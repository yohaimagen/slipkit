import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from pyproj import CRS, Transformer
from typing import Optional, Union, Tuple, List, Sequence
from pathlib import Path

from slipkit.core.data import GeodeticDataSet, Ramp
from slipkit.core.fault import AbstractFaultModel
from slipkit.core.physics import GreenFunctionBuilder


class InsarParser:
    """
    Parser for InSAR displacement data (GeoTiff format).

    Handles reading raster data, downsampling (Uniform or Quadtree), 
    coordinate projection (WGS84 -> Local UTM), and Line-of-Sight (LOS) vector generation.
    """

    @staticmethod
    def read_geotiff(
        filepath: Union[str, Path],
        origin_lon: float,
        origin_lat: float,
        downsample_method: str = 'uniform',
        downsample_factor: Union[int, Tuple[int, int]] = 1,
        heading: Union[float, str, Path] = 0.0,
        incidence: Union[float, str, Path] = 30.0,
        name: str = "InSAR_Dataset",
        nan_threshold: float = 0.5,
        quadtree_var_thresh: float = 1e-3
    ) -> GeodeticDataSet:
        """
        Reads an InSAR displacement GeoTiff, downsamples it, and converts it to a GeodeticDataSet.

        Args:
            filepath: Path to the displacement GeoTiff.
            origin_lon: Longitude of the local coordinate system origin.
            origin_lat: Latitude of the local coordinate system origin.
            downsample_method: 'uniform' or 'quadtree'.
            downsample_factor: 
                For 'uniform': Integer scalar (square block) or Tuple (row_factor, col_factor). 
                If 0 or 1, no downsampling is performed.
            heading: Flight direction (degrees). Scalar or path to raster.
            incidence: Incidence angle (degrees). Scalar or path to raster.
            name: Identifier for the dataset.
            nan_threshold: Fraction of NaNs allowed in a block before masking it (uniform only).
            quadtree_var_thresh: Variance threshold for quadtree splitting (quadtree only).

        Returns:
            GeodeticDataSet: Populated data container with local coordinates.
        """
        
        # 1. Read Displacement Raster
        with rasterio.open(filepath) as src:
            data_raw = src.read(1)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
            height, width = data_raw.shape

        # Mask NaNs/NoData
        if nodata is not None:
            data_masked = np.ma.masked_equal(data_raw, nodata)
        else:
            data_masked = np.ma.masked_invalid(data_raw)
            
        # Also mask standard NaNs if floating point
        data_masked = np.ma.masked_invalid(data_masked)

        if isinstance(downsample_factor, (int, float)):
            r_factor = c_factor = int(downsample_factor)
        else:
            r_factor, c_factor = downsample_factor
            
        # 2. Downsampling Strategy
        if downsample_method == 'uniform':
            # Check for "No Downsampling" condition (0 or 1)
            if r_factor <= 1 and c_factor <= 1:
                valid_mask = ~data_masked.mask
                data_final = data_masked[valid_mask].data
                rows, cols = np.where(valid_mask)
            else:
                data_final, rows, cols = InsarParser._uniform_median_downsample(
                    data_masked, r_factor, c_factor, nan_threshold
                )

        elif downsample_method == 'quadtree':
            # PASS THE FACTORS (K, L) TO QUADTREE
            data_final, rows, cols = InsarParser._quadtree_downsample(
                data_masked, 
                quadtree_var_thresh, 
                initial_block_size=(r_factor, c_factor) # <--- NEW ARGUMENT
            )
        
        else:
            raise ValueError(f"Unknown downsample method: {downsample_method}")

        # 3. Coordinate Conversion (Pixel -> Lat/Lon -> Local UTM)
        # transform * (col, row) -> (x, y)
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
        xs = np.array(xs)
        ys = np.array(ys)

        # Projections
        utm_zone = int((origin_lon + 180) / 6) + 1
        hemisphere = 'south' if origin_lat < 0 else 'north'
        
        # Default to EPSG:4326 if CRS is missing from GeoTiff
        source_crs = crs if crs else CRS.from_epsg(4326)
        target_crs = CRS.from_proj4(
            f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"
        )
        
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

        # Transform points
        easting, northing = transformer.transform(xs, ys)
        # Transform Origin
        origin_e, origin_n = transformer.transform(origin_lon, origin_lat)

        # Local Cartesian
        local_coords = np.column_stack((
            (easting - origin_e) * 1e-3,  # Convert to km
            (northing - origin_n) * 1e-3,  # Convert to km
            np.zeros_like(easting)  # Z=0 for now
        ))

        # 4. Compute Unit Vectors (using original image indices)
        unit_vecs = InsarParser._compute_los_vectors(
            heading, incidence, rows, cols, (height, width)
        )

        # 5. Sigma (Placeholder)
        sigma = np.ones_like(data_final) * 1.0

        return GeodeticDataSet(
            coords=local_coords,
            data=data_final,
            unit_vecs=unit_vecs,
            sigma=sigma,
            name=name
        )

    @staticmethod
    def read_los_table(
        filepath: Union[str, Path],
        origin_lon: float,
        origin_lat: float,
        downsample_method: str = 'quadtree',
        downsample_factor: Union[int, Tuple[int, int]] = 1,
        name: str = "InSAR_LOS_Table",
        nan_threshold: float = 0.5,
        quadtree_var_thresh: float = 1e-3,
        flip_look_sign: bool = True,
        ramp_degree: Optional[int] = None,
    ) -> GeodeticDataSet:
        """
        Reads a whitespace-delimited LOS displacement point table and converts it
        to a GeodeticDataSet, following the same path as :meth:`read_geotiff`.

        The table is the "InSAR data for Modelers" format with one row per pixel::

            #lon #lat #look_E #look_N #look_U #displacement

        Unlike a GeoTiff, the look vector is supplied per point (three columns)
        rather than derived from scalar heading/incidence. The points lie on a
        regular lon/lat grid with masked pixels dropped, so this method lays them
        back onto a raster grid and then reuses the identical downsampling and
        projection logic as :meth:`read_geotiff`.

        Displacement values are stored verbatim (same units as the file, mm for
        the Venezuela dataset); no unit conversion is applied.

        Args:
            filepath: Path to the LOS point table.
            origin_lon: Longitude of the local coordinate system origin.
            origin_lat: Latitude of the local coordinate system origin.
            downsample_method: 'uniform' or 'quadtree'.
            downsample_factor:
                For 'uniform': integer (square block) or (row_factor, col_factor).
                For 'quadtree': the initial uniform block size (K, L). 1 = no
                uniform downsampling for 'uniform'.
            name: Identifier for the dataset.
            nan_threshold: Fraction of NaNs allowed in a block before masking it
                (uniform only).
            quadtree_var_thresh: Variance threshold for quadtree splitting.
            flip_look_sign: If True (default), negate the look vectors so they
                match SlipKit's convention (as produced by :meth:`read_geotiff`),
                where ``predicted = u . look`` gives a *negative* LOS for uplift
                -- i.e. positive LOS = motion away from the satellite (GMTSAR
                range-increase convention). The "InSAR data for Modelers" tables
                ship look vectors pointing ground->satellite (look_U > 0), which
                is the opposite sign; without this flip the predicted LOS comes
                out inverted relative to the observed displacement.
            ramp_degree: If given, attach a polynomial nuisance ramp of this
                degree (0 = constant LOS offset, 1 = planar, 2 = quadratic) to
                be solved jointly with slip. ``None`` (default) adds no nuisance
                unknowns. See :class:`~slipkit.core.data.Ramp`.

        Returns:
            GeodeticDataSet: Local coordinates (km), per-point look vectors, and
            LOS displacement.
        """
        # 1. Read the point table (lon, lat, look_E, look_N, look_U, disp).
        df = InsarParser._read_los_points(filepath)
        lon = df["lon"].to_numpy()
        lat = df["lat"].to_numpy()
        disp = df["disp"].to_numpy()
        look = df[["look_e", "look_n", "look_u"]].to_numpy()

        # 2. Rasterize the points back onto their regular grid.
        # Infer pixel spacing from the (regular) unique coordinate values.
        uniq_lon = np.unique(lon)
        uniq_lat = np.unique(lat)
        dlon = np.median(np.diff(uniq_lon))
        dlat = np.median(np.diff(uniq_lat))

        lon_min = lon.min()
        lat_max = lat.max()

        cols = np.rint((lon - lon_min) / dlon).astype(np.intp)
        rows = np.rint((lat_max - lat) / dlat).astype(np.intp)  # row 0 = north
        n_rows = int(rows.max()) + 1
        n_cols = int(cols.max()) + 1

        # Affine transform equivalent to a GeoTiff's, so rasterio.transform.xy
        # (offset='center') recovers each pixel's lon/lat.
        transform = from_origin(
            lon_min - dlon / 2.0, lat_max + dlat / 2.0, dlon, dlat
        )

        # Displacement grid (the field the downsamplers operate on).
        disp_grid = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
        disp_grid[rows, cols] = disp
        data_masked = np.ma.masked_invalid(disp_grid)

        # Per-pixel look grids. Filled by nearest-valid so a downsample leaf
        # centre that lands on a dropped pixel still gets a look vector
        # (look vectors vary smoothly across the scene).
        look_grids = []
        for k in range(3):
            g = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
            g[rows, cols] = look[:, k]
            look_grids.append(InsarParser._fill_nearest(g))

        # 3. Downsample the displacement grid (identical to read_geotiff).
        if isinstance(downsample_factor, (int, float)):
            r_factor = c_factor = int(downsample_factor)
        else:
            r_factor, c_factor = downsample_factor

        if downsample_method == 'uniform':
            if r_factor <= 1 and c_factor <= 1:
                valid_mask = ~data_masked.mask
                data_final = data_masked[valid_mask].data
                out_rows, out_cols = np.where(valid_mask)
            else:
                data_final, out_rows, out_cols = InsarParser._uniform_median_downsample(
                    data_masked, r_factor, c_factor, nan_threshold
                )
        elif downsample_method == 'quadtree':
            data_final, out_rows, out_cols = InsarParser._quadtree_downsample(
                data_masked,
                quadtree_var_thresh,
                initial_block_size=(r_factor, c_factor),
            )
        else:
            raise ValueError(f"Unknown downsample method: {downsample_method}")

        out_rows = np.asarray(out_rows, dtype=np.intp)
        out_cols = np.asarray(out_cols, dtype=np.intp)

        # 4. Pixel -> lon/lat -> local UTM (km), identical to read_geotiff.
        xs, ys = rasterio.transform.xy(transform, out_rows, out_cols, offset='center')
        xs = np.array(xs)
        ys = np.array(ys)

        utm_zone = int((origin_lon + 180) / 6) + 1
        hemisphere = 'south' if origin_lat < 0 else 'north'

        source_crs = CRS.from_epsg(4326)
        target_crs = CRS.from_proj4(
            f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"
        )
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

        easting, northing = transformer.transform(xs, ys)
        origin_e, origin_n = transformer.transform(origin_lon, origin_lat)

        local_coords = np.column_stack((
            (easting - origin_e) * 1e-3,   # km
            (northing - origin_n) * 1e-3,  # km
            np.zeros_like(easting),        # Z = 0
        ))

        # 5. Per-point look vectors, sampled at the retained pixels.
        unit_vecs = np.column_stack([g[out_rows, out_cols] for g in look_grids])
        if flip_look_sign:
            # Match SlipKit's LOS convention (see flip_look_sign in the docstring).
            unit_vecs = -unit_vecs

        # 6. Sigma placeholder (matches read_geotiff).
        sigma = np.ones_like(data_final)

        return GeodeticDataSet(
            coords=local_coords,
            data=data_final,
            unit_vecs=unit_vecs,
            sigma=sigma,
            name=name,
            ramp=ramp_degree,
        )

    @staticmethod
    def _read_los_points(filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Reads a LOS point table into a DataFrame, whatever format it is in.

        Accepts GeoParquet (named columns, displacement in ``disp_mm``) or the
        original whitespace-delimited "InSAR data for Modelers" text table. Only
        the six numeric columns are read, so a parquet file's WKB ``geometry``
        column is never loaded or parsed.

        Args:
            filepath: Path to the point table.

        Returns:
            A DataFrame with columns ``lon, lat, look_e, look_n, look_u, disp``.
        """
        if Path(filepath).suffix == ".parquet":
            df = pd.read_parquet(
                filepath,
                columns=["lon", "lat", "look_e", "look_n", "look_u", "disp_mm"],
            ).rename(columns={"disp_mm": "disp"})
        else:
            df = pd.read_csv(
                filepath,
                sep=r"\s+",
                header=None,
                comment="#",
                names=["lon", "lat", "look_e", "look_n", "look_u", "disp"],
            )
        return df

    @staticmethod
    def _project_lonlat_to_local_km(
        lon: np.ndarray,
        lat: np.ndarray,
        origin_lon: float,
        origin_lat: float,
    ) -> np.ndarray:
        """
        Projects WGS84 lon/lat to local UTM Cartesian coordinates in kilometres.

        Uses the same projection convention as :meth:`read_geotiff` /
        :meth:`read_los_table`: the UTM zone is chosen from ``origin_lon`` and the
        result is expressed relative to (``origin_lon``, ``origin_lat``).

        Args:
            lon: (N,) longitudes in degrees.
            lat: (N,) latitudes in degrees.
            origin_lon: Longitude of the local-frame origin.
            origin_lat: Latitude of the local-frame origin.

        Returns:
            An ``(N, 3)`` array of ``(x, y, z)`` in km, with ``z = 0``.
        """
        utm_zone = int((origin_lon + 180) / 6) + 1
        hemisphere = 'south' if origin_lat < 0 else 'north'

        source_crs = CRS.from_epsg(4326)
        target_crs = CRS.from_proj4(
            f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"
        )
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

        easting, northing = transformer.transform(lon, lat)
        origin_e, origin_n = transformer.transform(origin_lon, origin_lat)

        return np.column_stack((
            (np.asarray(easting) - origin_e) * 1e-3,   # km
            (np.asarray(northing) - origin_n) * 1e-3,  # km
            np.zeros_like(np.asarray(easting), dtype=float),
        ))

    @staticmethod
    def _predict_chunked(
        faults: List[AbstractFaultModel],
        slip_vector: np.ndarray,
        engine: GreenFunctionBuilder,
        coords: np.ndarray,
        look: np.ndarray,
        chunk_size: Optional[int],
        progress: bool = True,
    ) -> np.ndarray:
        """
        Evaluates the forward model ``G @ slip`` at every observation point.

        The prediction is matrix-free: it goes through ``engine.predict`` (which,
        for :class:`~slipkit.core.physics.CutdeCpuEngine`, uses cutde's
        ``disp_free``), so the dense ``(N, 2M)`` kernel is never formed and memory
        stays O(N). cutde already parallelises each call across cores, so there is
        no process pool here; points are merely walked in chunks to drive the
        progress counter (and keep per-call arrays modest).

        Args:
            faults: Fault models, in slip-vector block order.
            slip_vector: The full concatenated slip vector.
            engine: Green's-function engine.
            coords: ``(N, 3)`` observation coords (local km).
            look: ``(N, 3)`` look vectors in the engine's convention.
            chunk_size: Points evaluated per cutde call. If ``None``, defaults to
                ``50_000``. Only affects progress granularity and per-call array
                size, not the result.
            progress: If True, prints a live "n/N points (k/K chunks)" counter as
                chunks complete.

        Returns:
            The ``(N,)`` predicted LOS.
        """
        n_points = coords.shape[0]

        # Slice the slip vector into per-fault blocks (matches SlipDistribution).
        slip_blocks: List[np.ndarray] = []
        offset = 0
        for fault in faults:
            width = fault.num_components() * fault.num_patches()
            slip_blocks.append(np.ascontiguousarray(slip_vector[offset:offset + width]))
            offset += width

        if chunk_size is None:
            chunk_size = 50_000
        chunk_size = max(1, min(int(chunk_size), n_points if n_points else 1))

        predicted = np.empty(n_points)
        starts = list(range(0, n_points, chunk_size))
        n_chunks = len(starts)

        for i, start in enumerate(starts, start=1):
            stop = min(start + chunk_size, n_points)
            chunk_ds = GeodeticDataSet(
                coords=coords[start:stop],
                data=np.zeros(stop - start),
                unit_vecs=look[start:stop],
                sigma=np.ones(stop - start),
                name="forward_chunk",
            )
            out = np.zeros(stop - start)
            for fault, slip_block in zip(faults, slip_blocks):
                out += engine.predict(fault, chunk_ds, slip_block)
            predicted[start:stop] = out

            if progress:
                end = "\n" if i == n_chunks else ""
                print(
                    f"\rForward model: {stop}/{n_points} points "
                    f"({i}/{n_chunks} chunks)",
                    end=end,
                    flush=True,
                )

        return predicted

    @staticmethod
    def write_los_table(
        filepath: Union[str, Path],
        lon: np.ndarray,
        lat: np.ndarray,
        look: np.ndarray,
        disp: np.ndarray,
        fmt: str = "%.6f",
    ) -> None:
        """
        Writes a LOS displacement point table in the "InSAR data for Modelers"
        format read by :meth:`read_los_table`.

        A ``.parquet`` destination writes GeoParquet instead, matching the
        layout :meth:`_read_los_points` expects and the project's own converter
        produces (``geometry, lon, lat, look_e, look_n, look_u, disp_mm``). It
        is far faster and ~10x smaller than the text form at full resolution,
        and drops straight into QGIS/GDAL/geopandas.

        Any other suffix writes the whitespace-delimited text table with no
        header, one row per point::

            lon lat look_E look_N look_U displacement

        Args:
            filepath: Destination path. ``.parquet`` selects GeoParquet, anything
                else the text table.
            lon: (N,) longitudes in degrees.
            lat: (N,) latitudes in degrees.
            look: (N, 3) look vectors ``[look_E, look_N, look_U]``.
            disp: (N,) displacement values (same units the reader expects, i.e.
                mm for the Venezuela dataset -- the parquet column is named
                ``disp_mm`` accordingly).
            fmt: ``numpy.savetxt`` format spec applied to every column. Ignored
                for parquet output.
        """
        lon = np.asarray(lon)
        lat = np.asarray(lat)
        look = np.asarray(look)
        disp = np.asarray(disp)
        if look.shape != (lon.shape[0], 3):
            raise ValueError(
                f"look must have shape (N, 3); got {look.shape} for N={lon.shape[0]}."
            )

        if Path(filepath).suffix == ".parquet":
            InsarParser._write_los_parquet(filepath, lon, lat, look, disp)
            return

        table = np.column_stack((lon, lat, look[:, 0], look[:, 1], look[:, 2], disp))
        np.savetxt(filepath, table, fmt=fmt, delimiter=" ")

    # GeoParquet schema of the LOS point tables: a WKB Point geometry column for
    # GIS tools, plus the same coordinates as plain floats so numeric readers
    # (i.e. _read_los_points) never parse WKB.
    _PARQUET_F64 = ("lon", "lat")
    _PARQUET_COLUMNS = ("lon", "lat", "look_e", "look_n", "look_u", "disp_mm")

    @staticmethod
    def _los_parquet_schema() -> "pa.Schema":
        """Returns the GeoParquet schema written by :meth:`write_los_table`."""
        geo_meta = {
            "version": "1.1.0",
            "primary_column": "geometry",
            "columns": {
                "geometry": {
                    "encoding": "WKB",
                    "geometry_types": ["Point"],
                    "crs": CRS.from_user_input("OGC:CRS84").to_json_dict(),
                    "edges": "planar",
                }
            },
        }
        fields = [("geometry", pa.binary())] + [
            (c, pa.float64() if c in InsarParser._PARQUET_F64 else pa.float32())
            for c in InsarParser._PARQUET_COLUMNS
        ]
        return pa.schema(fields, metadata={b"geo": json.dumps(geo_meta).encode()})

    @staticmethod
    def _wkb_points(lon: np.ndarray, lat: np.ndarray) -> "pa.Array":
        """
        Builds a little-endian WKB Point array (21 bytes per point).

        Assembled directly from the coordinate buffers, so writing GeoParquet
        needs no shapely/geopandas dependency.
        """
        n = len(lon)
        buf = np.empty((n, 21), dtype=np.uint8)
        buf[:, 0] = 1                   # little endian
        buf[:, 1] = 1                   # geometry type 1 = Point
        buf[:, 2:5] = 0
        buf[:, 5:13] = np.ascontiguousarray(lon, "<f8").view(np.uint8).reshape(n, 8)
        buf[:, 13:21] = np.ascontiguousarray(lat, "<f8").view(np.uint8).reshape(n, 8)
        offsets = np.arange(0, 21 * (n + 1), 21, dtype=np.int32)
        return pa.Array.from_buffers(
            pa.binary(), n, [None, pa.py_buffer(offsets), pa.py_buffer(buf)]
        )

    @staticmethod
    def _write_los_parquet(
        filepath: Union[str, Path],
        lon: np.ndarray,
        lat: np.ndarray,
        look: np.ndarray,
        disp: np.ndarray,
        row_group_size: int = 5_000_000,
    ) -> None:
        """
        Writes a LOS point table as zstd-compressed GeoParquet.

        Args:
            filepath: Destination ``.parquet`` path.
            lon: (N,) longitudes in degrees.
            lat: (N,) latitudes in degrees.
            look: (N, 3) look vectors ``[look_E, look_N, look_U]``.
            disp: (N,) displacement, written to the ``disp_mm`` column.
            row_group_size: Rows per parquet row group, so a full-resolution
                table streams instead of landing in one giant group.
        """
        schema = InsarParser._los_parquet_schema()
        columns = dict(
            lon=lon, lat=lat,
            look_e=look[:, 0], look_n=look[:, 1], look_u=look[:, 2],
            disp_mm=disp,
        )
        arrays = [InsarParser._wkb_points(lon, lat)] + [
            pa.array(columns[c], type=schema.field(c).type)
            for c in InsarParser._PARQUET_COLUMNS
        ]
        pq.write_table(
            pa.Table.from_arrays(arrays, schema=schema),
            filepath,
            compression="zstd",
            row_group_size=row_group_size,
        )

    @staticmethod
    def predict_los_table(
        input_filepath: Union[str, Path],
        output_filepath: Union[str, Path],
        faults: Union[AbstractFaultModel, Sequence[AbstractFaultModel]],
        slip: Union[np.ndarray, "object"],
        engine: GreenFunctionBuilder,
        origin_lon: float,
        origin_lat: float,
        disp_scale: float = 1000.0,
        flip_look_sign: bool = True,
        name: str = "Synthetic_LOS",
        chunk_size: Optional[int] = None,
        progress: bool = True,
        ramp_name: Optional[str] = None,
    ) -> GeodeticDataSet:
        """
        Computes the synthetic LOS displacement at the *full resolution* of an
        input LOS point table and writes it back in the same file format.

        This is a pure forward model, independent of the inversion machinery: it
        reads every point of ``input_filepath`` (no downsampling), evaluates the
        elastic response ``G @ slip`` at those points using the same fault(s),
        physics engine and projection origin as the inversion, and saves the
        predicted LOS to ``output_filepath`` as a drop-in replacement for the
        input (identical columns, original-convention look vectors, only the
        displacement column changed).

        The prediction is matrix-free (via ``engine.predict`` -> cutde
        ``disp_free``): the dense ``(N, 2M)`` Green's-function matrix is never
        built, so memory stays O(N) even at full data resolution. Points are
        walked in chunks only to drive the progress counter; cutde already
        parallelises each call across cores.

        Args:
            input_filepath: Path to the observed LOS point table to mirror
                (GeoParquet or the whitespace text table).
            output_filepath: Where to write the synthetic table. A ``.parquet``
                suffix writes GeoParquet, anything else the text table --
                independent of the input's format.
            faults: The fault model (or list of fault models) used in the
                inversion. Green's-function column blocks are laid out fault by
                fault, matching the slip-vector layout of ``SlipDistribution``.
            slip: The inverted slip. Either the raw ``(sum_i k_i M_i,)`` vector or
                a ``SlipDistribution`` (its ``slip_vector`` is used).
            engine: The Green's-function engine (e.g. ``CutdeCpuEngine``); must be
                the same physics used for the inversion so signs/units match.
            origin_lon: Longitude of the local-frame origin (same value used to
                build the fault mesh and read the data).
            origin_lat: Latitude of the local-frame origin.
            disp_scale: Multiplies the predicted LOS before writing, to convert
                the model's units to the file's units. The default ``1000.0``
                converts model metres to the file's millimetres; pass ``1.0`` if
                the model already produces the file's units.
            flip_look_sign: Must match the value used when the data were read
                (default ``True``). It negates the file's look vectors to
                SlipKit's LOS convention for the prediction; the *written* look
                vectors are always the original (un-flipped) file values.
            name: Identifier for the returned dataset.
            chunk_size: Number of observation points evaluated per cutde call
                (default ``50_000``). Only affects progress granularity and
                per-call array size; the result is identical for any chunk size.
            progress: If True (default), prints a live "n/N points" progress
                counter as chunks are computed.
            ramp_name: Name of a fitted nuisance ramp in ``slip.nuisance`` (i.e.
                the name of the dataset it was estimated for) to add to the
                elastic prediction, evaluated at these full-resolution points.
                ``None`` (default) writes the pure elastic forward model.

        Returns:
            A full-resolution :class:`GeodeticDataSet` whose ``data`` holds the
            predicted LOS in *model* units (i.e. before ``disp_scale``), with
            local-km coords and the look vectors used for prediction.
        """
        # 1. Read the full-resolution point table verbatim (no downsampling).
        df = InsarParser._read_los_points(input_filepath)
        lon = df["lon"].to_numpy()
        lat = df["lat"].to_numpy()
        look_file = df[["look_e", "look_n", "look_u"]].to_numpy()  # original convention

        # 2. Project to the local km frame the fault mesh lives in.
        coords = InsarParser._project_lonlat_to_local_km(
            lon, lat, origin_lon, origin_lat
        )

        # 3. Look vectors for prediction: match read_los_table's sign convention.
        look_pred = -look_file if flip_look_sign else look_file

        # 4. Normalise faults / slip.
        fault_list: List[AbstractFaultModel] = (
            [faults] if isinstance(faults, AbstractFaultModel) else list(faults)
        )
        slip_vector = np.asarray(getattr(slip, "slip_vector", slip)).ravel()

        expected = sum(f.num_components() * f.num_patches() for f in fault_list)
        if slip_vector.shape[0] != expected:
            raise ValueError(
                f"slip length ({slip_vector.shape[0]}) does not match the total "
                f"number of unknowns across faults ({expected})."
            )

        # Evaluate G @ slip matrix-free (no dense kernel; O(N) memory).
        predicted = InsarParser._predict_chunked(
            fault_list, slip_vector, engine, coords, look_pred,
            chunk_size=chunk_size, progress=progress,
        )

        # Add the fitted nuisance ramp, if one was requested. The ramp carries
        # its own centre/scale, so it evaluates consistently at these points
        # even though they are not the (downsampled) ones it was fitted on.
        if ramp_name is not None:
            nuisance = getattr(slip, "nuisance", {})
            if ramp_name not in nuisance:
                raise KeyError(
                    f"No fitted ramp named '{ramp_name}'. Available: "
                    f"{sorted(nuisance)}."
                )
            predicted = predicted + nuisance[ramp_name].evaluate(coords)

        # 5. Write the synthetic table in the input format (original look vectors,
        #    displacement scaled back to the file's units).
        InsarParser.write_los_table(
            output_filepath,
            lon=lon,
            lat=lat,
            look=look_file,
            disp=predicted * disp_scale,
        )

        # Return the full-resolution prediction (model units) for inspection.
        return GeodeticDataSet(
            coords=coords,
            data=predicted,
            unit_vecs=look_pred,
            sigma=np.ones(coords.shape[0]),
            name=name,
        )

    @staticmethod
    def _fill_nearest(grid: np.ndarray) -> np.ndarray:
        """
        Fills NaN cells of a 2-D grid with their nearest valid neighbour value.

        Used for the per-pixel look-vector grids, which are smooth and defined
        only where displacement is valid; nearest-fill lets a downsample leaf
        centre on a dropped pixel still receive a sensible look vector.

        Args:
            grid: 2-D array with NaNs marking missing cells.

        Returns:
            A copy of ``grid`` with all NaNs replaced by the nearest valid value.
        """
        from scipy.ndimage import distance_transform_edt

        invalid = np.isnan(grid)
        if not invalid.any():
            return grid
        ind = distance_transform_edt(
            invalid, return_distances=False, return_indices=True
        )
        return grid[tuple(ind)]

    @staticmethod
    def _uniform_median_downsample(
        data: np.ma.MaskedArray, 
        r_factor: int,
        c_factor: int,
        nan_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs uniform median downsampling with independent row/col factors.
        """
        # Ensure factors are at least 1 to avoid division by zero
        r_factor = max(1, r_factor)
        c_factor = max(1, c_factor)

        rows, cols = data.shape
        
        # Trim to be divisible
        new_rows = rows // r_factor
        new_cols = cols // c_factor
        
        if new_rows == 0 or new_cols == 0:
            raise ValueError("Downsample factor is larger than image dimensions.")

        trimmed = data[:new_rows * r_factor, :new_cols * c_factor]
        
        # Reshape: (new_rows, r_factor, new_cols, c_factor)
        reshaped = trimmed.reshape(new_rows, r_factor, new_cols, c_factor)
        
        # Transpose: (new_rows, new_cols, r_factor, c_factor)
        blocks = reshaped.transpose(0, 2, 1, 3)
        
        # Flatten blocks: (new_rows, new_cols, pixels_per_block)
        blocks_flat = blocks.reshape(new_rows, new_cols, -1)
        
        # Median calc
        downsampled_data = np.ma.median(blocks_flat, axis=2)
        
        # Validity Check
        mask_counts = np.sum(blocks_flat.mask, axis=2)
        total_pixels = r_factor * c_factor
        bad_blocks = (mask_counts / total_pixels) > nan_threshold
        downsampled_data[bad_blocks] = np.ma.masked

        # Extract valid
        valid_mask = ~downsampled_data.mask
        grid_rows, grid_cols = np.where(valid_mask)
        
        # Map back to original indices (center of block)
        original_rows = grid_rows * r_factor + (r_factor // 2)
        original_cols = grid_cols * c_factor + (c_factor // 2)
        
        return downsampled_data[valid_mask].data, original_rows, original_cols

    @staticmethod
    def _quadtree_downsample(
        data: np.ma.MaskedArray, 
        variance_thresh: float,
        initial_block_size: Tuple[int, int],
        min_size: int = 4,
        nan_allowed_fraction: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs Quadtree decomposition initialized on a uniform grid.
        
        Args:
            data: Masked displacement array.
            variance_thresh: Std dev threshold to trigger split.
            initial_block_size: (rows, cols) defining the 'K, L' uniform grid. 
                                Recursion starts within these blocks.
            min_size: Minimum pixel dimension of a leaf node.
            nan_allowed_fraction: Min fraction of valid pixels required.
        """
        final_values = []
        final_rows = []
        final_cols = []
        
        total_rows, total_cols = data.shape
        r_step, c_step = initial_block_size

        # ---------------------------------------------------------
        # Inner Recursive Function
        # ---------------------------------------------------------
        def _recursive_quad(r_start, c_start, r_len, c_len):
            # 1. Extract Block
            block = data[r_start : r_start + r_len, c_start : c_start + c_len]
            
            # 2. Check Validity
            valid_count = block.count()
            if valid_count == 0:
                return # Empty block

            total_pixels = r_len * c_len
            valid_fraction = valid_count / total_pixels
            
            # 3. Spatial Distribution (Fast Lux/Luy check)
            valid_mask = ~np.ma.getmaskarray(block)
            luy = np.sum(np.any(valid_mask, axis=1)) # Span in Y
            lux = np.sum(np.any(valid_mask, axis=0)) # Span in X

            # 4. Calc Stats
            std = np.std(block.compressed()) if valid_count > 1 else 0.0

            should_split = False
            
            # --- SPLIT LOGIC ---
            # Only split if:
            # A. We have enough valid data/geometry
            if (valid_fraction >= nan_allowed_fraction and 
                valid_count > 2 and 
                lux > 1 and luy > 1 and 
                (1/3 < lux/luy < 3)):
                
                # B. The variance is high AND we are larger than min_size
                if std > variance_thresh and r_len > min_size and c_len > min_size:
                    should_split = True
            
            if should_split:
                # Calculate sub-dimensions
                half_r = r_len // 2
                half_c = c_len // 2
                
                # Recursively call 4 quadrants
                # Top-Left
                _recursive_quad(r_start, c_start, half_r, half_c)
                # Top-Right
                _recursive_quad(r_start, c_start + half_c, half_r, c_len - half_c)
                # Bottom-Left
                _recursive_quad(r_start + half_r, c_start, r_len - half_r, half_c)
                # Bottom-Right
                _recursive_quad(r_start + half_r, c_start + half_c, r_len - half_r, c_len - half_c)
            else:
                # --- LEAF NODE ---
                # Calculate median and center indices
                if valid_count > 0:
                    median_val = np.ma.median(block)
                    if not np.ma.is_masked(median_val):
                        final_values.append(median_val)
                        final_rows.append(r_start + r_len // 2)
                        final_cols.append(c_start + c_len // 2)

        # ---------------------------------------------------------
        # Outer Loop: Uniform Sampling Initialization
        # ---------------------------------------------------------
        # We iterate over the image in strides of r_step (K) and c_step (L).
        # This treats every KxL block as a "Root" for a mini-quadtree.
        for r in range(0, total_rows, r_step):
            for c in range(0, total_cols, c_step):
                
                # Handle edge cases where image dims aren't divisible by step
                # The last block might be smaller than K x L
                curr_h = min(r_step, total_rows - r)
                curr_w = min(c_step, total_cols - c)
                
                # Start recursion for this uniform tile
                _recursive_quad(r, c, curr_h, curr_w)

        return np.array(final_values), np.array(final_rows), np.array(final_cols)

    @staticmethod
    def _compute_los_vectors(
        heading: Union[float, str, Path],
        incidence: Union[float, str, Path],
        rows: np.ndarray,
        cols: np.ndarray,
        shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Computes Look Vector [East, North, Up] for specified pixels.
        Vector points FROM ground TO satellite.
        """
        
        def get_angle_values(source, name_log):
            if isinstance(source, (float, int)):
                return np.full(rows.shape, float(source))
            elif isinstance(source, (str, Path)):
                # If path, we must sample the raster at specific row/cols
                with rasterio.open(source) as src:
                    if src.height != shape[0] or src.width != shape[1]:
                        raise ValueError(f"{name_log} raster dims {src.shape} != data dims {shape}.")
                    
                    # Read full array (optimization: windowed read for sparse points?)
                    # For now, assuming fit-in-memory for typical InSAR crops
                    arr = src.read(1)
                    return arr[rows, cols]
            else:
                raise TypeError(f"Invalid type for {name_log}")

        h_arr = get_angle_values(heading, "Heading")
        inc_arr = get_angle_values(incidence, "Incidence")

        h_rad = np.deg2rad(h_arr)
        inc_rad = np.deg2rad(inc_arr)

        # Standard InSAR LOS vector components
        # u_E = -sin(heading) * sin(incidence)
        # u_N =  cos(heading) * sin(incidence)
        # u_U =  cos(incidence)
        u_e = -np.sin(h_rad) * np.sin(inc_rad)
        u_n = np.cos(h_rad) * np.sin(inc_rad)
        u_u = np.cos(inc_rad)

        return -np.column_stack((u_e, u_n, u_u))