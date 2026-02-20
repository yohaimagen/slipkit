import numpy as np
import rasterio
from rasterio.enums import Resampling
from pyproj import CRS, Transformer
from typing import Optional, Union, Tuple, List
from pathlib import Path

from slipkit.core.data import GeodeticDataSet

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