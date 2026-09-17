import numpy as np
import pytest

from slipkit.core.fault import TriangularFaultMesh, StrikeSlipType, DipSlipType
from slipkit.core.physics import CutdeCpuEngine
from slipkit.core.inversion import SlipDistribution
from slipkit.utils.parsers import InsarParser


ORIGIN_LON, ORIGIN_LAT = -69.3, 10.023


@pytest.fixture
def fault_and_slip():
    """A 2-triangle fault (km frame) with a known strike-slip + dip-slip vector."""
    verts = np.array([[0, 0, 0], [2, 0, 0], [0, 0, -1], [2, 0, -1]], dtype=float)
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    fault = TriangularFaultMesh(
        (verts, faces),
        strike_slip_type=StrikeSlipType.RIGHT_LATERAL,
        dip_slip_type=DipSlipType.NORMAL,
    )
    m = fault.num_patches()
    slip = np.zeros(2 * m)
    slip[:m] = 1.0
    slip[m:] = 0.5
    return fault, slip


@pytest.fixture
def input_table(tmp_path):
    """Writes a small full-resolution LOS point table and returns its path + arrays."""
    rng = np.random.default_rng(0)
    n = 40
    lon = ORIGIN_LON + rng.uniform(-0.05, 0.05, n)
    lat = ORIGIN_LAT + rng.uniform(-0.05, 0.05, n)
    look = np.tile([-0.70, -0.13, 0.70], (n, 1))  # look_U > 0 (ground->sat)
    disp = rng.normal(0.0, 10.0, n)               # mm
    path = tmp_path / "in.txt"
    np.savetxt(path, np.column_stack([lon, lat, look, disp]), fmt="%.6f")
    return path, lon, lat, look


def test_write_los_table_roundtrip(tmp_path):
    lon = np.array([-69.3, -69.2])
    lat = np.array([10.0, 10.1])
    look = np.array([[-0.7, -0.1, 0.7], [-0.6, -0.2, 0.75]])
    disp = np.array([12.5, -3.25])

    out = tmp_path / "out.txt"
    InsarParser.write_los_table(out, lon, lat, look, disp)

    arr = np.loadtxt(out)
    assert arr.shape == (2, 6)
    np.testing.assert_allclose(arr[:, 0], lon, atol=1e-6)
    np.testing.assert_allclose(arr[:, 1], lat, atol=1e-6)
    np.testing.assert_allclose(arr[:, 2:5], look, atol=1e-6)
    np.testing.assert_allclose(arr[:, 5], disp, atol=1e-6)


def test_write_los_table_bad_look_shape(tmp_path):
    with pytest.raises(ValueError, match="look must have shape"):
        InsarParser.write_los_table(
            tmp_path / "x.txt",
            lon=np.zeros(3), lat=np.zeros(3),
            look=np.zeros((2, 3)), disp=np.zeros(3),
        )


def test_predict_los_table_matches_kernel(fault_and_slip, input_table, tmp_path):
    fault, slip = fault_and_slip
    in_path, lon, lat, look = input_table
    engine = CutdeCpuEngine(0.25)
    out_path = tmp_path / "syn.txt"

    ds = InsarParser.predict_los_table(
        in_path, out_path, fault, SlipDistribution(slip, [fault]),
        engine, ORIGIN_LON, ORIGIN_LAT,
    )

    # Full resolution: one synthetic point per input point.
    assert len(ds) == len(lon)

    # Written file mirrors the input format and preserves the original look vectors.
    arr = np.loadtxt(out_path)
    assert arr.shape == (len(lon), 6)
    np.testing.assert_allclose(arr[:, 0], lon, atol=1e-5)
    np.testing.assert_allclose(arr[:, 1], lat, atol=1e-5)
    np.testing.assert_allclose(arr[:, 2:5], look, atol=1e-6)

    # Displacement column is the model prediction scaled m -> mm by default.
    # (Tolerance accounts for the %.6f rounding applied when writing the file.)
    np.testing.assert_allclose(arr[:, 5], ds.data * 1000.0, rtol=1e-5, atol=1e-4)

    # Prediction equals G @ slip with the flipped-look convention used at read.
    # Project the coords the function actually used (lon/lat as stored in the
    # file, i.e. rounded to the written precision) so the check is exact.
    coords = InsarParser._project_lonlat_to_local_km(
        arr[:, 0], arr[:, 1], ORIGIN_LON, ORIGIN_LAT
    )
    from slipkit.core.data import GeodeticDataSet
    check_ds = GeodeticDataSet(
        coords=coords, data=np.zeros(len(lon)), unit_vecs=-look,
        sigma=np.ones(len(lon)), name="chk",
    )
    expected = engine.build_kernel(fault, check_ds) @ slip
    np.testing.assert_allclose(ds.data, expected, rtol=1e-6, atol=1e-9)


def test_predict_los_table_accepts_list_and_scale(fault_and_slip, input_table, tmp_path):
    fault, slip = fault_and_slip
    in_path, lon, _, _ = input_table
    engine = CutdeCpuEngine(0.25)

    ds_scalar = InsarParser.predict_los_table(
        in_path, tmp_path / "a.txt", fault, slip, engine,
        ORIGIN_LON, ORIGIN_LAT, disp_scale=1.0,
    )
    ds_list = InsarParser.predict_los_table(
        in_path, tmp_path / "b.txt", [fault], slip, engine,
        ORIGIN_LON, ORIGIN_LAT, disp_scale=1.0,
    )
    np.testing.assert_allclose(ds_scalar.data, ds_list.data)

    # disp_scale=1.0 writes the model units verbatim (up to %.6f file rounding).
    arr = np.loadtxt(tmp_path / "a.txt")
    np.testing.assert_allclose(arr[:, 5], ds_scalar.data, rtol=1e-5, atol=1e-6)


def test_predict_los_table_chunking_invariance(fault_and_slip, input_table, tmp_path):
    """The chunk size only affects progress cadence, never the result."""
    fault, slip = fault_and_slip
    in_path, _, _, _ = input_table
    engine = CutdeCpuEngine(0.25)

    # Reference: a single chunk covering everything.
    ref = InsarParser.predict_los_table(
        in_path, tmp_path / "ref.txt", fault, slip, engine,
        ORIGIN_LON, ORIGIN_LAT, chunk_size=10_000,
    )
    # Tiny chunks force many chunk boundaries.
    small = InsarParser.predict_los_table(
        in_path, tmp_path / "s.txt", fault, slip, engine,
        ORIGIN_LON, ORIGIN_LAT, chunk_size=7,
    )
    # Default (chunk_size=None).
    default = InsarParser.predict_los_table(
        in_path, tmp_path / "a.txt", fault, slip, engine,
        ORIGIN_LON, ORIGIN_LAT, chunk_size=None,
    )

    np.testing.assert_allclose(small.data, ref.data, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(default.data, ref.data, rtol=1e-12, atol=1e-14)


def test_predict_los_table_slip_length_mismatch(fault_and_slip, input_table, tmp_path):
    fault, _ = fault_and_slip
    in_path, _, _, _ = input_table
    engine = CutdeCpuEngine(0.25)
    with pytest.raises(ValueError, match="does not match"):
        InsarParser.predict_los_table(
            in_path, tmp_path / "z.txt", fault, np.ones(3), engine,
            ORIGIN_LON, ORIGIN_LAT,
        )


# --------------------------------------------------------------------------- #
# Parquet I/O
# --------------------------------------------------------------------------- #

def test_write_los_table_parquet_roundtrip(tmp_path):
    """A .parquet destination writes GeoParquet our own reader can read back."""
    import json
    import pandas as pd
    import pyarrow.parquet as pq

    lon = np.array([-69.3, -69.2])
    lat = np.array([10.0, 10.1])
    look = np.array([[-0.7, -0.1, 0.7], [-0.6, -0.2, 0.75]])
    disp = np.array([12.5, -3.25])

    out = tmp_path / "out.parquet"
    InsarParser.write_los_table(out, lon, lat, look, disp)

    df = InsarParser._read_los_points(out)
    np.testing.assert_allclose(df["lon"].to_numpy(), lon, atol=1e-9)
    np.testing.assert_allclose(df["lat"].to_numpy(), lat, atol=1e-9)
    np.testing.assert_allclose(df[["look_e", "look_n", "look_u"]].to_numpy(), look, atol=1e-6)
    np.testing.assert_allclose(df["disp"].to_numpy(), disp, atol=1e-6)

    # GeoParquet metadata and a decodable WKB Point column, for QGIS/GDAL.
    meta = json.loads(pq.read_schema(out).metadata[b"geo"])
    assert meta["primary_column"] == "geometry" and meta["version"] == "1.1.0"
    wkb = pd.read_parquet(out, columns=["geometry"])["geometry"].iloc[1]
    assert len(wkb) == 21 and wkb[0] == 1
    assert np.frombuffer(wkb[5:13], "<f8")[0] == lon[1]
    assert np.frombuffer(wkb[13:21], "<f8")[0] == lat[1]


def test_predict_los_table_formats_are_interchangeable(
    fault_and_slip, input_table, tmp_path
):
    """Parquet in / parquet out gives the same prediction as the text path.

    Agreement is to ~1e-6 relative, not exact: the text table round-trips
    through ``%.6f`` while the parquet schema stores the look vectors as
    float32 (both matching the observed data files), so the two paths see
    coordinates and look vectors that differ in the last few digits.
    """
    fault, slip = fault_and_slip
    txt_path, _, _, _ = input_table
    engine = CutdeCpuEngine(0.25)

    # Mirror the text input as parquet, sourced from the text file itself so
    # both paths start from the same (already rounded) coordinates.
    df = InsarParser._read_los_points(txt_path)
    parquet_in = tmp_path / "in.parquet"
    InsarParser.write_los_table(
        parquet_in,
        df["lon"].to_numpy(),
        df["lat"].to_numpy(),
        df[["look_e", "look_n", "look_u"]].to_numpy(),
        df["disp"].to_numpy(),
    )

    from_txt = InsarParser.predict_los_table(
        txt_path, tmp_path / "o.txt", fault, slip, engine,
        ORIGIN_LON, ORIGIN_LAT, progress=False,
    )
    from_parquet = InsarParser.predict_los_table(
        parquet_in, tmp_path / "o.parquet", fault, slip, engine,
        ORIGIN_LON, ORIGIN_LAT, progress=False,
    )
    np.testing.assert_allclose(from_parquet.data, from_txt.data, rtol=1e-6, atol=1e-9)

    # The written synthetic tables agree too, whichever format they are in.
    written_txt = np.loadtxt(tmp_path / "o.txt")[:, 5]
    written_parquet = InsarParser._read_los_points(tmp_path / "o.parquet")["disp"].to_numpy()
    np.testing.assert_allclose(written_parquet, written_txt, rtol=1e-5, atol=1e-5)
