import numpy as np
import pandas as pd
import pytest

from slipkit.core.fault import TriangularFaultMesh, StrikeSlipType, DipSlipType
from slipkit.core.data import GeodeticDataSet
from slipkit.core.physics import CutdeCpuEngine
from slipkit.utils.statistics import FitStatistics


def test_dataset_metrics_known_values():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.5, 1.5, 3.5, 3.5])  # residual = [-0.5, 0.5, -0.5, 0.5]
    m = FitStatistics.dataset_metrics(obs, pred)

    assert m["n"] == 4
    assert m["rms"] == pytest.approx(0.5)              # sqrt(mean(0.25))
    assert m["residual_norm"] == pytest.approx(1.0)    # sqrt(4 * 0.25)
    assert m["data_norm"] == pytest.approx(np.sqrt(30.0))
    # VR = (1 - 1.0/30.0) * 100
    assert m["variance_reduction_pct"] == pytest.approx((1 - 1.0 / 30.0) * 100)
    assert m["mean_residual"] == pytest.approx(0.0)
    assert m["max_abs_residual"] == pytest.approx(0.5)
    # sigma=None => weighted == unweighted, chi2 == sum(r^2)
    assert m["wrms"] == pytest.approx(0.5)
    assert m["chi2"] == pytest.approx(1.0)
    assert m["weighted_vr_pct"] == pytest.approx(m["variance_reduction_pct"])


def test_dataset_metrics_perfect_fit():
    obs = np.array([1.0, -2.0, 3.0])
    m = FitStatistics.dataset_metrics(obs, obs)
    assert m["rms"] == pytest.approx(0.0)
    assert m["variance_reduction_pct"] == pytest.approx(100.0)
    assert m["chi2"] == pytest.approx(0.0)
    assert m["correlation"] == pytest.approx(1.0)


def test_dataset_metrics_weighting_and_chi2():
    obs = np.array([0.0, 0.0])
    pred = np.array([2.0, 1.0])          # residual = [-2, -1]
    sigma = np.array([2.0, 1.0])         # weights = [0.5, 1.0]
    m = FitStatistics.dataset_metrics(obs, pred, sigma=sigma, n_params=1)
    # chi2 = (2/2)^2 + (1/1)^2 = 2
    assert m["chi2"] == pytest.approx(2.0)
    # reduced chi2 = chi2 / (n - n_params) = 2 / (2 - 1) = 2
    assert m["reduced_chi2"] == pytest.approx(2.0)
    # wrms = sqrt(sum((w r)^2) / sum(w^2)) = sqrt(2 / 1.25)
    assert m["wrms"] == pytest.approx(np.sqrt(2.0 / 1.25))


def test_dataset_metrics_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        FitStatistics.dataset_metrics(np.zeros(3), np.zeros(4))


def _make_dataset(name, obs, sigma=None):
    n = len(obs)
    coords = np.column_stack([np.arange(n), np.zeros(n), np.zeros(n)]).astype(float)
    uv = np.tile([0, 0, 1], (n, 1)).astype(float)
    return GeodeticDataSet(
        coords=coords, data=np.asarray(obs, float), unit_vecs=uv,
        sigma=np.ones(n) if sigma is None else np.asarray(sigma, float), name=name,
    )


def test_compute_with_precomputed_predicted():
    ds1 = _make_dataset("A", [1.0, 2.0, 3.0])
    ds2 = _make_dataset("B", [0.0, 4.0])
    pred1 = np.array([1.0, 2.0, 3.0])      # perfect
    pred2 = np.array([1.0, 3.0])           # residual = [-1, 1]

    df = FitStatistics.compute([ds1, ds2], predicted=[pred1, pred2])

    assert list(df.index) == ["A", "B", "Overall"]
    assert set(df.columns) >= {"rms", "variance_reduction_pct", "chi2", "n"}
    assert df.loc["A", "rms"] == pytest.approx(0.0)
    assert df.loc["A", "variance_reduction_pct"] == pytest.approx(100.0)

    # Overall aggregates all 5 points: residuals [0,0,0,-1,1] -> ss_res = 2.
    assert df.loc["Overall", "n"] == 5
    assert df.loc["Overall", "residual_norm"] == pytest.approx(np.sqrt(2.0))
    ss_dat = 1 + 4 + 9 + 0 + 16
    assert df.loc["Overall", "variance_reduction_pct"] == pytest.approx(
        (1 - 2.0 / ss_dat) * 100
    )
    # n column is integer-typed.
    assert df["n"].dtype == np.dtype(int) or np.issubdtype(df["n"].dtype, np.integer)


def test_compute_from_engine_matches_precomputed():
    verts = np.array([[0, 0, 0], [2, 0, 0], [0, 0, -1], [2, 0, -1]], float)
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    fault = TriangularFaultMesh(
        (verts, faces),
        strike_slip_type=StrikeSlipType.RIGHT_LATERAL,
        dip_slip_type=DipSlipType.NORMAL,
    )
    engine = CutdeCpuEngine(0.25)
    m = fault.num_patches()
    slip = np.zeros(2 * m)
    slip[:m] = 1.0
    slip[m:] = 0.5

    # Observation points on the surface.
    x = np.linspace(-3, 5, 6)
    y = np.linspace(-3, 5, 6)
    xv, yv = np.meshgrid(x, y)
    coords = np.column_stack([xv.ravel(), yv.ravel(), np.zeros(xv.size)])
    uv = np.tile([0.3, 0.2, 0.9], (coords.shape[0], 1))
    ds = GeodeticDataSet(
        coords=coords, data=np.zeros(coords.shape[0]), unit_vecs=uv,
        sigma=np.ones(coords.shape[0]), name="synthetic",
    )
    # Make the observed data the model prediction plus a small offset.
    pred = engine.build_kernel(fault, ds) @ slip
    ds.data = pred + 0.01

    df_engine = FitStatistics.compute([ds], faults=fault, slip=slip, engine=engine)
    df_pre = FitStatistics.compute([ds], predicted=[pred])

    # Predictions built internally match the direct kernel product.
    for col in ["rms", "variance_reduction_pct", "chi2", "residual_norm"]:
        assert df_engine.loc["synthetic", col] == pytest.approx(df_pre.loc["synthetic", col])

    # Constant 0.01 residual on every point -> rms == 0.01.
    assert df_engine.loc["synthetic", "rms"] == pytest.approx(0.01, abs=1e-9)

    # Overall reduced chi-square uses total unknowns (2M) as dof reduction.
    n = coords.shape[0]
    chi2 = df_engine.loc["Overall", "chi2"]
    assert df_engine.loc["Overall", "reduced_chi2"] == pytest.approx(chi2 / max(1, n - 2 * m))


def test_compute_requires_prediction_inputs():
    ds = _make_dataset("A", [1.0, 2.0])
    with pytest.raises(ValueError, match="Provide either"):
        FitStatistics.compute([ds])


def test_compute_predicted_length_mismatch():
    ds = _make_dataset("A", [1.0, 2.0])
    with pytest.raises(ValueError, match="entries but there are"):
        FitStatistics.compute([ds], predicted=[np.zeros(2), np.zeros(2)])
