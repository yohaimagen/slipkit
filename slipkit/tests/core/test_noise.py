import numpy as np
import pytest

from slipkit.core.data import GeodeticDataSet
from slipkit.core.noise import (
    DiagonalNoise,
    EmpiricalInsarNoise,
    NoiseModel,
    detrend_plane,
    empirical_variogram,
    estimate_insar_sigma,
    fit_variogram,
    select_quiet_region,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(coords_xy, data, sigma=None, name="ds"):
    n = coords_xy.shape[0]
    coords = np.column_stack([coords_xy[:, 0], coords_xy[:, 1], np.zeros(n)])
    uv = np.tile([0.0, 0.0, 1.0], (n, 1))
    return GeodeticDataSet(
        coords=coords.astype(float),
        data=np.asarray(data, dtype=float),
        unit_vecs=uv,
        sigma=np.ones(n) if sigma is None else np.asarray(sigma, dtype=float),
        name=name,
    )


def _grid(nx, ny, spacing=1.0):
    xs = np.arange(nx) * spacing
    ys = np.arange(ny) * spacing
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def _correlated_field(coords_xy, *, nugget, sill, length, rng):
    """Draws one Gaussian field with an exponential covariance."""
    from scipy.spatial.distance import pdist, squareform

    d = squareform(pdist(coords_xy))
    cov = sill * np.exp(-d / length)
    cov[np.diag_indices_from(cov)] += nugget
    chol = np.linalg.cholesky(cov + 1e-9 * np.eye(coords_xy.shape[0]))
    return chol @ rng.standard_normal(coords_xy.shape[0])


# ---------------------------------------------------------------------------
# DiagonalNoise
# ---------------------------------------------------------------------------

def test_diagonal_from_dataset_reproduces_sigma():
    coords = _grid(3, 3)
    sigma = np.linspace(0.5, 2.0, coords.shape[0])
    ds = _make_dataset(coords, np.zeros(coords.shape[0]), sigma=sigma)

    nm = DiagonalNoise.from_dataset(ds)
    assert isinstance(nm, NoiseModel)
    np.testing.assert_array_equal(nm.sigma(), sigma)
    assert len(nm) == coords.shape[0]
    assert nm.covariance() is None


def test_diagonal_from_dataset_without_sigma_raises():
    coords = _grid(2, 2)
    ds = _make_dataset(coords, np.zeros(coords.shape[0]))
    ds.sigma = None  # simulate a dataset that never got real uncertainties
    with pytest.raises(ValueError, match="no sigma"):
        DiagonalNoise.from_dataset(ds)


def test_diagonal_rejects_nonpositive_sigma():
    with pytest.raises(ValueError):
        DiagonalNoise(np.array([1.0, 0.0, 2.0]))
    with pytest.raises(ValueError):
        DiagonalNoise(np.array([1.0, -1.0]))


def test_diagonal_sample_has_expected_std():
    sigma = np.array([1.0, 3.0, 0.5])
    nm = DiagonalNoise(sigma)
    rng = np.random.default_rng(0)
    draws = np.stack([nm.sample(rng) for _ in range(4000)])
    emp = draws.std(axis=0)
    # Per-column empirical std tracks sigma.
    np.testing.assert_allclose(emp, sigma, rtol=0.1)


def test_diagonal_sample_is_reproducible():
    nm = DiagonalNoise(np.ones(5))
    a = nm.sample(np.random.default_rng(42))
    b = nm.sample(np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Variogram fitting
# ---------------------------------------------------------------------------

def test_empirical_variogram_shapes_and_range():
    coords = _grid(8, 8)
    rng = np.random.default_rng(1)
    vals = rng.standard_normal(coords.shape[0])
    centers, gamma, counts = empirical_variogram(coords, vals, n_bins=10)
    assert centers.ndim == gamma.ndim == counts.ndim == 1
    assert centers.size == gamma.size == counts.size
    assert np.all(counts > 0)
    assert np.all(np.isfinite(gamma))


def test_detrend_plane_removes_ramp():
    coords = _grid(12, 12)
    ramp = 3.0 + 2.0 * coords[:, 0] - 1.5 * coords[:, 1]
    resid = detrend_plane(coords, ramp)
    # A pure plane is removed to ~0.
    assert np.allclose(resid, 0.0, atol=1e-9)
    # A plane + noise leaves noise with no residual planar correlation.
    rng = np.random.default_rng(0)
    noisy = ramp + rng.standard_normal(coords.shape[0])
    r2 = detrend_plane(coords, noisy)
    assert abs(np.corrcoef(r2, coords[:, 0])[0, 1]) < 0.2
    assert abs(r2.mean()) < 0.3


def test_fit_variogram_caps_length_and_warns_on_ramp():
    # A pure ramp is non-stationary: its variogram grows without a sill, so the
    # fit must NOT run L off to infinity -- it is capped at the region extent and
    # a warning is raised.
    coords = _grid(20, 20)
    ramp = 0.5 * coords[:, 0]              # linear, never saturates
    with pytest.warns(UserWarning, match="saturate"):
        nugget, sill, length = fit_variogram(coords, ramp, n_bins=15)
    # L is bounded by the fitting extent (a physical constraint), not millions.
    centers, _, _ = empirical_variogram(coords, ramp, n_bins=15)
    assert length <= centers.max() + 1e-9


def test_fit_variogram_recovers_injected_parameters():
    coords = _grid(30, 30, spacing=1.0)
    nugget_true, sill_true, length_true = 0.2, 1.0, 5.0
    rng = np.random.default_rng(7)
    # Average several realizations to stabilise the empirical variogram.
    fits = []
    for k in range(6):
        field = _correlated_field(
            coords, nugget=nugget_true, sill=sill_true, length=length_true,
            rng=np.random.default_rng(100 + k),
        )
        fits.append(fit_variogram(coords, field, n_bins=20, max_dist=20.0))
    fits = np.array(fits)
    nugget, sill, length = fits.mean(axis=0)

    total = nugget + sill
    assert total == pytest.approx(nugget_true + sill_true, rel=0.35)
    assert length == pytest.approx(length_true, rel=0.6)


# ---------------------------------------------------------------------------
# Quiet-region selection
# ---------------------------------------------------------------------------

def test_select_quiet_region_bbox():
    coords = _grid(5, 5)
    ds = _make_dataset(coords, np.zeros(coords.shape[0]))
    mask = select_quiet_region(ds, region=(0.0, 0.0, 1.0, 1.0))
    # bbox [0,1]x[0,1] on a unit grid selects the 2x2 lower-left corner.
    assert mask.sum() == 4
    assert np.all(coords[mask, 0] <= 1.0)


def test_select_quiet_region_boolean_mask_passthrough():
    coords = _grid(3, 3)
    ds = _make_dataset(coords, np.zeros(coords.shape[0]))
    m = np.zeros(coords.shape[0], dtype=bool)
    m[:3] = True
    out = select_quiet_region(ds, region=m)
    np.testing.assert_array_equal(out, m)


def test_select_quiet_region_polygon():
    coords = _grid(5, 5)
    ds = _make_dataset(coords, np.zeros(coords.shape[0]))
    poly = np.array([[-0.5, -0.5], [2.5, -0.5], [2.5, 2.5], [-0.5, 2.5]])
    mask = select_quiet_region(ds, region=poly)
    assert mask.sum() == 9  # the 3x3 lower-left block


def test_select_quiet_region_auto_requires_faults():
    coords = _grid(4, 4)
    ds = _make_dataset(coords, np.zeros(coords.shape[0]))
    with pytest.raises(ValueError, match="fault geometry"):
        select_quiet_region(ds)


def test_select_quiet_region_auto_picks_far_low_points():
    # Near-fault points carry big signal; far points are quiet.
    coords = _grid(10, 1, spacing=1.0)
    data = np.where(coords[:, 0] < 3, 10.0, 0.05)
    ds = _make_dataset(coords, data)

    class _FakeFault:
        def get_centroids(self):
            return np.array([[0.0, 0.0, 0.0]])

    # Duck-typed fault passed as a list, so select_quiet_region iterates it and
    # only calls get_centroids (no AbstractFaultModel isinstance dependency).
    mask = select_quiet_region(ds, faults=[_FakeFault()], quantile=0.6)
    assert mask.sum() > 0
    # All selected points are in the far, low-signal half.
    assert np.all(coords[mask, 0] >= 3)


# ---------------------------------------------------------------------------
# EmpiricalInsarNoise
# ---------------------------------------------------------------------------

def test_empirical_insar_from_quiet_region_recovers_sigma_and_length():
    # Domain spans ~17 correlation lengths per side, so a single realization
    # carries enough independent structure for a stable variogram fit.
    coords = _grid(52, 52, spacing=1.0)
    nugget_true, sill_true, length_true = 0.1, 0.9, 3.0
    field = _correlated_field(
        coords, nugget=nugget_true, sill=sill_true, length=length_true,
        rng=np.random.default_rng(3),
    )
    ds = _make_dataset(coords, field, name="track")

    # Use the whole scene as the quiet region (true signal is zero everywhere).
    mask = np.ones(coords.shape[0], dtype=bool)
    nm = EmpiricalInsarNoise.from_quiet_region(
        ds, region=mask, n_bins=20, max_dist=15.0
    )

    total_true = np.sqrt(nugget_true + sill_true)
    assert nm.sigma()[0] == pytest.approx(total_true, rel=0.4)
    assert np.all(nm.sigma() == nm.sigma()[0])  # constant amplitude
    assert nm.length == pytest.approx(length_true, rel=0.7)


def test_from_reference_fits_on_reference_applies_to_target():
    # Full-resolution reference: fine grid with a known correlated field.
    ref_coords = _grid(40, 40, spacing=1.0)
    nugget_true, sill_true, length_true = 0.1, 0.9, 3.0
    field = _correlated_field(
        ref_coords, nugget=nugget_true, sill=sill_true, length=length_true,
        rng=np.random.default_rng(4),
    )
    ref = _make_dataset(ref_coords, field, name="track_full")

    # Downsampled target: a coarse, differently-spaced subset of points.
    tgt_coords = _grid(9, 9, spacing=4.0)
    tgt = _make_dataset(tgt_coords, np.zeros(tgt_coords.shape[0]), name="track_ds")

    nm = EmpiricalInsarNoise.from_reference(
        tgt, ref, region=np.ones(ref_coords.shape[0], dtype=bool),
        n_bins=18, max_dist=12.0,
    )
    # Parameters fitted from the reference...
    assert nm.length == pytest.approx(length_true, rel=0.7)
    assert nm.sigma()[0] == pytest.approx(np.sqrt(nugget_true + sill_true), rel=0.4)
    # ...but the model is sized to the TARGET points.
    assert len(nm.sigma()) == len(tgt)
    np.testing.assert_array_equal(nm._coords_xy, tgt_coords)


def test_diagonal_sample_batch_shape_and_std():
    sigma = np.array([1.0, 3.0, 0.5])
    nm = DiagonalNoise(sigma)
    batch = nm.sample_batch(np.random.default_rng(0), 3000)
    assert batch.shape == (3, 3000)
    np.testing.assert_allclose(batch.std(axis=1), sigma, rtol=0.1)


def test_correlated_sample_batch_matches_covariance_and_frees_cache():
    coords = _grid(7, 7, spacing=1.0)
    nm = EmpiricalInsarNoise(coords, nugget=0.2, sill=0.8, length=2.5, correlated=True)
    target = nm.covariance()
    batch = nm.sample_batch(np.random.default_rng(3), 6000)  # (49, 6000)
    assert batch.shape == (coords.shape[0], 6000)
    emp = np.cov(batch, rowvar=True)  # rows are points
    assert np.linalg.norm(emp - target) / np.linalg.norm(target) < 0.15
    # free_cache releases the O(N^2) factor.
    nm.free_cache()
    assert nm._cov is None and nm._chol is None


def test_from_reference_thins_large_reference():
    # Reference bigger than max_points exercises the thinning path.
    ref_coords = _grid(45, 45, spacing=1.0)          # 2025 points
    rng = np.random.default_rng(2)
    sigma_true = 1.2
    ref = _make_dataset(ref_coords, sigma_true * rng.standard_normal(ref_coords.shape[0]))
    tgt_coords = _grid(8, 8, spacing=5.0)
    tgt = _make_dataset(tgt_coords, np.zeros(tgt_coords.shape[0]), name="ds")

    # bbox over the whole scene (not a per-point bool mask) => thinning applies,
    # and no low-|LOS| filtering that would bias a pure-noise field's amplitude.
    bbox = (-1.0, -1.0, 45.0, 45.0)
    nm = EmpiricalInsarNoise.from_reference(
        tgt, ref, region=bbox, max_points=600, n_bins=15,
    )
    assert len(nm.sigma()) == len(tgt)
    # White-noise reference => total sigma ~ injected amplitude despite thinning.
    assert nm.sigma()[0] == pytest.approx(sigma_true, rel=0.3)


def test_empirical_insar_covariance_structure():
    coords = _grid(6, 6)
    nm = EmpiricalInsarNoise(
        coords, nugget=0.25, sill=0.75, length=2.0, correlated=True
    )
    cov = nm.covariance()
    assert cov is not None
    n = coords.shape[0]
    assert cov.shape == (n, n)
    # Symmetric, PSD, correct diagonal (nugget + sill == sigma^2).
    np.testing.assert_allclose(cov, cov.T)
    np.testing.assert_allclose(np.diag(cov), nm.sigma() ** 2)
    assert np.linalg.eigvalsh(cov).min() > 0


def test_empirical_insar_uncorrelated_has_no_full_covariance():
    coords = _grid(4, 4)
    nm = EmpiricalInsarNoise(
        coords, nugget=1.0, sill=0.0, length=1.0, correlated=True
    )
    assert nm.covariance() is None  # sill == 0 => diagonal
    nm2 = EmpiricalInsarNoise(
        coords, nugget=0.5, sill=0.5, length=1.0, correlated=False
    )
    assert nm2.covariance() is None  # correlated=False => diagonal


def test_correlated_sample_matches_covariance():
    coords = _grid(7, 7, spacing=1.0)
    nm = EmpiricalInsarNoise(
        coords, nugget=0.2, sill=0.8, length=2.5, correlated=True
    )
    rng = np.random.default_rng(11)
    draws = np.stack([nm.sample(rng) for _ in range(6000)])
    emp_cov = np.cov(draws, rowvar=False)
    target = nm.covariance()
    # Empirical covariance tracks the model covariance.
    assert np.linalg.norm(emp_cov - target) / np.linalg.norm(target) < 0.15


def test_correlated_std_exceeds_independent_for_smooth_operator():
    # The point of the correlated model (plan 1.4): a smoothing/averaging
    # operator applied to correlated noise has larger output variance than the
    # same operator on independent noise of equal per-point sigma.
    coords = _grid(12, 12, spacing=1.0)
    sigma_scalar = 1.0
    corr = EmpiricalInsarNoise(
        coords, nugget=0.1, sill=0.9, length=4.0, correlated=True
    )
    indep = DiagonalNoise(np.full(coords.shape[0], sigma_scalar))
    # Same per-point sigma for a fair comparison.
    np.testing.assert_allclose(corr.sigma(), indep.sigma(), rtol=1e-12)

    # A block-average "operator": mean over all points (maximally smooth).
    rng = np.random.default_rng(5)
    corr_means = [corr.sample(rng).mean() for _ in range(2000)]
    indep_means = [indep.sample(rng).mean() for _ in range(2000)]
    assert np.std(corr_means) > 3.0 * np.std(indep_means)


def test_empirical_insar_sample_reproducible():
    coords = _grid(4, 4)
    nm = EmpiricalInsarNoise(coords, nugget=0.3, sill=0.7, length=2.0)
    a = nm.sample(np.random.default_rng(9))
    b = nm.sample(np.random.default_rng(9))
    np.testing.assert_array_equal(a, b)


def test_empirical_insar_invalid_params():
    coords = _grid(3, 3)
    with pytest.raises(ValueError):
        EmpiricalInsarNoise(coords, nugget=-1.0, sill=1.0, length=1.0)
    with pytest.raises(ValueError):
        EmpiricalInsarNoise(coords, nugget=1.0, sill=1.0, length=0.0)
    with pytest.raises(ValueError):
        EmpiricalInsarNoise(coords, nugget=0.0, sill=0.0, length=1.0)


# ---------------------------------------------------------------------------
# estimate_insar_sigma
# ---------------------------------------------------------------------------

def test_estimate_insar_sigma_recovers_amplitude():
    coords = _grid(20, 20)
    rng = np.random.default_rng(2)
    sigma_true = 1.5
    ds = _make_dataset(coords, sigma_true * rng.standard_normal(coords.shape[0]))
    est = estimate_insar_sigma(ds, region=np.ones(coords.shape[0], dtype=bool))
    assert est == pytest.approx(sigma_true, rel=0.1)
