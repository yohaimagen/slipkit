import numpy as np
import pytest

from slipkit.core.fault import (
    TriangularFaultMesh,
    SlipComponent,
    CANONICAL_COMPONENT_ORDER,
)
from slipkit.core.data import GeodeticDataSet
from slipkit.core.physics import CutdeCpuEngine
from slipkit.core.inversion import InversionOrchestrator
from slipkit.core.resolution import ResolutionAnalyzer


# ---------------------------------------------------------------------------
# Toy meshes / operators
# ---------------------------------------------------------------------------

def _strip_mesh(m, spacing=1.0, components=CANONICAL_COMPONENT_ORDER):
    """A connected zig-zag strip of ``m`` triangles (a path-graph Laplacian)."""
    n_v = m + 2
    xs = np.arange(n_v) * (0.5 * spacing)
    ys = np.tile([0.0, spacing], n_v)[:n_v]
    verts = np.column_stack([xs, ys, np.zeros(n_v)]).astype(float)
    faces = np.array([[i, i + 1, i + 2] for i in range(m)], dtype=int)
    return TriangularFaultMesh((verts, faces), slip_components=components)


def _analyzer(g, fault, lam, **kw):
    l_base = fault.get_smoothing_matrix()
    # Repeat the Laplacian once per active component (mirrors LaplacianSmoothing).
    from scipy.sparse import block_diag
    l_block = block_diag([l_base] * fault.num_components(), format="csr")
    return ResolutionAnalyzer(g, l_block, lam, [fault], **kw)


# ---------------------------------------------------------------------------
# Identity / limiting-lambda behaviour
# ---------------------------------------------------------------------------

def test_identity_operator_gives_identity_resolution():
    fault = _strip_mesh(6, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    g = np.eye(m)                       # perfectly-conditioned, one obs per patch
    an = _analyzer(g, fault, lam=0.0)

    r = an.model_resolution()
    np.testing.assert_allclose(r, np.eye(m), atol=1e-9)
    np.testing.assert_allclose(an.resolution_diagonal(), np.ones(m), atol=1e-9)
    assert an.resolved_parameter_count() == pytest.approx(m)
    # An identity kernel has no spread.
    np.testing.assert_allclose(an.spread_length(), np.zeros(m), atol=1e-9)


def test_large_lambda_collapses_to_laplacian_nullspace():
    # A Laplacian has a 1-D null space (the constant mode), so heavy smoothing
    # collapses the estimate to the mesh mean: trace(R) -> 1 (the null-space
    # dimension) and diag(R) -> 1/M uniformly, NOT 0 (that would need a
    # full-rank damping term).
    fault = _strip_mesh(6, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    g = np.eye(m)
    an = _analyzer(g, fault, lam=1e5)

    diag = an.resolution_diagonal()
    np.testing.assert_allclose(diag, np.full(m, 1.0 / m), atol=1e-3)
    assert an.resolved_parameter_count() == pytest.approx(1.0, abs=1e-2)


def test_zero_lambda_wellposed_resolution_is_full():
    fault = _strip_mesh(5, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    rng = np.random.default_rng(0)
    g = np.eye(m) + 0.01 * rng.standard_normal((m, m))  # still well-conditioned
    an = _analyzer(g, fault, lam=0.0)
    np.testing.assert_allclose(an.resolution_diagonal(), np.ones(m), atol=1e-6)
    assert an.resolved_parameter_count() == pytest.approx(m, abs=1e-6)


# ---------------------------------------------------------------------------
# Covariance sanity
# ---------------------------------------------------------------------------

def test_covariance_symmetric_and_psd():
    fault = _strip_mesh(6, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    rng = np.random.default_rng(1)
    g = rng.standard_normal((10, m))
    an = _analyzer(g, fault, lam=0.4)

    c = an.model_covariance()
    np.testing.assert_allclose(c, c.T, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(c)) > -1e-10
    assert np.all(np.diag(c) >= 0.0)
    np.testing.assert_allclose(an.model_std(), np.sqrt(np.diag(c)), atol=1e-12)


def test_correlation_has_unit_diagonal():
    fault = _strip_mesh(5, components=[SlipComponent.STRIKE_SLIP])
    rng = np.random.default_rng(2)
    g = rng.standard_normal((12, fault.num_patches()))
    an = _analyzer(g, fault, lam=0.3)
    corr = an.model_correlation()
    np.testing.assert_allclose(np.diag(corr), np.ones(fault.num_patches()), atol=1e-10)
    assert np.all(np.abs(corr) <= 1.0 + 1e-9)


# ---------------------------------------------------------------------------
# Column-PSF validation and row != column
# ---------------------------------------------------------------------------

def test_column_psf_matches_unbounded_spike_recovery():
    # For an unbounded, noise-free spike test, recovering m_true = e_j gives
    # exactly R[:, j] (the column PSF). This is the analytic/empirical bridge.
    fault = _strip_mesh(6, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    rng = np.random.default_rng(3)
    g = rng.standard_normal((14, m))
    lam = 0.35
    an = _analyzer(g, fault, lam=lam)
    r = an.model_resolution()

    l_block = an.l_base
    for j in (0, 2, m - 1):
        m_true = np.zeros(m)
        m_true[j] = 1.0
        d_w = g @ m_true
        a_aug = np.vstack([g, lam * l_block])
        b_aug = np.concatenate([d_w, np.zeros(l_block.shape[0])])
        m_hat, *_ = np.linalg.lstsq(a_aug, b_aug, rcond=None)
        np.testing.assert_allclose(m_hat, r[:, j], atol=1e-8)
        # The public accessor returns the same column.
        np.testing.assert_allclose(
            an.resolution_kernel(j, kind="column"), r[:, j], atol=1e-12
        )


def test_rows_differ_from_columns():
    fault = _strip_mesh(6, components=[SlipComponent.STRIKE_SLIP])
    rng = np.random.default_rng(4)
    g = rng.standard_normal((9, fault.num_patches()))
    an = _analyzer(g, fault, lam=0.5)
    r = an.model_resolution()
    # R is not symmetric in general -- guard against silently symmetrizing.
    assert not np.allclose(r, r.T)
    row = an.resolution_kernel(2, kind="row")
    col = an.resolution_kernel(2, kind="column")
    assert not np.allclose(row, col)
    np.testing.assert_allclose(row, r[2, :])
    np.testing.assert_allclose(col, r[:, 2])


# ---------------------------------------------------------------------------
# Data resolution
# ---------------------------------------------------------------------------

def test_data_resolution_diagonal_matches_full():
    fault = _strip_mesh(5, components=[SlipComponent.STRIKE_SLIP])
    rng = np.random.default_rng(5)
    g = rng.standard_normal((11, fault.num_patches()))
    an = _analyzer(g, fault, lam=0.2)
    full = an.data_resolution()
    np.testing.assert_allclose(
        an.data_resolution_diagonal(), np.diag(full), atol=1e-12
    )
    # trace(N_d) == trace(R) (both equal the effective resolved-parameter count).
    assert np.trace(full) == pytest.approx(an.resolved_parameter_count())


# ---------------------------------------------------------------------------
# Component-aware slicing (two-component fault)
# ---------------------------------------------------------------------------

def test_component_slicing_and_leakage():
    fault = _strip_mesh(4)  # both SS and DS
    m = fault.num_patches()
    p = 2 * m
    rng = np.random.default_rng(6)
    g = rng.standard_normal((20, p))
    an = _analyzer(g, fault, lam=0.3)
    diag_full = np.diag(an.model_resolution())

    ss = an.resolution_diagonal(component="ss")
    ds = an.resolution_diagonal(component="ds")
    assert ss.shape == (m,) and ds.shape == (m,)
    np.testing.assert_allclose(ss, diag_full[:m])
    np.testing.assert_allclose(ds, diag_full[m:])
    # component=None returns the whole fault block.
    assert an.resolution_diagonal().shape == (p,)

    leak = an.cross_component_leakage()
    assert leak.shape == (2, m)
    assert np.all(leak >= 0.0)

    # spread_length must be told which component on a 2-component fault.
    with pytest.raises(ValueError, match="specify `component`"):
        an.spread_length()
    sp = an.spread_length(component="ss")
    assert sp.shape == (m,)

    assert an.model_std(component="ds").shape == (m,)


def test_spread_masks_low_resolution_patches():
    fault = _strip_mesh(6, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    g = np.eye(m)
    an = _analyzer(g, fault, lam=5e4)  # collapses to mean => diag(R) ~ 1/M
    # Threshold above the smoothed plateau masks every patch to NaN.
    sp = an.spread_length(mask_below=0.5)
    assert np.all(np.isnan(sp))
    # A threshold below the plateau masks nothing.
    sp2 = an.spread_length(mask_below=1e-3)
    assert not np.any(np.isnan(sp2))


def test_bad_component_and_patch_index_raise():
    fault = _strip_mesh(4, components=[SlipComponent.STRIKE_SLIP])
    g = np.eye(fault.num_patches())
    an = _analyzer(g, fault, lam=0.1)
    with pytest.raises(ValueError, match="not active"):
        an.resolution_diagonal(component="ds")
    with pytest.raises(IndexError):
        an.resolution_kernel(99)


# ---------------------------------------------------------------------------
# from_orchestrator extraction
# ---------------------------------------------------------------------------

def _orchestrator_setup():
    vertices = np.array(
        [
            [0, 0, -5000],
            [2000, 0, -5000],
            [0, 0, -6000],
            [2000, 0, -6000],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    fault = TriangularFaultMesh(
        (vertices, faces), slip_components=[SlipComponent.STRIKE_SLIP]
    )
    engine = CutdeCpuEngine(poisson_ratio=0.25)

    x = np.linspace(-5000, 7000, 5)
    y = np.linspace(-5000, 7000, 5)
    xv, yv = np.meshgrid(x, y)
    n = xv.size
    coords = np.column_stack([xv.ravel(), yv.ravel(), np.zeros(n)])
    uv = np.tile([0.0, 0.0, 1.0], (n, 1))
    sigma = 0.5 * np.ones(n)
    data = np.zeros(n)  # values don't matter for R/C_m
    ds = GeodeticDataSet(coords, data, uv, sigma, name="synthetic")

    inv = InversionOrchestrator()
    inv.add_fault(fault)
    inv.add_data(ds)
    inv.set_engine(engine)
    return inv, fault, engine, ds


def test_from_orchestrator_extracts_weighted_operator():
    inv, fault, engine, ds = _orchestrator_setup()
    lam = 0.7
    an = ResolutionAnalyzer.from_orchestrator(inv, lam)

    # Shapes agree with the problem size.
    assert an.n_params == fault.num_patches()
    assert an.n_data == len(ds)

    # G_w extracted by from_orchestrator equals (1/sigma) * raw kernel.
    g_raw = engine.build_kernel(fault, ds)
    g_w_expected = g_raw / ds.sigma[:, None]
    np.testing.assert_allclose(an.g_weighted, g_w_expected, rtol=1e-10, atol=1e-12)

    # Resolution diagonal is a valid 0..1 map; C_m is symmetric PSD.
    diag = an.resolution_diagonal()
    assert np.all(diag > -1e-9) and np.all(diag < 1.0 + 1e-9)
    c = an.model_covariance()
    np.testing.assert_allclose(c, c.T, atol=1e-10)

    # The unconstrained estimate matches a direct regularized least squares on
    # the extracted (G_w, d_w, L) -- confirms d_w was extracted correctly too.
    a_aug = np.vstack([an.g_weighted, lam * an.l_base])
    b_aug = np.concatenate([an.d_weighted, np.zeros(an.l_base.shape[0])])
    m_direct, *_ = np.linalg.lstsq(a_aug, b_aug, rcond=None)
    np.testing.assert_allclose(an.model_estimate(), m_direct, atol=1e-8)


def test_large_matrix_guard_warns():
    fault = _strip_mesh(3, components=[SlipComponent.STRIKE_SLIP])
    g = np.eye(fault.num_patches())
    an = _analyzer(g, fault, lam=0.1, max_dense_p=2)
    with pytest.warns(UserWarning, match="max_dense_p"):
        an.model_resolution()


# ---------------------------------------------------------------------------
# Optional TSVD / Picard
# ---------------------------------------------------------------------------

def test_singular_values_descending_and_match_numpy():
    fault = _strip_mesh(6, components=[SlipComponent.STRIKE_SLIP])
    rng = np.random.default_rng(0)
    g = rng.standard_normal((12, fault.num_patches()))
    an = _analyzer(g, fault, lam=0.3)
    s = an.singular_values()
    assert np.all(np.diff(s) <= 1e-12)  # descending
    np.testing.assert_allclose(s, np.linalg.svd(g, compute_uv=False))


def test_tsvd_full_rank_is_identity_and_diag_grows_with_k():
    fault = _strip_mesh(6, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    rng = np.random.default_rng(1)
    g = rng.standard_normal((m, m))  # square, full rank
    an = _analyzer(g, fault, lam=0.3)
    # Retaining all modes reconstructs the identity (V is orthogonal).
    full = an.tsvd_model_resolution(m)
    np.testing.assert_allclose(full, np.eye(m), atol=1e-9)
    # trace(V_k V_k^T) == k, and grows monotonically with k.
    traces = [np.trace(an.tsvd_model_resolution(k)) for k in range(1, m + 1)]
    np.testing.assert_allclose(traces, np.arange(1, m + 1), atol=1e-9)
    with pytest.raises(ValueError):
        an.tsvd_model_resolution(m + 1)


def test_picard_coefficients_shapes_and_ratio():
    fault = _strip_mesh(6, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    rng = np.random.default_rng(2)
    g = rng.standard_normal((10, m))
    d_w = rng.standard_normal(10)
    an = _analyzer(g, fault, lam=0.3, d_weighted=d_w)
    s, coeffs, ratios = an.picard_coefficients()
    assert s.shape == coeffs.shape == ratios.shape == (m,)
    np.testing.assert_allclose(ratios, coeffs / s)


def test_picard_without_data_raises():
    fault = _strip_mesh(4, components=[SlipComponent.STRIKE_SLIP])
    g = np.eye(fault.num_patches())
    an = _analyzer(g, fault, lam=0.1)
    with pytest.raises(ValueError, match="weighted data"):
        an.picard_coefficients()
