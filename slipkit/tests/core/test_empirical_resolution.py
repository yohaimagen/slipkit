import numpy as np
import pytest

from slipkit.core.fault import (
    TriangularFaultMesh,
    SlipComponent,
    CANONICAL_COMPONENT_ORDER,
)
from slipkit.core.data import GeodeticDataSet
from slipkit.core.physics import GreenFunctionBuilder
from slipkit.core.solvers import SolverStrategy
from slipkit.core.inversion import InversionOrchestrator
from slipkit.core.noise import DiagonalNoise
from slipkit.core.resolution import (
    ResolutionAnalyzer,
    SyntheticRecoveryTest,
    MonteCarloResolution,
)


# ---------------------------------------------------------------------------
# Toy fixtures: a controllable engine + an exact linear solver.
# ---------------------------------------------------------------------------

def _strip_mesh(m, spacing=1.0, components=CANONICAL_COMPONENT_ORDER):
    n_v = m + 2
    xs = np.arange(n_v) * (0.5 * spacing)
    ys = np.tile([0.0, spacing], n_v)[:n_v]
    verts = np.column_stack([xs, ys, np.zeros(n_v)]).astype(float)
    faces = np.array([[i, i + 1, i + 2] for i in range(m)], dtype=int)
    return TriangularFaultMesh((verts, faces), slip_components=components)


class _FakeEngine(GreenFunctionBuilder):
    """Returns a prescribed kernel per dataset (ignores physics)."""

    def __init__(self, g_by_name):
        self._g = g_by_name

    def build_kernel(self, fault, dataset):
        return self._g[dataset.name]


class _LstsqSolver(SolverStrategy):
    """Exact, unbounded least-squares solver (linear, for precise comparisons)."""

    def solve(self, A, b, bounds=None):
        m, *_ = np.linalg.lstsq(A, b, rcond=None)
        return m


def _dataset(name, n, sigma=None):
    coords = np.column_stack([np.arange(n), np.zeros(n), np.zeros(n)]).astype(float)
    uv = np.tile([0.0, 0.0, 1.0], (n, 1))
    sig = np.ones(n) if sigma is None else np.asarray(sigma, float)
    return GeodeticDataSet(coords, np.zeros(n), uv, sig, name=name)


def _inversion(g, fault, ds, solver):
    inv = InversionOrchestrator()
    inv.add_fault(fault)
    inv.add_data(ds)
    inv.set_engine(_FakeEngine({ds.name: g}))
    inv.set_solver(solver)
    return inv


# ---------------------------------------------------------------------------
# make_checkerboard
# ---------------------------------------------------------------------------

def test_checkerboard_pattern_values_and_parity():
    fault = _strip_mesh(8, components=[SlipComponent.STRIKE_SLIP])
    ds = _dataset("d", fault.num_patches())
    test = SyntheticRecoveryTest([fault], [ds], _FakeEngine({"d": np.eye(8)}), _LstsqSolver())
    cb = test.make_checkerboard(block_size=1.0, amplitude=2.0, baseline=0.5)
    # Values live in {baseline, baseline + amplitude}.
    assert set(np.unique(cb)).issubset({0.5, 2.5})
    # Both parities present on a strip spanning several blocks.
    assert 0.5 in cb and 2.5 in cb


def test_checkerboard_rejects_infeasible_pattern_under_bounds():
    fault = _strip_mesh(8, components=[SlipComponent.STRIKE_SLIP])
    ds = _dataset("d", fault.num_patches())
    m = fault.num_patches()
    bounds = (np.zeros(m), np.full(m, np.inf))
    test = SyntheticRecoveryTest(
        [fault], [ds], _FakeEngine({"d": np.eye(m)}), _LstsqSolver(), bounds=bounds
    )
    # A +/-A pattern (baseline=-1, amplitude=2 => {-1, +1}) violates slip >= 0.
    with pytest.raises(ValueError, match="unrecoverable"):
        test.make_checkerboard(block_size=1.0, baseline=-1.0, amplitude=2.0)
    # The default 0/A alternation is feasible.
    cb = test.make_checkerboard(block_size=1.0)
    assert np.all(cb >= 0.0)


def test_checkerboard_multicomponent_targets_all_components_by_default():
    fault = _strip_mesh(6)  # SS + DS
    m = fault.num_patches()
    ds = _dataset("d", 2 * m)
    test = SyntheticRecoveryTest(
        [fault], [ds], _FakeEngine({"d": np.eye(2 * m)}), _LstsqSolver()
    )
    cb = test.make_checkerboard(block_size=1.0)
    assert cb.shape == (2 * m,)
    # Same pattern imprinted on both component blocks.
    np.testing.assert_allclose(cb[:m], cb[m:])
    # Restricting to one component leaves the other zero.
    cb_ss = test.make_checkerboard(block_size=1.0, component="ss")
    assert np.any(cb_ss[:m] != 0) and np.all(cb_ss[m:] == 0)


# ---------------------------------------------------------------------------
# SyntheticRecoveryTest.run
# ---------------------------------------------------------------------------

def test_noise_free_smooth_target_recovers_closely():
    fault = _strip_mesh(10, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    g = np.eye(m)
    ds = _dataset("d", m)
    test = SyntheticRecoveryTest([fault], [ds], _FakeEngine({"d": g}), _LstsqSolver())
    smooth = np.linspace(0.2, 1.0, m)  # smooth ramp, cheap to recover
    res = test.run(smooth, lambda_spatial=0.05, noise=False)
    np.testing.assert_allclose(res.recovered.slip_vector, smooth, atol=0.05)
    assert res.metrics["correlation"] > 0.99
    assert res.difference.shape == (m,)
    assert res.synthetic_datasets[0].name == "d_synthetic"


def test_checkerboard_degrades_with_smaller_blocks():
    fault = _strip_mesh(12, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    ds = _dataset("d", m)
    test = SyntheticRecoveryTest(
        [fault], [ds], _FakeEngine({"d": np.eye(m)}), _LstsqSolver()
    )
    lam = 1.0
    # Fine checkerboard (block ~ one patch) is high-frequency => damped.
    fine = test.make_checkerboard(block_size=0.4)
    coarse = test.make_checkerboard(block_size=2.0)
    r_fine = test.run(fine, lam, noise=False)
    r_coarse = test.run(coarse, lam, noise=False)
    # Coarse pattern is recovered better than the fine one.
    assert r_coarse.metrics["rms_difference"] < r_fine.metrics["rms_difference"]
    assert r_coarse.metrics["correlation"] > r_fine.metrics["correlation"]


def test_empirical_psf_matches_analytical_column():
    # Unbounded + noise-free: recovering a unit spike reproduces column j of R.
    fault = _strip_mesh(7, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    rng = np.random.default_rng(0)
    g = np.eye(m) + 0.1 * rng.standard_normal((m, m))
    ds = _dataset("d", m)
    solver = _LstsqSolver()
    lam = 0.3

    test = SyntheticRecoveryTest([fault], [ds], _FakeEngine({"d": g}), solver)
    inv = _inversion(g, fault, ds, solver)
    an = ResolutionAnalyzer.from_orchestrator(inv, lam)

    for j in (0, 3, m - 1):
        spike = test.make_point_source(j)
        res = test.run(spike, lam, noise=False)
        col = an.resolution_kernel(j, kind="column")
        np.testing.assert_allclose(res.recovered.slip_vector, col, atol=1e-8)


# ---------------------------------------------------------------------------
# MonteCarloResolution
# ---------------------------------------------------------------------------

def test_mc_std_matches_analytical_sigma_on_linear_toy():
    fault = _strip_mesh(8, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    rng = np.random.default_rng(1)
    g = rng.standard_normal((14, m))
    ds = _dataset("d", 14)
    solver = _LstsqSolver()
    lam = 0.3

    inv = _inversion(g, fault, ds, solver)
    an = ResolutionAnalyzer.from_orchestrator(inv, lam)
    sigma_analytical = an.model_std()

    mc = MonteCarloResolution(inv, lam, n_jobs=1)
    ens = mc.perturb_data(600, seed=7)
    # Unconstrained linear solver => MC covariance == analytical C_m.
    np.testing.assert_allclose(ens.std, sigma_analytical, rtol=0.15, atol=1e-3)


def test_mc_correlated_noise_runs_and_frees_cache():
    from slipkit.core.noise import EmpiricalInsarNoise
    fault = _strip_mesh(6, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    rng = np.random.default_rng(1)
    g = rng.standard_normal((16, m))
    ds = _dataset("d", 16)
    coords_xy = ds.coords[:, :2]
    nm = EmpiricalInsarNoise(coords_xy, nugget=0.1, sill=0.9, length=3.0, correlated=True)
    inv = _inversion(g, fault, ds, _LstsqSolver())

    mc = MonteCarloResolution(inv, 0.3, n_jobs=1, noise_models=[nm])
    ens = mc.perturb_data(80, seed=0)
    assert ens.samples.shape == (80, m)
    assert np.all(ens.std > 0)
    # perturb_data pre-draws in the parent and frees the dense factor afterwards.
    assert nm._cov is None and nm._chol is None


def test_mc_is_deterministic_and_parallel_matches_serial():
    fault = _strip_mesh(6, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    rng = np.random.default_rng(2)
    g = rng.standard_normal((10, m))
    ds = _dataset("d", 10)
    inv = _inversion(g, fault, ds, _LstsqSolver())

    mc = MonteCarloResolution(inv, 0.2, n_jobs=1)
    a = mc.perturb_data(20, seed=3)
    b = mc.perturb_data(20, seed=3)
    np.testing.assert_array_equal(a.samples, b.samples)

    mc_par = MonteCarloResolution(inv, 0.2, n_jobs=2)
    c = mc_par.perturb_data(20, seed=3)
    # SeedSequence.spawn + ordered map => identical to the serial run.
    np.testing.assert_allclose(c.samples, a.samples, atol=1e-10)


def test_mc_constraint_bias_zero_for_linear_solver():
    fault = _strip_mesh(6, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    rng = np.random.default_rng(4)
    g = rng.standard_normal((12, m))
    ds = _dataset("d", 12)
    inv = _inversion(g, fault, ds, _LstsqSolver())
    mc = MonteCarloResolution(inv, 0.25, n_jobs=1)
    ens = mc.perturb_data(400, seed=5)
    # Unconstrained: <m> - m_hat ~ 0 (only sampling noise remains).
    assert np.max(np.abs(ens.constraint_bias)) < 0.1
    assert ens.running_std().shape == (400,)
    # Running std stabilises: late window varies little.
    rs = ens.running_std()
    assert abs(rs[-1] - rs[-50]) < 0.5 * rs[-1]


def test_mc_percentiles_and_distributions():
    fault = _strip_mesh(5, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    rng = np.random.default_rng(6)
    g = rng.standard_normal((9, m))
    ds = _dataset("d", 9)
    inv = _inversion(g, fault, ds, _LstsqSolver())
    ens = MonteCarloResolution(inv, 0.2, n_jobs=1).perturb_data(50, seed=1)

    p = ens.percentile([16, 84])
    assert p.shape == (2, m)
    assert np.all(p[1] >= p[0])
    assert ens.mean_distribution().slip_vector.shape == (m,)
    assert ens.std_distribution().slip_vector.shape == (m,)


# ---------------------------------------------------------------------------
# Jackknife / bootstrap over datasets
# ---------------------------------------------------------------------------

def _two_dataset_inversion(solver):
    fault = _strip_mesh(6, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    rng = np.random.default_rng(10)
    g1 = rng.standard_normal((8, m))
    g2 = rng.standard_normal((7, m))
    ds1 = _dataset("track1", 8)
    ds2 = _dataset("track2", 7)
    # Give the datasets non-trivial data so leaving one out changes the solution.
    ds1.data = g1 @ np.linspace(0.1, 1.0, m)
    ds2.data = g2 @ np.linspace(1.0, 0.1, m)
    inv = InversionOrchestrator()
    inv.add_fault(fault)
    inv.add_data(ds1)
    inv.add_data(ds2)
    inv.set_engine(_FakeEngine({"track1": g1, "track2": g2}))
    inv.set_solver(solver)
    return inv, m


def test_jackknife_over_datasets():
    inv, m = _two_dataset_inversion(_LstsqSolver())
    ens = MonteCarloResolution(inv, 0.3, n_jobs=1).jackknife_datasets()
    assert ens.samples.shape == (2, m)
    # Leave-one-out solutions differ from each other.
    assert not np.allclose(ens.samples[0], ens.samples[1])


def test_bootstrap_over_datasets():
    inv, m = _two_dataset_inversion(_LstsqSolver())
    ens = MonteCarloResolution(inv, 0.3, n_jobs=1).bootstrap_datasets(15, seed=0)
    assert ens.samples.shape == (15, m)
    assert ens.std.shape == (m,)


def test_jackknife_needs_two_datasets():
    fault = _strip_mesh(4, components=[SlipComponent.STRIKE_SLIP])
    ds = _dataset("d", fault.num_patches())
    inv = _inversion(np.eye(fault.num_patches()), fault, ds, _LstsqSolver())
    with pytest.raises(ValueError, match=">= 2 datasets"):
        MonteCarloResolution(inv, 0.1, n_jobs=1).jackknife_datasets()
