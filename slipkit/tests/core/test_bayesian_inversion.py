"""
Unit tests for the ALTar Bayesian slip inversion module.

Tests cover:
- AltarAssembler: kernel stacking, caching, multi-fault/dataset consistency.
- AltarDataExporter: HDF5 file shapes and covariance construction.
- AltarConfigBuilder: .pfg file syntax and attribute correctness.
- AltarResultImporter: mock HDF5 parsing into AltarSlipDistribution.
- AltarSlipDistribution: posterior statistics and convergence flag.

Integration tests that require ALTar to be installed are marked with
``pytest.mark.integration`` and are skipped by default.
"""

import os
import tempfile

import h5py
import numpy as np
import pandas as pd
import pytest

from slipkit.core.bayesian.assembler import AltarAssembler
from slipkit.core.bayesian.config import AltarConfigBuilder
from slipkit.core.bayesian.exporter import AltarDataExporter
from slipkit.core.bayesian.importer import AltarResultImporter
from slipkit.core.bayesian.results import AltarSlipDistribution
from slipkit.core.data import GeodeticDataSet
from slipkit.core.fault import TriangularFaultMesh
from slipkit.core.inversion import InversionOrchestrator
from slipkit.core.physics import CutdeCpuEngine


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_fault():
    """Single-triangle fault at 5 km depth."""
    vertices = np.array(
        [[0, 0, -5000], [2000, 0, -5000], [1000, 1732, -5000]], dtype=float
    )
    faces = np.array([[0, 1, 2]], dtype=int)
    return TriangularFaultMesh((vertices, faces))


@pytest.fixture
def synthetic_dataset(simple_fault):
    """Synthetic 3-component GNSS dataset for the simple fault."""
    engine = CutdeCpuEngine()
    n_pts = 9
    x = np.linspace(-2000, 4000, 3)
    y = np.linspace(-2000, 4000, 3)
    xv, yv = np.meshgrid(x, y)
    obs_xy = np.column_stack([xv.flatten(), yv.flatten()])

    coords = np.repeat(
        np.column_stack([obs_xy, np.zeros(n_pts)]), 3, axis=0
    )
    unit_vecs = np.tile(np.eye(3), (n_pts, 1))
    sigma = np.ones(n_pts * 3) * 0.01

    dummy_ds = GeodeticDataSet(
        coords=coords,
        data=np.zeros(n_pts * 3),
        unit_vecs=unit_vecs,
        sigma=sigma,
        name="dummy",
    )

    G_raw = engine.build_kernel(simple_fault, dummy_ds)
    true_slip = np.array([1.0, 0.5])
    clean_data = G_raw @ true_slip
    noise = np.random.default_rng(0).normal(0, 0.005, size=clean_data.shape)

    return GeodeticDataSet(
        coords=coords,
        data=clean_data + noise,
        unit_vecs=unit_vecs,
        sigma=sigma,
        name="synth",
    )


@pytest.fixture
def areas(simple_fault):
    return simple_fault.get_areas()


# ---------------------------------------------------------------------------
# AltarAssembler tests
# ---------------------------------------------------------------------------

class TestAltarAssembler:

    def test_output_shapes(self, simple_fault, synthetic_dataset):
        assembler = AltarAssembler()
        A, b = assembler.assemble(
            [simple_fault], [synthetic_dataset],
            CutdeCpuEngine(), None, 0.0
        )
        n_obs = len(synthetic_dataset.data)
        n_patches = simple_fault.num_patches()

        assert A.shape == (n_obs, 2 * n_patches)
        assert b.shape == (2 * n_obs,)

    def test_b_layout(self, simple_fault, synthetic_dataset):
        """b must be [d_obs | sigma] with correct values."""
        assembler = AltarAssembler()
        A, b = assembler.assemble(
            [simple_fault], [synthetic_dataset],
            CutdeCpuEngine(), None, 0.0
        )
        n_obs = len(synthetic_dataset.data)
        np.testing.assert_array_equal(b[:n_obs], synthetic_dataset.data)
        np.testing.assert_array_equal(b[n_obs:], synthetic_dataset.sigma)

    def test_a_no_rotation(self, simple_fault, synthetic_dataset):
        """G columns must match raw engine output (no rake rotation)."""
        engine = CutdeCpuEngine()
        assembler = AltarAssembler()
        A, _ = assembler.assemble(
            [simple_fault], [synthetic_dataset], engine, None, 0.0
        )
        G_raw = engine.build_kernel(simple_fault, synthetic_dataset)
        np.testing.assert_allclose(A, G_raw, atol=1e-10)

    def test_caching(self, simple_fault, synthetic_dataset):
        """Second call must return the same object (cache hit)."""
        assembler = AltarAssembler()
        engine = CutdeCpuEngine()
        A1, _ = assembler.assemble(
            [simple_fault], [synthetic_dataset], engine, None, 0.0
        )
        cached = assembler._G_raw_cache
        A2, _ = assembler.assemble(
            [simple_fault], [synthetic_dataset], engine, None, 0.0
        )
        assert assembler._G_raw_cache is cached

    def test_force_recompute(self, simple_fault, synthetic_dataset):
        """force_recompute=True must regenerate the cache."""
        assembler = AltarAssembler()
        engine = CutdeCpuEngine()
        assembler.assemble([simple_fault], [synthetic_dataset], engine, None, 0.0)
        old = assembler._G_raw_cache
        assembler.assemble(
            [simple_fault], [synthetic_dataset], engine, None, 0.0,
            force_recompute=True
        )
        assert assembler._G_raw_cache is not old

    def test_get_areas(self, simple_fault, areas):
        assembler = AltarAssembler()
        result = assembler.get_areas([simple_fault])
        np.testing.assert_allclose(result, areas)

    def test_multi_dataset_stacking(self, simple_fault, synthetic_dataset):
        """Two identical datasets must produce doubled N_obs."""
        assembler = AltarAssembler()
        engine = CutdeCpuEngine()
        A, b = assembler.assemble(
            [simple_fault], [synthetic_dataset, synthetic_dataset],
            engine, None, 0.0
        )
        n_obs_single = len(synthetic_dataset.data)
        assert A.shape[0] == 2 * n_obs_single
        assert b.shape[0] == 2 * (2 * n_obs_single)


# ---------------------------------------------------------------------------
# AltarDataExporter tests
# ---------------------------------------------------------------------------

class TestAltarDataExporter:

    def test_green_shape(self, simple_fault, synthetic_dataset, tmp_path):
        engine = CutdeCpuEngine()
        G = engine.build_kernel(simple_fault, synthetic_dataset)
        exporter = AltarDataExporter(str(tmp_path))
        path = exporter.export_greens_function(G)

        with h5py.File(path) as f:
            assert f["green"].shape == G.shape

    def test_data_shape(self, synthetic_dataset, tmp_path):
        exporter = AltarDataExporter(str(tmp_path))
        path = exporter.export_data(synthetic_dataset.data)

        with h5py.File(path) as f:
            assert f["data"].shape == (len(synthetic_dataset.data),)

    def test_cd_diagonal_only(self, synthetic_dataset, tmp_path):
        """Without Cp, Cd should be diag(sigma^2)."""
        exporter = AltarDataExporter(str(tmp_path))
        path = exporter.export_covariance(
            synthetic_dataset.sigma, synthetic_dataset.data, alpha_cp=0.0
        )
        with h5py.File(path) as f:
            cx = np.array(f["cd"])
        expected_diag = synthetic_dataset.sigma ** 2
        np.testing.assert_allclose(np.diag(cx), expected_diag, rtol=1e-10)
        off_diag = cx - np.diag(np.diag(cx))
        assert np.allclose(off_diag, 0.0)

    def test_cd_with_cp(self, synthetic_dataset, tmp_path):
        """With alpha_cp, diagonal must include Cp contribution."""
        alpha = 0.1
        exporter = AltarDataExporter(str(tmp_path))
        path = exporter.export_covariance(
            synthetic_dataset.sigma, synthetic_dataset.data, alpha_cp=alpha
        )
        with h5py.File(path) as f:
            cx = np.array(f["cd"])
        expected = synthetic_dataset.sigma ** 2 + (alpha * synthetic_dataset.data) ** 2
        np.testing.assert_allclose(np.diag(cx), expected, rtol=1e-10)

    def test_areas_km2_conversion(self, simple_fault, tmp_path):
        areas_m2 = simple_fault.get_areas()
        exporter = AltarDataExporter(str(tmp_path))
        path = exporter.export_areas(areas_m2)

        with h5py.File(path) as f:
            stored = np.array(f["areas"])
        np.testing.assert_allclose(stored, areas_m2 / 1e6, rtol=1e-10)

    def test_export_all_creates_files(self, simple_fault, synthetic_dataset, tmp_path):
        engine = CutdeCpuEngine()
        G = engine.build_kernel(simple_fault, synthetic_dataset)
        exporter = AltarDataExporter(str(tmp_path))
        paths = exporter.export_all(
            G,
            synthetic_dataset.data,
            synthetic_dataset.sigma,
            simple_fault.get_areas(),
        )
        for key in ("green", "data", "cd", "areas"):
            assert os.path.isfile(paths[key]), f"Missing file for '{key}'"


# ---------------------------------------------------------------------------
# AltarConfigBuilder tests
# ---------------------------------------------------------------------------

class TestAltarConfigBuilder:

    def _make_builder(self, tmp_path, **kwargs):
        defaults = dict(
            n_patches=9,
            n_observations=108,
            case_dir=str(tmp_path),
            areas_km2=np.full(9, 400.0),
            mw_mean=7.3,
        )
        defaults.update(kwargs)
        return AltarConfigBuilder(**defaults)

    def test_build_contains_app_name(self, tmp_path):
        b = self._make_builder(tmp_path)
        pfg = b.build()
        assert "slipmodel:" in pfg

    def test_patch_count_in_config(self, tmp_path):
        b = self._make_builder(tmp_path, n_patches=12)
        pfg = b.build()
        assert "patches = 12" in pfg

    def test_observations_in_config(self, tmp_path):
        b = self._make_builder(tmp_path, n_observations=200)
        pfg = b.build()
        assert "observations = 200" in pfg

    def test_cpu_model_key(self, tmp_path):
        b = self._make_builder(tmp_path, use_gpu=False)
        pfg = b.build()
        assert "altar.models.seismic.static" in pfg
        assert "cuda" not in pfg.split("model =")[1].split("\n")[0]

    def test_gpu_model_key(self, tmp_path):
        b = self._make_builder(tmp_path, use_gpu=True)
        pfg = b.build()
        assert "altar.models.seismic.cuda.static" in pfg

    def test_ramp_params_absent_when_zero(self, tmp_path):
        b = self._make_builder(tmp_path, n_ramp_params=0)
        pfg = b.build()
        # Check the ramp *parameter set block* is absent, not just the word
        # (the word "ramp" may appear in the tmp_path directory name).
        assert "ramp = contiguous" not in pfg
        assert "ramp = altar" not in pfg

    def test_ramp_params_present(self, tmp_path):
        b = self._make_builder(tmp_path, n_ramp_params=3)
        pfg = b.build()
        assert "ramp" in pfg
        assert "count = 3" in pfg

    def test_save_writes_file(self, tmp_path):
        b = self._make_builder(tmp_path)
        path = b.save(str(tmp_path / "test.pfg"))
        assert os.path.isfile(path)
        with open(path) as f:
            content = f.read()
        assert "slipmodel:" in content


# ---------------------------------------------------------------------------
# AltarResultImporter tests (uses mock HDF5)
# ---------------------------------------------------------------------------

def _write_mock_altar_output(results_dir: str, n_chains: int, n_patches: int, beta: float = 1.0):
    """Creates a minimal ``step_final.h5`` and ``BetaStatistics.txt``."""
    os.makedirs(results_dir, exist_ok=True)

    rng = np.random.default_rng(42)
    ss = rng.uniform(0, 2, size=(n_chains, n_patches))
    ds = rng.uniform(0, 3, size=(n_chains, n_patches))

    h5_path = os.path.join(results_dir, "step_final.h5")
    with h5py.File(h5_path, "w") as f:
        ann = f.create_group("Annealer")
        ann.create_dataset("beta", data=np.array(beta))
        bayes = f.create_group("Bayesian")
        bayes.create_dataset("posterior", data=rng.standard_normal(n_chains))
        psets = f.create_group("ParameterSets")
        psets.create_dataset("strikeslip", data=ss)
        psets.create_dataset("dipslip", data=ds)

    stats_path = os.path.join(results_dir, "BetaStatistics.txt")
    with open(stats_path, "w") as f:
        f.write("iteration, beta, scaling, (accepted, invalid, rejected)\n")
        f.write("0, 0.0, 0.3, (0, 0, 0)\n")
        f.write("1, 0.5, 0.25, (800, 0, 200)\n")
        f.write(f"2, {beta}, 0.28, (750, 0, 250)\n")

    return ss, ds


class TestAltarResultImporter:

    def test_loads_samples(self, tmp_path):
        n_chains, n_patches = 200, 5
        ss_true, ds_true = _write_mock_altar_output(str(tmp_path), n_chains, n_patches)
        importer = AltarResultImporter()
        result = importer.load(str(tmp_path), n_patches)

        np.testing.assert_array_equal(result.ss_samples, ss_true)
        np.testing.assert_array_equal(result.ds_samples, ds_true)

    def test_final_beta_stored(self, tmp_path):
        _write_mock_altar_output(str(tmp_path), 100, 4, beta=1.0)
        importer = AltarResultImporter()
        result = importer.load(str(tmp_path), 4)
        assert abs(result.final_beta - 1.0) < 1e-9

    def test_beta_statistics_parsed(self, tmp_path):
        _write_mock_altar_output(str(tmp_path), 100, 4)
        importer = AltarResultImporter()
        result = importer.load(str(tmp_path), 4)
        assert len(result.beta_statistics) == 3
        assert "beta" in result.beta_statistics.columns

    def test_missing_final_h5_raises(self, tmp_path):
        importer = AltarResultImporter()
        with pytest.raises(FileNotFoundError):
            importer.load(str(tmp_path), 5)

    def test_warns_on_non_converged(self, tmp_path):
        _write_mock_altar_output(str(tmp_path), 100, 4, beta=0.8)
        importer = AltarResultImporter()
        with pytest.warns(RuntimeWarning, match="converged"):
            importer.load(str(tmp_path), 4)


# ---------------------------------------------------------------------------
# AltarSlipDistribution tests
# ---------------------------------------------------------------------------

def _make_distribution(n_chains=200, n_patches=5):
    rng = np.random.default_rng(7)
    ss = rng.normal(1.0, 0.2, size=(n_chains, n_patches))
    ds = rng.normal(0.5, 0.1, size=(n_chains, n_patches))
    slip_vec = np.concatenate([ss.mean(0), ds.mean(0)])
    beta_stats = pd.DataFrame(
        {"iteration": [0, 1, 2], "beta": [0.0, 0.5, 1.0],
         "scaling": [0.3, 0.25, 0.28],
         "accepted": [0, 800, 750], "invalid": [0, 0, 0], "rejected": [0, 200, 250]}
    )
    return AltarSlipDistribution(
        slip_vector=slip_vec,
        faults=[],
        ss_samples=ss,
        ds_samples=ds,
        beta_statistics=beta_stats,
        final_beta=1.0,
    )


class TestAltarSlipDistribution:

    def test_get_mean_slip_shape(self):
        dist = _make_distribution(n_patches=5)
        mean = dist.get_mean_slip()
        assert mean.shape == (10,)

    def test_get_mean_slip_values(self):
        dist = _make_distribution(n_patches=5)
        expected = np.concatenate(
            [dist.ss_samples.mean(0), dist.ds_samples.mean(0)]
        )
        np.testing.assert_allclose(dist.get_mean_slip(), expected)

    def test_get_posterior_std_shape(self):
        dist = _make_distribution(n_patches=5)
        assert dist.get_posterior_std().shape == (10,)

    def test_get_credible_intervals_shape(self):
        dist = _make_distribution(n_patches=5)
        ci = dist.get_credible_intervals(0.95)
        assert ci.shape[0] == 10
        assert ci.shape[1] == 2

    def test_slip_magnitude_stats_keys(self):
        dist = _make_distribution(n_patches=5)
        stats = dist.get_slip_magnitude_stats()
        for key in ("mean", "std", "hdi_lower", "hdi_upper"):
            assert key in stats
            assert stats[key].shape == (5,)

    def test_is_converged_true(self):
        dist = _make_distribution()
        assert dist.is_converged()

    def test_is_converged_false(self):
        dist = _make_distribution()
        dist.final_beta = 0.9
        assert not dist.is_converged()

    def test_n_beta_steps(self):
        dist = _make_distribution()
        assert dist.n_beta_steps == 3


# ---------------------------------------------------------------------------
# End-to-end orchestration test (no ALTar required)
# ---------------------------------------------------------------------------

class TestAltarOrchestratorAssembly:
    """Verifies that AltarAssembler integrates correctly with InversionOrchestrator.

    The solver is mocked to avoid requiring ALTar to be installed.
    """

    def test_orchestrator_calls_assembler(self, simple_fault, synthetic_dataset, areas):
        from unittest.mock import MagicMock
        from slipkit.core.bayesian.assembler import AltarAssembler

        mock_solver = MagicMock()
        n_obs = len(synthetic_dataset.data)
        n_patches = simple_fault.num_patches()
        mock_solver.solve.return_value = np.zeros(2 * n_patches)

        assembler = AltarAssembler()
        orc = InversionOrchestrator()
        orc.add_fault(simple_fault)
        orc.add_data(synthetic_dataset)
        orc.set_engine(CutdeCpuEngine())
        orc.set_assembler(assembler)
        orc.set_solver(mock_solver)

        result = orc.run_inversion(lambda_spatial=0.0)

        mock_solver.solve.assert_called_once()
        A_arg, b_arg = mock_solver.solve.call_args[0][:2]
        assert A_arg.shape == (n_obs, 2 * n_patches)
        assert b_arg.shape == (2 * n_obs,)
