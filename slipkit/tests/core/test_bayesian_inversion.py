"""CPU bridge regressions. Real sampling requires SLIPKIT_RUN_ALTAR=1."""
import json
import os
import shutil
import sys
import h5py
import numpy as np
import pytest
from scipy.stats import truncnorm
from slipkit.core.bayesian import (AltarAssembler, AltarBayesianSolver, AltarConfigBuilder,
    AltarDataExporter, AltarProblem, AltarPosterior, AltarResultImporter, AltarSlipDistribution)
from slipkit.core.data import GeodeticDataSet
from slipkit.core.fault import TriangularFaultMesh, SlipComponent
from slipkit.core.inversion import InversionOrchestrator
from slipkit.core.physics import CutdeCpuEngine
from slipkit.core.bayesian.problem import parameter_layout
from slipkit.core.regularization import LaplacianSmoothing, DeepEdgeDamping


def fault(n=1, components=(SlipComponent.STRIKE_SLIP, SlipComponent.DIP_SLIP), scale=1.):
    vertices = np.array([[0., 0., -5.], [2., 0., -5.], [0., 2., -7.]])*scale
    return TriangularFaultMesh((vertices, np.tile([[0, 1, 2]], (n, 1))), slip_components=components)


def dataset(name='data', ramp=None):
    coords = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
    return GeodeticDataSet(coords, np.arange(4.)/10, np.tile([0., 0., 1.], (4, 1)), np.full(4, .1), name, ramp=ramp)


class SentinelEngine:
    def build_kernel(self, f, d):
        return np.tile(np.arange(f.num_patches()*f.num_components())+f.num_patches(), (len(d), 1)) + d.coords[:, :1]


def write_output(directory, samples, beta=1., named=None):
    directory.mkdir(parents=True, exist_ok=True)
    with h5py.File(directory/'step_final.h5', 'w') as out:
        out['Annealer/beta'] = beta
        if named is None:
            out['ParameterSets/theta'] = samples
        else:
            for name, values in named.items():
                out[f'ParameterSets/{name}'] = values
        for name in ('prior', 'likelihood', 'posterior'):
            out[f'Bayesian/{name}'] = np.zeros(len(samples))
    (directory/'BetaStatistics.txt').write_text('iteration, beta, scaling, accepted, invalid, rejected\n(0, 1.0, 0.1, 10, 2, 3)\n')


@pytest.mark.parametrize('components,n', [((SlipComponent.STRIKE_SLIP,), 1), ((SlipComponent.DIP_SLIP,), 2),
                                          ((SlipComponent.DIP_SLIP,), 3)])
def test_multiple_fault_layout_and_ramps(components, n):
    faults = [fault(2), fault(n, components)]
    datasets = [dataset(), dataset('ramp', 1)]
    assembler = AltarAssembler()
    problem = assembler.assemble_problem(faults, datasets, SentinelEngine(), LaplacianSmoothing(), 0.)
    p = 4+n+3
    assert problem.G.shape == (8, p)
    assert [(e['start'], e['stop']) for e in problem.layout] == [(0, 2), (2, 4), (4, 4+n), (4+n, p)]
    np.testing.assert_array_equal(problem.G[:4, -3:], 0.)
    samples = np.tile(np.arange(p), (5, 1)).astype(float)
    result = AltarSlipDistribution(AltarPosterior(samples, 1., layout=problem.layout), faults, datasets)
    np.testing.assert_equal(result.get_fault_vector(1), np.arange(4, 4+n))
    np.testing.assert_equal(result.get_component_samples(components[0], 1), samples[:, 4:4+n])
    np.testing.assert_allclose(result.nuisance_prediction(datasets[1]), problem.G[4:, -3:] @ samples[0, -3:])
    reconstructed = np.concatenate([result.get_fault_vector(i) for i in range(2)]+[result.nuisance['ramp'].coeffs])
    np.testing.assert_allclose(problem.G @ reconstructed, problem.G @ result.posterior.mean)
    absent = 'ds' if components[0] == SlipComponent.STRIKE_SLIP else 'ss'
    with pytest.raises(ValueError):
        result.get_component_samples(absent, 1)


def test_inputs_are_refreshed():
    f, d, assembler = fault(), dataset(), AltarAssembler()
    first = assembler.assemble_problem([f], [d], SentinelEngine(), None, 0.)
    d.data = d.data+1
    d.sigma = d.sigma*2
    d.coords[:, 0] += 3
    second = assembler.assemble_problem([f], [d], SentinelEngine(), None, 0.)
    np.testing.assert_allclose(second.data, first.data+1)
    np.testing.assert_allclose(second.covariance, first.covariance*4)
    assert not np.array_equal(first.G, second.G)


def test_noise_model_covariance():
    class Noise:
        def sigma(self): return np.full(4, .2)
        def covariance(self): return np.eye(4)*.03 + np.ones((4, 4))*.01
    model = Noise()
    problem = AltarAssembler(noise_models=[model]).assemble_problem([fault()], [dataset()], SentinelEngine(), None, 0.)
    np.testing.assert_equal(problem.covariance, model.covariance())


@pytest.mark.parametrize('covariance', [np.ones((2, 2)), np.array([[1., .2], [0., 1.]]), np.eye(3), np.eye(2)*np.nan])
def test_invalid_covariance(covariance):
    with pytest.raises(ValueError):
        AltarProblem(np.eye(2), np.zeros(2), covariance)


@pytest.mark.parametrize('sigma', [0., -1., np.nan])
def test_bad_sigma_before_launch(sigma, tmp_path):
    solver = AltarBayesianSolver(work_dir=tmp_path)
    with pytest.raises(ValueError):
        solver.solve(np.eye(2), np.array([0., 0., sigma, .1]))
    assert solver.last_posterior is None
    assert not list(tmp_path.iterdir())


def test_text_export_full_covariance_and_fixed_cp(tmp_path):
    g, d = np.eye(2), np.array([.2, .7])
    c = np.array([[.1, .02], [.02, .2]])
    files = AltarDataExporter(tmp_path).export_all(g, d, covariance=c, alpha_cp=.1)
    np.testing.assert_allclose(np.loadtxt(files['green']), g)
    np.testing.assert_allclose(np.loadtxt(files['data']), d)
    np.testing.assert_allclose(np.loadtxt(files['cd']), c+np.diag((.1*d)**2))
    assert set(files) == {'green', 'data', 'cd'}


@pytest.mark.parametrize('prior', ['gaussian', 'uniform'])
def test_config_uses_real_dimensions_and_settings(prior, tmp_path):
    config = AltarConfigBuilder(3, 4, tmp_path, output_dir='posterior', output_freq=2, seed=31, prior=prior).build()
    for text in ['parameters = 3', 'observations = 4', 'rng.seed = 31', 'output_dir = posterior', 'output_freq = 2']:
        assert text in config
    assert f'prior = altar.distributions.{prior}' in config
    with pytest.raises(ValueError):
        AltarConfigBuilder(2, 3, tmp_path, tasks=2)


def test_gaussian_transform_smoothing_and_correlated_likelihood():
    g = np.array([[1., .2], [.1, 1.], [1., 1.]])
    c = np.eye(3)*.02 + np.ones((3, 3))*.01
    s = np.array([[1., -1.]])
    problem = AltarProblem(g, np.array([.2, .7, .9]), c, smoothing=s)
    solver = AltarBayesianSolver(prior_mean=[.2, -.1], prior_scales=[.5, .8])
    offset, matrix = solver._transform(problem, None)
    q = np.diag(np.array([.5, .8])**-2) + s.T@s
    np.testing.assert_allclose(matrix.T @ q @ matrix, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(q @ offset, np.array([.2/.5**2, -.1/.8**2]))
    z = np.array([.4, -.2])
    x = offset+matrix@z
    physical_residual = problem.data-g@x
    transformed_residual = (problem.data-g@offset)-(g@matrix)@z
    np.testing.assert_allclose(physical_residual, transformed_residual)
    np.testing.assert_allclose(physical_residual @ np.linalg.solve(c, physical_residual),
                               transformed_residual @ np.linalg.solve(c, transformed_residual))
    mean, cov = problem.gaussian_reference([.2, -.1], [.5, .8])
    np.testing.assert_allclose((g.T @ np.linalg.solve(c, g)+q)@cov, np.eye(2), atol=1e-14)
    np.testing.assert_allclose((g.T @ np.linalg.solve(c, g)+q)@mean,
                               g.T@np.linalg.solve(c, problem.data)+q@offset)


def test_smoothing_and_deep_edge_assembly():
    f, d = fault(), dataset(ramp=0)
    reg = DeepEdgeDamping(alpha=2., tol=3.)
    problem = AltarAssembler().assemble_problem([f], [d], SentinelEngine(), reg, .4)
    expected = reg.build_smoothing_matrix([f], .4).toarray()
    np.testing.assert_allclose(problem.smoothing[:, :2], expected)
    np.testing.assert_equal(problem.smoothing[:, 2], 0.)


@pytest.mark.parametrize('prior,bounds,smoothing', [('gaussian', ([0, 0], [1, 1]), None),
    ('uniform', None, None), ('uniform', ([0, 0], [1, np.inf]), None),
    ('uniform', ([0, 0], [0, 1]), None), ('uniform', ([0, 0], [1, 1]), np.eye(2))])
def test_unsupported_prior_combinations(prior, bounds, smoothing):
    solver = AltarBayesianSolver(prior=prior)
    with pytest.raises(ValueError):
        solver._transform(AltarProblem(np.eye(2), np.zeros(2), np.eye(2), smoothing=smoothing), bounds)


def test_named_import_follows_manifest_and_physical_transform(tmp_path):
    samples = np.arange(15.).reshape(5, 3)
    write_output(tmp_path, samples, named={'aaa': samples[:, 2:], 'zzz': samples[:, :2]})
    manifest = dict(n_parameters=3, parameter_sets=[dict(name='zzz', width=2), dict(name='aaa', width=1)],
                    transform=dict(offset=[1, 2, 3], matrix=np.diag([2, 3, 4]).tolist()))
    result = AltarResultImporter().load(tmp_path, manifest=manifest)
    np.testing.assert_equal(result.samples, samples*[2, 3, 4]+[1, 2, 3])
    np.testing.assert_allclose(result.covariance, np.cov(result.samples, rowvar=False))
    with pytest.raises(ValueError):
        AltarResultImporter().load(tmp_path, 3)


@pytest.mark.parametrize('samples,beta', [(np.zeros((0, 2)), 1.), (np.zeros((3, 3)), 1.),
    (np.full((3, 2), np.nan), 1.), (np.zeros((3, 2)), .9), (np.zeros((3, 2)), np.nan)])
def test_invalid_import_rejected(samples, beta, tmp_path):
    write_output(tmp_path, samples, beta)
    with pytest.raises(ValueError):
        AltarResultImporter().load(tmp_path, 2)


def test_incomplete_diagnostic_opt_in(tmp_path):
    write_output(tmp_path, np.zeros((5, 2)), .5)
    posterior = AltarResultImporter().load(tmp_path, 2, allow_incomplete=True)
    assert not posterior.annealing_complete
    assert 'incomplete' in posterior.diagnostics['sampling_adequacy']


def test_bad_probability_array(tmp_path):
    write_output(tmp_path, np.zeros((5, 2)))
    with h5py.File(tmp_path/'step_final.h5', 'a') as archive:
        archive['Bayesian/prior'][0] = np.nan
    with pytest.raises(ValueError, match='prior'):
        AltarResultImporter().load(tmp_path, 2)


def test_result_statistics_units_and_intervals():
    samples = np.array([[1., 2.], [2., 3.], [3., 4.]])
    results = [AltarSlipDistribution(AltarPosterior(samples, 1., layout=parameter_layout([fault(scale=scale)])), [fault(scale=scale)]) for scale in [1, 1000]]
    np.testing.assert_equal(results[0].get_mean_slip(), [2., 3.])
    np.testing.assert_allclose(results[0].seismic_moment_samples(length_unit='km'),
                               results[1].seismic_moment_samples(length_unit='m'))
    np.testing.assert_allclose(results[0].get_slip_magnitude_stats()['mean'], np.sqrt((samples**2).sum(axis=1)).mean())
    assert results[0].get_credible_intervals().shape == (2, 2)
    with pytest.raises(ValueError):
        results[0].get_credible_intervals(1.)


def test_orchestrator_maps_geometry_and_resets_stale_result(tmp_path, monkeypatch):
    solver = AltarBayesianSolver(work_dir=tmp_path)
    orc = InversionOrchestrator()
    orc.add_fault(fault()); orc.add_data(dataset(ramp=0)); orc.set_engine(SentinelEngine())
    orc.set_assembler(AltarAssembler()); orc.set_solver(solver)
    def solve(problem, bounds=None):
        solver.last_posterior = AltarPosterior(np.ones((5, 3)), 1., layout=problem.layout)
        return solver.last_posterior.mean
    monkeypatch.setattr(solver, 'solve_problem', solve)
    result = orc.run_inversion(.2)
    assert isinstance(result, AltarSlipDistribution)
    assert result.get_fault_vector().shape == (2,)
    assert result.nuisance_samples['data'].shape == (5, 1)
    orc.datasets[0].sigma[:] = 0
    with pytest.raises(ValueError):
        orc.run_inversion(0.)
    assert solver.last_result is None and solver.last_posterior is None


def test_run_isolation_failure_and_cleanup(tmp_path, monkeypatch):
    solver = AltarBayesianSolver(work_dir=tmp_path, chains=5)
    monkeypatch.setattr(solver, '_preflight', lambda: ('linear', {}))
    def successful(binary, config, run):
        from pathlib import Path
        write_output(Path(run)/'results', np.ones((5, 2)))
    monkeypatch.setattr(solver, '_run_altar', successful)
    solver.solve(np.eye(2), np.array([0., 0., .1, .1]))
    first = solver.last_run_path
    def failed(*args): raise RuntimeError('deliberate launcher failure')
    monkeypatch.setattr(solver, '_run_altar', failed)
    with pytest.raises(RuntimeError):
        solver.solve(np.eye(2), np.array([0., 0., .1, .1]))
    assert solver.last_run_path != first
    assert solver.last_posterior is None and solver.last_result is None
    assert json.loads(open(solver.last_run_path+'/manifest.json').read())['status'] == 'failed'
    monkeypatch.setattr(solver, '_run_altar', successful)
    solver.keep_work_dir = False
    solver.solve(np.eye(2), np.array([0., 0., .1, .1]))
    assert solver.last_posterior.run_path is None and solver.last_posterior.step_files == []


skip_native_altar = pytest.mark.skipif(os.environ.get('SLIPKIT_RUN_ALTAR') != '1' or
    shutil.which('linear', path=os.path.dirname(sys.executable)) is None,
    reason='Set SLIPKIT_RUN_ALTAR=1 in the native AlTar environment.')


def real_altar(function):
    return pytest.mark.integration(skip_native_altar(function))


@real_altar
@pytest.mark.parametrize('seed', [17, 31, 47])
@pytest.mark.parametrize('prior', ['gaussian', 'uniform'])
def test_real_sampler_reference(seed, prior, tmp_path):
    g = np.eye(2)
    d = np.array([-.05, .75])
    sigma = np.array([.15, .2])
    bounds = (np.array([0., -.2]), np.array([1., 1.8])) if prior == 'uniform' else None
    solver = AltarBayesianSolver(work_dir=tmp_path, seed=seed, prior=prior, chains=1024, steps=200, output_freq=2, output_dir='posterior')
    problem = AltarProblem(g, d, np.diag(sigma**2))
    solver.solve_problem(problem, bounds)
    record = solver.get_last_posterior()
    if prior == 'gaussian':
        mean, cov = problem.gaussian_reference()
        std = np.sqrt(np.diag(cov))
    else:
        lo, hi = bounds
        ref = truncnorm((lo-d)/sigma, (hi-d)/sigma, loc=d, scale=sigma)
        mean, std = ref.mean(), ref.std()
        assert np.all(record.samples >= lo) and np.all(record.samples <= hi)
    # Fixed full-run tolerances, not an independence claim for terminal particles.
    assert np.max(np.abs(record.mean-mean)/std) < .2
    assert np.max(np.abs(record.samples.std(axis=0)/std-1)) < .2
    assert record.annealing_complete
    reloaded = AltarResultImporter().load(os.path.join(record.run_path, 'posterior'))
    np.testing.assert_equal(reloaded.samples, record.samples)
    archives = [os.path.basename(p) for p in record.step_files if 'final' not in p]
    assert all(int(name[5:8]) % 2 == 0 for name in archives)


@real_altar
def test_real_geometry_forward_round_trip(tmp_path):
    f, d, engine = fault(), dataset(), CutdeCpuEngine()
    g = engine.build_kernel(f, d)
    d.data = g @ np.array([.3, .1])
    orc = InversionOrchestrator()
    orc.add_fault(f); orc.add_data(d); orc.set_engine(engine)
    orc.set_assembler(AltarAssembler()); orc.set_solver(AltarBayesianSolver(work_dir=tmp_path, chains=256, steps=150))
    result = orc.run_inversion(0.)
    assert result.annealing_complete
    np.testing.assert_allclose(engine.predict(f, d, result.get_fault_vector()), g @ result.get_mean_slip())


def test_timeout_preserves_log_and_kills_children(tmp_path):
    import time
    script = tmp_path/'hanging.py'
    marker = tmp_path/'child-survived'
    child_code = f"import time, signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); from pathlib import Path; time.sleep(2); Path({str(marker)!r}).write_text('alive'); time.sleep(60)"
    script.write_text(f'import subprocess, sys, time\nsubprocess.Popen([sys.executable, "-c", {child_code!r}])\nprint("child launched", flush=True)\ntime.sleep(60)\n')
    solver = AltarBayesianSolver(timeout=.5)
    with pytest.raises(RuntimeError, match='timed out'):
        solver._run_altar(str(script), 'unused', str(tmp_path))
    assert 'child launched' in (tmp_path/'sampler.log').read_text()
    time.sleep(2)
    assert not marker.exists()


@real_altar
def test_real_gaussian_smoothing_ramp_and_full_covariance(tmp_path):
    d = dataset(ramp=0)
    c = np.eye(4)*.01 + np.ones((4, 4))*.003
    problem = AltarAssembler(covariance=c).assemble_problem([fault()], [d], SentinelEngine(), DeepEdgeDamping(), .3)
    scales, mean = [.5, .8, .2], [.1, -.2, .05]
    exact_mean, exact_cov = problem.gaussian_reference(mean, scales)
    solver = AltarBayesianSolver(work_dir=tmp_path, prior_mean=mean, prior_scales=scales, chains=1024, steps=200)
    solver.solve_problem(problem)
    posterior = solver.get_last_posterior()
    assert np.max(np.abs(posterior.mean-exact_mean)/np.sqrt(np.diag(exact_cov))) < .2
    assert np.max(np.abs(np.sqrt(np.diag(posterior.covariance)/np.diag(exact_cov))-1)) < .2


def test_geometry_scale_leaves_physical_predictions_unchanged():
    d_km = dataset()
    d_m = dataset()
    d_m.coords = d_m.coords*1000
    engine = CutdeCpuEngine()
    np.testing.assert_allclose(engine.build_kernel(fault(), d_km),
                               engine.build_kernel(fault(scale=1000), d_m), atol=1e-14)


@pytest.mark.parametrize('case', ['tiny', 'patch-9'])
def test_fixture_inputs_unchanged(case):
    import hashlib
    from pathlib import Path
    directory = Path(__file__).parents[1]/'fixtures/altar'/case
    hashes = json.loads((directory/'sha256.json').read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == expected


def test_credible_interval_hdi_is_explicit():
    from slipkit.core.bayesian.results import intervals
    samples = np.random.default_rng(0).normal(size=(100, 3))
    try:
        import arviz
    except ImportError:
        with pytest.raises(ImportError):
            intervals(samples, .95, 'hdi')
    else:
        ci = intervals(samples, .95, 'hdi')
        assert ci.shape == (3, 2)
    np.testing.assert_allclose(intervals(samples, .95, 'equal_tailed'), np.quantile(samples, [.025, .975], axis=0).T)


def test_empty_geometry_resets_previous_result():
    solver = AltarBayesianSolver()
    solver.last_result = object()
    solver.last_posterior = object()
    orchestrator = InversionOrchestrator()
    orchestrator.set_solver(solver)
    with pytest.raises(ValueError, match='No fault'):
        orchestrator.run_inversion(0.)
    assert solver.last_result is None and solver.last_posterior is None


def test_incomplete_record_cannot_be_public_result():
    with pytest.raises(ValueError, match='Incomplete'):
        AltarSlipDistribution(AltarPosterior(np.zeros((5, 2)), .5), [fault()])


def test_failed_run_intermediate_output_is_diagnostic_only(tmp_path):
    write_output(tmp_path, np.ones((5, 2)), 1.)
    (tmp_path/'step_final.h5').rename(tmp_path/'step_010.h5')
    manifest = dict(n_parameters=2, status='failed')
    with pytest.raises(ValueError, match='failed run'):
        AltarResultImporter().load(tmp_path, manifest=manifest)
    diagnostic = AltarResultImporter().load(tmp_path, manifest=manifest, allow_incomplete=True)
    assert diagnostic.diagnostics['diagnostic_only']
    with pytest.raises(ValueError, match='Incomplete'):
        AltarSlipDistribution(diagnostic, [fault()])


def test_uniform_rejects_explicit_gaussian_controls():
    with pytest.raises(ValueError, match='Gaussian prior controls'):
        AltarBayesianSolver(prior='uniform', prior_scales=.5)


def test_nested_recorder_directory_reimports_physical_samples(tmp_path):
    results = tmp_path/'posterior/nested'
    write_output(results, np.ones((5, 2)))
    manifest = dict(n_parameters=2, output_dir='posterior/nested',
                    transform=dict(offset=[.1, .2], matrix=np.eye(2).tolist()))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    posterior = AltarResultImporter().load(results)
    assert posterior.run_path == str(tmp_path)
    np.testing.assert_allclose(posterior.mean, [1.1, 1.2])


def test_small_variance_correlation_cannot_use_packed_adapter():
    c = np.eye(4)*1e-8
    c[0, 1] = c[1, 0] = 5e-9
    with pytest.raises(ValueError, match='Full covariance'):
        AltarAssembler(covariance=c).assemble([fault()], [dataset()], SentinelEngine(), None, 0.)


def test_semantic_layout_and_geometry_must_match():
    f = fault()
    layout = parameter_layout([f])
    layout[0]['component'], layout[1]['component'] = layout[1]['component'], layout[0]['component']
    with pytest.raises(ValueError, match='semantic'):
        AltarSlipDistribution(AltarPosterior(np.ones((5, 2)), 1., layout=layout), [f])
    layout = parameter_layout([f])
    f.vertices[0, 0] += .1
    with pytest.raises(ValueError, match='geometry'):
        AltarSlipDistribution(AltarPosterior(np.ones((5, 2)), 1., layout=layout), [f])


def test_ramp_identity_and_normalization_must_match():
    f, ds = fault(), dataset(ramp=1)
    posterior = AltarPosterior(np.ones((5, 5)), 1., layout=parameter_layout([f], [ds]))
    ds.ramp.scale *= 2
    with pytest.raises(ValueError, match='semantic'):
        AltarSlipDistribution(posterior, [f], [ds])


def test_frozen_inputs_and_diagonal_block_whitening():
    from scipy.linalg import block_diag
    g = np.arange(12.).reshape(4, 3)
    d = np.arange(4.)
    blocks = (np.array([.01, .02]), np.array([[1., -.4], [-.4, 2.]]))
    p = AltarProblem(g, d, blocks)
    g[:] = 100; d[:] = 100
    with pytest.raises(ValueError): p.G[0, 0] = 0
    c = block_diag(np.diag(blocks[0]), blocks[1])
    gw, dw, logdet = p.whiten(.1)
    effective = c+np.diag((.1*p.data)**2)
    np.testing.assert_allclose(gw.T @ gw, p.G.T @ np.linalg.solve(effective, p.G))
    np.testing.assert_allclose(dw @ dw, p.data @ np.linalg.solve(effective, p.data))
    np.testing.assert_allclose(logdet, np.linalg.slogdet(effective)[1])
    assert p.covariance_bytes == 48


def test_exact_gaussian_solver_shared_results():
    from slipkit.core.bayesian import GaussianBayesianSolver
    f, ds = fault(), dataset(ramp=0)
    c = np.eye(4)*.01+np.ones((4, 4))*.005
    problem = AltarAssembler(covariance=c).assemble_problem([f], [ds], SentinelEngine(), DeepEdgeDamping(), .3)
    scales, anchor = np.array([.5, .8, .2]), np.array([.1, -.2, .05])
    effective = c+np.diag((.1*problem.data)**2)
    q = np.diag(scales**-2) + problem.smoothing.T @ problem.smoothing + problem.G.T @ np.linalg.solve(effective, problem.G)
    reference = np.linalg.solve(q, scales**-2*anchor+problem.G.T @ np.linalg.solve(effective, problem.data))
    solver = GaussianBayesianSolver(prior_scales=scales, prior_mean=anchor, alpha_cp=.1, draws=20000)
    np.testing.assert_allclose(solver.solve_problem(problem), reference, atol=1e-13)
    posterior = solver.get_last_posterior()
    np.testing.assert_allclose(posterior.covariance, np.linalg.solve(q, np.eye(3)), atol=1e-13)
    assert np.max(np.abs(posterior.samples.mean(axis=0)-reference)/np.sqrt(np.diag(posterior.covariance))) < .04
    result = AltarSlipDistribution(posterior, [f], [ds])
    np.testing.assert_allclose(result.nuisance_prediction(ds), problem.G[:, -1]*reference[-1])
    with pytest.raises(ValueError, match='finite bounds'):
        solver.solve_problem(problem, ([0, 0, 0], [1, 1, 1]))
    assert solver.last_posterior is None
    orc = InversionOrchestrator()
    orc.add_fault(f); orc.add_data(ds); orc.set_engine(SentinelEngine())
    orc.set_assembler(AltarAssembler(covariance=c)); orc.set_solver(solver)
    assert isinstance(orc.run_inversion(.3), AltarSlipDistribution)


def test_binary_prior_transform_checksum_and_probability_normalization(tmp_path):
    import hashlib
    results = tmp_path/'results'
    write_output(results, np.ones((5, 2)))
    path = tmp_path/'prior_transform.npz'
    np.savez(path, offset=[1., 2.], matrix=np.array([[2., 1.], [0., 3.]]))
    manifest = dict(n_parameters=2, output_dir='results', likelihood_offset=2., prior_offset=3.,
                    transform=dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    posterior = AltarResultImporter().load(results, manifest=manifest)
    np.testing.assert_allclose(posterior.mean, [4., 5.])
    np.testing.assert_allclose(posterior.probabilities['likelihood'], 2.)
    np.testing.assert_allclose(posterior.probabilities['prior'], 3.)
    np.testing.assert_allclose(posterior.probabilities['posterior'], 5.)
    manifest['transform']['sha256'] = 'bad'
    with pytest.raises(ValueError, match='checksum'):
        AltarResultImporter().load(results, manifest=manifest)


def test_stage_progress_survives_interruption(tmp_path):
    import time
    from slipkit.core.bayesian.progress import save_progress
    (tmp_path/'sampler.log').write_text('altar: time: 2026-09-17T01:00:00\naltar: iteration: 0, beta: 0, scaling: 0.1\n'
        'altar: resampling: unique samples 90 out of 100\naltar: time: 2026-09-17T01:00:02\n'
        'altar: iteration: 1, beta: 0.01, scaling: 0.4\naltar: stats(accepted/invalid/rejected): (10, 5, 5)\n'
        'slipkit: peak_memory_bytes = 1000\n')
    report = save_progress(tmp_path, time.monotonic(), .1)
    stage = report['stages'][1]
    assert stage['actual_scaling'] == .1 and stage['next_scaling'] == .4
    assert stage['seconds'] == 2 and stage['acceptance'] == .5
    assert report['peak_memory_bytes'] == 1000
    assert report['stages'][0]['resampling_for_next_stage']['unique'] == 90


@real_altar
def test_native_whitened_likelihood_deterministic():
    from slipkit.core.bayesian.solver import check_native_likelihood
    check_native_likelihood()


@real_altar
@pytest.mark.parametrize('prior', ['gaussian', 'uniform'])
def test_vectorized_model_parity(prior):
    import altar
    from altar.models.linear.Linear import Linear
    from altar.bayesian.CoolingStep import CoolingStep
    from altar.distributions.Gaussian import Gaussian
    from altar.distributions.Uniform import Uniform
    from slipkit.core.bayesian.native_cpu import VectorizedLinear
    from types import SimpleNamespace
    rng = SimpleNamespace(rng=altar.rng())
    distribution = Gaussian() if prior == 'gaussian' else Uniform()
    distribution.parameters = 3
    if prior == 'gaussian':
        distribution.mean, distribution.sigma = 0., 1.
    else:
        distribution.support = (0, 1)
    distribution.initialize(rng=rng)
    g = np.array([[1., 2., .3], [0., -.1, 2.]])
    data = np.array([.2, .7])
    step = CoolingStep.alloc(samples=5, parameters=3)
    theta = np.random.default_rng(0).uniform(size=(5, 3))
    step.theta.ndarray()[:] = theta
    fast = VectorizedLinear(name='linear.model')
    fast.parameters, fast.observations = 3, 2
    fast.prior = distribution
    fast._g, fast._d, fast.normalization = g, data, -np.log(2*np.pi)
    step.prior.ndarray()[:] = 0
    fast.priorLikelihood(step)
    fast_prior = step.prior.ndarray().copy()
    step.prior.ndarray()[:] = 0
    distribution.priorLikelihood(theta=step.theta, likelihood=step.prior)
    np.testing.assert_allclose(fast_prior, step.prior.ndarray(), atol=1e-13)
    fast.dataLikelihood(step)
    expected = -.5*np.sum((theta @ g.T-data)**2, axis=1)-np.log(2*np.pi)
    np.testing.assert_allclose(step.data.ndarray(), expected, atol=1e-13)


@real_altar
@pytest.mark.parametrize('kernel', ['native', 'vectorized'])
def test_real_physical_likelihood_and_prior_normalization(kernel, tmp_path):
    g = np.array([[1., .2], [.3, 1.]])
    d = np.array([.2, .7])
    c = np.array([[1., .8], [.8, 2.]])
    anchor, scales = np.array([.2, -.1]), np.array([.5, .8])
    solver = AltarBayesianSolver(work_dir=tmp_path, chains=128, steps=80, cpu_kernel=kernel,
                                prior_mean=anchor, prior_scales=scales, alpha_cp=.1,
                                initial_scaling=.3, acceptance_weight=0., rejection_weight=.3)
    solver.solve_problem(AltarProblem(g, d, c))
    posterior = solver.get_last_posterior()
    residual = d-posterior.samples @ g.T
    effective = c+np.diag((.1*d)**2)
    expected_likelihood = -.5*(np.einsum('ij,ji->i', residual, np.linalg.solve(effective, residual.T))
                               + np.linalg.slogdet(effective)[1] + 2*np.log(2*np.pi))
    expected_prior = -.5*np.sum(((posterior.samples-anchor)/scales)**2, axis=1)-np.log(scales).sum()-np.log(2*np.pi)
    np.testing.assert_allclose(posterior.probabilities['likelihood'], expected_likelihood, atol=1e-10)
    np.testing.assert_allclose(posterior.probabilities['prior'], expected_prior, atol=1e-10)
    np.testing.assert_allclose(posterior.probabilities['posterior'], expected_likelihood+expected_prior, atol=1e-10)
    progress = json.loads(open(posterior.run_path+'/progress.json').read())
    assert len(progress['stages']) >= 2
    assert all(stage['next_scaling'] == .3 for stage in progress['stages'])
    from slipkit.core.bayesian import load_inference
    solver.save_result(tmp_path/'durable-native')
    restored = load_inference(tmp_path/'durable-native')
    np.testing.assert_array_equal(restored.posterior.samples, posterior.samples)
    np.testing.assert_allclose(restored.posterior.probabilities['likelihood'], expected_likelihood)
    assert restored.metadata['inference_settings']['cpu_kernel'] == kernel


def test_benchmark_preserves_reports_and_records_validation_failures(tmp_path, monkeypatch):
    from pathlib import Path
    from slipkit.core.bayesian import benchmark
    class BadImportSolver:
        def __init__(self, **kw): self.last_run_path = None
        def solve_problem(self, problem): raise ValueError('deliberate sample validation failure')
    monkeypatch.setattr(benchmark, 'AltarBayesianSolver', BadImportSolver)
    case = Path(__file__).parents[1]/'fixtures/altar/tiny'
    monkeypatch.setattr(sys, 'argv', ['benchmark', str(case), '--work-dir', str(tmp_path), '--seeds', '17'])
    previous = tmp_path/'benchmark.json'
    previous.write_text('old report')
    for _ in range(2):
        with pytest.raises(SystemExit, match='gate failed'):
            benchmark.main()
    assert previous.read_text() == 'old report'
    reports = list(tmp_path.glob('experiment-*/benchmark.json'))
    assert len(reports) == 2
    assert all('ValueError' in json.loads(p.read_text())[0]['error'] for p in reports)


def test_explicit_proposal_controls_and_scale_limits(tmp_path):
    config = AltarConfigBuilder(288, 351, tmp_path, initial_scaling=.16, acceptance_weight=0., rejection_weight=.16).build()
    assert 'scaling = 0.16' in config and 'acceptanceWeight = 0.0' in config and 'rejectionWeight = 0.16' in config
    with pytest.raises(ValueError, match='weights'):
        AltarConfigBuilder(2, 3, tmp_path, acceptance_weight=1., rejection_weight=.5)


@real_altar
def test_vectorized_uniform_sampler_round_trip(tmp_path):
    bounds = (np.array([0., -.2]), np.array([1., 1.8]))
    solver = AltarBayesianSolver(work_dir=tmp_path, prior='uniform', cpu_kernel='vectorized', chains=512, steps=100)
    solver.solve_problem(AltarProblem(np.eye(2), [-.05, .75], [.15**2, .2**2]), bounds)
    posterior = solver.get_last_posterior()
    ref = truncnorm((bounds[0]-[-.05, .75])/[.15, .2], (bounds[1]-[-.05, .75])/[.15, .2], loc=[-.05, .75], scale=[.15, .2])
    assert np.max(np.abs(posterior.mean-ref.mean())/ref.std()) < .2
    np.testing.assert_allclose(posterior.probabilities['prior'], -np.log(bounds[1]-bounds[0]).sum())
    assert np.all(posterior.samples >= bounds[0]) and np.all(posterior.samples <= bounds[1])


def test_benchmark_records_input_validation_failure(tmp_path, monkeypatch):
    from slipkit.core.bayesian import benchmark
    monkeypatch.setattr(sys, 'argv', ['benchmark', str(tmp_path/'missing-case'), '--work-dir', str(tmp_path)])
    with pytest.raises(SystemExit, match='Input/reference validation failed'):
        benchmark.main()
    report, = tmp_path.glob('experiment-*/benchmark.json')
    assert json.loads(report.read_text())[0]['status'] == 'failed'
    assert 'FileNotFoundError' in json.loads(report.read_text())[0]['error']


def test_sampler_keyboard_interrupt_cleans_up_and_propagates(tmp_path, monkeypatch):
    import subprocess
    import signal
    from slipkit.core.bayesian import solver as module
    class Process:
        pid = 12345
        def wait(self, timeout=None):
            if timeout is not None and timeout <= 1:
                raise KeyboardInterrupt
            return 0
    signals = []
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **kw: Process())
    monkeypatch.setattr(module.os, 'killpg', lambda pid, sig: signals.append((pid, sig)))
    solver = AltarBayesianSolver(work_dir=tmp_path)
    with pytest.raises(KeyboardInterrupt):
        solver._run_altar('unused', 'unused', str(tmp_path))
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]
    assert (tmp_path/'progress.json').exists()


@pytest.mark.parametrize('covariance', [np.array([.01, .04]), np.diag([.01, .04]), np.array([[.01, .005], [.005, .04]])])
def test_exporter_fixed_cp_preserves_noise(covariance, tmp_path):
    exporter = AltarDataExporter(tmp_path)
    expected = np.diag(covariance) if covariance.ndim == 1 else covariance.copy()
    expected[np.diag_indices(2)] += [.01, .04]
    original = covariance.copy()
    path = exporter.export_covariance(covariance=covariance, d_obs=np.array([1., 2.]), alpha_cp=.1)
    np.testing.assert_allclose(np.loadtxt(path), expected)
    paths = exporter.export_all(np.eye(2), np.array([1., 2.]), covariance=covariance, alpha_cp=.1)
    np.testing.assert_allclose(np.loadtxt(paths['cd']), expected)
    np.testing.assert_array_equal(covariance, original)


@pytest.mark.parametrize('exact', [True, False])
def test_durable_inference_geometry_ramps_and_summaries(tmp_path, exact):
    from slipkit.core.bayesian import GaussianBayesianSolver, load_inference
    f, d = fault(), dataset(ramp=1)
    p = AltarAssembler(covariance=np.diag(d.sigma**2)+.002*np.ones((4, 4))).assemble_problem([f], [d], CutdeCpuEngine(), None, 0.)
    p = AltarProblem(p.G, p.data, p.covariance, p.layout, np.array([[1., -1., 0., 0., 0.]]))
    solver = GaussianBayesianSolver(prior_mean=.2, prior_scales=.8, alpha_cp=.1, draws=2048)
    solver.solve_problem(p)
    if not exact:
        solver.last_posterior.exact_mean = solver.last_posterior.precision_factor = None
        solver.last_posterior.diagnostics = {'inference': 'native fixture'}
    result = AltarSlipDistribution(solver.last_posterior, [f], [d])
    destination = tmp_path/'inference'
    solver.save_result(destination, p, faults=[f], datasets=[d])
    restored = load_inference(destination)
    np.testing.assert_allclose(restored.problem.G @ restored.posterior.mean, p.G @ solver.last_posterior.mean)
    np.testing.assert_allclose(restored.posterior.covariance, solver.last_posterior.covariance)
    np.testing.assert_array_equal(restored.result.get_component_samples('ss'), result.get_component_samples('ss'))
    np.testing.assert_array_equal(restored.result.seismic_moment_samples(), result.seismic_moment_samples())
    np.testing.assert_array_equal(restored.result.nuisance_prediction(restored.datasets[0]), result.nuisance_prediction(d))
    assert restored.metadata['prior']['scales'] == [.8]*p.G.shape[1]
    assert restored.metadata['alpha_cp'] == .1
    assert not (destination/'sampler.log').exists()
    with pytest.raises(FileExistsError):
        solver.save_result(destination, p, faults=[f], datasets=[d])
    with open(destination/'arrays.npz', 'ab') as stream:
        stream.write(b'damaged')
    with pytest.raises(ValueError, match='checksum'):
        load_inference(destination)


def test_tempered_gaussian_diagnostics_detect_population_bias(tmp_path):
    from slipkit.core.bayesian.diagnostics import gaussian_stage_errors, save_stage_diagnostics
    from scipy.linalg import solve_triangular
    p = AltarProblem(np.array([[1., 2.], [.3, 1.]]), [.2, .6], np.array([[.1, .03], [.03, .2]]), smoothing=np.array([[1., -1.]]))
    mean, scales, beta = [.2, -.1], [.5, .8], .23
    anchor = np.diag(np.array(scales)**-2)
    q = anchor+p.smoothing.T@p.smoothing+beta*p.G.T@np.linalg.solve(p.covariance, p.G)
    expected = np.linalg.solve(q, anchor@mean+beta*p.G.T@np.linalg.solve(p.covariance, p.data))
    target, factor = p.gaussian_factor(mean, scales, beta=beta)
    np.testing.assert_allclose(target, expected, atol=1e-14)
    samples = target+solve_triangular(factor, np.random.default_rng(7).normal(size=(2, 4096))).T
    record = AltarPosterior(samples, beta)
    assert gaussian_stage_errors(p, record, mean=mean, scales=scales)['passes']
    record.samples += 1
    assert not gaussian_stage_errors(p, record, mean=mean, scales=scales)['passes']
    write_output(tmp_path/'results', record.samples, beta)
    manifest = dict(output_dir='results', n_parameters=2, status='failed', prior='gaussian')
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    report = save_stage_diagnostics(p, tmp_path, mean=mean, scales=scales)
    assert not report[0]['passes'] and report[0]['beta'] == beta
    assert (tmp_path/'stage_targets.json').exists()
    with pytest.raises(ValueError, match='stage imports'):
        AltarResultImporter().load(tmp_path/'results', manifest=manifest, allow_incomplete=True, stage_file='../step_final.h5')


def test_cuda_bundle_is_binary_and_has_no_identity_matrix(tmp_path):
    from slipkit.core.bayesian import AltarCudaConfigBuilder
    g, d = np.arange(20000.).reshape(10000, 2), np.ones(10000)
    paths = AltarDataExporter(tmp_path).export_cuda(g, d)
    assert set(paths) == {'green', 'data'}
    assert not (tmp_path/'cd.txt').exists()
    assert sum(p.stat().st_size for p in tmp_path.iterdir()) < 300000
    with h5py.File(paths['green']) as data:
        np.testing.assert_array_equal(data['green'], g)
    config = AltarCudaConfigBuilder(288, 351, tmp_path).build()
    assert 'parameters = 288' in config and 'job.gpus = 1' in config and 'float64' in config
    assert 'cd_file' not in config and 'cd_std = 1' in config
    assert 'min_mc_steps = 1000' in config and 'max_mc_steps = 4000' in config
    assert 'scaling = 0.1' in config
    assert 'useFixedScaling = True' in AltarCudaConfigBuilder(2, 3, tmp_path, sampler='fixed').build()
