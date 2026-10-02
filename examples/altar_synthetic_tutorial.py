# %% [markdown]
# # CPU AlTar posterior: a self-contained synthetic example
# Install the separate native AlTar/Pyre framework first. Geometry is in km;
# slip, displacement, sigma and normalized ramp coefficients are in metres.
# This example uses an unbounded independent Gaussian prior, not a rake constraint.

# %%
import tempfile
import numpy as np
from slipkit.core.fault import TriangularFaultMesh
from slipkit.core.data import GeodeticDataSet
from slipkit.core.physics import CutdeCpuEngine
from slipkit.core.inversion import InversionOrchestrator
from slipkit.core.bayesian import AltarAssembler, AltarBayesianSolver, GaussianBayesianSolver

vertices = np.array([[0., 0., -5.], [2., 0., -5.], [0., 2., -7.]])
fault = TriangularFaultMesh((vertices, np.array([[0, 1, 2]])))
xy = np.array([[-2., -2.], [0., 0.], [2., 2.], [4., 0.]])
coords = np.repeat(np.column_stack((xy, np.zeros(len(xy)))), 3, axis=0)
dataset = GeodeticDataSet(coords, np.zeros(len(coords)), np.tile(np.eye(3), (len(xy), 1)),
                          np.full(len(coords), .01), 'GNSS')
engine = CutdeCpuEngine()
G = engine.build_kernel(fault, dataset)
truth = np.array([.3, .1])
dataset.data = G @ truth + np.random.default_rng(0).normal(0., dataset.sigma)

# %%
assembler = AltarAssembler(length_unit='km')
solver = AltarBayesianSolver(work_dir=tempfile.mkdtemp(prefix='slipkit-tutorial-'),
                            prior_scales=[.5, .3], prior_mean=0., seed=17,
                            chains=1024, steps=200, output_freq=2)
# For this fully Gaussian model, explicitly select exact inference instead:
# solver = GaussianBayesianSolver(prior_scales=[.5, .3], prior_mean=0., draws=4096, seed=17)
orchestrator = InversionOrchestrator()
orchestrator.add_fault(fault)
orchestrator.add_data(dataset)
orchestrator.set_engine(engine)
orchestrator.set_assembler(assembler)
orchestrator.set_solver(solver)
result = orchestrator.run_inversion(lambda_spatial=0.)
print('Annealing complete:', result.annealing_complete)
print('Sampling diagnostics:', result.posterior.diagnostics)
print('Posterior mean slip (m):', result.get_mean_slip())
print('Posterior standard deviation (m):', result.get_posterior_std())
print('95% equal-tailed intervals (m):', result.get_credible_intervals())
print('Retained run:', result.posterior.run_path)

# %%
problem = assembler.assemble_problem([fault], [dataset], engine, None, 0.)
exact_mean, exact_covariance = problem.gaussian_reference(scales=[.5, .3])
mean_error = np.abs(result.get_mean_slip()-exact_mean)/np.sqrt(np.diag(exact_covariance))
variance_error = np.abs(np.diag(result.posterior.covariance)/np.diag(exact_covariance)-1.)
print('Exact Gaussian mean:', exact_mean)
print('Normalized mean error:', mean_error)
print('Relative marginal variance error:', variance_error)
assert np.max(mean_error) < .25
assert np.max(variance_error) < .3
np.testing.assert_allclose(engine.predict(fault, dataset, result.get_fault_vector()),
                           G @ result.get_mean_slip())
print('Moment samples (N m):', result.seismic_moment_samples(length_unit='km')[:5])
