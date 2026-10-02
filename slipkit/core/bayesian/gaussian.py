"""Explicit exact inference for the fixed linear Gaussian model; no AlTar launch."""
import numpy as np
from scipy.linalg import solve_triangular
from .solver import AltarBayesianSolver
from .problem import AltarProblem
from .results import AltarPosterior


class GaussianBayesianSolver(AltarBayesianSolver):
    """Reuse the matrix/orchestrator contract, explicitly selecting exact inference."""
    def __init__(self, *, prior_scales=.5, prior_mean=0., draws=4096, seed=17, alpha_cp=0.):
        super().__init__(prior_scales=prior_scales, prior_mean=prior_mean, chains=draws,
                         steps=1, seed=seed, alpha_cp=alpha_cp)

    def solve_problem(self, problem, bounds=None):
        self.reset()
        if not isinstance(problem, AltarProblem):
            raise TypeError('solve_problem requires AltarProblem.')
        if bounds is not None:
            p = problem.G.shape[1]
            lo, hi = (np.broadcast_to(np.asarray(v), (p,)) for v in bounds)
            if not (np.all(np.isneginf(lo)) and np.all(np.isposinf(hi))):
                raise ValueError('Exact Gaussian inference does not support finite bounds or clipping.')
        mean, factor = problem.gaussian_factor(self.prior_mean, self.prior_scales, self.alpha_cp)
        z = np.random.default_rng(self.seed).normal(size=(len(mean), self.chains))
        samples = mean + solve_triangular(factor, z).T
        self.last_posterior = AltarPosterior(samples, 1., layout=problem.layout,
            exact_mean=mean, precision_factor=factor,
            diagnostics={'inference': 'exact Gaussian', 'sampling_adequacy': 'independent Gaussian draws',
                         'draws': self.chains, 'seed': self.seed})
        self.last_problem = problem
        return mean.copy()
