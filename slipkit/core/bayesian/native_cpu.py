"""Explicit vectorized CPU linear adapter; leaves native AlTar installation intact."""
import numpy as np
import resource
import sys
import time
import altar
from altar.models.linear.Linear import Linear
from altar.distributions.Gaussian import Gaussian
from altar.distributions.Uniform import Uniform


class VectorizedLinear(Linear, family='slipkit.models.linear'):
    @altar.export
    def initialize(self, application):
        super().initialize(application)
        for distribution in (self.prep, self.prior):
            if isinstance(distribution, Gaussian):
                if distribution.mean != 0 or distribution.sigma != 1:
                    raise ValueError('Vectorized adapter requires a standardized unit Gaussian.')
            elif isinstance(distribution, Uniform):
                if tuple(distribution.support) != (0, 1):
                    raise ValueError('Vectorized adapter requires Uniform(0,1).')
            else:
                raise ValueError('Unverified prior type in vectorized CPU adapter.')
        self._memory_recorded = 0.
        self._g = self.G.ndarray()
        self._d = self.d.ndarray()
        return self

    def computeCovarianceInverse(self, cd):
        # ponytail: native text loader remains dense; a new loader needs separate verification.
        covariance = cd.ndarray()
        if not np.all(np.diag(covariance) == 1) or np.count_nonzero(covariance) != covariance.shape[0]:
            raise ValueError('Vectorized adapter accepts already whitened identity noise only.')
        return None

    def initializeResiduals(self, samples, data):
        # The batched evaluator owns its workspace; no duplicate native residual population.
        return None

    def computeNormalization(self, observations, cd):
        return -.5*observations*np.log(2*np.pi)

    @altar.export
    def priorLikelihood(self, step):
        theta = self.restrict(theta=step.theta).ndarray()
        values = step.prior.ndarray()
        if isinstance(self.prior, Gaussian):
            values[:] += -.5*(np.einsum('ij,ij->i', theta, theta)+theta.shape[1]*np.log(2*np.pi))
        else:
            values[:] += np.where(np.all((theta >= 0) & (theta <= 1), axis=1), 0., -np.inf)
        return self

    @altar.export
    def dataLikelihood(self, step):
        theta = self.restrict(theta=step.theta).ndarray()
        # ponytail: 512-particle batches bound workspaces; native text Cd remains dense.
        for start in range(0, len(theta), 512):
            stop = start+512
            residuals = theta[start:stop] @ self._g.T - self._d
            step.data.ndarray()[start:stop] = self.normalization-.5*np.einsum('ij,ij->i', residuals, residuals)
        if hasattr(self, '_memory_recorded') and time.monotonic()-self._memory_recorded >= 2:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            print(f'slipkit: peak_memory_bytes = {int(peak if sys.platform == "darwin" else peak*1024)}', flush=True)
            self._memory_recorded = time.monotonic()
        return self

    @altar.export
    def verify(self, step, mask):
        theta = self.restrict(theta=step.theta).ndarray()
        invalid = ~np.isfinite(theta).all(axis=1)
        if isinstance(self.prior, Uniform):
            invalid |= np.any((theta < 0) | (theta > 1), axis=1)
        mask.ndarray()[:] += invalid
        return mask


class Application(altar.shells.application, family='slipkit.applications.linear'):
    model = altar.models.model()
    model.default = VectorizedLinear(name="linear.model")


if __name__ == '__main__':
    status = Application(name='linear').run()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f'slipkit: peak_memory_bytes = {int(peak if sys.platform == "darwin" else peak*1024)}', flush=True)
    raise SystemExit(status)
