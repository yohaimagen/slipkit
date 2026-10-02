"""Thin native CUDA static launcher with explicit scalar identity noise."""
import numpy as np
import json
import os
from pathlib import Path
from altar.cuda.distributions.cudaGaussian import cudaGaussian
from altar.cuda.distributions.cudaUniform import cudaUniform
import altar
import altar.cuda
from altar.cuda.data.cudaDataL2 import cudaDataL2
from altar.models.seismic.cuda.cudaStatic import cudaStatic


class IdentityNoise(cudaDataL2, family='slipkit.cuda.identitynoise'):
    @altar.export
    def initialize(self, application):
        # Upstream cd_std still allocates/inverts N*N; bypass that for verified whitened inputs.
        if self.cd_file is not None or self.cd_std != 1.:
            raise ValueError('CUDA identity adapter requires whitened inputs and cd_std=1 without cd_file.')
        self.device = application.controller.worker.device
        self.precision = application.job.gpuprecision
        self.ifs, self.error = application.pfs['inputs'], application.error
        self.samples = application.job.chains
        self.dataobs = self.loadFile(self.data_file, shape=self.observations)
        if not np.isfinite(self.dataobs).all():
            raise ValueError('Nonfinite CUDA observations.')
        self.cd = None
        self.gcd_inv = 1.
        self.normalization = -.5*self.observations*np.log(2*np.pi)
        # ponytail: native observation/residual populations remain O(particles*observations); budget before scaling.
        self.gdataObsBatch = altar.cuda.matrix(shape=(self.samples, self.observations), dtype=self.precision)
        vector = altar.cuda.vector(source=self.dataobs, dtype=self.precision)
        self.gdataObsBatch.duplicateVector(src=vector)
        return self

    def updateCovariance(self, cp=None):
        raise ValueError('Noise is fixed and already whitened; CUDA covariance updates are unsupported.')


class Static(cudaStatic, family='slipkit.cuda.static'):
    dataobs = altar.cuda.data.data()
    dataobs.default = IdentityNoise(name='slipmodel.model.dataobs')

    @altar.export
    def initialize(self, application):
        super().initialize(application)
        if not np.isfinite(self.GF).all():
            raise ValueError('Nonfinite CUDA Green matrix.')
        Path('gpu-process.json').write_text(json.dumps(dict(pid=os.getpid())))
        # Seed the shared native generator; upstream prior kernels seed from clock64.
        from cuda import cuda as libcuda
        self.bridge_seed = int(application.rng.seed)
        libcuda.curand_setseed(self.device.curand_generator, self.bridge_seed)
        # Native proposals use the manager's default generator, acceptance uses worker.device.
        libcuda.curand_setseed(altar.cuda.curand.get_current_generator(), self.bridge_seed)
        self.check_numerical_contract()
        return self

    def cuInitSample(self, theta, batch):
        if all(isinstance(p.prior, cudaUniform) for p in self.psets.values()):
            altar.cuda.curand.uniform(self.device.curand_generator, out=theta)
        else:
            altar.cuda.curand.gaussian(self.device.curand_generator, out=theta)
        return self

    def check_numerical_contract(self):
        uniform = all(isinstance(p.prior, cudaUniform) for p in self.psets.values())
        for p in self.psets.values():
            if isinstance(p.prior, cudaGaussian):
                if p.prior.mean != 0 or p.prior.sigma != 1:
                    raise ValueError('CUDA bridge requires unit Gaussian prior coordinates.')
            elif not isinstance(p.prior, cudaUniform) or tuple(p.prior.support) != (0, 1):
                raise ValueError('CUDA bridge requires unit Gaussian or Uniform(0,1) coordinates.')
        batch = min(3, self.samples)
        values = np.array([.2, .5, .8] if uniform else [0., -.3, .4])[:batch]
        host = np.repeat(values[:, None], self.parameters, axis=1)
        theta = altar.cuda.matrix(source=host, dtype=self.precision)
        prior = altar.cuda.vector(shape=batch, dtype=self.precision).zero()
        score = altar.cuda.vector(shape=batch, dtype=self.precision).zero()
        self.cuEvalPrior(theta, prior, batch)
        self.cuEvalLikelihood(theta, score, batch)
        expected_prior = np.zeros(batch)
        for p in self.psets.values():
            if isinstance(p.prior, cudaGaussian):
                expected_prior -= .5*p.count*(values**2+np.log(2*np.pi))
        observed_prior = prior.copy_to_host(type='numpy')
        # Native Gaussian uses a float PI literal even in float64. Restore only its constant.
        offset = float(expected_prior[0]-observed_prior[0])
        np.testing.assert_allclose(observed_prior+offset, expected_prior, rtol=1e-10, atol=1e-10)
        residual = host @ self.GF.T-self.dataobs.dataobs
        np.testing.assert_allclose(self.gDataPred.copy_to_host(type='numpy')[:batch], residual, rtol=1e-10, atol=1e-9)
        expected_score = self.dataobs.normalization-.5*np.einsum('ij,ij->i', residual, residual)
        np.testing.assert_allclose(score.copy_to_host(type='numpy'), expected_score, rtol=1e-10, atol=1e-8)
        mask = altar.cuda.vector(shape=batch, dtype='int32').zero()
        self.cuVerify(theta, mask, batch)
        if np.any(mask.copy_to_host(type='numpy')):
            raise ValueError('Native CUDA rejected vectors within prior support.')
        if uniform:
            outside = altar.cuda.matrix(source=np.full(host.shape, -1.), dtype=self.precision)
            self.cuVerify(outside, mask.zero(), batch)
            if not np.all(mask.copy_to_host(type='numpy') > 0):
                raise ValueError('Native CUDA failed to reject outside-box vectors.')
        Path('numerical-parity.json').write_text(json.dumps(dict(passes=True, prior_normalization_offset=offset,
            parameters=self.parameters, observations=self.observations, precision=self.precision,
            seed=self.bridge_seed, initialization='seeded native cuRAND')))


class Application(altar.shells.cudaapplication, family='slipkit.cuda.application'):
    model = altar.models.model()
    model.default = Static(name='slipmodel.model')


if __name__ == '__main__':
    raise SystemExit(Application(name='slipmodel').run())
