"""Serial CPU AlTar execution with explicit priors and isolated run provenance."""
import hashlib
import inspect
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import warnings
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from slipkit.core.solvers import SolverStrategy
from .config import AltarConfigBuilder
from .exporter import AltarDataExporter
from .importer import AltarResultImporter
from .problem import AltarProblem, file_hash


class AltarBayesianSolver(SolverStrategy):
    def __init__(self, work_dir='./altar_run', alpha_cp=0., ss_prior_sigma=None,
                 chains=1024, steps=1000, tasks=1, output_freq=1, keep_work_dir=True,
                 *, prior='gaussian', prior_scales=None, prior_mean=None, seed=17,
                 output_dir='results', timeout=3600., launcher=None, cp_policy='fixed',
                 initial_scaling=.1, acceptance_weight=8/9, rejection_weight=1/9, cpu_kernel='native'):
        if prior_scales is not None and ss_prior_sigma is not None:
            raise ValueError('Supply prior_scales or legacy ss_prior_sigma, not both.')
        if prior not in ('gaussian', 'uniform'):
            raise ValueError('Only Gaussian and independent finite uniform priors are supported.')
        if prior == 'uniform' and any(v is not None for v in (prior_scales, prior_mean, ss_prior_sigma)):
            raise ValueError('Gaussian prior controls do not apply to a uniform prior.')
        if cp_policy != 'fixed':
            raise ValueError('Only fixed observation-based Cp is supported; stage/proposal updates are unsupported.')
        if not np.isfinite(alpha_cp) or alpha_cp < 0:
            raise ValueError('alpha_cp must be finite and nonnegative.')
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite and positive.')
        if os.path.isabs(output_dir) or '..' in output_dir.split(os.sep) or output_dir.split(os.sep)[0] in ('', '.', 'case', 'linear.pfg', 'manifest.json', 'sampler.log', 'physical_problem.npz', 'prior_transform.npz', 'progress.json', 'progress.tmp', 'numerical-parity.json', 'gpu-process.json', 'stage_targets.json', 'host-resource.txt'):
            raise ValueError('output_dir must name a relative directory within each isolated run.')
        if cpu_kernel not in ('native', 'vectorized') or (cpu_kernel == 'vectorized' and launcher is not None):
            raise ValueError('cpu_kernel must be native or vectorized; vectorized uses its own explicit launcher.')
        self.cpu_kernel = cpu_kernel
        self.initial_scaling, self.acceptance_weight, self.rejection_weight = initial_scaling, acceptance_weight, rejection_weight
        self.work_dir = os.path.abspath(work_dir)
        self.alpha_cp, self.prior, self.cp_policy = alpha_cp, prior, cp_policy
        self.prior_scales = prior_scales if prior_scales is not None else (.5 if ss_prior_sigma is None else ss_prior_sigma)
        self.prior_mean = 0. if prior_mean is None else prior_mean
        self.chains, self.steps, self.tasks = chains, steps, tasks
        self.seed, self.output_dir, self.output_freq = seed, output_dir, output_freq
        self.keep_work_dir, self.timeout, self.launcher = keep_work_dir, timeout, launcher
        self.last_posterior = self.last_result = self.last_run_path = self.last_problem = self.last_manifest = None
        # Validate execution options without creating files.
        AltarConfigBuilder(1, 1, '.', chains, steps, tasks, output_dir, output_freq, seed, prior, initial_scaling, acceptance_weight, rejection_weight)
        if chains < 2:
            raise ValueError('At least two particles are required.')

    def reset(self):
        self.last_posterior = self.last_result = self.last_run_path = self.last_problem = self.last_manifest = None

    def get_last_posterior(self):
        return self.last_posterior

    def get_last_result(self):
        """Geometry-aware result exists only after an orchestrated inversion."""
        return self.last_result

    def save_result(self, path, problem=None, *, faults=(), datasets=(), length_unit='km'):
        """Explicit durable bundle for either solver, including actual prior settings."""
        from .artifact import save_inference
        if self.last_posterior is None:
            raise ValueError('No posterior is available to save.')
        problem = self.last_problem if problem is None else problem
        prior = dict(kind=self.prior)
        if self.prior == 'gaussian':
            prior.update(anchor_mean=np.broadcast_to(self.prior_mean, (problem.G.shape[1],)),
                         scales=np.broadcast_to(self.prior_scales, (problem.G.shape[1],)))
        else:
            offset, widths = self.last_prior_transform
            prior.update(lower=offset.tolist(), upper=(offset+widths).tolist())
        return save_inference(path, problem, self.last_posterior, prior=prior,
            alpha_cp=self.alpha_cp, seed=self.seed, faults=faults, datasets=datasets,
            length_unit=length_unit, inference_settings=self.last_manifest)

    def solve(self, A, b, bounds=None):
        """Legacy matrix adapter: b=[raw observations, positive sigma]."""
        self.reset()
        a, packed = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
        if a.ndim != 2 or packed.shape != (2*a.shape[0],):
            raise ValueError('Expected A=(N_obs,N_param) and b=[data|sigma] with 2*N_obs entries.')
        n = a.shape[0]
        sigma = packed[n:]
        if not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError('sigma must be finite and strictly positive.')
        return self.solve_problem(AltarProblem(a, packed[:n], sigma**2), bounds)

    def _transform(self, problem, bounds):
        p = problem.G.shape[1]
        if self.prior == 'uniform':
            if bounds is None:
                raise ValueError('Uniform prior requires explicit finite lower and upper bounds.')
            if problem.smoothing is not None and np.any(problem.smoothing):
                raise ValueError('Uniform bounds with Gaussian smoothing are not supported.')
            if np.any(np.asarray(self.prior_mean) != 0) or np.any(np.asarray(self.prior_scales) != .5):
                raise ValueError('Gaussian prior_mean/scales do not apply to a uniform prior.')
            lo, hi = (np.broadcast_to(np.asarray(v, dtype=float), (p,)).copy() for v in bounds)
            if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi <= lo):
                raise ValueError('Uniform bounds must be finite and strictly increasing; fixed parameters are unsupported.')
            return lo, hi-lo
        if bounds is not None:
            lo, hi = (np.broadcast_to(np.asarray(v, dtype=float), (p,)) for v in bounds)
            if not (np.all(np.isneginf(lo)) and np.all(np.isposinf(hi))):
                raise ValueError('Bounded Gaussian priors are unsupported; choose uniform explicitly for a uniform box prior.')
        scales = np.broadcast_to(np.asarray(self.prior_scales, dtype=float), (p,))
        mean = np.broadcast_to(np.asarray(self.prior_mean, dtype=float), (p,))
        if not np.isfinite(scales).all() or np.any(scales <= 0) or not np.isfinite(mean).all():
            raise ValueError('Gaussian means must be finite and scales finite and positive.')
        if problem.smoothing is None or not np.any(problem.smoothing):
            return mean.copy(), scales.copy()
        anchor = scales**-2
        q = np.diag(anchor)
        if problem.smoothing is not None:
            q += problem.smoothing.T @ problem.smoothing
        r = np.linalg.cholesky(q).T
        offset = cho_solve((r, False), anchor*mean)
        matrix = solve_triangular(r, np.eye(p))
        return offset, matrix

    def _preflight(self):
        """The local native model contract is checked before exporting inputs."""
        try:
            import altar
            import pyre
            from altar.models.linear.Linear import Linear
            from altar.distributions.Uniform import Uniform
            from altar.distributions.Gaussian import Gaussian
            from altar.bayesian.Recorder import Recorder
            from altar.norms.L2 import L2
            from altar.bayesian.Metropolis import Metropolis
        except ImportError as exc:
            raise RuntimeError('CPU AlTar/Pyre native framework is unavailable in this Python environment.') from exc
        binary = self.launcher or shutil.which('linear', path=os.path.dirname(sys.executable)) or shutil.which('linear')
        if binary is None or not os.path.isfile(binary):
            raise RuntimeError("AlTar 'linear' launcher not found; use the native AlTar/Pyre environment.")
        hashes = {}
        for cls in (Linear, Uniform, Gaussian, Recorder, L2, Metropolis):
            path = inspect.getfile(cls)
            with open(path, 'rb') as stream:
                hashes[cls.__name__] = hashlib.sha256(stream.read()).hexdigest()
        expected = dict(
            Linear='63a9cbb73c2c49308df466e129ad8a1f3d65745fa8423763118ea19a810d129e',
            Uniform='92aeb08f391bdd427f3faa5f316dc13a5b349959d0e7305f30a576c4532d39c0',
            Gaussian='f35b1c946d307a678c0973f819be169cddd12f6130cfae7ac79c448a04572ef9',
            L2='c45ab5a7c9bbf4a7dcb54dcdc360aaf8b4d491960c4e0bb4826be2953fdead14',
            Metropolis='e60896b19caaa3ce3ef29f270586831a829f8977321bff5d936d55fb7d518d7d',
            Recorder='3fd34b5173cfb0a9885606d9cd1e374b712e9ed51f0a17d4a899d47de71c6245',
        )
        if hashes != expected:
            raise RuntimeError('Native sources differ from the verified CPU backend contract; validate this revision before use.')
        check_native_likelihood()
        if self.cpu_kernel == 'vectorized':
            binary = os.path.join(os.path.dirname(__file__), 'native_cpu.py')
            with open(binary, 'rb') as stream:
                hashes['bridge_cpu_adapter'] = hashlib.sha256(stream.read()).hexdigest()
        from altar import meta as altar_meta
        from pyre import meta as pyre_meta
        return os.path.abspath(binary), dict(python=sys.executable, platform=platform.platform(),
                                            source_hashes=hashes, altar_path=altar.__file__, pyre_path=pyre.__file__,
                                            altar_version=altar_meta.version, pyre_version=pyre_meta.version)

    def _after_run(self, run, manifest):
        pass

    def _extra_manifest(self):
        return {}

    backend_name = 'serial CPU linear'

    def _export_inputs(self, exporter, green, data):
        return exporter.export_whitened(green, data)

    def _configuration(self, problem, case, results):
        return AltarConfigBuilder(problem.G.shape[1], len(problem.data), case,
            self.chains, self.steps, self.tasks, results, self.output_freq, self.seed, self.prior,
            self.initial_scaling, self.acceptance_weight, self.rejection_weight)

    def solve_problem(self, problem, bounds=None):
        self.reset()
        if not isinstance(problem, AltarProblem):
            raise TypeError('solve_problem requires AltarProblem.')
        offset, matrix = self._transform(problem, bounds)
        binary, backend = self._preflight()
        if self.chains <= problem.G.shape[1]:
            warnings.warn('Population is no larger than parameter count; sampling adequacy needs separate validation.', stacklevel=2)
        os.makedirs(self.work_dir, exist_ok=True)
        run = tempfile.mkdtemp(prefix='run-', dir=self.work_dir)
        self.last_run_path = run
        results = os.path.join(run, self.output_dir)
        case = os.path.join(run, 'case')
        g, d, logdet = problem.whiten(self.alpha_cp)
        transformed_g = g*matrix if matrix.ndim == 1 else g @ matrix
        transformed_d = d-g @ offset
        paths = self._export_inputs(AltarDataExporter(case), transformed_g, transformed_d)
        config = self._configuration(problem, case, results)
        pfg = config.save(os.path.join(run, 'linear.pfg'))
        manifest = dict(backend=self.backend_name, environment=backend, n_parameters=problem.G.shape[1],
                        n_observations=len(problem.data), layout=problem.layout, prior=self.prior,
                        noise_representation='blocks' if isinstance(problem.covariance, tuple) else
                                             'diagonal variances' if problem.covariance.ndim == 1 else 'full covariance',
                        transform=dict(offset=offset.tolist(), scales=matrix.tolist()) if matrix.ndim == 1 else {}, seed=self.seed,
                        chains=self.chains, steps=self.steps, output_dir=self.output_dir, output_freq=self.output_freq,
                        cp_policy='fixed observation-based', alpha_cp=self.alpha_cp, representation='observation whitened once, including fixed Cp; affine prior coordinates',
                        cpu_kernel=self.cpu_kernel, initial_scaling=self.initial_scaling, acceptance_weight=self.acceptance_weight,
                        rejection_weight=self.rejection_weight, likelihood_offset=-.5*logdet,
                        prior_offset=-float(np.log(matrix).sum() if matrix.ndim == 1 else np.linalg.slogdet(matrix)[1]),
                        parameter_sets=[dict(name='theta', width=problem.G.shape[1])], status='running')
        manifest.update(self._extra_manifest())
        if matrix.ndim == 2:
            transform_path = os.path.join(run, 'prior_transform.npz')
            np.savez(transform_path, offset=offset, matrix=matrix)
            with open(transform_path, 'rb') as stream:
                manifest['transform'] = dict(file='prior_transform.npz', sha256=hashlib.sha256(stream.read()).hexdigest())
        np.savez(os.path.join(run, 'physical_problem.npz'), G=problem.G, data=problem.data,
                 **problem.covariance_arrays, alpha_cp=self.alpha_cp, smoothing=problem.smoothing if problem.smoothing is not None else np.zeros((0, problem.G.shape[1])))
        manifest['input_sha256'] = {}
        for name, path in paths.items():
            manifest['input_sha256'][name] = file_hash(path)
        manifest_path = os.path.join(run, 'manifest.json')
        def save_manifest():
            with open(manifest_path, 'w') as stream:
                json.dump(manifest, stream, indent=2)
        save_manifest()
        try:
            self._run_altar(binary, pfg, run)
            self._after_run(run, manifest)
            posterior = AltarResultImporter().load(results, manifest=manifest)
            manifest['status'] = 'complete'
            save_manifest()
        except BaseException as exc:
            manifest['status'] = 'failed'
            manifest['error'] = str(exc)
            save_manifest()
            raise
        if not self.keep_work_dir:
            shutil.rmtree(run)
            posterior.run_path = None
            posterior.step_files = []
            self.last_run_path = None
        self.last_posterior, self.last_problem = posterior, problem
        self.last_prior_transform = (offset, matrix)
        self.last_manifest = manifest
        return posterior.mean

    def _progress(self, run, started):
        from .progress import save_progress
        return save_progress(run, started, self.initial_scaling)

    def _command(self, binary, pfg):
        return [sys.executable, binary, f'--config={pfg}']

    def _run_altar(self, binary, pfg, run):
        log = os.path.join(run, 'sampler.log')
        command = self._command(binary, pfg)
        if sys.platform != 'darwin' and os.path.isfile('/usr/bin/time'):
            command = ['/usr/bin/time', '-v'] + command
        started = time.monotonic()
        with open(log, 'w') as stream:
            process = subprocess.Popen(command, cwd=run, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while True:
                    remaining = self.timeout-(time.monotonic()-started)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, self.timeout)
                    try:
                        code = process.wait(timeout=min(1., remaining))
                        break
                    except subprocess.TimeoutExpired:
                        self._progress(run, started)
            except BaseException as exc:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                except ProcessLookupError:
                    pass
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    self._progress(run, started)
                if isinstance(exc, KeyboardInterrupt):
                    raise
                raise RuntimeError(f'AlTar interrupted or timed out. Logs retained at {log}') from exc
        self._progress(run, started)
        if code:
            raise RuntimeError(f'AlTar exited with code {code}. Logs retained at {log}')


def check_native_likelihood():
    """Release gate for the actual whitened native norm and normalization."""
    import altar
    from altar.models.linear.Linear import Linear
    from altar.norms.L2 import L2
    for correlation in (.8, -.8, 0.):
        for scale in (1., 1e-8, 1e3):
            covariance = scale*np.array([[1., correlation], [correlation, 2.]])
            c = altar.matrix(shape=(2, 2)); c.ndarray()[:] = np.eye(2)
            factor = Linear.computeCovarianceInverse(None, c)
            normalization = Linear.computeNormalization(None, 2, c)
            lc = np.linalg.cholesky(covariance)
            offset = -np.log(np.diag(lc)).sum()
            for residual in ([1., 2.], [2., -1.], [0., 1.]):
                residual = np.array(residual)
                v = altar.vector(shape=2); v.ndarray()[:] = solve_triangular(lc, residual, lower=True)
                actual = normalization-.5*L2.withCovariance(None, v, factor)**2+offset
                expected = -.5*(residual @ np.linalg.solve(covariance, residual) + np.linalg.slogdet(covariance)[1] + 2*np.log(2*np.pi))
                if not np.isclose(actual, expected, rtol=1e-10, atol=1e-8):
                    raise RuntimeError('Whitened native Gaussian likelihood failed numerical parity.')
