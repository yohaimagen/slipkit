"""Explicit single-GPU native AlTar static contract; never falls back to CPU."""
import hashlib
import inspect
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from .config import AltarConfigBuilder
from .solver import AltarBayesianSolver


class AltarCudaConfigBuilder(AltarConfigBuilder):
    def __init__(self, *args, sampler='adaptive', min_steps=1000, max_steps=4000,
                 target_correlation=.2, gpu_id=0, **kwargs):
        super().__init__(*args, **kwargs)
        if sampler not in ('fixed', 'adaptive'):
            raise ValueError('CUDA sampler must be fixed or adaptive.')
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in (min_steps, max_steps)) or min_steps > max_steps:
            raise ValueError('CUDA update limits must be positive, with min_steps <= max_steps.')
        if not 0 < target_correlation < 1 or not isinstance(gpu_id, int) or isinstance(gpu_id, bool) or gpu_id < 0:
            raise ValueError('Use correlation in (0,1) and a nonnegative GPU index.')
        self.sampler, self.min_steps, self.max_steps = sampler, min_steps, max_steps
        self.target_correlation, self.gpu_id = target_correlation, gpu_id

    def build(self):
        lines = ['slipmodel:', '    model:', f'        case = {self.case_dir}',
                 '        green = green.h5', '        psets_list = [theta]', '        dataobs:',
                 f'            observations = {self.n_observations}', '            data_file = data.h5',
                 '            cd_std = 1', '        psets:', '            theta = altar.cuda.models.parameterset',
                 '            theta:', f'                count = {self.n_parameters}',
                 f'                prior = altar.cuda.distributions.{self.prior}', '                prior:']
        lines += (['                    mean = 0', '                    sigma = 1'] if self.prior == 'gaussian' else ['                    support = (0, 1)'])
        lines += [f'    rng.seed = {self.seed}', '    controller:',
            f'        sampler = altar.cuda.bayesian.{"adaptivemetropolis" if self.sampler == "adaptive" else "metropolis"}', '        sampler:']
        if self.sampler == 'adaptive':
            # Pinned upstream initializes the sampler before the model parameter count;
            # its sqrt(count) divisor is therefore one. Set and verify the actual scale.
            lines += [f'            scaling = {self.initial_scaling}', f'            parameters = {self.n_parameters}',
                f'            min_mc_steps = {self.min_steps}',
                f'            max_mc_steps = {self.max_steps}', f'            corr_check_steps = {math.gcd(100, self.min_steps, self.max_steps)}',
                f'            target_correlation = {self.target_correlation}']
        else:
            lines += [f'            scaling = {self.initial_scaling}', '            useFixedScaling = True']
        lines += ['        archiver:', f'            output_dir = {self.output_dir}', f'            output_freq = {self.output_freq}',
            '    job.tasks = 1', '    job.gpus = 1', '    job.gpuprecision = float64', f'    job.gpuids = [{self.gpu_id}]',
            f'    job.chains = {self.chains}', f'    job.steps = {self.steps}', '']
        return '\n'.join(lines)


class AltarCudaBayesianSolver(AltarBayesianSolver):
    backend_name = 'single GPU native CUDA static, float64'

    def __init__(self, *, sampler='adaptive', min_steps=1000, max_steps=4000,
                 target_correlation=.2, gpu_id=0, **kwargs):
        if kwargs.get('launcher') is not None or kwargs.get('cpu_kernel', 'native') != 'native':
            raise ValueError('CUDA uses its explicit native launcher, not CPU/custom launchers.')
        super().__init__(**kwargs)
        self.cuda_options = dict(sampler=sampler, min_steps=min_steps, max_steps=max_steps,
                                 target_correlation=target_correlation, gpu_id=gpu_id)
        AltarCudaConfigBuilder(1, 1, '.', **self.cuda_options)

    def _progress(self, run, started):
        import json
        path = Path(run)/'progress.json'
        previous = json.loads(path.read_text()) if path.exists() else {}
        report = super()._progress(run, started)
        peak = previous.get('gpu_peak_memory_bytes')
        host_peak = previous.get('peak_memory_bytes')
        state = Path(run)/'gpu-process.json'
        if state.exists():
            pid = json.loads(state.read_text())['pid']
            status = Path(f'/proc/{pid}/status')
            if status.exists():
                for line in status.read_text().splitlines():
                    if line.startswith('VmHWM:'):
                        host_peak = max(host_peak or 0, int(line.split()[1])*1024)
                        break
            try:
                rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory,gpu_uuid',
                    '--format=csv,noheader,nounits'], text=True, timeout=3).splitlines()
                for row in rows:
                    fields = [v.strip() for v in row.split(',')]
                    if len(fields) == 3 and fields[0] == str(pid) and fields[1].isdigit():
                        peak = max(peak or 0, int(fields[1])*1024**2)
                        report['gpu_uuid'] = fields[2]
            except (subprocess.SubprocessError, OSError):
                pass  # A missing measurement remains explicitly null; numerical checks are unaffected.
        report['gpu_peak_memory_bytes'] = peak
        report['gpu_uuid'] = report.get('gpu_uuid', previous.get('gpu_uuid'))
        report['peak_memory_bytes'] = max(host_peak or 0, report.get('peak_memory_bytes') or 0) or None
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(path)
        return report

    def _after_run(self, run, manifest):
        import json
        import re
        checks = json.loads((Path(run)/'numerical-parity.json').read_text())
        if not checks['passes'] or checks['precision'] != 'float64':
            raise RuntimeError('Native CUDA did not pass the deterministic float64 contract.')
        if self.cuda_options['sampler'] == 'adaptive':
            log = (Path(run)/'sampler.log').read_text()
            initial = re.search(r'Adaptive Metropolis Sampler: initial scaling\s+([\d.eE+-]+)', log)
            if initial is None or not math.isclose(float(initial[1]), self.initial_scaling, rel_tol=1e-10):
                raise RuntimeError('Native adaptive proposal scale differs from requested initial scale.')
            checks['actual_initial_scaling'] = float(initial[1])
        manifest['prior_offset'] += checks['prior_normalization_offset']
        manifest['numerical_parity'] = checks

    def _command(self, binary, pfg):
        # Module execution prevents our cuda.py from shadowing Pyre's top-level cuda package.
        return [sys.executable, '-m', 'slipkit.core.bayesian.native_cuda', f'--config={pfg}']

    def _configuration(self, problem, case, results):
        if self.prior == 'gaussian' and self.chains*problem.G.shape[1] % 2:
            raise ValueError('Native cuRAND Gaussian initialization requires an even particles*parameters count.')
        return AltarCudaConfigBuilder(problem.G.shape[1], len(problem.data), case,
            chains=self.chains, steps=self.steps, output_dir=results, output_freq=self.output_freq,
            seed=self.seed, prior=self.prior, initial_scaling=self.initial_scaling, **self.cuda_options)

    def _export_inputs(self, exporter, green, data):
        return exporter.export_cuda(green, data)

    def _extra_manifest(self):
        return dict(cuda=self.cuda_options, precision='float64', identity_noise='scalar; no covariance allocation',
                    randomness=dict(seed=self.seed, initialization='seeded native cuRAND',
                        bitwise_reproducible=False,
                        limitation='Native atomicAdd valid-particle queue changes acceptance-draw assignment.'))

    def _preflight(self):
        try:
            import cuda
            from altar.cuda.data.cudaDataL2 import cudaDataL2
            from altar.models.seismic.cuda.cudaStatic import cudaStatic
            from altar.cuda.bayesian.cudaMetropolis import cudaMetropolis
            from altar.cuda.bayesian.cudaAdaptiveMetropolis import cudaAdaptiveMetropolis
            from altar.cuda.models.cudaBayesian import cudaBayesian
            from altar.cuda.distributions.cudaGaussian import cudaGaussian
            from altar.cuda.distributions.cudaUniform import cudaUniform
        except ImportError as exc:
            raise RuntimeError('Native AlTar CUDA build unavailable; CPU fallback is disabled.') from exc
        classes = (cudaDataL2, cudaStatic, cudaMetropolis, cudaAdaptiveMetropolis, cudaBayesian, cudaGaussian, cudaUniform)
        hashes = {cls.__name__: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest() for cls in classes}
        import json
        expected = json.loads(Path(__file__).with_name('cuda_contract.json').read_text())
        if hashes != expected['source_sha256']:
            raise RuntimeError('CUDA sources differ from the pinned numerical contract; validate before use.')
        # Real device access, not an import-only claim; the native application also checks its worker.
        devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total', '--format=csv,noheader'], text=True)
        launcher = Path(__file__).with_name('native_cuda.py')
        hashes['bridge_cuda_adapter'] = hashlib.sha256(launcher.read_bytes()).hexdigest()
        return str(launcher), dict(python=sys.executable, platform=platform.platform(), source_hashes=hashes,
                                  native_commits=expected['commits'], gpu_inventory=devices)
