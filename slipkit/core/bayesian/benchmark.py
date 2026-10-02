"""Bounded, reproducible CPU experiments; each invocation preserves its own report."""
import argparse
import json
import os
import platform
import hashlib
from pathlib import Path
import tempfile
import time
import numpy as np
from .problem import AltarProblem
from .solver import AltarBayesianSolver
from .gaussian import GaussianBayesianSolver
from .cuda import AltarCudaBayesianSolver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--work-dir', default='./altar-benchmark')
    parser.add_argument('--backend', choices=['altar', 'gaussian', 'cuda'], default='altar')
    parser.add_argument('--cpu-kernel', choices=['native', 'vectorized'], default='native')
    parser.add_argument('--chains', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--timeout', type=float, default=3600.)
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 31, 47])
    parser.add_argument('--prior-scale', type=float, default=.5)
    parser.add_argument('--initial-scaling', type=float, default=.1)
    parser.add_argument('--acceptance-weight', type=float, default=8/9)
    parser.add_argument('--rejection-weight', type=float, default=1/9)
    parser.add_argument('--fixed-scales', type=float, nargs='+')
    parser.add_argument('--cuda-sampler', choices=['fixed', 'adaptive'], default='adaptive')
    parser.add_argument('--min-steps', type=int, default=1000)
    parser.add_argument('--max-steps', type=int, default=4000)
    parser.add_argument('--target-correlation', type=float, default=.2)
    parser.add_argument('--gpu-id', type=int, default=0)
    args = parser.parse_args()
    experiment_started = time.monotonic()
    parent = Path(args.work_dir)
    parent.mkdir(parents=True, exist_ok=True)
    experiment = Path(tempfile.mkdtemp(prefix='experiment-', dir=parent))
    path = experiment/'benchmark.json'
    print(f'Report: {path}', flush=True)
    try:
        def load_array(stem, *, ndmin=1):
            binary = args.case/(stem+'.npy')
            return np.load(binary, allow_pickle=False) if binary.exists() else np.loadtxt(args.case/(stem+'.txt'), ndmin=ndmin)
        problem = AltarProblem(load_array('green', ndmin=2), load_array('data'), load_array('cd', ndmin=2))
        mean, cov = problem.gaussian_reference(scales=args.prior_scale)
        p = len(mean)
        std = np.sqrt(np.diag(cov))
        rng = np.random.default_rng(0)
        _, eigenvectors = np.linalg.eigh(cov)
        contrasts = np.diff(np.eye(p), axis=0)[::max(1, p//8)]
        projections = np.vstack((np.eye(p), rng.normal(size=(8, p)), eigenvectors[:, :3].T,
                                 eigenvectors[:, -3:].T, contrasts))
        reference_variance = np.einsum('ij,jk,ik->i', projections, cov, projections)
    except Exception as exc:
        path.write_text(json.dumps([dict(status='failed', passes=False, case=str(args.case),
            error=f'{type(exc).__name__}: {exc}')], indent=2)+'\n')
        raise SystemExit(f'Input/reference validation failed; report retained in {path}') from exc
    setup_seconds = time.monotonic()-experiment_started
    metadata = dict(setup_seconds=setup_seconds, environment=dict(platform=platform.platform(), numpy=np.__version__,
        threads={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS')}),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    (experiment/'experiment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    report = []
    scales = args.fixed_scales or [args.initial_scaling]
    for scale in scales:
        for seed in args.seeds:
            solver = None
            start = time.monotonic()
            entry = dict(seed=seed, particles=args.chains, parameters=p, observations=len(problem.data),
                         covariance_bytes=problem.covariance_bytes, backend=args.backend,
                         cpu_kernel=args.cpu_kernel if args.backend == 'altar' else None, initial_scaling=scale)
            try:
                if args.backend == 'gaussian':
                    solver = GaussianBayesianSolver(seed=seed, draws=args.chains, prior_scales=args.prior_scale)
                else:
                    factory = AltarCudaBayesianSolver if args.backend == 'cuda' else AltarBayesianSolver
                    cuda_options = dict(sampler=args.cuda_sampler, min_steps=args.min_steps, max_steps=args.max_steps, target_correlation=args.target_correlation, gpu_id=args.gpu_id) if args.backend == 'cuda' else {}
                    solver = factory(**cuda_options, work_dir=str(experiment), seed=seed, chains=args.chains,
                        steps=args.steps, prior_scales=args.prior_scale, timeout=args.timeout, cpu_kernel=args.cpu_kernel,
                        initial_scaling=scale, acceptance_weight=0. if args.fixed_scales else args.acceptance_weight,
                        rejection_weight=scale if args.fixed_scales else args.rejection_weight)
                solver.solve_problem(problem)
                entry['inference_seconds'] = time.monotonic()-start
                validation_started = time.monotonic()
                posterior = solver.get_last_posterior()
                variance = (posterior.samples @ projections.T).var(axis=0, ddof=1)
                mean_error = float(np.max(np.abs(posterior.samples.mean(axis=0)-mean)/std))
                variance_error = float(np.max(np.abs(variance/reference_variance-1)))
                entry.update(max_normalized_mean_error=mean_error, max_projection_variance_relative_error=variance_error,
                             passes=mean_error < .25 and variance_error < .3, status='complete',
                             posterior_validation_seconds=time.monotonic()-validation_started)
            except (Exception, KeyboardInterrupt) as exc:
                entry.update(passes=False, status='failed', error=f'{type(exc).__name__}: {exc}')
                if isinstance(exc, KeyboardInterrupt):
                    entry['interrupted'] = True
            entry.update(seconds=time.monotonic()-start, run_path=solver.last_run_path if solver else None)
            if entry['run_path']:
                progress = Path(entry['run_path'])/'progress.json'
                if progress.exists():
                    entry['progress'] = json.loads(progress.read_text())
                from .diagnostics import save_stage_diagnostics
                stage_started = time.monotonic()
                try:
                    entry['stage_targets'] = save_stage_diagnostics(problem, entry['run_path'], scales=args.prior_scale)
                    entry['stage_gate_passes'] = bool(entry['stage_targets']) and all(
                        stage.get('passes', False) for stage in entry['stage_targets'])
                    entry['passes'] = entry['passes'] and entry['stage_gate_passes']
                except Exception as exc:
                    entry.update(passes=False, stage_diagnostics_error=f'{type(exc).__name__}: {exc}')
                entry['stage_validation_seconds'] = time.monotonic()-stage_started
            entry.update(total_seconds=time.monotonic()-start, experiment_elapsed_seconds=time.monotonic()-experiment_started,
                         setup_seconds=setup_seconds)
            report.append(entry)
            path.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(entry), flush=True)
            if entry.get('interrupted'):
                raise SystemExit(f'Interrupted; report retained in {path}')
    if not all(r['passes'] for r in report):
        raise SystemExit(f'Accuracy/completion gate failed; diagnostics retained in {path}')


if __name__ == '__main__':
    main()
