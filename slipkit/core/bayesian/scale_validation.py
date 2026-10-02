"""Reproducible synthetic observations on real Cutde triangular geometry; no earthquake claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time
import numpy as np
from scipy.linalg import lstsq
from slipkit.core.data import GeodeticDataSet
from slipkit.core.fault import TriangularFaultMesh
from slipkit.core.physics import CutdeCpuEngine
from .assembler import AltarAssembler
from .gaussian import GaussianBayesianSolver
from .artifact import load_inference
from .results import AltarSlipDistribution
from .diagnostics import gaussian_stage_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--nx', type=int, default=25)
    parser.add_argument('--ny', type=int, default=20)
    parser.add_argument('--observations', type=int, default=10000)
    parser.add_argument('--draws', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    if min(args.nx, args.ny, args.observations) <= 0:
        parser.error('Mesh and observation dimensions must be positive.')
    total = time.monotonic()
    parent = Path(args.work_dir)
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='geometry-', dir=parent))
    report = dict(synthetic_observations=True, physical_forward_model='Cutde halfspace',
        target='free-sign SS/DS, independent Gaussian anchor mean 0, scale .5 m; no smoothing, ramps or constraints',
        noise='independent sigma=.01 m; Cp=0', length_unit='km', seed=args.seed,
        budget=dict(total_seconds=180, peak_memory_bytes=2000000000),
        source_sha256={str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__).parents[1]/'physics.py', *Path(__file__).parent.glob('*.py')]},
        environment=dict(platform=platform.platform(), numpy=np.__version__, threads={k: os.environ.get(k) for k in
                         ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}))
    try:
        rng = np.random.default_rng(args.seed)
        x, y = np.meshgrid(np.linspace(-25, 25, args.nx+1), np.linspace(0, 30, args.ny+1))
        vertices = np.column_stack((x.ravel(), y.ravel(), -5-.5*y.ravel()))
        faces = []
        for row in range(args.ny):
            for col in range(args.nx):
                i = row*(args.nx+1)+col
                faces.extend(((i, i+1, i+args.nx+1), (i+1, i+args.nx+2, i+args.nx+1)))
        fault = TriangularFaultMesh((vertices, np.asarray(faces)))
        coords = np.column_stack((rng.uniform(-50, 50, (args.observations, 2)), np.zeros(args.observations)))
        dataset = GeodeticDataSet(coords, np.zeros(args.observations),
            np.tile([.2, -.4, np.sqrt(.8)], (args.observations, 1)), np.full(args.observations, .01), 'synthetic-los')
        engine = CutdeCpuEngine(observation_chunk_size=128)
        centers = vertices[np.asarray(faces)].mean(axis=1)
        truth = np.concatenate((.2+.1*np.cos(centers[:, 0]/20), .4+.1*np.sin(centers[:, 1]/20)))
        dataset.data = engine.predict(fault, dataset, truth)+rng.normal(0, .01, args.observations)
        report['setup_seconds'] = time.monotonic()-total
        started = time.monotonic()
        problem = AltarAssembler().assemble_problem([fault], [dataset], engine, None, 0.)
        report.update(assembly_seconds=time.monotonic()-started, triangles=fault.num_patches(),
            parameters=problem.G.shape[1], observations=len(dataset), kernel_bytes=problem.G.nbytes,
            covariance_bytes=problem.covariance_bytes)
        solver = GaussianBayesianSolver(draws=args.draws, seed=args.seed)
        started = time.monotonic()
        solver.solve_problem(problem)
        report['inference_seconds'] = time.monotonic()-started
        started = time.monotonic()
        # Augmented QR reference avoids relying only on normal-equation residuals.
        gw, dw, _ = problem.whiten()
        p = gw.shape[1]
        augmented = np.vstack((gw, 2*np.eye(p)))
        reference, _, _, _ = lstsq(augmented, np.concatenate((dw, np.zeros(p))), lapack_driver='gelsy')
        posterior = solver.get_last_posterior()
        mean_error = float(np.max(np.abs(posterior.mean-reference)/posterior.std))
        projection = np.random.default_rng(111).normal(size=p)
        from scipy.linalg import solve_triangular
        target_variance = np.linalg.norm(solve_triangular(posterior.precision_factor.T, projection, lower=True))**2
        variance_error = float(abs(np.var(posterior.samples@projection, ddof=1)/target_variance-1))
        diagnostics = gaussian_stage_errors(problem, posterior)
        report['particle_diagnostics'] = diagnostics
        report.update(qr_max_normalized_mean_error=mean_error, projection_variance_error=variance_error,
            validation_seconds=time.monotonic()-started, passes=mean_error < 1e-7 and variance_error < .30 and diagnostics['passes'])
        del augmented, gw
        started = time.monotonic()
        solver.save_result(run/'inference', faults=[fault], datasets=[dataset])
        restored = load_inference(run/'inference')
        np.testing.assert_allclose(restored.problem.G@restored.posterior.mean, problem.G@posterior.mean)
        np.testing.assert_array_equal(restored.result.seismic_moment_samples(),
                                     AltarSlipDistribution(posterior, [fault], [dataset]).seismic_moment_samples())
        report['persistence_seconds'] = time.monotonic()-started
    except Exception as exc:
        report.update(passes=False, error=f'{type(exc).__name__}: {exc}')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report.update(total_seconds=time.monotonic()-total, process_peak_memory_bytes=int(peak if sys.platform == 'darwin' else peak*1024))
    report['passes'] = report['passes'] and report['total_seconds'] < report['budget']['total_seconds'] and report['process_peak_memory_bytes'] < report['budget']['peak_memory_bytes']
    (run/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(run/'report.json'), **report)), flush=True)
    if not report['passes']:
        raise SystemExit('Geometry validation gate failed.')


if __name__ == '__main__':
    main()
