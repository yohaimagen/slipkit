"""Gaussian particle checks against the actual tempered target, with draw baselines."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from .importer import AltarResultImporter


def gaussian_stage_errors(problem, posterior, *, mean=0., scales=.5, alpha_cp=0., baseline_seed=111):
    target, factor = problem.gaussian_factor(mean, scales, alpha_cp, beta=posterior.final_beta)
    covariance = solve_triangular(factor, np.eye(len(target)))
    covariance = covariance @ covariance.T
    std = np.sqrt(np.diag(covariance))
    _, modes = np.linalg.eigh(covariance)
    p = len(target)
    projections = np.vstack((np.eye(p), modes[:, :3].T, modes[:, -3:].T,
                             np.diff(np.eye(p), axis=0)[::max(1, p//8)]))
    variances = np.einsum('ij,jk,ik->i', projections, covariance, projections)
    def errors(samples):
        return dict(max_normalized_mean_error=float(np.max(np.abs(samples.mean(axis=0)-target)/std)),
            max_projection_variance_relative_error=float(np.max(np.abs(
                (samples @ projections.T).var(axis=0, ddof=1)/variances-1))))
    reference = target + solve_triangular(factor, np.random.default_rng(baseline_seed).normal(size=(p, len(posterior.samples)))).T
    metrics = errors(posterior.samples)
    return dict(beta=posterior.final_beta, particles=len(posterior.samples), **metrics,
                passes=metrics['max_normalized_mean_error'] < .25 and metrics['max_projection_variance_relative_error'] < .30,
                independent_draw_baseline=dict(seed=baseline_seed, **errors(reference)))


def save_stage_diagnostics(problem, run, *, mean=0., scales=.5, alpha_cp=0.):
    """Preserve every recorded stage, including failures; never expose a public result."""
    run = Path(run)
    manifest = json.loads((run/'manifest.json').read_text())
    results = run/manifest['output_dir']
    report = []
    for path in sorted(results.glob('step_*.h5')):
        try:
            record = AltarResultImporter().load(results, manifest=manifest, allow_incomplete=True, stage_file=path.name)
            item = gaussian_stage_errors(problem, record, mean=mean, scales=scales, alpha_cp=alpha_cp)
            item['archive'] = path.name
        except Exception as exc:
            item = dict(archive=path.name, passes=False, error=f'{type(exc).__name__}: {exc}')
        report.append(item)
        (run/'stage_targets.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
