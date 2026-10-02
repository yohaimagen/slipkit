"""Fit the Mw 7.6 synthetic case and score its withheld observations."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from slipkit.core.bayesian import AltarProblem, AltarBayesianSolver, AltarCudaBayesianSolver


def shared_edge_pairs(faces):
    """Return triangle pairs that share a mesh edge."""
    owner = {}
    pairs = []
    for triangle, face in enumerate(faces):
        for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = tuple(sorted(edge))
            if edge in owner:
                pairs.append((owner[edge], triangle))
            else:
                owner[edge] = triangle
    return np.asarray(pairs, dtype=int)


def score(g, d, h, heldout, slip, truth, areas, rigidity, adjacency):
    moment = rigidity*1e6*np.dot(areas, slip)
    return dict(train_rmse_m=float(np.sqrt(np.mean((g@slip-d)**2))),
                holdout_rmse_m=float(np.sqrt(np.mean((h@slip-heldout)**2))),
                slip_rmse_m=float(np.sqrt(np.mean((slip-truth)**2))),
                shared_edge_rms_m=float(np.sqrt(np.mean(np.diff(slip[adjacency], axis=1)**2))),
                true_shared_edge_rms_m=float(np.sqrt(np.mean(np.diff(truth[adjacency], axis=1)**2))),
                moment_Nm=float(moment), mw=float(2/3*(np.log10(moment)-9.1)),
                minimum_slip_m=float(slip.min()), maximum_slip_m=float(slip.max()))


def reduced_problem(g, d, covariance):
    """QR preserves all slip-dependent Gaussian likelihood terms."""
    sigma = np.sqrt(covariance)
    whitened_g, whitened_d = g/sigma[:, None], d/sigma
    q, r = np.linalg.qr(whitened_g, mode='reduced')
    projected = q.T@whitened_d
    residual_constant = float(whitened_d@whitened_d-projected@projected)
    if residual_constant < 0:
        raise ValueError('QR residual constant must be nonnegative.')
    rng = np.random.default_rng(331)
    for slip in (np.zeros(g.shape[1]), rng.uniform(0., 4., g.shape[1])):
        full = np.linalg.norm(whitened_g@slip-whitened_d)**2
        compact = np.linalg.norm(r@slip-projected)**2+residual_constant
        np.testing.assert_allclose(compact, full, rtol=1e-10, atol=1e-4)
    log_likelihood_offset = -.5*(residual_constant+(len(d)-len(projected))*np.log(2*np.pi)
                                   +np.log(covariance).sum())
    return AltarProblem(r, projected, np.ones(len(projected))), log_likelihood_offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--backend', choices=('map', 'cpu', 'cuda'), required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--chains', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--timeout', type=float, default=7200)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--scaling', type=float, default=.1)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--smoothing', type=float, default=0.,
                        help='MAP-only shared-edge prior precision in 1/metre')
    args = parser.parse_args()
    if args.smoothing and args.backend != 'map':
        parser.error('--smoothing is currently supported only by the MAP baseline')
    args.work_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    metadata = json.loads((args.case/'report.json').read_text())
    g = np.load(args.case/'case/green.npy')
    d = np.load(args.case/'case/data.npy')
    covariance = np.load(args.case/'case/cd.npy')
    h = np.load(args.case/'case/holdout_green.npy')
    heldout = np.load(args.case/'case/holdout_data.npy')
    with np.load(args.case/'synthetic.npz') as synthetic:
        truth = synthetic['slip_m']
        areas = synthetic['areas_km2']
        adjacency = shared_edge_pairs(synthetic['faces'])
        clean = synthetic['displacement_clean_m']
        holdout_mask = synthetic['holdout']
    upper = metadata['prior_upper_m']
    assert g.shape == (metadata['training_observations'], metadata['parameters'])
    assert h.shape == (metadata['heldout_observations'], metadata['parameters'])
    assert np.allclose(covariance, metadata['noise_sigma_m']**2)
    np.testing.assert_allclose(h@truth, clean[holdout_mask], rtol=1e-10, atol=1e-9)
    if args.backend == 'map':
        # Box-constrained MAP with an optional shared-edge Gaussian difference prior.
        sigma2 = float(covariance[0])
        gram, rhs = (g.T@g)/sigma2, (g.T@d)/sigma2
        if args.smoothing:
            laplacian = np.zeros_like(gram)
            np.add.at(laplacian, (adjacency[:, 0], adjacency[:, 0]), 1.)
            np.add.at(laplacian, (adjacency[:, 1], adjacency[:, 1]), 1.)
            np.add.at(laplacian, (adjacency[:, 0], adjacency[:, 1]), -1.)
            np.add.at(laplacian, (adjacency[:, 1], adjacency[:, 0]), -1.)
            gram += args.smoothing**2*laplacian
        scale = float(np.max(np.diag(gram)))
        def objective(slip):
            gradient = gram@slip-rhs
            return (float((slip@gram@slip-2*rhs@slip)/(2*scale)), gradient/scale)
        fit = minimize(objective, np.full(g.shape[1], 1.4), jac=True, method='L-BFGS-B',
                       bounds=[(0, upper)]*g.shape[1], options=dict(maxiter=20000, ftol=1e-14, gtol=1e-4))
        slip = fit.x
        projected = np.where(((slip <= 1e-9) & (fit.jac > 0)) |
                             ((slip >= upper-1e-9) & (fit.jac < 0)), 0., fit.jac)
        result = dict(backend='box-constrained MAP', optimizer_success=bool(fit.success),
                      optimizer_message=str(fit.message), optimizer_iterations=int(fit.nit),
                      projected_gradient_max=float(np.max(np.abs(projected))),
                      smoothing_precision_per_m=args.smoothing,
                      **score(g,d,h,heldout,slip,truth,areas,metadata['rigidity_Pa'],adjacency))
        np.save(args.work_dir/'map_slip.npy', slip)
    else:
        problem, log_likelihood_offset = reduced_problem(g, d, covariance)
        factory = AltarCudaBayesianSolver if args.backend == 'cuda' else AltarBayesianSolver
        options = dict(sampler='fixed', gpu_id=args.gpu_id) if args.backend == 'cuda' else dict(acceptance_weight=0., rejection_weight=args.scaling)
        solver = factory(prior='uniform', work_dir=str(args.work_dir/'native-runs'),
                         chains=args.chains, steps=args.steps, timeout=args.timeout,
                         seed=args.seed, initial_scaling=args.scaling, **options)
        try:
            solver.solve_problem(problem, bounds=(0., upper))
        except BaseException as exc:
            (args.work_dir/f'{args.backend}_failure.json').write_text(json.dumps(dict(error=repr(exc),
                run_path=solver.last_run_path, elapsed_seconds=time.monotonic()-started), indent=2)+'\n')
            raise
        posterior = solver.get_last_posterior()
        slip = posterior.samples.mean(axis=0)
        result = dict(backend=f'native AlTar {args.backend.upper()} uniform posterior', final_beta=float(posterior.final_beta),
                      chains=args.chains, steps=args.steps, seed=args.seed, scaling=args.scaling,
                      run_path=solver.last_run_path, native_pseudo_observations=len(problem.data),
                      physical_training_observations=len(d), full_log_likelihood_offset=log_likelihood_offset,
                      **score(g,d,h,heldout,slip,truth,areas,metadata['rigidity_Pa'],adjacency))
        np.save(args.work_dir/'posterior_mean_slip.npy', slip)
        solver.save_result(args.work_dir/'inference')
    prior_kind = 'bounded shared-edge Gaussian' if args.smoothing else 'uniform'
    result.update(case=str(args.case), prior=dict(kind=prior_kind, lower_m=0., upper_m=upper,
                  shared_edge_precision_per_m=args.smoothing),
                  elapsed_seconds=time.monotonic()-started)
    (args.work_dir/f'{args.backend}_report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
