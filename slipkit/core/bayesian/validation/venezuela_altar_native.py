"""Run a native CUDA AlTar parity sample for the exact Venezuela posterior."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from slipkit.core.bayesian import AltarCudaBayesianSolver, AltarProblem, load_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exact_bundle", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--chains", type=int, default=512)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--scaling", type=float, default=.03)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=7200.)
    parser.add_argument("--label", default="native CUDA AlTar inversion")
    args = parser.parse_args()
    if args.work_dir.exists():
        raise FileExistsError(args.work_dir)
    args.work_dir.mkdir(parents=True)
    started = time.monotonic()

    exact = load_inference(args.exact_bundle)
    problem = exact.problem
    gw, dw, logdet = problem.whiten(exact.metadata["alpha_cp"])
    q, r = np.linalg.qr(gw, mode="reduced")
    projected = q.T @ dw
    residual_constant = float(dw @ dw - projected @ projected)
    if residual_constant < -1e-6:
        raise ValueError("QR residual constant is negative.")
    residual_constant = max(0., residual_constant)
    reduced = AltarProblem(r, projected, np.ones(len(projected)),
                           problem.layout, problem.smoothing)
    rng = np.random.default_rng(90210)
    for model in (exact.posterior.mean, rng.normal(size=problem.G.shape[1])):
        full = np.linalg.norm(gw @ model-dw)**2
        compact = np.linalg.norm(r @ model-projected)**2+residual_constant
        np.testing.assert_allclose(compact, full, rtol=1e-10, atol=1e-3)

    prior = exact.metadata["prior"]
    solver = AltarCudaBayesianSolver(
        prior="gaussian",
        prior_mean=np.asarray(prior["anchor_mean"]),
        prior_scales=np.asarray(prior["scales"]),
        work_dir=str(args.work_dir / "native-runs"),
        chains=args.chains,
        steps=args.steps,
        timeout=args.timeout,
        seed=args.seed,
        initial_scaling=args.scaling,
        sampler="fixed",
        gpu_id=args.gpu_id,
    )
    try:
        solver.solve_problem(reduced)
    except BaseException as exc:
        (args.work_dir / "failure.json").write_text(json.dumps(dict(
            error=repr(exc), run_path=solver.last_run_path,
            elapsed_seconds=time.monotonic()-started,
        ), indent=2)+"\n")
        raise
    posterior = solver.get_last_posterior()
    exact_std = exact.posterior.std
    standardized_mean_error = (posterior.mean-exact.posterior.mean)/exact_std
    rms_error = float(np.sqrt(np.mean(standardized_mean_error**2)))
    report = dict(
        purpose=args.label,
        exact_bundle=str(args.exact_bundle),
        parameters=problem.G.shape[1],
        physical_observations=len(problem.data),
        qr_observations=len(reduced.data),
        qr_residual_constant=residual_constant,
        full_logdet_covariance=logdet,
        chains=args.chains,
        steps=args.steps,
        scaling=args.scaling,
        seed=args.seed,
        gpu_id=args.gpu_id,
        final_beta=float(posterior.final_beta),
        run_path=solver.last_run_path,
        rms_standardized_mean_error=rms_error,
        maximum_absolute_standardized_mean_error=float(np.max(np.abs(standardized_mean_error))),
        elapsed_seconds=time.monotonic()-started,
        adequacy=("native mean agrees with the exact reference within one posterior "
                  "standard deviation RMS" if rms_error <= 1 else
                  "native execution complete; posterior-mean agreement with the exact "
                  "reference needs improvement"),
    )
    solver.save_result(args.work_dir / "inference", reduced)
    (args.work_dir / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
