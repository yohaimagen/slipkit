"""Prepare exact problem bundles for Venezuela depth/regularization variants."""
import argparse
import json
from pathlib import Path

import numpy as np

from slipkit.core.bayesian import AltarAssembler, AltarProblem, GaussianBayesianSolver
from slipkit.core.physics import CutdeCpuEngine
from slipkit.core.regularization import DeepEdgeDamping
from venezuela_altar import load_inputs, prior_vectors


def scale_mesh(source, destination, depth_km):
    lines = Path(source).read_text().splitlines()
    start = lines.index("$Nodes") + 2
    stop = lines.index("$EndNodes")
    old_depth = max(-float(lines[i].split()[3]) for i in range(start, stop))
    scale = depth_km / old_depth
    for i in range(start, stop):
        fields = lines[i].split()
        fields[3] = f"{float(fields[3]) * scale:.16g}"
        lines[i] = " ".join(fields)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")


def save_reference(output, problem, fault, datasets, prior_mean, prior_scales,
                   depth_km, constrained, seed):
    if output.exists():
        return
    output.mkdir(parents=True)
    solver = GaussianBayesianSolver(
        prior_mean=prior_mean, prior_scales=prior_scales, draws=2, seed=seed,
    )
    solver.solve_problem(problem)
    solver.save_result(
        output / "inference", problem, faults=[fault], datasets=datasets,
        length_unit="km",
    )
    report = {
        "purpose": "exact analytical reference for a queued native AlTar inversion",
        "fault_depth_km": depth_km,
        "regularization": (
            "Laplacian smoothing plus deep-edge damping alpha=10"
            if constrained else "none; Gaussian parameter prior only"
        ),
        "smoothing_rows": 0 if problem.smoothing is None else len(problem.smoothing),
        "parameters": problem.G.shape[1],
        "observations": len(problem.data),
        "exact_draws_saved": 2,
        "seed": seed,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--depths", type=float, nargs="+", default=(25., 40.))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    source_mesh = args.root / "input/fault.msh"
    bayesian_lambda = .21261123338996568 / .116301
    for depth in args.depths:
        tag = f"depth{depth:g}"
        input_dir = args.root / "variants" / tag / "input"
        mesh = input_dir / "fault.msh"
        if not mesh.exists():
            scale_mesh(source_mesh, mesh, depth)
        data_link = input_dir / "data"
        if not data_link.exists():
            data_link.symlink_to(args.root / "input/data", target_is_directory=True)

        fault, datasets = load_inputs(input_dir)
        constrained = AltarAssembler().assemble_problem(
            [fault], datasets, CutdeCpuEngine(observation_chunk_size=128),
            DeepEdgeDamping(alpha=10., tol=2.), bayesian_lambda,
        )
        unconstrained = AltarProblem(
            constrained.G, constrained.data, constrained.covariance,
            constrained.layout, None,
        )
        prior_mean, prior_scales = prior_vectors(fault, datasets, 5., .5)
        for name, problem, use_constraints in (
            ("constrained", constrained, True),
            ("unconstrained", unconstrained, False),
        ):
            save_reference(
                args.root / "runs" / f"exact-{tag}-{name}", problem, fault,
                datasets, prior_mean, prior_scales, depth, use_constraints,
                args.seed,
            )


if __name__ == "__main__":
    main()
