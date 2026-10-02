"""Exact Gaussian Bayesian inversion for the six-track Venezuela InSAR case."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from slipkit.core.bayesian import (
    AltarAssembler,
    AltarSlipDistribution,
    GaussianBayesianSolver,
    load_inference,
)
from slipkit.core.data import GeodeticDataSet
from slipkit.core.fault import SlipComponent, StrikeSlipType, TriangularFaultMesh
from slipkit.core.physics import CutdeCpuEngine
from slipkit.core.regularization import DeepEdgeDamping


TRACKS = (
    ("ALOS2_T134D", 0.1459),
    ("ALOS2_T135D", 0.1327),
    ("NSR_T162A", 0.0877),
    ("NSR_T61A", 0.1570),
    ("NSR_T54D", 0.0867),
    ("NSR_T126D", 0.0889),
)


def load_inputs(input_dir):
    """Load the exact quadtree points exported by the deterministic notebook."""
    input_dir = Path(input_dir)
    fault = TriangularFaultMesh(
        str(input_dir / "fault.msh"),
        strike_slip_type=StrikeSlipType.RIGHT_LATERAL,
        slip_components=[SlipComponent.STRIKE_SLIP, SlipComponent.DIP_SLIP],
    )
    datasets = []
    for name, noise_sigma in TRACKS:
        with np.load(input_dir / "data" / f"fit_{name}.npz") as source:
            coords = np.column_stack((source["x_km"], source["y_km"],
                                      np.zeros(len(source["x_km"]))))
            dataset = GeodeticDataSet(
                coords=coords,
                data=source["observed"],
                unit_vecs=source["unit_vecs"],
                sigma=np.full(len(coords), noise_sigma),
                name=name,
                ramp=1,
            )
        datasets.append(dataset)
    return fault, datasets


def prior_vectors(fault, datasets, slip_scale, ramp_scale):
    n_slip = fault.num_patches() * fault.num_components()
    n_ramp = sum(dataset.ramp.num_params for dataset in datasets)
    scales = np.concatenate((np.full(n_slip, slip_scale), np.full(n_ramp, ramp_scale)))
    return np.zeros_like(scales), scales


def fit_metrics(problem, posterior, datasets):
    prediction = problem.G @ posterior.mean
    residual = problem.data - prediction
    rows = []
    start = 0
    for dataset in datasets:
        stop = start + len(dataset)
        r = residual[start:stop]
        rows.append(dict(
            name=dataset.name,
            observations=len(dataset),
            noise_sigma_m=float(dataset.sigma[0]),
            residual_mean_m=float(r.mean()),
            residual_rms_m=float(np.sqrt(np.mean(r**2))),
            normalized_rms=float(np.sqrt(np.mean((r/dataset.sigma)**2))),
            variance_reduction_pct=float(100*(1-r.var()/dataset.data.var())),
        ))
        start = stop
    return prediction, residual, rows


def save_plots(output_dir, fault, result, residual, datasets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    vertices, faces = fault.get_mesh_geometry()
    # The mesh is nearly vertical: local x follows strike, local y captures
    # small trace curvature, and negative z is depth.
    tri = Triangulation(vertices[:, 0], -vertices[:, 2], faces)
    fields = (result.get_component("ss"), result.get_component("ds"), result.total_slip())
    labels = ("Strike slip", "Dip slip (notebook kernel sign)", "Slip magnitude")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    limit = max(np.max(np.abs(field)) for field in fields[:2])
    for ax, field, label in zip(axes, fields, labels):
        if label == "Slip magnitude":
            image = ax.tripcolor(tri, facecolors=field, shading="flat", cmap="viridis",
                                 vmin=0, vmax=np.max(field))
        else:
            image = ax.tripcolor(tri, facecolors=field, shading="flat", cmap="RdBu_r",
                                 vmin=-limit, vmax=limit)
        ax.set(title=label, xlabel="Along-fault local x (km)", ylabel="Depth (km)",
               ylim=(30, 0), aspect="auto")
        fig.colorbar(image, ax=ax, label="Posterior mean slip (m)")
    fig.suptitle("Venezuela six-track exact Gaussian Bayesian posterior")
    fig.savefig(output_dir / "posterior_mean_slip.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    start = 0
    for ax, dataset in zip(axes.ravel(), datasets):
        stop = start + len(dataset)
        values = residual[start:stop]
        vmax = np.percentile(np.abs(values), 98)
        image = ax.scatter(dataset.coords[:, 0], dataset.coords[:, 1], c=values,
                           s=6, cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=True)
        ax.set(title=f"{dataset.name}: posterior-mean residual", xlabel="x (km)",
               ylabel="y (km)", aspect="equal")
        fig.colorbar(image, ax=ax, label="LOS residual (m)")
        start = stop
    fig.savefig(output_dir / "posterior_mean_residuals.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--slip-prior-scale", type=float, default=5.)
    parser.add_argument("--ramp-prior-scale", type=float, default=.5)
    parser.add_argument("--deterministic-lambda", type=float, default=.21261123338996568)
    parser.add_argument("--reference-rms", type=float, default=.116301)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()

    fault, datasets = load_inputs(args.input_dir)
    bayesian_lambda = args.deterministic_lambda / args.reference_rms
    problem = AltarAssembler().assemble_problem(
        [fault], datasets, CutdeCpuEngine(observation_chunk_size=128),
        DeepEdgeDamping(alpha=10., tol=2.), bayesian_lambda,
    )
    prior_mean, prior_scales = prior_vectors(
        fault, datasets, args.slip_prior_scale, args.ramp_prior_scale,
    )
    solver = GaussianBayesianSolver(
        prior_mean=prior_mean,
        prior_scales=prior_scales,
        draws=args.draws,
        seed=args.seed,
    )
    solver.solve_problem(problem)
    posterior = solver.get_last_posterior()
    result = AltarSlipDistribution(posterior, [fault], datasets)
    prediction, residual, fits = fit_metrics(problem, posterior, datasets)

    moment = result.seismic_moment_samples(shear_modulus=3.3e10, length_unit="km")
    mw = 2/3*(np.log10(moment)-9.1)
    n = fault.num_patches()
    ss, ds = posterior.samples[:, :n], posterior.samples[:, n:2*n]
    component_summary = dict(
        strike_slip=dict(
            posterior_mean_min_m=float(result.get_component("ss").min()),
            posterior_mean_max_m=float(result.get_component("ss").max()),
            fraction_mean_negative=float(np.mean(result.get_component("ss") < 0)),
            median_probability_positive=float(np.median(np.mean(ss > 0, axis=0))),
        ),
        dip_slip=dict(
            posterior_mean_min_m=float(result.get_component("ds").min()),
            posterior_mean_max_m=float(result.get_component("ds").max()),
            fraction_mean_negative=float(np.mean(result.get_component("ds") < 0)),
            median_probability_positive=float(np.median(np.mean(ds > 0, axis=0))),
        ),
    )
    ramp_summary = {}
    for dataset in datasets:
        samples = result.nuisance_samples[dataset.name]
        ramp_summary[dataset.name] = dict(
            mean_m=samples.mean(axis=0).tolist(),
            std_m=samples.std(axis=0, ddof=1).tolist(),
            q025_m=np.quantile(samples, .025, axis=0).tolist(),
            q975_m=np.quantile(samples, .975, axis=0).tolist(),
            center_km=dataset.ramp.center.tolist(),
            scale_km=dataset.ramp.scale,
        )

    bundle = args.output_dir / "inference"
    solver.save_result(bundle, problem, faults=[fault], datasets=datasets, length_unit="km")
    restored = load_inference(bundle)
    np.testing.assert_allclose(restored.posterior.mean, posterior.mean, rtol=0, atol=0)
    np.testing.assert_allclose(restored.result.slip_vector, result.slip_vector, rtol=0, atol=0)
    save_plots(args.output_dir, fault, result, residual, datasets)
    np.savez(
        args.output_dir / "posterior_summary.npz",
        posterior_mean=posterior.mean,
        posterior_std=posterior.std,
        prediction=prediction,
        residual=residual,
        moment_samples_Nm=moment,
        mw_samples=mw,
    )
    report = dict(
        event="Venezuela 2026-06-24",
        inference="exact linear Gaussian posterior for the AlTar problem contract",
        native_altar_launched=False,
        reason="The fixed linear Gaussian target is solved exactly; no MCMC approximation is required.",
        fault_patches=fault.num_patches(),
        slip_components=[component.value for component in fault.active_components()],
        observations=len(problem.data),
        parameters=problem.G.shape[1],
        slip_parameters=2*fault.num_patches(),
        ramp_parameters=problem.G.shape[1]-2*fault.num_patches(),
        ramp_degree=1,
        likelihood_noise="independent diagonal per-track empirical residual RMS",
        track_fits=fits,
        prior=dict(
            kind="free-sign Gaussian anchor plus Laplacian and deep-edge Gaussian smoothing",
            mean=0.,
            slip_scale_m=args.slip_prior_scale,
            ramp_coefficient_scale_m=args.ramp_prior_scale,
            deterministic_lambda=args.deterministic_lambda,
            reference_rms_m=args.reference_rms,
            bayesian_smoothing_precision=float(bayesian_lambda),
            deep_edge_alpha=10.,
            deep_edge_tolerance_km=2.,
        ),
        component_summary=component_summary,
        ramps=ramp_summary,
        moment_Nm=dict(mean=float(moment.mean()), std=float(moment.std(ddof=1)),
                       q025=float(np.quantile(moment, .025)), q975=float(np.quantile(moment, .975))),
        mw=dict(mean=float(mw.mean()), std=float(mw.std(ddof=1)),
                q025=float(np.quantile(mw, .025)), q975=float(np.quantile(mw, .975))),
        draws=args.draws,
        seed=args.seed,
        bundle_arrays_sha256=restored.metadata["arrays_sha256"],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-started,
        limitations=[
            "Noise sigmas are plug-in estimates from deterministic residual RMS values.",
            "Spatial atmospheric covariance is not modeled.",
            "The Gaussian slip prior is free-sign and does not enforce the deterministic nonnegative bounds.",
            "The source notebook left dip-slip type unspecified; this run preserves its actual kernel sign convention.",
            "Fault geometry, elastic structure, Poisson ratio, and ramp degree are fixed.",
        ],
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
