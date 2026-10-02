# SlipKit–AlTar: published earthquake applications and revised GPU scaling plan

Date: 2026-09-17; updated 2026-09-18 with the available remote GPU host. Scope: literature review and planning; no sampler or bridge implementation changes, and no new GPU benchmark.

## Revised conclusion

The user's objection is supported by the literature. AlTar/CATMIP has handled real earthquake inversions, including an application with 2,000 parameters. The unsuccessful local 144-element experiment does not establish a dimensional limit of AlTar or a need to replace its sampler.

This addendum **supersedes the implementation priority** in the [second-round plan](/Users/ymagen/slipkit/ALTAR_BRIDGE_REVIEW_ROUND_2_AND_SCALING_PLAN.md): validate the existing native CUDA AlTar route before investing in additional custom sampler work or a custom CPU loader. Retain the exact Gaussian solver as a reference for numerical validation and an optional solver when that target is scientifically appropriate.

The previous correctness findings remain valid for the implementations reviewed. Successful published applications neither invalidate a local covariance bug nor validate our particular bridge.

## Published evidence

| Source | Verified evidence | Relevance and limit |
|---|---|---|
| [Minson, Simons & Beck (2013), theory and algorithm](https://academic.oup.com/gji/article/194/3/1701/645931), Section 5 and Figure 12; DOI 10.1093/gji/ggt180 | CATMIP includes a **144-patch synthetic static inversion**, with experiments varying population and chain length. | Direct evidence that 144 patches are not an intrinsic algorithm limit. Our fixture has not been verified as an identical reproduction of their experiment. |
| [Minson et al. (2014), Tohoku finite-fault inversion](https://authors.library.caltech.edu/records/7rycv-9ta47); DOI 10.1093/gji/ggu170 | Real Mw 9 kinematic inversion combining geodetic and tsunami observations. The repository records CATMIP execution on NASA's Pleiades supercomputer. | A real finite-fault application of massively parallel inference. This review verified the repository abstract and computing statement, not every numerical setting in its PDF. |
| [Duputel et al. (2015), Iquique sequence](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015GL065402), Sections 2–3; DOI 10.1002/2015GL065402 | Real Mw 8.1 joint static–kinematic inversion with AlTar: approximately **140,000 chains, 16 billion model evaluations, 24 hours on 90 GPUs**. Also a static AlTar inversion for the Mw 7.7 aftershock. | Particularly relevant to the requested earthquake scale. Uses mixed priors and avoids spatial smoothing. The historical hardware budget is evidence of substantial work, not a hardware requirement for our simpler static problem. |
| [Jiang & Simons (2016), Tohoku seafloor deformation](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016JB013760), Section 2.6; DOI 10.1002/2016JB013760 | Hybrid CPU–GPU AlTar with **2,000 parameters**, approximately **10,000 chains × 1,000 updates per tempering stage**. | Comparable parameter dimension to 1,000 elements with two slip components. Its parameters describe seafloor displacement and propagation speed; it is not the identical 1,000-triangle strike/dip-slip problem. |

These papers establish feasibility of the software family and approach. They do not prove that an arbitrary current build, default configuration, or modified bridge reproduces those results. Study-specific forward models and historical versions also differ from today's documented static component.

Earthquake magnitude, number of elements, parameter count, and sampling difficulty must be kept separate. An Mw 7.6 event does not mathematically require 1,000 independently resolved slip elements. That can be a useful mesh, but resolution and uncertainty must justify the interpretation of its smallest features. With two free components, 1,000 elements means 2,000 parameters before nuisance terms.

## What this changes about the 144-element diagnosis

The current [configuration builder](/Users/ymagen/slipkit/slipkit/core/bayesian/config.py:14) rejects multiple tasks and explicitly emits `job.tasks = 1` and `job.gpus = 0`. Its validated execution contract is serial CPU linear inference. It does not exercise the documented native seismic CUDA backend.

The retained experiment assessed in round two used 2,048 particles and 100 updates per stage. It stopped near beta 0.0247 after about ten minutes; its intermediate distribution disagreed materially with the exact Gaussian target despite approximately 28% acceptance. The saved adapter also differs from the current source. These are observations about that retained run, not a completed benchmark of the present code.

As a workload comparison, 2,048 × 100 is 204,800 proposal attempts per stage, whereas the Jiang–Simons example above is approximately 10 million: roughly 49 times as many. This is not a prescription to multiply our runtime by 49, because the targets and implementations differ. It does show why a brief local run cannot establish failure of the published approach.

The local experiment also has different prior assumptions from the published mixed-prior examples. Reproducing only patch count is not reproducing an inference problem. More updates and an adequate population are plausible remedies to test; available evidence does not isolate a unique cause of the local stage error.

## Will GPUs solve the issue?

**They can address the computational throughput bottleneck and enable substantially more sampling. They do not, by themselves, establish correctness or convergence.** No measured SlipKit GPU speedup is available yet.

| Issue | Expected GPU effect | Required check |
|---|---|---|
| Repeated dense predictions and likelihood calculations across many particles | A suitable native CUDA implementation can accelerate this workload. | Measure end-to-end stage time and posterior accuracy at equal settings. |
| Too few updates to explore correlated directions | Faster updates make a larger budget practical; hardware alone does not change the required exploration. | Increase updates and check intermediate-target errors and independent runs. |
| Inaccurate population covariance or loss of diversity | More affordable particles can help; simply moving the same population to a GPU does not restore missing diversity. | Inspect proposal conditioning, weight concentration and ancestry/diversity. |
| Incorrect covariance handling or parameter ordering | No remedy. An incorrect answer can be produced faster. | Small deterministic prediction and likelihood comparisons. |
| Host-side kernel assembly, I/O, dense covariance copies | Not automatically accelerated by AlTar CUDA. | Profile these separately from sampling and include them in total runtime. |
| Insufficient information to resolve each small fault element | No remedy. | Report uncertainty, correlations and spatially aggregated quantities. |

The documented GPU backend uses **NVIDIA CUDA**. The Apple Silicon GPU in the current Mac is not compatible with that backend. Use the available SSH target `hise_jump` for the pilot, after confirming its GPU and CUDA compatibility as specified below. The installation guide points to CUDA-specific AlTar and Pyre repositories because those extensions are not fully merged into its referenced main branches. Pin a tested pair and its build environment rather than assuming the installed CPU package supports them. [Installation guide](https://altar2.readthedocs.io/en/latest/cuda/Installation.html).

For scale intuition, with 10,000 observations and 2,000 parameters, a float64 Green's matrix occupies 160 MB. A population of 10,000 × 2,000 occupies another 160 MB. A fully materialized 10,000 × 10,000 residual population would be 800 MB; dense observation covariance would be another 800 MB. Proposal factors, additional copies, workspaces and archives add more. These are array-size calculations, not a measured peak-memory estimate or a guarantee that a particular card will fit the run.

There is also a population-size concern independent of hardware: a sample covariance from N particles has rank at most N−1. For 2,000 parameters, retaining just 2,048 particles gives little margin for estimating a full covariance, even before uneven weights and duplicated ancestry. A rank check is a minimum condition, not evidence of an accurate proposal.

## Existing AlTar capabilities to use first

The native `altar.models.seismic.cuda.static` component accepts externally calculated Green's functions and HDF5 inputs. It supports named parameter sets with separate priors, including Gaussian strike slip and uniform dip slip. Parameter ordering must match matrix columns. For unit-noise whitened inputs, the documented scalar `cd_std` option should avoid exporting an identity covariance; verify actual memory behavior in the selected build. The optional moment `prep` controls initialization and must not be described as a posterior moment prior. [Static model documentation](https://altar.readthedocs.io/en/cuda/cuda/Static.html).

The CUDA framework already supplies `altar.cuda.bayesian.adaptivemetropolis`, with configurable minimum/maximum updates and correlation-based stopping. Explicitly supply the actual parameter count for its dimension-dependent proposal scale. Evaluate this existing sampler before designing another adaptation mechanism. Its internal correlation threshold is a tuning aid, not an independent guarantee of convergence. [Framework documentation](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html).

For fixed-geometry static inversion, the simplest integration candidate is therefore SlipKit exporting correctly ordered physical inputs and metadata, native AlTar performing inference, and SlipKit importing results. A thin export/configuration/import change may be necessary; new AlTar sampling algorithms are not currently justified. Coupled spatial or moment priors should only prompt extensions after the scientific target is explicitly chosen and shown to exceed native capabilities.

## Revised implementation and validation sequence

1. **Freeze the target and reproduce a native example.** Record mesh, observations, units, kernel columns, noise, priors and nuisance terms. On `hise_jump`, complete the host checks below, then pin AlTar/Pyre commits, CUDA and numerical libraries. Run the upstream small static example independently of SlipKit. Preserve configuration, logs and outputs.

2. **Prove bridge-to-native numerical equivalence.** Export a tiny SlipKit problem. Compare predictions, prior support and log likelihood at fixed parameter vectors. Test a correlated covariance case. Begin with float64; evaluate float32 only after a reference exists. If whitening is used, apply it exactly once and preserve normalization when comparing densities/evidence. Do not assume the previously diagnosed CPU norm defect is either present or absent in CUDA.

3. **Complete the 144-element Gaussian reference.** Keep the existing Gaussian target unchanged so exact distributions are available at every beta. Compare native fixed-length and native adaptive sampling, with multiple seeds. Retain stage means, selected covariance projections, acceptance, diversity, update counts, runtime and memory. Require beta one and agreement within predeclared Monte Carlo tolerances calibrated with independent reference draws; a timeout or acceptance percentage is not a pass.

4. **Validate the intended mixed-prior problem separately.** If bounded dip slip and Gaussian strike slip match the science, use native parameter sets. Validate a small constrained case against a trusted numerical reference, then increase size. Do not compare a changed prior to the old Gaussian posterior, clip Gaussian draws, or add smoothing merely to make a run finish.

5. **Scale through approximately 400 and 1,000 elements.** Vary observation count as well as parameter count. Treat approximately 10,000 total particles and order-1,000 updates per stage as a literature-informed pilot scale to investigate, not a convergence guarantee or an immediate full-run commitment. Measure early-stage cost first. Check how the build distributes chains and computes population covariance across workers before configuring MPI. Assess uncertainty in total moment, integrated slip and predictions as well as individual coefficients.

6. **Choose hardware and any further code changes from the results.** Compare one-GPU and, where available, multi-GPU execution with the same global population to measure speedup. Separately increase the population to test accuracy. If runtime remains unacceptable, identify whether time is spent in assembly, likelihoods, proposal operations, communication or disk output before optimizing that component. Authorize production-scale runs using measured stage costs and observed temperature progression, not a promised linear extrapolation.

These steps move native CUDA validation ahead of the previous plan's custom CPU-loader and constrained-sampler extensions. The exact Gaussian route remains useful, but is no longer a prerequisite or substitute for investigating the user's preferred native AlTar route.

## GPU implementation validation on `hise_jump`

The user has confirmed access to a GPU machine through the SSH alias **`hise_jump`**. Local SSH configuration resolution on 2026-09-18 identifies hostname `heisenbug`, user `ymagen`, port 22, and proxy jump `LMU`. Connect using `ssh hise_jump` so the configured jump route is retained. Only local configuration resolution has been checked; remote connectivity, hardware, available resources and AlTar installation have not yet been verified.

This is the designated execution host for the native CUDA validation sequence above. Complete these stages in order:

| Stage | Work on the remote host | Evidence required to proceed |
|---|---|---|
| H0: access and resources | Confirm login, whether this is a compute host or a gateway, and whether a scheduler allocation is needed. Inspect GPU model/count, total and available VRAM, driver, CUDA toolkit/compiler, CPU architecture, RAM and free storage. Start with `nvidia-smi` and `nvcc --version`; a driver-reported CUDA version alone does not establish that the toolkit is installed. | Saved hardware/environment inventory and the actual GPU execution location. If GPU access requires an allocation or another node, record that route. |
| H1: identify or build native CUDA AlTar | Inspect existing environments/modules and AlTar/Pyre versions and paths. Reuse a compatible installation; otherwise create an isolated user-owned environment using a pinned CUDA-compatible AlTar/Pyre pair. Check compiler, CUDA libraries, MPI and HDF5 compatibility. | Reproducible build/environment record; native static CUDA model and selected sampler load successfully. |
| H2: demonstrate GPU execution | Run a small upstream static example on one allocated GPU. Inspect the generated configuration, loaded component paths, logs and device activity to establish that native CUDA code executes. | Successful output with recorded device, precision, timings and GPU memory use; successful imports alone are insufficient. |
| H3: validate SlipKit inputs | Transfer a frozen small input bundle with hashes, parameter-layout metadata and configuration. Compare CPU/reference predictions and likelihoods to native CUDA, including correlated-noise handling. | Numerical agreement at specified float64 tolerances, correct parameter order and verified archive import. |
| H4: complete the 144-element gate | Run the Gaussian reference through beta one with at least three independent seeds. Compare native fixed-length and adaptive settings, checking stage distributions against exact targets. | Mean/variance and selected eigenmode diagnostics within calibrated Monte Carlo tolerances; wall time, update counts, peak host/GPU memory and failures retained. |
| H5: scale toward the earthquake model | Increase to approximately 400 and 1,000 elements with realistic observation counts. Test mixed priors separately. If multiple GPUs are available, compare one versus multiple devices at fixed total particle count before increasing the population. | Accuracy and resource report for each size, including full-run cost and the largest validated problem. |

Use a dedicated remote work directory and preserve the exact SlipKit working-tree snapshot: the local repository contains uncommitted changes, so a commit identifier alone cannot reproduce its state. Transfer only the required source and input artifacts. Keep configurations, seeds, source/input hashes, build logs, run logs and posterior archives together; return a compact report and required results to the local project.

The remote report should explicitly answer: **Does the existing native AlTar CUDA implementation run correctly on this host? Does the 144-element target converge accurately? What time and memory does a validated 1,000-element target require?** Report same-host CPU/GPU speedup where a compatible CPU baseline is available. Comparisons with the Mac must be labeled cross-machine comparisons, not pure GPU acceleration measurements. No remote login, installation or GPU run was performed as part of this plan update.

## Access and remaining evidence gaps

The main texts of the 2013, 2015 and 2016 papers were accessible. A paywall did not block the central conclusion. Some author-hosted PDFs and supplementary downloads failed, and the exact historical executable/configurations were not recovered. For a faithful paper reproduction, the most useful additional material would be the original AlTar configuration files, input matrices and supplementary computational settings, especially for Iquique. We have not established that a published study used precisely our triangular discretization and constraints, or measured our 1,000-element case on any GPU.

No implementation files were changed for this addendum. It revises the plan based on primary literature, documentation, the current configuration builder and the experiments already documented in round two.
