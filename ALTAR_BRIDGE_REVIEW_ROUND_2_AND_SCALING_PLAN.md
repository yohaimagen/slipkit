# SlipKit–AlTar: second review and large-model implementation plan

**Date:** 2026-09-17. **Scope:** review and planning; no implementation changes.

**Planning update, 2026-09-18:** The [literature and GPU scaling addendum](/Users/ymagen/slipkit/ALTAR_LITERATURE_AND_GPU_SCALING_ADDENDUM.md#gpu-implementation-validation-on-hise_jump) supersedes the implementation priorities below. Native CUDA AlTar validation now comes first, using the available SSH target `hise_jump` (`heisenbug` through `LMU`). Its remote hardware and installation still need verification; the addendum defines the access, build, numerical-parity, 144-element and 1,000-element validation stages.

This reviews the implementation response in [docs/altar_cpu.md](/Users/ymagen/slipkit/docs/altar_cpu.md:182), its retained experiment reports, and the current working tree after [round one](/Users/ymagen/slipkit/ALTAR_BRIDGE_CODE_REVIEW.md). Git HEAD remains `933e2c45079d6ab3478e2455866f299c99a5b486`; substantial uncommitted changes, including the new Gaussian solver and vectorized CPU model, are included. Existing documents were evaluated as evidence, not treated as instructions.

## 1. Decision

**Accept the main correctness fixes. Do not yet accept large-model AlTar sampling as validated.** The response implements observation whitening correctly, enforces parameter/geometry identity, preserves diagonal and block covariance, exposes proposal controls, and adds an explicitly selected exact Gaussian solver. These are meaningful improvements.

The next work should target two distinct deliverables:

1. **A practical 1,000-element Gaussian inversion**, using the new direct solver, with real mesh/data assembly, structured noise and reproducible saved results.
2. **A scientifically specified constrained AlTar inversion**, with demonstrated mixing at intermediate temperatures and complete 144-element validation before attempting 1,000 elements.

The second route is still needed when the target requires bounds, positivity combined with smoothing, mixed priors or moment/rake constraints. Gaussian draws must not be clipped to simulate those constraints. Since no constraint choice was supplied during this review, the plan preserves both routes and treats the exact target definition as the first implementation decision.

## 2. Disposition of the first review

| First-round finding | Current implementation | Second-round assessment |
|---|---|---|
| Native correlated-noise norm uses the wrong triangular orientation | `problem.whiten()` transforms observations and kernels; native runs receive identity covariance; importer restores normalization offsets | **Resolved for bridge-mediated runs.** The installed native raw-covariance defect still exists, but this bridge avoids it. Native numerical parity and physical density tests pass. |
| Proposal scaling oscillates | Explicit initial scale and adaptation weights; fixed-scale experiments; saved progress | **Controls implemented; mixing remains unresolved.** Fixed-scale retained runs remove the alternating near-zero acceptance pattern, but that is not a posterior accuracy result. |
| Legacy adapter silently removes small covariance entries | Exact diagonal representation and explicit rejection of correlated packed inputs | **Resolved in the legacy assembler.** A separate exporter regression remains below. |
| Geometry attachment ignores semantic order | Canonical layout includes components, geometry hashes, signs and ramp identity/basis metadata | **Resolved for tested attachments.** Deliberate mismatches now fail. |
| Dense physical covariance and transform storage | Diagonal vectors, covariance blocks, cached noise factors, vector affine scales and binary dense transforms | **Partially resolved.** Native loading still allocates a dense identity matrix; kernel construction and several copies remain large. |
| Gaussian problems unnecessarily require MCMC | `GaussianBayesianSolver` shares physical result handling, computes a precision factor and draws independent samples | **Implemented and numerically verified.** Persistence and real-model scale validation remain. |

The documented static CUDA route is still a separate backend contract: named parameter sets, model/launcher, sampler, input formats and deployment need their own adapter and numerical verification. The CPU vectorized model does not provide CUDA compatibility by itself. [AlTar static documentation](https://altar.readthedocs.io/en/cuda/cuda/Static.html).

## 3. Remaining findings

### R2-1 — P2: diagonal covariance export with nonzero Cp now raises an exception

Location: [exporter.py:39](/Users/ymagen/slipkit/slipkit/core/bayesian/exporter.py:39).

`validate_covariance` now compresses a diagonal matrix to a variance vector. `export_covariance` still adds a two-dimensional diagonal Cp matrix in-place. Therefore both vector and diagonal-matrix inputs fail when `alpha_cp > 0`.

Reproduction:

```python
AltarDataExporter(output_dir).export_covariance(
    covariance=np.array([0.01, 0.04]),
    d_obs=np.array([1.0, 2.0]), alpha_cp=0.1)
# ValueError: non-broadcastable output operand with shape (2,)
# doesn't match the broadcast shape (2,2)
```

The expected exported covariance is `diag([0.02, 0.08])`. Supplying `diag([0.01, 0.04])` instead also fails. The same defect is reachable through `export_all` with diagonal noise. The main solver uses `problem.whiten()` and is unaffected; this is a public exporter regression, not evidence that the corrected inference route is wrong.

**Fix:** update the variance vector directly, or update only matrix diagonal indices, as `problem.whiten()` already does. Add a regression for both representations and verify the diagonal entries numerically.

### R2-2 — P1 for the scale objective: an identity covariance still costs quadratic observation storage

Locations: [exporter.py:46](/Users/ymagen/slipkit/slipkit/core/bayesian/exporter.py:46), [native_cpu.py:14](/Users/ymagen/slipkit/slipkit/core/bayesian/native_cpu.py:14).

Whitening fixes likelihood correctness, but `export_whitened` still creates and writes `np.eye(n)`. The vectorized model invokes the parent loader, which allocates/loads the full matrix before the overridden covariance method verifies it is identity. Batching particles avoids the full residual population, but does not remove this matrix.

At 10,000 observations the identity alone is 800 MB in float64. With the current text formatting, its file is approximately 2.5 GB. At 50,000 observations these become 20 GB and approximately 62.5 GB, before parser overhead or other arrays. It contains no information worth storing.

**Fix:** give the already bridge-owned `VectorizedLinear` a minimal verified loader for binary `G`/`d` and an explicit identity-noise contract. Bypass covariance allocation altogether. Keep the original native loader available as a reference backend. Do not add a broad storage framework; one array format, validated shapes/finite values, provenance and a parity test suffice. NumPy binary arrays can use the existing stack; a future CUDA adapter may use its documented HDF5 support.

### R2-3 — High-priority validation gap: acceptable Metropolis acceptance has not established equilibrium

The retained 144-element seed-17 run used 288 parameters, 2,048 particles, 100 updates per stage and fixed scale `0.16`. It stopped after about 600 seconds, at stage 31 and beta `0.024674370255382935`. Its final recorded acceptance was `28.46%`.

Because this is a Gaussian problem, the exact distribution at that intermediate beta is available:

```text
Q_beta = Q_prior + beta * Gᵀ C⁻¹ G
h_beta = D⁻² m + beta * Gᵀ C⁻¹ d
mean_beta = solve(Q_beta, h_beta)
```

I compared the saved physical particles with **this intermediate target**, not with the final posterior. The recorded prior is independent zero-mean Gaussian with scale 0.5 m, and Cp is zero.

| Diagnostic | Saved stage-31 particles | One independent Gaussian population of the same size |
|---|---:|---:|
| Maximum coordinate mean error / target standard deviation | 1.0191 | 0.0695 |
| Maximum relative projection variance error | 0.4413 | 0.1082 |

Projections comprised coordinate axes and the three smallest/largest covariance eigenmodes. The comparison population used seed 111. One comparison is not a calibrated significance test, but the discrepancy is enough to reject acceptance rate alone as an adequacy criterion. Population bias, dependence and finite-stage equilibration all need attention; these measurements do not isolate a unique cause.

**Version caveat:** that run's adapter checksum differs from the current source, consistent with the response explaining that its experiments preceded the last batched evaluator change. This diagnoses the retained experiment; it is not a claim that the current code has been run to the same state or has identical timing.

**Required change to validation:** for Gaussian experiments, save stage-wise errors against the exact tempered target. Tune updates and proposal scale using those errors and particle diversity, not acceptance alone. Then rerun a frozen current revision. Preserve failed stages as diagnostics, never as final posterior output.

### R2-4 — P2 for reproducible science: the exact solver has no durable inference artifact

Location: [gaussian.py:24](/Users/ymagen/slipkit/slipkit/core/bayesian/gaussian.py:24).

The Gaussian solver returns a mean, precision factor and particles in memory. Unlike AlTar runs, it writes no input/layout/prior manifest or result artifact; `last_run_path` remains `None`. A benchmark JSON containing summary errors cannot reconstruct a real earthquake posterior. The physical problem's layout alone also does not contain the fault mesh needed to rebuild a geometry-aware result.

**Improve before production:** add a small explicit save/load facility shared by both result routes. Store numeric inputs or immutable references with hashes, geometry/ramp metadata, effective-noise policy, prior anchor/scales, smoothing definition, seed, exact mean and precision factor. Particles may be stored or reproducibly regenerated with the numerical environment recorded. Verify save/reload predictions and component/moment summaries. Keep an in-memory mode for experiments; do not force AlTar subprocess machinery into the Gaussian path.

### R2-5 — P2 for scale readiness: current measurements omit expensive production stages

Locations: [physics.py:106](/Users/ymagen/slipkit/slipkit/core/physics.py:106), [problem.py:131](/Users/ymagen/slipkit/slipkit/core/bayesian/problem.py:131), [benchmark.py:34](/Users/ymagen/slipkit/slipkit/core/bayesian/benchmark.py:34).

The synthetic checks start from generated matrices; they do not measure real Cutde assembly. `CutdeCpuEngine.build_kernel` materializes nine displacement/slip responses per observation/triangle before projection. For 10,000 observations and 1,000 triangles, that is about **720 MB**, versus **160 MB** for the final two-component kernel. Projection temporaries and assembler copies add more. At 50,000 observations, the nine-response array alone is about 3.6 GB.

`whiten` also builds lists of transformed blocks and then stacks them, introducing further complete kernel copies. Benchmark timing starts after input loading, exact covariance construction and eigendecomposition; its elapsed field is not total job time. Current tests are useful, but these exclusions must remain visible.

**Improve:** build/project the Green matrix in observation chunks, preallocate its final destination, and benchmark the whole path. Retain covariance blocks only where cross-block independence is scientifically justified. Correlated whitening within a dense block requires its full factor; arbitrary chunk boundaries must not discard cross-observation covariance.

## 4. What can already scale: a fresh numerical experiment

I ran a new synthetic Gaussian check with **10,000 observations, 2,000 parameters and 4,096 independent posterior draws**, using the current solver and requested BLAS thread limits of one. The matrix was a reproducible random design scaled by `1/sqrt(n)`; noise was diagonal unit variance and the prior standard deviation was 0.5.

| Quantity | Measured result |
|---|---:|
| Matrix/data generation plus physical problem construction | 0.146 s |
| Exact solve and posterior draws | 1.322 s |
| Relative normal-equation residual, independently recomputed | 7.54e-16 |
| Relative variance error in one preselected random projection | 0.0003855 |
| Process peak resident memory | 984,203,264 bytes, approximately 984 MB |

This single local experiment establishes that **2,000 parameters are not intrinsically too large for the current Gaussian algebra**. It is not a runtime promise, a Cutde benchmark, a realistic spatial-covariance test, or validation of a constrained posterior. The design is better conditioned than many geodetic problems. It extends the response's 351-observation synthetic checks to a larger observation count, while keeping the interpretation narrow.

For roughly 1,000 fault elements, two active components imply about 2,000 parameters plus nuisance terms. If Gaussian assumptions are scientifically acceptable, first complete the real assembly and persistence work; a GPU is not a prerequisite demonstrated by these measurements.

## 5. Define the scientific target before choosing the large-model sampler

Write the intended target explicitly. For a bounded, smoothed Gaussian slip model, a useful form is:

```text
p(x | d) ∝ 1{lower <= x <= upper}
           exp[-0.5 * (data_misfit(x) + anchor_penalty(x) + smoothing_penalty(x))]

data_misfit    = (Gx-d)ᵀ C⁻¹ (Gx-d)
anchor_penalty = ||D⁻¹(x-m)||²
smoothing_penalty = ||Sx||²
```

The current Gaussian solver handles the case without finite bounds. The current uniform solver handles an independent finite box **without** those Gaussian prior penalties. Neither currently handles the full expression above. “Uniform bounds” and “bounded Gaussian smoothing” are different models.

Specify which SS/DS components can have either sign, physical bounds, ramp priors, noise blocks, smoothing length/strength, and whether rake/moment information is a prior or only a reported posterior quantity. A magnitude near Mw 7.6 does not determine parameter count or justify the fixture's 0.5 m prior. Define mesh refinement tests that preserve comparable physical prior behavior; fixed coefficients on a topological Laplacian do not automatically do so.

### A minimal AlTar extension if finite bounds plus smoothing are required

There is a route that can reuse the existing native uniform initializer and most of the new vectorized model, rather than creating a general prior framework. Use a finite uniform box as the **annealing base distribution**, then temper data and Gaussian penalty factors together:

```text
pi_beta(x) ∝ UniformBox(x)
             * exp[-0.5 * beta * (data_misfit + anchor_penalty + smoothing_penalty)]
```

At beta zero, initialization is exactly the native box distribution. At beta one, the target is the bounded smoothed posterior above. This is a mathematical proposal for a new capability, not a feature already implemented or claimed by the documentation. It changes the annealing path, not the final target.

Implement the penalty energy separately and label physical likelihood, prior factors and annealing potential distinctly. The bounded correlated prior's normalizing constant is generally unavailable: do not reuse the current importer to claim fully normalized physical prior/posterior densities or evidence. Dense smoothing transforms also turn box bounds into coupled constraints, so preserve a clearly documented coordinate system. Verify a small coupled bounded problem against numerical integration before trusting larger examples. This route is suitable only where the chosen base box is finite; Gaussian/unbounded ramp components need explicit treatment rather than invented finite bounds.

If the uniform base is far from plausible slip fields, this path may still require many annealing stages. Exact initialization does not guarantee efficient mixing. Evaluate that experimentally before investing in more backends. The AlTar framework's annealed prior-times-likelihood construction supports the mathematical reasoning, but the new penalty accounting and validation are SlipKit work. [AlTar framework documentation](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html).

## 6. Scaling architecture: implement the smallest useful changes in order

### A. Gaussian production route

1. Fix R2-1 and add durable results (R2-4).
2. Chunk Cutde construction over observations and write projected active-component columns into a preallocated kernel. Verify chunked versus existing kernels, signs and predictions on several meshes.
3. Avoid unnecessary full-kernel copies during assembly and whitening. For diagonal or independently factored covariance blocks, accumulate `H = sum(Gw_blockᵀ Gw_block)` and `g = sum(Gw_blockᵀ dw_block)` when a streaming solve is needed. Add the Gaussian prior precision once, not once per block.
4. Retain sparse `S` until accumulating its contribution. Factor the resulting dense parameter precision for the initial 2,000-parameter target. This is a reasonable starting point; a matrix-free solver framework is premature until measured memory/time or parameter count demands it.
5. Draw particles in batches when necessary; allow exact linear summaries without generating particles. Reuse the precision factor. Computing every marginal standard deviation currently solves against the full identity, so request/cache only the summaries needed for large repeated workflows.
6. Cross-check representative difficult problems against augmented least-squares/QR algebra. Forming normal equations can lose accuracy in ill-conditioned weakly regularized cases; a small normal-equation residual alone does not prove an accurate covariance.

Keep all observations and covariance correlations represented correctly. Statistical downsampling, covariance approximation and a coarser slip basis are scientific approximations and need separate error assessment.

### B. AlTar computation and mixing route

1. Remove identity covariance export/loading (R2-2) from the bridge-owned model and verify likelihood parity, including normalization.
2. Freeze the evaluator revision and profile proposal generation, covariance estimation/conditioning, acceptance bookkeeping, likelihood, recording and preparation separately. The vectorized prior optimization does not accelerate these other native operations.
3. Complete small and 144-element Gaussian reference experiments with stage-target diagnostics. Increase updates per stage or change proposals when measured equilibration is poor; use stable acceptance only as one diagnostic. Population conditioning cannot manufacture information when particles occupy too few directions.
4. Implement only the scientifically required constrained target and validate it independently. Do not use exact Gaussian success as evidence that constraints work.
5. For fixed linear noise with many more observations than parameters, consider exact observation compression before hardware changes. A thin QR of the whitened design gives `Gw = Q R`, `y = Qᵀ dw`, and `||Gw x-dw||² = ||R x-y||² + k`, where `k` is the orthogonal residual norm squared. Preserve `k`, the original observation count and determinant normalization when reporting physical likelihoods. Account for rank deficiency and compare residual formulations to avoid cancellation. This reduces repeated observation-space work without reducing the slip parameter dimension. It is less useful for the 351-by-288 fixture than for a 10,000-by-2,000 problem.
6. If the validated CPU route misses the chosen runtime budget, move the measured bottleneck to a verified CUDA/MPI backend. Native whole-population proposals and covariance adaptation still scale strongly with parameter count. A faster likelihood alone cannot solve slow mixing.

For a CUDA adapter, start with float64 parity on small targets and the 144-element reference, then evaluate float32 deliberately. Verify parameter-set order, archive layout and per-worker particle counts. Do not assume a CUDA setting accelerates the current Mac CPU deployment. One verified accelerated backend is preferable to simultaneous MPI, CUDA and custom-sampler projects.

### C. Memory budget to track

Illustrative float64 arrays for 1,000 triangles, both components, 10,000 scalar observations and 4,096 particles:

| Array/workspace | Size | Action |
|---|---:|---|
| Projected `G` | 160 MB | Keep or stream, depending on workflow |
| Cutde nine-response temporary | 720 MB | Build in observation chunks |
| Physical diagonal variance vector | 0.08 MB | Preserve vector representation |
| Unnecessary identity covariance | 800 MB | Eliminate from vectorized native loading |
| One 2,000-by-2,000 precision/covariance/factor | 32 MB | Budget the simultaneous factors/copies |
| One particle population | 65.5 MB | Candidates and originals are separate |
| Current 512-particle residual batch | 41 MB | Bound with a documented batch budget |

These are individual arrays, not peak process memory. At 50,000 observations `G` is 800 MB and the Cutde temporary is 3.6 GB. A full correlated physical covariance itself would be 20 GB: removing the *identity* file does not solve a genuinely dense correlated-noise model. Preserve exact independent blocks where appropriate; introduce sparse/low-rank/operator noise only with covariance and likelihood validation.

## 7. Concrete implementation milestones and stopping rules

| Milestone | Primary files/responsibility | Required result before proceeding |
|---|---|---|
| M0: target and reproducibility | Scientific specification; frozen code/environment; saved fixtures | Explicit components, bounds/prior factors, ramps, noise and smoothing semantics; declared runtime/memory budget |
| M1: close regressions and persist exact inference | `exporter.py`, result/manifest save/load | Diagonal Cp export passes; Gaussian result reload reproduces predictions, ordering and summaries |
| M2: real 1,000-element Gaussian case | `physics.py`, `assembler.py`, `problem.py`, `gaussian.py` | Chunked Green assembly parity, exact Gaussian checks, full-path timings and memory on real geometry; credible uncertainty summaries |
| M3: scalable vectorized input | `native_cpu.py`, `exporter.py`, `solver.py` | No quadratic identity array/file; deterministic likelihood and prior parity; tiny full runs pass |
| M4: complete 144-element AlTar gate | `benchmark.py`, progress/diagnostic handling, sampler configuration | Three independent seeds reach beta one and pass mean/variance/eigenmode checks; stage errors and mixing are documented |
| M5: required constraints | Explicit target factors and initialization in the selected backend | Tiny coupled bounded reference agrees; physical likelihood and prior-factor reporting remain correct |
| M6: increase mesh/data scale | 144 → approximately 400 → 1,000 elements; observation-count ladder | Each level meets accuracy, memory and runtime budgets before the next; numerical and scientific approximation errors are distinguished |
| M7: acceleration if needed | One selected CUDA or MPI adapter | Numerical equivalence, preserved target, measured full-run speedup and successful large-model validation |

**Default experiment policy:** conduct one controlled performance experiment at a time; record hardware, BLAS thread settings, source hashes, dimensions and covariance type. Use separate subprocesses for memory measurements so high-water values do not accumulate across experiments. Keep setup, assembly, factorization, sampling and validation times separate, plus total elapsed time. Do not compare the response's overlapping runs as controlled speedup measurements.

For Gaussian sample checks retain the existing initial gates of maximum standardized mean error `<0.25` and projection variance error `<0.30`, with independent draw baselines at each dimension. Include weak/strong modes, spatial contrasts and scientifically important moment/prediction summaries. For deterministic Gaussian means/covariances use substantially tighter numerical tolerances against independent references.

For constrained targets these Gaussian gates are not a reference distribution. Use low-dimensional integration or another independently validated method on small cases, then compare repeated complete runs and posterior predictive behavior at larger sizes. Do not compute ordinary chain diagnostics by pretending terminal particle rows are successive MCMC draws.

Stop and revise the experiment when stage distributions remain materially biased, runtime/memory exceeds the predeclared budget, or an approximation changes predictions/uncertainties beyond tolerance. Do not automatically double particles, lower noise, remove bounds or weaken smoothing just to finish. A coarse slip basis can help, but it changes the represented posterior; quantify forward approximation and omitted uncertainty before adopting it.

## 8. Verification and remaining limits

Fresh checks in this round:

- **68 bridge tests passed with native AlTar enabled**, including all 14 optional native checks, in 53.71 seconds.
- **151 other repository tests passed** in a separate run. The pre-existing visualization test importing removed `slipkit.utils.viz` remains excluded. Combined: **219 passing tests** across the two runs.
- Reproduced the diagonal/vector Cp exporter exception without changing implementation code.
- Recomputed the exact tempered Gaussian target for the saved stage-31 particles, and compared against an independent same-size Gaussian population.
- Ran the current 2,000-parameter/10,000-observation synthetic Gaussian check reported above.
- Inspected retained 9-element, 144-element, proposal-sweep and exact-Gaussian reports. The newer 9-element vectorized attempts also timed out under their short 120-second budgets; they do not replace completed reference benchmarks.
- Checked the retained large-run adapter checksum against current code and confirmed they differ. No new full 144-element AlTar run or real 1,000-element earthquake inversion was completed in this review.

Audit measurements and source hashes are retained under `/tmp/slipkit-round2/` (`evidence.json`, `scale.json`, `scale_probe.py`, `source_hashes.json`). Temporary files may be removed by the operating system; the important measurements and conclusions are included here.

The recommended next implementation is **M1–M3**, together with the scientific target decision. The exact Gaussian route already has promising numerical capacity at the requested parameter count. The constrained AlTar route needs a mixing and backend validation milestone, not another blanket claim that raising timeout makes it scale.
