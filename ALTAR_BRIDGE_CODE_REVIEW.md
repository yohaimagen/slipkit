# SlipKit–AlTar bridge: code review and route to 1,000 fault elements

Review date: 2026-09-17. Repository: `/Users/ymagen/slipkit`. Reviewed HEAD: `933e2c45079d6ab3478e2455866f299c99a5b486`, **including the substantial uncommitted bridge implementation**. HEAD alone does not identify the reviewed code. This review changes no implementation files.

## Assessment

The bridge is substantially improved and implements most of the CPU-first compatibility plan. It can now return geometry-aware results, preserve active-component ordering, include ramps and Gaussian smoothing, enforce explicit prior choices, isolate runs, and reject incomplete output. The earlier empty-fault importer failure is fixed.

**It is not yet ready for production correlated-noise inversions or the proposed 1,000-element sampling workload.** There is a confirmed native likelihood error for correlated covariance, two reproducible bridge correctness gaps, and a measured sampler-scaling problem. The 144-element failure is a one-hour timeout during annealing, aggravated by oscillating proposal sizes and expensive CPU operations. It is not a 144-element limit or the old importer exception.

For fixed geometry, fixed covariance and Gaussian priors/smoothing, use an exact Gaussian posterior as a practical inference route as well as the AlTar validation reference. For bounded or otherwise non-Gaussian scientific models, continue AlTar development with explicit numerical and performance gates.

## 1. Prioritized findings

### P1 — Full covariance is exported to a native likelihood that evaluates the wrong quadratic form

Bridge location: [solver.py:154](/Users/ymagen/slipkit/slipkit/core/bayesian/solver.py:154), where the full covariance is passed unchanged to CPU `linear`. Native locations: [Linear.py:300](/Users/ymagen/miniconda3/envs/slipkit/lib/python3.12/site-packages/altar/models/linear/Linear.py:300) and [L2.py:43](/Users/ymagen/miniconda3/envs/slipkit/lib/python3.12/site-packages/altar/norms/L2.py:43).

The installed model computes a lower Cholesky factor `L` of `C⁻¹`, so `C⁻¹ = L Lᵀ`. Its norm then computes `||L r||²`. The required Gaussian term is `rᵀ C⁻¹ r = ||Lᵀ r||²`. They agree for diagonal covariance but generally disagree for correlated noise.

**Verified directly against the installed native routines:**

```text
C = [[1.0, 0.8], [0.8, 2.0]]
r = [1.0, 2.0]
Required rᵀ C⁻¹ r:      2.058823529411765
Installed native norm²:  2.333893671801105
```

Consequently, an apparently successful correlated-noise inversion samples the wrong posterior. Shape, symmetry and positive-definiteness validation cannot detect this.

The existing native full-covariance integration test passes despite this error. For its own fixture, the wrong target differs from the correct target by only `0.05387` posterior standard deviations in the mean and `0.00809` in relative standard deviation, below the test's `0.2` thresholds. I reran that test and it passed. A passing stochastic test is therefore insufficient evidence of likelihood correctness.

**Required fix:** correct the native triangular transpose, validate it and update the backend contract. A bridge-only alternative is exact observation whitening: factor `C = Lc Lcᵀ`, export `Lc⁻¹ G` and `Lc⁻¹ d`, and send identity covariance. Apply whitening once, including the fixed Cp contribution, and record the representation. If reporting physical log likelihoods, restore the constant normalization offset introduced by whitening. This alternative fixes correctness but does not by itself remove the native dense identity-matrix operations.

Add deterministic native likelihood comparisons against NumPy/SciPy for unequal variances, positive and negative correlations, several residual directions and scaled covariances. Compare log likelihoods including normalization, not just posterior summaries. Temporarily reject correlated covariance if neither correction is delivered. The current source-hash check includes `Linear`, distributions and `Recorder`, but excludes `L2`; broaden provenance to the numerical dependency actually responsible for this error and make numerical checks the release gate.

### P1 for the scaling objective — Default proposal adaptation repeatedly makes almost every proposal fail

Bridge location: [config.py:34](/Users/ymagen/slipkit/slipkit/core/bayesian/config.py:34). Native implementation: [Metropolis.py:310](/Users/ymagen/miniconda3/envs/slipkit/lib/python3.12/site-packages/altar/bayesian/Metropolis.py:310).

The bridge exposes particle count and step count but leaves native proposal adaptation at its defaults. The installed update is approximately:

```text
next_scaling = (8/9) * current_acceptance_rate + 1/9
```

In the retained 288-parameter run it alternates between modest proposals with roughly 40–48% acceptance and much larger proposals with under 0.2% acceptance. This is observed behavior, not just a concern about high-dimensional MCMC.

| Completed stage | Accepted / attempted | Acceptance | Scaling recorded after adaptation, for the next stage |
|---|---:|---:|---:|
| 1 | 246,992 / 614,400 | 40.20% | 0.46845 |
| 2 | 355 / 614,400 | 0.0578% | 0.11162 |
| 19 | 292,790 / 614,400 | 47.65% | 0.53471 |
| 20 | 483 / 614,400 | 0.0786% | 0.11181 |

**Required improvement:** expose and record a small set of validated CPU sampler controls: initial scaling, acceptance weight and rejection weight. Test a fixed-scale baseline, achievable in this native implementation with acceptance weight zero and rejection weight equal to the desired scale. Set the initial scale consistently. Compare several scales on the exact Gaussian target before choosing a default. Do not silently alter the native installation or assume that the documented CUDA adaptive sampler is available in this CPU build.

A dimension-based initial scale is a useful experiment, not a convergence guarantee. Increasing particles, steps or timeout while retaining this oscillation spends more time without addressing the underlying wasted proposals.

### P2 — The legacy packed adapter silently drops small but significant correlations

Location: [assembler.py:99](/Users/ymagen/slipkit/slipkit/core/bayesian/assembler.py:99).

`np.allclose(C, diag(diag(C)))` uses an absolute tolerance of `1e-8`. In covariance units of metres squared, this is not a safe definition of independent errors. A reproduced case with diagonal variance `1e-8` and off-diagonal covariance `5e-9` has correlation `0.5`, yet the legacy adapter accepts it and returns only standard deviations of `0.0001` m. Correlations disappear silently.

**Fix:** require exactly zero off-diagonal entries for this explicitly diagonal-only adapter, or use a documented dimensionless correlation tolerance. Prefer rejecting any nonzero off-diagonal entries. Add a regression using small physical variances. This affects the public legacy `assemble` route; the newer orchestrator's `assemble_problem` route preserves the matrix.

### P2 — Geometry attachment checks width but ignores declared parameter semantics

Location: [results.py:57](/Users/ymagen/slipkit/slipkit/core/bayesian/results.py:57), with only contiguous-width validation in [problem.py:20](/Users/ymagen/slipkit/slipkit/core/bayesian/problem.py:20).

`AltarSlipDistribution` derives ordering from the supplied faults and datasets but does not compare that ordering with `posterior.layout`. I constructed a valid-width posterior declaring `[DS, SS]` and attached an ordinary `[SS, DS]` fault: construction succeeded, and `ss_samples` returned the declared DS column. Reattaching a reimported posterior to different same-width faults or reordered ramp datasets can silently mislabel slip and change predictions or moment estimates.

**Fix:** derive the expected semantic layout and compare fault index, component, block width, dataset identity and ramp order before geometry attachment. Reject mismatches or perform an explicit validated permutation. For durable reimport, also store geometry identifiers/hashes and ramp basis metadata, including normalization origin and scale. Width alone cannot establish that supplied geometry is the geometry used to construct `G`. Keep the geometry-free matrix interface available deliberately.

### P2 for large problems — Dense representations and repeated factorizations impose avoidable costs

Locations: [assembler.py:84](/Users/ymagen/slipkit/slipkit/core/bayesian/assembler.py:84), [problem.py:7](/Users/ymagen/slipkit/slipkit/core/bayesian/problem.py:7), [solver.py:97](/Users/ymagen/slipkit/slipkit/core/bayesian/solver.py:97), [solver.py:144](/Users/ymagen/slipkit/slipkit/core/bayesian/solver.py:144), and [exporter.py:49](/Users/ymagen/slipkit/slipkit/core/bayesian/exporter.py:49).

Independent noise becomes a dense matrix; repeated problem construction and export repeat covariance Cholesky checks; independent priors become dense diagonal matrices followed by dense factorization; the affine transform becomes nested Python lists and JSON. Native CPU `linear` additionally inverts/factors covariance and applies a dense triangular matrix to every residual, even when covariance is diagonal.

These are not the primary memory issue at 351 observations, but they obstruct realistic large InSAR runs. Preserve diagonal noise as a vector, retain block structure where available, validate/factor a frozen input snapshot once per solve, and represent independent affine transforms by vectors. Use binary arrays for genuinely dense transforms with small manifest references and checksums. Changing only the file format cannot repair CPU compute scaling; the pinned CPU loader also needs verified support before replacing its text inputs.

## 2. Why the current 144-element test did not work

The concrete failed run is `/tmp/slipkit-144-benchmark/run-rsw60rdy`. Its manifest records a failed timeout/interruption, and the accompanying benchmark documentation identifies the configured one-hour timeout. Its saved configuration has:

| Quantity | Value |
|---|---:|
| Fault elements | 144 |
| Parameters | 288: SS and DS per element |
| Scalar observations | 351 |
| Particles | 2,048 |
| Metropolis updates per annealing stage | 300 |
| Seed | 17 |
| Last saved stage | 20 |
| Last saved beta | 0.007934392775245408 |
| Final posterior archive | Absent |

The importer correctly refuses to present these intermediate samples as a posterior. At this beta the target is approximately `prior × likelihood^0.007934`, not the requested `prior × likelihood`. Beta is **not a percentage-of-runtime progress bar**; dividing one hour by beta would not give a justified runtime estimate.

There are three separable contributors:

1. **Proposal oscillation**, demonstrated above, leaves alternate stages nearly stationary.
2. **High cost per update.** Each stage attempts `2,048 × 300 = 614,400` particle moves, with dense numerical work and Python loops.
3. **A changing, anisotropic target.** Using the saved data and the run's 0.5 m independent Gaussian prior, eigenvalues of the prior-whitened information matrix span approximately `1.85e-9` to `429.62`. The posterior precision condition number in these coordinates is approximately `430.62`. Prior standardization does not make this posterior isotropic. This informs tuning; it does not prove that the problem is numerically unsolvable.

The covariance in this specific run is diagonal. Therefore the correlated-covariance error in finding 1 is independent of this timeout.

### Measured CPU costs

I timed installed native methods using the saved problem dimensions and synthetic particle values. Each method was warmed up, then measured three times. These are isolated operation measurements, not a complete run profile or accuracy test.

| Native operation, one population update | Median elapsed time |
|---|---:|
| Gaussian prior log likelihood | 0.1197 s |
| Data log likelihood | 0.0525 s |
| Random proposal generation and dense transformation | 0.0631 s |
| Sum × 300 updates | Approximately 70.6 s per stage |

The prior evaluator calls a density and logarithm in nested Python iteration over particles and parameters. The data evaluator performs matrix multiplication and a separate covariance-weighted residual norm for every particle. Proposal generation applies a dense parameter-space factor. The summed microbenchmark excludes acceptance bookkeeping, stage covariance estimation, resampling, archiving and other overhead. Later saved log intervals were roughly 96–110 seconds per stage. Earlier intervals were much slower; machine contention and other overhead cannot be separated from those logs.

Simply extending the timeout may eventually finish this experiment, but it would not establish adequate mixing or validate 1,000-element performance. The older historical 144-element output described in the original plan is a different run: reaching beta one there did not establish posterior accuracy. Do not conflate that historical statistical failure with this new timeout.

## 3. What the implementation gets right, and what the documentation changes

The reviewed implementation satisfies important parts of the [compatibility plan](/Users/ymagen/slipkit/ALTAR_COMPATIBILITY_ANALYSIS_AND_PLAN.md): actual parameter counts; fault-major active components and ramps; raw measurement covariance; proper Gaussian smoothing through an affine prior transform; native finite independent uniform bounds; generic posterior import followed by geometry-aware construction; explicit named-set ordering; isolated run directories; child-process timeout cleanup; physical sample transforms; and a distinction between beta completion and sampling adequacy. Preserve these improvements.

The documented static route uses `slipmodel`, `altar.models.seismic.cuda.static`, ordered parameter sets and a CUDA sampler. It describes mixed priors and moment-based initialization. The bridge currently supports the separately verified CPU `linear` route; changing a model name or GPU count does not implement that other contract. Green files should follow the explicit documented observation-by-parameter layout. HDF5 is an appropriate future format for a verified CUDA adapter. [AlTar Static documentation](https://altar.readthedocs.io/en/cuda/cuda/Static.html).

The framework documentation distinguishes CPU, CUDA and MPI execution, exposes proposal controls, and documents adaptive Metropolis under CUDA. Its archive distinguishes parameter particles from proposal covariance. These support the existing import design and motivate explicit backend-specific configuration and profiling. They do not establish CPU/GPU numerical equivalence on this installation. [AlTar Framework documentation](https://altar.readthedocs.io/en/cuda/cuda/AlTarFramework.html).

The earlier plan needs two substantive amendments: numerical validation must include the **native likelihood**, because local input algebra alone missed finding 1; and Gaussian problems should have a direct inference option, because AlTar compatibility is not a reason to run expensive MCMC when the full posterior is analytically available.

The current rejection of mixed priors, bounded Gaussian smoothing, fixed parameters, rake constraints and moment constraints is honest and preferable to ignoring them. These remain scientific capability gaps if required for the earthquake model. A moment-shaped initializer is not automatically a moment prior; inspect the chosen implementation's density as well as its initialization behavior.

## 4. Recommended route for approximately 1,000 elements

### Establish the actual statistical problem

An earthquake magnitude around Mw 7.6 does not itself determine computational cost. The relevant quantities are active slip components, observation count, covariance structure, nuisance parameters and prior/constraint choices. For 1,000 elements:

- One active slip component gives approximately 1,000 parameters plus ramps.
- Both SS and DS give approximately 2,000 parameters plus ramps.
- Keeping a fine physical mesh does not require treating every component as an independently resolved degree of freedom. Any coarse basis or dimensional reduction must be scientifically justified and checked for forward error and uncertainty loss.

The existing fixture's 0.5 m prior scale is a benchmark assumption, not a justified prior for the proposed earthquake. Define plausible slip, rake, spatial correlation and moment behavior from the intended scientific model. Use mesh-aware smoothing and test sensitivity to mesh refinement; blindly reusing a smoothing coefficient can change the implied prior when element size changes.

### Gaussian route: exact inference first

For fixed `G`, fixed effective covariance `C`, Gaussian anchor mean `m`, scales `D`, and smoothing operator `S`, compute:

```text
Qprior = D⁻² + Sᵀ S
Qpost  = Qprior + Gᵀ C⁻¹ G
h      = D⁻² m + Gᵀ C⁻¹ d
mean   = solve(Qpost, h)
```

With `Qpost = Rᵀ R`, draw independent posterior samples as `mean + solve(R, z)`, where `z ~ N(0,I)`. Use triangular solves and factor reuse rather than explicit inverses. This preserves Gaussian smoothing and linear ramps. Nonlinear summaries such as total scalar moment can then be computed from the samples without changing the inference method.

`AltarProblem.gaussian_reference` already contains most of the mean/covariance algebra. On the saved 144-element problem, its posterior calculation took **0.0049 seconds** in this audit, excluding data loading and problem validation. This is a single local measurement and is not a promised 2,000-parameter runtime. Turn it into an explicitly selected Gaussian solver using the shared result representation; do not silently substitute it for an AlTar request. Sampling from the precision factor avoids materializing the full covariance unless requested.

This route is exact only for the stated model. Positivity, truncated Gaussian smoothing, nonlinear geometry, hierarchical noise or a nonlinear moment prior require other inference machinery. Clipping Gaussian draws does not implement a truncated posterior.

### AlTar route: improve numerical work before scaling particle counts

Retain AlTar for the desired non-Gaussian route and as a cross-check. First fix correctness, expose proposal tuning, and measure short bounded experiments. Optimize the observed expensive operations: vectorized stable log densities, batched covariance handling and diagonal-noise fast paths. For fixed linear Gaussian noise, a verified sufficient-statistics likelihood can reuse `H=GᵀC⁻¹G`, `g=GᵀC⁻¹d`, and `c=dᵀC⁻¹d` rather than repeatedly operating in observation space. Compare its cost and numerical accuracy with the whitened forward form; it helps most when observation count substantially exceeds parameter count and still costs parameter-space matrix work.

Choose one next backend based on measured requirements. For the documented CUDA route, use a compatible NVIDIA deployment and build a separate static adapter, reusing the physical problem/layout and result layer. Confirm launcher, parameter-set order, prior semantics, archive shape, precision and likelihood values before performance claims. For MPI, verify per-worker versus total particles and actual speedup; parallelizing this implementation does not automatically remove its Python and covariance costs. The current audited Mac CPU deployment does not supply a validated CUDA path.

### Concrete memory and work budget

For an illustrative **2,000-parameter, 10,000-observation, 4,096-particle** float64 run, individual arrays alone require:

| Array | Decimal MB |
|---|---:|
| Green matrix, observations × parameters | 160.0 |
| Dense measurement covariance | 800.0 |
| One parameter covariance/factor | 32.0 |
| One particle population | 65.5 |
| One residual population, observations × particles | 327.7 |

This is not peak memory: candidates, originals, factors, native copies, serialization and workspaces add substantially. Observation count is an explicit illustration, not an assumption about the user's eventual data. At 50,000 observations one dense covariance alone is 20 GB. Diagonal/block structure and justified observation reduction matter at least as much as mesh size.

Per Metropolis population update, native forward multiplication scales as `O(N*n*p)`, dense covariance residual work as `O(N*n²)`, and proposal transformation as `O(N*p²)`. At fixed particle count, raising parameters from 288 to 2,000 multiplies the parameter-quadratic term by about **48.2**; doubling particles to 4,096 makes that approximately **96.5**. These are operation-count ratios, not wall-time predictions. Stage count and mixing also change.

The default 1,024 particles are fewer than 2,000 parameters: their empirical covariance has rank at most 1,023. Native conditioning can make a proposal matrix invertible but cannot create missing posterior information. A population of 4,096 is an experiment to budget, not a certified adequate sample size.

## 5. Implementation and acceptance plan

| Order | Work | Acceptance gate |
|---|---|---|
| 1 | Fix correlated likelihood; reject correlation loss; validate semantic geometry attachment | Deterministic likelihood parity, small-variance correlation rejection, and deliberate layout-mismatch rejection |
| 2 | Add explicit direct Gaussian solver through shared physical results | Mean/covariance agree with independent dense references; factor-generated sample statistics and predictions agree; include nonzero means, smoothing, ramps and correlated C |
| 3 | Expose CPU proposal controls and preserve progress on interruption | Short 144-element experiments report beta, actual/next scale, acceptance, resampling diversity, stage time and peak memory; remove observed alternating near-zero-acceptance behavior |
| 4 | Optimize diagonal covariance, affine transforms and measured native hotspots | Numerical parity at each change; before/after timings on fixed inputs and seeds; no hidden prior or likelihood changes |
| 5 | Validate complete 144-element runs | Three declared seeds reach beta one and pass exact Gaussian mean, variance and covariance-direction checks; preserve all artifacts and failures |
| 6 | Add only the required constrained-prior/backend capability | Small bounded/mixed-prior reference cases and CPU/backend likelihood equivalence before large runs |
| 7 | Scale through intermediate sizes to 1,000 elements | Record parameters, observations, noise representation, particles, stages, elapsed time, memory and accuracy; establish an explicit compute budget and scientific uncertainty checks |

For stage 5, keep the current benchmark mean-error `<0.25` posterior standard deviations and projection variance-error `<0.30` as initial regression gates. Extend projections beyond coordinate axes and eight random directions to include weak/strong eigenmodes and scientific quantities. These thresholds are engineering checks, not proof of posterior calibration. At larger dimension, interpret maxima with a dimension-aware Monte Carlo baseline rather than silently relaxing thresholds until a run passes.

Check posterior predictive residuals, spatial contrasts, slip magnitude and moment distributions; compare independent complete populations. Terminal CATMIP particles are not a time-series chain suitable for naive R-hat or ESS calculations. Persist stage statistics incrementally: the interrupted 144 run has no `BetaStatistics.txt`, although its log contains the information. Also make the benchmark record validation/import failures, not just `RuntimeError`, and preserve prior reports when rerunning into the same parent directory.

For an expensive run, the decision to increase timeout should follow measured stage behavior and posterior-quality evidence. Do not lower observation uncertainty, widen priors, change covariance or reduce the mesh merely to make a benchmark finish without documenting the scientific consequences.

## 6. Verification performed and limitations

- Read the bridge, its orchestrator integration, tests, backend contract, original plan and the linked AlTar documentation; inspected installed native likelihood, prior and sampler implementations.
- Re-ran bridge tests: **43 passed, 8 optional native tests skipped**.
- Re-ran available repository tests excluding `slipkit/tests/utils/test_viz.py`: **194 passed, 8 skipped**. The excluded pre-existing test imports the removed `slipkit.utils.viz` module; this is not an AlTar regression.
- Explicitly enabled and reran the native correlated-covariance/smoothing/ramp test: **1 passed** in 8.92 seconds. Its tolerance masks the confirmed likelihood error, as quantified above. The other seven native tests were not rerun in this review.
- Reproduced the covariance norm discrepancy, legacy correlation loss and layout mismatch acceptance without modifying package code.
- Inspected the actual failed 144-element manifest, configuration, intermediate HDF5 files, saved physical problem and sampler log. Did not repeat the hour-long run or claim a successful 144-element posterior.
- Independently timed three native operations and the exact Gaussian reference. The existing 9-element three-seed report was inspected, not regenerated; it reports passing accuracy gates with full-run times of 1,029/612/603 seconds.
- No 1,000-element run, CUDA build, MPI run or new constrained prior was implemented or validated. Those remain explicit implementation milestones.

Audit helper and numerical output are retained at `/tmp/slipkit-bridge-review/checks.py` and `/tmp/slipkit-bridge-review/evidence.json`; temporary files may be removed by the operating system. The key reproductions and measurements are embedded in this review so its conclusions do not depend on retaining those files.

## 7. Minimal native reproduction of the highest-priority defect

Run with the reviewed AlTar/Pyre environment:

```python
import altar
import numpy as np
from altar.models.linear.Linear import Linear
from altar.norms.L2 import L2

C = np.array([[1.0, 0.8], [0.8, 2.0]])
r = np.array([1.0, 2.0])
native_C = altar.matrix(shape=(2, 2))
native_C.ndarray()[:] = C
factor = Linear.computeCovarianceInverse(None, native_C)
native_r = altar.vector(shape=2)
native_r[0], native_r[1] = r
print(L2.withCovariance(None, native_r, factor) ** 2)
print(r @ np.linalg.solve(C, r))
# Installed native: 2.333893671801105
# Required:         2.058823529411765
```

These methods do not use instance state in the invoked paths, allowing the small deterministic check without launching an annealing run. This should become a supported-backend regression, alongside complete posterior tests.
