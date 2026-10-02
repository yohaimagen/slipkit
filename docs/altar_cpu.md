# SlipKit CPU AlTar contract

The supported route is **serial CPU linear**, tested with Python 3.12.9,
AlTar 2.0.2 revision 6646198, Pyre 1.9.6 revision 86ff61856, NumPy 2.4.2,
SciPy 1.17.0, h5py 3.12.1 and cutde 25.7.24. Native builds can differ despite
identical version strings; the solver verifies the audited native source hashes.
A changed revision must be validated with tiny fixtures before updating the
contract in `solver.py`. No unverified backend fallback occurs.

Install the native framework and its GSL/BLAS/LAPACK bindings using your AlTar
build/deployment process, preserving the exact source revisions. This repository
does not distribute native AlTar/Pyre builds or a portable native binary lock.
The audited machine interpreter is `/Users/ymagen/miniconda3/envs/slipkit/bin/python`,
with `linear` alongside it. The solver runs that launcher with the current Python.
`environment-altar.yml` pins the numerical stack; `pip install -e '.[bayesian,test]'`
installs SlipKit and Python extras. ArviZ is optional (`.[analysis]`) and required
only when explicitly asking for HDIs. The native installation remains a prerequisite for AltarBayesianSolver.
GaussianBayesianSolver uses the numerical stack without launching or importing AlTar.

## Problem and column order

`AltarAssembler.assemble_problem` returns `AltarProblem(G, data, covariance,
layout, smoothing)`. The physical problem retains raw inputs. Before a native launch the bridge
whitens G and d exactly once using the Cholesky factor of effective measurement
covariance, including fixed Cp, and exports identity covariance. This works
around the audited native correlated-covariance transpose defect without changing
the native installation. Preflight also checks deterministic normalized likelihood
parity using native L2 routines over both correlation signs and scaled covariances.
Columns are each fault's active SS then DS blocks, followed by each dataset's
normalized ramp coefficients. Any number of faults and either component selection
are supported. Observations, sigma, geometry and coordinates are recomputed every
run. `AltarAssembler(covariance=Cd)` supplies a full joint covariance; alternatively
`noise_models=[...]` supplies one existing NoiseModel per dataset. Without either,
the data covariance is a variance vector (sigma squared). Independent noise
stays a vector; multiple NoiseModels retain independent covariance blocks as a
tuple. Exactly diagonal matrices are compressed to vectors. Frozen numeric inputs
are copied and made read-only, and covariance factors are reused within the solve. Full matrices must be finite,
symmetric and positive definite. No jitter is silently inserted.

Geometry and observation coordinates must share a length unit (default km).
Green functions are scale invariant; slip, displacement and sigma are metres.
Mesh areas are in mesh units squared. `get_areas` on the Bayesian assembler converts
to m² using its explicit length_unit. Result moment functions take `length_unit`
explicitly (default km), multiply areas by 1e6 for km meshes, and use rigidity in Pa
and total vector slip per particle to produce N m. No fabricated area file is sent
to CPU linear. Sign conventions are those already applied by the physics engine,
recorded per fault; Gaussian priors are free-sign even for oriented kernels.

## Priors and unsupported requests

`AltarBayesianSolver(prior_scales=..., prior_mean=...)` specifies the Gaussian
anchor: `||(x-prior_mean)/prior_scales||² + ||Sx||²`. Scalars broadcast; vectors
follow the full column order, including ramps. `ss_prior_sigma` is a compatibility
alias for all scales; it is not SS-specific. S is the existing regularization
manager output including lambda, padded with zero ramp columns. The proper
Gaussian anchor removes the Laplacian nullspace. For nonzero anchor means the
actual prior mean is `solve(Q, D^-2 * prior_mean)`. The solver standardizes this
prior to a native unit Gaussian and maps all samples back to physical coordinates.

`prior='uniform'` with `bounds=(lo, hi)` means an independent uniform box prior,
not a clipped Gaussian. Unequal finite bounds use a native Uniform(0,1) affine
transform. Smoothing, fixed parameters, infinite bounds, Gaussian prior controls
with uniform priors, truncated Gaussians, mixed priors, rake/moment constraints,
MPI (`tasks>1`) and CUDA are unsupported and rejected. The inspected CPU Moment
inherits uniform density and lacks the CUDA moment constraint controls; the CPU
seismic implementation remains incompatible. No custom extension is warranted
for the two supported priors. Additional unsupported keywords raise TypeError.

`alpha_cp` adds a fixed observation-based diagonal `(alpha_cp * data)²` to Cd
before prior transformation. Only `cp_policy='fixed'` is accepted. Stage-updated
or proposal-dependent covariance needs a separately verified model implementation.

## Execution and result meaning

Use AltarAssembler and AltarBayesianSolver together on InversionOrchestrator.
The orchestrator constructs AltarSlipDistribution once with real faults and ramp
metadata. Matrix callers can use `solve(G, concatenate([data,sigma]))` for diagonal
noise, or `solve_problem(problem)`, then `get_last_posterior()` for generic arrays.
`get_last_result()` is geometry-aware only after an orchestrated inversion.
Geometry attachment compares the entire semantic layout, including fault index,
component, geometry checksum, signs, dataset identity and ramp order, degree,
normalization center/scale and basis checksum. A geometry-free matrix posterior
cannot be silently attached to faults; construct through the orchestrator or
provide a matching canonical layout from `parameter_layout`. The old importer
geometry-free result constructor and patch-count configuration
API have been replaced with a posterior record and actual parameter counts.

Each run has a fresh `run-*` directory, text inputs, physical arrays, ordered
layout, affine transform, source/input hashes, seed, recorder settings, manifest,
outputs and combined sampler.log. The default timeout is one hour; interruption
kills the entire subprocess group. Failures retain logs and clear prior results.
`keep_work_dir=False` deletes successful runs and clears all artifact paths;
failed runs are always retained. Never point the importer at a previous run to
recover a failed invocation. Recorder output_dir must remain within the run.

`AltarResultImporter.load(results_dir)` discovers the parent manifest and restores
physical samples; named sets require explicit manifest parameter_sets order and
widths. Without a manifest, supply `n_parameters` for raw theta import. Nonfinite,
wrong-shaped or beta<1 archives are rejected. `allow_incomplete=True` provides
explicit diagnostic-only access to a final archive or the latest saved numbered
stage, including failed manifests; those records cannot construct a public slip result.
This option cannot be enabled on the solver success path.
New manifests save the likelihood normalization and affine prior Jacobian offsets.
The importer restores physical log prior, likelihood and posterior densities using
those constants. Old manifests without offsets retain their original backend
coordinate density convention. Evidence is not exposed. Independent transforms
use scale/offset vectors; dense smoothing transforms use a checksummed binary NPZ
file. Old JSON matrix transforms remain readable. Covariance is computed from
physical terminal samples, never Annealer/covariance (proposal covariance).

`result.get_component_samples(component, fault_index)` returns the correct block;
absent components raise ValueError. `ss_samples` and `ds_samples` refer to the first
fault only. Mean/std/interval arrays are slip-only, fault-major. Nuisance particle
arrays are separate in `nuisance_samples`; fitted mean ramps use the base nuisance
interface. Credible intervals default to explicitly equal-tailed intervals;
`method='hdi'` requires ArviZ and never silently substitutes quantiles.
`annealing_complete` only means beta reached one. It is not a convergence or
independence certification. Diagnostics state sampling adequacy is not assessed.
The old `is_converged()` emits a deprecation warning. Acceptance includes invalid
proposals in the denominator; no universal optimal rate is claimed.

## Validation and scaling

Run the regression suite and opt-in sampler tests as described in README. Small
Gaussian and unequal finite uniform cases use three explicit seeds and fixed
full-run tolerances on normalized means and standard deviations. A separate real
case exercises ramps, full covariance and Gaussian smoothing. `gaussian_reference`
uses Cholesky solves to calculate the exact target, including nonzero anchors.

Run the 9-patch fixture with:

```
python -m slipkit.core.bayesian.benchmark slipkit/tests/fixtures/altar/patch-9 --work-dir /tmp/altar-9-check
```

Fixtures preserve the old numerical inputs with SHA-256 checksums. The benchmark
records elapsed time, allocated physical covariance bytes, mean error and marginal/random/eigenmode/column-contrast
projection variance error for each seed. Its accuracy gate is maximum normalized
mean error <0.25 and maximum projection relative variance error <0.30.
For 288 parameters, start experimental runs with 1024–2048 particles and increase
particles/steps if the gate fails. The 144-patch historical files can be passed to
the same benchmark; they are not an accuracy standard. Dense Cd costs 8*n*n bytes
before copies and native factorization, and dense prior factors cost O(p²) memory
and O(p³) factorization. Use existing downsampling before large dense cases.
MPI/GPU equivalence and a validated 144-patch performance envelope remain separate
experiments; they are not supported capabilities of this delivery.

## Historical validation before the code review fixes

These historical measurements preceded the correlated-noise correction. Their
stochastic full-covariance test did not detect the native transpose defect; use
the deterministic checks and corrected measurements below for the current contract.

The self-contained script ran successfully and compared physical means and
variances with its exact Gaussian target. The small native sampler cases passed
three explicit seeds for both prior families, plus a Gaussian full-covariance,
smoothing and ramp case. The full available suite passed 193 tests (8 optional
native checks skipped); a pre-existing `tests/utils/test_viz.py` cannot collect
because it imports the removed `slipkit.utils.viz` module and was excluded. Core
fault/physics/inversion/regularization/data/noise checks independently passed 80 tests.

The 9-patch benchmark (18 parameters, 1024 particles, 300 steps) passed seeds
17/31/47: maximum normalized mean errors 0.0596/0.0727/0.0603 and maximum
projection variance relative errors 0.0619/0.0876/0.0749. Full-run wall times
were 1029/612/603 seconds; the first run overlapped other validation processes,
so these are observed timings, not controlled performance medians. Cd itself
uses 93,312 bytes, excluding native copies and work arrays.

The 144-patch experiment used 288 parameters, 2048 particles, 300 steps, seed 17.
It reached only beta 0.007934 at recorded step 20 before the one-hour timeout.
It did not complete annealing and provides no validated physical posterior or
accuracy envelope. Failure logs and intermediate states are retained under
`/tmp/slipkit-144-benchmark/run-rsw60rdy`; the 9-patch report is
`/tmp/slipkit-9-benchmark/benchmark.json`. The benchmark accepts `--timeout` for
explicit longer experiments and saves failure reports. This native CPU build
needs further performance evaluation before a realistic 144-patch sampling
workflow can be promised. MPI/CUDA remain unsupported on this deployment.

The original working-tree patch and Python package snapshot were preserved in
`/tmp/slipkit-implementation-baseline`. Existing historical runs were not changed.

## Review fixes and explicit exact inference

For a fully Gaussian fixed linear problem select `GaussianBayesianSolver` explicitly:

```python
from slipkit.core.bayesian import GaussianBayesianSolver
solver = GaussianBayesianSolver(prior_scales=.5, prior_mean=0., draws=4096, seed=17)
# Set this solver and AltarAssembler on the existing InversionOrchestrator.
# Matrix callers can use solver.solve_problem(problem), then get_last_posterior().
```

This does not launch AlTar or fall back from a failed native request. It solves
posterior normal equations by Cholesky, retains the precision factor, reports
exact means and standard deviations, and generates independent Gaussian particles
with triangular solves. Full analytic covariance is formed only when requested.
Smoothing, nonzero anchor means, ramps, block/correlated noise and fixed Cp retain
exactly their existing model meanings. Finite bounds and clipping are rejected.
Use Gaussian particles for nonlinear slip-magnitude and moment summaries.

For AlTar expose `initial_scaling`, `acceptance_weight`, `rejection_weight`.
Native defaults are preserved; no high-dimensional tuning default is promised.
A fixed scale s uses `initial_scaling=s, acceptance_weight=0, rejection_weight=s`.
All resulting scales must lie in [0.01,1], matching the native clamp. Every manifest
records these controls. `cpu_kernel='vectorized'` explicitly selects a bridge-owned
CPU linear adapter with stable vectorized unit Gaussian/uniform log densities,
batched whitened residual norms and a verified identity-noise factor bypass.
It leaves the native installation intact and records its own source checksum.
Proposal generation, acceptance bookkeeping and stage covariance estimation still
use the native sampler. Text inputs and identity covariance remain dense in that
loader; this is a compute optimization, not a sparse observation loader.

`progress.json` is updated during execution and on interruption. It retains beta,
actual and next proposal scale, acceptance including invalid proposals, resampling
for the next stage, and stage durations. The vectorized adapter records process
peak resident memory; missing memory measurement is explicitly null. The importer
can use saved progress when native BetaStatistics.txt was not written. These
intermediate populations remain diagnostic-only.

Every benchmark invocation now creates a fresh `experiment-*` directory under the
chosen parent. Earlier reports cannot be overwritten. All solver/import/validation
exceptions produce a saved failure report; interruption stops the experiment.
The benchmark adds extreme covariance eigenmodes and column contrasts to its fixed
projection gates. It retains the original mean <0.25 and variance <0.30 thresholds.

Examples of explicitly bounded experiments:

```
python -m slipkit.core.bayesian.benchmark CASE --backend gaussian --chains 4096
python -m slipkit.core.bayesian.benchmark CASE --cpu-kernel vectorized --fixed-scales .08 .12 .16 --chains 1024 --steps 30 --timeout 45 --seeds 17
```

Short tuning runs are not successful posterior runs. Neither their acceptance
rates nor synthetic large Gaussian timings validate constrained earthquake
inference, a native 1,000-element sampler, MPI or CUDA.

## Measurements after the review fixes

The available repository suite passed **204 tests with 14 optional native tests
skipped**, excluding the pre-existing obsolete visualization import. Additional
input-validation and interrupt-cleanup regressions pass. All 14 native checks
passed in a separately enabled bridge run (66 tests total before the final two
regressions were added; those passed separately). Native verification is
opt-in with `SLIPKIT_RUN_ALTAR=1`; it includes deterministic likelihood checks and
end-to-end physical likelihood/prior normalization for both CPU kernels.

On the saved 144-element case (288 parameters, 351 observations), exact Gaussian
inference with 4,096 independent draws passed seeds 17/31/47. Maximum normalized
particle-mean errors were 0.04772/0.05366/0.04154; maximum projection variance errors
were 0.07694/0.08046/0.06985, including extreme eigenmodes and column contrasts.
Measured solve/draw/check times were 0.277/0.161/0.251 seconds, excluding initial
input loading and reference construction. The current diagonal covariance occupies
2,808 bytes; the native text loader still creates a dense identity matrix.

A seed-17 evaluator comparison used 1,024 particles, 288 parameters and 351
observations, with deterministic numerical parity before timing. Three-call median
prior times were 0.2543 seconds native versus 0.000223 vectorized; data likelihood
times were 0.3042 versus 0.05603 seconds. These overlapped other validation work and
are local observations, not isolated performance guarantees or full-sampler speedups.

Synthetic fixed linear Gaussian checks with 351 observations and 4,096 draws took
0.382/2.307/4.557 seconds at 288/1,000/2,000 parameters. Normal-equation relative
residuals were below 1.2e-15; maximum standardized draw-mean errors were below 0.052
and variance errors below 0.084. Process high-water memory was 210/368/676 MB,
cumulative across the sequential checks and including validation workspaces.
These synthetic matrices do not validate a real 1,000-element earthquake model.

Short saved-144 tuning runs at fixed scales 0.08/0.12/0.16 (1,024 particles,
30 Metropolis steps, seed 17, 45-second budgets) reported early-stage acceptance
between 22% and 54%, without alternating near-zero acceptance. They timed out at
beta below 0.004 and do not provide posterior samples. Larger bounded AlTar
experiments also have not established the three-seed 144-element completion gate;
the seed-17 attempt used 2,048 particles and 100 Metropolis steps, reached
beta 0.024674 after 31 completed stages, and timed out at 600 seconds with
344 MB peak resident memory. Seeds 31/47 were deliberately interrupted after
that budget failure; their partial diagnostics are retained. No completed native
144-element posterior is claimed. These runs overlapped other checks, and were
started before the final batched evaluator update; each manifest pins its version.
The exact Gaussian successes must not be interpreted as native sampler validation.

Reports retained for inspection (temporary files may be removed by the OS):

- Exact saved-144: `/tmp/slipkit-review-validation/experiment-llahijap/benchmark.json`.
- Synthetic checks and evaluator timings: `/tmp/slipkit-review-validation/synthetic-scale.json` and `profile.json`.
- Short proposal sweep: `/tmp/slipkit-review-tuning/experiment-ut0bkwal/benchmark.json`.
- Larger 144-element attempt: `/tmp/slipkit-review-144-complete/experiment-3b4p9brd/benchmark.json`.
- Bounded 9-element attempts: `/tmp/slipkit-review-9/experiment-g54rchvw/benchmark.json`.

Constrained priors, validated native 1,000-element sampling, MPI and CUDA remain
explicit future capabilities. The intended scientific constraint/backend has not
been specified or validated on this CPU deployment. No scientific bounds, noise,
mesh or prior were changed to force an accuracy gate to pass.

## Round 2 revisions and native CUDA pilot

The 2026-09-18 addendum prioritizes native CUDA validation. The CPU vectorized
loader remains unchanged pending that evaluation; it still reads dense text
identity noise. No custom constrained sampler has been added without a scientific
target. The public covariance exporter now adds fixed Cp correctly for variance
vectors, diagonal matrices and correlated full matrices; `export_all` shares the fix.

Cutde assembly now projects observations in configurable chunks (default 512).
`CutdeCpuEngine(observation_chunk_size=128)` bounds the nine-response workspace;
the final active-component kernel is preallocated. The Bayesian assembler also
preallocates fault/ramp columns, and whitening fills its final arrays directly.
Chunking changes no covariance blocks or physical correlations.

Both solver routes expose explicit durable results without imposing native run
machinery on Gaussian inference:

```python
solver.save_result("inference-bundle", faults=faults, datasets=datasets, length_unit="km")
from slipkit.core.bayesian import load_inference
saved = load_inference("inference-bundle")
posterior = saved.posterior
result = saved.result  # Requires the saved triangular geometry.
```

Save after `solve_problem` or an orchestrated inversion. Omitting geometry is
supported for matrix callers. Bundles contain frozen numeric inputs, covariance
blocks, smoothing, actual prior anchor/scales or uniform bounds, fixed Cp policy,
seed, environment/source hashes, geometry, dataset/ramp metadata, particles and
exact mean/precision factor when available. Native run settings/provenance are
included. Arrays are binary, checksummed using bounded reads and loaded without
pickle. Existing destinations are rejected. Gaussian experiments remain in memory
unless explicitly saved. Moment calculations retain the existing explicit units
argument; use the bundle's `metadata['length_unit']`.

`AltarProblem.gaussian_factor(..., beta=beta)` provides the actual tempered target.
The benchmark saves `stage_targets.json` for every recorded Gaussian stage,
including interrupted output, with coordinate/eigenmode/contrast errors and a
same-size independent-draw baseline. Thresholds remain mean <0.25 and variance
<0.30. Explicit stage imports require `allow_incomplete=True` and remain diagnostic
only, including a beta-one stage archive. Stage checks report distribution bias;
acceptance and beta completion alone do not certify sampling accuracy. Benchmark
metadata includes thread settings/source hashes and separate setup, inference,
posterior validation, stage validation and total timings. Setup includes input
loading, reference factorization/covariance and eigendecomposition.

### Real-geometry Gaussian capacity check

A controlled local process with requested BLAS thread limits of one used 1,000
triangles, both components (2,000 parameters), 10,000 synthetic LOS observations,
independent 0.01 m noise, no Cp/smoothing/ramps/constraints, and a free-sign zero-mean
0.5 m Gaussian anchor. This is a reproducible tilted planar mesh and a real Cutde
forward response; it is not a validated earthquake dataset or justified seismic prior.
The declared budget was 180 seconds and 2 GB process peak memory.

- Setup/forward data generation: 8.07 s; assembly: 23.30 s.
- Exact inference/draws: 1.404 s; independent QR/particle validation: 21.43 s.
- Save/reload: 0.462 s; total: 54.67 s; peak resident memory: 1.502 GB.
- Maximum exact mean difference from augmented QR: 3.74e-12 target standard deviations.
- Particle maximum mean/variance errors: 0.05712/0.08220; same-size independent
  Gaussian baseline (seed 111): 0.06258/0.07891. Extreme eigenmodes and contrasts included.
- Reload predictions and component/moment samples agree with the original result.

Frozen source hashes, full measurements and the reconstructible posterior are saved
at `/Users/ymagen/slipkit/altar_runing_example_144/round2-geometry/geometry-36vscyvg/`.
Reproduce with a separate process:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m slipkit.core.bayesian.scale_validation --work-dir VALIDATION_DIR --observations 10000
```

### hise_jump native execution

All remote work is under `/export/dump/ymagen/slipkit-altar-round2-20260918`, as
requested; the remote home directory is not used for source, environments, builds,
caches, logs or outputs. Host heisenbug has two NVIDIA RTX 3090 GPUs (24 GB each),
driver 535.309.01, CUDA toolkit 11.8, GCC 11.3, and approximately 503 GiB host RAM.
Other GPU workloads were present and were left running. No scheduler commands
were found; direct native GPU execution succeeded.

Built and installed the documented source pair in the isolated directory:
Pyre `86ff6185625139200c9e4d2f8a655c0911fd4645`,
AlTar `6646198a928d0ea3c24f8d32b4e04b2fc4fa471a`, targeting architecture 86.
The upstream nine-patch static example used float64, 256 particles and 100 updates
per stage. It reached beta one in 2.86 s including process startup, with about
304 MB peak host resident memory. Device monitoring recorded the native process
using about 298 MiB on GPU 0. This is an execution smoke test with the upstream
moment/strike-slip target, not posterior accuracy validation or a CPU/GPU speedup.
Local evidence: `/Users/ymagen/slipkit/altar_runing_example_144/round2-native/`.

A thin **experimental** `AltarCudaBayesianSolver` and native static launcher are
implemented locally for the unchanged Gaussian/uniform CPU target, using one
GPU and float64. Binary HDF5 inputs preserve existing columns and affine coordinates.
The upstream `cd_std` implementation still creates/inverts an N-by-N covariance,
so the thin bridge's identity-noise subclass uses a scalar factor and the same
native likelihood kernels instead. No identity array or file is exported. Native
Metropolis or AdaptiveMetropolis are selected explicitly; the adaptive sampler
receives the actual parameter count and bounded update/correlation settings.
Deterministic fixed-vector prediction, likelihood, prior and support checks run
before sampling. The original upstream model remains the reference implementation.
Native source hashes are pinned; no CPU fallback is permitted by the adapter.

The user approved the scoped 27-file source/input transfer. Required modules
and unchanged 144-element arrays were hash-verified on `hise_jump`, with all remote
work under `/export/dump/ymagen/slipkit-altar-round2-20260918`. No credentials,
Git history, unrelated files or rejected broad payload were transferred. Native
Pyre/AlTar commits and class hashes are pinned; a CUDA import is never treated as
a CPU fallback.

### Native CUDA numerical validation (H3–H4)

Three tiny Gaussian cases with positive, negative and zero observation-noise
correlation passed float64 physical prediction, prior, likelihood and posterior
checks. Both native fixed and adaptive samplers passed an independent bounded
Uniform(0,1) target against truncated-normal moments. The native Gaussian log
normalizer uses a float `PI` literal; the bridge checks fixed vectors and restores
only that measured constant on imported physical prior probabilities.

The upstream prior initializer seeds from `clock64`. The bridge uses native cuRAND
with the declared seed, which gives identical initial populations for the same
seed. Final particles are not bitwise reproducible: native `cudaMetropolis.cu` uses
`atomicAdd` to queue valid particles, changing which acceptance draw each particle
receives. This is recorded in run manifests; accuracy is evaluated across seeds.

The unchanged 144-element target has 288 active parameters, 351 observations,
Gaussian scale 0.5 m and fixed Cp=0. With 1,024 particles and seeds 17, 31, 47:

| Native sampler | Runs reaching beta one | Saved stages passing exact tempered target | Maximum normalized mean error | Maximum projection variance error | Native time per run |
|---|---:|---:|---:|---:|---:|
| Fixed scale 0.14, 1,000 updates/stage | 3/3 | 216/216 | 0.102 | 0.187 | 234.8–236.5 s |
| Adaptive scale 0.14, 1,000–4,000 updates/stage | 3/3 | 216/216 | 0.104 | 0.196 | 508.3–517.6 s |

All six runs met the prespecified mean <0.25 standard deviations and variance
<0.30 relative-error gates. Fixed runs used 70,000 updates; corrected adaptive
runs used about 72,000 updates. The first adaptive attempt started at scale 1.0
because the pinned native build initializes the sampler before model dimensions
are available; it reached beta one but failed several early variance stages. The
bridge now sets and verifies the actual initial scale. The failed attempt remains
saved as evidence and is excluded from the pass counts above.

The 4,096-particle fixed pilot (seed 17) also reached beta one, passing 75/75
saved-stage checks. It took 568.1 s for native inference
and 589.5 s including stage validation; its final mean and
variance errors were 0.056 and
0.067. Peak native process memory
was 374 MB host and
346 MB GPU. Its checksummed,
reload-tested bundle is in `altar_runing_example_144/round2-native/bridge/144-pilot/inference`.

### Larger mesh evidence (H5 pilot)

A synthetic tilted 1,000-triangle Cutde mesh with 10,000 independent-noise LOS
observations and 2,000 SS/DS parameters passed the exact Gaussian route on the
remote host. It used 2,048 independent draws, finished in 175.1 s with 948 MB
peak host memory, and matched an independent augmented-QR mean within 2.2e-12
posterior standard deviations. Its particle mean/variance maxima were 0.074/0.106,
near the same-size independent-draw baseline 0.068/0.106. This is a real forward
kernel on synthetic observations, not an earthquake-data or constrained-prior
validation. The local 4,096-draw version and its durable bundle are reported above.

Bounded GPU pilots used 4,096 particles on GPU 1, separately from the 144-element
runs on GPU 0. At 400 triangles, 800 parameters and 2,000 observations, the first
1,000-update stage took 96.2 s and reached beta 0.00075; both saved stages passed
exact-target checks. The 120 s cap stopped the incomplete run. At 1,000 triangles,
2,000 parameters and 10,000 observations, a 1,000-update stage did not complete
within 120 s. A 100-update stage took 149.5 s and reached beta 0.000071; its two
saved stages passed exact-target checks before the 180 s cap stopped the run.
The repeated 1,000-element partial run retained peak memory of 798 MB host and
1,401 MB on its GPU. Other users' GPU workloads were present, so these are
measured host-specific pilot costs, not isolated throughput or speedup claims.

The pinned upstream AlTar MPI example also completed on two GPUs: its two
workers selected devices 0 and 1, and the combined 256-particle nine-patch archive
reached beta one. This confirms a native multi-GPU execution path, not SlipKit
bridge parity or a speed comparison at fixed total particles. The bridge stays
single-GPU until worker-specific seed, identity-noise, importer and stage checks
are validated together. The smoke archive is saved as
`round2-native/bridge/upstream-multigpu-evidence.tar.gz`.

The largest **fully validated native GPU posterior** here is the 144-element
Gaussian target. The 1,000-element Gaussian target is validated through the exact
solver, while native GPU runs at that size establish only early-stage accuracy,
memory fit and a substantial throughput limit. Mixed bounded/smoothed priors,
float32 and multi-GPU sampling remain unvalidated. A scientific definition of
slip bounds, smoothing and any rake/moment constraints is required before a
constrained earthquake run can be called a validation.

Evidence and reloadable bundles are under
`/Users/ymagen/slipkit/altar_runing_example_144/round2-native/bridge/`;
remote source snapshots, raw logs, stages and reports remain under
`/export/dump/ymagen/slipkit-altar-round2-20260918`. Local verification:
229 available repository tests passed with native CPU checks enabled (obsolete
visualization import excluded), including final CUDA export/config and benchmark
checks.
