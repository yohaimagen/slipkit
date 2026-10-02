# Mw 7.6 Synthetic AlTar Inversion: Implementation and Handoff

## Purpose

This document gives a future agent enough context to continue the SlipKit–AlTar
bridge work without repeating the completed investigation. It records the
round-2 bridge validation, the large synthetic earthquake experiment, the exact
inversion configuration, the results, and the main unresolved scientific issue.

The immediate next objective is to add a spatially coherent slip prior while
retaining finite nonnegative slip bounds, then repeat the synthetic recovery and
validate it against held-out surface observations.

## Non-negotiable remote-host constraint

The GPU host alias is `hise_jump`.

All work on that host must stay below:

```text
/export/dump/ymagen
```

Do not create working directories, environments, caches, inputs, or results in
the remote home directory. The existing task root is:

```text
/export/dump/ymagen/slipkit-altar-round2-20260918
```

The host has two NVIDIA GeForce RTX 3090 GPUs. The current SlipKit CUDA bridge
runs one GPU per inversion process.

## Local repository and environments

The Git repository root is:

```text
/Users/ymagen/slipkit
```

The main local native environment is:

```text
/Users/ymagen/miniconda3/envs/slipkit/bin/python
```

Use these environment controls for deterministic local numerical work:

```bash
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
CUTDE_USE_BACKEND=cpp
```

The repository contains unrelated pre-existing modifications and untracked
files. Do not discard or reset them. The new Mw 7.6 files described below are
also currently uncommitted.

## Earlier round-2 bridge work already completed

The second review and scaling plan was implemented before the Mw 7.6 experiment.
The important completed items are:

1. Native CPU and native CUDA AlTar paths use explicit, audited launchers.
2. CUDA runs in float64 and never silently fall back to CPU.
3. Gaussian and independent finite uniform priors are supported.
4. Uniform priors require explicit finite bounds.
5. Observation covariance is whitened once before native execution.
6. Fixed observation-based `Cp` is supported.
7. Gaussian smoothing is supported on the Gaussian-prior path.
8. Uniform bounds combined with Gaussian smoothing are deliberately rejected by
   the current solver.
9. Native CUDA initialization is seeded. Final samples are not bitwise
   reproducible because the upstream atomic queue changes the assignment of
   acceptance random draws.
10. Geometry-aware, checksummed inference bundles can be saved and reloaded.

The primary bridge implementation is in:

```text
slipkit/core/bayesian/solver.py
slipkit/core/bayesian/cuda.py
slipkit/core/bayesian/native_cuda.py
slipkit/core/bayesian/problem.py
slipkit/core/bayesian/artifact.py
slipkit/core/bayesian/assembler.py
slipkit/core/bayesian/diagnostics.py
```

Detailed bridge documentation is in:

```text
docs/altar_cpu.md
```

### Completed numerical validation before the Mw 7.6 case

- The relevant local suite passed 229 tests. The obsolete
  `slipkit.utils.viz` test was excluded.
- A 144-element Gaussian case, with 288 parameters and 351 observations, was
  run on native CUDA with three fixed-sampler seeds and three corrected
  adaptive-sampler seeds.
- All six runs reached beta one and all saved temperature stages passed their
  exact tempered-target checks.
- These runs used 1,024 particles. Fixed runs took roughly 235 seconds and
  adaptive runs roughly 510 seconds.
- A 4,096-particle pilot also reached beta one and passed.
- Six checksummed bundles were copied back and reloaded locally.
- A larger 1,000-triangle, 2,000-parameter, 10,000-observation synthetic scale
  case passed the exact solver. Its native GPU early-stage cost was measured,
  but a full posterior was not established.
- An upstream two-GPU smoke test reached beta one. SlipKit's bridge remains a
  single-GPU solver, although independent processes can target different GPUs.

The prior round-2 evidence is stored locally below:

```text
/Users/ymagen/slipkit/altar_runing_example_144/round2-native/bridge
```

## Mw 7.6 synthetic earthquake definition

The generator is:

```text
slipkit/core/bayesian/validation/mw76_synthetic.py
```

The finalized case is:

```text
/Users/ymagen/slipkit/altar_runing_example_144/round2-mw76/mw76-7my8z3m8
```

Its metadata are in `report.json`, and all geometry, truth, observations, and
splits are in `synthetic.npz`.

### Fault geometry

- Vertical, right-lateral strike-slip fault.
- Length: 300 km.
- Depth: 25 km.
- Fault area: 7,500 km².
- Depth rows: 0, 3, 7, 12, 18, and 25 km.
- Nominal along-strike spacing increases from 3 km at the surface to 8 km at
  maximum depth.
- Final mesh: 594 triangular patches.
- One strike-slip parameter per triangle, for 594 parameters total.
- Median shallow characteristic element size: 3.0 km.
- Median deep characteristic element size: approximately 6.99 km.

### Invented slip and moment

The true slip is a smooth, nonnegative, separable along-strike/depth function.
It was normalized using a rigidity of 30 GPa to achieve Mw 7.6:

```text
target moment = 3.162277660168379e20 N m
area-weighted mean slip = 1.4054567378526126 m
maximum true slip = 2.291365195578278 m
```

The proposed synthetic uniform prior was used exactly:

```text
lower bound = 0 m
upper bound = maximum true slip + 2 m
            = 4.291365195578278 m
```

This upper bound uses knowledge of the hidden truth. It is acceptable for this
synthetic bridge experiment, but it is an oracle choice and must not be
presented as a deployable real-earthquake prior.

### Forward model and noise

- Forward engine: CUTDE elastic half-space.
- Observed component: horizontal displacement parallel to strike.
- Independent Gaussian noise standard deviation: 0.01 m.
- Random seed: 17.
- The same mesh and elastic forward model were used to generate and invert the
  data. This is an optimistic inverse-crime test. It isolates inversion and
  bridge behavior but does not include geometry or elastic-model error.
- Surface points exactly on the fault trace were avoided because the
  surface-reaching dislocation is singular there.

## Adaptive surface observations

The initial 10,000-point version used an approximately uniform 50 by 200 grid.
It was replaced with adaptive sampling at the user's request.

The final case starts from a 100 by 400 raster spanning:

```text
along strike: -180 to 180 km
across fault: -45 to 45 km
base spacing: approximately 0.9 km
```

A quadtree rule recursively subdivides blocks where the clean predicted
displacement standard deviation exceeds a tuned threshold. A refinement floor
within 5 km of the trace ensures that the observation density is highest next
to the fault. The selected standard-deviation threshold is approximately
0.01188 m.

The final survey has 10,003 observations:

| Distance from trace | Sites | Sites per km of cross-fault width |
|---|---:|---:|
| 0–5 km | 3,708 | 370.8 |
| 5–15 km | 3,802 | 190.1 |
| 15–45 km | 2,493 | 41.5 |

The split is stratified by these distance bands:

```text
training observations: 8,002
held-out observations: 2,001
```

The generator saves both training and held-out Green matrices and asserts that
matrix multiplication reproduces the matrix-free CUTDE predictions.

The observation-layout figure is:

```text
/Users/ymagen/slipkit/altar_runing_example_144/round2-mw76/mw76-7my8z3m8/sampling_map.png
```

## Inversion implementation

The inversion driver is:

```text
slipkit/core/bayesian/validation/mw76_invert.py
```

It provides three paths:

- `map`: box-constrained Gaussian-likelihood maximum a posteriori estimate.
- `cpu`: native AlTar CPU uniform-prior posterior.
- `cuda`: native AlTar CUDA uniform-prior posterior.

### Exact QR likelihood reduction

Direct native sampling with 8,002 rows was too expensive. A local native CPU
pilot with 1,024 particles, 100 updates per temperature, and a 180-second
timeout did not advance beyond its initial stage and allocated roughly 1.6 GB
of run data.

For independent Gaussian noise, the driver now reduces the whitened linear
likelihood exactly. If

```text
Gw = Q R
```

is the reduced QR factorization of the whitened 8,002 by 594 Green matrix, then
for any slip vector `s`:

```text
||Gw s - dw||² = ||R s - Qᵀ dw||² + constant
```

The sampler therefore evaluates a 594 by 594 problem while preserving every
slip-dependent term of the full 8,002-observation Gaussian likelihood. This is
not observation subsampling. The driver asserts numerical equality for test
vectors before launching AlTar and records the constant likelihood offset.

The locally saved reduced problem is:

```text
/Users/ymagen/slipkit/altar_runing_example_144/round2-mw76/inversion/reduced_problem.npz
```

## Inversion runs and results

### Box-constrained MAP baseline

The converged L-BFGS-B baseline used the same 0–4.291365 m bounds:

```text
training RMS = 0.009765845 m
held-out RMS = 0.010256364 m
slip RMS error = 0.845221 m
recovered Mw = 7.599265
```

This already demonstrated that an excellent data fit and moment recovery do not
imply recovery of the slip distribution.

The report is:

```text
/Users/ymagen/slipkit/altar_runing_example_144/round2-mw76/inversion/map_report.json
```

### Short CUDA pilot

The reduced CUDA pilot used:

```text
particles = 1,024
updates per temperature = 100
fixed scale = 0.03
seed = 17
GPU = 0
```

It reached beta one in approximately 244 seconds and used about 310 MiB of GPU
memory. Its posterior mean fit was poor:

```text
training RMS = 0.1353 m
held-out RMS = 0.1361 m
```

The pilot proved that the annealing path could finish, but 100 updates per
temperature were inadequate for mixing.

### Full CUDA posterior

The substantive run used:

```text
prior = independent Uniform(0, 4.291365195578278 m) for every patch
particles = 1,024
updates per temperature = 1,000
fixed proposal scale = 0.03
seed = 17
GPU = 0
physical observations represented = 8,002
native QR pseudo-observations = 594
parameters = 594
```

It completed 113 annealing stages, reached beta one, and took approximately
1,010 seconds, or 16.8 minutes. Final-stage acceptance was approximately 64%.
All 1,024 final particles are distinct.

The posterior-mean results are:

```text
training RMS = 0.010115090 m
held-out RMS = 0.010346180 m
slip RMS error = 0.707718581 m
recovered moment = 3.2372169877510134e20 N m
recovered Mw = 7.606781
posterior-mean minimum slip = 0.044889 m
posterior-mean maximum slip = 4.199218 m
```

Moment uncertainty from the final population is narrow within this fixed-model
experiment:

```text
posterior mean Mw = 7.606781
2.5% Mw quantile = 7.606029
97.5% Mw quantile = 7.607494
```

These intervals do not include geometry, elastic, noise-model, or forward-model
uncertainty and should not be interpreted as real-event uncertainty.

## Main scientific conclusion

The user correctly identified the central result: the problem is poorly
constrained at the patch-slip level even though the data are reproduced.

The evidence is:

1. Training and held-out displacement residuals are both at the injected 1 cm
   noise level.
2. Total moment and Mw are recovered accurately.
3. The posterior-mean slip map is visibly unlike the smooth true model.
4. Patch slip RMS error remains approximately 0.71 m.
5. The Green-matrix condition number is approximately `1.58e6`; the normal
   matrix condition is therefore on the order of `2.5e12`.
6. Shared-edge RMS slip difference is 0.129 m in the truth but 1.103 m in the
   posterior mean, about 8.5 times rougher.
7. Only about 42% of true patch values fall within the nominal patchwise 95%
   posterior intervals. This is a diagnostic of the mismatch between the
   independent box prior and the deliberately smooth truth, not a calibrated
   frequentist coverage statement because the truth was not drawn from the
   stated prior.

Dense sampling near the fault adds many measurements but not an equivalent
number of independent constraints on neighboring and deeper patches. Many
oscillatory slip combinations lie in weakly observed directions and generate
almost indistinguishable surface deformation. The independent uniform prior
has no spatial-coherence preference, so it permits checkerboard-like solutions
and can fit noise through adjacent patches.

The recovery plot is:

```text
/Users/ymagen/slipkit/altar_runing_example_144/round2-mw76/inversion/gpu-full/slip_recovery.png
```

## Local final artifacts

The compact completed result is below:

```text
/Users/ymagen/slipkit/altar_runing_example_144/round2-mw76/inversion/gpu-full
```

Important files:

```text
cuda_report.json              physical fit and run summary
posterior_diagnostics.json    uncertainty, conditioning, and roughness metrics
posterior_mean_slip.npy       posterior mean in canonical patch order
slip_recovery.png             true, recovered, and error fault-plane maps
inference/manifest.json       reloadable bundle metadata
inference/arrays.npz          checksummed posterior and reduced numeric problem
diagnostics/manifest.json     native run manifest
diagnostics/progress.json     all annealing stages
diagnostics/numerical-parity.json
diagnostics/host-resource.txt
diagnostics/sampler.log
```

The bundle arrays SHA-256 recorded after a successful local reload is:

```text
1bf6fa33cc1068f9b04cff4a151589db4e8fd9a88cae007790ab96dd353c09af
```

## Remote final artifacts

The complete full run remains at:

```text
/export/dump/ymagen/slipkit-altar-round2-20260918/runs/mw76/full-s03
```

Its native run, including every HDF5 temperature stage, is:

```text
/export/dump/ymagen/slipkit-altar-round2-20260918/runs/mw76/full-s03/native-runs/run-x37nr_r9
```

The synthetic inputs on the GPU host are:

```text
/export/dump/ymagen/slipkit-altar-round2-20260918/inputs/mw76/mw76-7my8z3m8
```

The remote runtime environment used was:

```bash
T=/export/dump/ymagen/slipkit-altar-round2-20260918
export PATH="$T/native/bin:$PATH"
export LD_LIBRARY_PATH="$T/native/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$T/snapshot-seeded-v4:$T/native/packages:$T/python-deps"
export XDG_CACHE_HOME="$T/cache"
export TMPDIR="$T/tmp"
export PYTHONPYCACHEPREFIX="$T/cache/pycache"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export CUTDE_USE_BACKEND=cpp
```

## Verification completed for the Mw 7.6 work

- Forward prediction and assembled Green-matrix prediction agree for the true
  slip on training and held-out sets.
- All generated arrays were checked for finite values.
- Training and held-out dimensions match the recorded metadata.
- The QR-reduced squared residual plus its constant matches the full whitened
  squared residual numerically for independent test slip vectors.
- The full native CUDA run reached beta one.
- The native CUDA deterministic float64 contract passed.
- The final bundle was copied back and reloaded locally.
- Reloaded samples exactly reproduce the saved posterior mean.
- All samples obey the finite uniform bounds.
- Both new validation modules pass Python bytecode compilation.

## Known limitations

1. Only one full 1,000-update random seed has been run for this uniform case.
   Repeat seeds are required before making a strong stochastic-convergence
   claim.
2. There is no spatial smoothing or correlated slip prior in the bounded run.
3. Current `AltarBayesianSolver` intentionally rejects a `uniform` prior when
   `problem.smoothing` is nonzero.
4. The observation component is along-strike horizontal displacement, not a
   realistic mixture of GNSS components and satellite line-of-sight tracks.
5. Noise is independent and homoscedastic at 1 cm.
6. Quadtree site selection uses the clean synthetic displacement. A real
   workflow would apply it to observed noisy imagery and handle masks.
7. The same geometry and forward engine are used for generation and inversion.
8. The prior upper limit is derived from the hidden true maximum.
9. A dense survey cannot by itself overcome the depth and neighboring-patch
   null space.
10. The present posterior uncertainty is conditional on a fixed fault,
    half-space, rigidity, and noise model.

## Recommended next implementation

The next scientifically justified experiment is a bounded, spatially coherent
slip prior. It should retain the nonnegative finite bounds while penalizing
differences between triangles that share an edge.

### Construct the smoothing operator

Build a sparse shared-edge incidence matrix `L`. For every adjacent triangle
pair `(i, j)`, add one row with appropriately normalized entries:

```text
L[row, i] = +w_ij
L[row, j] = -w_ij
```

Weights should account for patch geometry, for example shared-edge length and
centroid distance, and should be documented with units. Verify that constant
slip lies in the null space. Consider a separate weak damping term if an
intrinsic smoothness prior leaves an unwanted null mode.

### Preserve the meaning of the prior

Do not silently call smoothing rows observational data. The model should record
that the target density is:

```text
p(s) proportional to
    1[0 <= s <= upper]
    exp(-0.5 * lambda² * ||L s||²)
```

This is a truncated Gaussian Markov random-field prior, not an independent
uniform prior. The native likelihood can technically represent the quadratic
factor with augmented rows, but artifacts, probability reporting, normalizing
constants, and diagnostics must identify it as a prior term. Prefer an explicit
bridge implementation or a carefully audited transform over a semantic
shortcut.

### Choose smoothing with held-out prediction and recovery diagnostics

Run a predeclared grid of smoothing strengths. For each value record:

- training RMS;
- held-out RMS;
- slip RMS error, because this is synthetic;
- moment and Mw;
- shared-edge roughness;
- maximum and fraction of slips near either bound;
- depth-binned resolution or recovery;
- multiple random-seed agreement.

Select the weakest smoothing that removes the patch-scale checkerboard without
materially degrading held-out prediction. Do not select only by training fit.

### Compare with a coarser parameterization

The requested 3–8 km mesh should remain the primary case, but also invert a
coarser mesh as a resolution test. Compare its held-out error and slip recovery.
If a coarser mesh has the same predictive skill with much more stable slip, that
is direct evidence that 594 independent patches exceed the data resolution.

### Add more realistic observations after resolving the prior

After the bounded smooth-prior path is verified, add complementary observation
components or geometries, such as three-component GNSS and multiple InSAR
look directions. These may improve shallow and component resolution, but should
not be assumed to solve deep-slip non-uniqueness without an explicit resolution
analysis.

## Suggested continuation sequence

1. Read `docs/altar_cpu.md` and the current solver uniform/smoothing guards.
2. Load the local final bundle and reproduce `cuda_report.json` metrics.
3. Implement and unit-test the shared-edge smoothing operator on small meshes.
4. Define the exact bounded smooth-prior density and artifact metadata.
5. Add deterministic small-problem target checks before launching the large
   synthetic case.
6. Run local or reduced MAP smoothing-strength scans to identify a sensible
   range.
7. Run at least three native CUDA seeds for the chosen strengths on `hise_jump`,
   always under `/export/dump/ymagen`.
8. Compare held-out predictions, moment, slip recovery, roughness, bounds, and
   depth resolution.
9. Update `docs/altar_cpu.md` with the new model contract and evidence.

## Reproduction commands

Generate a fresh local case:

```bash
cd /Users/ymagen/slipkit
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUTDE_USE_BACKEND=cpp \
  /Users/ymagen/miniconda3/envs/slipkit/bin/python \
  -m slipkit.core.bayesian.validation.mw76_synthetic \
  --work-dir /Users/ymagen/slipkit/altar_runing_example_144/round2-mw76
```

Run the local constrained MAP baseline:

```bash
cd /Users/ymagen/slipkit
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/ymagen/miniconda3/envs/slipkit/bin/python \
  -m slipkit.core.bayesian.validation.mw76_invert \
  /Users/ymagen/slipkit/altar_runing_example_144/round2-mw76/mw76-7my8z3m8 \
  --backend map \
  --work-dir /Users/ymagen/slipkit/altar_runing_example_144/round2-mw76/inversion
```

The completed remote full-run configuration was equivalent to:

```bash
T=/export/dump/ymagen/slipkit-altar-round2-20260918
cd "$T"
/usr/bin/python3 runs/mw76/mw76_invert.py \
  inputs/mw76/mw76-7my8z3m8 \
  --backend cuda \
  --work-dir runs/mw76/full-s03 \
  --chains 1024 \
  --steps 1000 \
  --timeout 3600 \
  --scaling 0.03 \
  --gpu-id 0
```

Set the full remote environment shown earlier before running this command.
