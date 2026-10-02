# Venezuela AlTar Bayesian inversion

This directory contains a reproducible six-track InSAR Bayesian inversion based
on `venezuela_inversion_with_ramp.ipynb`.

## Model

- 931 triangular fault patches
- strike-slip and dip-slip components: 1,862 slip parameters
- six independent degree-1 orbital ramps: 18 nuisance parameters
- 11,528 quadtree observations
- independent per-track noise sigmas estimated from the deterministic ramped
  residual RMS values
- zero-mean free-sign Gaussian slip prior, 5 m marginal anchor scale
- zero-mean Gaussian ramp prior, 0.5 m coefficient scale
- Laplacian smoothing and deep-edge damping

The fixed model is linear Gaussian. `GaussianBayesianSolver` evaluates its full
posterior exactly and produces independent Gaussian draws using the same
`AltarProblem`, parameter layout, ramp, covariance, smoothing, artifact, and
result contracts as the native bridge. Native AlTar MCMC is deliberately not
used because it would approximate a posterior that is already available in
closed form.

## Run

```bash
./run_exact.sh
```

The verified result is in:

```text
runs/exact-gaussian-seed17
```

Important outputs:

```text
report.json
posterior_summary.npz
posterior_mean_slip.png
posterior_mean_residuals.png
inference/manifest.json
inference/arrays.npz
```

## Interpretation limits

The uncertainty interval is conditional on fixed fault geometry, elastic
structure, Poisson ratio, ramp degree, regularization, and plug-in diagonal
noise sigmas. Spatially correlated atmospheric noise is not yet included. The
Gaussian slip prior is free-sign and does not impose the deterministic
nonnegative slip bounds. The source notebook did not explicitly set a dip-slip
kinematic type; this package preserves the kernel convention it actually used.
