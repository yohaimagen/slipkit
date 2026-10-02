# SlipKit: Earthquake Slip Inversion Package

SlipKit is a modular Python package designed for static earthquake fault slip inversion ($Gm = d$).
## Key Features:
*   **Physics Backend:** Leverages `cutde` (Triangle Dislocation Elements) for efficient elastic Green's functions computation on both CPU and GPU.
*   **Coordinate System:** Operates in local Cartesian coordinates (km) for consistent and accurate calculations.
*   **Regularization:** Implements Laplacian smoothing on unstructured fault meshes to promote geologically realistic slip distributions, plus optional deep-edge damping (`DeepEdgeDamping`) that pulls slip towards zero on the fault's deepest patches with a tunable coefficient.
*   **Extensibility:** Designed with Abstract Base Classes for Solvers, Fault Models, and Green's Functions, allowing users to easily integrate their own custom implementations.
*   **Resolution & Uncertainty:** Analytical resolution/covariance (`slipkit.core.resolution.ResolutionAnalyzer`) and empirical, bound-respecting Monte-Carlo error propagation + checkerboard recovery (`SyntheticRecoveryTest`, `MonteCarloResolution`), driven by per-dataset noise models (`slipkit.core.noise`). See below.

## Resolution & Uncertainty Analysis

SlipKit can quantify *how well the data resolve the slip model* for the linear
least-squares backend (`NnlsSolver` / `BoundedLsqSolver`):

*   **Per-dataset noise (`slipkit.core.noise`)** — `DiagonalNoise` for datasets
    with reported sigma (GNSS), and `EmpiricalInsarNoise.from_quiet_region` for
    InSAR (noise amplitude *and* spatial correlation fitted from a quiet,
    far-field region). The chosen model propagates into `Sigma`, the analytical
    covariance, and the Monte-Carlo draws.
*   **Analytical (`ResolutionAnalyzer`)** — model resolution `R`, resolution
    diagonal, Backus–Gilbert resolution-length map, posterior std `sigma_m`,
    cross-component (rake) leakage, and optional TSVD/Picard diagnostics. Fast,
    constraint-free diagnostics of the underlying linear operator.
*   **Empirical (`SyntheticRecoveryTest`, `MonteCarloResolution`)** — checkerboard
    / restoration / point-spread recovery tests and Monte-Carlo / jackknife /
    bootstrap ensembles run through the *real* solver and bounds. The MC
    ensemble is the authoritative, bound-respecting uncertainty.
*   **Visualization (`slipkit.utils.visualizers.ResolutionVisualizer`)** — maps
    all of the above onto the fault plane.

A worked example is in `venezuela_resolution.ipynb` (a dedicated companion to
`venezuela_inversion.ipynb`).

## Installation

Install core Python dependencies with `python -m pip install -e .`.
For the CPU AlTar bridge use `python -m pip install -e '.[bayesian,test]'`
inside a separately provisioned native AlTar/Pyre environment. The ordinary
Python packages called altar/pyre are not a substitute for that framework.
`environment-altar.yml` records the tested Python numerical stack; it does
not install the native framework. The verified environment uses AlTar 2.0.2
revision 6646198 and Pyre 1.9.6 revision 86ff61856. Preflight checks native
model/distribution/recorder source hashes against the tested implementation.
See [CPU bridge contract](docs/altar_cpu.md) for installation limits and API details.

## Usage

Run `python examples/altar_synthetic_tutorial.py` from the native environment.
The matching notebook uses the same tiny mesh defined directly from arrays.
It checks sampled means and uncertainties against an exact Gaussian reference.

Run bridge regressions with `python -m pytest slipkit/tests/core/test_bayesian_inversion.py`.
Add `SLIPKIT_RUN_ALTAR=1` to include the opt-in real sampler tests.
Older integration plans and PyMC examples are historical, not installation instructions
for the current CPU AlTar bridge.

## Exact Gaussian inference and CPU tuning

For fixed linear geometry/noise and Gaussian priors, select
`GaussianBayesianSolver(prior_scales=..., prior_mean=..., draws=4096, seed=17)`
with the same AltarAssembler and orchestrator. It returns exact posterior means
and independent physical posterior draws without launching AlTar. It does not
implement finite bounds or silently replace an AlTar request.

Native requests now whiten observations once to correct the installed correlated
covariance norm defect. Optional `cpu_kernel="vectorized"` optimizes standardized
CPU likelihoods without changing the native installation. Proposal controls and
incremental progress/failure reports are documented in [the CPU contract](docs/altar_cpu.md).
