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

(Installation instructions will go here)

## Usage

(Quick start usage examples will go here)
