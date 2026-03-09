# Implementation Plan: Bayesian Static Slip Inversion for SlipKit

This document outlines the strategy for implementing a Bayesian static earthquake slip inversion module within the `SlipKit` package. This implementation maps the mathematical framework from Minson, Simons, and Beck (2013) into the modular architecture of `SlipKit` using PyMC.

## 1. Scientific Objective
The goal is to move beyond point-estimate inversions (least-squares) by providing a full probabilistic characterization of fault slip. This includes:
* **Quantifying Uncertainty:** Estimating the posterior distribution of slip on each fault patch.
* **Dynamic Error Modeling:** Incorporating a "Model Prediction Error" ($\mathbf{C}_p$) that accounts for Green's function inaccuracies, scaling with the amplitude of observations.
* **Joint Parameter Estimation:** Estimating both the slip distribution and the fractional model error ($\alpha$) simultaneously.

## 2. Integration with SlipKit Architecture

The Bayesian module will be implemented as a specialized "pair" consisting of an **Assembler** and a **Solver**, designed to work within the existing `InversionOrchestrator` framework.

### 2.1. The Bayesian Assembler (`BayesianAssembler`)
Inheriting from `AbstractAssembler`, this class handles the preparation of matrices specifically for the Bayesian framework.

* **Responsibilities:**
    * **Kernel Rotation:** Transforms the raw Strike-Slip ($G_{ss}$) and Dip-Slip ($G_{ds}$) kernels into Rake-Parallel ($G_{\parallel}$) and Rake-Perpendicular ($G_{\perp}$) kernels based on a target rake angle $\phi$.
    * **Data Packing:** Unlike the `VanillaAssembler`, the `BayesianAssembler` returns the raw Green's functions, data observations, and uncertainties ($\sigma$). **It explicitly ignores the `RegularizationManager`**, as spatial smoothing is replaced by prior distributions.
* **Implementation Details:**
    * Returns an augmented matrix `A` (horizontal concatenation $[G_{\parallel}, G_{\perp}]$).
    * Returns an augmented vector `b` (stacked $\mathbf{d}_{obs}$ and $\sigma_{data}$).

### 2.2. The Bayesian Solver (`BayesianSolver`)
Inheriting from `SolverStrategy`, this class implements the PyMC probabilistic model and seeding logic.

* **Responsibilities:**
    * **Model Construction:** Unpacks the matrices and builds the PyMC model context.
    * **Initialization:** Implements the Dirichlet-seeding trick for SMC to ensure a physically plausible starting population.
    * **Probabilistic Sampling:** Executes `pm.sample_smc`.

## 3. Detailed Mathematical Implementation

### 3.1. Kernel Rotation Logic
The transformation from Strike/Dip to Parallel/Perpendicular components:
$$G_{\parallel} = G_{ss} \cos \phi + G_{ds} \sin \phi$$
$$G_{\perp} = -G_{ss} \sin \phi + G_{ds} \cos \phi$$

### 3.2. PyMC Model Structure (Prior Enforcement)
Unlike least-squares, which uses Laplacian smoothing matrices ($S_{reg}$), the Bayesian approach enforces "regularization" through the Choice of Probability Density Functions (PDFs):

* **Rake-Parallel ($U_{\parallel}$):** `pm.Uniform(lower=-0.1, upper=max_slip)`. 
    * *Reasoning:* A hard lower bound slightly below zero prevents the sampler from sticking at the boundary while enforcing the physical constraint of positive slip.
* **Rake-Perpendicular ($U_{\perp}$):** `pm.Normal(mu=0.0, sigma=rake_tolerance)`. 
    * *Reasoning:* Acts as a penalty against large deviations from the target rake direction, replacing the need for cross-component smoothing.
* **Fractional Model Error ($\alpha$):** `pm.LogNormal(mu=np.log(0.05), sigma=1.0)`.

### 3.3. Initialization: Seismic Moment & Dirichlet Seeding
The seismic moment is **not** used as a prior during sampling. Instead, it is used strictly to seed the initial population for the SMC sampler to avoid evaluating physically impossible models at $\beta=0$.

1. **Target Moment:** Draw a target $M_w$ from a Gaussian distribution centered on the earthquake catalog magnitude.
2. **Moment Partitioning:** Use a **Dirichlet distribution** to partition this total moment across all fault patches.
3. **Seeding:** Pass these partitioned slip models as the `initvals` to `pm.sample_smc`. This ensures the algorithm starts with a population that is physically plausible in terms of total seismic moment.

## 4. Output & Result Handling
The result of `sample()` will be a `BayesianSlipDistribution` object.
* **Attributes:** Stores the `arviz.InferenceData` object.
* **Methods:** Provides mean slip, credible intervals, and visualization of the $\alpha$ posterior.

## 5. Implementation Phases
1. **Phase 1 (Assembler):** Implement `BayesianAssembler` with kernel rotation and raw data extraction.
2. **Phase 2 (Solver):** Implement `BayesianSolver` with the PyMC model and the `_generate_dirichlet_seeds` helper.
3. **Phase 3 (Distribution):** Implement `BayesianSlipDistribution` for posterior analysis.
4. **Phase 4 (Validation):** Verify with a synthetic test case, ensuring compatibility with the `InversionOrchestrator` API.
