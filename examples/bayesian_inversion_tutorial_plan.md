# Implementation Plan: Synthetic Bayesian Slip Inversion Tutorial

This document outlines the implementation plan for a comprehensive synthetic tutorial demonstrating the Bayesian Static Inversion module in `SlipKit`. The tutorial will follow a notebook-style structure in a `.py` file.

## 1. Scientific Objective
The goal is to demonstrate the power of the Bayesian approach in recovering a known slip distribution and quantifying its uncertainty, especially when dealing with noisy, multi-sensor geodetic data. We will focus on a strike-slip fault system and joint inversion of InSAR and GNSS data.

## 2. Tutorial Workflow

### 2.1. System Setup
* **Fault Geometry**: Load the `simple_strike_slip.msh` file (generated from `make_simple_Strike_slip.geo`).
* **Physics Engine**: Initialize `CutdeCpuEngine` for elastic Green's function calculations.
* **Coordinate System**: Use local Cartesian coordinates (km) as defined in the mesh.

### 2.2. Synthetic Slip Generation
* Define a "true" slip distribution on the fault mesh.
* **Pattern**: A centered Gaussian slip distribution or a specific high-slip patch to simulate a localized rupture.
* **Components**: Primarily rake-parallel (strike-slip) with a small rake-perpendicular component to test the rake-tolerance parameter.

### 2.3. Synthetic Data Generation (Forward Modeling)
* **InSAR Dataset 1 (Ascending)**: Simulate LOS displacements with a specific ascending look vector.
* **InSAR Dataset 2 (Descending)**: Simulate LOS displacements with a specific descending look vector.
* **GNSS Dataset**: Simulate 3-component (E, N, U) displacements at a sparse network of surface stations.
* **Forward Model**: Use the physics engine to project the "true" slip into surface displacements.

### 2.4. Noise Injection & Error Modeling
* Add Gaussian noise to the synthetic datasets.
* **Variable Noise**: Implement a parameter to easily scale the noise level (from 0% to realistic levels).
* **Uncertainty ($\sigma$)**: Assign realistic data uncertainties to the `GeodeticDataSet` objects.

### 2.5. Bayesian Inversion Configuration
* **Patch Areas**: Calculate the area of each triangular patch (required for Dirichlet seeding).
* **Bayesian Assembler**: Initialize `BayesianAssembler` with the target rake (e.g., 0° for pure strike-slip).
* **Bayesian Solver**:
    * Set priors for $M_w$ based on the synthetic moment.
    * Configure SMC parameters: `draws=2000`, `chains=4`, `p_acc_rate=0.8`.
* **Orchestration**: Setup `InversionOrchestrator` with the Bayesian components.

### 2.6. Inference & Sampling
* Execute `orchestrator.run_inversion()`.
* Monitor SMC stages and acceptance rates.

### 2.7. Results & Comparison
* **Mean Recovery**: Compare the posterior mean slip distribution with the "true" synthetic distribution.
* **Uncertainty Mapping**: Visualize the 95% credible intervals (HDI) on the 3D mesh.
* **Model Error ($\alpha$)**: Analyze the posterior of the fractional model error parameter.
* **Diagnostics**: Generate ArviZ rank plots to verify chain convergence.

## 3. Implementation Details (Python Script)
The tutorial will be implemented in `examples/bayesian_synthetic_tutorial.py` using standard `SlipKit` visualization tools and `arviz`.
