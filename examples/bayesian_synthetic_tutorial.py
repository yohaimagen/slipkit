"""
SlipKit Tutorial: Synthetic Bayesian Slip Inversion
===================================================

This tutorial demonstrates how to use the Bayesian inversion module in SlipKit
to invert synthetic geodetic data (InSAR and GNSS) for fault slip.

Scientific Objective:
Recover a known slip distribution and quantify its uncertainty using 
Sequential Monte Carlo (SMC) sampling.
"""

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import os

from slipkit.core.fault import TriangularFaultMesh
from slipkit.core.data import GeodeticDataSet
from slipkit.core.physics import CutdeCpuEngine
from slipkit.core.inversion import InversionOrchestrator
from slipkit.core.bayesian import BayesianAssembler, BayesianSolver, BayesianSlipDistribution
from slipkit.utils.visualizers import SlipVisualizer, ForwardModelVisualizer

# ---------------------------------------------------------
# 1. SETUP & PHYSICS
# ---------------------------------------------------------
# Use the simple strike-slip mesh generated from make_simple_Strike_slip.geo
# If the file is not in examples/, adjust the path accordingly.
MESH_FILE = "examples/simple_strike_slip.msh"

if not os.path.exists(MESH_FILE):
    raise FileNotFoundError(f"Mesh file {MESH_FILE} not found. Please generate it first.")

# Load the fault mesh
fault = TriangularFaultMesh(MESH_FILE)
n_patches = fault.num_patches()
print(f"Loaded fault with {n_patches} patches.")

# Initialize the physics engine
engine = CutdeCpuEngine(poisson_ratio=0.25)

# ---------------------------------------------------------
# 2. SYNTHETIC SLIP DISTRIBUTION
# ---------------------------------------------------------
# Create a Gaussian slip distribution centered on the fault
centroids = fault.get_centroids()
center_y = 30.0
center_z = -10.0
sigma = 5.0

# Calculate distance from center for each patch
dist = np.sqrt((centroids[:, 1] - center_y)**2 + (centroids[:, 2] - center_z)**2)
max_slip_true = 2.0
true_slip_par = max_slip_true * np.exp(-0.5 * (dist / sigma)**2)
true_slip_perp = 0.1 * true_slip_par  # Small rake deviation

# Full slip vector: [u_par, u_perp]
# Note: SlipKit core expects [u_ss, u_ds] but the Bayesian module 
# handles the rotation. For data generation, we'll manually rotate to SS/DS.
target_rake_deg = 0.0 # Pure strike-slip orientation
rad = np.radians(target_rake_deg)
cos, sin = np.cos(rad), np.sin(rad)

# R = [[cos, sin], [-sin, cos]] -> [par, perp]^T = R [ss, ds]^T
# [ss, ds]^T = R^T [par, perp]^T = [[cos, -sin], [sin, cos]] [par, perp]^T
true_ss = true_slip_par * cos - true_slip_perp * sin
true_ds = true_slip_par * sin + true_slip_perp * cos
true_slip_raw = np.concatenate([true_ss, true_ds])

print(f"Synthetic Slip: Max Parallel = {np.max(true_slip_par):.2f}m")

# Visualize the true slip
SlipVisualizer.plot_slip_components(fault, true_slip_raw)
plt.show()

# ---------------------------------------------------------
# 3. SYNTHETIC DATA GENERATION
# ---------------------------------------------------------
noise_level = 0.05 # 5% relative noise

def create_synthetic_insar(name, heading, incidence, fault, true_slip, engine, noise=0.0):
    # Create a grid of observation points, shifted slightly from x=20 to avoid NaNs
    x = np.linspace(0.1, 40.1, 20)
    y = np.linspace(10.1, 50.1, 20)
    xv, yv = np.meshgrid(x, y)
    coords = np.column_stack([xv.flatten(), yv.flatten(), np.zeros_like(xv.flatten())])
    
    # Calculate LOS unit vector
    h_rad = np.radians(heading)
    i_rad = np.radians(incidence)
    los_vec = np.array([
        np.sin(i_rad) * np.cos(h_rad), 
        -np.sin(i_rad) * np.sin(h_rad), 
        np.cos(i_rad)
    ])
    unit_vecs = np.tile(los_vec, (len(coords), 1))
    
    dummy_ds = GeodeticDataSet(coords, np.zeros(len(coords)), unit_vecs, np.ones(len(coords)), name)
    G = engine.build_kernel(fault, dummy_ds)
    clean_data = G @ true_slip
    
    if np.any(np.isnan(clean_data)):
        print(f"Warning: NaNs detected in {name} clean data. Replacing with 0.")
        clean_data = np.nan_to_num(clean_data)

    sigma = np.max(np.abs(clean_data)) * noise + 0.001
    noisy_data = clean_data + np.random.normal(0, sigma, size=len(clean_data))
    
    return GeodeticDataSet(coords, noisy_data, unit_vecs, np.full_like(noisy_data, sigma), name)

def create_synthetic_gnss(name, fault, true_slip, engine, noise=0.0):
    # Sparse stations, shifted
    x = np.linspace(5.1, 35.1, 5)
    y = np.linspace(15.1, 45.1, 5)
    xv, yv = np.meshgrid(x, y)
    coords = np.column_stack([xv.flatten(), yv.flatten(), np.zeros_like(xv.flatten())])
    
    # 3 components per station
    full_coords = np.repeat(coords, 3, axis=0)
    unit_vecs = np.tile(np.eye(3), (len(coords), 1))
    
    dummy_ds = GeodeticDataSet(full_coords, np.zeros(len(full_coords)), unit_vecs, np.ones(len(full_coords)), name)
    G = engine.build_kernel(fault, dummy_ds)
    clean_data = G @ true_slip
    
    if np.any(np.isnan(clean_data)):
        print(f"Warning: NaNs detected in {name} clean data. Replacing with 0.")
        clean_data = np.nan_to_num(clean_data)

    sigma = 0.002 # 2mm fixed noise for GNSS
    noisy_data = clean_data + np.random.normal(0, sigma, size=len(clean_data))
    
    return GeodeticDataSet(full_coords, noisy_data, unit_vecs, np.full_like(noisy_data, sigma), name)

# Generate datasets
insar_asc = create_synthetic_insar("InSAR_Asc", heading=-10, incidence=35, fault=fault, true_slip=true_slip_raw, engine=engine, noise=noise_level)
insar_des = create_synthetic_insar("InSAR_Des", heading=190, incidence=35, fault=fault, true_slip=true_slip_raw, engine=engine, noise=noise_level)
gnss = create_synthetic_gnss("GNSS", fault=fault, true_slip=true_slip_raw, engine=engine, noise=noise_level)

print(f"Generated {len(insar_asc) + len(insar_des) + len(gnss)} data points.")

# ---------------------------------------------------------
# 4. BAYESIAN INVERSION CONFIGURATION
# ---------------------------------------------------------
# Calculate patch areas for Dirichlet seeding
areas_km2 = fault.get_areas()
areas_m2 = areas_km2 * 1e6
print(f"Total Fault Area: {np.sum(areas_km2):.2f} km^2")

# Setup Assembler
assembler = BayesianAssembler(target_rake_deg=target_rake_deg)

# Setup Solver
# Estimated Mw: True Mo = mu * Area * mean_slip
mu = 30e9
# True Mo = sum(mu * area_i * slip_i)
true_mo = np.sum(mu * areas_m2 * true_slip_par)
true_mw = (2/3) * (np.log10(true_mo) - 9.1)
print(f"True Mw: {true_mw:.2f}")

solver = BayesianSolver(
    mu_mw=true_mw, 
    sigma_mw=0.1, 
    areas=areas_m2, # Pass in m^2
    draws=2000, 
    chains=4,
    random_seed=42
)

# Orchestrate
orchestrator = InversionOrchestrator()
orchestrator.add_fault(fault)
orchestrator.add_data(insar_asc)
orchestrator.add_data(insar_des)
orchestrator.add_data(gnss)
orchestrator.set_engine(engine)
orchestrator.set_assembler(assembler)
orchestrator.set_solver(solver)

# ---------------------------------------------------------
# 5. INVERSION
# ---------------------------------------------------------
print("Running Bayesian Inversion...")
# lambda_spatial is ignored by BayesianAssembler
result = orchestrator.run_inversion(lambda_spatial=0.0) 

# Retrieve Bayesian results
idata = solver.get_inference_data()
bayesian_result = BayesianSlipDistribution(result.slip_vector, [fault], idata)

# ---------------------------------------------------------
# 6. RESULTS & DIAGNOSTICS
# ---------------------------------------------------------
# Compare posterior mean with truth
posterior_mean = bayesian_result.slip_vector
mean_par = idata.posterior.u_par.mean(dim=["chain", "draw"]).values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(true_slip_par, 'k-', label="True Parallel")
ax1.plot(mean_par, 'r--', label="Posterior Mean")
ax1.set_title("Rake-Parallel Slip Recovery")
ax1.legend()

# Uncertainty plot
hdi = bayesian_result.get_credible_intervals(hdi_prob=0.95)
hdi_par = hdi[:n_patches]
ax2.fill_between(range(n_patches), hdi_par[:, 0], hdi_par[:, 1], color='r', alpha=0.3, label="95% HDI")
ax2.plot(true_slip_par, 'k-')
ax2.set_title("Uncertainty (95% HDI)")
ax2.legend()
plt.show()

# Alpha posterior
bayesian_result.plot_alpha_posterior()

# SMC Diagnostics
bayesian_result.plot_diagnostics()

print("Tutorial Complete.")
