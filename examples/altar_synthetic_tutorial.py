# %% [markdown]
# # ALTar Bayesian Static Slip Inversion — Synthetic Tutorial
#
# This notebook demonstrates the ALTar-backed Bayesian inversion module in
# SlipKit. It mirrors the structure of `bayesian_synthetic_tutorial.py` but
# replaces the PyMC SMC backend with ALTar's CATMIP algorithm.
#
# **Key differences from the PyMC tutorial:**
# - `AltarAssembler` stacks raw Strike-Slip / Dip-Slip kernels directly —
#   no rake rotation. Rake constraints live in the prior distributions.
# - `AltarBayesianSolver` exports HDF5 files, generates a `.pfg` config, and
#   runs `slipmodel` as a subprocess.
# - Results are an `AltarSlipDistribution` with per-patch posterior samples,
#   convergence diagnostics, and HDI intervals.
#
# **Prerequisites:**
# - ALTar 2.0 installed and `slipmodel` on your PATH.
#   See https://altar.readthedocs.io for installation instructions.
# - The mesh file `simple_strike_slip.msh` in this directory.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt

from slipkit.core.fault import TriangularFaultMesh
from slipkit.core.data import GeodeticDataSet
from slipkit.core.physics import CutdeCpuEngine
from slipkit.core.inversion import InversionOrchestrator
from slipkit.core.bayesian import (
    AltarAssembler,
    AltarBayesianSolver,
    AltarSlipDistribution,
)
from slipkit.utils.visualizers import SlipVisualizer

# %%
# ------------------------------------------------------------------
# 1. GEOMETRY: load the fault mesh
# ------------------------------------------------------------------
MESH_FILE = os.path.join(os.path.dirname(__file__), "simple_strike_slip.msh")

if not os.path.exists(MESH_FILE):
    raise FileNotFoundError(
        f"Mesh file not found: {MESH_FILE}\n"
        "Generate it first with gmsh using make_simple_Strike_slip.geo"
    )

fault = TriangularFaultMesh(MESH_FILE)
n_patches = fault.num_patches()
areas_m2 = fault.get_areas()          # (M,) patch areas in m²
areas_km2 = areas_m2 / 1e6

print(f"Loaded fault: {n_patches} patches, total area = {areas_km2.sum():.1f} km²")

# Physics engine
engine = CutdeCpuEngine(poisson_ratio=0.25)

# %%
# ------------------------------------------------------------------
# 2. SYNTHETIC TRUTH: define a Gaussian slip patch in SS/DS space
#
# Unlike the PyMC tutorial, we work directly in Strike-Slip / Dip-Slip
# coordinates because the ALTar assembler does NOT rotate kernels.
# A pure strike-slip event has true_ds ≈ 0.
# ------------------------------------------------------------------
centroids = fault.get_centroids()     # (M, 3) in km (x, y, z)

# Slip is concentrated at fault centre
center_y = 30.0
center_z = -10.0
sigma_patch = 5.0

dist = np.sqrt(
    (centroids[:, 1] - center_y) ** 2
    + (centroids[:, 2] - center_z) ** 2
)

max_slip = 2.0                        # metres at peak
true_ss = max_slip * np.exp(-0.5 * (dist / sigma_patch) ** 2)
true_ds = 0.1 * true_ss              # small dip-slip (~5° rake deviation)

# Full slip vector layout: [ss_1,...,ss_M, ds_1,...,ds_M]
true_slip = np.concatenate([true_ss, true_ds])

# Moment magnitude of the synthetic event
mu_pa = 30e9                          # shear modulus in Pa
true_mo = np.sum(mu_pa * areas_m2 * true_ss)
true_mw = (2 / 3) * (np.log10(true_mo) - 9.1)
print(f"Synthetic truth: max SS = {true_ss.max():.2f} m, "
      f"max DS = {true_ds.max():.3f} m, Mw = {true_mw:.2f}")

# Visualise true slip
SlipVisualizer.plot_slip_components(fault, true_slip)

# %%
# ------------------------------------------------------------------
# 3. SYNTHETIC DATA GENERATION
# ------------------------------------------------------------------

def make_insar(name, heading_deg, incidence_deg, fault, true_slip, engine,
               noise_frac=0.0, grid_n=20):
    """Creates a synthetic InSAR LOS dataset on a regular grid."""
    x = np.linspace(0.1, 40.1, grid_n)
    y = np.linspace(10.1, 50.1, grid_n)
    xv, yv = np.meshgrid(x, y)
    coords = np.column_stack([xv.ravel(), yv.ravel(), np.zeros(xv.size)])

    h = np.radians(heading_deg)
    i = np.radians(incidence_deg)
    los = np.array([np.sin(i) * np.cos(h), -np.sin(i) * np.sin(h), np.cos(i)])
    unit_vecs = np.tile(los, (len(coords), 1))

    dummy = GeodeticDataSet(coords, np.zeros(len(coords)), unit_vecs,
                            np.ones(len(coords)), name)
    G = engine.build_kernel(fault, dummy)
    clean = np.nan_to_num(G @ true_slip)

    sigma = np.abs(clean).max() * noise_frac + 0.001
    data = clean + np.random.default_rng(0).normal(0, sigma, len(clean))
    return GeodeticDataSet(coords, data, unit_vecs,
                           np.full(len(data), sigma), name)


def make_gnss(name, fault, true_slip, engine, noise_m=0.002, grid_n=5):
    """Creates a synthetic 3-component GNSS dataset."""
    x = np.linspace(5.1, 35.1, grid_n)
    y = np.linspace(15.1, 45.1, grid_n)
    xv, yv = np.meshgrid(x, y)
    n_sta = xv.size
    coords = np.column_stack([xv.ravel(), yv.ravel(), np.zeros(n_sta)])

    full_coords = np.repeat(coords, 3, axis=0)
    unit_vecs = np.tile(np.eye(3), (n_sta, 1))

    dummy = GeodeticDataSet(full_coords, np.zeros(len(full_coords)),
                            unit_vecs, np.ones(len(full_coords)), name)
    G = engine.build_kernel(fault, dummy)
    clean = np.nan_to_num(G @ true_slip)

    data = clean + np.random.default_rng(1).normal(0, noise_m, len(clean))
    return GeodeticDataSet(full_coords, data, unit_vecs,
                           np.full(len(data), noise_m), name)


NOISE = 0.02        # 2% fractional InSAR noise

insar_asc = make_insar("InSAR_Asc", heading_deg=-10,  incidence_deg=35,
                        fault=fault, true_slip=true_slip, engine=engine,
                        noise_frac=NOISE, grid_n=20)
insar_des = make_insar("InSAR_Des", heading_deg=190, incidence_deg=35,
                        fault=fault, true_slip=true_slip, engine=engine,
                        noise_frac=NOISE, grid_n=20)
gnss      = make_gnss("GNSS", fault=fault, true_slip=true_slip, engine=engine)

n_total = len(insar_asc) + len(insar_des) + len(gnss)
print(f"Generated {n_total} data points "
      f"({len(insar_asc)} asc + {len(insar_des)} des InSAR, {len(gnss)} GNSS components)")

# %%
# Quick look at the synthetic data
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, ds, title in zip(axes, [insar_asc, insar_des],
                          ["InSAR Ascending", "InSAR Descending"]):
    sc = ax.scatter(ds.coords[:, 0], ds.coords[:, 1], c=ds.data,
                    cmap="RdBu_r", s=8)
    fig.colorbar(sc, ax=ax, label="LOS displacement (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
plt.tight_layout()
plt.show()

# %%
# GNSS three components
gnss_n = gnss.coords.shape[0] // 3
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, label in enumerate(["East", "North", "Up"]):
    sc = axes[i].scatter(gnss.coords[::3, 0], gnss.coords[::3, 1],
                         c=gnss.data[i::3], cmap="RdBu_r", s=40)
    fig.colorbar(sc, ax=axes[i], label="Displacement (m)")
    axes[i].set_title(f"GNSS {label}  (N={gnss_n})")
    axes[i].set_aspect("equal")
plt.tight_layout()
plt.show()

# %%
# ------------------------------------------------------------------
# 4. ALTAR BAYESIAN INVERSION SETUP
# ------------------------------------------------------------------

# --- Assembler ---
# AltarAssembler stacks raw [G_ss | G_ds] — no rake rotation.
# Rake is constrained via prior distributions in the solver config.
assembler = AltarAssembler()

# --- Solver ---
# Priors that reflect the known physics of a strike-slip event:
#   - ss_prior_sigma = 1.0  → generous Gaussian prior on strike-slip
#   - ds_prior_support  → uniform prior allowing slight dip-slip
#
# Moment seeding: ALTar's Moment distribution initialises samples so
# the total seismic moment is consistent with the catalog Mw.

WORK_DIR = "./altar_synthetic_run"   # all ALTar I/O goes here

solver = AltarBayesianSolver(
    mw_mean=true_mw,
    mw_sigma=0.2,
    areas_m2=areas_m2,
    work_dir=WORK_DIR,
    alpha_cp=0.05,              # 5% fractional model error (static Cp)
    ss_prior_sigma=1.0,         # Gaussian prior σ on strike-slip (m)
    ds_prior_support=(-0.3, 1.0),  # Uniform prior bounds on dip-slip (m)
    chains=2**10,               # number of Markov chains
    steps=500,                  # Metropolis steps per beta step
    use_gpu=False,              # set True if a CUDA GPU is available
    output_freq=3,              # save HDF5 every 3 beta steps
    keep_work_dir=True,         # keep files after run for inspection
)

# --- Orchestrator ---
orc = InversionOrchestrator()
orc.add_fault(fault)
orc.add_data(insar_asc)
orc.add_data(insar_des)
orc.add_data(gnss)
orc.set_engine(engine)
orc.set_assembler(assembler)
orc.set_solver(solver)

# %%
# ------------------------------------------------------------------
# 5. RUN INVERSION
#
# This cell exports input HDF5 files, writes slipmodel.pfg, and
# invokes ALTar as a subprocess.  Progress is printed to stdout.
# lambda_spatial is ignored by AltarAssembler.
# ------------------------------------------------------------------
print("Starting ALTar Bayesian inversion …")
print(f"Working directory: {os.path.abspath(WORK_DIR)}")

result = orc.run_inversion(lambda_spatial=0.0)

print("\nInversion complete.")
print(f"Converged (beta=1):  {result.is_converged()}")
print(f"Beta steps taken:    {result.n_beta_steps}")
print(f"Final beta:          {result.final_beta:.6f}")

# %%
# ------------------------------------------------------------------
# 6. CONVERGENCE DIAGNOSTICS
# ------------------------------------------------------------------

# Beta trajectory and acceptance rate
result.plot_annealing_convergence()

# %%
# ------------------------------------------------------------------
# 7. POSTERIOR STATISTICS
# ------------------------------------------------------------------
mean_slip   = result.get_mean_slip()        # (2M,)  [mean_ss | mean_ds]
std_slip    = result.get_posterior_std()    # (2M,)
ci_95       = result.get_credible_intervals(hdi_prob=0.95)  # (2M, 2)
mag_stats   = result.get_slip_magnitude_stats()

mean_ss = mean_slip[:n_patches]
mean_ds = mean_slip[n_patches:]
std_ss  = std_slip[:n_patches]
std_ds  = std_slip[n_patches:]

print(f"\n--- Posterior summary ---")
print(f"Max mean SS:  {mean_ss.max():.3f} m   (truth: {true_ss.max():.3f} m)")
print(f"Max mean DS:  {mean_ds.max():.3f} m   (truth: {true_ds.max():.3f} m)")
print(f"Max SS std:   {std_ss.max():.3f} m")
print(f"Max DS std:   {std_ds.max():.3f} m")

# %%
# ------------------------------------------------------------------
# 8. VISUAL COMPARISON: truth vs posterior mean
# ------------------------------------------------------------------
print("\nTrue slip:")
SlipVisualizer.plot_slip_components(fault, true_slip)

print("Posterior mean slip:")
SlipVisualizer.plot_slip_components(fault, mean_slip)

# %%
# ------------------------------------------------------------------
# 9. SLIP MAGNITUDE UNCERTAINTY
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, vals, title in zip(
    axes,
    [mag_stats["mean"], mag_stats["std"],
     mag_stats["hdi_upper"] - mag_stats["hdi_lower"]],
    ["Posterior mean magnitude (m)",
     "Posterior std (m)",
     "95% HDI width (m)"],
):
    sc = ax.scatter(
        centroids[:, 0], centroids[:, 1],
        c=vals, cmap="plasma", s=40, vmin=0
    )
    fig.colorbar(sc, ax=ax, label="metres")
    ax.set_title(title)
    ax.set_aspect("equal")

plt.tight_layout()
plt.show()

# %%
# ------------------------------------------------------------------
# 10. PATCH-LEVEL POSTERIOR MARGINALS
#     (shows the full 1-D posterior for the highest-slip patches)
# ------------------------------------------------------------------
top_patches = np.argsort(mean_ss)[-6:][::-1]   # 6 patches with most SS
result.plot_slip_marginals(patch_indices=list(top_patches))

# %%
# ------------------------------------------------------------------
# 11. RECOVERY CHECK: SS residual per patch
# ------------------------------------------------------------------
residual_ss = mean_ss - true_ss
residual_ds = mean_ds - true_ds

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, res, title in zip(
    axes,
    [residual_ss, residual_ds],
    ["SS residual: mean − truth (m)", "DS residual: mean − truth (m)"],
):
    lim = np.abs(res).max()
    sc = ax.scatter(
        centroids[:, 0], centroids[:, 1],
        c=res, cmap="RdBu_r", vmin=-lim, vmax=lim, s=40
    )
    fig.colorbar(sc, ax=ax, label="metres")
    ax.set_title(title)
    ax.set_aspect("equal")

plt.tight_layout()
plt.show()

rms_ss = np.sqrt(np.mean(residual_ss ** 2))
rms_ds = np.sqrt(np.mean(residual_ds ** 2))
print(f"\nRMS SS residual: {rms_ss:.4f} m")
print(f"RMS DS residual: {rms_ds:.4f} m")

# %%
# ------------------------------------------------------------------
# 12. INSPECT ALTAR OUTPUT FILES
# ------------------------------------------------------------------
results_dir = os.path.join(WORK_DIR, "results")
h5_files = sorted(f for f in os.listdir(results_dir) if f.endswith(".h5"))
print(f"\nALTar output files in {results_dir}:")
for f in h5_files:
    size_kb = os.path.getsize(os.path.join(results_dir, f)) / 1024
    print(f"  {f:30s}  {size_kb:8.1f} kB")

# Annealing trajectory
print("\nAnnealing trajectory (last 5 steps):")
print(result.beta_statistics.tail(5).to_string(index=False))

# %%
