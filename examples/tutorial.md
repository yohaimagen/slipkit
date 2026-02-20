# Multi-Sensor InSAR Slip Inversion with SlipKit

This tutorial demonstrates how to use the `slipkit` library to perform a slip inversion of the [2025 Mw7.1 Tingri earthquake](https://earthquake.usgs.gov/earthquakes/eventpage/us6000pi9w/executive) using multiple InSAR datasets. We will invert data from ALOS and Sentinel-1 satellites to resolve the slip distribution on a triangular fault mesh.

**Prerequisites:**
- `slipkit`
- `cutde`
- `numpy`
- `matplotlib`

```python
import numpy as np
%load_ext autoreload
%autoreload 2
from slipkit.core.fault import TriangularFaultMesh
from slipkit.core.data import GeodeticDataSet
from slipkit.core.physics import CutdeCpuEngine
from slipkit.core.solvers import NnlsSolver
from slipkit.core.inversion import InversionOrchestrator

from slipkit.utils.visualizers import SlipVisualizer
from slipkit.utils.visualizers import SarDataFitVisualizer
from slipkit.utils.visualizers import LCurveVisualizer
from slipkit.utils.visualizers import ForwardModelVisualizer


from slipkit.utils.parsers import InsarParser
import matplotlib.pyplot as plt
```

## System Setup & Physics

First, we define the fault geometry and the physics engine. The fault geometry is loaded from a `.msh` file, which contains the 3D geometry of the fault surface. This file can be generated from various mesh generators, in this case it was created with `gmsh` from the `make_fault_mesh.geo` script. We also specify the expected slip components.

### Note: the fault mesh need to be in the same coordinate system as the data, we project the data from geographic coordinate system (lat, lon) to UTM in km with origin_lon, origin_lat

```python
origin_lon = 87.0
origin_lat = 28.0
```

The `CutdeCpuEngine` is our physics engine, responsible for calculating the elastic Green's functions that relate fault slip ona triangular fault patch to surface displacement.

```python
# Define the path to your data files
# Please replace this with the actual path to your data
DATA_DIR = "./"

fault = TriangularFaultMesh(
    mesh_input=f"{DATA_DIR}fault.msh", 
    strike_slip_type='left-lateral', 
    dip_slip_type='normal'
)

# Initialize the physics engine
engine = CutdeCpuEngine(poisson_ratio=0.25)
```

To ensure the mesh loaded correctly, we can visualize its geometry.

```python
SlipVisualizer.plot_slip_components(
    fault=fault, 
    slip_vector=np.ones(fault.num_patches() * 2),  # Example slip vector
    cmap='plasma',
    plot_edges=False,
    elev=10, 
    azim=0,
)
plt.tight_layout()
plt.show()
```

## Forward Modeling "Sanity Check"

Before inverting real data, it's a good practice to perform a synthetic "sanity check". We simulate a known slip distribution (e.g., pure dip-slip) and check if the resulting surface deformation patterns (East, North, Up) make physical sense. This helps verify that our coordinate systems and fault orientation are correct. For instance, a pure dip-slip fault should produce quadrants of uplift and subsidence.

```python
# Example: pure dip-slip
slip_vector_synthetic = np.concatenate([
    np.zeros(fault.num_patches()),
    np.ones(fault.num_patches())
])

# Define grid extent based on some relative values or default values
# For this example, we'll use a fixed extent for demonstration.
x_extent = (0, 100) # Example: meters
y_extent = (30, 130) # Example: meters
grid_resolution = 100

ForwardModelVisualizer.plot_enu_response(
    fault=fault, 
    slip_vector=slip_vector_synthetic, 
    engine=engine,
    x_extent=x_extent,
    y_extent=y_extent,
    grid_resolution=grid_resolution,
    title="Surface Displacement Patterns for Dip-Slip Faulting",
    figsize=(15, 5)
)
```

## Data Ingestion & Downsampling

InSAR interferograms can contain millions of pixels. To make the inversion computationally tractable, we downsample the data. We use Quadtree downsampling (`downsample_method='quadtree'`) based on data variance (`quadtree_var_thresh`) to reduce the dataset size while preserving high-gradient deformation features near the fault.

Here, we load three InSAR datasets: ALOS Descending, ALOS Ascending, and Sentinel-1 Ascending.

```python
# ALOS Descending
alos_des = InsarParser.read_geotiff(
    filepath=f'{DATA_DIR}ALOS_241229_250112_des.tiff',
    origin_lon=origin_lon,
    origin_lat=origin_lat,
    downsample_method='quadtree',
    downsample_factor=(64, 64),
    quadtree_var_thresh=0.01,
    heading=190.9,
    incidence=34.9
)

# ALOS Ascending
alos_ase = InsarParser.read_geotiff(
    filepath=f'{DATA_DIR}ALOS_240921_250125_ase.tif',
    origin_lon=origin_lon,
    origin_lat=origin_lat,
    downsample_method='quadtree',
    downsample_factor=(64, 64),
    quadtree_var_thresh=0.01,
    heading=-9.8,
    incidence=34.9
)

# Sentinel-1 Ascending
s1_ase = InsarParser.read_geotiff(
    filepath=f'{DATA_DIR}S1_20250105_20250117_ase.tif',
    origin_lon=origin_lon,
    origin_lat=origin_lat,
    downsample_method='quadtree',
    downsample_factor=(64, 64),
    quadtree_var_thresh=0.01,
    heading=-12.62,
    incidence=39.6593
)

# Plot the downsampled datasets
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
datasets = [alos_des, alos_ase, s1_ase]
titles = ["ALOS Descending", "ALOS Ascending", "Sentinel-1 Ascending"]

for ax, ds, title in zip(axes, datasets, titles):
    sc = ax.scatter(ds.coords[:, 0], ds.coords[:, 1], c=ds.data, cmap='viridis', s=5)
    fig.colorbar(sc, ax=ax, label='Displacement (m)')
    ax.set_title(f'Downsampled LOS: {title} (N={ds.coords.shape[0]})')
    ax.set_aspect('equal', 'box')

plt.tight_layout()
plt.show()
```

## Inversion Configuration

Now, we configure the `InversionOrchestrator`. This is the main user-facing API that ties together the data, fault model, physics engine, and solver. We add the fault model and all three datasets to the orchestrator. We also set the solver to `NnlsSolver` (Non-Negative Least Squares), which will constrain our solution to have non-negative slip values.

```python
inversion = InversionOrchestrator()
inversion.add_fault(fault)
inversion.add_data(alos_des)
inversion.add_data(s1_ase)
inversion.add_data(alos_ase)
inversion.set_engine(engine)
inversion.set_solver(NnlsSolver())
```

### Regularization and the L-Curve

To solve this ill-posed inverse problem, we use Tikhonov regularization. We want to minimize the following cost function, which balances fitting the data and keeping the model physically realistic (i.e., smooth):

$$ \min_m \|Gm - d\|_2^2 + \lambda^2 \|Lm\|_2^2 $$

- $G$: Green's Functions (Elastic Kernels).
- $d$: Observed displacements (InSAR).
- $L$: Finite-difference Laplacian (Smoothing Matrix).
- $m$: Slip vector (parameter to solve).
- $\lambda$: Regularization parameter (smoothing coefficient).

The L-curve helps us find the optimal balance between data misfit ($\|Gm - d\|_2$) and model roughness ($\|Lm\|_2$). We will run the inversion for a range of $\lambda$ values and plot the L-curve to find the "corner", which represents the optimal $\lambda$.

```python
# Run the L-curve analysis
lambda_values = np.logspace(-2, 1, 40)
lambdas, misfits, roughnesses = inversion.run_l_curve(lambdas=lambda_values)

# Plot the L-curve
LCurveVisualizer.plot_l_curve(lambdas, misfits, roughnesses)
```

## Running the Inversion

From the L-curve, we can pick an optimal `lambda` value at the corner of the curve. We then run the inversion with this `lambda` and non-negativity constraints. The bounds `(0, inf)` are used because we assume the physical slip direction is known and we want to prevent non-physical "back-slip".

```python
optimal_lambda, _, _ = LCurveVisualizer.find_corner(lambdas, misfits, roughnesses)

# Bounds for non-negative slip
n_patches = fault.num_patches()
lower_bounds = np.zeros(2 * n_patches)
upper_bounds = np.full(2 * n_patches, np.inf)
bounds = (lower_bounds, upper_bounds)

# Run the final inversion with the optimal lambda
result = inversion.run_inversion(lambda_spatial=optimal_lambda, bounds=bounds)
inverted_slip = result.slip_vector
```

## Results & Model Assessment

### Slip Distribution

Now, we visualize the resulting slip distribution on the fault.

```python
SlipVisualizer.plot_slip_components(
    fault=fault, 
    slip_vector=inverted_slip, 
    cmap='RdBu_r', 
    plot_edges=False,
    elev=10, 
    azim=180,
)
plt.tight_layout()
plt.show()
```

### Residual Analysis

Finally, we assess the model fit by analyzing the residuals (Observed - Predicted) for each dataset. A good model should result in residuals that look like random noise. If a coherent signal remains in the residuals, it might indicate that the model is missing some fault complexity, or that there is atmospheric noise in the InSAR data. In this specific example the high residuals centered at $\sim(40, 80)$ is the displacement due to slip along not modeled antithetic fault.

```python
# Calculate and plot residuals for each dataset
for ds in [alos_des, alos_ase, s1_ase]:
    # Calculate predicted data for this dataset
    g_matrix = engine.build_kernel(fault, ds)
    predicted_data = g_matrix @ inverted_slip
    
    # Plot the data fit and residuals
    SarDataFitVisualizer.plot_data_fit(
        observed_data=ds,
        predicted_data=predicted_data,
        fault=fault
    )
```
