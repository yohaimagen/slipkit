import numpy as np
from slipkit.core.fault import TriangularFaultMesh
from slipkit.core.data import GeodeticDataSet
from slipkit.core.physics import CutdeCpuEngine
from slipkit.core.solvers import BoundedLsqSolver
from slipkit.core.inversion import InversionOrchestrator

def create_simple_fault():
    """
    Creates a simple dipping rectangular fault made of two triangles.
    The fault dips at 45 degrees.
    """
    vertices = np.array([
        [-1000., -1000., -1000.],
        [ 1000., -1000., -1000.],
        [-1000.,  1000., -3000.],
        [ 1000.,  1000., -3000.]
    ])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    
    # Let's make it a right-lateral normal fault for demonstration
    return TriangularFaultMesh(
        mesh_input=(vertices, faces),
    )

def main():
    """
    Runs a full example of a simple slip inversion.
    """
    # 1. Create the fault model
    fault = create_simple_fault()
    n_patches = fault.num_patches()

    # 2. Define a grid of observation points on the surface (z=0)
    grid_res = 1000
    x_coords = np.arange(-4000, 4001, grid_res)
    y_coords = np.arange(-4000, 4001, grid_res)
    xv, yv = np.meshgrid(x_coords, y_coords)
    obs_pts = np.vstack([xv.flatten(), yv.flatten(), np.zeros_like(xv.flatten())]).T
    n_obs = obs_pts.shape[0]

    # For this example, we'll assume we are observing vertical displacement
    unit_vecs = np.array([[0., 0., 1.]] * n_obs)

    # 3. Create a synthetic "true" slip distribution
    # The slip vector is ordered [strike_slip_vals..., dip_slip_vals...]
    true_slip = np.zeros(2 * n_patches)
    # Assign 1.0m of right-lateral slip to both patches
    true_slip[:n_patches] = 1.0
    # Assign 0.5m of normal slip to both patches
    true_slip[n_patches:] = 0.5

    # 4. Generate synthetic "true" displacement data (forward model)
    # We need a dummy dataset to build the Green's function
    dummy_dataset = GeodeticDataSet(coords=obs_pts, data=np.zeros(n_obs), unit_vecs=unit_vecs, sigma=np.ones(n_obs))
    
    engine = CutdeCpuEngine(poisson_ratio=0.25)
    G = engine.build_kernel(fault, dummy_dataset)
    
    # Calculate true displacements: d = G * m
    true_disp = G @ true_slip

    # 5. Add Gaussian noise to create the final "observed" data
    noise_std = 0.01  # 1 cm of noise
    noise = np.random.normal(0, noise_std, size=n_obs)
    observed_disp = true_disp + noise

    # Create the final dataset for the inversion
    dataset = GeodeticDataSet(
        name="synthetic_data",
        coords=obs_pts,
        data=observed_disp,
        unit_vecs=unit_vecs,
        sigma=np.full(n_obs, noise_std)
    )

    # 6. Set up the inversion
    inversion = InversionOrchestrator()
    inversion.add_fault(fault)
    inversion.add_data(dataset)
    inversion.set_engine(engine)
    inversion.set_solver(BoundedLsqSolver()) # Use a solver that supports bounds

    # Define bounds for the slip. Since we simulated positive slip,
    # we'll constrain the inversion to only find positive slip values.
    # This is a common practice when the sense of slip is known.
    lower_bounds = np.zeros(2 * n_patches)
    upper_bounds = np.full(2 * n_patches, np.inf) # No upper limit
    bounds = (lower_bounds, upper_bounds)

    # 7. Run the inversion
    # The regularization parameter `lambda_spatial` smooths the slip distribution.
    # For a simple 2-patch fault, smoothing is not critical, but we include it.
    lambda_spatial = 0.05
    result = inversion.run_inversion(lambda_spatial=lambda_spatial, bounds=bounds)
    inverted_slip = result.slip_vector

    # 8. Report the results
    print("--- Slip Inversion Example ---")
    print(f"Number of fault patches: {n_patches}")
    print(f"Number of observation points: {n_obs}")
    print(f"Noise level (std dev): {noise_std} m")
    print(f"Regularization lambda: {lambda_spatial}
")

    print("Slip components are ordered as [strike_slip..., dip_slip...]")
    # Using np.round for cleaner output
    print(f"True slip vector:
{np.round(true_slip, 3)}")
    print(f"Inverted slip vector:
{np.round(inverted_slip, 3)}
")

    # Calculate the L2 norm of the misfit
    misfit = np.linalg.norm(inverted_slip - true_slip)
    print(f"L2 misfit between true and inverted slip: {misfit:.4f}")

if __name__ == "__main__":
    main()
