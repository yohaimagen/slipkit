import numpy as np
import matplotlib.pyplot as plt
from slipkit.core.fault import TriangularFaultMesh
from slipkit.core.data import GeodeticDataSet
from slipkit.core.physics import CutdeCpuEngine
from slipkit.utils.viz import GreenFunctionVisualizer

def create_dipping_fault(dip_angle_deg: float, strike_length: float = 20.0, depth: float = 10.0):
    """Creates a dipping rectangular fault."""
    dip_rad = np.deg2rad(dip_angle_deg)
    y_half = strike_length / 2.0
    
    # Horizontal offset due to dip
    x_offset = depth / np.tan(dip_rad)
    
    vertices = np.array([
        [0, -y_half, 0],          # Top-left
        [x_offset, -y_half, -depth],  # Bottom-left
        [x_offset, y_half, -depth],   # Bottom-right
        [0, y_half, 0],           # Top-right
    ])
    faces = np.array([[0, 1, 3], [1, 2, 3]])
    return TriangularFaultMesh((vertices, faces))

def plot_enu_response(ax_row, title, fault, slip, engine, coords):
    """Plots E, N, U components for a given scenario on a row of axes."""
    components = ['East', 'North', 'Up']
    unit_vectors = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    
    for i, (comp, vec) in enumerate(zip(components, unit_vectors)):
        unit_vecs_comp = np.zeros_like(coords)
        unit_vecs_comp[:] = vec
        dataset = GeodeticDataSet(
            name=f"dataset_{comp}",
            coords=coords,
            data=np.zeros(len(coords)),
            unit_vecs=unit_vecs_comp,
            sigma=np.ones(len(coords)),
        )
        
        # Use a modified plotting logic here
        g_matrix = engine.build_kernel(fault, dataset)
        displacement = g_matrix @ slip

        xx = coords[:, 0].reshape(100, 100)
        yy = coords[:, 1].reshape(100, 100)
        disp_grid = displacement.reshape(100, 100)

        ax = ax_row[i]
        sc = ax.contourf(xx, yy, disp_grid, cmap='coolwarm', levels=20)
        fig.colorbar(sc, ax=ax, label='Displacement (m)')
        ax.set_title(f"{title}\n{comp} Component")
        ax.set_aspect('equal', 'box')
        
        # Plot fault trace
        verts, _ = fault.get_mesh_geometry()
        ax.plot(verts[[0, 3], 0], verts[[0, 3], 1], 'k-', lw=2)


# --- Main Script ---
if __name__ == "__main__":
    # 1. Setup plotting grid
    fig, axes = plt.subplots(4, 3, figsize=(15, 20), constrained_layout=True)
    fig.suptitle("Surface Displacement Patterns for Different Faulting Styles", fontsize=16)

    # 2. Setup shared components
    engine = CutdeCpuEngine(poisson_ratio=0.25)
    n_x, n_y = 100, 100
    x = np.linspace(-30, 30, n_x)
    y = np.linspace(-30, 30, n_y)
    xx, yy = np.meshgrid(x, y)
    obs_coords = np.vstack([xx.ravel(), yy.ravel(), np.zeros(n_x * n_y)]).T.copy()

    # --- Define Fault Geometries ---
    fault_vertical = TriangularFaultMesh((
        np.array([[0, -10, 0], [0, -10, -10], [0, 10, -10], [0, 10, 0]]),
        np.array([[0, 1, 3], [1, 2, 3]])
    ))
    fault_dipping = create_dipping_fault(dip_angle_deg=60)

    # --- Scenarios ---

    # Scenario 1: Right-lateral strike-slip (Vertical Fault)
    # Following user's convention: positive for right-lateral strike-slip
    slip_right_lateral = np.array([1.0, 1.0, 0.0, 0.0])
    plot_enu_response(axes[0], "Right-Lateral Strike-Slip", fault_vertical, slip_right_lateral, engine, obs_coords)

    # Scenario 2: Left-lateral strike-slip (Vertical Fault)
    # Following user's convention: negative for left-lateral strike-slip
    slip_left_lateral = np.array([-1.0, -1.0, 0.0, 0.0])
    plot_enu_response(axes[1], "Left-Lateral Strike-Slip", fault_vertical, slip_left_lateral, engine, obs_coords)

    # Scenario 3: Normal Faulting (Dip=60)
    # Following user's convention: positive for normal faulting
    slip_normal = np.array([0.0, 0.0, 1.0, 1.0])
    plot_enu_response(axes[2], "Normal Faulting (Dip=60)", fault_dipping, slip_normal, engine, obs_coords)

    # Scenario 4: Reverse Faulting (Dip=60)
    # Following user's convention: negative for reverse faulting
    slip_reverse = np.array([0.0, 0.0, -1.0, -1.0])
    plot_enu_response(axes[3], "Reverse Faulting (Dip=60)", fault_dipping, slip_reverse, engine, obs_coords)
    
    # Save the final figure
    output_filename = "advanced_fault_visualization.png"
    fig.savefig(output_filename, dpi=300)
    print(f"Advanced visualization saved to {output_filename}")
    plt.close(fig)
