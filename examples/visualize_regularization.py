import matplotlib.pyplot as plt
import numpy as np

from slipkit.core.fault import TriangularFaultMesh
from slipkit.utils.visualizers import RegularizationVisualizer

def create_simple_grid_mesh(nx, ny, dz=0.0):
    """
    Creates a simple rectangular grid of triangular fault patches for demonstration.

    Args:
        nx: Number of divisions along the x-axis.
        ny: Number of divisions along the y-axis.
        dz: Constant z-coordinate (depth) for all vertices.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Vertices and faces of the mesh.
    """
    verts = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            verts.append([float(i), float(j), dz])
    verts = np.array(verts)

    faces = []
    for j in range(ny):
        for i in range(nx):
            bl_idx = j * (nx + 1) + i
            v0 = bl_idx
            v1 = bl_idx + 1
            v2 = bl_idx + (nx + 1) + 1
            v3 = bl_idx + (nx + 1)

            # Two triangles per square
            faces.append([v0, v1, v2]) # Triangle 1
            faces.append([v0, v2, v3]) # Triangle 2
            
    faces = np.array(faces)
    return verts, faces

if __name__ == "__main__":
    # 1. Create a dummy fault mesh
    # Let's create a 2x2 grid, which will result in 8 triangular patches
    print("Creating a 2x2 grid fault mesh (8 patches)...")
    verts, faces = create_simple_grid_mesh(nx=2, ny=2, dz=0.0)
    fault_mesh = TriangularFaultMesh(mesh_input=(verts, faces))
    print(f"Mesh created with {fault_mesh.num_patches()} patches.")

    # 2. Choose a target patch index to visualize
    target_idx = 3 # Choose an arbitrary patch (e.g., one in the middle if possible)
    print(f"Visualizing neighbors for target patch index: {target_idx}")

    # 3. Plot the fault neighbors
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    RegularizationVisualizer.plot_fault_neighbors(
        fault=fault_mesh,
        target_patch_idx=target_idx,
        ax=ax,
        color_target='red',
        color_neighbors='blue',
        color_others='lightgray',
        # save_to="fault_neighbors_example.png" # Uncomment to save the figure
    )

    plt.show()

    print("\nExample finished. If 'fault_neighbors_example.png' was specified, it has been saved.")
    print("A plot window should have appeared showing the fault mesh with highlighted patches.")