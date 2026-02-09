import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # This is the correct import for 3D plotting

from slipkit.core.fault import TriangularFaultMesh

class RegularizationVisualizer:
    """
    A visualization tool for regularization-related aspects of fault models.
    """

    @staticmethod
    def plot_fault_neighbors(
        fault: TriangularFaultMesh,
        target_patch_idx: int,
        ax=None,
        color_target: str = 'red',
        color_neighbors: str = 'blue',
        color_others: str = 'lightgray',
        save_to: str = None,
    ):
        """
        Plots a 3D fault plane, highlighting a target patch and its neighbors.

        Args:
            fault: The TriangularFaultMesh object.
            target_patch_idx: The index of the patch to highlight as the target.
            ax: A matplotlib 3D axes object to plot on. If None, a new one is created.
            color_target: Color for the target patch.
            color_neighbors: Color for the neighboring patches.
            color_others: Color for all other patches.
            save_to: If not None, the path to save the figure to.
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.get_figure()

        n_patches = fault.num_patches()
        if not 0 <= target_patch_idx < n_patches:
            raise ValueError(
                f"target_patch_idx must be between 0 and {n_patches - 1}."
            )

        # Get mesh geometry
        verts, faces = fault.get_mesh_geometry()

        # Get the adjacency information from the smoothing matrix
        # The Laplacian matrix has non-zero entries (excluding diagonal) for neighbors
        smoothing_matrix = fault.get_smoothing_matrix(type='laplacian')
        
        # Find neighbors of the target patch
        # In a Laplacian matrix L, L[i,j] != 0 if i and j are neighbors (for i != j)
        neighbors_indices = set()
        # Find column indices where row `target_patch_idx` has non-zero values (excluding itself)
        row_data = smoothing_matrix.getrow(target_patch_idx)
        for col_idx in row_data.indices:
            if col_idx != target_patch_idx:
                neighbors_indices.add(col_idx)

        # Assign colors to each patch
        patch_colors = []
        for i in range(n_patches):
            if i == target_patch_idx:
                patch_colors.append(color_target)
            elif i in neighbors_indices:
                patch_colors.append(color_neighbors)
            else:
                patch_colors.append(color_others)

        # Plot each triangular patch
        for i, face in enumerate(faces):
            patch_verts = verts[face]
            ax.plot_trisurf(
                patch_verts[:, 0],
                patch_verts[:, 1],
                patch_verts[:, 2],
                triangles=[[0, 1, 2]],
                color=patch_colors[i],
                edgecolor='k',
                linewidth=0.5,
                alpha=0.8,
            )
        
        ax.set_title(f'Fault Plane with Target Patch {target_patch_idx} and Neighbors')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_aspect('auto') # or 'equal' if desired, but 'auto' is usually better for 3D

        # Set equal aspect ratio for 3D plot
        max_range = np.array([verts[:,0].max()-verts[:,0].min(), 
                              verts[:,1].max()-verts[:,1].min(), 
                              verts[:,2].max()-verts[:,2].min()]).max() / 2.0
        
        mid_x = (verts[:,0].max()+verts[:,0].min()) * 0.5
        mid_y = (verts[:,1].max()+verts[:,1].min()) * 0.5
        mid_z = (verts[:,2].max()+verts[:,2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)


        if save_to:
            fig.savefig(save_to, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        return ax

