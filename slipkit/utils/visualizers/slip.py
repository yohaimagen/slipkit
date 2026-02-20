import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from slipkit.core.fault import TriangularFaultMesh


# Assuming TriangularFaultMesh is already defined as in your snippet

class SlipVisualizer:
    """
    A visualization tool for plotting slip distributions on fault meshes.
    """

    @staticmethod
    def plot_slip_components(
        fault: TriangularFaultMesh,
        slip_vector: np.ndarray,
        figsize: tuple = (16, 8),
        cmap: str = 'viridis',
        plot_edges: bool = True,
        elev: float = 30,  # Default elevation angle
        azim: float = -60, # Default azimuth angle
        save_to: str = None,
    ):
        """
        Plots the Strike-Slip and Dip-Slip components side-by-side on 3D axes.

        Args:
            fault: The TriangularFaultMesh object.
            slip_vector: A 1D numpy array of length 2 * n_patches. 
                         Structure must be [strike_slip_1...N, dip_slip_1...N].
            figsize: Tuple for figure dimensions.
            cmap: Matplotlib colormap name (e.g., 'viridis', 'plasma', 'seismic').
            plot_edges: If True, draws faint lines for triangle edges.
            save_to: If not None, saves the figure to this path.
        """
        n_patches = fault.num_patches()
        
        # Validation
        if slip_vector.shape[0] != 2 * n_patches:
            raise ValueError(
                f"Slip vector length ({len(slip_vector)}) does not match "
                f"2 * num_patches ({2 * n_patches}). "
                "Ensure vector format is [SS_components, DS_components]."
            )

        # Decompose vector
        ss_slip = slip_vector[:n_patches]
        ds_slip = slip_vector[n_patches:]

        # Create Figure
        fig = plt.figure(figsize=figsize)
        
        # --- Plot 1: Strike-Slip ---
        ax1 = fig.add_subplot(121, projection='3d')
        SlipVisualizer._plot_mesh_on_ax(
            ax1, fault, ss_slip, "Strike-Slip Distribution", cmap, plot_edges, elev, azim
        )

        # --- Plot 2: Dip-Slip ---
        ax2 = fig.add_subplot(122, projection='3d')
        SlipVisualizer._plot_mesh_on_ax(
            ax2, fault, ds_slip, "Dip-Slip Distribution", cmap, plot_edges, elev, azim
        )

        plt.tight_layout()

        if save_to:
            plt.savefig(save_to, bbox_inches='tight', dpi=300)
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def _plot_mesh_on_ax(ax, fault, data, title, cmap_name, plot_edges, elev, azim):
        """Helper to render the colored mesh on a specific axis."""
        verts, faces = fault.get_mesh_geometry()
        
        # Prepare Geometry: shape (M, 3, 3) -> M triangles, 3 corners, 3 coords
        # This allows us to use Poly3DCollection which is much faster than loop plotting
        triangles = verts[faces]

        # Normalize data for colormap
        norm = mcolors.Normalize(vmin=np.min(data), vmax=np.max(data))
        cmap = plt.get_cmap(cmap_name)
        
        # Create colors for each face based on the data value
        # facecolors expects an RGBA tuple per face
        face_colors = cmap(norm(data))

        # Create the collection
        mesh_poly = Poly3DCollection(
            triangles,
            facecolors=face_colors,
            edgecolors='k' if plot_edges else None,
            linewidths=0.1 if plot_edges else 0,
            alpha=0.9
        )
        
        ax.add_collection3d(mesh_poly)

        # Set initial viewing angle
        ax.view_init(elev=elev, azim=azim)

        # Auto-scaling logic (critical for geological meshes)
        # We must manually set limits because add_collection3d doesn't auto-update limits
        scale = np.concatenate([verts[:, 0], verts[:, 1], verts[:, 2]]).flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        
        # Better aspect ratio handling
        max_range = np.array([
            verts[:, 0].max() - verts[:, 0].min(),
            verts[:, 1].max() - verts[:, 1].min(),
            verts[:, 2].max() - verts[:, 2].min()
        ]).max() / 2.0

        mid_x = (verts[:, 0].max() + verts[:, 0].min()) * 0.5
        mid_y = (verts[:, 1].max() + verts[:, 1].min()) * 0.5
        mid_z = (verts[:, 2].max() + verts[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Labels
        ax.set_title(title)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        # Colorbar
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(data)
        plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Slip (m)')