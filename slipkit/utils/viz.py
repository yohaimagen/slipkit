import matplotlib.pyplot as plt
import numpy as np

from slipkit.core.fault import TriangularFaultMesh
from slipkit.core.data import GeodeticDataSet
from slipkit.core.physics import CutdeCpuEngine

class GreenFunctionVisualizer:
    """
    A visualization tool for sanity-checking Green's functions.
    """

    @staticmethod
    def plot_response(
        fault: TriangularFaultMesh,
        dataset: GeodeticDataSet,
        patch_idx: int,
        slip_component: str,
        engine: CutdeCpuEngine,
        ax=None,
        save_to: str = None,
    ):
        """
        Plots the displacement response for a single patch and slip component.

        Args:
            fault: The fault model.
            dataset: The geodetic dataset with observation points.
            patch_idx: The index of the fault patch to visualize.
            slip_component: The slip component, either 'strike' or 'dip'.
            engine: The physics engine to compute the Green's function.
            ax: A matplotlib axes object to plot on. If None, a new one is created.
            save_to: If not None, the path to save the figure to.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        n_patches = fault.num_patches()
        if not 0 <= patch_idx < n_patches:
            raise ValueError(f"patch_idx must be between 0 and {n_patches - 1}.")

        g_matrix = engine.build_kernel(fault, dataset)

        if slip_component == 'strike':
            displacements = g_matrix[:, patch_idx]
        elif slip_component == 'dip':
            displacements = g_matrix[:, n_patches + patch_idx]
        else:
            raise ValueError("slip_component must be 'strike' or 'dip'.")

        x = dataset.coords[:, 0]
        y = dataset.coords[:, 1]
        
        sc = ax.scatter(x, y, c=displacements, cmap='coolwarm', s=15)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(f'Projected displacement (m)')

        # Plot fault patch projection on the xy plane
        verts, _ = fault.get_mesh_geometry()
        fault_patch = verts[fault.faces[patch_idx]]
        
        # Draw the patch edges
        for i in range(3):
            ax.plot(
                [fault_patch[i, 0], fault_patch[(i + 1) % 3, 0]],
                [fault_patch[i, 1], fault_patch[(i + 1) % 3, 1]],
                'k-', lw=2
            )

        ax.set_title(f'Displacement Field for Patch {patch_idx} ({slip_component}-slip)')
        ax.set_xlabel('X coordinate (m)')
        ax.set_ylabel('Y coordinate (m)')
        ax.set_aspect('equal', 'box')
        ax.grid(True, linestyle='--', alpha=0.6)

        if save_to:
            fig.savefig(save_to, bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        return ax

    @staticmethod
    def plot_total_response(
        fault: TriangularFaultMesh,
        dataset: GeodeticDataSet,
        slip_distribution: np.ndarray,
        engine: CutdeCpuEngine,
        ax=None,
        save_to: str = None,
    ):
        """
        Plots the total displacement response from the entire fault.

        Args:
            fault: The fault model.
            dataset: The geodetic dataset.
            slip_distribution: A (2M,) numpy array with strike-slip values
                               followed by dip-slip values.
            engine: The physics engine.
            ax: A matplotlib axes object. If None, one is created.
            save_to: Path to save the figure to.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

        n_patches = fault.num_patches()
        if slip_distribution.shape != (2 * n_patches,):
            raise ValueError(f"slip_distribution must have shape ({2 * n_patches},).")

        g_matrix = engine.build_kernel(fault, dataset)
        total_displacement = g_matrix @ slip_distribution

        x = dataset.coords[:, 0]
        y = dataset.coords[:, 1]

        sc = ax.scatter(x, y, c=total_displacement, cmap='coolwarm', s=15)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('Total Projected Displacement (m)')

        # Plot the entire fault geometry
        verts, faces = fault.get_mesh_geometry()
        for face in faces:
            fault_patch = verts[face]
            for i in range(3):
                ax.plot(
                    [fault_patch[i, 0], fault_patch[(i + 1) % 3, 0]],
                    [fault_patch[i, 1], fault_patch[(i + 1) % 3, 1]],
                    'k-', lw=1.5
                )

        ax.set_title('Total Displacement Field from All Patches')
        ax.set_xlabel('X coordinate (m)')
        ax.set_ylabel('Y coordinate (m)')
        ax.set_aspect('equal', 'box')
        ax.grid(True, linestyle='--', alpha=0.6)

        if save_to:
            fig.savefig(save_to, bbox_inches='tight', dpi=300)
            plt.close(fig)

        return ax

