# slipkit/utils/visualizers/forward_model.py
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, List, Tuple, Union
from slipkit.core.data import GeodeticDataSet
from slipkit.core.fault import AbstractFaultModel
from slipkit.core.physics import GreenFunctionBuilder


class ForwardModelVisualizer:
    """
    A visualizer for performing and plotting forward modeling "sanity checks".
    """

    @staticmethod
    def _get_fault_outline_segments(fault_mesh: AbstractFaultModel) -> List[np.ndarray]:
        """
        Extracts the unique outline segments (edges) of a triangular fault mesh.
        (Copied from SarDataFitVisualizer for reusability)

        Parameters
        ----------
        fault_mesh : AbstractFaultModel
            The fault mesh object, expected to be a TriangularFaultMesh or similar
            that provides vertices and faces.

        Returns
        -------
        List[np.ndarray]
            A list of 2-point numpy arrays, where each array represents a segment
            (e.g., [[x1, y1], [x2, y2]]).
        """
        verts, faces = fault_mesh.get_mesh_geometry()
        edge_counts = {}

        # Count how many times each edge appears
        for face in faces:
            # Get edges for current face, sort vertex indices to treat (v1,v2) and (v2,v1) as same
            edges = [
                tuple(sorted((face[0], face[1]))),
                tuple(sorted((face[1], face[2]))),
                tuple(sorted((face[2], face[0]))),
            ]
            for edge in edges:
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        
        outline_segments = []
        for edge, count in edge_counts.items():
            if count == 1:  # Edges that appear only once are on the boundary
                v1, v2 = edge
                outline_segments.append(np.array([verts[v1, :2], verts[v2, :2]]))
        
        return outline_segments

    @staticmethod
    def plot_enu_response(
        fault: Union[AbstractFaultModel, List[AbstractFaultModel]],
        slip_vector: np.ndarray,
        engine: GreenFunctionBuilder,
        x_extent: Tuple[float, float],
        y_extent: Tuple[float, float],
        grid_resolution: int = 100,
        title: str = "Surface Displacement Patterns",
        figsize: Tuple[float, float] = (15, 5),
        cmap: str = 'coolwarm',
        levels: int = 20,
        fault_color: str = 'k',
        fault_alpha: float = 0.5,
        fault_linewidth: float = 0.5,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        return_fig_ax: bool = False,
        **kwargs
    ):
        """
        Plots East, North, and Up displacement components for a given fault slip.

        This function serves as a "sanity check" to visualize the expected surface
        deformation pattern from a synthetic fault slip.

        Parameters
        ----------
        fault : Union[AbstractFaultModel, List[AbstractFaultModel]]
            The fault model or a list of fault models.
        slip_vector : np.ndarray
            A 1D numpy array representing the slip on the fault patches.
        engine : GreenFunctionBuilder
            The physics engine used to calculate Green's functions.
        x_extent : Tuple[float, float]
            Tuple (min_x, max_x) defining the extent of the observation grid in meters.
        y_extent : Tuple[float, float]
            Tuple (min_y, max_y) defining the extent of the observation grid in meters.
        grid_resolution : int, optional
            Number of points along each axis of the observation grid. Default is 100.
        title : str, optional
            Overall title for the figure. Default is "Surface Displacement Patterns".
        figsize : Tuple[float, float], optional
            Figure size (width, height) in inches. Default is (15, 5).
        cmap : str, optional
            Colormap for the displacement plots. Default is 'coolwarm'.
        levels : int, optional
            Number of contour levels. Default is 20.
        fault_color : str, optional
            Color for the fault trace. Default is 'k' (black).
        fault_alpha : float, optional
            Transparency for the fault trace. Default is 0.5.
        fault_linewidth : float, optional
            Linewidth for the fault trace. Default is 0.5.
        vmin : Optional[float]
            Minimum value for the colormap. If None, calculated automatically (5th percentile).
        vmax : Optional[float]
            Maximum value for the colormap. If None, calculated automatically (95th percentile).
        return_fig_ax : bool, optional
            If True, returns the matplotlib figure and axes objects. Default is False.
        **kwargs
            Additional keyword arguments passed to plt.contourf.

        Returns
        -------
        Optional[Tuple[plt.Figure, np.ndarray]]
            A tuple of (Figure, Axes) if `return_fig_ax` is True.
        """
        x = np.linspace(x_extent[0], x_extent[1], grid_resolution)
        y = np.linspace(y_extent[0], y_extent[1], grid_resolution)
        xx, yy = np.meshgrid(x, y)
        
        # Observation points at z=0 (surface)
        obs_coords = np.vstack([xx.ravel(), yy.ravel(), np.zeros(grid_resolution * grid_resolution)]).T.copy()
        
        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
        fig.suptitle(title, fontsize=16)

        components_data = ['East', 'North', 'Up']
        unit_vectors_data = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
        
        for i, (comp, vec) in enumerate(zip(components_data, unit_vectors_data)):
            unit_vecs_comp = np.zeros_like(obs_coords)
            unit_vecs_comp[:] = vec # Assign unit vector for this component
            
            # Create a dummy GeodeticDataSet for Green's function calculation
            dataset = GeodeticDataSet(
                name=f"Synthetic_{comp}",
                coords=obs_coords,
                data=np.zeros(len(obs_coords)), # Data values are not used for forward modeling
                unit_vecs=unit_vecs_comp,
                sigma=np.ones(len(obs_coords)), # Sigma values are not used
            )
            
            # Calculate Green's function matrix for the current dataset
            # If there are multiple faults, concatenate their G matrices
            g_matrix_parts = []
            faults_for_kernel = fault if isinstance(fault, list) else [fault]
            for f_item in faults_for_kernel:
                g_matrix_parts.append(engine.build_kernel(f_item, dataset))
            g_matrix = np.concatenate(g_matrix_parts, axis=1)
            
            # Compute displacement for the given slip vector
            displacement = g_matrix @ slip_vector

            # Reshape displacement for contour plotting
            disp_grid = displacement.reshape(grid_resolution, grid_resolution)

            ax = axes[i]
            
            # Determine vmin/vmax for current subplot
            current_vmin, current_vmax = vmin, vmax
            if current_vmin is None or current_vmax is None:
                # Calculate 5th and 95th percentiles of the absolute displacement
                data_range = np.abs(displacement[~np.isnan(displacement)])
                if len(data_range) > 0:
                    current_vmin = -np.percentile(data_range, 95)
                    current_vmax = np.percentile(data_range, 95)
                else:
                    current_vmin, current_vmax = -0.1, 0.1 # Default small range if no valid data


            sc = ax.contourf(
                xx, yy, disp_grid, 
                cmap=cmap, 
                levels=levels, 
                vmin=current_vmin,
                vmax=current_vmax,
                **kwargs
            )
            fig.colorbar(sc, ax=ax, label='Displacement (m)')
            ax.set_title(f"{comp} Component")
            ax.set_xlabel("East (m)")
            ax.set_ylabel("North (m)")
            ax.set_aspect('equal', 'box')
            
            # Plot fault outlines
            if fault is not None:
                faults_to_plot = fault if isinstance(fault, list) else [fault]
                for f in faults_to_plot:
                    outline_segments = ForwardModelVisualizer._get_fault_outline_segments(f)
                    for segment in outline_segments:
                        ax.plot(
                            segment[:, 0],
                            segment[:, 1],
                            color=fault_color,
                            alpha=fault_alpha,
                            linewidth=fault_linewidth,
                            zorder=10,
                        )
        
        if return_fig_ax:
            return fig, axes
        
        plt.show()