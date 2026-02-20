# slipkit/utils/visualizers/data_fit.py
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, List, Union
from slipkit.core.data import GeodeticDataSet
from slipkit.core.fault import AbstractFaultModel

class SarDataFitVisualizer:
    """
    A visualizer for comparing observed geodetic data with predicted model outputs
    and visualizing residuals.
    """

    @staticmethod
    def _get_fault_outline_segments(fault_mesh: AbstractFaultModel) -> List[np.ndarray]:
        """
        Extracts the unique outline segments (edges) of a triangular fault mesh.

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
    def plot_data_fit(
        observed_data: GeodeticDataSet,
        predicted_data: np.ndarray,
        fault: Optional[Union[AbstractFaultModel, List[AbstractFaultModel]]] = None,
        fig_width: float = 15.0,
        fig_height: float = 5.0,
        cmap: str = 'viridis',
        scatter_size: float = 5.0,
        fault_color: str = 'r',
        fault_alpha: float = 0.5,
        fault_linewidth: float = 1.5,
        ax_titles: Optional[List[str]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        return_fig_ax: bool = False,
        **kwargs
    ):
        """
        Plots observed data, predicted data, and residuals side-by-side.

        Parameters
        ----------
        observed_data : GeodeticDataSet
            The observed geodetic dataset.
        predicted_data : np.ndarray
            The predicted displacement values from the model. Should have the
            same shape as observed_data.data.
        fault : Optional[Union[AbstractFaultModel, List[AbstractFaultModel]]]
            An optional fault model or list of fault models to plot the fault trace on the maps.
        fig_width : float
            Width of the matplotlib figure.
        fig_height : float
            Height of the matplotlib figure.
        cmap : str
            Colormap to use for the scatter plots.
        scatter_size : float
            Size of the scatter points.
        fault_color : str
            Color for the fault trace.
        fault_alpha : float
            Transparency for the fault trace.
        fault_linewidth : float
            Linewidth for the fault trace.
        ax_titles : Optional[List[str]]
            Custom titles for the subplots: [Observed, Predicted, Residuals].
            If None, default titles are used.
        vmin : Optional[float]
            Minimum value for the colormap. If None, calculated automatically (5th percentile).
        vmax : Optional[float]
            Maximum value for the colormap. If None, calculated automatically (95th percentile).
        return_fig_ax : bool
            If True, returns the matplotlib figure and axes objects.
        **kwargs
            Additional keyword arguments passed to plt.scatter.

        Returns
        -------
        Optional[Tuple[plt.Figure, np.ndarray]]
            A tuple of (Figure, Axes) if `return_fig_ax` is True.
        """
        if observed_data.data.shape != predicted_data.shape:
            raise ValueError(
                "Shape of observed_data.data and predicted_data must match."
            )

        residuals = observed_data.data - predicted_data

        fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), constrained_layout=True)

        plots = [
            (observed_data.data, ax_titles[0] if ax_titles else "Observed Data"),
            (predicted_data, ax_titles[1] if ax_titles else "Predicted Data"),
            (residuals, ax_titles[2] if ax_titles else "Residuals"),
        ]

        # Determine a common colorbar range for observed and predicted data
        vmax_obs_pred = max(
            np.nanmax(np.abs(observed_data.data)), np.nanmax(np.abs(predicted_data))
        )
        vmin_obs_pred = -vmax_obs_pred

        # Determine a common colorbar range for residuals
        vmax_res = np.nanmax(np.abs(residuals))
        vmin_res = -vmax_res


        for i, (data_to_plot, title) in enumerate(plots):
            ax = axes[i]
            
            # Determine vmin/vmax for current subplot
            if vmin is None or vmax is None:
                # Calculate 5th and 95th percentiles if vmin/vmax not provided
                # For observed and predicted, use absolute values for symmetric colormap
                if i < 2:
                    data_range = np.abs(data_to_plot[~np.isnan(data_to_plot)])
                else: # For residuals, use raw values
                    data_range = data_to_plot[~np.isnan(data_to_plot)]

                if len(data_range) > 0:
                    current_vmin = np.percentile(data_range, 5) if i < 2 else np.percentile(data_range, 5)
                    current_vmax = np.percentile(data_range, 95) if i < 2 else np.percentile(data_range, 95)
                else:
                    current_vmin, current_vmax = -0.1, 0.1 # Default small range if no valid data
            else:
                current_vmin, current_vmax = vmin, vmax


            sc = ax.scatter(
                observed_data.coords[:, 0],
                observed_data.coords[:, 1],
                c=data_to_plot,
                cmap=cmap,
                s=scatter_size,
                vmin=current_vmin,
                vmax=current_vmax,
                **kwargs
            )
            fig.colorbar(sc, ax=ax, label="Displacement (m)")
            ax.set_title(title)
            ax.set_xlabel("East (m)")
            ax.set_ylabel("North (m)")
            ax.set_aspect('equal', 'box')

            if fault is not None:
                faults_to_plot = fault if isinstance(fault, list) else [fault]
                for f in faults_to_plot:
                    outline_segments = SarDataFitVisualizer._get_fault_outline_segments(f)
                    for segment in outline_segments:
                        ax.plot(
                            segment[:, 0],
                            segment[:, 1],
                            color=fault_color,
                            alpha=fault_alpha,
                            linewidth=fault_linewidth,
                            zorder=10,
                        )
        
        fig.suptitle(f"Data Fit for {observed_data.name}", fontsize=16)

        if return_fig_ax:
            return fig, axes
        
        plt.show()
