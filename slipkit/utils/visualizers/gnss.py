import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple
from slipkit.core.data import GeodeticDataSet
from slipkit.core.fault import AbstractFaultModel

class GnssVisualizer:
    """
    Visualizer for GNSS displacement data.
    """

    @staticmethod
    def plot_gnss_vectors(
        dataset: GeodeticDataSet,
        scale: float = 1.0,
        quiver_kwargs: Optional[dict] = None,
        scatter_kwargs: Optional[dict] = None,
        cmap: str = "RdYlBu_r",
        fig_width: float = 8.0,
        fig_height: float = 6.0,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        show_colorbar: bool = True,
        return_fig_ax: bool = False
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """
        Plots GNSS horizontal displacement vectors and vertical displacement as colored points.

        The dataset is expected to be a stacked GeodeticDataSet containing E, N, and U components
        in that order (as produced by GnssParser.read_csv).

        Args:
            dataset: The GeodeticDataSet containing the GNSS data.
            scale: Scale factor for the quiver vectors.
            quiver_kwargs: Additional arguments for plt.quiver.
            scatter_kwargs: Additional arguments for plt.scatter.
            cmap: Colormap for the vertical (Up) displacement.
            fig_width: Width of the figure if ax is None.
            fig_height: Height of the figure if ax is None.
            title: Plot title.
            ax: Existing matplotlib axis to plot on.
            show_colorbar: Whether to show the colorbar for the vertical component.
            return_fig_ax: If True, returns (fig, ax).

        Returns:
            Optional[Tuple[plt.Figure, plt.Axes]]: If return_fig_ax is True.
        """
        # Determine number of stations (assume E, N, U are all present)
        # GnssParser stacks them as E, then N, then U.
        total_n = len(dataset)
        num_stations = total_n // 3
        
        if total_n % 3 != 0:
             # If not exactly 3 components, we try to deduce from unit vectors
             # or just take the first N unique coordinates.
             # But for standard use, we expect 3.
             # Let's find unique coordinates to be safe.
             _, unique_indices = np.unique(dataset.coords, axis=0, return_index=True)
             num_stations = len(unique_indices)
             # This might be slow for many points, but GNSS usually has few.
        
        # Extract E, N, U
        # We need to map the data points back to their stations.
        # Since GnssParser.read_csv stacks them:
        # E is at 0:num_stations
        # N is at num_stations:2*num_stations
        # U is at 2*num_stations:3*num_stations
        
        x = dataset.coords[0:num_stations, 0]
        y = dataset.coords[0:num_stations, 1]
        
        # We search for the components based on unit vectors to be robust
        e_mask = np.all(np.isclose(dataset.unit_vecs, [1, 0, 0]), axis=1)
        n_mask = np.all(np.isclose(dataset.unit_vecs, [0, 1, 0]), axis=1)
        u_mask = np.all(np.isclose(dataset.unit_vecs, [0, 0, 1]), axis=1)
        
        de = dataset.data[e_mask]
        dn = dataset.data[n_mask]
        du = dataset.data[u_mask]
        
        # Ensure we have consistent lengths
        n_pts = min(len(de), len(dn), len(du))
        if n_pts == 0:
            raise ValueError("Dataset does not contain all E, N, U components required for this plot.")
            
        # Coordinates for these points (take from E mask)
        coords_e = dataset.coords[e_mask][:n_pts]
        x = coords_e[:, 0]
        y = coords_e[:, 1]
        de = de[:n_pts]
        dn = dn[:n_pts]
        du = du[:n_pts]

        if ax is None:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        else:
            fig = ax.get_figure()

        # 1. Plot Vertical Component as scatter
        skwargs = {"s": 50, "edgecolors": "k", "zorder": 2}
        if scatter_kwargs:
            skwargs.update(scatter_kwargs)
        
        sc = ax.scatter(x, y, c=du, cmap=cmap, **skwargs)
        
        if show_colorbar:
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("Vertical Displacement (m)")

        # 2. Plot Horizontal Vectors
        qkwargs = {"angles": "xy", "scale_units": "xy", "scale": 1.0/scale, "width": 0.005, "zorder": 3}
        if quiver_kwargs:
            qkwargs.update(quiver_kwargs)
            
        q = ax.quiver(x, y, de, dn, **qkwargs)
        
        # Add a reference vector if possible
        # ax.quiverkey(q, 0.9, 0.1, 0.01, r'$1 cm$', labelpos='E', coordinates='figure')

        ax.set_aspect("equal")
        ax.set_xlabel("East (km)")
        ax.set_ylabel("North (km)")
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"GNSS Displacements: {dataset.name}")

        if return_fig_ax:
            return fig, ax
        
        plt.show()
