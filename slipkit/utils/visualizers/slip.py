import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection

from slipkit.core.fault import TriangularFaultMesh, SlipComponent


_COMPONENT_TITLE = {
    SlipComponent.STRIKE_SLIP: "Strike-Slip Distribution",
    SlipComponent.DIP_SLIP: "Dip-Slip Distribution",
}


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
        elev: float = 30,   # Default elevation angle
        azim: float = -60,  # Default azimuth angle
        xlim: tuple = None,
        ylim: tuple = None,
        zlim: tuple = None,
        box_aspect: tuple = None,
        z_exaggeration: float = 1.0,
        pad_frac: float = 0.05,
        vmin: float = None,
        vmax: float = None,
        save_to: str = None,
    ):
        """
        Plots each active slip component of the fault on side-by-side 3D axes.

        One panel is drawn per active component (so a strike-slip-only fault
        yields a single panel). By default the axes are tightly fitted to the
        fault geometry and the box is scaled to the *true* proportions of the
        mesh, so a long, shallow fault fills the frame instead of shrinking into
        a cube. Use ``zlim`` / ``z_exaggeration`` / ``box_aspect`` to control the
        depth view.

        Args:
            fault: The TriangularFaultMesh object.
            slip_vector: A 1D numpy array of length ``num_components * n_patches``,
                         laid out as consecutive per-component blocks in canonical
                         order (strike-slip before dip-slip).
            figsize: Tuple for figure dimensions.
            cmap: Matplotlib colormap name (e.g., 'viridis', 'plasma', 'seismic').
            plot_edges: If True, draws faint lines for triangle edges.
            elev: Camera elevation angle (degrees).
            azim: Camera azimuth angle (degrees).
            xlim: Optional (min, max) to override the X (along-strike) axis limits.
            ylim: Optional (min, max) to override the Y axis limits.
            zlim: Optional (min, max) to override the Z (depth) axis limits.
                  e.g. ``zlim=(-20, 0)`` to crop to the fault depth range.
            box_aspect: Optional (ax, ay, az) relative visual lengths passed to
                        ``ax.set_box_aspect``. If None, the true data extents are
                        used (multiplied by ``z_exaggeration`` on the z term).
            z_exaggeration: Vertical exaggeration factor applied to the default
                            box aspect (ignored if ``box_aspect`` is given).
            pad_frac: Fractional padding added around the data when limits are
                      auto-computed (ignored on any axis given an explicit limit).
            vmin: Optional lower bound for the color scale (applied to all panels).
            vmax: Optional upper bound for the color scale (applied to all panels).
            save_to: If not None, saves the figure to this path instead of showing.

        Returns:
            The matplotlib Figure object (for further customization).
        """
        SlipVisualizer._validate_length(fault, slip_vector)

        components = fault.active_components()
        k = len(components)

        fig = plt.figure(figsize=figsize)
        for i, component in enumerate(components):
            ax = fig.add_subplot(1, k, i + 1, projection='3d')
            data = slip_vector[fault.component_slice(component)]
            SlipVisualizer._plot_mesh_on_ax(
                ax, fault, data, _COMPONENT_TITLE[component], cmap, plot_edges,
                elev, azim, xlim, ylim, zlim, box_aspect, z_exaggeration,
                pad_frac, vmin, vmax,
            )

        plt.tight_layout()

        if save_to:
            plt.savefig(save_to, bbox_inches='tight', dpi=300)
            plt.close(fig)
        else:
            plt.show()

        return fig

    @staticmethod
    def _validate_length(fault, slip_vector):
        """Raises if the slip vector width does not match the fault's components."""
        expected = fault.num_components() * fault.num_patches()
        if slip_vector.shape[0] != expected:
            active = [str(c) for c in fault.active_components()]
            raise ValueError(
                f"Slip vector length ({len(slip_vector)}) does not match "
                f"num_components * num_patches ({expected}). "
                f"Active components: {active}."
            )

    @staticmethod
    def _select_component(fault, slip_vector, component):
        """
        Returns (component_data, title) for a single slip component of a fault.

        Args:
            fault: The fault model (source of the component layout).
            slip_vector: Full vector of length num_components * n_patches.
            component: A SlipComponent or string alias ('ss'/'ds', etc.).

        Raises:
            ValueError: If the vector width is wrong or the component is inactive.
        """
        SlipVisualizer._validate_length(fault, slip_vector)
        component = SlipComponent.coerce(component)
        sl = fault.component_slice(component)
        if sl is None:
            active = [str(c) for c in fault.active_components()]
            raise ValueError(
                f"Component '{component}' is not active on this fault. "
                f"Active components: {active}."
            )
        return slip_vector[sl], _COMPONENT_TITLE[component]

    @staticmethod
    def plot_slip_component(
        fault: TriangularFaultMesh,
        slip_vector: np.ndarray,
        component: str = 'strike_slip',
        figsize: tuple = (10, 7),
        cmap: str = 'viridis',
        plot_edges: bool = True,
        elev: float = 30,
        azim: float = -60,
        xlim: tuple = None,
        ylim: tuple = None,
        zlim: tuple = None,
        box_aspect: tuple = None,
        z_exaggeration: float = 1.0,
        pad_frac: float = 0.05,
        vmin: float = None,
        vmax: float = None,
        title: str = None,
        save_to: str = None,
    ):
        """
        Plots a single slip component (strike-slip OR dip-slip) on one 3D axis.

        Same view/zoom controls as :meth:`plot_slip_components`; see that method
        for details on ``zlim`` / ``z_exaggeration`` / ``box_aspect``.

        Args:
            fault: The TriangularFaultMesh object.
            slip_vector: A 1D numpy array of length ``num_components * n_patches``.
            component: Which component to draw: 'strike_slip'/'ss' or
                       'dip_slip'/'ds'. Must be active on the fault.
            title: Optional custom title (defaults to the component name).
            (all other args are as in :meth:`plot_slip_components`.)

        Returns:
            The matplotlib Figure object.
        """
        data, default_title = SlipVisualizer._select_component(
            fault, slip_vector, component
        )

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        SlipVisualizer._plot_mesh_on_ax(
            ax, fault, data, title or default_title, cmap, plot_edges,
            elev, azim, xlim, ylim, zlim, box_aspect, z_exaggeration,
            pad_frac, vmin, vmax,
        )

        plt.tight_layout()

        if save_to:
            plt.savefig(save_to, bbox_inches='tight', dpi=300)
            plt.close(fig)
        else:
            plt.show()

        return fig

    @staticmethod
    def _plot_mesh_on_ax(
        ax, fault, data, title, cmap_name, plot_edges, elev, azim,
        xlim, ylim, zlim, box_aspect, z_exaggeration, pad_frac, vmin, vmax,
    ):
        """Helper to render the colored mesh on a specific 3D axis."""
        verts, faces = fault.get_mesh_geometry()

        # Prepare Geometry: shape (M, 3, 3) -> M triangles, 3 corners, 3 coords
        triangles = verts[faces]

        # Normalize data for colormap (respecting optional user clim)
        v_lo = np.min(data) if vmin is None else vmin
        v_hi = np.max(data) if vmax is None else vmax
        norm = mcolors.Normalize(vmin=v_lo, vmax=v_hi)
        cmap = plt.get_cmap(cmap_name)
        face_colors = cmap(norm(data))

        mesh_poly = Poly3DCollection(
            triangles,
            facecolors=face_colors,
            edgecolors='k' if plot_edges else None,
            linewidths=0.1 if plot_edges else 0,
            alpha=0.9,
        )
        ax.add_collection3d(mesh_poly)

        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)

        # --- Axis limits: tight fit to data (with padding) unless overridden ---
        def _auto_limits(coord):
            lo, hi = coord.min(), coord.max()
            span = hi - lo
            if span == 0:
                span = 1.0
            pad = span * pad_frac
            return lo - pad, hi + pad

        x_lo, x_hi = xlim if xlim is not None else _auto_limits(verts[:, 0])
        y_lo, y_hi = ylim if ylim is not None else _auto_limits(verts[:, 1])
        z_lo, z_hi = zlim if zlim is not None else _auto_limits(verts[:, 2])

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_zlim(z_lo, z_hi)

        # --- Box aspect: keep true proportions so the fault fills the frame ---
        if box_aspect is not None:
            ax.set_box_aspect(box_aspect)
        else:
            dx = x_hi - x_lo
            dy = y_hi - y_lo
            dz = (z_hi - z_lo) * z_exaggeration
            ax.set_box_aspect((dx, dy, dz))

        # Labels
        ax.set_title(title)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        # Colorbar
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(data)
        plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Slip (m)')

    @staticmethod
    def plot_slip_components_2d(
        fault: TriangularFaultMesh,
        slip_vector: np.ndarray,
        figsize: tuple = (16, 6),
        cmap: str = 'viridis',
        plot_edges: bool = True,
        strike_vec: tuple = None,
        depth_positive_down: bool = True,
        equal_aspect: bool = True,
        vmin: float = None,
        vmax: float = None,
        save_to: str = None,
    ):
        """
        Plots the fault plane in 2D: along-strike distance (x) vs. depth (y).

        The 3D fault is "unrolled" by projecting every vertex onto the fault's
        horizontal strike direction (its principal horizontal axis, or a
        user-supplied ``strike_vec``) for the x-coordinate, and using the
        vertical coordinate for depth. This is exact for a planar vertical fault
        and a good approximation for gently non-planar faults. One panel is drawn
        per active component.

        Args:
            fault: The TriangularFaultMesh object.
            slip_vector: A 1D numpy array of length ``num_components * n_patches``,
                         laid out as consecutive per-component blocks in canonical
                         order (strike-slip before dip-slip).
            figsize: Tuple for figure dimensions.
            cmap: Matplotlib colormap name.
            plot_edges: If True, draws faint triangle edges.
            strike_vec: Optional (2,) horizontal (x, y) strike direction. If None,
                        it is estimated from the principal axis of the centroids.
            depth_positive_down: If True, the y-axis shows positive depth
                        increasing downward (top of plot = surface). If False,
                        the native Z coordinate is plotted as-is.
            equal_aspect: If True, use a 1:1 data aspect so patches are undistorted.
            vmin: Optional lower bound for the color scale (all panels).
            vmax: Optional upper bound for the color scale (all panels).
            save_to: If not None, saves the figure to this path instead of showing.

        Returns:
            The matplotlib Figure object.
        """
        SlipVisualizer._validate_length(fault, slip_vector)

        # Project every vertex to (along-strike, depth): shape (M, 3, 2)
        tri_2d = SlipVisualizer._project_to_plane(
            fault, strike_vec, depth_positive_down
        )

        components = fault.active_components()
        k = len(components)

        fig, axes = plt.subplots(1, k, figsize=figsize, squeeze=False)
        for ax, component in zip(axes.ravel(), components):
            data = slip_vector[fault.component_slice(component)]
            SlipVisualizer._plot_mesh_2d(
                ax, tri_2d, data, _COMPONENT_TITLE[component], cmap,
                plot_edges, depth_positive_down, equal_aspect, vmin, vmax,
            )

        plt.tight_layout()

        if save_to:
            plt.savefig(save_to, bbox_inches='tight', dpi=300)
            plt.close(fig)
        else:
            plt.show()

        return fig

    @staticmethod
    def _project_to_plane(fault, strike_vec, depth_positive_down):
        """
        Projects mesh vertices to (along-strike, depth) and returns the (M, 3, 2)
        array of 2D triangle vertices used by the 2D plots.
        """
        verts, faces = fault.get_mesh_geometry()

        if strike_vec is None:
            cen = fault.get_centroids()
            horiz = cen[:, :2] - cen[:, :2].mean(axis=0)
            _, eigvecs = np.linalg.eigh(horiz.T @ horiz)
            s_hat = eigvecs[:, -1]
        else:
            s_hat = np.asarray(strike_vec, dtype=float)
            s_hat = s_hat / np.linalg.norm(s_hat)

        s_all = verts[:, :2] @ s_hat
        s_all = s_all - s_all.min()

        depth_all = -verts[:, 2] if depth_positive_down else verts[:, 2]

        return np.stack([s_all[faces], depth_all[faces]], axis=-1)

    @staticmethod
    def plot_slip_component_2d(
        fault: TriangularFaultMesh,
        slip_vector: np.ndarray,
        component: str = 'strike_slip',
        figsize: tuple = (12, 4),
        cmap: str = 'viridis',
        plot_edges: bool = True,
        strike_vec: tuple = None,
        depth_positive_down: bool = True,
        equal_aspect: bool = True,
        vmin: float = None,
        vmax: float = None,
        title: str = None,
        save_to: str = None,
    ):
        """
        Plots a single slip component on the 2D fault plane (along-strike vs. depth).

        Single-panel version of :meth:`plot_slip_components_2d`; see that method
        for details on the projection and options.

        Args:
            fault: The TriangularFaultMesh object.
            slip_vector: A 1D numpy array of length ``num_components * n_patches``.
            component: Which component to draw: 'strike_slip'/'ss' or
                       'dip_slip'/'ds'. Must be active on the fault.
            title: Optional custom title (defaults to the component name).
            (all other args are as in :meth:`plot_slip_components_2d`.)

        Returns:
            The matplotlib Figure object.
        """
        data, default_title = SlipVisualizer._select_component(
            fault, slip_vector, component
        )
        tri_2d = SlipVisualizer._project_to_plane(
            fault, strike_vec, depth_positive_down
        )

        fig, ax = plt.subplots(figsize=figsize)
        SlipVisualizer._plot_mesh_2d(
            ax, tri_2d, data, title or default_title, cmap,
            plot_edges, depth_positive_down, equal_aspect, vmin, vmax,
        )

        plt.tight_layout()

        if save_to:
            plt.savefig(save_to, bbox_inches='tight', dpi=300)
            plt.close(fig)
        else:
            plt.show()

        return fig

    @staticmethod
    def _plot_mesh_2d(
        ax, tri_2d, data, title, cmap_name, plot_edges,
        depth_positive_down, equal_aspect, vmin, vmax,
    ):
        """Helper to render the projected 2D fault plane on an axis."""
        pc = PolyCollection(
            tri_2d,
            cmap=cmap_name,
            edgecolors='k' if plot_edges else 'none',
            linewidths=0.1 if plot_edges else 0,
        )
        pc.set_array(np.asarray(data))
        pc.set_clim(
            np.min(data) if vmin is None else vmin,
            np.max(data) if vmax is None else vmax,
        )
        ax.add_collection(pc)
        ax.autoscale()

        if equal_aspect:
            ax.set_aspect('equal', 'box')

        # Surface at top: for positive-down depth, invert so 0 is at the top.
        if depth_positive_down:
            ax.invert_yaxis()

        ax.set_title(title)
        ax.set_xlabel('Along-strike distance (m)')
        ax.set_ylabel('Depth (m)' if depth_positive_down else 'Z (m)')

        plt.colorbar(pc, ax=ax, shrink=0.8, aspect=20, label='Slip (m)')
