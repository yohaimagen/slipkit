"""
Visualisation of resolution / uncertainty products on the fault plane.

``ResolutionVisualizer`` mirrors :class:`~slipkit.utils.visualizers.slip.SlipVisualizer`
conventions -- it "unrolls" the mesh onto the (along-strike, depth) plane and
renders one panel per active slip component with a shared colour scale. It plots
the per-patch quantities produced by
:class:`~slipkit.core.resolution.ResolutionAnalyzer` (analytical) and the
empirical ensembles / recovery tests (Phase 2).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

from slipkit.core.fault import TriangularFaultMesh, SlipComponent
from slipkit.utils.visualizers.slip import SlipVisualizer, _COMPONENT_TITLE


class ResolutionVisualizer:
    """Plots resolution, spread, kernels and uncertainty maps on the fault plane."""

    # ------------------------------------------------------------------ #
    # Public: per-patch scalar maps (one panel per active component)
    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_resolution_diagonal(
        fault: TriangularFaultMesh,
        R: np.ndarray,
        *,
        cmap: str = "viridis",
        figsize: tuple = None,
        plot_edges: bool = True,
        save_to: str = None,
    ):
        """Plots ``diag(R)`` (0-1) per patch, one panel per component.

        Args:
            fault: The fault model.
            R: Either the full resolution matrix ``(P, P)`` (its diagonal is
                taken) or a per-patch diagonal vector.
            cmap: Colormap (sequential; the scale is fixed to ``[0, 1]``).
            figsize / plot_edges / save_to: As in :class:`SlipVisualizer`.

        Returns:
            The matplotlib Figure.
        """
        maps = ResolutionVisualizer._coerce_to_components(fault, R, take_diag=True)
        return ResolutionVisualizer._grid(
            fault, maps, save_to=save_to, figsize=figsize, plot_edges=plot_edges,
            cmap=cmap, label="diag(R)", vmin=0.0, vmax=1.0,
            title_fmt="Resolution diag(R) — {comp}",
        )

    @staticmethod
    def plot_spread_length(
        fault: TriangularFaultMesh,
        spread: np.ndarray,
        *,
        cmap: str = "magma",
        figsize: tuple = None,
        plot_edges: bool = True,
        save_to: str = None,
    ):
        """Plots the Backus-Gilbert resolution length per patch (mesh units).

        Args:
            fault: The fault model.
            spread: ``(M,)`` per-patch resolution length (single component). NaNs
                (masked, low-resolution patches) are drawn transparent.
            cmap / figsize / plot_edges / save_to: As above.

        Returns:
            The matplotlib Figure.
        """
        maps = ResolutionVisualizer._coerce_to_components(fault, spread)
        return ResolutionVisualizer._grid(
            fault, maps, save_to=save_to, figsize=figsize, plot_edges=plot_edges,
            cmap=cmap, label="Resolution length", vmin=None, vmax=None,
            title_fmt="Backus-Gilbert spread — {comp}",
        )

    @staticmethod
    def plot_uncertainty(
        fault: TriangularFaultMesh,
        std: np.ndarray,
        *,
        title: str = "Model uncertainty",
        cmap: str = "viridis",
        figsize: tuple = None,
        plot_edges: bool = True,
        vmin: float = None,
        vmax: float = None,
        save_to: str = None,
    ):
        """Plots a per-patch standard-deviation map (analytical ``sigma_m`` or MC ``sigma``).

        Pass an explicit ``vmin``/``vmax`` to render analytical and empirical
        uncertainty on the *same* scale for a fair side-by-side comparison.

        Args:
            fault: The fault model.
            std: ``(M,)`` (single component) or ``(P,)`` per-patch std.
            title: Panel title prefix.
            vmin / vmax: Shared colour limits (recommended for comparisons).
            cmap / figsize / plot_edges / save_to: As above.

        Returns:
            The matplotlib Figure.
        """
        maps = ResolutionVisualizer._coerce_to_components(fault, std)
        return ResolutionVisualizer._grid(
            fault, maps, save_to=save_to, figsize=figsize, plot_edges=plot_edges,
            cmap=cmap, label="σ (slip units)", vmin=vmin, vmax=vmax,
            title_fmt=title + " — {comp}",
        )

    @staticmethod
    def plot_kernel(
        fault: TriangularFaultMesh,
        R: np.ndarray,
        patch_index: int,
        *,
        cmap: str = "seismic",
        figsize: tuple = None,
        plot_edges: bool = True,
        save_to: str = None,
    ):
        """Plots the analytical point-spread function (a column of ``R``) for one patch.

        The column spreads a unit *true* spike across the whole estimate, so its
        component panels also expose cross-component leakage. A diverging cmap on
        a symmetric scale is used (the kernel takes both signs).

        Args:
            fault: The fault model.
            R: The full resolution matrix ``(P, P)``.
            patch_index: Column index (global unknown) whose PSF to draw.
            cmap / figsize / plot_edges / save_to: As above.

        Returns:
            The matplotlib Figure.
        """
        R = np.asarray(R)
        if R.ndim != 2:
            raise ValueError("plot_kernel needs the full (P, P) resolution matrix.")
        column = R[:, patch_index]
        maps = ResolutionVisualizer._coerce_to_components(fault, column)
        vlim = ResolutionVisualizer._sym_limit(column)
        return ResolutionVisualizer._grid(
            fault, maps, save_to=save_to, figsize=figsize, plot_edges=plot_edges,
            cmap=cmap, label="PSF", vmin=-vlim, vmax=vlim,
            title_fmt=f"PSF of patch {patch_index} — {{comp}}",
        )

    # ------------------------------------------------------------------ #
    # Public: multi-panel comparisons (input | recovered | diff)
    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_checkerboard(
        fault: TriangularFaultMesh,
        input_slip: np.ndarray,
        recovered_slip: np.ndarray,
        *,
        component: str = None,
        cmap: str = "viridis",
        diff_cmap: str = "seismic",
        figsize: tuple = None,
        plot_edges: bool = True,
        save_to: str = None,
    ):
        """Plots a recovery test as input | recovered | difference columns.

        Args:
            fault: The fault model.
            input_slip: ``(P,)`` target slip.
            recovered_slip: ``(P,)`` recovered slip (or a ``SlipDistribution``).
            component: Restrict to one component; ``None`` draws every active one
                as a row.
            cmap: Sequential cmap for the input/recovered columns (shared scale).
            diff_cmap: Diverging cmap for the (recovered - input) column.
            figsize / plot_edges / save_to: As above.

        Returns:
            The matplotlib Figure.
        """
        inp = np.asarray(getattr(input_slip, "slip_vector", input_slip), dtype=float)
        rec = np.asarray(getattr(recovered_slip, "slip_vector", recovered_slip), dtype=float)
        in_maps = ResolutionVisualizer._coerce_to_components(fault, inp, component=component)
        re_maps = ResolutionVisualizer._coerce_to_components(fault, rec, component=component)

        tri = SlipVisualizer._project_to_plane(fault, None, True)
        comps = [c for c, _ in in_maps]
        n = len(comps)
        fig, axes = plt.subplots(
            n, 3, figsize=figsize or (15, 4 * n), squeeze=False
        )
        for row, (comp, in_data) in enumerate(in_maps):
            re_data = dict(re_maps)[comp]
            diff = re_data - in_data
            shared = ResolutionVisualizer._pair_limits(in_data, re_data)
            dlim = ResolutionVisualizer._sym_limit(diff)
            label = str(comp) if comp is not None else ""
            ResolutionVisualizer._render(
                axes[row, 0], tri, in_data, f"Input — {label}", cmap, "Slip", *shared, plot_edges)
            ResolutionVisualizer._render(
                axes[row, 1], tri, re_data, f"Recovered — {label}", cmap, "Slip", *shared, plot_edges)
            ResolutionVisualizer._render(
                axes[row, 2], tri, diff, f"Difference — {label}", diff_cmap, "Δ Slip",
                -dlim, dlim, plot_edges)
        return ResolutionVisualizer._finish(fig, save_to)

    @staticmethod
    def plot_psf_comparison(
        fault: TriangularFaultMesh,
        analytical_kernel: np.ndarray,
        empirical_kernel: np.ndarray,
        *,
        component: str = None,
        cmap: str = "seismic",
        figsize: tuple = None,
        plot_edges: bool = True,
        save_to: str = None,
    ):
        """Plots analytical vs empirical PSF (and their difference) on a shared scale.

        Args:
            fault: The fault model.
            analytical_kernel: ``(P,)`` column of ``R`` (see
                :meth:`ResolutionAnalyzer.resolution_kernel`).
            empirical_kernel: ``(P,)`` recovered spike (see
                :meth:`SyntheticRecoveryTest.run`).
            component: Restrict to one component; ``None`` draws every active one.
            cmap / figsize / plot_edges / save_to: As above.

        Returns:
            The matplotlib Figure.
        """
        ana = np.asarray(getattr(analytical_kernel, "slip_vector", analytical_kernel), float)
        emp = np.asarray(getattr(empirical_kernel, "slip_vector", empirical_kernel), float)
        a_maps = ResolutionVisualizer._coerce_to_components(fault, ana, component=component)
        e_maps = dict(ResolutionVisualizer._coerce_to_components(fault, emp, component=component))

        tri = SlipVisualizer._project_to_plane(fault, None, True)
        n = len(a_maps)
        fig, axes = plt.subplots(n, 3, figsize=figsize or (15, 4 * n), squeeze=False)
        for row, (comp, a_data) in enumerate(a_maps):
            e_data = e_maps[comp]
            diff = e_data - a_data
            vlim = max(
                ResolutionVisualizer._sym_limit(a_data),
                ResolutionVisualizer._sym_limit(e_data),
            )
            dlim = ResolutionVisualizer._sym_limit(diff)
            label = str(comp) if comp is not None else ""
            ResolutionVisualizer._render(
                axes[row, 0], tri, a_data, f"Analytical PSF — {label}", cmap, "PSF",
                -vlim, vlim, plot_edges)
            ResolutionVisualizer._render(
                axes[row, 1], tri, e_data, f"Empirical PSF — {label}", cmap, "PSF",
                -vlim, vlim, plot_edges)
            ResolutionVisualizer._render(
                axes[row, 2], tri, diff, f"Difference — {label}", cmap, "Δ",
                -dlim, dlim, plot_edges)
        return ResolutionVisualizer._finish(fig, save_to)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce_to_components(fault, values, *, take_diag=False, component=None):
        """Splits a value array into ``[(component, (M,) data), ...]`` panels.

        Accepts a full ``(P,)`` block (split by ``component_slice``), a ``(P, P)``
        matrix (``take_diag`` -> its diagonal), or a bare ``(M,)`` array for a
        single-component fault (component reported as ``None`` if unresolved).
        """
        v = np.asarray(values, dtype=float)
        if take_diag and v.ndim == 2:
            v = np.diag(v)
        if v.ndim != 1:
            raise ValueError("Expected a 1-D per-patch array (or a matrix for diag).")

        m = fault.num_patches()
        active = fault.active_components()
        if component is not None:
            comp = SlipComponent.coerce(component)
            if comp not in active:
                raise ValueError(f"Component '{comp}' is not active on this fault.")
            sl = fault.component_slice(comp)
            block = v if v.shape[0] == m else v[sl]
            return [(comp, block)]

        if v.shape[0] == fault.num_components() * m:
            return [(c, v[fault.component_slice(c)]) for c in active]
        if v.shape[0] == m:
            return [(active[0] if fault.num_components() == 1 else None, v)]
        raise ValueError(
            f"Value length {v.shape[0]} matches neither M={m} nor "
            f"num_components*M={fault.num_components() * m}."
        )

    @staticmethod
    def _grid(fault, maps, *, save_to, figsize, plot_edges, cmap, label,
              vmin, vmax, title_fmt):
        """Renders one panel per component with a shared colour scale."""
        tri = SlipVisualizer._project_to_plane(fault, None, True)
        n = len(maps)
        fig, axes = plt.subplots(
            1, n, figsize=figsize or (7 * n, 5), squeeze=False
        )
        for ax, (comp, data) in zip(axes.ravel(), maps):
            comp_name = str(comp) if comp is not None else "map"
            ResolutionVisualizer._render(
                ax, tri, data, title_fmt.format(comp=comp_name), cmap, label,
                vmin, vmax, plot_edges)
        return ResolutionVisualizer._finish(fig, save_to)

    @staticmethod
    def _render(ax, tri_2d, data, title, cmap, label, vmin, vmax, plot_edges):
        """Draws one coloured fault-plane panel (NaNs transparent)."""
        pc = PolyCollection(
            tri_2d, cmap=cmap,
            edgecolors="k" if plot_edges else "none",
            linewidths=0.1 if plot_edges else 0,
        )
        arr = np.asarray(data, dtype=float)
        pc.set_array(arr)
        finite = arr[np.isfinite(arr)]
        lo = (float(np.min(finite)) if finite.size else 0.0) if vmin is None else vmin
        hi = (float(np.max(finite)) if finite.size else 1.0) if vmax is None else vmax
        if lo == hi:
            hi = lo + 1e-9
        pc.set_clim(lo, hi)
        ax.add_collection(pc)
        ax.autoscale()
        ax.set_aspect("equal", "box")
        ax.invert_yaxis()  # depth positive down => surface at top
        ax.set_title(title)
        ax.set_xlabel("Along-strike distance")
        ax.set_ylabel("Depth")
        plt.colorbar(pc, ax=ax, shrink=0.8, aspect=20, label=label)

    @staticmethod
    def _sym_limit(data):
        """Symmetric colour limit for a diverging map (max |finite value|)."""
        arr = np.asarray(data, dtype=float)
        finite = np.abs(arr[np.isfinite(arr)])
        v = float(finite.max()) if finite.size else 1.0
        return v if v > 0 else 1.0

    @staticmethod
    def _pair_limits(a, b):
        """Shared (vmin, vmax) across two sequential maps."""
        both = np.concatenate([np.asarray(a, float).ravel(), np.asarray(b, float).ravel()])
        finite = both[np.isfinite(both)]
        if finite.size == 0:
            return 0.0, 1.0
        return float(finite.min()), float(finite.max())

    @staticmethod
    def _finish(fig, save_to):
        """Shared tight-layout + save/show tail."""
        fig.tight_layout()
        if save_to:
            fig.savefig(save_to, bbox_inches="tight", dpi=300)
            plt.close(fig)
        else:
            plt.show()
        return fig
