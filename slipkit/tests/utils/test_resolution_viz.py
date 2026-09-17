import os

import matplotlib
matplotlib.use("Agg")  # headless backend for tests

import numpy as np
import pytest

from slipkit.core.fault import (
    TriangularFaultMesh,
    SlipComponent,
    CANONICAL_COMPONENT_ORDER,
)
from slipkit.utils.visualizers import ResolutionVisualizer


def _strip_mesh(m, spacing=1.0, components=CANONICAL_COMPONENT_ORDER):
    n_v = m + 2
    xs = np.arange(n_v) * (0.5 * spacing)
    ys = np.tile([0.0, spacing], n_v)[:n_v]
    verts = np.column_stack([xs, ys, np.zeros(n_v)]).astype(float)
    faces = np.array([[i, i + 1, i + 2] for i in range(m)], dtype=int)
    return TriangularFaultMesh((verts, faces), slip_components=components)


def test_plot_resolution_diagonal_from_matrix(tmp_path):
    fault = _strip_mesh(6, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    R = np.eye(m) * 0.7
    out = tmp_path / "diag.png"
    fig = ResolutionVisualizer.plot_resolution_diagonal(fault, R, save_to=str(out))
    assert os.path.exists(out)
    assert fig is not None


def test_plot_spread_with_nans(tmp_path):
    fault = _strip_mesh(6, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    spread = np.linspace(0.5, 2.0, m)
    spread[0] = np.nan  # masked patch must not crash the renderer
    out = tmp_path / "spread.png"
    ResolutionVisualizer.plot_spread_length(fault, spread, save_to=str(out))
    assert os.path.exists(out)


def test_plot_uncertainty_two_components(tmp_path):
    fault = _strip_mesh(5)  # SS + DS
    p = 2 * fault.num_patches()
    std = np.linspace(0.1, 0.5, p)
    out = tmp_path / "unc.png"
    fig = ResolutionVisualizer.plot_uncertainty(
        fault, std, vmin=0.0, vmax=0.6, save_to=str(out)
    )
    # One panel per active component.
    assert len(fig.axes) >= 2
    assert os.path.exists(out)


def test_plot_kernel_requires_matrix(tmp_path):
    fault = _strip_mesh(5, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    R = np.random.default_rng(0).standard_normal((m, m))
    out = tmp_path / "kernel.png"
    ResolutionVisualizer.plot_kernel(fault, R, patch_index=2, save_to=str(out))
    assert os.path.exists(out)
    with pytest.raises(ValueError, match="full \\(P, P\\)"):
        ResolutionVisualizer.plot_kernel(fault, np.diag(R), patch_index=2)


def test_plot_checkerboard_three_panels(tmp_path):
    fault = _strip_mesh(6, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    inp = np.tile([0.0, 1.0], m)[:m]
    rec = inp * 0.8 + 0.05
    out = tmp_path / "cb.png"
    fig = ResolutionVisualizer.plot_checkerboard(fault, inp, rec, save_to=str(out))
    # input | recovered | difference
    assert len([a for a in fig.axes if a.get_title()]) == 3
    assert os.path.exists(out)


def test_plot_psf_comparison(tmp_path):
    fault = _strip_mesh(6, components=[SlipComponent.STRIKE_SLIP])
    m = fault.num_patches()
    ana = np.zeros(m); ana[3] = 1.0
    emp = np.zeros(m); emp[3] = 0.7; emp[2] = 0.1
    out = tmp_path / "psf.png"
    ResolutionVisualizer.plot_psf_comparison(fault, ana, emp, save_to=str(out))
    assert os.path.exists(out)


def test_coerce_rejects_bad_length():
    fault = _strip_mesh(5, components=[SlipComponent.STRIKE_SLIP])
    with pytest.raises(ValueError, match="matches neither"):
        ResolutionVisualizer._coerce_to_components(fault, np.zeros(3))
