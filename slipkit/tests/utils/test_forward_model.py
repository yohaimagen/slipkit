import pytest
import numpy as np
import matplotlib.pyplot as plt

from slipkit.core.fault import TriangularFaultMesh
from slipkit.core.physics import CutdeCpuEngine
from slipkit.utils.visualizers.forward_model import ForwardModelVisualizer

@pytest.fixture
def sample_fault_mesh():
    """Provides a sample TriangularFaultMesh for testing using raw arrays."""
    verts = np.array([
        [0.0, 0.0, -1000.0],
        [1000.0, 0.0, -1000.0],
        [0.0, 1000.0, -2000.0],
        [1000.0, 1000.0, -2000.0],
    ])
    faces = np.array([
        [0, 1, 2],
        [1, 3, 2],
    ])
    return TriangularFaultMesh(mesh_input=(verts, faces))

@pytest.fixture
def sample_fault_mesh_2():
    """Provides a second sample TriangularFaultMesh for testing with multiple faults."""
    verts = np.array([
        [1000.0, 1000.0, -500.0],
        [2000.0, 1000.0, -500.0],
        [1000.0, 2000.0, -1500.0],
        [2000.0, 2000.0, -1500.0],
    ])
    faces = np.array([
        [0, 1, 2],
        [1, 3, 2],
    ])
    return TriangularFaultMesh(mesh_input=(verts, faces))

@pytest.fixture
def sample_engine():
    """Provides a sample CutdeCpuEngine."""
    return CutdeCpuEngine(poisson_ratio=0.25)

def test_plot_enu_response_basic(sample_fault_mesh, sample_engine):
    """
    Test that plot_enu_response runs without error and returns figure/axes
    with default grid parameters.
    """
    # Example slip vector (pure dip-slip)
    slip = np.concatenate([
        np.zeros(sample_fault_mesh.num_patches()),
        np.ones(sample_fault_mesh.num_patches())
    ])
    
    # Define arbitrary extent for testing
    x_extent = (-2000.0, 2000.0)
    y_extent = (-2000.0, 2000.0)

    fig, axes = ForwardModelVisualizer.plot_enu_response(
        fault=sample_fault_mesh,
        slip_vector=slip,
        engine=sample_engine,
        x_extent=x_extent,
        y_extent=y_extent,
        return_fig_ax=True,
    )
    
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (3,) # Expect 3 subplots for E, N, U
    plt.close(fig)

def test_plot_enu_response_custom_grid_resolution(sample_fault_mesh, sample_engine):
    """
    Test that plot_enu_response runs with a custom grid resolution.
    """
    slip = np.concatenate([
        np.zeros(sample_fault_mesh.num_patches()),
        np.ones(sample_fault_mesh.num_patches())
    ])
    
    x_extent = (-2000.0, 2000.0)
    y_extent = (-2000.0, 2000.0)
    custom_resolution = 50 # Lower resolution for faster test

    fig, axes = ForwardModelVisualizer.plot_enu_response(
        fault=sample_fault_mesh,
        slip_vector=slip,
        engine=sample_engine,
        x_extent=x_extent,
        y_extent=y_extent,
        grid_resolution=custom_resolution,
        return_fig_ax=True,
    )
    
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)

def test_plot_enu_response_custom_figsize(sample_fault_mesh, sample_engine):
    """
    Test that plot_enu_response runs with a custom figsize.
    """
    slip = np.concatenate([
        np.zeros(sample_fault_mesh.num_patches()),
        np.ones(sample_fault_mesh.num_patches())
    ])
    
    x_extent = (-2000.0, 2000.0)
    y_extent = (-2000.0, 2000.0)
    custom_figsize = (10, 3)

    fig, axes = ForwardModelVisualizer.plot_enu_response(
        fault=sample_fault_mesh,
        slip_vector=slip,
        engine=sample_engine,
        x_extent=x_extent,
        y_extent=y_extent,
        figsize=custom_figsize,
        return_fig_ax=True,
    )
    
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)

def test_plot_enu_response_with_fault_list(
    sample_fault_mesh, sample_fault_mesh_2, sample_engine
):
    """
    Test that plot_enu_response runs with a list of fault models.
    """
    fault_list = [sample_fault_mesh, sample_fault_mesh_2]
    # Slip vector needs to match total number of patches across all faults
    total_patches = sum(f.num_patches() for f in fault_list)
    slip = np.ones(total_patches * 2)

    x_extent = (-2000.0, 3000.0)
    y_extent = (-2000.0, 3000.0)

    fig, axes = ForwardModelVisualizer.plot_enu_response(
        fault=fault_list,
        slip_vector=slip,
        engine=sample_engine,
        x_extent=x_extent,
        y_extent=y_extent,
        return_fig_ax=True,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)

def test_plot_enu_response_with_vmin_vmax(sample_fault_mesh, sample_engine):
    """
    Test that plot_enu_response runs with explicit vmin and vmax.
    """
    slip = np.concatenate([
        np.zeros(sample_fault_mesh.num_patches()),
        np.ones(sample_fault_mesh.num_patches())
    ])
    
    x_extent = (-2000.0, 2000.0)
    y_extent = (-2000.0, 2000.0)

    fig, axes = ForwardModelVisualizer.plot_enu_response(
        fault=sample_fault_mesh,
        slip_vector=slip,
        engine=sample_engine,
        x_extent=x_extent,
        y_extent=y_extent,
        vmin=-0.1,
        vmax=0.1,
        return_fig_ax=True,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)