# slipkit/tests/utils/test_data_fit.py
import pytest
import numpy as np
import matplotlib.pyplot as plt

from slipkit.core.data import GeodeticDataSet
from slipkit.core.fault import TriangularFaultMesh
from slipkit.utils.visualizers.sar_data_fit import SarDataFitVisualizer

@pytest.fixture
def sample_geodetic_dataset():
    """Provides a sample GeodeticDataSet for testing."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    data = np.array([0.1, 0.2, 0.3, 0.4])
    unit_vecs = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    sigma = np.array([0.01, 0.01, 0.01, 0.01])
    return GeodeticDataSet(
        name="test_dataset", coords=coords, data=data, unit_vecs=unit_vecs, sigma=sigma
    )

@pytest.fixture
def sample_predicted_data():
    """Provides sample predicted data for testing."""
    return np.array([0.11, 0.19, 0.32, 0.38])

@pytest.fixture
def sample_fault_mesh():
    """Provides a sample TriangularFaultMesh for testing using raw arrays."""
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    faces = np.array([
        [0, 1, 2],
        [1, 3, 2],
    ])
    # Pass verts and faces as a tuple for mesh_input
    return TriangularFaultMesh(mesh_input=(verts, faces))

def test_plot_data_fit_basic(sample_geodetic_dataset, sample_predicted_data):
    """Test that plot_data_fit runs without error and returns figure/axes."""
    fig, axes = SarDataFitVisualizer.plot_data_fit(
        observed_data=sample_geodetic_dataset,
        predicted_data=sample_predicted_data,
        return_fig_ax=True,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (3,)
    plt.close(fig)  # Close the figure to prevent it from showing up

def test_plot_data_fit_with_fault(
    sample_geodetic_dataset, sample_predicted_data, sample_fault_mesh
):
    """Test that plot_data_fit runs with a fault model."""
    fig, axes = SarDataFitVisualizer.plot_data_fit(
        observed_data=sample_geodetic_dataset,
        predicted_data=sample_predicted_data,
        fault=sample_fault_mesh,
        return_fig_ax=True,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)

def test_plot_data_fit_value_error(sample_geodetic_dataset):
    """Test that plot_data_fit raises ValueError for mismatched shapes."""
    predicted_data_mismatch = np.array([0.1, 0.2])  # Mismatched shape
    with pytest.raises(ValueError, match="Shape of observed_data.data and predicted_data must match."):
        SarDataFitVisualizer.plot_data_fit(
            observed_data=sample_geodetic_dataset,
            predicted_data=predicted_data_mismatch,
            return_fig_ax=True,
        )

def test_plot_data_fit_with_fault_list(
    sample_geodetic_dataset, sample_predicted_data, sample_fault_mesh
):
    """Test that plot_data_fit runs with a list of fault models."""
    fault_list = [sample_fault_mesh, sample_fault_mesh] # Use the same mesh twice for simplicity
    fig, axes = SarDataFitVisualizer.plot_data_fit(
        observed_data=sample_geodetic_dataset,
        predicted_data=sample_predicted_data,
        fault=fault_list,
        return_fig_ax=True,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)

def test_plot_data_fit_with_vmin_vmax(sample_geodetic_dataset, sample_predicted_data):
    """Test that plot_data_fit runs with explicit vmin and vmax."""
    fig, axes = SarDataFitVisualizer.plot_data_fit(
        observed_data=sample_geodetic_dataset,
        predicted_data=sample_predicted_data,
        vmin=-0.5,
        vmax=0.5,
        return_fig_ax=True,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(fig)