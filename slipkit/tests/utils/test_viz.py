import os
import numpy as np
import pytest
from slipkit.utils.viz import GreenFunctionVisualizer
from slipkit.core.fault import TriangularFaultMesh
from slipkit.core.data import GeodeticDataSet
from slipkit.core.physics import CutdeCpuEngine

@pytest.fixture
def sample_fault_for_viz():
    """A simple fault mesh for visualization tests."""
    vertices = np.array([
        [0, 0, -1], [1, 0, -1], [0.5, 1, -1], # Patch 0
        [1, 0, -1], [2, 0, -1], [1.5, 1, -1], # Patch 1
    ])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    return TriangularFaultMesh((vertices, faces))

@pytest.fixture
def sample_dataset_for_viz():
    """A simple dataset on a grid for visualization tests."""
    n_x, n_y = 20, 20
    x = np.linspace(-1, 3, n_x)
    y = np.linspace(-1, 2, n_y)
    xx, yy = np.meshgrid(x, y)
    coords = np.vstack([xx.ravel(), yy.ravel(), np.zeros(n_x * n_y)]).T.copy()
    
    # Dummy LOS pointing in the z direction
    unit_vecs = np.zeros_like(coords)
    unit_vecs[:, 2] = 1.0

    return GeodeticDataSet(
        name="viz_test_data",
        coords=coords,
        data=np.random.rand(len(coords)),
        unit_vecs=unit_vecs,
        sigma=np.ones(len(coords)),
    )

def test_green_function_visualizer_plot_response(
    tmp_path, sample_fault_for_viz, sample_dataset_for_viz
):
    """
    Test that GreenFunctionVisualizer.plot_response runs and saves a file.
    This is an integration test.
    """
    output_path = tmp_path / "green_function_strike.png"
    
    engine = CutdeCpuEngine(poisson_ratio=0.25)

    try:
        GreenFunctionVisualizer.plot_response(
            fault=sample_fault_for_viz,
            dataset=sample_dataset_for_viz,
            patch_idx=0,
            slip_component='strike',
            engine=engine,
            save_to=str(output_path),
        )
    except Exception as e:
        pytest.fail(f"plot_response failed with an exception: {e}")

    assert os.path.exists(output_path), "Plot file was not created."

def test_plot_total_response(
    tmp_path, sample_fault_for_viz, sample_dataset_for_viz
):
    """Test that plot_total_response runs and saves a file."""
    output_path = tmp_path / "total_response.png"
    engine = CutdeCpuEngine()
    
    n_patches = sample_fault_for_viz.num_patches()
    # Assume 1m of strike-slip on the first patch, 0.5m of dip-slip on the second
    slip_distribution = np.zeros(2 * n_patches)
    slip_distribution[0] = 1.0
    slip_distribution[n_patches + 1] = 0.5

    try:
        GreenFunctionVisualizer.plot_total_response(
            fault=sample_fault_for_viz,
            dataset=sample_dataset_for_viz,
            slip_distribution=slip_distribution,
            engine=engine,
            save_to=str(output_path),
        )
    except Exception as e:
        pytest.fail(f"plot_total_response failed with an exception: {e}")

    assert os.path.exists(output_path)

def test_plot_response_invalid_patch_idx(
    sample_fault_for_viz, sample_dataset_for_viz
):
    """Test that an invalid patch index raises a ValueError."""
    engine = CutdeCpuEngine()
    with pytest.raises(ValueError, match="patch_idx must be between 0 and 1"):
        GreenFunctionVisualizer.plot_response(
            fault=sample_fault_for_viz,
            dataset=sample_dataset_for_viz,
            patch_idx=99, # Invalid
            slip_component='strike',
            engine=engine,
        )

def test_plot_response_invalid_slip_component(
    sample_fault_for_viz, sample_dataset_for_viz
):
    """Test that an invalid slip component raises a ValueError."""
    engine = CutdeCpuEngine()
    with pytest.raises(ValueError, match="slip_component must be 'strike' or 'dip'"):
        GreenFunctionVisualizer.plot_response(
            fault=sample_fault_for_viz,
            dataset=sample_dataset_for_viz,
            patch_idx=0,
            slip_component='invalid', # Invalid
            engine=engine,
        )

def test_plot_total_response_invalid_slip_shape(
    sample_fault_for_viz, sample_dataset_for_viz
):
    """Test that an invalid slip distribution shape raises a ValueError."""
    engine = CutdeCpuEngine()
    invalid_slip = np.ones(sample_fault_for_viz.num_patches()) # Shape (M,) not (2M,)
    with pytest.raises(ValueError, match=r"slip_distribution must have shape \(4,\)."):
        GreenFunctionVisualizer.plot_total_response(
            fault=sample_fault_for_viz,
            dataset=sample_dataset_for_viz,
            slip_distribution=invalid_slip,
            engine=engine,
        )