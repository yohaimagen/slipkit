import numpy as np
import pytest
from unittest.mock import patch
from slipkit.core.physics import CutdeCpuEngine
from slipkit.core.fault import TriangularFaultMesh
from slipkit.core.data import GeodeticDataSet

@pytest.fixture
def sample_fault_mesh():
    """A fixture for a simple triangular fault mesh."""
    vertices = np.array([[0, 0, -1], [1, 0, -1], [0, 1, -1], [1, 1, -1]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    return TriangularFaultMesh((vertices, faces))

@pytest.fixture
def sample_geodetic_dataset():
    """A fixture for a simple geodetic dataset."""
    n_obs = 10
    return GeodeticDataSet(
        name="test_data",
        coords=np.random.rand(n_obs, 3),
        data=np.random.rand(n_obs),
        unit_vecs=np.random.rand(n_obs, 3),
        sigma=np.ones(n_obs),
    )

@patch("slipkit.core.physics.HS.disp_matrix")
def test_cutde_cpu_engine_build_kernel(
    mock_disp_matrix, sample_fault_mesh, sample_geodetic_dataset
):
    """
    Test the build_kernel method of CutdeCpuEngine.
    Mocks the cutde call and verifies the matrix assembly logic.
    """
    n_obs = len(sample_geodetic_dataset)
    n_patches = sample_fault_mesh.num_patches()
    
    # Create a mock return value for disp_matrix with the correct shape (N, 3, M, 3)
    mock_g_raw = np.ones((n_obs, 3, n_patches, 3))
    mock_disp_matrix.return_value = mock_g_raw
    
    engine = CutdeCpuEngine(poisson_ratio=0.25)
    g_matrix = engine.build_kernel(sample_fault_mesh, sample_geodetic_dataset)
    
    # 1. Verify the shape of the output G matrix
    assert g_matrix.shape == (n_obs, 2 * n_patches)
    
    # 2. Verify the content based on the mock data
    # The mock disp_matrix has ones. The calculation for each element in G should be:
    # g[i, j] = sum(disp_mat[i, j, :, 0] * unit_vecs[i, :]) for strike-slip
    # g[i, M+j] = sum(disp_mat[i, j, :, 1] * unit_vecs[i, :]) for dip-slip
    
    # disp_mat is all ones, so this simplifies to:
    # g[i, j] = sum(unit_vecs[i, :])
    
    unit_vecs = sample_geodetic_dataset.unit_vecs
    expected_projection = np.sum(unit_vecs, axis=1) # Shape: (n_obs,)

    # Create the expected (N, 2M) matrix
    expected_g = np.zeros((n_obs, 2 * n_patches))
    for i in range(n_obs):
        for j in range(n_patches):
            # Since mock_g_raw is all ones, the response is just the sum of unit_vec components
            expected_g[i, j] = expected_projection[i] # Strike-slip
            expected_g[i, j + n_patches] = expected_projection[i] # Dip-slip

    assert np.allclose(g_matrix, expected_g)
    
    # 3. Verify that cutde was called with the correct arguments
    mock_disp_matrix.assert_called_once()
    call_args, call_kwargs = mock_disp_matrix.call_args
    assert np.array_equal(call_kwargs['obs_pts'], sample_geodetic_dataset.coords)
    
    expected_tris = sample_fault_mesh.vertices[sample_fault_mesh.faces]
    assert np.array_equal(call_kwargs['tris'], expected_tris)
    assert call_kwargs['nu'] == 0.25

def test_cutde_cpu_engine_init():
    """Test the initializer of CutdeCpuEngine."""
    engine = CutdeCpuEngine(poisson_ratio=0.3)
    assert engine.nu == 0.3
