import numpy as np
import pytest
import warnings
from scipy.sparse import issparse, block_diag, csr_matrix
from slipkit.core.fault import TriangularFaultMesh
from slipkit.core.regularization import LaplacianSmoothing, RegularizationManager

@pytest.fixture
def simple_fault_mesh():
    """A fixture for a simple triangular fault mesh with a known Laplacian."""
    vertices = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [1.0, 1.0, 0.0],  # 3
    ])
    # f0 neighbors f1
    faces = np.array([
        [0, 1, 2],  # f0
        [1, 3, 2],  # f1
    ])
    # Expected L for this mesh:
    # f0: neighbors=[f1], degree=1
    # f1: neighbors=[f0], degree=1
    # L = [[1, -1],
    #      [-1, 1]]
    return TriangularFaultMesh((vertices, faces))

@pytest.fixture
def another_simple_fault_mesh():
    """Another fixture for a simple triangular fault mesh."""
    vertices = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
    ])
    faces = np.array([
        [0, 1, 2],  # f0
    ])
    # Expected L for this mesh:
    # f0: neighbors=[], degree=0 (isolated patch)
    # L = [[0]]
    return TriangularFaultMesh((vertices, faces))


def test_laplacian_smoothing_single_fault(simple_fault_mesh):
    """
    Test LaplacianSmoothing with a single fault, ensuring correct block-diagonal
    matrix assembly for two slip components and no warning.
    """
    reg_manager = LaplacianSmoothing()
    lambda_spatial = 0.5

    # Capture warnings to ensure none are raised for a single fault
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        smoothing_matrix = reg_manager.build_smoothing_matrix(
            [simple_fault_mesh], lambda_spatial
        )
        assert len(w) == 0, "No warnings should be raised for a single fault."

    assert issparse(smoothing_matrix)
    
    # A single fault with 2 patches, each having 2 components, so 2*2 = 4x4 matrix
    assert smoothing_matrix.shape == (4, 4)

    # Expected L from simple_fault_mesh:
    # L = [[1, -1],
    #      [-1, 1]]
    
    # Expected S_full = [[lambda*L, 0],
    #                    [0, lambda*L]]
    # S_full = [[0.5, -0.5, 0.0, 0.0],
    #           [-0.5, 0.5, 0.0, 0.0],
    #           [0.0, 0.0, 0.5, -0.5],
    #           [0.0, 0.0, -0.5, 0.5]]
    
    expected_l_single = simple_fault_mesh.get_smoothing_matrix().toarray()
    expected_weighted_l = lambda_spatial * expected_l_single
    expected_s_full = block_diag([expected_weighted_l, expected_weighted_l], format='csr').toarray()

    assert np.allclose(smoothing_matrix.toarray(), expected_s_full)


def test_laplacian_smoothing_multiple_faults(simple_fault_mesh, another_simple_fault_mesh):
    """
    Test LaplacianSmoothing with multiple faults, ensuring correct block-diagonal
    matrix assembly and that a warning is raised.
    """
    reg_manager = LaplacianSmoothing()
    lambda_spatial = 1.0

    faults = [simple_fault_mesh, another_simple_fault_mesh]

    # Capture warnings to ensure the multi-fault warning is raised
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        smoothing_matrix = reg_manager.build_smoothing_matrix(faults, lambda_spatial)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Multi-fault stitching is not yet implemented" in str(w[-1].message)

    assert issparse(smoothing_matrix)

    # simple_fault_mesh has 2 patches, another_simple_fault_mesh has 1 patch.
    # Total patches = 2 + 1 = 3
    # Total degrees of freedom = 3 patches * 2 components/patch = 6
    assert smoothing_matrix.shape == (6, 6)

    # Expected L for simple_fault_mesh (L1):
    # L1 = [[1, -1],
    #       [-1, 1]]
    # Expected L for another_simple_fault_mesh (L2):
    # L2 = [[0]]

    # Expected S_full = [[lambda*L1, 0, 0, 0],
    #                    [0, lambda*L1, 0, 0],
    #                    [0, 0, lambda*L2, 0],
    #                    [0, 0, 0, lambda*L2]]
    
    l1_single = simple_fault_mesh.get_smoothing_matrix().toarray()
    l2_single = another_simple_fault_mesh.get_smoothing_matrix().toarray()

    weighted_l1 = lambda_spatial * l1_single
    weighted_l2 = lambda_spatial * l2_single

    block1 = block_diag([weighted_l1, weighted_l1], format='csr').toarray() # 4x4
    block2 = block_diag([weighted_l2, weighted_l2], format='csr').toarray() # 2x2

    expected_s_full = block_diag([block1, block2], format='csr').toarray()
    
    assert np.allclose(smoothing_matrix.toarray(), expected_s_full)


def test_laplacian_smoothing_no_faults():
    """
    Test LaplacianSmoothing with an empty list of faults, ensuring an empty
    sparse matrix is returned.
    """
    reg_manager = LaplacianSmoothing()
    lambda_spatial = 0.1

    smoothing_matrix = reg_manager.build_smoothing_matrix([], lambda_spatial)

    assert issparse(smoothing_matrix)
    assert smoothing_matrix.shape == (0, 0)
    assert smoothing_matrix.nnz == 0

def test_abstract_regularization_manager_instantiation():
    """Verify that RegularizationManager cannot be instantiated directly."""
    expected_regex = (
        "Can't instantiate abstract class RegularizationManager without an "
        "implementation for abstract method 'build_smoothing_matrix'"
    )
    with pytest.raises(TypeError, match=expected_regex):
        RegularizationManager()
