import numpy as np
import pytest
import warnings
from scipy.sparse import issparse, block_diag, csr_matrix
from slipkit.core.fault import TriangularFaultMesh, SlipComponent
from slipkit.core.regularization import (
    LaplacianSmoothing,
    RegularizationManager,
    DeepEdgeDamping,
)

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

def test_laplacian_smoothing_single_component(simple_fault_mesh):
    """
    A single-component fault yields an (M, M) smoothing matrix equal to lambda*L
    (no duplicated block).
    """
    vertices = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
    ])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    fault = TriangularFaultMesh(
        (vertices, faces), slip_components=[SlipComponent.STRIKE_SLIP]
    )

    lambda_spatial = 0.5
    S = LaplacianSmoothing().build_smoothing_matrix([fault], lambda_spatial)

    m = fault.num_patches()
    assert S.shape == (m, m)  # not 2M
    expected = lambda_spatial * fault.get_smoothing_matrix().toarray()
    assert np.allclose(S.toarray(), expected)


def test_abstract_regularization_manager_instantiation():
    """Verify that RegularizationManager cannot be instantiated directly."""
    expected_regex = (
        "Can't instantiate abstract class RegularizationManager without an "
        "implementation for abstract method 'build_smoothing_matrix'"
    )
    with pytest.raises(TypeError, match=expected_regex):
        RegularizationManager()


@pytest.fixture
def dipping_fault_mesh():
    """A 4-triangle strip dipping down-dip in +y, spanning depths 0 to 2."""
    vertices = np.array([
        [0.0, 0.0,  0.0],  # 0
        [1.0, 0.0,  0.0],  # 1
        [0.0, 1.0, -1.0],  # 2
        [1.0, 1.0, -1.0],  # 3
        [0.0, 2.0, -2.0],  # 4
        [1.0, 2.0, -2.0],  # 5
    ])
    faces = np.array([
        [0, 1, 2],  # f0, centroid depth 0.333
        [1, 3, 2],  # f1, centroid depth 0.667
        [2, 3, 4],  # f2, centroid depth 1.333
        [3, 5, 4],  # f3, centroid depth 1.667
    ])
    return TriangularFaultMesh((vertices, faces))


def test_deep_patch_indices_selects_deepest_band(dipping_fault_mesh):
    """Only patches within `tol` of the deepest centroid are selected."""
    np.testing.assert_array_equal(
        dipping_fault_mesh.deep_patch_indices(tol=0.1), [3]
    )
    np.testing.assert_array_equal(
        dipping_fault_mesh.deep_patch_indices(tol=0.5), [2, 3]
    )


def test_deep_edge_damping_appends_rows_for_all_active_components(dipping_fault_mesh):
    """One damped row per deep patch per active component, at alpha * lambda."""
    fault, alpha, lam = dipping_fault_mesh, 3.0, 2.0
    manager = DeepEdgeDamping(alpha=alpha, tol=0.5)
    deep = fault.deep_patch_indices(tol=0.5)  # [2, 3]

    s_base = LaplacianSmoothing().build_smoothing_matrix([fault], lam)
    s_full = manager.build_smoothing_matrix([fault], lam).toarray()

    n_extra = len(deep) * fault.num_components()  # 2 patches x (ss, ds)
    assert s_full.shape == (s_base.shape[0] + n_extra, s_base.shape[1])
    # The base block is untouched.
    np.testing.assert_allclose(s_full[:s_base.shape[0]], s_base.toarray())

    # Each extra row damps exactly one column, and both components are covered.
    extra = s_full[s_base.shape[0]:]
    assert np.count_nonzero(extra) == n_extra
    expected_cols = sorted(
        list(deep) + list(deep + fault.num_patches())
    )
    np.testing.assert_array_equal(sorted(np.nonzero(extra)[1]), expected_cols)
    np.testing.assert_allclose(extra[extra != 0], alpha * lam)


def test_deep_edge_damping_single_component_fault(dipping_fault_mesh):
    """A fault solving for one component only gets that component damped."""
    verts, faces = dipping_fault_mesh.get_mesh_geometry()
    fault = TriangularFaultMesh(
        (verts, faces), slip_components=[SlipComponent.DIP_SLIP]
    )
    manager = DeepEdgeDamping(alpha=1.0, tol=0.5)
    cols = manager.constrained_columns([fault])
    # Single-component block: columns are the patch indices themselves.
    np.testing.assert_array_equal(cols, fault.deep_patch_indices(tol=0.5))


def test_deep_edge_damping_component_subset_and_offsets(dipping_fault_mesh):
    """`components` restricts damping; multi-fault columns are globally offset."""
    fault = dipping_fault_mesh
    manager = DeepEdgeDamping(alpha=1.0, tol=0.5, components=["ds"])
    deep = fault.deep_patch_indices(tol=0.5)
    m = fault.num_patches()

    np.testing.assert_array_equal(manager.constrained_columns([fault]), deep + m)
    # Second fault's columns are shifted by the first fault's block width.
    np.testing.assert_array_equal(
        manager.constrained_columns([fault, fault]),
        np.concatenate([deep + m, deep + m + fault.num_components() * m]),
    )


def test_deep_edge_damping_disabled_matches_base(dipping_fault_mesh):
    """alpha = 0 leaves the base regularization untouched."""
    base = LaplacianSmoothing().build_smoothing_matrix([dipping_fault_mesh], 1.0)
    damped = DeepEdgeDamping(alpha=0.0).build_smoothing_matrix(
        [dipping_fault_mesh], 1.0
    )
    np.testing.assert_allclose(damped.toarray(), base.toarray())
