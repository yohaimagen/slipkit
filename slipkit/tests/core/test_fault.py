import numpy as np
import pytest
from scipy.sparse import issparse
from slipkit.core.fault import (
    AbstractFaultModel,
    TriangularFaultMesh,
    SlipComponent,
)

# Test for AbstractFaultModel
def test_abstract_fault_model_instantiation():
    """Verify that AbstractFaultModel cannot be instantiated directly."""
    expected_regex = (
        "Can't instantiate abstract class AbstractFaultModel without an "
        "implementation for abstract methods 'get_centroids', 'get_mesh_geometry', "
        "'get_smoothing_matrix', 'num_patches'"
    )
    with pytest.raises(TypeError, match=expected_regex):
        AbstractFaultModel()

# Tests for TriangularFaultMesh
@pytest.fixture
def sample_mesh_data():
    """Returns sample vertices and faces for TriangularFaultMesh."""
    vertices = np.array([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [1.0, 1.0, 0.0],  # 3
        [2.0, 0.0, 0.0],  # 4
    ])
    # f0 neighbors f1
    # f1 neighbors f0, f2
    # f2 neighbors f1
    faces = np.array([
        [0, 1, 2],  # f0
        [1, 3, 2],  # f1
        [1, 4, 3],  # f2
    ])
    return vertices, faces

def test_triangular_fault_mesh_init_arrays(sample_mesh_data):
    """Test initialization with numpy arrays and verify attributes."""
    vertices, faces = sample_mesh_data
    mesh = TriangularFaultMesh(mesh_input=(vertices, faces))
    assert np.array_equal(mesh.vertices, vertices)
    assert np.array_equal(mesh.faces, faces)
    assert mesh.num_patches() == 3

def test_triangular_fault_mesh_init_from_file():
    """Test initialization from a .msh file."""
    mesh = TriangularFaultMesh(mesh_input="slipkit/tests/core/sample_mesh.msh")
    assert mesh.num_patches() == 3
    assert mesh.vertices.shape == (5, 3)

def test_init_invalid_faces_shape():
    """Test ValueError for invalid faces array shape."""
    vertices = np.random.rand(5, 3)
    invalid_faces = np.random.randint(0, 4, size=(3, 2))
    with pytest.raises(ValueError, match=r"Faces array must represent triangles \(N, 3\)."):
        TriangularFaultMesh(mesh_input=(vertices, invalid_faces))

def test_init_invalid_input_type():
    """Test ValueError for invalid mesh_input type."""
    with pytest.raises(ValueError, match=r"mesh_input must be a file path \(str\) or a tuple of \(vertices, faces\) arrays."):
        TriangularFaultMesh(mesh_input=[1, 2, 3])

def test_get_mesh_geometry(sample_mesh_data):
    """Test the get_mesh_geometry method."""
    vertices, faces = sample_mesh_data
    mesh = TriangularFaultMesh((vertices, faces))
    v, f = mesh.get_mesh_geometry()
    assert np.array_equal(v, vertices)
    assert np.array_equal(f, faces)

def test_get_centroids(sample_mesh_data):
    """Test the get_centroids method."""
    vertices, faces = sample_mesh_data
    mesh = TriangularFaultMesh((vertices, faces))
    centroids = mesh.get_centroids()
    
    expected_c0 = np.mean(vertices[[0, 1, 2]], axis=0)
    expected_c1 = np.mean(vertices[[1, 3, 2]], axis=0)
    
    assert centroids.shape == (3, 3)
    assert np.allclose(centroids[0], expected_c0)
    assert np.allclose(centroids[1], expected_c1)

def test_get_smoothing_matrix(sample_mesh_data):
    """Test the get_smoothing_matrix method."""
    vertices, faces = sample_mesh_data
    mesh = TriangularFaultMesh((vertices, faces))
    laplacian = mesh.get_smoothing_matrix()

    assert issparse(laplacian)
    assert laplacian.shape == (3, 3)

    # Adjacency: f0-f1, f1-f2
    # L_ii = degree, L_ij = -1 for neighbors
    expected = np.array([
        [1, -1, 0],
        [-1, 2, -1],
        [0, -1, 1],
    ])
    
    assert np.allclose(laplacian.toarray(), expected)
    # Verify row sums are zero
    assert np.allclose(laplacian.sum(axis=1), np.zeros((3, 1)))

def test_unsupported_smoothing_type(sample_mesh_data):
    """Test that an unsupported smoothing type raises an error."""
    vertices, faces = sample_mesh_data
    mesh = TriangularFaultMesh((vertices, faces))
    with pytest.raises(NotImplementedError):
        mesh.get_smoothing_matrix(type='gradient')


# --- Slip component selection ---

def test_default_components_are_both(sample_mesh_data):
    """By default a fault inverts for strike-slip and dip-slip in canonical order."""
    vertices, faces = sample_mesh_data
    mesh = TriangularFaultMesh((vertices, faces))
    assert mesh.num_components() == 2
    assert mesh.active_components() == (
        SlipComponent.STRIKE_SLIP,
        SlipComponent.DIP_SLIP,
    )


def test_strike_slip_only_components(sample_mesh_data):
    """A strike-slip-only fault reports a single component and slice."""
    vertices, faces = sample_mesh_data
    mesh = TriangularFaultMesh(
        (vertices, faces), slip_components=[SlipComponent.STRIKE_SLIP]
    )
    m = mesh.num_patches()
    assert mesh.num_components() == 1
    assert mesh.active_components() == (SlipComponent.STRIKE_SLIP,)
    assert mesh.component_slice(SlipComponent.STRIKE_SLIP) == slice(0, m)
    # dip-slip is inactive -> no slice
    assert mesh.component_slice(SlipComponent.DIP_SLIP) is None


def test_dip_slip_only_components(sample_mesh_data):
    """A dip-slip-only fault puts dip-slip in the first block."""
    vertices, faces = sample_mesh_data
    mesh = TriangularFaultMesh(
        (vertices, faces), slip_components=[SlipComponent.DIP_SLIP]
    )
    m = mesh.num_patches()
    assert mesh.num_components() == 1
    assert mesh.component_slice(SlipComponent.DIP_SLIP) == slice(0, m)
    assert mesh.component_slice(SlipComponent.STRIKE_SLIP) is None


def test_components_canonicalized(sample_mesh_data):
    """Order is canonicalized regardless of the order passed in."""
    vertices, faces = sample_mesh_data
    m = TriangularFaultMesh(
        (vertices, faces),
        slip_components=[SlipComponent.DIP_SLIP, SlipComponent.STRIKE_SLIP],
    )
    n = m.num_patches()
    assert m.active_components() == (
        SlipComponent.STRIKE_SLIP,
        SlipComponent.DIP_SLIP,
    )
    assert m.component_slice(SlipComponent.STRIKE_SLIP) == slice(0, n)
    assert m.component_slice(SlipComponent.DIP_SLIP) == slice(n, 2 * n)


def test_component_slice_accepts_string(sample_mesh_data):
    """component_slice coerces string aliases."""
    vertices, faces = sample_mesh_data
    m = TriangularFaultMesh((vertices, faces))
    assert m.component_slice("ss") == m.component_slice(SlipComponent.STRIKE_SLIP)
    assert m.component_slice("dip") == m.component_slice(SlipComponent.DIP_SLIP)


def test_empty_components_raises(sample_mesh_data):
    """An empty component list is rejected."""
    vertices, faces = sample_mesh_data
    with pytest.raises(ValueError, match="at least one SlipComponent"):
        TriangularFaultMesh((vertices, faces), slip_components=[])


def test_duplicate_components_raises(sample_mesh_data):
    """Duplicate components are rejected."""
    vertices, faces = sample_mesh_data
    with pytest.raises(ValueError, match="duplicate"):
        TriangularFaultMesh(
            (vertices, faces),
            slip_components=[SlipComponent.STRIKE_SLIP, SlipComponent.STRIKE_SLIP],
        )


def test_slip_component_coerce():
    """SlipComponent.coerce maps aliases and rejects unknowns."""
    assert SlipComponent.coerce("ss") is SlipComponent.STRIKE_SLIP
    assert SlipComponent.coerce("strike") is SlipComponent.STRIKE_SLIP
    assert SlipComponent.coerce("ds") is SlipComponent.DIP_SLIP
    assert SlipComponent.coerce(SlipComponent.DIP_SLIP) is SlipComponent.DIP_SLIP
    with pytest.raises(ValueError, match="Unknown slip component"):
        SlipComponent.coerce("tensile")