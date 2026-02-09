import numpy as np
import pytest
from slipkit.core.fault import AbstractFaultModel, TriangularFaultMesh

# Test for AbstractFaultModel
def test_abstract_fault_model_instantiation():
    """Verify that AbstractFaultModel cannot be instantiated directly."""
    # The actual error message includes the names of the abstract methods
    expected_regex = (
        "Can't instantiate abstract class AbstractFaultModel without an "
        "implementation for abstract methods 'get_centroids', 'get_mesh_geometry', "
        "'get_smoothing_matrix', 'num_patches'"
    )
    with pytest.raises(TypeError, match=expected_regex):
        AbstractFaultModel()

# Tests for TriangularFaultMesh
@pytest.fixture
def sample_triangular_mesh_data():
    """Returns sample vertices and faces for TriangularFaultMesh."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    faces = np.array([
        [0, 1, 2],
        [1, 3, 2],
        [1, 4, 3],
    ])
    return vertices, faces

def test_triangular_fault_mesh_init_arrays(sample_triangular_mesh_data):
    """
    Test initialization of TriangularFaultMesh with direct numpy arrays
    and verify stored attributes and num_patches.
    """
    vertices, faces = sample_triangular_mesh_data
    mesh = TriangularFaultMesh(mesh_input=(vertices, faces))

    assert np.array_equal(mesh.vertices, vertices)
    assert np.array_equal(mesh.faces, faces)
    assert mesh.num_patches() == faces.shape[0]
    assert mesh.num_patches() == 3

def test_triangular_fault_mesh_init_file_not_implemented():
    """
    Test that initialization with a file path raises NotImplementedError.
    """
    with pytest.raises(NotImplementedError, match="File parsing for .msh or .stl is not yet implemented."):
        TriangularFaultMesh(mesh_input="path/to/mesh.msh")

def test_triangular_fault_mesh_init_invalid_faces_shape():
    """
    Test that initialization with an invalid faces array shape raises ValueError.
    """
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    invalid_faces = np.array([[0, 1], [1, 2]]) # Not (N, 3)
    with pytest.raises(ValueError, match=r"Faces array must represent triangles \(N, 3\)\."):
        TriangularFaultMesh(mesh_input=(vertices, invalid_faces))

def test_triangular_fault_mesh_init_invalid_input_type():
    """
    Test that initialization with an invalid input type raises ValueError.
    """
    with pytest.raises(ValueError, match=r"mesh_input must be a file path \(str\) or a tuple of \(vertices, faces\) arrays\."):
        TriangularFaultMesh(mesh_input=[1, 2, 3]) # Invalid type

def test_triangular_fault_mesh_num_patches(sample_triangular_mesh_data):
    """
    Test the num_patches method directly.
    """
    vertices, faces = sample_triangular_mesh_data
    mesh = TriangularFaultMesh(mesh_input=(vertices, faces))
    assert mesh.num_patches() == faces.shape[0]
