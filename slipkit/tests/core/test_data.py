import numpy as np
import pytest
from slipkit.core.data import GeodeticDataSet

@pytest.fixture
def sample_dataset():
    """Returns a sample GeodeticDataSet for testing."""
    n_points = 50
    return GeodeticDataSet(
        name="test_dataset",
        coords=np.random.rand(n_points, 3),
        data=np.random.rand(n_points),
        unit_vecs=np.random.rand(n_points, 3),
        sigma=np.random.rand(n_points),
    )

def test_geodetic_dataset_shapes(sample_dataset: GeodeticDataSet):
    """Verify the shapes of the dataset attributes."""
    n_points = 50
    assert len(sample_dataset) == n_points
    assert sample_dataset.coords.shape == (n_points, 3)
    assert sample_dataset.data.shape == (n_points,)
    assert sample_dataset.unit_vecs.shape == (n_points, 3)
    assert sample_dataset.sigma.shape == (n_points,)

def test_geodetic_dataset_nuisance(sample_dataset: GeodeticDataSet):
    """Verify that the nuisance basis is None."""
    assert sample_dataset.get_nuisance_basis() is None

def test_init_shape_mismatch():
    """Test that ValueError is raised on shape mismatch."""
    with pytest.raises(ValueError, match="All input arrays must have the same length"):
        GeodeticDataSet(
            name="mismatch",
            coords=np.random.rand(10, 3),
            data=np.random.rand(10),
            unit_vecs=np.random.rand(9, 3), # Mismatch here
            sigma=np.random.rand(10),
        )

def test_init_coord_dims():
    """Test that ValueError is raised on incorrect coordinate dimensions."""
    with pytest.raises(ValueError, match="Coordinates array must have shape"):
        GeodeticDataSet(
            name="wrong_dims",
            coords=np.random.rand(10, 2), # Mismatch here
            data=np.random.rand(10),
            unit_vecs=np.random.rand(10, 3),
            sigma=np.random.rand(10),
        )
