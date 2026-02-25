import pytest
import numpy as np
import matplotlib.pyplot as plt
from slipkit.utils.visualizers.gnss import GnssVisualizer
from slipkit.core.data import GeodeticDataSet

def test_plot_gnss_vectors():
    # Create a dummy GNSS GeodeticDataSet
    num_stations = 5
    coords = np.random.rand(num_stations, 3)
    
    # Stack E, N, U
    all_coords = np.vstack([coords, coords, coords])
    data = np.random.rand(num_stations * 3)
    
    unit_vecs = np.vstack([
        np.tile([1, 0, 0], (num_stations, 1)),
        np.tile([0, 1, 0], (num_stations, 1)),
        np.tile([0, 0, 1], (num_stations, 1))
    ])
    
    sigma = np.ones(num_stations * 3)
    
    dataset = GeodeticDataSet(
        coords=all_coords,
        data=data,
        unit_vecs=unit_vecs,
        sigma=sigma,
        name="Test_GNSS"
    )
    
    # Test plotting without showing (using return_fig_ax)
    fig_ax = GnssVisualizer.plot_gnss_vectors(dataset, return_fig_ax=True)
    
    assert fig_ax is not None
    fig, ax = fig_ax
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    
    plt.close(fig)

def test_plot_gnss_vectors_missing_components():
    # Create a dataset with only E and N
    num_stations = 5
    coords = np.random.rand(num_stations, 3)
    all_coords = np.vstack([coords, coords])
    data = np.random.rand(num_stations * 2)
    unit_vecs = np.vstack([
        np.tile([1, 0, 0], (num_stations, 1)),
        np.tile([0, 1, 0], (num_stations, 1))
    ])
    sigma = np.ones(num_stations * 2)
    
    dataset = GeodeticDataSet(
        coords=all_coords,
        data=data,
        unit_vecs=unit_vecs,
        sigma=sigma,
        name="Test_GNSS_Partial"
    )
    
    with pytest.raises(ValueError, match="Dataset does not contain all E, N, U components"):
        GnssVisualizer.plot_gnss_vectors(dataset)
