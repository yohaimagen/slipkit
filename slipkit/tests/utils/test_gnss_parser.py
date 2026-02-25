import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from slipkit.utils.parsers.gnss_parser import GnssParser
from slipkit.core.data import GeodeticDataSet

@pytest.fixture
def sample_gnss_csv(tmp_path):
    csv_path = tmp_path / "gnss_data.csv"
    data = {
        'Station': ['ST01', 'ST02'],
        'Lat': [35.0, 35.1],
        'Lon': [135.0, 135.1],
        'E': [0.1, 0.2],
        'N': [0.05, 0.06],
        'U': [-0.01, -0.02],
        'SigE': [0.01, 0.01],
        'SigN': [0.01, 0.01],
        'SigU': [0.02, 0.02]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path

def test_gnss_parser_read_csv(sample_gnss_csv):
    origin_lon = 135.0
    origin_lat = 35.0
    
    dataset = GnssParser.read_csv(sample_gnss_csv, origin_lon, origin_lat)
    
    assert isinstance(dataset, GeodeticDataSet)
    # 2 stations * 3 components = 6 data points
    assert len(dataset) == 6
    
    # Check data values
    # Order should be E (ST01, ST02), N (ST01, ST02), U (ST01, ST02)
    expected_data = np.array([0.1, 0.2, 0.05, 0.06, -0.01, -0.02])
    np.testing.assert_allclose(dataset.data, expected_data)
    
    # Check unit vectors
    # E: [1, 0, 0], N: [0, 1, 0], U: [0, 0, 1]
    assert np.allclose(dataset.unit_vecs[0], [1, 0, 0])
    assert np.allclose(dataset.unit_vecs[1], [1, 0, 0])
    assert np.allclose(dataset.unit_vecs[2], [0, 1, 0])
    assert np.allclose(dataset.unit_vecs[3], [0, 1, 0])
    assert np.allclose(dataset.unit_vecs[4], [0, 0, 1])
    assert np.allclose(dataset.unit_vecs[5], [0, 0, 1])

def test_gnss_parser_partial_components(sample_gnss_csv):
    origin_lon = 135.0
    origin_lat = 35.0
    
    dataset = GnssParser.read_csv(
        sample_gnss_csv, 
        origin_lon, 
        origin_lat, 
        components=['E', 'N']
    )
    
    # 2 stations * 2 components = 4 data points
    assert len(dataset) == 4
    expected_data = np.array([0.1, 0.2, 0.05, 0.06])
    np.testing.assert_allclose(dataset.data, expected_data)
