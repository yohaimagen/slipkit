# slipkit/tests/utils/test_l_curve.py
import pytest
import numpy as np
import matplotlib.pyplot as plt

from slipkit.utils.visualizers.l_curve import LCurveVisualizer

@pytest.fixture
def sample_l_curve_data():
    """Provides sample L-curve data for testing."""
    lambdas = np.logspace(-3, 3, 10)
    # Generate synthetic data that resembles an L-curve
    misfits = 1 / (1 + lambdas**2) + 0.1
    roughnesses = lambdas**2 / (1 + lambdas**2) + 0.1
    return lambdas, misfits, roughnesses

def test_find_corner(sample_l_curve_data):
    """Test that find_corner runs without error and returns plausible values."""
    lambdas, misfits, roughnesses = sample_l_curve_data
    optimal_lambda, corner_misfit, corner_roughness = LCurveVisualizer.find_corner(
        lambdas, misfits, roughnesses
    )
    assert isinstance(optimal_lambda, float)
    assert isinstance(corner_misfit, float)
    assert isinstance(corner_roughness, float)
    assert optimal_lambda > 0

def test_plot_l_curve_no_corner(sample_l_curve_data):
    """Test that plot_l_curve runs without error when plot_corner is False."""
    lambdas, misfits, roughnesses = sample_l_curve_data
    fig, ax = plt.subplots(1, 1)
    LCurveVisualizer.plot_l_curve(
        lambdas, misfits, roughnesses, ax=ax, plot_corner=False
    )
    plt.close(fig)

def test_plot_l_curve_with_corner(sample_l_curve_data):
    """Test that plot_l_curve runs without error when plot_corner is True."""
    lambdas, misfits, roughnesses = sample_l_curve_data
    fig, ax = plt.subplots(1, 1)
    LCurveVisualizer.plot_l_curve(
        lambdas, misfits, roughnesses, ax=ax, plot_corner=True
    )
    plt.close(fig)
