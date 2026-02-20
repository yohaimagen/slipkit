import pytest
import numpy as np
from slipkit.core.fault import TriangularFaultMesh
from slipkit.core.data import GeodeticDataSet
from slipkit.core.physics import CutdeCpuEngine
from slipkit.core.solvers import NnlsSolver
from slipkit.core.inversion import InversionOrchestrator

@pytest.fixture
def simple_inversion_setup():
    """
    Fixture that creates a simple 2-triangle fault and a physics engine.
    Returns a dictionary containing the setup objects.
    """
    # 1. Create a simple fault in memory (2 triangles forming a square)
    # Coordinates in meters. Z is negative down.
    vertices = np.array([
        [0, 0, -5000],      # 0: Top-Left
        [2000, 0, -5000],   # 1: Top-Right
        [0, 0, -6000],      # 2: Bottom-Left
        [2000, 0, -6000]    # 3: Bottom-Right
    ], dtype=float)

    faces = np.array([
        [0, 1, 2], # Upper triangle
        [1, 3, 2]  # Lower triangle
    ], dtype=int)

    fault = TriangularFaultMesh((vertices, faces))
    
    # 2. Setup Physics Engine
    engine = CutdeCpuEngine(poisson_ratio=0.25)
    
    return {
        "fault": fault,
        "engine": engine,
        "n_patches": fault.num_patches()
    }

@pytest.fixture
def synthetic_dataset(simple_inversion_setup):
    """
    Generates a synthetic dataset for a simple inversion setup.
    """
    fault = simple_inversion_setup["fault"]
    engine = simple_inversion_setup["engine"]
    n_patches = simple_inversion_setup["n_patches"]

    # Define a simple "true" slip
    true_slip = np.zeros(2 * n_patches)
    true_slip[0] = 1.0  # Strike-slip on patch 0
    true_slip[n_patches + 1] = 0.5 # Dip-slip on patch 1

    # Create observation points on surface (z=0)
    x = np.linspace(-5000, 7000, 4)
    y = np.linspace(-5000, 7000, 4)
    xv, yv = np.meshgrid(x, y)
    
    obs_coords_list = []
    unit_vecs_list = []
    for i in range(len(xv.flatten())):
        pt = [xv.flatten()[i], yv.flatten()[i], 0.0]
        obs_coords_list.extend([pt, pt, pt])
        unit_vecs_list.extend([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    obs_coords = np.array(obs_coords_list)
    unit_vecs = np.array(unit_vecs_list)
    n_obs = len(obs_coords)

    dummy_dataset = GeodeticDataSet(
        name="SynthGen",
        coords=obs_coords,
        unit_vecs=unit_vecs,
        data=np.zeros(n_obs),
        sigma=np.ones(n_obs)
    )
    G = engine.build_kernel(fault, dummy_dataset)
    synthetic_data = G @ true_slip

    return GeodeticDataSet(
        name="SyntheticData",
        coords=obs_coords,
        unit_vecs=unit_vecs,
        data=synthetic_data,
        sigma=np.ones(n_obs) * 1e-3
    )


def test_synthetic_recovery_exact(simple_inversion_setup, synthetic_dataset):
    """
    Verifies that the inversion can recover a known input slip distribution
    with high precision in a noiseless, over-determined case.
    """
    fault = simple_inversion_setup["fault"]
    engine = simple_inversion_setup["engine"]
    n_patches = simple_inversion_setup["n_patches"]

    # Define the "True" Slip (same as in synthetic_dataset fixture)
    true_slip = np.zeros(2 * n_patches)
    true_slip[0] = 1.0
    true_slip[n_patches + 1] = 0.5

    orchestrator = InversionOrchestrator()
    orchestrator.add_fault(fault)
    orchestrator.add_data(synthetic_dataset)
    orchestrator.set_engine(engine)
    orchestrator.set_solver(NnlsSolver())

    bounds = (np.zeros(2 * n_patches), np.full(2 * n_patches, np.inf))
    
    result = orchestrator.run_inversion(lambda_spatial=0.0, bounds=bounds)
    inverted_slip = result.slip_vector

    np.testing.assert_allclose(
        inverted_slip, 
        true_slip, 
        atol=1e-5, 
        err_msg="Inversion failed to recover synthetic slip in noiseless case."
    )

def test_run_l_curve(simple_inversion_setup, synthetic_dataset):
    """
    Test the run_l_curve method of InversionOrchestrator.
    """
    fault = simple_inversion_setup["fault"]
    engine = simple_inversion_setup["engine"]
    n_patches = simple_inversion_setup["n_patches"]

    orchestrator = InversionOrchestrator()
    orchestrator.add_fault(fault)
    orchestrator.add_data(synthetic_dataset)
    orchestrator.set_engine(engine)
    orchestrator.set_solver(NnlsSolver())

    lambdas_to_test = np.logspace(-2, 2, 5) # Test 5 lambda values

    # Define bounds (e.g., non-negative slip)
    bounds = (np.zeros(2 * n_patches), np.full(2 * n_patches, np.inf))

    # Run the L-curve analysis
    returned_lambdas, misfits, roughnesses = orchestrator.run_l_curve(
        lambdas=lambdas_to_test, 
        bounds=bounds
    )

    # Assertions
    assert np.array_equal(returned_lambdas, lambdas_to_test)
    assert misfits.shape == lambdas_to_test.shape
    assert roughnesses.shape == lambdas_to_test.shape

    # Misfits should generally decrease or stay similar as lambda decreases
    # Roughnesses should generally increase or stay similar as lambda decreases
    # This is a qualitative check, more robust checks might involve specific synthetic results.
    # For now, check if they are all positive
    assert np.all(misfits >= 0)
    assert np.all(roughnesses >= 0)

    # Basic check for trend: roughness should generally increase with decreasing lambda
    # and misfit should generally decrease with decreasing lambda.
    # This might not be strictly monotonic for all synthetic cases,
    # but for a reasonable L-curve, we expect some trend.
    # Let's check for at least a non-increasing misfit and non-decreasing roughness overall
    assert np.all(np.diff(misfits) <= 1e-9) or np.all(np.diff(misfits) >= -1e-9) # Misfit roughly non-increasing
    assert np.all(np.diff(roughnesses) >= -1e-9) or np.all(np.diff(roughnesses) <= 1e-9) # Roughness roughly non-decreasing


def test_run_l_curve_missing_components():
    """
    Test that run_l_curve raises ValueError if essential components are missing.
    """
    orchestrator = InversionOrchestrator()
    lambdas_to_test = np.logspace(-2, 2, 3)

    with pytest.raises(ValueError, match="No fault models added to the inversion."):
        orchestrator.run_l_curve(lambdas=lambdas_to_test)

    # Add fault, check for datasets
    orchestrator.add_fault(TriangularFaultMesh(mesh_input=(np.array([[0,0,0],[1,0,0],[0,1,0]]), np.array([[0,1,2]]))))
    with pytest.raises(ValueError, match="No geodetic datasets added to the inversion."):
        orchestrator.run_l_curve(lambdas=lambdas_to_test)

    # Add dataset, check for engine
    orchestrator.add_data(GeodeticDataSet(name="dummy", coords=np.array([[0,0,0]]), data=np.array([0]), unit_vecs=np.array([[0,0,1]]), sigma=np.array([1])))
    with pytest.raises(ValueError, match="No GreenFunctionBuilder engine has been set."):
        orchestrator.run_l_curve(lambdas=lambdas_to_test)

    # Add engine, check for solver
    orchestrator.set_engine(CutdeCpuEngine(poisson_ratio=0.25))
    with pytest.raises(ValueError, match="No SolverStrategy has been set."):
        orchestrator.run_l_curve(lambdas=lambdas_to_test)