import pytest
import numpy as np
import pymc as pm
import arviz as az
from slipkit.core.bayesian.assembler import BayesianAssembler
from slipkit.core.bayesian.solver import BayesianSolver
from slipkit.core.bayesian.results import BayesianSlipDistribution
from slipkit.core.inversion import InversionOrchestrator
from slipkit.core.fault import TriangularFaultMesh
from slipkit.core.data import GeodeticDataSet
from slipkit.core.physics import CutdeCpuEngine

@pytest.fixture
def bayesian_setup():
    """Sets up a simple fault and synthetic data for Bayesian testing."""
    # 1. Simple Fault (1 triangle)
    vertices = np.array([
        [0, 0, -5000],
        [2000, 0, -5000],
        [1000, 1732, -5000]
    ], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    fault = TriangularFaultMesh((vertices, faces))
    n_patches = fault.num_patches()
    areas = np.array([1.732e6]) # Area of the triangle in m^2

    # 2. Synthetic Data Generation
    engine = CutdeCpuEngine()
    
    # Target rake = 45 degrees
    target_rake = 45.0
    true_u_par = 1.5
    true_u_perp = 0.2
    
    # Observation points
    x = np.linspace(-2000, 4000, 5)
    y = np.linspace(-2000, 4000, 5)
    xv, yv = np.meshgrid(x, y)
    obs_coords = np.column_stack([xv.flatten(), yv.flatten(), np.zeros_like(xv.flatten())])
    n_pts = len(obs_coords)
    
    # We need 3 components per point (E, N, U)
    full_obs_coords = np.repeat(obs_coords, 3, axis=0)
    unit_vecs = np.tile(np.eye(3), (n_pts, 1))
    
    dummy_ds = GeodeticDataSet(
        coords=full_obs_coords, 
        data=np.zeros(len(full_obs_coords)), 
        unit_vecs=unit_vecs, 
        sigma=np.ones(len(full_obs_coords)), 
        name="Synth"
    )
    
    # Standard G (Strike, Dip)
    G_raw = engine.build_kernel(fault, dummy_ds)
    G_ss = G_raw[:, :n_patches]
    G_ds = G_raw[:, n_patches:]
    
    # Convert true slip (par, perp) to (ss, ds)
    # par = ss*cos + ds*sin
    # perp = -ss*sin + ds*cos
    # Solving for ss, ds:
    # [cos sin] [ss] = [par]
    # [-sin cos] [ds] = [perp]
    # Rotation matrix R = [[cos, sin], [-sin, cos]] is orthogonal. 
    # [ss, ds]^T = R^T [par, perp]^T = [[cos, -sin], [sin, cos]] [par, perp]^T
    rad = np.radians(target_rake)
    cos, sin = np.cos(rad), np.sin(rad)
    true_ss = true_u_par * cos - true_u_perp * sin
    true_ds = true_u_par * sin + true_u_perp * cos
    
    true_slip_raw = np.concatenate([np.full(n_patches, true_ss), np.full(n_patches, true_ds)])
    
    # Generate noiseless data
    clean_data = G_raw @ true_slip_raw
    
    # Add "Model Error" noise (alpha = 0.05)
    alpha_true = 0.05
    noise_sigma = 0.01 # Small constant data noise
    total_sigma = np.sqrt(noise_sigma**2 + (alpha_true * clean_data)**2)
    observed_data = clean_data + np.random.normal(0, total_sigma)
    
    dataset = GeodeticDataSet(
        coords=full_obs_coords,
        data=observed_data,
        unit_vecs=unit_vecs,
        sigma=np.full_like(observed_data, noise_sigma),
        name="BayesianSynth"
    )
    
    return {
        "fault": fault,
        "dataset": dataset,
        "engine": engine,
        "areas": areas,
        "target_rake": target_rake,
        "true_u_par": true_u_par,
        "true_u_perp": true_u_perp,
        "alpha_true": alpha_true
    }

def test_bayesian_assembler_rotation(bayesian_setup):
    """Verifies that the BayesianAssembler rotates kernels correctly."""
    setup = bayesian_setup
    assembler = BayesianAssembler(target_rake_deg=setup["target_rake"])
    
    A, b = assembler.assemble(
        [setup["fault"]], 
        [setup["dataset"]], 
        setup["engine"], 
        None, 0.0
    )
    
    n_patches = setup["fault"].num_patches()
    G_par = A[:, :n_patches]
    G_perp = A[:, n_patches:]
    
    # Check rotation manually
    # G_par = G_ss * cos + G_ds * sin
    # G_perp = -G_ss * sin + G_ds * cos
    G_raw = setup["engine"].build_kernel(setup["fault"], setup["dataset"])
    G_ss = G_raw[:, :n_patches]
    G_ds = G_raw[:, n_patches:]
    
    rad = np.radians(setup["target_rake"])
    expected_G_par = G_ss * np.cos(rad) + G_ds * np.sin(rad)
    expected_G_perp = -G_ss * np.sin(rad) + G_ds * np.cos(rad)
    
    np.testing.assert_allclose(G_par, expected_G_par, atol=1e-7)
    np.testing.assert_allclose(G_perp, expected_G_perp, atol=1e-7)

def test_bayesian_orchestration(bayesian_setup):
    """Verifies the end-to-end orchestration of a Bayesian inversion."""
    setup = bayesian_setup
    
    # We use very few draws/chains for a fast unit test
    solver = BayesianSolver(
        mu_mw=6.0, 
        sigma_mw=0.1, 
        areas=setup["areas"],
        draws=500, 
        chains=2,
        random_seed=42
    )
    
    assembler = BayesianAssembler(target_rake_deg=setup["target_rake"])
    
    orchestrator = InversionOrchestrator()
    orchestrator.add_fault(setup["fault"])
    orchestrator.add_data(setup["dataset"])
    orchestrator.set_engine(setup["engine"])
    orchestrator.set_assembler(assembler)
    orchestrator.set_solver(solver)
    
    # run_inversion currently returns a SlipDistribution. 
    # We need to ensure it works with the Bayesian components.
    # Note: In a real scenario, we might want it to return a BayesianSlipDistribution.
    # For now, let's verify if the core logic executes.
    result = orchestrator.run_inversion(lambda_spatial=0.0)
    
    assert isinstance(result.slip_vector, np.ndarray)
    assert len(result.slip_vector) == 2 * setup["fault"].num_patches()
    
    # Retrieve the trace from the solver
    trace = solver.get_inference_data()
    assert trace is not None
    assert "u_par" in trace.posterior
    assert "u_perp" in trace.posterior
    assert "alpha" in trace.posterior
    
    # Check if mean recovery is reasonable (with 500 draws on a 1-patch problem it should be okay)
    mean_u_par = trace.posterior["u_par"].mean().values
    assert np.abs(mean_u_par - setup["true_u_par"]) < 0.5 # Generous tolerance for fast test
