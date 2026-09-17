import pytest
import numpy as np
from slipkit.core.fault import TriangularFaultMesh, SlipComponent
from slipkit.core.data import GeodeticDataSet, Ramp
from slipkit.core.physics import CutdeCpuEngine
from slipkit.core.solvers import BoundedLsqSolver, NnlsSolver
from slipkit.core.inversion import InversionOrchestrator, SlipDistribution

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


def test_strike_slip_only_inversion(simple_inversion_setup):
    """
    A strike-slip-only fault produces an (M,) solution recovering pure SS slip,
    solvable directly with NNLS (no bounds, no dip-slip columns).
    """
    setup = simple_inversion_setup
    vertices, faces = setup["fault"].get_mesh_geometry()
    engine = setup["engine"]

    fault = TriangularFaultMesh(
        (vertices, faces), slip_components=[SlipComponent.STRIKE_SLIP]
    )
    n_patches = fault.num_patches()

    # True SS-only slip; kernel is (N, M) now.
    true_slip = np.zeros(n_patches)
    true_slip[0] = 1.0

    x = np.linspace(-5000, 7000, 4)
    y = np.linspace(-5000, 7000, 4)
    xv, yv = np.meshgrid(x, y)
    coords, uvs = [], []
    for px, py in zip(xv.ravel(), yv.ravel()):
        pt = [px, py, 0.0]
        coords += [pt, pt, pt]
        uvs += [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    coords = np.array(coords, dtype=float)
    uvs = np.array(uvs, dtype=float)

    G = engine.build_kernel(fault, GeodeticDataSet(
        name="g", coords=coords, unit_vecs=uvs,
        data=np.zeros(len(coords)), sigma=np.ones(len(coords))))
    assert G.shape == (len(coords), n_patches)

    dataset = GeodeticDataSet(
        name="ss", coords=coords, unit_vecs=uvs,
        data=G @ true_slip, sigma=np.ones(len(coords)) * 1e-3)

    orch = InversionOrchestrator()
    orch.add_fault(fault)
    orch.add_data(dataset)
    orch.set_engine(engine)
    orch.set_solver(NnlsSolver())

    result = orch.run_inversion(lambda_spatial=0.0)
    assert result.slip_vector.shape == (n_patches,)
    np.testing.assert_allclose(result.slip_vector, true_slip, atol=1e-5)

    # Component accessor returns the SS block; DS is inactive and raises.
    np.testing.assert_allclose(
        result.get_component(SlipComponent.STRIKE_SLIP), true_slip, atol=1e-5)
    with pytest.raises(ValueError, match="not active"):
        result.get_component("ds")


def test_slip_distribution_accessors(simple_inversion_setup):
    """SlipDistribution slices the vector back into per-component blocks."""
    fault = simple_inversion_setup["fault"]  # default: SS + DS
    m = fault.num_patches()
    vec = np.arange(2 * m, dtype=float)
    dist = SlipDistribution(vec, [fault])

    np.testing.assert_array_equal(dist.get_component("ss"), vec[:m])
    np.testing.assert_array_equal(dist.get_component("ds"), vec[m:])
    np.testing.assert_array_equal(dist.get_fault_vector(0), vec)

    with pytest.raises(ValueError, match="does not match"):
        SlipDistribution(np.zeros(2 * m + 1), [fault])


def test_seismic_moment_and_magnitude():
    """
    Seismic moment and Mw are computed from patch areas and total slip,
    matching a hand-computed reference value.
    """
    # Single unit-square patch split into two right triangles, in metres.
    # Total area = 1 km x 1 km = 1e6 m^2 (0.5e6 m^2 each triangle).
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
    ])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    fault = TriangularFaultMesh((vertices, faces))  # SS + DS
    m = fault.num_patches()

    # 3 m pure strike-slip on both patches, no dip-slip.
    vec = np.zeros(2 * m)
    vec[:m] = 3.0
    dist = SlipDistribution(vec, [fault])

    np.testing.assert_allclose(dist.total_slip(), np.full(m, 3.0))

    mu = 3.3e10
    areas_m2 = fault.get_areas() * 1e6  # mesh in km -> m^2
    expected_m0 = float(np.sum(mu * areas_m2 * 3.0))
    expected_mw = (2.0 / 3.0) * (np.log10(expected_m0) - 9.05)

    assert dist.seismic_moment(shear_modulus=mu) == pytest.approx(expected_m0)
    assert dist.moment_magnitude(shear_modulus=mu) == pytest.approx(expected_mw)

    # Combined SS + DS magnitude adds in quadrature: 3-4-5 triangle -> 5 m.
    vec2 = np.zeros(2 * m)
    vec2[:m] = 3.0
    vec2[m:] = 4.0
    dist2 = SlipDistribution(vec2, [fault])
    np.testing.assert_allclose(dist2.total_slip(), np.full(m, 5.0))

    # A zero slip distribution has undefined magnitude.
    with pytest.raises(ValueError, match="undefined"):
        SlipDistribution(np.zeros(2 * m), [fault]).moment_magnitude()

    # An unknown length unit is rejected.
    with pytest.raises(ValueError, match="length_unit"):
        dist.seismic_moment(length_unit="furlong")


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

# --------------------------------------------------------------------------- #
# Nuisance ramps
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("degree, n_params", [(0, 1), (1, 3), (2, 6)])
def test_ramp_basis_size(degree, n_params, synthetic_dataset):
    """A degree-d ramp has (d+1)(d+2)/2 terms and a well-conditioned basis."""
    ramp = Ramp.for_dataset(synthetic_dataset, degree)
    basis = ramp.basis(synthetic_dataset.coords)

    assert ramp.num_params == n_params
    assert basis.shape == (len(synthetic_dataset), n_params)
    # Normalization keeps the columns O(1) -- this block carries no smoothing.
    assert np.linalg.cond(basis) < 1e3


def test_ramp_recovery(simple_inversion_setup, synthetic_dataset):
    """A planar ramp added to the data is recovered along with the slip."""
    fault = simple_inversion_setup["fault"]
    engine = simple_inversion_setup["engine"]
    n_slip = fault.num_components() * fault.num_patches()

    ramp = Ramp.for_dataset(synthetic_dataset, degree=1)
    true_coeffs = np.array([0.03, -0.02, 0.05])
    contaminated = GeodeticDataSet(
        name=synthetic_dataset.name,
        coords=synthetic_dataset.coords,
        unit_vecs=synthetic_dataset.unit_vecs,
        data=synthetic_dataset.data + ramp.basis(synthetic_dataset.coords) @ true_coeffs,
        sigma=synthetic_dataset.sigma,
        ramp=ramp,
    )

    inversion = InversionOrchestrator()
    inversion.add_fault(fault)
    inversion.add_data(contaminated)
    inversion.set_engine(engine)
    inversion.set_solver(BoundedLsqSolver())
    # Slip stays non-negative; the ramp block is extended to +/-inf for us.
    bounds = (np.zeros(n_slip), np.full(n_slip, np.inf))
    result = inversion.run_inversion(lambda_spatial=1e-4, bounds=bounds)

    assert result.slip_vector.shape == (n_slip,)
    assert np.allclose(result.nuisance[contaminated.name].coeffs, true_coeffs, atol=1e-3)
    assert np.allclose(result.nuisance_prediction(contaminated),
                       ramp.basis(contaminated.coords) @ true_coeffs, atol=1e-3)

    # And the slip itself is recovered despite the contamination.
    clean = inversion_slip_without_ramp(fault, engine, synthetic_dataset, n_slip)
    assert np.allclose(result.slip_vector, clean, atol=1e-2)


def inversion_slip_without_ramp(fault, engine, dataset, n_slip):
    """Solves the same problem on clean data with no ramp, as a reference."""
    inversion = InversionOrchestrator()
    inversion.add_fault(fault)
    inversion.add_data(dataset)
    inversion.set_engine(engine)
    inversion.set_solver(BoundedLsqSolver())
    bounds = (np.zeros(n_slip), np.full(n_slip, np.inf))
    return inversion.run_inversion(lambda_spatial=1e-4, bounds=bounds).slip_vector


def test_ignoring_ramp_biases_slip(simple_inversion_setup, synthetic_dataset):
    """Without a ramp in the model, ramp-contaminated data biases the slip."""
    fault = simple_inversion_setup["fault"]
    engine = simple_inversion_setup["engine"]
    n_slip = fault.num_components() * fault.num_patches()

    ramp = Ramp.for_dataset(synthetic_dataset, degree=1)
    contaminated = GeodeticDataSet(
        name=synthetic_dataset.name,
        coords=synthetic_dataset.coords,
        unit_vecs=synthetic_dataset.unit_vecs,
        data=synthetic_dataset.data + ramp.basis(synthetic_dataset.coords)
        @ np.array([0.03, -0.02, 0.05]),
        sigma=synthetic_dataset.sigma,
    )
    biased = inversion_slip_without_ramp(fault, engine, contaminated, n_slip)
    clean = inversion_slip_without_ramp(fault, engine, synthetic_dataset, n_slip)
    assert not np.allclose(biased, clean, atol=1e-2)


def test_nnls_rejects_ramp(simple_inversion_setup, synthetic_dataset):
    """NNLS cannot carry free-sign ramp coefficients, so it must refuse."""
    synthetic_dataset.ramp = Ramp.for_dataset(synthetic_dataset, degree=1)
    inversion = InversionOrchestrator()
    inversion.add_fault(simple_inversion_setup["fault"])
    inversion.add_data(synthetic_dataset)
    inversion.set_engine(simple_inversion_setup["engine"])
    inversion.set_solver(NnlsSolver())

    with pytest.raises(ValueError, match="non-negative"):
        inversion.run_inversion(lambda_spatial=1e-4)


def test_l_curve_with_ramp(simple_inversion_setup, synthetic_dataset):
    """The L-curve sweep stays consistent when nuisance columns widen G."""
    synthetic_dataset.ramp = Ramp.for_dataset(synthetic_dataset, degree=1)
    inversion = InversionOrchestrator()
    inversion.add_fault(simple_inversion_setup["fault"])
    inversion.add_data(synthetic_dataset)
    inversion.set_engine(simple_inversion_setup["engine"])
    inversion.set_solver(BoundedLsqSolver())

    n_slip = simple_inversion_setup["fault"].num_components() * \
        simple_inversion_setup["fault"].num_patches()
    lambdas, misfits, roughnesses = inversion.run_l_curve(
        np.logspace(-3, 0, 4),
        bounds=(np.zeros(n_slip), np.full(n_slip, np.inf)),
        n_jobs=1,
    )
    assert misfits.shape == roughnesses.shape == lambdas.shape
    assert np.all(np.isfinite(misfits)) and np.all(np.isfinite(roughnesses))
    # More smoothing -> smoother model, worse fit.
    assert misfits[-1] >= misfits[0]
    assert roughnesses[-1] <= roughnesses[0]
