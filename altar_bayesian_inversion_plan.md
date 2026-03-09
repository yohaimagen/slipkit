# Implementation Plan: ALTar-Based Static Bayesian Slip Inversion for SlipKit

## 1. Motivation: ALTar vs. PyMC

The existing Bayesian module uses PyMC's Sequential Monte Carlo (`pm.sample_smc`). While
functional for small meshes, it has limitations in scalability and performance. ALTar
(Al-Tar) is a dedicated Bayesian inversion package from Caltech/ParaSim that implements
the CATMIP algorithm (the canonical algorithm from Minson, Simons & Beck, 2013) with
GPU-accelerated C++/CUDA kernels and MPI parallelism. Key advantages:

| Concern              | PyMC SMC                        | ALTar CATMIP                            |
|----------------------|---------------------------------|-----------------------------------------|
| Forward model speed  | Pure Python/Theano              | C++/CUDA batched GEMM                   |
| GPU support          | No                              | Yes (CUDA, float32/float64)             |
| Parallelism          | Multi-core (GIL limited)        | MPI across nodes/GPUs                   |
| Checkpointing        | None                            | Per-β-step HDF5                         |
| Seeding (Moment)     | Manual Dirichlet (Python loop)  | Built-in `altar.models.seismic.moment`  |
| COV scheduler        | Fixed step heuristic            | Adaptive COV scheduler                  |
| Cp (epistemic error) | Sampled α parameter             | Static or dynamic Cp matrix             |
| Output format        | ArviZ InferenceData             | HDF5 (`step_nnn.h5`, `step_final.h5`)   |
| Scalability          | < ~200 patches (practical)      | > 1000 patches (tested in literature)   |

---

## 2. ALTar Conceptual Mapping

### 2.1 CATMIP Algorithm (review)

ALTar implements CATMIP, which anneals from the prior to the posterior through tempered
distributions `P_m(θ|d) = P(θ) P(d|θ)^β_m` with β stepping from 0 → 1. At each β-step:
1. Importance weights `w_i = P(d|θ_i)^(Δβ)` are computed.
2. Samples are resampled by weight (COV scheduler targets ESS ≈ N_samples/2).
3. Metropolis burn-in equilibrates samples to the new β distribution.
4. Repeat until β = 1 (the true posterior).

### 2.2 Static Model Parameter Vector

In ALTar's static model, the parameter vector `θ` has `N_param = 2 * N_patches` components
arranged as contiguous parameter sets:

```
θ = [strikeslip_1, ..., strikeslip_M, dipslip_1, ..., dipslip_M]
```

The Green's function matrix `G` has shape `(N_obs, N_param)`, row-major:
```
G[obs_i][param_j]   where j ∈ [0, M-1] → strike-slip columns
                         j ∈ [M, 2M-1] → dip-slip columns
```

This is the same layout that SlipKit's `CutdeCpuEngine.build_kernel()` already produces:
`G_raw[:, :M]` = strike-slip, `G_raw[:, M:]` = dip-slip. **No rotation is needed** when
using ALTar (unlike the existing PyMC `BayesianAssembler` which rotates to rake-parallel/
perpendicular). ALTar expresses the rake constraint through the _prior distribution_ on each
parameter set, not through kernel rotation.

### 2.3 Data Covariance

ALTar's `dataobs` component accepts a full `Cd` matrix `(N_obs, N_obs)`. SlipKit's
`GeodeticDataSet.sigma` is a 1-D diagonal. The mapping is:

```
Cd = diag(sigma²)   [diagonal, block-wise for multiple datasets]
```

For epistemic uncertainty (Cp), ALTar supports two modes:

- **Static Cp**: `Cx = Cd + Cp` passed as a single pre-computed matrix to `cd_file`.
- **Dynamic Cp**: Updated at each β-step via `top`/`bottom` model hooks.

The Minson (2013) approach — `Cp_ii = α² * d_obs_i²` — is a static Cp once an α value
is chosen, or it can be made dynamic by re-computing `Cp` from the current mean model
at each β-step. **Phase 1 will implement static Cp** (pre-chosen α). Dynamic Cp is a
Phase 3 enhancement.

---

## 3. New Module: `slipkit/core/bayesian/altar/`

All ALTar-specific code lives in a new subpackage. The existing PyMC-based module
(`slipkit/core/bayesian/assembler.py`, `solver.py`, `results.py`) is **unchanged** and
remains the default Bayesian backend.

```
slipkit/core/bayesian/
├── __init__.py           (existing)
├── assembler.py          (existing PyMC assembler, unchanged)
├── solver.py             (existing PyMC solver, unchanged)
├── results.py            (existing PyMC results, unchanged)
└── altar/
    ├── __init__.py
    ├── assembler.py      → AltarAssembler
    ├── exporter.py       → AltarDataExporter
    ├── config.py         → AltarConfigBuilder
    ├── solver.py         → AltarBayesianSolver
    ├── importer.py       → AltarResultImporter
    └── results.py        → AltarSlipDistribution
```

---

## 4. Class Definitions

### 4.1 `AltarAssembler(AbstractAssembler)`
**File:** `slipkit/core/bayesian/altar/assembler.py`

Assembles the linear system for ALTar: raw SS/DS Green's functions without rake rotation
and without regularization. Caches `G` (expensive) and reuses it.

**Key difference from `BayesianAssembler`**: No rake rotation. Returns raw `G_ss/G_ds`
stacked as `[G_ss | G_ds]` horizontally. Prior distributions on each column set encode
the "rake constraint" instead.

**Attributes:**
- `_G_raw_cache`: `Optional[np.ndarray]` — cached `(N_obs, 2M)` kernel matrix
- `_d_obs_cache`: `Optional[np.ndarray]` — stacked observations vector `(N_obs,)`
- `_sigma_cache`: `Optional[np.ndarray]` — stacked diagonal uncertainties `(N_obs,)`
- `_areas_cache`: `Optional[np.ndarray]` — stacked patch areas `(M,)` for moment seeding

**Methods:**
- `assemble(faults, datasets, engine, regularization_manager, lambda_spatial)`:
  - Returns `(A, b)` where `A = [G_ss | G_ds]` and `b = concatenate([d_obs, sigma])`.
  - `lambda_spatial` and `regularization_manager` are **intentionally ignored**.
  - Uses caching for `G`.
- `get_areas(faults)`: Returns `(M,)` concatenated patch areas across all faults.
- `clear_cache()`: Clears all cached arrays.

### 4.2 `AltarDataExporter`
**File:** `slipkit/core/bayesian/altar/exporter.py`

Converts SlipKit data structures to ALTar-compatible HDF5 files (or binary/text).
This is a pure utility class — no inheritance required.

**Constructor:**
```python
AltarDataExporter(output_dir: str)
```

**Methods:**
- `export_greens_function(G: np.ndarray, filepath: str)`:
  - Writes `G` as HDF5 dataset with shape `(N_obs, N_param)`.
  - ALTar expects `Nparam` as the leading dimension in the description, but in
    practice the matrix is stored row-major as `(N_obs, N_param)`.
- `export_data(d_obs: np.ndarray, filepath: str)`:
  - Writes data vector `(N_obs,)` to HDF5.
- `export_covariance(sigma: np.ndarray, filepath: str, alpha: float = 0.0)`:
  - Constructs `Cd = diag(sigma²)`.
  - If `alpha > 0`: computes static `Cp = diag((alpha * d_obs)²)` and exports
    `Cx = Cd + Cp` as the full `(N_obs, N_obs)` matrix.
  - If `alpha == 0`: exports `Cd` only.
  - ALTar accepts sparse diagonal matrices stored as full matrices in HDF5.
- `export_areas(areas: np.ndarray, filepath: str)`:
  - Writes patch areas `(M,)` in km² (ALTar Moment distribution expects km²).
- `export_all(G, d_obs, sigma, areas, alpha, case_dir)`:
  - Convenience method: exports all files to a single `case_dir` directory.
  - Returns a dict of output file paths for `AltarConfigBuilder`.

### 4.3 `AltarConfigBuilder`
**File:** `slipkit/core/bayesian/altar/config.py`

Generates ALTar `.pfg` configuration files programmatically from Python parameters.

**Constructor:**
```python
AltarConfigBuilder(
    app_name: str = "slipmodel",
    n_patches: int,
    n_observations: int,
    case_dir: str,
    green_file: str = "green.h5",
    data_file: str = "data.h5",
    cd_file: str = "cd.h5",
)
```

**Configurable parameters (with defaults matching ALTar static model):**

*Parameter Sets:*
- `ss_prior`: `"gaussian"` — prior for strike-slip (Gaussian centered at 0)
- `ss_prior_sigma`: `float = 0.5` — σ for Gaussian strike-slip prior
- `ds_prior`: `"uniform"` — prior for dip-slip
- `ds_prior_support`: `Tuple[float, float] = (-0.5, 20.0)` — m/s bounds for dip-slip
- `use_moment_prep`: `bool = True` — use Moment distribution to seed dip-slip
- `mw_mean`: `float` — mean moment magnitude for Moment seeding
- `mw_sigma`: `float = 0.2` — σ for Mw Gaussian in Moment seeding
- `rigidity_gpa`: `float = 30.0` — shear modulus in GPa for Moment distribution
- `areas_file`: `str = "areas.h5"` — patch areas for Moment distribution

*InSAR Ramp (optional):*
- `n_ramp_params`: `int = 0` — if >0, adds a `ramp` parameter set
- `ramp_prior_support`: `Tuple[float, float] = (-1.0, 1.0)`

*Sampler/Job:*
- `sampler`: `str = "altar.cuda.bayesian.metropolis"` (or CPU: `"altar.bayesian.metropolis"`)
- `chains`: `int = 2**10` — number of Markov chains
- `steps`: `int = 1000` — MC burn-in steps per β-step
- `tasks`: `int = 1` — MPI tasks (threads)
- `hosts`: `int = 1` — MPI hosts
- `gpus`: `int = 0` — 0 for CPU, 1+ for GPU
- `gpu_precision`: `str = "float32"`
- `output_dir`: `str = "results"` — ALTar result directory
- `output_freq`: `int = 1` — save every N β-steps

**Methods:**
- `build() -> str`: Returns the complete `.pfg` file as a string.
- `save(filepath: str)`: Writes `.pfg` to disk.

**Generated `.pfg` structure (example):**
```pfg
slipmodel:
    model = altar.models.seismic.cuda.static
    model:
        case = {case_dir}
        patches = {n_patches}
        green = {green_file}
        dataobs = altar.cuda.data.datal2
        dataobs:
            observations = {n_observations}
            data_file = {data_file}
            cd_file = {cd_file}
        psets_list = [strikeslip, dipslip]
        psets:
            strikeslip = altar.cuda.models.parameterset
            strikeslip:
                count = {n_patches}
                prior = altar.cuda.distributions.gaussian
                prior.mean = 0
                prior.sigma = {ss_prior_sigma}
            dipslip = altar.cuda.models.parameterset
            dipslip:
                count = {n_patches}
                prep = altar.models.seismic.cuda.moment
                prep:
                    Mw_mean = {mw_mean}
                    Mw_sigma = {mw_sigma}
                    Mu = [{rigidity_gpa}]
                    area = {areas}
                prior = altar.cuda.distributions.uniform
                prior.support = {ds_prior_support}
    controller:
        sampler = {sampler}
        archiver:
            output_dir = {output_dir}
            output_freq = {output_freq}
    job:
        tasks = {tasks}
        hosts = {hosts}
        gpus  = {gpus}
        gpuprecision = {gpu_precision}
        chains = {chains}
        steps  = {steps}
```

### 4.4 `AltarBayesianSolver(SolverStrategy)`
**File:** `slipkit/core/bayesian/altar/solver.py`

The main orchestrator: prepares data, generates config, runs ALTar, and imports results.

**Constructor:**
```python
AltarBayesianSolver(
    mw_mean: float,
    mw_sigma: float,
    areas: np.ndarray,                  # patch areas (m²) from TriangularFaultMesh
    work_dir: str = "./altar_run",      # working directory for all I/O
    alpha_cp: float = 0.0,             # static Cp fractional error (0 = disabled)
    ss_prior_sigma: float = 0.5,
    ds_prior_support: Tuple[float, float] = (-0.5, 20.0),
    n_ramp_params: int = 0,
    chains: int = 2**10,
    steps: int = 1000,
    tasks: int = 1,
    hosts: int = 1,
    gpus: int = 0,
    gpu_precision: str = "float32",
    output_freq: int = 1,
    use_gpu_sampler: bool = False,
    keep_work_dir: bool = True,         # retain files after run for inspection
)
```

**Methods:**
- `solve(A: np.ndarray, b: np.ndarray, bounds=None) -> np.ndarray`:
  - Unpacks `A = [G_ss | G_ds]` and `b = [d_obs | sigma]` from `AltarAssembler`.
  - Calls `_export_data()`, `_write_config()`, `_run_altar()`, `_import_results()`.
  - Returns posterior mean slip vector `(2M,)` for compatibility with `SlipDistribution`.
- `_export_data(G, d_obs, sigma)`: Delegates to `AltarDataExporter`.
- `_write_config(n_patches, n_obs)`: Delegates to `AltarConfigBuilder`.
- `_run_altar()`: Executes `slipmodel --config=<pfg>` as a subprocess via `subprocess.run()`.
  - Captures stdout/stderr and raises `RuntimeError` on non-zero exit.
  - Logs β progress by tailing `BetaStatistics.txt` if `verbose=True`.
- `_import_results() -> AltarSlipDistribution`: Delegates to `AltarResultImporter`.
- `get_last_result() -> Optional[AltarSlipDistribution]`: Returns last result object.

**Subprocess execution note**: ALTar is invoked as a shell command (`slipmodel`), not
imported as a Python library. This avoids the complexity of pyre's component system and
keeps SlipKit's dependency on ALTar as an optional external tool rather than a hard import.

### 4.5 `AltarResultImporter`
**File:** `slipkit/core/bayesian/altar/importer.py`

Reads ALTar's HDF5 output and constructs an `AltarSlipDistribution`.

**Methods:**
- `load(results_dir: str, n_patches: int) -> AltarSlipDistribution`:
  - Reads `step_final.h5` (last β = 1 samples).
  - Extracts `ParameterSets/strikeslip` → `(N_chains, M)` and
    `ParameterSets/dipslip` → `(N_chains, M)`.
  - Reads `Annealer/beta` (should be 1.0) and `Bayesian/prior/likelihood/posterior`.
  - Reads `BetaStatistics.txt` for convergence trajectory.
  - Loads intermediate steps if `load_all_steps=True`.
- `load_beta_statistics(results_dir: str) -> pd.DataFrame`:
  - Parses `BetaStatistics.txt` into a DataFrame with columns
    `[iteration, beta, scaling, accepted, invalid, rejected]`.
- `_check_convergence(beta_stats: pd.DataFrame) -> bool`:
  - Warns if final `beta < 1.0 - tolerance`.
  - Warns if final acceptance rate is below 0.1 or above 0.9.

### 4.6 `AltarSlipDistribution(SlipDistribution)`
**File:** `slipkit/core/bayesian/altar/results.py`

Stores ALTar posterior samples and provides analysis/visualization methods.

**Attributes (in addition to base `SlipDistribution`):**
- `ss_samples`: `np.ndarray` — `(N_chains, M)` strike-slip posterior samples
- `ds_samples`: `np.ndarray` — `(N_chains, M)` dip-slip posterior samples
- `beta_statistics`: `pd.DataFrame` — annealing trajectory
- `n_beta_steps`: `int` — total number of β-steps
- `final_beta`: `float` — should be 1.0; warns if not
- `step_files`: `List[str]` — paths to `step_nnn.h5` for intermediate analysis

**Methods:**
- `get_mean_slip() -> np.ndarray`: Returns `(2M,)` posterior mean `[mean_ss, mean_ds]`.
- `get_posterior_std() -> np.ndarray`: Returns `(2M,)` posterior standard deviation.
- `get_credible_intervals(hdi_prob: float = 0.95) -> np.ndarray`:
  - Returns `(2M, 2)` HDI bounds using `arviz.hdi`.
- `get_slip_magnitude_stats() -> Dict[str, np.ndarray]`:
  - Computes `||slip|| = sqrt(ss² + ds²)` per patch across all samples.
  - Returns `{"mean": ..., "std": ..., "hdi_lower": ..., "hdi_upper": ...}`.
- `plot_annealing_convergence()`:
  - Plots `beta` vs. iteration and acceptance rate from `beta_statistics`.
  - Marks the β-step where β first exceeds 0.5 and 0.9.
- `plot_credible_intervals(component: str = "magnitude")`:
  - Maps 95% CI to the 3D fault mesh using `SlipVisualizer`.
- `plot_slip_marginals(patch_indices: List[int] = None)`:
  - Plots 1D posterior histograms for selected patches using matplotlib.
- `is_converged(tolerance: float = 1e-3) -> bool`:
  - Returns `True` if `final_beta >= 1.0 - tolerance`.

---

## 5. Data Flow

```
User code
    │
    ├─ InversionOrchestrator
    │       ├─ faults: List[TriangularFaultMesh]
    │       ├─ datasets: List[GeodeticDataSet]
    │       ├─ engine: CutdeCpuEngine
    │       ├─ assembler: AltarAssembler          ← new
    │       └─ solver: AltarBayesianSolver        ← new
    │
    ▼ orchestrator.run_inversion(lambda_spatial=0)
    │
    ├─ AltarAssembler.assemble()
    │       ├─ engine.build_kernel(fault, dataset) for each pair
    │       ├─ Stacks G horizontally across faults
    │       ├─ Stacks G vertically across datasets
    │       └─ Returns A=[G_ss|G_ds], b=[d_obs|sigma]
    │
    ├─ AltarBayesianSolver.solve(A, b)
    │       │
    │       ├─ AltarDataExporter.export_all()
    │       │       ├─ green.h5     (N_obs × 2M)
    │       │       ├─ data.h5      (N_obs,)
    │       │       ├─ cd.h5        (N_obs × N_obs)  Cx = Cd [+ Cp if alpha>0]
    │       │       └─ areas.h5     (M,) in km²
    │       │
    │       ├─ AltarConfigBuilder.build() → slipmodel.pfg
    │       │
    │       ├─ subprocess.run("slipmodel --config=slipmodel.pfg")
    │       │       └─ ALTar writes results/ (step_nnn.h5, BetaStatistics.txt)
    │       │
    │       └─ AltarResultImporter.load(results_dir, n_patches)
    │               └─ Returns AltarSlipDistribution
    │
    └─ InversionOrchestrator returns AltarSlipDistribution
```

---

## 6. Handling Multiple Datasets

When multiple `GeodeticDataSet` objects are added (e.g., InSAR + GNSS), the stacking is:

- **G**: Vertically stack `G_dataset1` and `G_dataset2` → `(N_obs_total, 2M)`.
- **d**: Concatenate `d_dataset1` and `d_dataset2` → `(N_obs_total,)`.
- **Cd**: Block-diagonal matrix: `diag(sigma1², ..., sigmaN²)`. ALTar treats all
  observations jointly in a single L2 likelihood, so this is correct.
- **InSAR ramp**: If `dataset.get_nuisance_basis()` returns a non-None matrix `(N, k)`,
  append `k` extra columns to `G` representing ramp functions and add a `ramp` parameter
  set with `count=k` to the `.pfg` config.

---

## 7. Rake Constraint via ALTar Priors

The PyMC implementation enforces rake via kernel rotation + a `Normal(mu=0)` prior on
`u_perp`. In ALTar, the equivalent is:

- **Pure dip-slip event** (e.g., thrust): Set `ss_prior` to `Gaussian(mean=0, sigma=small)`.
  This concentrates strike-slip near zero without hard rotation.
- **Pure strike-slip event**: Set `ds_prior` to `Gaussian(mean=0, sigma=small)`.
- **Variable rake**: Use `Uniform` for both with physical bounds.
- **Known rake**: Use `Gaussian(mean=expected_ss, sigma=small)` for strikeslip and
  `Gaussian(mean=expected_ds, sigma=small)` for dipslip.

The `AltarConfigBuilder` exposes `ss_prior_sigma` and `ds_prior_sigma` to control this.

---

## 8. Epistemic Uncertainty (Cp)

### 8.1 Static Cp (Phase 1)
Pre-compute `Cp = diag((α * d_obs)²)` where `α = alpha_cp` is a user-supplied scalar.
This represents the Minson (2013) fractional model error: the uncertainty in the forward
model is proportional to the predicted signal amplitude.

`AltarDataExporter.export_covariance()` computes `Cx = Cd + Cp` and writes it as the
`cd_file`. ALTar then uses `Cx` in the L2 likelihood.

### 8.2 Dynamic Cp (Phase 3 — future)
Implement a custom ALTar model class (inheriting from `altar.models.BayesianL2`) that
overrides the `top()` hook. At the start of each β-step, recompute `Cp` from the current
mean model `G @ theta_mean` and update `dataobs.cd` before the likelihood is evaluated.
This matches the Minson (2013) updated-Cp algorithm.

---

## 9. Implementation Phases

### Phase 1: Data Export + Config Generation (foundation)
**Goal:** Export SlipKit data to ALTar-readable HDF5 and generate valid `.pfg` files.
Verify by running ALTar manually on the exported files.

1. Implement `AltarAssembler` (no rotation, no regularization).
2. Implement `AltarDataExporter` (HDF5 output: `green.h5`, `data.h5`, `cd.h5`, `areas.h5`).
3. Implement `AltarConfigBuilder` (generate `.pfg` for basic `strikeslip + dipslip`).
4. Manual validation: load exported files in Python + a known ALTar example, check shapes.

### Phase 2: Result Import + Distribution
**Goal:** Parse ALTar's HDF5 output into SlipKit's result objects.

1. Implement `AltarResultImporter` (read `step_final.h5`, `BetaStatistics.txt`).
2. Implement `AltarSlipDistribution` with posterior stats and convergence diagnostics.
3. Unit tests with pre-computed synthetic ALTar outputs (mock HDF5 files).

### Phase 3: Full Solver + Orchestration
**Goal:** End-to-end `run_inversion()` calling ALTar as a subprocess.

1. Implement `AltarBayesianSolver.solve()` with subprocess execution.
2. Integration test on a synthetic 9-patch fault (matches ALTar's `9patch` example).
3. Validate posterior mean against the known synthetic truth.
4. Validate convergence: `final_beta == 1.0`, acceptance rates reasonable.

### Phase 4: Cp + InSAR Ramp + Multi-dataset
**Goal:** Full feature parity with Minson (2013).

1. Add static Cp (`alpha_cp > 0`) to `AltarDataExporter`.
2. Add InSAR ramp columns to `AltarAssembler` via `get_nuisance_basis()`.
3. Extend `AltarConfigBuilder` with `ramp` parameter set.
4. Test on synthetic InSAR dataset with known ramp + slip.

### Phase 5: Documentation + Tutorial
**Goal:** Provide a Jupyter notebook mirroring the existing `bayesian_synthetic_tutorial.py`
but using the ALTar backend.

1. Write `examples/altar_synthetic_tutorial.ipynb` showing:
   - Setup: `AltarAssembler` + `AltarBayesianSolver`.
   - Configuration tuning: chains, steps, GPU/CPU.
   - Result analysis: `plot_annealing_convergence()`, `plot_credible_intervals()`.
2. Document ALTar installation prerequisites in `README.md`.

---

## 10. API — User Workflow

```python
from slipkit.core.fault import TriangularFaultMesh
from slipkit.core.physics import CutdeCpuEngine
from slipkit.core.inversion import InversionOrchestrator
from slipkit.core.bayesian.altar import AltarAssembler, AltarBayesianSolver
from slipkit.utils.parsers import GnssParser, InsarParser

# 1. Load geometry and data
mesh = TriangularFaultMesh("fault.msh")
gnss = GnssParser.read("stations.csv")
insar = InsarParser.read("track_103.tif", los_vecs=...)

# 2. Configure the ALTar backend
areas_m2 = mesh.get_areas()  # (M,) in m²

solver = AltarBayesianSolver(
    mw_mean=7.3,
    mw_sigma=0.2,
    areas=areas_m2,
    work_dir="./my_altar_run",
    alpha_cp=0.05,          # 5% fractional model error
    chains=2**12,
    steps=2**10,
    gpus=1,                 # use GPU
    use_gpu_sampler=True,
)

assembler = AltarAssembler()

# 3. Orchestrate
inv = InversionOrchestrator()
inv.add_fault(mesh)
inv.add_data(gnss)
inv.add_data(insar)
inv.set_engine(CutdeCpuEngine(poisson=0.25))
inv.set_assembler(assembler)
inv.set_solver(solver)

# 4. Run (lambda_spatial is ignored by AltarAssembler)
result = inv.run_inversion(lambda_spatial=0.0)

# 5. Analyse
print(result.is_converged())               # True if β reached 1.0
result.plot_annealing_convergence()        # β-step trajectory
result.plot_credible_intervals()           # 3D mesh + 95% CI
mean_slip = result.get_mean_slip()         # (2M,) posterior mean
ci = result.get_credible_intervals(0.95)   # (2M, 2) HDI bounds
```

---

## 11. Testing Strategy

### Unit Tests (`slipkit/tests/core/test_altar_*.py`)

| Test file                        | What it tests                                              |
|----------------------------------|------------------------------------------------------------|
| `test_altar_assembler.py`        | G shape, caching, multi-fault/dataset stacking             |
| `test_altar_exporter.py`         | HDF5 file shapes, Cd construction, Cp with alpha > 0      |
| `test_altar_config.py`           | Valid `.pfg` syntax, correct patch count, ramp params      |
| `test_altar_importer.py`         | Mock HDF5 → AltarSlipDistribution attribute correctness   |
| `test_altar_results.py`          | HDI, std, `is_converged()`, `plot_annealing_convergence()` |

### Integration Test (`slipkit/tests/core/test_altar_integration.py`)

Requires ALTar to be installed. Runs a full inversion on a synthetic 9-patch fault mesh
and validates:
- `result.is_converged() == True`
- `||result.get_mean_slip() - truth|| / ||truth|| < 0.1` (10% relative error)
- `result.n_beta_steps > 5` (sampler actually annealed)
- All plots render without error

---

## 12. Dependencies and Prerequisites

### New Required (for ALTar backend)
- `ALTar 2.0` (installed separately, not a pip package; see ALTar installation guide)
- `pyre` (ALTar's component framework, installed with ALTar)
- `h5py` (for reading/writing HDF5 files)

### Already Available in SlipKit
- `numpy`, `scipy`, `pandas`, `matplotlib`, `arviz`

### Optional
- CUDA-capable GPU + CUDA toolkit (for GPU acceleration)
- OpenMPI (for multi-node runs)

The ALTar backend should gracefully fail at **import time** (not at class instantiation)
if `h5py` is missing, and at **subprocess invocation** if `slipmodel` is not on `PATH`,
with a clear error message pointing to ALTar installation docs.

---

## 13. Key Design Decisions

1. **Subprocess over Python API**: ALTar's pyre component system is complex and requires
   an `Application` instance to instantiate components. Running ALTar as a subprocess
   via `subprocess.run("slipmodel ...")` is simpler, more robust to ALTar version changes,
   and keeps SlipKit's ALTar dependency optional.

2. **No rake rotation in AltarAssembler**: Unlike the PyMC `BayesianAssembler`, rake
   constraint is expressed as prior distributions, matching ALTar's native model structure.
   Users control rake behaviour via `ss_prior_sigma` and `ds_prior_sigma`.

3. **Static Cp first**: Dynamic Cp (updating at each β-step) requires a custom ALTar model
   class and is a Phase 3 enhancement. Static Cp covers the most common use case from
   Minson (2013) and is sufficient for first validation.

4. **Preserve existing PyMC module**: The `BayesianAssembler` + `BayesianSolver` remain
   as the default Bayesian backend for users without ALTar installed and for small meshes
   where PyMC SMC is sufficient.

5. **HDF5 as the primary I/O format**: Preferred over raw binary or text for ALTar because
   it includes shape/precision metadata, avoiding reshaping bugs. ALTar 2.0 detects format
   from the `.h5` suffix automatically.
