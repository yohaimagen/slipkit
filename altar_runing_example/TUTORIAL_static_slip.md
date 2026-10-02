# AlTar 2.0 — Static Slip Inversion Tutorial (Mac / CPU)

> **Quick context on CUDA vs CPU:** The AlTar documentation and all `.pfg` example
> files reference `altar.models.seismic.cuda.static`, which requires an **NVIDIA GPU**.
> On Mac (Apple Silicon), there is no NVIDIA GPU, so that path will not run.
> This tutorial uses `altar.models.linear`, which implements the **identical forward
> model** (`d = G θ`) and the same CATMIP Bayesian sampler, but runs entirely on CPU.
> The configuration syntax and output format are the same.
> At the end of this tutorial, a reference config for CUDA machines is also provided.

---

## The Problem: 9-Patch Static Slip Inversion

We have a fault plane divided into **9 rectangular patches**. We want to recover
the spatial distribution of coseismic slip from surface displacement observations.

**Forward model:**
```
d = G θ
```
| Symbol | Meaning | Shape |
|--------|---------|-------|
| `θ` | slip parameters (strike + dip for each patch) | 18 (= 2 × 9) |
| `G` | Green's functions (surface displacement per unit slip) | 108 × 18 |
| `d` | observed surface displacements | 108 |

The Bayesian inversion recovers a posterior **distribution** over θ given d, not
just a single best-fit model. This quantifies both the recovered slip and its
uncertainty.

---

## Directory Structure

Use the 9-patch example data that ships with AlTar. Set up a working directory:

```bash
# activate your conda environment
conda activate slipkit   # or whichever env has altar installed

# create a clean working directory
mkdir altar_runing_example
cd altar_runing_example

# copy the example input data and config
cp -r ~/tools/src/altar/models/linear/examples/patch-9 .
cp ~/tools/src/altar/models/linear/examples/linear.pfg .
```

> **Note:** The `python3 -c "import altar; ..."` trick to locate example files does **not**
> work for a source install of ALTar (the path resolution goes outside the install prefix).
> Use the direct `~/tools/src/altar/models/linear/examples/` path shown above.

Your directory should look like:
```
altar_runing_example/
├── patch-9/
│   ├── green.txt   — Green's function: 108×18 matrix (Nobs × Nparams)
│   ├── data.txt    — observed displacements: 108 values
│   └── cd.txt      — data covariance matrix: 108×108
└── linear.pfg      — configuration file (copied from examples)
```

### Data file format

All input files are plain text, one value per line (or whitespace-separated):

- `green.txt` — 108 × 18 = 1944 values, row-major (all params for obs 1, then obs 2, …)
- `data.txt`  — 108 displacement values (units: meters)
- `cd.txt`    — 108 × 108 = 11664 values, the data covariance matrix

---

## Configuration File

The `linear.pfg` file was copied directly from the AlTar examples directory. Open it
to review or adjust settings:

```bash
code linear.pfg   # or any editor
```

Key settings to know:

```ini
; Static slip inversion — 9-patch fault, CPU (Mac compatible)
;
; Run with:
;   linear --config=linear.pfg

linear:

    ; ── forward model ──────────────────────────────────────────────────────
    model:
        ; input data directory (contains green.txt, data.txt, cd.txt)
        case = patch-9

        ; total number of slip parameters: 2 (strike + dip) × 9 patches
        parameters = 18

        ; number of surface displacement observations
        observations = 108

        ; initializer: draw initial samples from a Gaussian
        ; sigma ~ expected slip magnitude in meters
        prep:
            parameters = {linear.model.parameters}
            sigma = 0.5

        ; prior distribution: Gaussian centered at 0
        ; broad sigma leaves the data to constrain the posterior
        prior:
            parameters = {linear.model.parameters}
            sigma = 0.5

    ; ── run configuration ──────────────────────────────────────────────────
    job.tasks = 1   ; single process (no MPI)
    job.gpus  = 0   ; no GPU — runs on CPU
    job.chains = 2**8   ; number of Markov chains (256 in the default example)
    job.steps  = 1000   ; MCMC burn-in steps per beta iteration

    ; ── optional: profiler monitor (prints timing info) ────────────────────
    ; monitors:
    ;     prof = altar.bayesian.profiler

    ; ── MPI parallel run (optional, see MPI section below) ─────────────────
    ; job.tasks = 4
    ; shell = mpi.shells.mpirun
```

---

## Running the Inversion

```bash
linear --config=linear.pfg
```

You will see output like:
```
altar: time: 2026-03-20T13:49:47.375537
altar: iteration: 0, beta: 0, scaling: 0.1
altar: stats(accepted/invalid/rejected): (0, 0, 0)
altar: step
  β: 0
  θ: (256 samples) x (18 parameters)
  ...
altar: resampling: ...
altar: iteration: 1, beta: 0.000152, scaling: 0.85
...
altar: iteration: 23, beta: 1.0  ← converged when β = 1
```

> The default `linear.pfg` example uses `job.chains = 2**8` (256 chains). Increase to
> `2**10` (1024) or higher for better posterior statistics in production runs.

The sampler runs a **cascading annealing** schedule: `β` increases from 0 to 1,
gradually introducing the data likelihood. When `β = 1` the samples are drawn
from the true posterior. A typical run takes **1–5 minutes** on a Mac M-series CPU.

Results are saved in:
```
altar_runing_example/results/
├── BetaStatistics.txt   — convergence log (one row per beta step)
├── step_000.h5          — samples at β = 0 (prior)
├── step_001.h5          — samples at β step 1
├── ...
└── step_final.h5        — samples at β = 1 (posterior)
```

---

## Reading and Analyzing Results

Create a notebook for interactive analysis:

```bash
touch plot.ipynb
code plot.ipynb   # or jupyter lab plot.ipynb
```

Paste the following into cells:

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ── Load posterior samples ──────────────────────────────────────────────────
with h5py.File("results/step_final.h5", "r") as f:
    theta  = f["ParameterSets/theta"][()]   # shape: (N_chains, 18)
    llk    = f["Bayesian/likelihood"][()]   # log-likelihood for each sample
    prior  = f["Bayesian/prior"][()]
    post   = f["Bayesian/posterior"][()]
    beta   = f["Annealer/beta"][()]         # should be 1.0

n_samples, n_params = theta.shape
n_patches = n_params // 2

print(f"Beta at final step: {beta}")
print(f"Samples: {n_samples},  Parameters: {n_params}")

# ── Split into strike and dip components ───────────────────────────────────
strike = theta[:, :n_patches]    # (N_samples, 9) — meters
dip    = theta[:, n_patches:]    # (N_samples, 9) — meters

# ── Mean model and uncertainty ─────────────────────────────────────────────
strike_mean = strike.mean(axis=0)
strike_std  = strike.std(axis=0)
dip_mean    = dip.mean(axis=0)
dip_std     = dip.std(axis=0)

print("\nPatch | Strike mean ± std (m) | Dip mean ± std (m)")
print("-" * 55)
for i in range(n_patches):
    print(f"  {i+1:2d}  |  {strike_mean[i]:+.3f} ± {strike_std[i]:.3f}     "
          f"|  {dip_mean[i]:.3f} ± {dip_std[i]:.3f}")

# ── Plot: slip distribution across patches ─────────────────────────────────
patches = np.arange(1, n_patches + 1)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

axes[0].errorbar(patches, strike_mean, yerr=strike_std, fmt="o-", capsize=4)
axes[0].axhline(0, color="gray", lw=0.8, ls="--")
axes[0].set_title("Strike Slip")
axes[0].set_xlabel("Patch")
axes[0].set_ylabel("Slip (m)")

axes[1].errorbar(patches, dip_mean, yerr=dip_std, fmt="s-", capsize=4, color="C1")
axes[1].axhline(0, color="gray", lw=0.8, ls="--")
axes[1].set_title("Dip Slip")
axes[1].set_xlabel("Patch")

plt.suptitle("Static Slip Inversion — Posterior Mean ± 1σ")
plt.tight_layout()
plt.savefig("results/slip_posterior.pdf")
print("\nSaved: results/slip_posterior.pdf")

# ── Plot: annealing convergence ────────────────────────────────────────────
import csv
betas, scalings = [], []
with open("results/BetaStatistics.txt") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        betas.append(float(row[1]))
        scalings.append(float(row[2]))

fig, ax = plt.subplots(figsize=(7, 3))
ax.semilogy(betas, "o-")
ax.set_xlabel("Beta step")
ax.set_ylabel("β (log scale)")
ax.set_title("Annealing Schedule (β → 1 = converged)")
plt.tight_layout()
plt.savefig("results/annealing.pdf")
print("Saved: results/annealing.pdf")
```

### Expected results for the 9-patch synthetic test case

The synthetic dataset was generated with known true slips:
- **Strike slip:** ~0.17 m (uniform across patches)
- **Dip slip:** ~1.0 m (uniform across patches)

Your recovered posterior should have means close to these values. The standard
deviation of the posterior gives the **uncertainty** — how well the surface
observations constrain each patch's slip.

---

## Understanding the Output Files

Each `step_NNN.h5` file contains a snapshot of the sampler at a given β:

| HDF5 path | Shape | Description |
|-----------|-------|-------------|
| `ParameterSets/theta` | (N_chains, 18) | Slip parameter samples |
| `Bayesian/likelihood` | (N_chains,) | Log data likelihood |
| `Bayesian/prior` | (N_chains,) | Log prior probability |
| `Bayesian/posterior` | (N_chains,) | Log posterior (llk + prior) |
| `Annealer/beta` | scalar | Current β value |
| `Annealer/covariance` | (18, 18) | Metropolis proposal covariance |

`step_000.h5` has β ≈ 0 → samples from prior only.
`step_final.h5` has β = 1 → samples from posterior.

---

## Tuning the Run

| Parameter | Location in .pfg | Effect |
|-----------|-----------------|--------|
| `job.chains` | `job.chains = 2**8` | More chains → better statistics, slower (default example uses 256) |
| `job.steps` | `job.steps = 1000` | More steps → better mixing per β, slower |
| `prior.sigma` | `prior: sigma = 0.5` | Wider prior = less constraint, fine for exploring |
| `prep.sigma` | `prep: sigma = 0.5` | Controls the spread of the initial samples |

The default `linear.pfg` ships with `job.tasks = 4` and `job.chains = 2**10` (1024 total,
256 per task). **Set `job.tasks = 1` for single-process runs** to avoid the broken
threaded path (see Parallelism section below).
For production runs, use `job.chains = 2**12` or larger.
For quick testing, use `job.chains = 2**7` and `job.steps = 200`.

---

## Parallelism

### What works and what doesn't

AlTar has three annealing paths:

| Mode | Triggered by | Status |
|------|-------------|--------|
| Sequential (single process) | `job.tasks = 1`, no shell | **Works** |
| Threaded (`job.tasks > 1`, no MPI shell) | `job.tasks > 1` without `shell = mpi.shells.mpirun` | **Broken** — `ThreadedAnnealing` is an unimplemented stub |
| MPI | `job.tasks > 1` **and** `shell = mpi.shells.mpirun` | **Works** |

> **Important:** Setting `job.tasks = 4` alone is not enough for MPI — and it will crash
> because it routes to the unimplemented `ThreadedAnnealing`. You must also set
> `shell = mpi.shells.mpirun` so that pyre knows to use the MPI annealing path.
> Simply running under `mpirun -n 4` without setting the shell in the config has the
> same problem.

### Single-process run (recommended for testing)

Keep `linear.pfg` with:
```ini
job.tasks = 1
job.gpus  = 0
; shell line absent or commented out
```

```bash
linear --config=linear.pfg
```

### MPI parallel run (multiple CPU cores)

#### Step 1 — tell pyre where to find mpirun

By default pyre discovers MPI through MacPorts, which fails on conda installs.
Create `~/.pyre/mpi.pfg` **once** (not per-run):

```ini
; ~/.pyre/mpi.pfg
mpi.shells.mpirun:
    mpi = openmpi#mpi_conda
    extra = -mca btl self,tcp -mca oob_tcp_if_include lo0 -mca btl_tcp_if_include lo0
    ; -mca btl self,tcp         — disables VADER shared memory (avoids macOS cleanup warnings)
    ; -mca oob_tcp_if_include   — restricts out-of-band TCP to loopback (avoids IPv6 multi-address confusion)
    ; -mca btl_tcp_if_include   — restricts data-plane TCP to loopback
    ; all three are needed for stable single-node multi-process runs on macOS

pyre.externals.mpi.openmpi#mpi_conda:
    version = 4.1.1
    launcher = /Users/ymagen/miniconda3/envs/slipkit/bin/mpirun
    prefix = /Users/ymagen/miniconda3/envs/slipkit
    bindir = {mpi_conda.prefix}/bin
    incdir = {mpi_conda.prefix}/include
    libdir = {mpi_conda.prefix}/lib
```

#### Step 2 — enable MPI shell in `linear.pfg`

```ini
job.tasks = 4              ; number of MPI processes
job.gpus  = 0
job.chains = 2**10         ; chains split evenly across tasks (256 per task)
shell = mpi.shells.mpirun  ; tells pyre to use MPIAnnealing, not ThreadedAnnealing
```

#### Step 3 — run (no `mpirun` prefix)

```bash
linear --config=linear.pfg
```

> **Do NOT prefix with `mpirun -n 4`.**
> When `shell = mpi.shells.mpirun` is set, pyre builds and launches the mpirun command
> itself (passing `--shell.auto=no` to the children so they don't double-spawn).
> Running `mpirun -n 4 linear ...` puts you inside mpirun already, then pyre tries to
> spawn *another* mpirun on top → crash.

The `shell = mpi.shells.mpirun` line is what switches pyre into MPI mode internally.
Without it, `job.tasks > 1` triggers the broken `ThreadedAnnealing` stub and crashes.

---

## Using Your Own Data

Replace the `patch-9` directory with your own case:

```
my_case/
├── green.txt   — Nobs × Nparam matrix, row-major, plain text
├── data.txt    — Nobs values (observed displacements)
└── cd.txt      — Nobs × Nobs covariance matrix
```

Then update `slipmodel.pfg`:
```ini
linear:
    model:
        case = my_case
        parameters = <2 × number_of_patches>
        observations = <number_of_observations>
        prep:
            parameters = {linear.model.parameters}
            sigma = <expected slip in meters>
        prior:
            parameters = {linear.model.parameters}
            sigma = <expected slip in meters>
```

**Tip:** Choose `prior.sigma` and `prep.sigma` based on your prior expectation of
slip magnitude. For a Mw 7 earthquake, dip slips of 1–5 m are typical; set
`sigma = 3.0` for a broad prior.

---

## Reference: CUDA Config (for Linux/Cluster with NVIDIA GPU)

When running on a CUDA machine, use this configuration instead.
Note the different component names (`altar.cuda.*`), different file format (`.h5`),
and different attribute names for the model:

```ini
; static slip inversion — CUDA version (NVIDIA GPU required)
slipmodel = altar.shells.altar
slipmodel:

    model = altar.models.seismic.cuda.static
    model:
        case = 9patch
        patches = 9

        green = static.gf.h5   ; HDF5 file with dataset "static.gf", shape (108, 18)

        dataobs = altar.cuda.data.datal2
        dataobs:
            observations = 108
            data_file = static.data.h5
            cd_file = static.Cd.h5

        psets_list = [strikeslip, dipslip]
        psets:
            strikeslip = altar.cuda.models.parameterset
            dipslip = altar.cuda.models.parameterset

            strikeslip:
                count = {slipmodel.model.patches}
                prior = altar.cuda.distributions.gaussian
                prior.mean = 0
                prior.sigma = 0.5

            dipslip:
                count = {slipmodel.model.patches}
                prior = altar.models.seismic.cuda.moment
                prior:
                    support = (-0.5, 20)
                    Mw_mean = 7.3
                    Mw_sigma = 0.2
                    Mu = [30]    ; shear modulus in GPa
                    area = [400] ; patch area in km²
                    moment_constraint = True

    controller:
        sampler = altar.cuda.bayesian.metropolis
        archiver:
            output_dir = results/static
            output_freq = 3

    job:
        tasks = 1
        gpus = 1
        gpuprecision = float32
        chains = 2**10
        steps = 1000
```

Run with:
```bash
slipmodel --config=static.pfg
# or
slipmodel.plexus sample --config=static.pfg
```

**Key differences between CPU and CUDA configs:**

| | CPU (`linear`) | CUDA (`seismic.cuda.static`) |
|--|---------------|------------------------------|
| App command | `linear` | `slipmodel` |
| Model family | `altar.models.linear` | `altar.models.seismic.cuda.static` |
| Input files | `.txt` plain text | `.h5` HDF5 |
| Data config | `model.green_file`, `model.data_file` | `model.dataobs` component |
| Parameter sets | `model.prep`, `model.prior` | `model.psets_list` + per-set configs |
| Prior (dip) | Gaussian (simple) | Moment magnitude distribution |
| GPU required | No | Yes (NVIDIA only) |
