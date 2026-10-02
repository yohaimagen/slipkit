set -eu
TASK_DIR=/export/dump/ymagen/slipkit-altar-round2-20260918
cd "$TASK_DIR"
export PATH="$TASK_DIR/native/bin:$PATH" LD_LIBRARY_PATH="$TASK_DIR/native/lib:${LD_LIBRARY_PATH:-}" PYTHONPATH="$TASK_DIR/snapshot:$TASK_DIR/native/packages:$TASK_DIR/python-deps" XDG_CACHE_HOME="$TASK_DIR/cache" PIP_CACHE_DIR="$TASK_DIR/cache/pip" TMPDIR="$TASK_DIR/tmp" PYTHONPYCACHEPREFIX="$TASK_DIR/cache/pycache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUTDE_USE_BACKEND=cpp
mkdir -p runs/tiny-validation
python3 - <<'PY'
import numpy as np,json
from pathlib import Path
from slipkit.core.bayesian import AltarCudaBayesianSolver,AltarProblem
p=AltarProblem(np.eye(2),np.array([.2,.7]),np.array([.1,.1]))
results=[];runs=[]
for seed in [17,17,31]:
 s=AltarCudaBayesianSolver(work_dir='runs/seed-validation',sampler='fixed',seed=seed,chains=512,steps=200,initial_scaling=.3,timeout=90)
 s.solve_problem(p);results.append(s.get_last_posterior().samples);runs.append(s.last_run_path)
report=dict(same_seed_identical=bool(np.array_equal(results[0],results[1])),different_seed_distinct=bool(not np.array_equal(results[0],results[2])),runs=runs)
Path('runs/seed-validation/report.json').write_text(json.dumps(report,indent=2));print(report)
import h5py
initial=[]
for run in runs:
 with h5py.File(Path(run)/'results/step_000.h5') as h:initial.append(h['ParameterSets/theta'][:])
report.update(same_seed_initial_identical=bool(np.array_equal(initial[0],initial[1])),different_seed_initial_distinct=bool(not np.array_equal(initial[0],initial[2])),limitation='Native atomicAdd queue changes acceptance-draw assignment; final samples are not bitwise reproducible.')
Path('runs/seed-validation/report.json').write_text(json.dumps(report,indent=2));print(report)
assert report['same_seed_initial_identical'] and report['different_seed_initial_distinct']
PY
