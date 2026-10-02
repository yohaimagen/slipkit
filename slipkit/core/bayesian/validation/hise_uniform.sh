set -eu
TASK_DIR=/export/dump/ymagen/slipkit-altar-round2-20260918
cd "$TASK_DIR"
export PATH="$TASK_DIR/native/bin:$PATH" LD_LIBRARY_PATH="$TASK_DIR/native/lib:${LD_LIBRARY_PATH:-}" PYTHONPATH="$TASK_DIR/snapshot:$TASK_DIR/native/packages:$TASK_DIR/python-deps" XDG_CACHE_HOME="$TASK_DIR/cache" PIP_CACHE_DIR="$TASK_DIR/cache/pip" TMPDIR="$TASK_DIR/tmp" PYTHONPYCACHEPREFIX="$TASK_DIR/cache/pycache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUTDE_USE_BACKEND=cpp
mkdir -p runs/tiny-validation
python3 - <<'PY'
import numpy as np,json
from pathlib import Path
from scipy.stats import truncnorm
from slipkit.core.bayesian import AltarCudaBayesianSolver,AltarProblem
p=AltarProblem(np.eye(2),np.array([.2,.7]),np.array([.04,.04]))
report=[]
for sampler in ['fixed','adaptive']:
 s=AltarCudaBayesianSolver(work_dir='runs/uniform-validation',sampler=sampler,prior='uniform',chains=4096,steps=200,min_steps=200,max_steps=800,initial_scaling=.3,timeout=90)
 s.solve_problem(p,bounds=(0.,1.));r=s.get_last_posterior();mean,var=truncnorm.stats(-p.data/.2,(1-p.data)/.2,loc=p.data,scale=.2,moments='mv')
 me=float(np.max(abs(r.samples.mean(0)-mean)/np.sqrt(var)));ve=float(np.max(abs(r.samples.var(0,ddof=1)/var-1)))
 assert np.all((r.samples>=0)&(r.samples<=1))
 report.append(dict(sampler=sampler,run=s.last_run_path,mean_error=me,variance_error=ve,passes=me<.25 and ve<.30))
 Path('runs/uniform-validation/report.json').write_text(json.dumps(report,indent=2));print(report[-1],flush=True)
 assert report[-1]['passes']
PY
