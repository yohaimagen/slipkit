set -eu
TASK_DIR=/export/dump/ymagen/slipkit-altar-round2-20260918
cd "$TASK_DIR"
export PATH="$TASK_DIR/native/bin:$PATH" LD_LIBRARY_PATH="$TASK_DIR/native/lib:${LD_LIBRARY_PATH:-}" PYTHONPATH="$TASK_DIR/snapshot:$TASK_DIR/native/packages:$TASK_DIR/python-deps" XDG_CACHE_HOME="$TASK_DIR/cache" PIP_CACHE_DIR="$TASK_DIR/cache/pip" TMPDIR="$TASK_DIR/tmp" PYTHONPYCACHEPREFIX="$TASK_DIR/cache/pycache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUTDE_USE_BACKEND=cpp
mkdir -p runs/tiny-validation
python3 - <<'PY' > runs/tiny-validation/driver.log 2>&1
import numpy as np,json,traceback
from pathlib import Path
from slipkit.core.bayesian import AltarCudaBayesianSolver, AltarProblem
from slipkit.core.bayesian.diagnostics import gaussian_stage_errors
root=Path('runs/tiny-validation'); report=[]
for correlation in [.8,-.8,0.]:
 c=.02*np.array([[1.,correlation],[correlation,2.]])
 p=AltarProblem(np.array([[1.,.2],[.3,1.]]),np.array([.2,.7]),c)
 s=AltarCudaBayesianSolver(work_dir=root,sampler='fixed',chains=512,steps=200,initial_scaling=.3,prior_mean=[.2,-.1],prior_scales=[.5,.8],alpha_cp=.1,timeout=90)
 try:
  s.solve_problem(p); r=s.get_last_posterior()
  residual=p.data-r.samples@p.G.T; ce=c+np.diag((.1*p.data)**2)
  ll=-.5*(np.einsum('ij,ji->i',residual,np.linalg.solve(ce,residual.T))+np.linalg.slogdet(ce)[1]+2*np.log(2*np.pi))
  lp=-.5*np.sum(((r.samples-[.2,-.1])/[.5,.8])**2,axis=1)-np.log([.5,.8]).sum()-np.log(2*np.pi)
  np.testing.assert_allclose(r.probabilities['likelihood'],ll,rtol=1e-10,atol=1e-9)
  np.testing.assert_allclose(r.probabilities['prior'],lp,rtol=1e-10,atol=1e-9)
  metrics=gaussian_stage_errors(p,r,mean=[.2,-.1],scales=[.5,.8],alpha_cp=.1)
  assert metrics['passes'],metrics
  report.append(dict(correlation=correlation,passes=True,run=s.last_run_path,metrics=metrics))
 except Exception as e:
  report.append(dict(correlation=correlation,passes=False,error=str(e),run=s.last_run_path));traceback.print_exc()
 (root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report[-1]),flush=True)
 if not report[-1]['passes']:raise SystemExit(1)
PY
cat runs/tiny-validation/driver.log
