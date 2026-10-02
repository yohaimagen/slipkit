set -eu
TASK_DIR=/export/dump/ymagen/slipkit-altar-round2-20260918
cd "$TASK_DIR"
export PATH="$TASK_DIR/native/bin:$PATH" LD_LIBRARY_PATH="$TASK_DIR/native/lib:${LD_LIBRARY_PATH:-}" PYTHONPATH="$TASK_DIR/snapshot-scale-v1:$TASK_DIR/native/packages:$TASK_DIR/python-deps" XDG_CACHE_HOME="$TASK_DIR/cache" PIP_CACHE_DIR="$TASK_DIR/cache/pip" TMPDIR="$TASK_DIR/tmp" PYTHONPYCACHEPREFIX="$TASK_DIR/cache/pycache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUTDE_USE_BACKEND=cpp
test ! -e snapshot-scale-v1
cp -a snapshot snapshot-scale-v1
python3 - <<'PY'
from pathlib import Path
import numpy as np,json
from slipkit.core.bayesian import load_inference
bundle=next(Path('runs/geometry-1000').glob('geometry-*/inference'))
artifact=load_inference(bundle)
p=artifact.problem
assert p.G.shape==(10000,2000)
root=Path('runs/scale-cases');root.mkdir(exist_ok=True)
for triangles,obs in [(400,2000),(1000,10000)]:
 case=root/str(triangles);case.mkdir(exist_ok=True)
 if triangles==400:
  columns=np.r_[0:400,1000:1400]
  g=np.ascontiguousarray(p.G[:obs,columns]);truth=np.r_[np.full(400,.2),np.full(400,.4)]
  data=g@truth+np.random.default_rng(17).normal(0,.01,obs)
 else:g,data=p.G,p.data
 np.save(case/'green.npy',g);np.save(case/'data.npy',data);np.save(case/'cd.npy',np.full(obs,.0001))
 (case/'source.json').write_text(json.dumps(dict(triangles=triangles,parameters=g.shape[1],observations=obs,source_bundle=str(bundle),synthetic=True,noise_variance=.0001),indent=2)+'\n')
 print(case,g.shape,flush=True)
PY
for triangles in 400 1000; do
 python3 -m slipkit.core.bayesian.benchmark "runs/scale-cases/$triangles" --backend cuda --cuda-sampler fixed --chains 4096 --steps 1000 --fixed-scales .14 --seeds 17 --timeout 120 --work-dir "runs/scale-gpu-$triangles" > "runs/scale-gpu-$triangles-driver.log" 2>&1 || true
done
