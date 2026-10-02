set -eu
TASK_DIR=/export/dump/ymagen/slipkit-altar-round2-20260918
cd "$TASK_DIR"
export PATH="$TASK_DIR/native/bin:$PATH" LD_LIBRARY_PATH="$TASK_DIR/native/lib:${LD_LIBRARY_PATH:-}" PYTHONPATH="$TASK_DIR/snapshot-seeded-v1:$TASK_DIR/native/packages:$TASK_DIR/python-deps" XDG_CACHE_HOME="$TASK_DIR/cache" PIP_CACHE_DIR="$TASK_DIR/cache/pip" TMPDIR="$TASK_DIR/tmp" PYTHONPYCACHEPREFIX="$TASK_DIR/cache/pycache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 - <<'PY'
from pathlib import Path
import json,numpy as np
from slipkit.core.bayesian import AltarProblem,load_inference
from slipkit.core.bayesian.importer import AltarResultImporter
from slipkit.core.bayesian.artifact import save_inference
root=Path('runs/144-fixed/experiment-3u0ejjml')
run=next(root.glob('run-*'))
m=json.loads((run/'manifest.json').read_text())
c=Path('inputs/patch-144')
p=AltarProblem(np.loadtxt(c/'green.txt',ndmin=2),np.loadtxt(c/'data.txt'),np.loadtxt(c/'cd.txt',ndmin=2))
r=AltarResultImporter().load(run/m['output_dir'],manifest=m)
target=root/'inference'
save_inference(target,p,r,prior=dict(kind='gaussian',anchor_mean=np.zeros(p.G.shape[1]),scales=np.full(p.G.shape[1],.5)),seed=17,inference_settings=m)
saved=load_inference(target)
np.testing.assert_array_equal(saved.posterior.samples,r.samples)
np.testing.assert_allclose(saved.problem.G@saved.posterior.samples[:2].T,p.G@r.samples[:2].T)
print(json.dumps(dict(path=str(target),particles=len(r.samples),beta=r.final_beta,sha256=saved.metadata['arrays_sha256'])))
PY
tar -czf runs/144-fixed/144-pilot-inference.tar.gz -C runs/144-fixed/experiment-3u0ejjml inference
