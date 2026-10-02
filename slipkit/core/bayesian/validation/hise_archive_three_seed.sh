set -eu
TASK_DIR=/export/dump/ymagen/slipkit-altar-round2-20260918
cd "$TASK_DIR"
export PATH="$TASK_DIR/native/bin:$PATH" LD_LIBRARY_PATH="$TASK_DIR/native/lib:${LD_LIBRARY_PATH:-}" PYTHONPATH="$TASK_DIR/snapshot-seeded-v4:$TASK_DIR/native/packages:$TASK_DIR/python-deps" XDG_CACHE_HOME="$TASK_DIR/cache" PIP_CACHE_DIR="$TASK_DIR/cache/pip" TMPDIR="$TASK_DIR/tmp" PYTHONPYCACHEPREFIX="$TASK_DIR/cache/pycache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 - <<'PY'
from pathlib import Path
import json,numpy as np
from slipkit.core.bayesian import AltarProblem,load_inference
from slipkit.core.bayesian.importer import AltarResultImporter
from slipkit.core.bayesian.artifact import save_inference
root=Path('runs/144-three-seed-bundles');root.mkdir(exist_ok=True)
c=Path('inputs/patch-144')
p=AltarProblem(np.loadtxt(c/'green.txt',ndmin=2),np.loadtxt(c/'data.txt'),np.loadtxt(c/'cd.txt',ndmin=2))
for label,source in [('fixed','144-fixed-multiseed'),('adaptive','144-adaptive-corrected600')]:
 report=next(Path('runs',source).glob('experiment-*/benchmark.json'))
 for row in json.loads(report.read_text()):
  if not row['passes'] or not all(stage.get('passes') for stage in row['stage_targets']):
   raise ValueError(f'{label} seed {row["seed"]} failed a required gate.')
  run=Path(row['run_path']);manifest=json.loads((run/'manifest.json').read_text())
  posterior=AltarResultImporter().load(run/manifest['output_dir'],manifest=manifest)
  target=root/f'{label}-seed-{row["seed"]}'
  save_inference(target,p,posterior,prior=dict(kind='gaussian',anchor_mean=np.zeros(p.G.shape[1]),scales=np.full(p.G.shape[1],.5)),seed=row['seed'],inference_settings=manifest)
  restored=load_inference(target)
  np.testing.assert_array_equal(restored.posterior.samples,posterior.samples)
  print(label,row['seed'],len(posterior.samples),posterior.final_beta,restored.metadata['arrays_sha256'],flush=True)
PY
tar -czf runs/144-three-seed-bundles.tar.gz -C runs 144-three-seed-bundles
