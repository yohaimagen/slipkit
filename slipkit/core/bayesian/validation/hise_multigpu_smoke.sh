set -eu
TASK_DIR=/export/dump/ymagen/slipkit-altar-round2-20260918
cd "$TASK_DIR"
export PATH="$TASK_DIR/native/bin:$PATH" LD_LIBRARY_PATH="$TASK_DIR/native/lib:${LD_LIBRARY_PATH:-}" PYTHONPATH="$TASK_DIR/native/packages:$TASK_DIR/python-deps" XDG_CACHE_HOME="$TASK_DIR/cache" TMPDIR="$TASK_DIR/tmp" PYTHONPYCACHEPREFIX="$TASK_DIR/cache/pycache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p runs/upstream-multigpu-smoke
cp -a src/altar/models/seismic/examples/9patch runs/upstream-multigpu-smoke/
cd runs/upstream-multigpu-smoke
sed -e 's/gpuprecision = float32/gpuprecision = float64/' -e 's/chains = 2\*\*10/chains = 128/' -e 's/steps = 1000/steps = 100/' -e 's/output_freq = 3/output_freq = 1/' -e 's/gpuids = \[0, 0\]/gpuids = [0, 1]/' "$TASK_DIR/src/altar/models/seismic/examples/static_mpi.pfg" > static.pfg
/usr/bin/time -v timeout --signal=INT --kill-after=10 90 slipmodel --config=static.pfg > sampler.log 2>&1 &
RUN_PID=$!
while kill -0 "$RUN_PID" 2>/dev/null; do nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits >> devices-during.csv; sleep 1; done
wait "$RUN_PID"
tail -20 sampler.log
