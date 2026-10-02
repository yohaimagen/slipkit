set -eu
TASK_DIR=/export/dump/ymagen/slipkit-altar-round2-20260918
cd "$TASK_DIR"
export PATH="$TASK_DIR/native/bin:$PATH" LD_LIBRARY_PATH="$TASK_DIR/native/lib:${LD_LIBRARY_PATH:-}" PYTHONPATH="$TASK_DIR/snapshot-scale-v3:$TASK_DIR/native/packages:$TASK_DIR/python-deps" XDG_CACHE_HOME="$TASK_DIR/cache" PIP_CACHE_DIR="$TASK_DIR/cache/pip" TMPDIR="$TASK_DIR/tmp" PYTHONPYCACHEPREFIX="$TASK_DIR/cache/pycache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUTDE_USE_BACKEND=cpp
test ! -e snapshot-scale-v3
cp -a snapshot snapshot-scale-v3
python3 -m slipkit.core.bayesian.benchmark runs/scale-cases/1000 --backend cuda --cuda-sampler fixed --gpu-id 1 --chains 4096 --steps 100 --fixed-scales .14 --seeds 17 --timeout 180 --work-dir runs/scale-gpu1-1000-hostmem > runs/scale-gpu1-1000-hostmem-driver.log 2>&1 || true
