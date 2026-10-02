set -eu
TASK_DIR=/export/dump/ymagen/slipkit-altar-round2-20260918
cd "$TASK_DIR"
export PATH="$TASK_DIR/native/bin:$PATH" LD_LIBRARY_PATH="$TASK_DIR/native/lib:${LD_LIBRARY_PATH:-}" PYTHONPATH="$TASK_DIR/snapshot-seeded-v2:$TASK_DIR/native/packages:$TASK_DIR/python-deps" XDG_CACHE_HOME="$TASK_DIR/cache" PIP_CACHE_DIR="$TASK_DIR/cache/pip" TMPDIR="$TASK_DIR/tmp" PYTHONPYCACHEPREFIX="$TASK_DIR/cache/pycache" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUTDE_USE_BACKEND=cpp
test ! -e snapshot-seeded-v2
cp -a snapshot snapshot-seeded-v2
python3 -m slipkit.core.bayesian.benchmark inputs/patch-144 --backend cuda --cuda-sampler fixed --chains 1024 --steps 1000 --fixed-scales .14 --seeds 17 31 47 --timeout 300 --work-dir runs/144-fixed-multiseed > runs/144-fixed-multiseed-driver.log 2>&1
python3 -m slipkit.core.bayesian.benchmark inputs/patch-144 --backend cuda --cuda-sampler adaptive --chains 1024 --steps 1000 --min-steps 1000 --max-steps 4000 --seeds 17 31 47 --timeout 300 --work-dir runs/144-adaptive-multiseed > runs/144-adaptive-multiseed-driver.log 2>&1
