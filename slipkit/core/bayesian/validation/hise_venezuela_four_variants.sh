#!/usr/bin/env bash
set -euo pipefail

T=/export/dump/ymagen/slipkit-altar-round2-20260918
V=/export/dump/ymagen/venezuela-altar-20260923
LOG=$V/four-variant-queue.log

export PATH="$T/native/bin:$PATH"
export LD_LIBRARY_PATH="$T/native/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$T/snapshot-seeded-v4:$T/native/packages:$T/python-deps"
export XDG_CACHE_HOME="$V/cache"
export MPLCONFIGDIR="$V/cache/matplotlib"
export TMPDIR="$V/tmp"
export PYTHONPYCACHEPREFIX="$V/cache/pycache"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export CUTDE_USE_BACKEND=cpp

cd "$V"
printf '%s preparing exact reference bundles\n' "$(date --iso-8601=seconds)" >> "$LOG"
/usr/bin/python3 venezuela_altar_variants.py --root "$V" >> "$LOG" 2>&1

jobs=(
    depth25-constrained
    depth40-constrained
    depth25-unconstrained
    depth40-unconstrained
)

for job in "${jobs[@]}"; do
    exact="$V/runs/exact-$job/inference"
    output="$V/runs/native-cuda-$job-gpu1-c2048-s1000"
    if [[ -f "$output/report.json" ]]; then
        printf '%s skipping completed %s\n' "$(date --iso-8601=seconds)" "$job" >> "$LOG"
        continue
    fi

    stable=0
    while (( stable < 10 )); do
        IFS=, read -r memory utilization < <(
            nvidia-smi -i 1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
        )
        memory=${memory// /}
        utilization=${utilization// /}
        printf '%s job=%s memory_MiB=%s utilization_pct=%s stable_checks=%s\n' \
            "$(date --iso-8601=seconds)" "$job" "$memory" "$utilization" "$stable" >> "$LOG"
        if (( memory < 1000 && utilization < 10 )); then
            stable=$((stable + 1))
        else
            stable=0
        fi
        if (( stable < 10 )); then sleep 30; fi
    done

    printf '%s launching %s\n' "$(date --iso-8601=seconds)" "$job" >> "$LOG"
    /usr/bin/python3 venezuela_altar_native.py "$exact" \
        --work-dir "$output" \
        --chains 2048 \
        --steps 1000 \
        --scaling 0.03 \
        --gpu-id 1 \
        --timeout 28800 \
        --label "Venezuela native CUDA AlTar: $job" \
        >> "$LOG" 2>&1
    printf '%s completed %s\n' "$(date --iso-8601=seconds)" "$job" >> "$LOG"
done

printf '%s all four variants completed\n' "$(date --iso-8601=seconds)" >> "$LOG"
