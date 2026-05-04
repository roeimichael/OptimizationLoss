#!/bin/bash
# Non-interactive runner for a multi-methodology anchor directory.
# Iterates the methodology configs under <anchor_dir>, dispatches each to its
# runner module. Bypasses main.py's interactive select_gpu prompt -- caller
# MUST set CUDA_VISIBLE_DEVICES before invocation.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_anchor.sh \
#       results/pending_runs/multi/tissuemnist/single_GE/L50_G50/MobileNetV3/seed_1/

ANCHOR_DIR="$1"
if [ -z "$ANCHOR_DIR" ]; then
    echo "Usage: CUDA_VISIBLE_DEVICES=N $0 <anchor_dir>"
    exit 2
fi
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    echo "ERROR: set CUDA_VISIBLE_DEVICES first (e.g. CUDA_VISIBLE_DEVICES=0)"
    exit 2
fi
if [ ! -d "$ANCHOR_DIR" ]; then
    echo "ERROR: $ANCHOR_DIR does not exist"
    exit 2
fi

SKIP_GPU_CHECK=0
if [ "$2" = "--skip-gpu-check" ]; then SKIP_GPU_CHECK=1; fi

# === GPU exclusivity check ===
# dsisco02 currently crashes the whole host if two processes share one GPU.
# Returns 0 if all target GPUs are clear, 3 if any has another compute app.
# Called before the run AND between each methodology dispatch (foreign users
# can land on a GPU mid-anchor; we must re-validate continuously).
gpu_clear() {
    local context="$1"
    [ "$SKIP_GPU_CHECK" = "1" ] && return 0
    for gpu in $(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' '); do
        # nvidia-smi -i N filters output to GPU N. Empty stdout = clear.
        local busy
        busy=$(nvidia-smi -i "$gpu" --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
        if [ -n "$busy" ]; then
            echo "ERROR ($context): GPU $gpu is NOT clear -- compute apps present:"
            echo "$busy" | sed 's/^/    /'
            echo "  dsisco02 driver bug: two processes on one GPU crashes the host."
            return 3
        fi
        echo "  $context: GPU $gpu is clear"
    done
    return 0
}

if ! gpu_clear "preflight"; then
    echo "  Pick a different GPU or wait. Override with: $0 $1 --skip-gpu-check"
    exit 3
fi

# Prefer optloss conda env if available
PY=python
for cand in "$HOME/anaconda3/envs/optloss/bin/python" \
            "$HOME/miniconda3/envs/optloss/bin/python"; do
    if [ -x "$cand" ]; then PY="$cand"; break; fi
done

echo "=== run_anchor.sh ==="
echo "  anchor: $ANCHOR_DIR"
echo "  GPU(s): $CUDA_VISIBLE_DEVICES"
echo "  python: $PY ($($PY -V 2>&1))"
echo ""

runner_for() {
    # All methodologies dispatch through src.experiments.runner; the runner
    # itself looks at config[methodology] to pick the train fn.
    echo "src.experiments.runner"
}

ok=0
fail=0
skip=0
configs=$(find "$ANCHOR_DIR" -name config.json -type f | sort)
total=$(echo "$configs" | grep -c .)
i=0

for cp in $configs; do
    i=$((i + 1))
    methodology=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['methodology'])" "$cp")
    module=$(runner_for "$methodology")
    if [ -z "$module" ]; then
        echo "[$i/$total] $methodology: UNKNOWN -- skipping"
        skip=$((skip + 1))
        continue
    fi
    status=$($PY -c "import json,sys; print(json.load(open(sys.argv[1])).get('status','pending'))" "$cp")
    if [ "$status" = "completed" ]; then
        echo "[$i/$total] $methodology: already completed -- skipping"
        skip=$((skip + 1))
        continue
    fi
    # Re-validate GPU is still exclusively ours before each dispatch.
    if ! gpu_clear "[$i/$total] pre-$methodology"; then
        echo "[$i/$total] $methodology: ABORTING REMAINING ($((total - i + 1)) configs) -- GPU intruded by another user"
        fail=$((fail + 1))
        break
    fi
    echo ""
    echo "================================================================"
    echo "[$i/$total] $methodology -> $module"
    echo "  config: $cp"
    echo "================================================================"
    t0=$(date +%s)
    if $PY -u -m "$module" "$cp"; then
        elapsed=$(( $(date +%s) - t0 ))
        echo "[$i/$total] $methodology: OK (${elapsed}s)"
        ok=$((ok + 1))
    else
        elapsed=$(( $(date +%s) - t0 ))
        echo "[$i/$total] $methodology: FAILED (exit $?) after ${elapsed}s"
        fail=$((fail + 1))
    fi
done

echo ""
echo "=== Summary ==="
echo "  ok=$ok  fail=$fail  skip=$skip  total=$total"
exit $fail
