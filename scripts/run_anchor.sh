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
    case "$1" in
        our_approach)               echo "src.experiments.run_experiment" ;;
        fioretto_ldf)               echo "fioretto_research.run_fioretto" ;;
        heuristic|po_lp|danits_lp)  echo "src.experiments.run_heuristic" ;;
        *)                           echo "" ;;
    esac
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
