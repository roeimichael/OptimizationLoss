#!/bin/bash
# claude_probe_queue.sh <gpu> <seed> [<seed> ...] -- the targeted-step dose-response probe, one gpu, seeds in order.
set -u
GPU="$1"; shift
SHA=e7e020858de86ad9defebf90e667f6795f2fcf6e
REL=/home/dsi/michaer8/tralo-rebuild/releases/$SHA
PY=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
DATA=/home/dsi/michaer8/tralo-rebuild/data/knee-chen-v1/KneeXrayData/ClsKLData/kneeKL224
OUT=/home/dsi/michaer8/tralo-rebuild/runs/claude-step-probe-20260926
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$OUT"
cd "$REL" || exit 2
for SEED in "$@"; do
  if [ -e "$OUT/seed$SEED" ]; then echo "SKIP $SEED"; continue; fi
  echo "[$(date '+%F %T')] gpu$GPU START $SEED"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u -m tralo.step_probe "$DATA" "experiments/configs/claude_probe_$SEED.json" "$OUT/seed$SEED" > "$OUT/seed$SEED.log" 2>&1 < /dev/null
  echo "[$(date '+%F %T')] gpu$GPU END $SEED exit $?"
done
echo QUEUE DONE
