#!/bin/bash
# claude_yuval_queue.sh <release-sha> <run-dir> <gpu> <seed> [<seed> ...] -- knee_yuval seeds in order on one gpu.
set -u
SHA="$1"; OUT="$2"; GPU="$3"; shift 3
REL=/home/dsi/michaer8/tralo-rebuild/releases/$SHA
PY=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
DATA=/home/dsi/michaer8/tralo-rebuild/data/knee-chen-v1/KneeXrayData/ClsKLData/kneeKL224
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$OUT"
cd "$REL" || exit 2
for SEED in "$@"; do
  if [ -e "$OUT/seed$SEED" ]; then echo "SKIP $SEED"; continue; fi
  echo "[$(date '+%F %T')] gpu$GPU START $SEED"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u -m tralo.knee_yuval "$DATA" "experiments/configs/claude_yuval_$SEED.json" "$OUT/seed$SEED" > "$OUT/seed$SEED.log" 2>&1 < /dev/null
  rc=$?; echo "[$(date "+%F %T")] gpu$GPU END $SEED exit $rc"
done
echo QUEUE DONE
