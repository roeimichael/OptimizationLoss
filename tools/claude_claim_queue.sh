#!/bin/bash
# claude_claim_queue.sh <release-sha> <run-dir> <gpu> <seed> [<seed> ...] -- knee_yuval seeds on one gpu.
# Every queue may get the same seed list: a seed is taken only by the queue whose mkdir of
# <run-dir>/.claim_<seed> succeeds (atomic), so queues started later on freed GPUs share the work.
set -u
SHA="$1"; OUT="$2"; GPU="$3"; shift 3
REL=/home/dsi/michaer8/tralo-rebuild/releases/$SHA
PY=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
DATA=/home/dsi/michaer8/tralo-rebuild/data/knee-chen-v1/KneeXrayData/ClsKLData/kneeKL224
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$OUT"
cd "$REL" || exit 2
for SEED in "$@"; do
  mkdir "$OUT/.claim_$SEED" 2>/dev/null || continue
  if [ -e "$OUT/seed$SEED" ]; then echo "SKIP $SEED"; continue; fi
  echo "[$(date '+%F %T')] gpu$GPU START $SEED"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u -m tralo.knee_yuval "$DATA" "experiments/configs/claude_yuval_$SEED.json" "$OUT/seed$SEED" > "$OUT/seed$SEED.log" 2>&1 < /dev/null
  rc=$?; echo "[$(date "+%F %T")] gpu$GPU END $SEED exit $rc"
  # a failure (usually CUDA OOM) would recur on every job this queue claims next: stop, keep the rest unclaimed
  [ "$rc" -eq 0 ] || { echo "QUEUE STOPPED after a failure"; exit "$rc"; }
done
echo QUEUE DONE
