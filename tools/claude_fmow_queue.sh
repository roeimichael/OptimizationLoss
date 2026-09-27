#!/bin/bash
# claude_fmow_queue.sh <release-sha> <run-dir> <gpu> <job> [<job> ...] -- tralo.fmow_yuval jobs on one gpu.
# tools/claude_claim_queue.sh for fmow2: a job is taken only by the queue whose mkdir of <run-dir>/.claim_<job>
# succeeds (atomic); its config is experiments/configs/claude_fmow_<job>.json and its output <run-dir>/seed<job>.
set -u
SHA="$1"; OUT="$2"; GPU="$3"; shift 3
REL=/home/dsi/michaer8/tralo-rebuild/releases/$SHA
PY=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
DATA=/home/dsi/michaer8/optloss-audit/data/fmow2/oodslice
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$OUT"
cd "$REL" || exit 2
for JOB in "$@"; do
  mkdir "$OUT/.claim_$JOB" 2>/dev/null || continue
  if [ -e "$OUT/seed$JOB" ]; then echo "SKIP $JOB"; continue; fi
  echo "[$(date '+%F %T')] gpu$GPU START $JOB"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u -m tralo.fmow_yuval "$DATA" "experiments/configs/claude_fmow_$JOB.json" "$OUT/seed$JOB" > "$OUT/seed$JOB.log" 2>&1 < /dev/null
  rc=$?; echo "[$(date "+%F %T")] gpu$GPU END $JOB exit $rc"
  [ "$rc" -eq 0 ] || { echo "QUEUE STOPPED after a failure"; exit "$rc"; }
done
echo QUEUE DONE
