#!/bin/bash
# claude_bandcons_queue.sh <gpu> <seed> [<seed> ...]
# Runs the preregistered controller/sham seeds back to back on ONE gpu of dsisco01.
# Refuses to start a seed while any other user's process holds that gpu, and never
# re-runs a seed whose output directory already exists (the runner refuses too).
set -u
GPU="$1"; shift
SHA="${SHA:?set SHA}"
REL=/home/dsi/michaer8/tralo-rebuild/releases/$SHA
PY=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
DATA=/home/dsi/michaer8/tralo-rebuild/data/knee-chen-v1/KneeXrayData/ClsKLData/kneeKL224
OUT=/home/dsi/michaer8/tralo-rebuild/runs/claude-bandcons-${BLOCK:?set BLOCK}
# dsisco01 runs another user's ~60 niced CPU jobs; an uncapped torch process spins ~44 threads and
# slowed epochs 10x (24 s -> 4 min). v4's CPU augmentation contains a reduction, so the thread count
# CAN change results bitwise: keep it fixed at 8 for every seed of the study.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
say() { echo "[$(date '+%F %T')] gpu$GPU $*"; }
cd "$REL" || exit 2
for SEED in "$@"; do
  if [ -e "$OUT/seed$SEED" ]; then say "SKIP $SEED -- output exists"; continue; fi
  while :; do
    others=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | while read p; do
      u=$(ps -o user= -p "$p" 2>/dev/null | xargs); [ -n "$u" ] && [ "$u" != michaer8 ] && echo "$u"; done)
    [ -z "$others" ] && break
    say "gpu held by $others -- waiting"; sleep 120
  done
  say "START seed $SEED"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u -m tralo.knee_e2e_v4 "$DATA" \
    "experiments/configs/claude_bandcons_$SEED.json" "$OUT/seed$SEED" > "$OUT/seed$SEED.log" 2>&1 < /dev/null
  say "END seed $SEED exit $?"
done
say "QUEUE DONE"
