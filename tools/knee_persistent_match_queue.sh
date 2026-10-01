#!/usr/bin/env bash
# One exclusive dsisco02 GPU queue for the fixed knee pilot or full block.
# Usage: bash tools/knee_persistent_match_queue.sh SHA ROOT GPU_INDEX MAX_SECONDS BACKBONE:SEED [...]
set -u

[[ $# -ge 5 ]] || { echo 'usage: queue SHA ROOT GPU_INDEX MAX_SECONDS BACKBONE:SEED [...]'; exit 2; }
SHA=$1; ROOT=$2; GPU=$3; MAX_SECONDS=$4; shift 4
REL=/home/dsi/michaer8/tralo-rebuild/releases/$SHA
DATA=/home/dsi/michaer8/tralo-rebuild/data/knee-chen-v1/KneeXrayData/ClsKLData/kneeKL224
PY=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
BASE=/home/dsi/michaer8/tralo-rebuild/runs/knee-persistent-match-20261001

[[ $(hostname -s) = dsisco02 ]] || { echo 'knee comparison restricted to dsisco02'; exit 2; }
[[ $SHA =~ ^[0-9a-f]{40}$ && $GPU =~ ^[0-3]$ && $MAX_SECONDS =~ ^[0-9]+$ ]] ||
  { echo 'invalid release, GPU index, or time ceiling'; exit 2; }
[[ $MAX_SECONDS -gt 0 && $MAX_SECONDS -le 43200 && $ROOT = "$BASE"/* ]] ||
  { echo 'time ceiling or output root outside the authorized study'; exit 2; }
[[ -d $REL && -d $DATA && -x $PY ]] || { echo 'release, data, or interpreter missing'; exit 2; }
[[ $(git -C "$REL" rev-parse HEAD) = "$SHA" && -z $(git -C "$REL" status --porcelain) ]] ||
  { echo 'release source is not immutable and clean'; exit 2; }
mkdir -p "$ROOT" || exit 2
cd "$REL" || exit 2
UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
[[ $UUID = GPU-* ]] || { echo 'physical GPU UUID unavailable'; exit 2; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
QUEUE_STARTED=$(date +%s)

for JOB in "$@"; do
  [[ $JOB =~ ^(efficientnet_b5|mobilenet_v3_large|vit_b_16):(6700|67(0[1-9]|1[0-2]))$ ]] ||
    { echo "job outside fixed design: $JOB"; exit 2; }
  BACKBONE=${JOB%:*}; SEED=${JOB#*:}
  CONFIG=$REL/experiments/configs/knee_persistent_${BACKBONE}_${SEED}.json
  [[ -f $CONFIG ]] || { echo "missing immutable config $CONFIG"; exit 2; }
  OUT=$ROOT/$BACKBONE/seed$SEED
  CLAIM=$ROOT/.claim_${BACKBONE}_${SEED}
  [[ ! -e $OUT && ! -e $CLAIM ]] || { echo "duplicate run refused: $JOB"; exit 2; }
  if ! PIDS=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); then
    echo "GPU process query failed; stopping before $JOB"; exit 2
  fi
  PIDS=$(printf '%s\n' "$PIDS" | sed '/^$/d')
  if [[ -n $PIDS ]]; then
    echo "GPU $GPU UUID $UUID has compute PIDs $PIDS; stopping before $JOB"; exit 2
  fi
  REMAINING=$((MAX_SECONDS - $(date +%s) + QUEUE_STARTED))
  [[ $REMAINING -gt 0 ]] || { echo "queue time ceiling reached before $JOB"; exit 124; }
  mkdir "$CLAIM" || { echo "exclusive claim failed: $JOB"; exit 2; }
  if ! PIDS=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); then
    echo "GPU recheck failed after claim; stopping before $JOB"; exit 2
  fi
  PIDS=$(printf '%s\n' "$PIDS" | sed '/^$/d')
  [[ -z $PIDS ]] || { echo "GPU $GPU UUID $UUID claimed by PIDs $PIDS after recheck; stopping"; exit 2; }
  mkdir -p "$(dirname "$OUT")" || exit 2
  echo "$(date -u +%FT%TZ) START $JOB GPU_INDEX=$GPU GPU_UUID=$UUID RELEASE=$SHA CONFIG_SHA256=$(sha256sum "$CONFIG" | cut -d' ' -f1) QUEUE_SECONDS_REMAINING=$REMAINING"
  CUDA_VISIBLE_DEVICES="$UUID" timeout --signal=TERM --kill-after=30s "${REMAINING}s" \
    "$PY" -u -m tralo.knee_persistent_match "$DATA" "$CONFIG" "$OUT" \
    > "$ROOT/${BACKBONE}_seed${SEED}.log" 2>&1 < /dev/null
  RC=$?
  echo "$(date -u +%FT%TZ) END $JOB EXIT=$RC GPU_INDEX=$GPU GPU_UUID=$UUID"
  [[ $RC -eq 0 ]] || exit "$RC"
done
