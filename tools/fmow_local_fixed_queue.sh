#!/usr/bin/env bash
# Fixed-dose fmow2 local-direction study; one exclusive GPU queue.
# Usage: bash tools/fmow_local_fixed_queue.sh RELEASE_SHA RUN_ROOT GPU_INDEX JOB [JOB...]
# JOB: 6199_step, 6199_ref, or one of 6200_step..6211_step.
set -u

[[ $# -ge 4 ]] || { echo "usage: queue RELEASE_SHA RUN_ROOT GPU_INDEX JOB [JOB...]"; exit 2; }
SHA=$1; ROOT=$2; GPU=$3; shift 3
REL=/home/dsi/michaer8/tralo-rebuild/releases/$SHA
DATA=/home/dsi/michaer8/optloss-audit/data/fmow2/oodslice
PY=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
CONFIGS=$REL/experiments/configs/fmow_local_fixed_20260928

[[ $SHA =~ ^[0-9a-f]{40}$ ]] || { echo "invalid release SHA"; exit 2; }
[[ $ROOT = /* && $GPU =~ ^[0-9]+$ && $# -gt 0 ]] || { echo "invalid queue arguments"; exit 2; }
[[ -d $REL && -d $CONFIGS && -x $PY ]] || { echo "missing release/configs/interpreter"; exit 2; }
mkdir -p "$ROOT" || exit 2
cd "$REL" || exit 2
UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
[[ $UUID = GPU-* ]] || { echo "GPU UUID unavailable"; exit 2; }
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

for JOB in "$@"; do
  case "$JOB" in
    6199_step|6199_ref) SEED=6199 ;;
    *_step)
      SEED=${JOB%_step}
      [[ $SEED =~ ^[0-9]{4}$ && $SEED -ge 6200 && $SEED -le 6211 ]] ||
        { echo "job outside fixed seeds: $JOB"; exit 2; } ;;
    *) echo "invalid job $JOB"; exit 2 ;;
  esac
  CONFIG=$CONFIGS/fmow_local_${JOB}.json
  [[ -f $CONFIG ]] || { echo "missing config $CONFIG"; exit 2; }
  OUT=$ROOT/seed$SEED
  [[ $JOB = 6199_ref ]] && OUT=$ROOT/seed6199_ref
  CLAIM=$ROOT/.claim_$JOB
  if [[ -e $OUT || -e $CLAIM ]]; then
    echo "refusing duplicate output or claim for $JOB"; exit 2
  fi
  mkdir "$CLAIM" || { echo "claim failed for $JOB"; exit 2; }
  if ! PIDS=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); then
    echo "GPU process query failed; queue stopped before $JOB"; exit 2
  fi
  PIDS=$(printf '%s\n' "$PIDS" | sed '/^$/d')
  if [[ -n $PIDS ]]; then
    echo "GPU $GPU $UUID has compute PIDs: $PIDS; queue stopped before $JOB";
    exit 2
  fi
  echo "$(date -u +%FT%TZ) START $JOB GPU_INDEX=$GPU GPU_UUID=$UUID RELEASE=$SHA CONFIG_SHA256=$(sha256sum "$CONFIG" | cut -d' ' -f1)"
  CUDA_VISIBLE_DEVICES="$UUID" "$PY" -u -m tralo.fmow_local "$DATA" "$CONFIG" "$OUT" > "$ROOT/seed$JOB.log" 2>&1 < /dev/null
  RC=$?
  echo "$(date -u +%FT%TZ) END $JOB EXIT=$RC GPU_INDEX=$GPU GPU_UUID=$UUID"
  [[ $RC -eq 0 ]] || exit "$RC"
done
