#!/usr/bin/env bash
# One exclusive card for the registered fmow2 snapshot-PHR direction study.
# This launcher prepares jobs only; the protocol's separate GPU approval and
# source/data/gradient gates must be satisfied before anyone invokes it.
# Usage: bash tools/fmow_local_alm_queue.sh RELEASE_SHA NEW_RUN_ROOT GPU_INDEX pilot-step|pilot-ref|full
set -u

usage() { echo "usage: $0 RELEASE_SHA NEW_RUN_ROOT GPU_INDEX pilot-step|pilot-ref|full" >&2; exit 2; }
fail() { echo "$*" >&2; exit 2; }
[[ $# -eq 4 ]] || usage
SHA=$1; ROOT=$2; GPU=$3; MODE=$4
[[ $SHA =~ ^[0-9a-f]{40}$ ]] || fail "invalid release SHA"
[[ $ROOT = /* && $ROOT != / && $GPU =~ ^[0-9]+$ ]] || fail "invalid run root or GPU index"
case $MODE in
  pilot-step) JOBS=(6300_step) ;;
  pilot-ref) JOBS=(6300_ref) ;;
  full) JOBS=(); for SEED in {6301..6312}; do JOBS+=("${SEED}_step"); done ;;
  *) usage ;;
esac

REL=/home/dsi/michaer8/tralo-rebuild/releases/$SHA
RUNS=/home/dsi/michaer8/tralo-rebuild/runs
DATA=/home/dsi/michaer8/optloss-audit/data/fmow2/oodslice
PY=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
CONFIGS=$REL/experiments/configs/fmow_local_alm_20260930
[[ -d $REL && -d $DATA && -d $CONFIGS && -x $PY ]] || fail "missing release, data, configs or interpreter"
RUNS_CANON=$(realpath -e -- "$RUNS") || fail "runs directory unavailable"
ROOT_CANON=$(realpath -m -- "$ROOT") || fail "cannot resolve run root"
[[ $ROOT_CANON = "$RUNS_CANON/"* ]] || fail "run root must be below $RUNS_CANON"
ROOT=$ROOT_CANON
[[ -d $(dirname "$ROOT") && ! -e $ROOT && ! -L $ROOT ]] ||
  fail "run root must be new and its parent must exist: $ROOT"

check_release() {
  [[ $(git -C "$REL" rev-parse HEAD 2>/dev/null) = "$SHA" ]] || fail "release HEAD differs from $SHA"
  if ! DIRTY=$(git -C "$REL" status --porcelain --untracked-files=all 2>/dev/null); then
    fail "cannot inspect release cleanliness"
  fi
  [[ -z $DIRTY ]] || fail "release checkout is dirty"
}
check_release
cd "$REL" || fail "cannot enter release"
export PYTHONDONTWRITEBYTECODE=1

# Check the entire fixed list before claiming a root. In particular, a full
# queue cannot silently omit a seed or substitute a pilot/reference config.
CONFIG_ARGS=()
for JOB in "${JOBS[@]}"; do
  CONFIG=$CONFIGS/fmow_local_${JOB}.json
  [[ -f $CONFIG && ! -L $CONFIG ]] || fail "missing or linked config: $CONFIG"
  git -C "$REL" ls-files --error-unmatch "experiments/configs/fmow_local_alm_20260930/fmow_local_${JOB}.json" >/dev/null 2>&1 ||
    fail "config is not tracked: $CONFIG"
  CONFIG_ARGS+=("$CONFIG")
done
"$PY" - "${CONFIG_ARGS[@]}" <<'PY' || fail "config validation failed"
import json
import sys
from pathlib import Path
from tralo.fmow_local import validate

for path in sys.argv[1:]:
    job = Path(path).stem.removeprefix("fmow_local_")
    seed, arm = job.split("_")
    config = json.loads(Path(path).read_text())
    validate(config)
    if (config["study"] != "local_alm_direction_v1"
            or config["seed"] != int(seed)
            or config["snapshot_steps"] is not (arm == "step")):
        raise ValueError(f"config does not match job {job}")
PY

UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
[[ $UUID = GPU-* ]] || fail "GPU UUID unavailable"
HOST=$(hostname -f 2>/dev/null) || fail "host identity unavailable"
[[ -n $HOST ]] || fail "host identity unavailable"
CLAIMS=$RUNS_CANON/.fmow-local-alm-claims
if ! mkdir "$CLAIMS" 2>/dev/null; then
  [[ -d $CLAIMS && ! -L $CLAIMS ]] || fail "study claim directory unavailable"
fi
ROOT_CREATED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

for JOB in "${JOBS[@]}"; do
  check_release
  CURRENT_UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  [[ $CURRENT_UUID = "$UUID" ]] || fail "GPU index ownership changed before $JOB"
  # Query by UUID so an index remap cannot move this job to a different card.
  if ! PIDS=$(nvidia-smi -i "$UUID" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); then
    fail "GPU process query failed before $JOB"
  fi
  PIDS=$(printf '%s\n' "$PIDS" | sed '/^[[:space:]]*$/d')
  [[ -z $PIDS ]] || fail "GPU $UUID has compute PIDs before $JOB: $PIDS"

  SEED=${JOB%%_*}
  OUT=$ROOT/seed$SEED
  [[ $JOB = 6300_ref ]] && OUT=$ROOT/seed6300_ref
  CONFIG=$CONFIGS/fmow_local_${JOB}.json
  LOG=$ROOT/seed${JOB}.log
  LAUNCH=$ROOT/seed${JOB}.launch.json
  COMPLETE=$ROOT/seed${JOB}.complete.json
  # This shared claim is the study-wide, atomic no-retry boundary. Keep it
  # even after failures or interruption; a new run root cannot reuse a seed.
  CLAIM=$CLAIMS/$JOB
  mkdir "$CLAIM" 2>/dev/null || fail "study job already claimed: $JOB ($CLAIM)"
  "$PY" - "$CLAIM/owner.json" "$JOB" "$HOST" "$UUID" "$SHA" "$ROOT" <<'PY' || fail "claim receipt failed for $JOB"
import json
import sys
from datetime import datetime, timezone

receipt, job, host, gpu_uuid, commit, root = sys.argv[1:]
with open(receipt, "x", encoding="utf-8") as stream:
    json.dump(dict(job=job, host=host, gpu_uuid=gpu_uuid, release_commit=commit,
                   run_root=root, claimed_utc=datetime.now(timezone.utc).isoformat()),
              stream, sort_keys=True, indent=2)
    stream.write("\n")
PY
  if [[ $ROOT_CREATED -eq 0 ]]; then
    mkdir "$ROOT" || fail "could not claim new run root"
    ROOT_CREATED=1
  fi
  [[ ! -e $OUT && ! -e $LOG && ! -e $LAUNCH && ! -e $COMPLETE ]] ||
    fail "refusing existing artifact for $JOB"

  # Exclusive creation records identity before launch. An interruption leaves
  # a receipt and cannot be mistaken for permission to retry or overwrite.
  "$PY" - "$LAUNCH" "$JOB" "$SEED" "$HOST" "$GPU" "$UUID" "$SHA" "$CONFIG" "$DATA" "$ROOT" "$OUT" <<'PY' || fail "launch receipt failed for $JOB"
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from tralo.knee_experiment import source

receipt, job, seed, host, gpu_index, gpu_uuid, commit, config, data, root, output = sys.argv[1:]
payload = dict(job=job, seed=int(seed), host=host, gpu_index=int(gpu_index),
               gpu_uuid=gpu_uuid, precision="fp32", release_commit=commit,
               source_sha256=source(),
               config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
               data_root=data, run_root=root, output_dir=output,
               started_utc=datetime.now(timezone.utc).isoformat())
with open(receipt, "x", encoding="utf-8") as stream:
    json.dump(payload, stream, sort_keys=True, indent=2)
    stream.write("\n")
PY
  echo "$(date -u +%FT%TZ) START $JOB HOST=$HOST GPU_INDEX=$GPU GPU_UUID=$UUID RELEASE=$SHA"
  CUDA_VISIBLE_DEVICES="$UUID" "$PY" -u -m tralo.fmow_local "$DATA" "$CONFIG" "$OUT" > "$LOG" 2>&1 < /dev/null
  RC=$?
  "$PY" - "$COMPLETE" "$JOB" "$SEED" "$RC" "$HOST" "$UUID" "$SHA" "$ROOT" "$OUT" <<'PY' || fail "completion receipt failed for $JOB"
import json
import sys
from datetime import datetime, timezone

receipt, job, seed, exit_code, host, gpu_uuid, commit, root, output = sys.argv[1:]
payload = dict(job=job, seed=int(seed), exit_code=int(exit_code), host=host,
               gpu_uuid=gpu_uuid, release_commit=commit, run_root=root,
               output_dir=output, ended_utc=datetime.now(timezone.utc).isoformat())
with open(receipt, "x", encoding="utf-8") as stream:
    json.dump(payload, stream, sort_keys=True, indent=2)
    stream.write("\n")
PY
  echo "$(date -u +%FT%TZ) END $JOB EXIT=$RC HOST=$HOST GPU_UUID=$UUID"
  [[ $RC -eq 0 ]] || exit "$RC"
done
