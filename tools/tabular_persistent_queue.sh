#!/usr/bin/env bash
# One guarded pilot, independent label-blind gate, then four fixed seeds.
# Usage: bash tools/tabular_persistent_queue.sh RELEASE NEW_ROOT GPU_INDEX DATASET BACKBONE
set -u
fail() { echo "$*" >&2; exit 2; }
[[ $# -eq 5 ]] || fail "expected RELEASE NEW_ROOT GPU_INDEX DATASET BACKBONE"
SHA=$1 ROOT=$2 GPU=$3 DATASET=$4 BACKBONE=$5
[[ $SHA =~ ^[0-9a-f]{40}$ && $ROOT = /* && $ROOT != / && $GPU =~ ^[0-9]+$ ]] || fail "invalid identity"
[[ $DATASET = isic2020 || $DATASET = celeba ]] || fail "unfrozen dataset"
[[ $BACKBONE = mobilenet_v3_large || $BACKBONE = vit_b_16 || $BACKBONE = convnext_tiny ]] || fail "unfrozen backbone"

REL=/home/dsi/michaer8/tralo-rebuild/releases/$SHA
RUNS=/tmp/tralo-weekend-michaer8-20261001/runs
REGISTRY=/home/dsi/michaer8/tralo-rebuild/runs
PY=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
CONFIGS=$REL/experiments/configs/tabular_persistent_20261002
case $DATASET in
  isic2020) DATA=/tmp/tralo-isic2020-michaer8-20261001/prepared_v2 ;;
  celeba) DATA=/tmp/tralo-celeba-michaer8-20261001/prepared_v2 ;;
esac
[[ -d $REL && -d $DATA && -x $PY && -d $CONFIGS ]] || fail "release/data/Python missing"
RUNS_CANON=$(realpath -e -- "$RUNS") || fail "run parent missing"
ROOT_CANON=$(realpath -m -- "$ROOT") || fail "bad run root"
[[ $ROOT_CANON = "$RUNS_CANON/"* && -d $(dirname "$ROOT_CANON") &&
   ! -e $ROOT_CANON && ! -L $ROOT_CANON ]] || fail "run root already exists or outside host-local study"
ROOT=$ROOT_CANON
HOST=$(hostname -f) || fail "host unavailable"
[[ $HOST = dsisco02* ]] || fail "host-local study restricted to dsisco02"
UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
[[ $UUID = GPU-* ]] || fail "physical UUID unavailable"
check_free() {
  local current pids
  current=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  [[ $current = "$UUID" ]] || fail "selected GPU UUID changed"
  pids=$(nvidia-smi -i "$UUID" --query-compute-apps=pid --format=csv,noheader 2>/dev/null) || fail "GPU process query failed"
  [[ -z $(printf '%s\n' "$pids" | sed '/^[[:space:]]*$/d') ]] || fail "GPU occupied by compute PID(s): $pids"
}
check_release() {
  [[ $(git -c gc.auto=0 -C "$REL" rev-parse HEAD 2>/dev/null) = "$SHA" ]] || fail "release HEAD changed"
  [[ -z $(git -c gc.auto=0 -C "$REL" status --porcelain --untracked-files=all 2>/dev/null) ]] || fail "release dirty"
}
check_release
cd "$REL" || fail "release unavailable"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$PY" - "$DATA" "$DATASET" "$BACKBONE" "$CONFIGS" <<'PY' || fail "data/weights/config preflight failed"
import json, sys
from pathlib import Path
from tralo.tabular_image_data import load_runner_cohort
from tralo.tabular_backbones import verified_pretrained_weight
from tralo.tabular_persistent_train import validate_config
from tralo.tabular_quota_policy import caps_for_unlabeled_pool
root, dataset, backbone, configs = sys.argv[1:]
manifest, rows = load_runner_cohort(root, dataset)
quota = caps_for_unlabeled_pool(dataset, [r['group'] for r in rows['development_pool']])
verified_pretrained_weight(backbone)
for seed in range(6800, 6805):
    config = json.loads((Path(configs) / f'{dataset}_{backbone}_{seed}.json').read_text())
    validate_config(config)
    if (config['dataset'], config['backbone'], config['seed'], config['pilot']) != (
            dataset, backbone, seed, seed == 6800):
        raise RuntimeError('frozen job/config mismatch')
assert len(quota) == 2 and manifest['dataset'] == dataset
PY
check_free
LOCKS=$REGISTRY/.fmow-persistent-local-gpu-locks
CLAIMS=$REGISTRY/.tabular-persistent-claims
mkdir "$LOCKS" 2>/dev/null || [[ -d $LOCKS && ! -L $LOCKS ]] || fail "shared lock directory unavailable"
mkdir "$CLAIMS" 2>/dev/null || [[ -d $CLAIMS && ! -L $CLAIMS ]] || fail "claim directory unavailable"
exec 9>"$LOCKS/$UUID.lock" || fail "cannot open physical GPU lock"
flock -n 9 || fail "another study queue owns physical GPU"
check_free
mkdir "$ROOT" || fail "fresh root claim failed"
mkdir "$ROOT/pilot" "$ROOT/full" || fail "run subdirectories unavailable"
QUEUE_START=$(date +%s)
MAX_QUEUE_SECONDS=86400
MAX_JOB_SECONDS=21600

run_seed() {
  local SEED=$1 PHASE=$2 OUT=$3 CONFIG LOG LAUNCH COMPLETE CLAIM START END RC LIMIT
  check_release
  check_free
  CONFIG=$CONFIGS/${DATASET}_${BACKBONE}_${SEED}.json
  [[ -f $CONFIG && ! -L $CONFIG ]] || fail "missing or linked config"
  CLAIM=$CLAIMS/${DATASET}_${BACKBONE}_${SEED}
  LOG=$ROOT/${PHASE}_${SEED}.log
  LAUNCH=$ROOT/${PHASE}_${SEED}.launch.json
  COMPLETE=$ROOT/${PHASE}_${SEED}.complete.json
  [[ ! -e $OUT && ! -e $LOG && ! -e $LAUNCH && ! -e $COMPLETE ]] || fail "seed artifacts already exist"
  START=$(date +%s)
  LIMIT=$((MAX_QUEUE_SECONDS - START + QUEUE_START))
  (( LIMIT > MAX_JOB_SECONDS )) && LIMIT=$MAX_JOB_SECONDS
  (( LIMIT > 0 )) || fail "queue cost ceiling exhausted before seed claim"
  mkdir "$CLAIM" || fail "seed already claimed: ${DATASET}/${BACKBONE}/${SEED}; never retry blindly"
  "$PY" - "$LAUNCH" "$CLAIM/owner.json" "$SEED" "$HOST" "$UUID" "$SHA" "$CONFIG" "$DATA" "$OUT" <<'PY' || fail "launch receipt failed"
import json, sys
from datetime import datetime, timezone
from pathlib import Path
from tralo.knee_experiment import digest, source
launch, claim, seed, host, uuid, sha, config, data, out = sys.argv[1:]
record = dict(seed=int(seed), host=host, gpu_uuid=uuid, release_commit=sha,
              source_sha256=source(), config_sha256=digest(config),
              prepared_manifest_sha256=digest(Path(data)/'manifest.json'),
              output_dir=out, started_utc=datetime.now(timezone.utc).isoformat())
for name in (claim, launch):
    with open(name, 'x', encoding='utf-8') as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write('\n')
PY
  check_free
  echo "$(date -u +%FT%TZ) START $SEED DATASET=$DATASET BACKBONE=$BACKBONE GPU_UUID=$UUID"
  timeout -k 60s "${LIMIT}s" env CUDA_VISIBLE_DEVICES="$UUID" \
    "$PY" -u -m tralo.tabular_persistent_train "$DATA" "$CONFIG" "$OUT" > "$LOG" 2>&1 < /dev/null
  RC=$?
  END=$(date +%s)
  "$PY" - "$COMPLETE" "$SEED" "$RC" "$HOST" "$UUID" "$SHA" "$OUT" "$((END-START))" <<'PY' || fail "completion receipt failed"
import json, sys
from datetime import datetime, timezone
path, seed, rc, host, uuid, sha, out, elapsed = sys.argv[1:]
with open(path, 'x', encoding='utf-8') as stream:
    json.dump(dict(seed=int(seed), exit_code=int(rc), host=host, gpu_uuid=uuid,
                   release_commit=sha, output_dir=out, elapsed_seconds=int(elapsed),
                   ended_utc=datetime.now(timezone.utc).isoformat()), stream,
              sort_keys=True, indent=2)
    stream.write('\n')
PY
  echo "$(date -u +%FT%TZ) END $SEED EXIT=$RC SECONDS=$((END-START))"
  [[ $RC -eq 0 ]] || fail "seed failed; inspect evidence before any new job"
}

run_seed 6800 pilot "$ROOT/pilot/seed6800"
check_free
GATE=$ROOT/pilot_gate.json
GATE_LOG=$ROOT/pilot_gate.log
echo "$(date -u +%FT%TZ) GATE 6800 START"
CUDA_VISIBLE_DEVICES="$UUID" "$PY" -u -m analysis.score_tabular_persistent \
  --gate "$ROOT/pilot/seed6800" "$DATA" "$GATE" > "$GATE_LOG" 2>&1 < /dev/null
GATE_RC=$?
[[ $GATE_RC -eq 0 && -f $GATE ]] || fail "pilot gate failed; preserve raw run and gate log"
"$PY" - "$GATE" "$SHA" "$ROOT/pilot/seed6800" <<'PY' || fail "pilot gate/cost/source mismatch"
import json, sys
from pathlib import Path
from tralo.knee_experiment import digest
gate = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[3]); summary = json.loads((root/'summary.json').read_text())
if (gate['status'] != 'label_blind_integrity_pass' or
        gate['seed'] != 6800 or gate['summary_sha256'] != digest(root/'summary.json') or
        gate['dataset'] != summary['config']['dataset'] or
        gate['backbone'] != summary['config']['backbone'] or
        gate['projected_four_seed_gpu_hours'] * 1.5 > 24):
    raise RuntimeError('pilot cannot authorize a 24-hour cell')
PY
check_free
PILOT_SECONDS=$(( $(date +%s) - QUEUE_START ))
for SEED in 6801 6802 6803 6804; do
  REMAINING=$((MAX_QUEUE_SECONDS - $(date +%s) + QUEUE_START))
  (( REMAINING > PILOT_SECONDS * 6 / 5 )) || fail "insufficient budget to claim next fixed seed"
  run_seed "$SEED" full "$ROOT/full/seed$SEED"
done
echo "$(date -u +%FT%TZ) ALL_FIXED_SEEDS_COMPLETE DATASET=$DATASET BACKBONE=$BACKBONE"
