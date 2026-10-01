#!/usr/bin/env bash
# Fixed persistent fmow2 study. Prepare/validate an immutable release before invoking.
# Usage: bash tools/fmow_persistent_local_queue.sh RELEASE_SHA NEW_ROOT GPU_INDEX pilot-step|pilot-ref|full [OLD_GATE PILOT_ROOT REF_ROOT]
set -u

fail() { echo "$*" >&2; exit 2; }
[[ $# -eq 4 || $# -eq 7 ]] || fail "usage: $0 RELEASE_SHA NEW_ROOT GPU_INDEX pilot-step|pilot-ref|full [OLD_GATE PILOT_ROOT REF_ROOT]"
SHA=$1 ROOT=$2 GPU=$3 MODE=$4
[[ $SHA =~ ^[0-9a-f]{40}$ && $ROOT = /* && $ROOT != / && $GPU =~ ^[0-9]+$ ]] || fail "invalid release, root or GPU"
case $MODE in
  pilot-step) [[ $# -eq 4 ]] || fail "pilot takes no gate receipt"; JOBS=(6700_step) ;;
  pilot-ref) [[ $# -eq 4 ]] || fail "pilot takes no gate receipt"; JOBS=(6700_ref) ;;
  full) [[ $# -eq 7 && $5 = /* && -f $5 && ! -L $5 && $6 = /* && $7 = /* ]] ||
          fail "full queue requires pilot gate receipt and preserved step/reference roots";
        GATE=$5; PILOT=$6; REF=$7;
        JOBS=(); for SEED in {6701..6712}; do JOBS+=("${SEED}_step"); done ;;
  *) fail "unknown fixed mode" ;;
esac

REL=/home/dsi/michaer8/tralo-rebuild/releases/$SHA
RUNS=/tmp/tralo-weekend-michaer8-20261001/runs
# Small cross-host claims, cost receipts and GPU leases stay on shared storage;
# model checkpoints and prediction arrays stay on the spacious host-local disk.
REGISTRY=/home/dsi/michaer8/tralo-rebuild/runs
DATA=/home/dsi/michaer8/optloss-audit/data/fmow2/oodslice
PY=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
CONFIGS=$REL/experiments/configs/fmow_persistent_local_20261001
[[ -d $REL && -d $DATA && -d $CONFIGS && -x $PY ]] || fail "release, data, configs or Python missing"
RUNS_CANON=$(realpath -e -- "$RUNS") || fail "runs directory unavailable"
REGISTRY_CANON=$(realpath -e -- "$REGISTRY") || fail "shared registry unavailable"
ROOT_CANON=$(realpath -m -- "$ROOT") || fail "run root unavailable"
[[ $ROOT_CANON = "$RUNS_CANON/"* && -d $(dirname "$ROOT_CANON") && ! -e $ROOT_CANON && ! -L $ROOT_CANON ]] ||
  fail "run root must be new below runs with existing parent"
ROOT=$ROOT_CANON

check_release() {
  [[ $(git -c gc.auto=0 -C "$REL" rev-parse HEAD 2>/dev/null) = "$SHA" ]] || fail "release HEAD differs"
  local dirty
  dirty=$(git -c gc.auto=0 -C "$REL" status --porcelain --untracked-files=all 2>/dev/null) || fail "release status failed"
  [[ -z $dirty ]] || fail "release checkout dirty"
}
check_release
cd "$REL" || fail "cannot enter release"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

CONFIG_ARGS=()
for JOB in "${JOBS[@]}"; do
  CONFIG=$CONFIGS/fmow_persistent_${JOB}.json
  [[ -f $CONFIG && ! -L $CONFIG ]] || fail "missing or linked config $CONFIG"
  git -c gc.auto=0 -C "$REL" ls-files --error-unmatch "experiments/configs/fmow_persistent_local_20261001/fmow_persistent_${JOB}.json" >/dev/null 2>&1 ||
    fail "config not tracked $CONFIG"
  CONFIG_ARGS+=("$CONFIG")
done
"$PY" - "${CONFIG_ARGS[@]}" <<'PY' || fail "fixed config validation failed"
import json, sys
from pathlib import Path
from tralo.fmow_persistent_local import validate_config
for name in sys.argv[1:]:
    path = Path(name)
    seed, role = path.stem.removeprefix('fmow_persistent_').split('_')
    config = json.loads(path.read_text())
    validate_config(config)
    if (config['seed'] != int(seed) or config['reference'] != (role == 'ref') or
            config['pilot'] != (int(seed) == 6700)):
        raise ValueError(f'job/config mismatch: {path}')
PY

UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
[[ $UUID = GPU-* ]] || fail "physical GPU UUID unavailable"
HOST=$(hostname -f 2>/dev/null) || fail "host identity unavailable"
[[ -n $HOST ]] || fail "host identity unavailable"
[[ $HOST = dsisco02* ]] || fail "local /tmp study roots are restricted to dsisco02"

check_gpu_free() {
  local job=$1 current pids
  current=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  [[ $current = "$UUID" ]] || fail "GPU index/UUID changed before $job"
  pids=$(nvidia-smi -i "$UUID" --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ||
    fail "cannot inspect GPU process ownership before $job"
  pids=$(printf '%s\n' "$pids" | sed '/^[[:space:]]*$/d')
  [[ -z $pids ]] || fail "GPU $UUID occupied before $job by compute PID(s): $pids"
}
# The initial check precedes any claim, and checks repeat immediately before each job.
check_gpu_free initial
CLAIMS=$REGISTRY_CANON/.fmow-persistent-local-claims
LOCKS=$REGISTRY_CANON/.fmow-persistent-local-gpu-locks
mkdir "$CLAIMS" 2>/dev/null || [[ -d $CLAIMS && ! -L $CLAIMS ]] || fail "claim directory unavailable"
mkdir "$LOCKS" 2>/dev/null || [[ -d $LOCKS && ! -L $LOCKS ]] || fail "lock directory unavailable"
exec 9>"$LOCKS/$UUID.lock" || fail "cannot open GPU lock"
flock -n 9 || fail "another study queue owns $UUID"
check_gpu_free preflight
if [[ $MODE = full ]]; then
  PILOT=$(realpath -e -- "$PILOT") || fail "pilot root unavailable"
  REF=$(realpath -e -- "$REF") || fail "reference root unavailable"
  [[ $PILOT = "$RUNS_CANON/"* && $REF = "$RUNS_CANON/"* &&
     -d $PILOT && -d $REF && $PILOT != "$REF" ]] || fail "preserved pilot/reference roots invalid"
  "$PY" - "$PILOT" "$REF" "$GATE" "$SHA" <<'PY' || fail "pilot/full release bytes differ"
import json, sys
from pathlib import Path
pilot, reference, gate_file = map(Path, sys.argv[1:4])
commit = sys.argv[4]
gate = json.loads(gate_file.read_text())
for root, role in ((pilot, 'step'), (reference, 'ref')):
    job = f'6700_{role}'
    launch = json.loads((root.parent / f'{job}.launch.json').read_text())
    if (launch.get('release_commit') != commit or
            gate.get('pilot_launch' if role == 'step' else 'reference_launch') != launch):
        raise RuntimeError(f'{job} differs from exact full release')
PY
fi
mkdir "$ROOT" || fail "could not claim fresh run root"
COSTS=$REGISTRY_CANON/.fmow-persistent-local-cost
mkdir "$COSTS" 2>/dev/null || [[ -d $COSTS && ! -L $COSTS ]] || fail "cost registry unavailable"
ROOT_HASH=$(printf '%s' "$ROOT" | sha256sum | cut -d' ' -f1)
ATTEMPT=$COSTS/$ROOT_HASH
mkdir "$ATTEMPT" || fail "GPU attempt already registered; never reuse a root"
"$PY" - "$ATTEMPT/attempt.json" "$ROOT" "$MODE" "$HOST" "$UUID" "$SHA" <<'PY' || fail "cost attempt registration failed"
import json, sys
from datetime import datetime, timezone
path, root, mode, host, uuid, commit = sys.argv[1:]
with open(path, 'x', encoding='utf-8') as stream:
    json.dump(dict(study='fmow_persistent_local_v1', run_root=root, mode=mode,
                   host=host, gpu_uuid=uuid, release_commit=commit,
                   started_utc=datetime.now(timezone.utc).isoformat()),
              stream, sort_keys=True, indent=2)
    stream.write('\n')
PY
"$PY" - "$ATTEMPT/preflight.start.json" "$ROOT" <<'PY' || fail "preflight start receipt failed"
import json, sys
from datetime import datetime, timezone
with open(sys.argv[1], 'x', encoding='utf-8') as stream:
    json.dump(dict(run_root=sys.argv[2], started_utc=datetime.now(timezone.utc).isoformat()), stream)
    stream.write('\n')
PY
"$PY" - "$ATTEMPT/preflight.json" "$ROOT" "$DATA" "${CONFIG_ARGS[@]}" <<'PY' || fail "real-data/source/preprocessing preflight failed; seed unclaimed"
import json, sys, time
from pathlib import Path
from tralo.fmow_persistent_local import (PREPROCESSING, label_free_pool_identity,
                                         load_label_free_training_data,
                                         pretrained_weight_provenance)
from tralo.knee_experiment import source
from analysis.score_fmow_persistent_local import (_data_hashes, _split_and_pool,
    _training_input_fingerprints, base)
path, root, data = sys.argv[1:4]
config_paths = sys.argv[4:]
started = time.monotonic()
record = dict(run_root=root, source_sha256=source(), passed=False)
try:
    configs = [json.loads(Path(p).read_text()) for p in config_paths]
    config = configs[0]
    record['data_files'] = _data_hashes(data)
    images, train_labels, pool, roles = load_label_free_training_data(data)
    split, independent_pool = _split_and_pool(data)
    if (label_free_pool_identity(pool) != independent_pool or
            {key: roles[key] for key in split} != split or
            len(train_labels) != 17670 or len(pool) != 1673 or
            PREPROCESSING != dict(size=[224, 224], color='RGB',
                                  mean=[.485, .456, .406], std=[.229, .224, .225],
                                  train_augmentation=dict(horizontal_flip=.5,
                                      rotation_degrees=3, affine_translate=[.1, .1],
                                      affine_scale=[.9, 1.1], color_jitter=.2))):
        raise RuntimeError('real-data roles, cohort or preprocessing differ')
    record['pretrained_weight'] = pretrained_weight_provenance(config['pretrained_sha256'])
    record['preprocessing'] = PREPROCESSING
    record.update(_training_input_fingerprints(
        images, train_labels, split, sorted({item['seed'] for item in configs})))
    record['quotas'] = base._quotas([row['location'] for row in pool])
    record['train_count'] = len(split['train'])
    record['stop_count'] = len(split['stop'])
    record['development_count'] = len(split['dev'])
    record['reserved_count'] = len(split['reserved_countries'])
    record['reserved_images'] = len(images['test']) - len(split['dev'])
    record['passed'] = True
except Exception as error:
    record['error'] = repr(error)
finally:
    record['elapsed_seconds'] = time.monotonic() - started
    with open(path, 'x', encoding='utf-8') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
if not record['passed']:
    raise SystemExit('preflight failed: ' + record['error'])
PY

check_gpu_free arithmetic-smoke
"$PY" - "$ATTEMPT/smoke.start.json" "$ROOT" "$UUID" <<'PY' || fail "CUDA smoke start receipt failed"
import json, sys
from datetime import datetime, timezone
with open(sys.argv[1], 'x', encoding='utf-8') as stream:
    json.dump(dict(run_root=sys.argv[2], gpu_uuid=sys.argv[3],
                   started_utc=datetime.now(timezone.utc).isoformat()), stream)
    stream.write('\n')
PY
check_gpu_free before-arithmetic-launch
CUDA_VISIBLE_DEVICES="$UUID" "$PY" - "$ATTEMPT/smoke.json" "$ROOT" "$UUID" "$DATA" <<'PY' || fail "exclusive CUDA/real-backbone smoke failed; seed unclaimed"
import json, math, sys, time
import torch
import torch.nn.functional as F
from tralo.fmow_persistent_local import focal_loss, load_label_free_training_data
from tralo.fmow_yuval import ArrayImages, make_model, transforms_for
from tralo.knee_experiment import cuda_setup, source
path, root, uuid, data = sys.argv[1:]
started = time.monotonic()
record = dict(run_root=root, gpu_uuid=uuid, source_sha256=source(), passed=False)
try:
    cuda_setup()
    images, train_labels, pool, roles = load_label_free_training_data(data)
    _, evaluate = transforms_for()
    train = ArrayImages(images['train'], roles['train'], train_labels)
    batch, labels = train.batch([0, 1], evaluate)
    model = make_model(backbone='mobilenet_v3_large').cuda()
    batch, labels = batch.cuda(), labels.cuda()
    model.train()
    logits = model(batch)
    ce = F.cross_entropy(logits, labels)
    ce.backward()
    backbone = next(model.features.parameters()).grad
    grad_norm = float(backbone.double().norm()) if backbone is not None else 0.
    if not math.isfinite(grad_norm) or grad_norm <= 0:
        raise RuntimeError('real MobileNet backbone CE gradient absent')
    model.zero_grad(set_to_none=True)
    focal = focal_loss(model(batch), labels)
    focal.backward()
    focal_norm = float(next(model.features.parameters()).grad.double().norm())
    if not math.isfinite(focal_norm) or focal_norm <= 0:
        raise RuntimeError('real MobileNet backbone focal gradient absent')
    torch.cuda.synchronize()
    record.update(ce_loss=float(ce.detach()), focal_loss=float(focal.detach()),
                  ce_backbone_grad_norm=grad_norm, focal_backbone_grad_norm=focal_norm,
                  gpu_name=torch.cuda.get_device_name(), precision='fp32_tf32_off',
                  passed=True)
except Exception as error:
    record['error'] = repr(error)
finally:
    record['elapsed_seconds'] = time.monotonic() - started
    with open(path, 'x', encoding='utf-8') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
if not record['passed']:
    raise SystemExit('CUDA smoke failed: ' + record['error'])
PY
check_gpu_free after-smoke

if [[ $MODE = full ]]; then
  FRESH=$ROOT/fresh_full_gate.json
  "$PY" - "$ATTEMPT/fresh_gate.start.json" "$ROOT" "$UUID" <<'PY' || fail "fresh-gate start receipt failed"
import json, sys
from datetime import datetime, timezone
with open(sys.argv[1], 'x', encoding='utf-8') as stream:
    json.dump(dict(run_root=sys.argv[2], gpu_uuid=sys.argv[3],
                   started_utc=datetime.now(timezone.utc).isoformat()), stream)
    stream.write('\n')
PY
  # Replay holds this same physical-card lease itself; release and reacquire it
  # around the independent label-blind recount. Seed claims are still untouched.
  flock -u 9 || fail "cannot hand GPU lease to independent scorer"
  START=$(date +%s.%N)
  CUDA_VISIBLE_DEVICES="$UUID" "$PY" analysis/score_fmow_persistent_local.py --replay-device cuda:0 \
    --fresh-full-gate "$GATE" "$PILOT" "$REF" "$DATA" "$COSTS" "$ROOT" "$FRESH"
  GATE_RC=$?
  END=$(date +%s.%N)
  "$PY" - "$ATTEMPT/fresh_gate.json" "$ROOT" "$UUID" "$START" "$END" "$GATE_RC" <<'PY' || fail "fresh-gate completion receipt failed"
import json, sys
from decimal import Decimal
path, root, uuid, start, end, rc = sys.argv[1:]
with open(path, 'x', encoding='utf-8') as stream:
    json.dump(dict(run_root=root, gpu_uuid=uuid,
                   elapsed_seconds=float(Decimal(end)-Decimal(start)),
                   exit_code=int(rc)), stream, sort_keys=True, indent=2)
    stream.write('\n')
PY
  [[ $GATE_RC -eq 0 ]] || fail "fresh pilot artifact/replay/cost gate failed; no seed claimed"
  flock -n 9 || fail "GPU lease changed during fresh gate"
  check_gpu_free after-fresh-gate
  "$PY" - "$FRESH" "$ATTEMPT/fresh_gate.json" <<'PY' || fail "fresh gate exceeded actual GPU cost ceiling"
import json, sys
from pathlib import Path
gate = json.loads(Path(sys.argv[1]).read_text())
spent = json.loads(Path(sys.argv[2]).read_text())
fresh_replay = gate['pilot_replay']['seconds'] + gate['reference_replay']['seconds']
if (gate['status'] != 'full_dispatch_fresh_integrity_pass' or
        gate['projected_gpu_hours'] + max(0., spent['elapsed_seconds']-fresh_replay)/3600 > 24):
    raise RuntimeError('full-gate measured GPU cost exceeds fixed ceiling')
PY
fi

for JOB in "${JOBS[@]}"; do
  check_release
  check_gpu_free "$JOB"
  SEED=${JOB%%_*}
  CONFIG=$CONFIGS/fmow_persistent_${JOB}.json
  OUT=$ROOT/seed$SEED
  [[ $JOB = 6700_ref ]] && OUT=$ROOT/seed6700_ref
  LOG=$ROOT/$JOB.log
  LAUNCH=$ROOT/$JOB.launch.json
  COMPLETE=$ROOT/$JOB.complete.json
  CLAIM=$CLAIMS/$JOB
  mkdir "$CLAIM" 2>/dev/null || fail "seed/role already claimed: $JOB; inspect receipts, never retry blindly"
  [[ ! -e $OUT && ! -e $LOG && ! -e $LAUNCH && ! -e $COMPLETE ]] || fail "job artifact already exists"
  "$PY" - "$LAUNCH" "$CLAIM/owner.json" "$JOB" "$HOST" "$GPU" "$UUID" "$SHA" "$CONFIG" "$DATA" "$ROOT" "$OUT" <<'PY' || fail "launch receipt failed"
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
from tralo.knee_experiment import source
launch, claim, job, host, index, uuid, commit, config, data, root, output = sys.argv[1:]
row = dict(job=job, seed=int(job[:4]), reference=job.endswith('ref'), host=host,
           gpu_index=int(index), gpu_uuid=uuid, precision='fp32_tf32_off',
           release_commit=commit, source_sha256=source(),
           config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
           data_root=data, run_root=root, output_dir=output,
           started_utc=datetime.now(timezone.utc).isoformat())
for path in (claim, launch):
    with open(path, 'x', encoding='utf-8') as stream:
        json.dump(row, stream, sort_keys=True, indent=2)
        stream.write('\n')
PY
  # Hashing/receipt creation can race with other users; refuse and preserve evidence.
  check_gpu_free "$JOB"
  echo "$(date -u +%FT%TZ) START $JOB HOST=$HOST GPU_UUID=$UUID RELEASE=$SHA"
  CUDA_VISIBLE_DEVICES="$UUID" "$PY" -u -m tralo.fmow_persistent_local "$DATA" "$CONFIG" "$OUT" > "$LOG" 2>&1 < /dev/null
  RC=$?
  "$PY" - "$COMPLETE" "$JOB" "$RC" "$HOST" "$UUID" "$SHA" "$OUT" <<'PY' || fail "completion receipt failed"
import json, sys
from datetime import datetime, timezone
path, job, rc, host, uuid, commit, output = sys.argv[1:]
with open(path, 'x', encoding='utf-8') as stream:
    json.dump(dict(job=job, seed=int(job[:4]), exit_code=int(rc), host=host,
                   gpu_uuid=uuid, release_commit=commit, output_dir=output,
                   ended_utc=datetime.now(timezone.utc).isoformat()),
              stream, sort_keys=True, indent=2)
    stream.write('\n')
PY
  echo "$(date -u +%FT%TZ) END $JOB EXIT=$RC HOST=$HOST GPU_UUID=$UUID"
  [[ $RC -eq 0 ]] || exit "$RC"
done
