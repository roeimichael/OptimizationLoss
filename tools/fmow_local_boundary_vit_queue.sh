#!/usr/bin/env bash
# One exclusive card for the registered fmow2 ViT-B/16 calibrated-boundary study.
# The immutable release runs exclusive memory and real-image numerical checks
# before claiming any study seed.
# Usage: bash tools/fmow_local_boundary_vit_queue.sh RELEASE_SHA NEW_RUN_ROOT GPU_INDEX pilot-step|pilot-ref|full
set -u

usage() { echo "usage: $0 RELEASE_SHA NEW_RUN_ROOT GPU_INDEX pilot-step|pilot-ref|full" >&2; exit 2; }
fail() { echo "$*" >&2; exit 2; }
[[ $# -eq 4 ]] || usage
SHA=$1; ROOT=$2; GPU=$3; MODE=$4
[[ $SHA =~ ^[0-9a-f]{40}$ ]] || fail "invalid release SHA"
[[ $ROOT = /* && $ROOT != / && $GPU =~ ^[0-9]+$ ]] || fail "invalid run root or GPU index"
case $MODE in
  pilot-step) JOBS=(6500_step) ;;
  pilot-ref) JOBS=(6500_ref) ;;
  full) JOBS=(); for SEED in {6501..6512}; do JOBS+=("${SEED}_step"); done ;;
  *) usage ;;
esac

REL=/home/dsi/michaer8/tralo-rebuild/releases/$SHA
RUNS=/home/dsi/michaer8/tralo-rebuild/runs
DATA=/home/dsi/michaer8/optloss-audit/data/fmow2/oodslice
PY=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
CONFIGS=$REL/experiments/configs/fmow_local_boundary_vit_20260930
[[ -d $REL && -d $DATA && -d $CONFIGS && -x $PY ]] || fail "missing release, data, configs or interpreter"
RUNS_CANON=$(realpath -e -- "$RUNS") || fail "runs directory unavailable"
ROOT_CANON=$(realpath -m -- "$ROOT") || fail "cannot resolve run root"
[[ $ROOT_CANON = "$RUNS_CANON/"* ]] || fail "run root must be below $RUNS_CANON"
ROOT=$ROOT_CANON
[[ -d $(dirname "$ROOT") && ! -e $ROOT && ! -L $ROOT ]] ||
  fail "run root must be new and its parent must exist: $ROOT"

check_release() {
  [[ $(git -c gc.auto=0 -C "$REL" rev-parse HEAD 2>/dev/null) = "$SHA" ]] || fail "release HEAD differs from $SHA"
  if ! DIRTY=$(git -c gc.auto=0 -C "$REL" status --porcelain --untracked-files=all 2>/dev/null); then
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
  git -c gc.auto=0 -C "$REL" ls-files --error-unmatch "experiments/configs/fmow_local_boundary_vit_20260930/fmow_local_${JOB}.json" >/dev/null 2>&1 ||
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
    if (config["study"] != "local_boundary_vit_v1"
            or config["seed"] != int(seed)
            or config["snapshot_steps"] is not (arm == "step")):
        raise ValueError(f"config does not match job {job}")
PY

UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
[[ $UUID = GPU-* ]] || fail "GPU UUID unavailable"
HOST=$(hostname -f 2>/dev/null) || fail "host identity unavailable"
[[ -n $HOST ]] || fail "host identity unavailable"
CLAIMS=$RUNS_CANON/.fmow-local-boundary-vit-claims
if ! mkdir "$CLAIMS" 2>/dev/null; then
  [[ -d $CLAIMS && ! -L $CLAIMS ]] || fail "study claim directory unavailable"
fi
for JOB in "${JOBS[@]}"; do
  [[ ! -e $CLAIMS/$JOB && ! -L $CLAIMS/$JOB ]] ||
    fail "study job already claimed: $JOB ($CLAIMS/$JOB)"
done
LOCKS=$RUNS_CANON/.fmow-local-boundary-gpu-locks
if ! mkdir "$LOCKS" 2>/dev/null; then
  [[ -d $LOCKS && ! -L $LOCKS ]] || fail "study GPU lock directory unavailable"
fi
# Hold one cooperating queue per physical card, including between seeds.
exec 9>"$LOCKS/$UUID.lock" || fail "cannot open GPU lock"
flock -n 9 || fail "another boundary queue owns GPU $UUID"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

check_gpu_free() {
  local job=$1 current_uuid pids
  current_uuid=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')
  [[ $current_uuid = "$UUID" ]] || fail "GPU index ownership changed before $job"
  # Query by UUID so an index remap cannot move this job to a different card.
  if ! pids=$(nvidia-smi -i "$UUID" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); then
    fail "GPU process query failed before $job"
  fi
  pids=$(printf '%s\n' "$pids" | sed '/^[[:space:]]*$/d')
  [[ -z $pids ]] || fail "GPU $UUID has compute PIDs before $job: $pids"
}

check_gpu_free "ViT memory smoke"
mkdir "$ROOT" || fail "could not claim new run root"
PILOT_GATE_RECHECK=
if [[ $MODE = full ]]; then
  STEP_ROOT=${FMOW_VIT_PILOT_STEP_ROOT:-}
  REF_ROOT=${FMOW_VIT_PILOT_REF_ROOT:-}
  PILOT_GATE_INPUT=${FMOW_VIT_PILOT_GATE_RECEIPT:-}
  [[ -n $STEP_ROOT && -n $REF_ROOT && -n $PILOT_GATE_INPUT && -f $PILOT_GATE_INPUT && ! -L $PILOT_GATE_INPUT ]] ||
    fail "full queue requires both pilot roots and a regular-file pilot gate receipt; preserve $ROOT"
  PILOT_GATE_RECHECK=$ROOT/vit_pilot_gate_recheck.json
  "$PY" analysis/score_fmow_boundary_vit.py --gate "$STEP_ROOT" "$REF_ROOT" "$PILOT_GATE_RECHECK" \
    > "$ROOT/vit_pilot_gate_recheck.log" 2>&1 ||
    fail "independent pilot gate recomputation failed; preserve $ROOT"
  "$PY" - "$PILOT_GATE_INPUT" "$PILOT_GATE_RECHECK" <<'PY' || fail "supplied pilot gate differs from independent recomputation; preserve $ROOT"
import json
import sys
from pathlib import Path

supplied, rechecked = [json.loads(Path(path).read_text()) for path in sys.argv[1:]]
if (supplied != rechecked or
        rechecked.get("status") != "vit_pilot_integrity_pass"):
    raise ValueError("pilot gate is absent, failed or differs from recomputation")
PY
fi
SMOKE=$ROOT/vit_memory_smoke.json
SMOKE_LOG=$ROOT/vit_memory_smoke.log
SMOKE_LAUNCH=$ROOT/vit_memory_smoke.launch.json
SMOKE_COMPLETE=$ROOT/vit_memory_smoke.complete.json
"$PY" - "$SMOKE_LAUNCH" "$SHA" "$HOST" "$GPU" "$UUID" "$SMOKE" <<'PY' || fail "smoke launch receipt failed"
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from tralo.knee_experiment import digest

path, commit, host, gpu_index, gpu_uuid, output = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump(dict(release_commit=commit, host=host, gpu_index=int(gpu_index),
                   gpu_uuid=gpu_uuid, receipt_path=output,
                   generator_sha256=digest(Path("tools/fmow_local_boundary_vit_smoke.py")),
                   started_utc=datetime.now(timezone.utc).isoformat()),
              stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
check_gpu_free "ViT memory smoke"
FMOW_VIT_INHERITED_LOCK_FD=9 "$PY" -u -m tools.fmow_local_boundary_vit_smoke \
  "$SHA" "$DATA" "$GPU" "$SMOKE" > "$SMOKE_LOG" 2>&1 < /dev/null
SMOKE_RC=$?
if [[ $SMOKE_RC -ne 0 && ! -e $SMOKE && ! -L $SMOKE ]]; then
  "$PY" - "$SMOKE" "$SHA" "$HOST" "$UUID" "$SMOKE_RC" <<'PY' || fail "negative smoke receipt failed"
import json
import sys
from datetime import datetime, timezone

path, commit, host, gpu_uuid, code = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump(dict(memory_smoke_passed=False, release_commit=commit, host=host,
                   gpu_uuid=gpu_uuid, exit_code=int(code),
                   failure_message="generator exited before writing a receipt; see smoke log",
                   ended_utc=datetime.now(timezone.utc).isoformat()),
              stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
fi
"$PY" - "$SMOKE_COMPLETE" "$SMOKE_RC" "$SHA" "$HOST" "$UUID" "$SMOKE" <<'PY' || fail "smoke completion receipt failed"
import json
import sys
from datetime import datetime, timezone

path, code, commit, host, gpu_uuid, output = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump(dict(exit_code=int(code), release_commit=commit, host=host,
                   gpu_uuid=gpu_uuid, receipt_path=output,
                   ended_utc=datetime.now(timezone.utc).isoformat()),
              stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
[[ $SMOKE_RC -eq 0 ]] || fail "ViT memory smoke failed (exit $SMOKE_RC); preserve $ROOT"
"$PY" - "$SMOKE" "$SMOKE_COMPLETE" "$SHA" "$HOST" "$GPU" "$UUID" <<'PY' || fail "ViT memory-smoke provenance failed"
import json
import sys
from pathlib import Path
from tralo.fmow_local import VIT_WEIGHT_SHA256, vit_weight_provenance
from tralo.knee_experiment import digest, source
from tools.fmow_local_boundary_vit_smoke import validate_success_receipt

receipt, completion, commit, host, gpu_index, gpu_uuid = sys.argv[1:]
smoke = json.loads(Path(receipt).read_text())
done = json.loads(Path(completion).read_text())
expected = dict(release_commit=commit, host=host, gpu_uuid=gpu_uuid,
                gpu_index=int(gpu_index), backbone="vit_b_16", batch_size=16,
                development_batch_size=8, weight_sha256=VIT_WEIGHT_SHA256,
                precision="fp32", label_free=True, memory_smoke_passed=True)
if any(smoke.get(key) != value for key, value in expected.items()):
    raise ValueError("memory smoke does not match this release, card or fixed recipe")
if done != {"exit_code": 0, "release_commit": commit, "host": host,
            "gpu_uuid": gpu_uuid, "receipt_path": receipt,
            "ended_utc": done.get("ended_utc")}:
    raise ValueError("memory-smoke completion receipt differs")
if (smoke.get("source_sha256") != source() or
        smoke.get("smoke_generator_sha256") != digest(
            Path("tools/fmow_local_boundary_vit_smoke.py")) or
        not isinstance(smoke.get("device_name"), str) or not smoke["device_name"]):
    raise ValueError("memory smoke source or device identity differs")
validate_success_receipt(smoke)
vit_weight_provenance()
PY
REAL=$ROOT/vit_real_preflight.json
REAL_LOG=$ROOT/vit_real_preflight.log
REAL_LAUNCH=$ROOT/vit_real_preflight.launch.json
REAL_COMPLETE=$ROOT/vit_real_preflight.complete.json
"$PY" - "$REAL_LAUNCH" "$SHA" "$HOST" "$GPU" "$UUID" "$REAL" <<'PY' || fail "real-image preflight launch receipt failed"
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from tralo.knee_experiment import digest

path, commit, host, gpu_index, gpu_uuid, output = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump(dict(release_commit=commit, host=host, gpu_index=int(gpu_index),
                   gpu_uuid=gpu_uuid, receipt_path=output,
                   generator_sha256=digest(Path("tools/fmow_local_boundary_vit_real_preflight.py")),
                   started_utc=datetime.now(timezone.utc).isoformat()),
              stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
check_gpu_free "ViT real-image preflight"
FMOW_VIT_INHERITED_LOCK_FD=9 "$PY" -u -m tools.fmow_local_boundary_vit_real_preflight \
  "$SHA" "$DATA" "$GPU" "$REAL" > "$REAL_LOG" 2>&1 < /dev/null
REAL_RC=$?
if [[ $REAL_RC -ne 0 && ! -e $REAL && ! -L $REAL ]]; then
  "$PY" - "$REAL" "$SHA" "$HOST" "$UUID" "$REAL_RC" <<'PY' || fail "negative real-image receipt failed"
import json
import sys
from datetime import datetime, timezone

path, commit, host, gpu_uuid, code = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump(dict(preflight_passed=False, release_commit=commit, host=host,
                   gpu_uuid=gpu_uuid, exit_code=int(code),
                   failure_message="generator exited before writing a receipt; see preflight log",
                   ended_utc=datetime.now(timezone.utc).isoformat()),
              stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
fi
"$PY" - "$REAL_COMPLETE" "$REAL_RC" "$SHA" "$HOST" "$UUID" "$REAL" <<'PY' || fail "real-image completion receipt failed"
import json
import sys
from datetime import datetime, timezone

path, code, commit, host, gpu_uuid, output = sys.argv[1:]
with open(path, "x", encoding="utf-8") as stream:
    json.dump(dict(exit_code=int(code), release_commit=commit, host=host,
                   gpu_uuid=gpu_uuid, receipt_path=output,
                   ended_utc=datetime.now(timezone.utc).isoformat()),
              stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
[[ $REAL_RC -eq 0 ]] || fail "ViT real-image preflight failed (exit $REAL_RC); preserve $ROOT"
"$PY" - "$REAL" "$REAL_COMPLETE" "$SHA" "$HOST" "$GPU" "$UUID" <<'PY' || fail "ViT real-image preflight provenance failed"
import json
import sys
from pathlib import Path
from tralo.fmow_local import VIT_WEIGHT_SHA256, vit_weight_provenance
from tralo.fmow_yuval import FILES
from tralo.knee_experiment import digest, source
from tools.fmow_local_boundary_vit_real_preflight import validate_success_receipt

receipt, completion, commit, host, gpu_index, gpu_uuid = sys.argv[1:]
real = json.loads(Path(receipt).read_text())
done = json.loads(Path(completion).read_text())
expected = dict(release_commit=commit, host=host, gpu_uuid=gpu_uuid,
                gpu_index=int(gpu_index), backbone="vit_b_16",
                weight_sha256=VIT_WEIGHT_SHA256, precision="fp32",
                development_batch_size=8, chunk_sizes=[8, 7],
                development_labels_accessed=False, preflight_passed=True,
                data_file_sha256=FILES)
if any(real.get(key) != value for key, value in expected.items()):
    raise ValueError("real-image preflight differs from this release, card or data")
if done != {"exit_code": 0, "release_commit": commit, "host": host,
            "gpu_uuid": gpu_uuid, "receipt_path": receipt,
            "ended_utc": done.get("ended_utc")}:
    raise ValueError("real-image completion receipt differs")
if (real.get("source_sha256") != source() or
        real.get("preflight_generator_sha256") != digest(
            Path("tools/fmow_local_boundary_vit_real_preflight.py")) or
        real.get("pretrained_weight") != vit_weight_provenance() or
        not isinstance(real.get("device_name"), str) or not real["device_name"]):
    raise ValueError("real-image preflight source, checkpoint or device differs")
artifacts = Path(receipt).with_suffix(".artifacts")
if any(digest(artifacts / name) != sha for name, sha in real["artifact_sha256"].items()):
    raise ValueError("real-image preflight artifact bytes differ")
validate_success_receipt(real)
PY
COST_GATE=
if [[ $MODE = full ]]; then
  COST_GATE=$ROOT/vit_cost_gate.json
  "$PY" - "$COST_GATE" "$STEP_ROOT" "$REF_ROOT" "$ROOT" "$SHA" "$HOST" "$RUNS_CANON" "$PILOT_GATE_RECHECK" <<'PY' || fail "ViT full queue exceeds or cannot verify 24 GPU-hour ceiling; preserve $ROOT"
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

output, step_root, ref_root, full_root, commit, host, runs_root, pilot_gate = sys.argv[1:]
result = dict(release_commit=commit, host=host, ceiling_gpu_hours=24.0,
              gate_passed=False, pilot_step_root=step_root,
              pilot_ref_root=ref_root, full_root=full_root,
              pilot_gate_receipt_path=pilot_gate,
              pilot_gate_receipt_sha256=hashlib.sha256(Path(pilot_gate).read_bytes()).hexdigest())

def read(root, name):
    return json.loads((root / name).read_text())

def duration(root, job):
    launch = read(root, f"seed{job}.launch.json")
    done = read(root, f"seed{job}.complete.json")
    if (launch.get("release_commit") != commit or done.get("release_commit") != commit or
            launch.get("host") != host or done.get("host") != host or
            done.get("exit_code") != 0):
        raise ValueError("pilot seed provenance or completion differs")
    seconds = (datetime.fromisoformat(done["ended_utc"]) -
               datetime.fromisoformat(launch["started_utc"])).total_seconds()
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("invalid pilot seed runtime")
    return seconds

def smoke_duration(root):
    smoke = read(root, "vit_memory_smoke.json")
    launch = read(root, "vit_memory_smoke.launch.json")
    done = read(root, "vit_memory_smoke.complete.json")
    if (smoke.get("release_commit") != commit or smoke.get("host") != host or
            smoke.get("memory_smoke_passed") is not True or
            launch.get("release_commit") != commit or done.get("release_commit") != commit or
            launch.get("host") != host or done.get("host") != host or
            done.get("exit_code") != 0):
        raise ValueError("pilot/full smoke provenance differs")
    seconds = (datetime.fromisoformat(done["ended_utc"]) -
               datetime.fromisoformat(launch["started_utc"])).total_seconds()
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("invalid GPU smoke runtime")
    return seconds

def preflight_duration(root):
    real = read(root, "vit_real_preflight.json")
    launch = read(root, "vit_real_preflight.launch.json")
    done = read(root, "vit_real_preflight.complete.json")
    if (real.get("release_commit") != commit or real.get("host") != host or
            real.get("preflight_passed") is not True or
            launch.get("release_commit") != commit or done.get("release_commit") != commit or
            launch.get("host") != host or done.get("host") != host or
            done.get("exit_code") != 0):
        raise ValueError("pilot/full real-image preflight provenance differs")
    seconds = (datetime.fromisoformat(done["ended_utc"]) -
               datetime.fromisoformat(launch["started_utc"])).total_seconds()
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("invalid real-image preflight runtime")
    return seconds

try:
    runs = Path(runs_root).resolve(strict=True)
    step = Path(step_root).resolve(strict=True)
    ref = Path(ref_root).resolve(strict=True)
    full = Path(full_root).resolve(strict=True)
    if (not step.is_relative_to(runs) or not ref.is_relative_to(runs) or
            not full.is_relative_to(runs) or len({step, ref, full}) != 3):
        raise ValueError("pilot and full roots must be distinct under owned runs")
    step_seconds = duration(step, "6500_step")
    ref_seconds = duration(ref, "6500_ref")
    smoke_seconds = {"pilot_step": smoke_duration(step),
                     "pilot_ref": smoke_duration(ref),
                     "full": smoke_duration(full)}
    preflight_seconds = {"pilot_step": preflight_duration(step),
                         "pilot_ref": preflight_duration(ref),
                         "full": preflight_duration(full)}
    # Preserve a 25% runtime margin for the fixed training/step workload.
    projected_hours = (1.25 * (13 * step_seconds + ref_seconds) +
                       sum(smoke_seconds.values()) +
                       sum(preflight_seconds.values())) / 3600
    result.update(pilot_step_seconds=step_seconds, pilot_ref_seconds=ref_seconds,
                  smoke_seconds=smoke_seconds,
                  preflight_seconds=preflight_seconds,
                  projected_gpu_hours=projected_hours,
                  gate_passed=projected_hours <= 24.0)
except Exception as exc:
    result.update(failure_type=type(exc).__name__, failure_message=str(exc))

with open(output, "x", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write("\n")
if not result["gate_passed"]:
    raise SystemExit(2)
PY
fi

for JOB in "${JOBS[@]}"; do
  check_release
  check_gpu_free "$JOB"

  SEED=${JOB%%_*}
  OUT=$ROOT/seed$SEED
  [[ $JOB = 6500_ref ]] && OUT=$ROOT/seed6500_ref
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
  [[ ! -e $OUT && ! -e $LOG && ! -e $LAUNCH && ! -e $COMPLETE ]] ||
    fail "refusing existing artifact for $JOB"

  # Exclusive creation records identity before launch. An interruption leaves
  # a receipt and cannot be mistaken for permission to retry or overwrite.
  "$PY" - "$LAUNCH" "$JOB" "$SEED" "$HOST" "$GPU" "$UUID" "$SHA" "$CONFIG" "$DATA" "$ROOT" "$OUT" "$SMOKE" "$SMOKE_LAUNCH" "$SMOKE_COMPLETE" "$REAL" "$REAL_LAUNCH" "$REAL_COMPLETE" "$COST_GATE" "$PILOT_GATE_RECHECK" <<'PY' || fail "launch receipt failed for $JOB"
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from tralo.knee_experiment import source

receipt, job, seed, host, gpu_index, gpu_uuid, commit, config, data, root, output, smoke, smoke_launch, smoke_complete, real, real_launch, real_complete, cost_gate, pilot_gate = sys.argv[1:]
payload = dict(job=job, seed=int(seed), host=host, gpu_index=int(gpu_index),
               gpu_uuid=gpu_uuid, precision="fp32", release_commit=commit,
               source_sha256=source(),
               config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
               data_root=data, run_root=root, output_dir=output,
               memory_smoke_receipt_path=smoke,
               memory_smoke_receipt_sha256=hashlib.sha256(Path(smoke).read_bytes()).hexdigest(),
               memory_smoke_launch_path=smoke_launch,
               memory_smoke_launch_sha256=hashlib.sha256(Path(smoke_launch).read_bytes()).hexdigest(),
               memory_smoke_complete_path=smoke_complete,
               memory_smoke_complete_sha256=hashlib.sha256(Path(smoke_complete).read_bytes()).hexdigest(),
               memory_smoke_execution="queue_executed",
               real_preflight_receipt_path=real,
               real_preflight_receipt_sha256=hashlib.sha256(Path(real).read_bytes()).hexdigest(),
               real_preflight_launch_path=real_launch,
               real_preflight_launch_sha256=hashlib.sha256(Path(real_launch).read_bytes()).hexdigest(),
               real_preflight_complete_path=real_complete,
               real_preflight_complete_sha256=hashlib.sha256(Path(real_complete).read_bytes()).hexdigest(),
               real_preflight_execution="queue_executed",
               started_utc=datetime.now(timezone.utc).isoformat())
if cost_gate:
    payload.update(cost_gate_receipt_path=cost_gate,
                   cost_gate_receipt_sha256=hashlib.sha256(Path(cost_gate).read_bytes()).hexdigest(),
                   pilot_gate_receipt_path=pilot_gate,
                   pilot_gate_receipt_sha256=hashlib.sha256(Path(pilot_gate).read_bytes()).hexdigest())
with open(receipt, "x", encoding="utf-8") as stream:
    json.dump(payload, stream, sort_keys=True, indent=2)
    stream.write("\n")
PY
  # Receipts/source hashing take time; refuse if another user claimed the card
  # during that interval. Keep the claim and receipt as forensic evidence.
  check_gpu_free "$JOB"
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
