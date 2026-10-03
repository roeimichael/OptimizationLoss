#!/usr/bin/env bash
# Continue only the four unclaimed fixed seeds after a separate, successful
# label-blind pilot re-audit. Never re-run the completed pilot.
# Usage: bash tools/tabular_calibrated_fixed_after_gate.sh SCRIPT_SHA DATASET GPU_INDEX GATE_JSON [--check-only]
set -euo pipefail
fail() { echo "$*" >&2; exit 2; }
[[ $# -ge 4 && $# -le 5 ]] || fail "expected SCRIPT_SHA DATASET GPU_INDEX GATE_JSON [--check-only]"
MODE=${5:-run}
[[ $MODE = run || $MODE = --check-only ]] || fail "invalid mode"
SCRIPT_SHA=$1 DATASET=$2 GPU=$3 GATE_SOURCE=$4
[[ $SCRIPT_SHA =~ ^[0-9a-f]{40}$ && $GPU =~ ^[0-9]+$ && $GATE_SOURCE = /* ]] || fail "invalid identity"
RUNNER_SHA=034ae537bd6eb0a474eff286ea5225471a260e2c
SCORER_SHA=2619a5d2e0b8bdab0e4eadb6525ba249a1d9d847
case "$DATASET" in
  celeba)
    BASE=6880 HOST_PREFIX=dsisco02
    DATA=/tmp/tralo-celeba-michaer8-20261001/prepared_v2 ;;
  isic2020)
    BASE=6890 HOST_PREFIX=dsisco01
    DATA=/tmp/tralo-isic2020-michaer8-20261001/prepared_v4_rgbcache ;;
  *) fail "unregistered calibrated dataset" ;;
esac
PARENT=/tmp/tralo-weekend-michaer8-20261001
ROOT=$PARENT/runs/tabular_${DATASET}_mnv3_calibrated_${BASE}_$((BASE+4))
SCRIPT_REL=/home/dsi/michaer8/tralo-rebuild/releases/$SCRIPT_SHA
RUNNER_REL=/home/dsi/michaer8/tralo-rebuild/releases/$RUNNER_SHA
SCORER_REL=/home/dsi/michaer8/tralo-rebuild/releases/$SCORER_SHA
PY=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
REGISTRY=/home/dsi/michaer8/tralo-rebuild/runs
LOCKS=$REGISTRY/.fmow-persistent-local-gpu-locks
CLAIMS=$REGISTRY/.tabular-persistent-claims
[[ $(hostname -f) = "$HOST_PREFIX"* && -x $PY && -d $DATA &&
   -d $ROOT/pilot && -d $ROOT/full && -f $ROOT/pilot_${BASE}.complete.json &&
   -f $ROOT/pilot/seed${BASE}/summary.json && -f $GATE_SOURCE &&
   ! -L $GATE_SOURCE ]] || fail "pilot/data/host unavailable"
for identity in "$SCRIPT_SHA:$SCRIPT_REL" "$RUNNER_SHA:$RUNNER_REL" "$SCORER_SHA:$SCORER_REL"; do
  sha=${identity%%:*} rel=${identity#*:}
  [[ -d $rel && $(git -c gc.auto=0 -C "$rel" rev-parse HEAD) = "$sha" &&
     -z $(git -c gc.auto=0 -C "$rel" status --porcelain --untracked-files=all) ]] ||
    fail "immutable release missing or changed: $sha"
done
UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
[[ $UUID = GPU-* ]] || fail "physical GPU UUID unavailable"
check_free() {
  [[ $(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]') = "$UUID" ]] ||
    fail "physical GPU changed"
  local pids
  pids=$(nvidia-smi -i "$UUID" --query-compute-apps=pid --format=csv,noheader) || fail "compute query failed"
  [[ -z $(printf '%s\n' "$pids" | sed '/^[[:space:]]*$/d') ]] || fail "GPU occupied by PID(s): $pids"
}
check_memory() {
  if [[ $DATASET = isic2020 ]]; then
    local available
    available=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    [[ $available =~ ^[0-9]+$ && $available -ge 83886080 ]] || fail "cached cell needs 80 GiB available"
  fi
}
mkdir -p "$LOCKS" "$CLAIMS"
[[ ! -L $LOCKS && ! -L $CLAIMS ]] || fail "linked lock registry"
exec 9>"$LOCKS/$UUID.lock"
flock -n 9 || fail "physical GPU lock occupied"
check_free
check_memory
cd "$RUNNER_REL"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$PY" - "$DATASET" "$BASE" "$ROOT" "$GATE_SOURCE" "$SCORER_REL" "$RUNNER_SHA" "$UUID" "$DATA" <<'PY'
import json, sys
from pathlib import Path
from tralo.knee_experiment import digest, source
from tralo.tabular_persistent_train import validate_config
from tralo.tabular_image_data import load_runner_cohort
from tralo.tabular_backbones import verified_pretrained_weight
dataset, base, root, gate_path, scorer_rel, runner_sha, uuid, data = sys.argv[1:]
base, root = int(base), Path(root)
gate = json.loads(Path(gate_path).read_text())
summary_path = root/'pilot'/f'seed{base}'/'summary.json'
summary = json.loads(summary_path.read_text())
launch = json.loads((root/f'pilot_{base}.launch.json').read_text())
complete = json.loads((root/f'pilot_{base}.complete.json').read_text())
if not (gate['status'] == 'label_blind_integrity_pass' and
        gate['development_labels_accessed'] is False and gate['seed'] == base and
        gate['dataset'] == dataset and gate['backbone'] == 'mobilenet_v3_large' and
        gate['summary_sha256'] == digest(summary_path) and
        gate['scorer_sha256'] == digest(Path(scorer_rel)/'analysis'/'score_tabular_persistent.py') and
        set(gate['applied_correction_counts']) == {
            'level1_tralo','level1_phr','level2_tralo','level2_phr'} and
        all(n > 0 for n in gate['applied_correction_counts'].values()) and
        gate['projected_four_seed_gpu_hours'] * 1.5 <= 24 and
        launch['release_commit'] == complete['release_commit'] == runner_sha and
        launch['gpu_uuid'] == complete['gpu_uuid'] == uuid and
        launch['source_sha256'] == summary['identity']['source_sha256'] == source() and
        launch['prepared_manifest_sha256'] == digest(Path(data)/'manifest.json') and
        complete['exit_code'] == 0 and complete['seed'] == base and
        complete['output_dir'] == str(root/'pilot'/f'seed{base}') and
        summary['config']['seed'] == base):
    raise RuntimeError('pilot gate or runner provenance mismatch')
manifest, rows = load_runner_cohort(data, dataset)
assert manifest['dataset'] == dataset and rows['development_pool']
verified_pretrained_weight('mobilenet_v3_large')
for seed in range(base+1, base+5):
    config = json.loads((Path.cwd()/'experiments'/'configs'/'tabular_persistent_20261002'/
                         f'{dataset}_mobilenet_v3_large_{seed}.json').read_text())
    validate_config(config)
    assert config['seed'] == seed and config['pilot'] is False
PY
[[ ! -e $ROOT/.fixed-continuation-claim && ! -e $ROOT/pilot_gate.json ]] ||
  fail "fixed continuation already claimed"
for SEED in $(seq "$((BASE+1))" "$((BASE+4))"); do
  [[ ! -e $CLAIMS/${DATASET}_mobilenet_v3_large_${SEED} &&
     ! -e $ROOT/full/seed${SEED} && ! -e $ROOT/full_${SEED}.launch.json &&
     ! -e $ROOT/full_${SEED}.log && ! -e $ROOT/full_${SEED}.complete.json ]] ||
    fail "fixed seed already claimed: $SEED"
done
PILOT_SECONDS=$("$PY" - "$ROOT/pilot_${BASE}.complete.json" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1]))['elapsed_seconds']))
PY
)
MAX_CELL_SECONDS=86400
MAX_JOB_SECONDS=21600
remaining_cell_seconds() {
  "$PY" "$SCRIPT_REL/tools/tabular_cell_budget.py" "$ROOT" "$BASE" "$MAX_CELL_SECONDS"
}
REMAINING=$(remaining_cell_seconds) || fail "cell GPU-hour accounting failed"
(( REMAINING > PILOT_SECONDS * 4 * 3 / 2 )) || fail "insufficient measured cell budget for fixed block"
if [[ $MODE = --check-only ]]; then
  echo "preflight_pass dataset=$DATASET base=$BASE gpu_uuid=$UUID remaining_seconds=$REMAINING pilot_seconds=$PILOT_SECONDS"
  exit 0
fi
mkdir "$ROOT/.fixed-continuation-claim" || fail "fixed continuation claim collision"
"$PY" - "$GATE_SOURCE" "$ROOT/pilot_gate.json" <<'PY'
from pathlib import Path
import sys
source, target = map(Path, sys.argv[1:])
with target.open('xb') as output:
    output.write(source.read_bytes())
PY
"$PY" - "$ROOT/fixed_continuation.launch.json" "$SCRIPT_SHA" "$SCORER_SHA" "$RUNNER_SHA" "$GATE_SOURCE" "$UUID" <<'PY'
import json,socket,sys
from datetime import datetime,timezone
from tralo.knee_experiment import digest
path,script,scorer,runner,gate,uuid=sys.argv[1:]
with open(path,'x',encoding='utf-8') as stream:
    json.dump(dict(script_release_commit=script,scorer_release_commit=scorer,
                   runner_release_commit=runner,pilot_gate_sha256=digest(gate),
                   gpu_uuid=uuid,host=socket.getfqdn(),
                   started_utc=datetime.now(timezone.utc).isoformat()),
              stream,sort_keys=True,indent=2)
    stream.write('\n')
PY
for SEED in $(seq "$((BASE+1))" "$((BASE+4))"); do
  REMAINING=$(remaining_cell_seconds) || fail "cell GPU-hour accounting failed"
  (( REMAINING > PILOT_SECONDS * 6 / 5 )) || fail "insufficient cell budget before fixed seed"
  LIMIT=$REMAINING
  (( LIMIT > MAX_JOB_SECONDS )) && LIMIT=$MAX_JOB_SECONDS
  check_free
  check_memory
  CLAIM=$CLAIMS/${DATASET}_mobilenet_v3_large_${SEED}
  OUT=$ROOT/full/seed${SEED}
  CONFIG=$RUNNER_REL/experiments/configs/tabular_persistent_20261002/${DATASET}_mobilenet_v3_large_${SEED}.json
  LOG=$ROOT/full_${SEED}.log
  LAUNCH=$ROOT/full_${SEED}.launch.json
  COMPLETE=$ROOT/full_${SEED}.complete.json
  [[ ! -e $CLAIM && ! -e $OUT && ! -e $LOG && ! -e $LAUNCH && ! -e $COMPLETE ]] ||
    fail "fixed seed already claimed: $SEED"
  mkdir "$CLAIM" || fail "atomic fixed seed claim collision: $SEED"
  START=$(date +%s)
  "$PY" - "$LAUNCH" "$CLAIM/owner.json" "$SEED" "$UUID" "$RUNNER_SHA" "$CONFIG" "$DATA" "$OUT" "$SCRIPT_SHA" "$SCORER_SHA" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
from tralo.knee_experiment import digest,source
launch,claim,seed,uuid,runner_sha,config,data,out,script_sha,scorer_sha=sys.argv[1:]
row=dict(seed=int(seed),host=__import__('socket').getfqdn(),gpu_uuid=uuid,
         release_commit=runner_sha,source_sha256=source(),config_sha256=digest(config),
         prepared_manifest_sha256=digest(Path(data)/'manifest.json'),output_dir=out,
         continuation_script_commit=script_sha,scorer_release_commit=scorer_sha,
         started_utc=datetime.now(timezone.utc).isoformat())
for path in (launch,claim):
    with open(path,'x',encoding='utf-8') as stream:
        json.dump(row,stream,sort_keys=True,indent=2)
        stream.write('\n')
PY
  check_free
  echo "$(date -u +%FT%TZ) START $SEED GPU_UUID=$UUID RUNNER=$RUNNER_SHA SCORER=$SCORER_SHA"
  set +e
  CUDA_VISIBLE_DEVICES="$UUID" timeout -k 60s "${LIMIT}s" \
    "$PY" -u -m tralo.tabular_persistent_train "$DATA" "$CONFIG" "$OUT" > "$LOG" 2>&1 < /dev/null
  RC=$?
  set -e
  END=$(date +%s)
  "$PY" - "$COMPLETE" "$SEED" "$RC" "$UUID" "$RUNNER_SHA" "$OUT" "$((END-START))" <<'PY'
import json,sys,socket
from datetime import datetime,timezone
path,seed,rc,uuid,sha,out,elapsed=sys.argv[1:]
with open(path,'x',encoding='utf-8') as stream:
    json.dump(dict(seed=int(seed),exit_code=int(rc),host=socket.getfqdn(),gpu_uuid=uuid,
                   release_commit=sha,output_dir=out,elapsed_seconds=int(elapsed),
                   ended_utc=datetime.now(timezone.utc).isoformat()),stream,
              sort_keys=True,indent=2)
    stream.write('\n')
PY
  echo "$(date -u +%FT%TZ) END $SEED EXIT=$RC SECONDS=$((END-START))"
  (( RC == 0 )) || fail "fixed seed failed; preserve evidence"
  check_free
  GATE=$ROOT/full_${SEED}.gate.json
  GATE_LOG=$ROOT/full_${SEED}.gate.log
  [[ ! -e $GATE && ! -e $GATE_LOG ]] || fail "fixed seed gate already exists: $SEED"
  echo "$(date -u +%FT%TZ) GATE $SEED START"
  set +e
  (cd "$SCORER_REL" && CUDA_VISIBLE_DEVICES="$UUID" "$PY" -u -m analysis.score_tabular_persistent \
    --gate "$OUT" "$DATA" "$GATE" > "$GATE_LOG" 2>&1 < /dev/null)
  GATE_RC=$?
  set -e
  (( GATE_RC == 0 )) && [[ -f $GATE ]] || fail "fixed seed label-blind gate failed; preserve evidence"
  echo "$(date -u +%FT%TZ) GATE $SEED PASSED"
done
"$PY" - "$ROOT/fixed_continuation.complete.json" "$SCRIPT_SHA" "$UUID" <<'PY'
import json,socket,sys
from datetime import datetime,timezone
path,sha,uuid=sys.argv[1:]
with open(path,'x',encoding='utf-8') as stream:
    json.dump(dict(status='all_four_fixed_seeds_gated',script_release_commit=sha,
                   gpu_uuid=uuid,host=socket.getfqdn(),
                   ended_utc=datetime.now(timezone.utc).isoformat()),
              stream,sort_keys=True,indent=2)
    stream.write('\n')
PY
echo "$(date -u +%FT%TZ) ALL_FIXED_SEEDS_COMPLETE DATASET=$DATASET"
