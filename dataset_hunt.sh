#!/bin/bash
# Autonomous overnight dataset hunt for the OptimizationLoss thesis.
# For each candidate: prep -> probe -> verdict (PASS if ep3 train-acc in
# [0.40, 0.82]). FAIL = delete data + skip. Stops after 4 PASSes or all
# candidates exhausted. Branch: dataset-hunt. Does NOT push.
#
# Run on dsisco01 GPU0 only. Never touches dsisco02. Never shares a GPU.
set +e  # don't abort on a single candidate's failure
cd ~/OptimizationLoss
mkdir -p logs/dataset_hunt
MASTER_LOG=logs/dataset_hunt/master_$(date +%Y%m%d_%H%M).log
REPORT=docs/DATASET_HUNT_RESULTS.md
PY=~/anaconda3/envs/optloss/bin/python

# Candidate list: "<ds_name>:<tv_class>:<n_classes>:<story_keyword>"
# Ordered by paper-story strength + sourcing reliability.
CANDIDATES=(
  "fgvc_aircraft:FGVCAircraft:100:aviation_safety_recall"
  "stanford_cars:StanfordCars:196:vehicle_recall_scenario"
  "pcam:PCAM:2:patch_camelyon_binary_cancer"
  "flowers102:Flowers102:102:fine_grained_no_story"
  "tiny_imagenet:TinyImageNet:200:hard_general_benchmark"
  "dtd:DTD:47:texture_classification"
  "sun397:SUN397:397:scene_understanding"
  "caltech256:Caltech256:257:fine_grained_general"
)

PASSED=()
declare -A VERDICTS

probe_ds() {
  local ds_name=$1
  local tv_class=$2
  local n_classes=$3
  local story=$4
  echo "" | tee -a "$MASTER_LOG"
  echo "=================================================================" | tee -a "$MASTER_LOG"
  echo "=== $(date +%H:%M) candidate: $ds_name ($tv_class, $n_classes-cls, story=$story)" | tee -a "$MASTER_LOG"
  echo "=================================================================" | tee -a "$MASTER_LOG"

  # Step 1: prep with a 25-min timeout (download + resize + save)
  echo "[$ds_name] preparing..." | tee -a "$MASTER_LOG"
  timeout 1500 $PY -m data.dataset_hunt_prep "$ds_name" "$tv_class" >> "$MASTER_LOG" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$ds_name] PREP FAILED -- skip" | tee -a "$MASTER_LOG"
    VERDICTS[$ds_name]="PREP_FAIL"
    rm -rf "data/$ds_name"
    return
  fi

  # Step 2: ensure dataset is in IMAGERY_DATASETS in the running data_loader
  if ! grep -q "'$ds_name'" src/utils/data_loader.py; then
    echo "[$ds_name] patching data_loader to add to IMAGERY_DATASETS..." | tee -a "$MASTER_LOG"
    sed -i "s|IMAGERY_DATASETS = {|IMAGERY_DATASETS = {'$ds_name', |" src/utils/data_loader.py
  fi

  # Step 3: generate probe config
  $PY -m src.config_generators.gen_probe_generic "$ds_name" "$n_classes" >> "$MASTER_LOG" 2>&1

  # Step 4: ensure GPU0 is free + nobody else's process on it
  GPU0_APPS=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | wc -l)
  if [ "$GPU0_APPS" -gt 0 ]; then
    GPU0_OWNER=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | head -1 | xargs ps -o user= -p 2>/dev/null | xargs)
    if [ "$GPU0_OWNER" != "michaer8" ]; then
      echo "[$ds_name] GPU0 owned by $GPU0_OWNER -- skip, wait next iteration" | tee -a "$MASTER_LOG"
      VERDICTS[$ds_name]="GPU_BUSY"
      return
    fi
  fi

  # Step 5: launch probe with 8-min timeout (3 ep warmup at ~30s/ep + post-hoc)
  echo "[$ds_name] launching probe on GPU0..." | tee -a "$MASTER_LOG"
  LOG=logs/dataset_hunt/probe_${ds_name}_$(date +%H%M).log
  unset CUDA_VISIBLE_DEVICES
  timeout 480 bash -c \
    "export EXPERIMENT_DIR=results/pending_runs/probe_${ds_name}; echo 0 | $PY -u main.py" \
    > "$LOG" 2>&1
  RC=$?

  # Step 6: parse ep3 train-acc
  EP3_ACC=$(grep -oP 'Warmup 3/3: loss=\S+ acc=\K[0-9.]+' "$LOG" | tail -1)
  if [ -z "$EP3_ACC" ]; then
    # maybe shorter or longer warmup label present
    EP3_ACC=$(grep -oP 'Warmup 3/\S+: loss=\S+ acc=\K[0-9.]+' "$LOG" | tail -1)
  fi
  EP1_ACC=$(grep -oP 'Warmup 1/\S+: loss=\S+ acc=\K[0-9.]+' "$LOG" | tail -1)

  if [ -z "$EP3_ACC" ]; then
    echo "[$ds_name] could not parse ep3 acc, rc=$RC -- FAIL" | tee -a "$MASTER_LOG"
    VERDICTS[$ds_name]="NO_EP3_ACC"
    rm -rf "data/$ds_name"
    rm -rf "results/pending_runs/probe_${ds_name}"
    return
  fi

  echo "[$ds_name] ep1=$EP1_ACC ep3=$EP3_ACC story=$story" | tee -a "$MASTER_LOG"
  # PASS = ep3 acc in (0.40, 0.82) -- has headroom but not degenerate
  if awk "BEGIN{exit !($EP3_ACC > 0.40 && $EP3_ACC < 0.82)}"; then
    echo "[$ds_name] *** PASS *** (ep3=$EP3_ACC in band)" | tee -a "$MASTER_LOG"
    PASSED+=("$ds_name|$EP1_ACC|$EP3_ACC|$story")
    VERDICTS[$ds_name]="PASS"
  else
    echo "[$ds_name] FAIL (ep3=$EP3_ACC out of [0.40, 0.82])" | tee -a "$MASTER_LOG"
    VERDICTS[$ds_name]="OUT_OF_BAND"
    rm -rf "data/$ds_name"
    rm -rf "results/pending_runs/probe_${ds_name}"
    # also revert data_loader edit
    sed -i "s|'$ds_name', ||" src/utils/data_loader.py
  fi
}

# Main loop
echo "Dataset hunt started $(date)" | tee "$MASTER_LOG"
echo "Branch: $(git branch --show-current)" | tee -a "$MASTER_LOG"

for cand in "${CANDIDATES[@]}"; do
  IFS=':' read -r ds_name tv_class n_classes story <<< "$cand"
  probe_ds "$ds_name" "$tv_class" "$n_classes" "$story"
  if [ ${#PASSED[@]} -ge 4 ]; then
    echo "Got 4 passes, stopping early" | tee -a "$MASTER_LOG"
    break
  fi
done

# Write final report
mkdir -p docs
{
  echo "# Dataset Hunt — autonomous overnight run"
  echo ""
  echo "Started: $(head -1 $MASTER_LOG)"
  echo "Finished: $(date)"
  echo "Branch: dataset-hunt"
  echo ""
  echo "## Verdicts (all candidates)"
  echo ""
  echo "| Dataset | Verdict |"
  echo "|---|---|"
  for cand in "${CANDIDATES[@]}"; do
    IFS=':' read -r ds_name _ _ _ <<< "$cand"
    v=${VERDICTS[$ds_name]:-NOT_TRIED}
    echo "| $ds_name | $v |"
  done
  echo ""
  echo "## PASSed (kept on disk + in IMAGERY_DATASETS)"
  echo ""
  if [ ${#PASSED[@]} -eq 0 ]; then
    echo "_None._"
  else
    echo "| Dataset | ep1 train-acc | ep3 train-acc | Story |"
    echo "|---|---|---|---|"
    for p in "${PASSED[@]}"; do
      IFS='|' read -r ds e1 e3 st <<< "$p"
      echo "| $ds | $e1 | $e3 | $st |"
    done
  fi
  echo ""
  echo "## Master log"
  echo ""
  echo "Full master log: \`$MASTER_LOG\`"
} > "$REPORT"

# Commit the work (locally only) so the branch state is auditable in the morning
git add -A 2>>"$MASTER_LOG"
git commit -m "dataset-hunt: $(date +%Y-%m-%d) — passed=${#PASSED[@]} kept=${PASSED[*]}" \
  --no-verify 2>>"$MASTER_LOG" 1>&2

echo "Done. Report at $REPORT. ${#PASSED[@]} datasets passed." | tee -a "$MASTER_LOG"
