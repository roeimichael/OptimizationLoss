#!/bin/bash
# Dataset hunt v2: bug-fixed (no git stash that nukes data_loader fix).
# PASS = ep3 train-acc < 0.82 (HARD dataset, no saturation). No lower bound.
# Iterates candidates, deletes FAILed ones, keeps PASSing ones.
set +e
cd ~/OptimizationLoss
mkdir -p logs/dataset_hunt
MASTER_LOG=logs/dataset_hunt/master_v2_$(date +%Y%m%d_%H%M).log
REPORT=docs/DATASET_HUNT_RESULTS.md
PY=~/anaconda3/envs/optloss/bin/python

# Candidate list: "<ds_name>:<tv_class>:<n_classes>:<story>"
# Only torchvision classes with known-working signatures.
CANDIDATES=(
  "flowers102:Flowers102:102:fine_grained_flowers"
  "dtd:DTD:47:texture_classification"
  "tiny_imagenet:TinyImageNet:200:hard_general_benchmark"
  "food101:Food101:101:food_recall_scenario"
  "oxford_pet:OxfordIIITPet:37:pet_id"
  "fgvc_aircraft:FGVCAircraft:100:aviation_safety_recall"
)

PASSED=()
declare -A VERDICTS

probe_ds() {
  local ds_name=$1
  local tv_class=$2
  local n_classes=$3
  local story=$4
  echo "" | tee -a "$MASTER_LOG"
  echo "=== $(date +%H:%M) $ds_name ($tv_class, $n_classes-cls, $story)" | tee -a "$MASTER_LOG"

  # Step 1: prep with 25-min timeout
  timeout 1500 $PY -m data.dataset_hunt_prep "$ds_name" "$tv_class" >> "$MASTER_LOG" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "[$ds_name] PREP_FAIL rc=$RC" | tee -a "$MASTER_LOG"
    VERDICTS[$ds_name]="PREP_FAIL"
    rm -rf "data/$ds_name"
    return
  fi

  # Step 2: add to IMAGERY_DATASETS if not already there
  if ! grep -q "'$ds_name'" src/utils/data_loader.py; then
    sed -i "s|IMAGERY_DATASETS = {|IMAGERY_DATASETS = {'$ds_name', |" src/utils/data_loader.py
  fi

  # Step 3: gen probe config
  $PY -m src.config_generators.gen_probe_generic "$ds_name" "$n_classes" >> "$MASTER_LOG" 2>&1

  # Step 4: GPU0 free check (don't share)
  GPU0_APPS=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$GPU0_APPS" -gt 0 ]; then
    OWNER=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | head -1 | xargs ps -o user= -p 2>/dev/null | xargs)
    if [ "$OWNER" != "michaer8" ]; then
      echo "[$ds_name] GPU0 owned by $OWNER, skip" | tee -a "$MASTER_LOG"
      VERDICTS[$ds_name]="GPU_BUSY"
      return
    fi
  fi

  # Step 5: probe (8-min budget for 3 ep warmup + post-hoc)
  LOG=logs/dataset_hunt/probe_v2_${ds_name}_$(date +%H%M).log
  unset CUDA_VISIBLE_DEVICES
  timeout 480 bash -c "export EXPERIMENT_DIR=results/pending_runs/probe_${ds_name}; echo 0 | $PY -u main.py" \
    > "$LOG" 2>&1
  RC=$?

  EP1_ACC=$(grep -oP 'Warmup 1/\S+: loss=\S+ acc=\K[0-9.]+' "$LOG" | head -1)
  EP3_ACC=$(grep -oP 'Warmup 3/\S+: loss=\S+ acc=\K[0-9.]+' "$LOG" | tail -1)

  if [ -z "$EP3_ACC" ]; then
    ERR=$(grep -oE "Error|Traceback|TypeError|ValueError|UFunc" "$LOG" | head -1)
    echo "[$ds_name] no ep3 acc (rc=$RC, err=$ERR), FAIL" | tee -a "$MASTER_LOG"
    VERDICTS[$ds_name]="NO_EP3 err=$ERR"
    rm -rf "data/$ds_name"
    rm -rf "results/pending_runs/probe_${ds_name}"
    sed -i "s|'$ds_name', ||" src/utils/data_loader.py
    return
  fi

  echo "[$ds_name] ep1=$EP1_ACC ep3=$EP3_ACC" | tee -a "$MASTER_LOG"
  # PASS = ep3 < 0.82 (no lower bound -- we want HARD)
  if awk "BEGIN{exit !($EP3_ACC < 0.82)}"; then
    echo "[$ds_name] *** PASS *** ep3=$EP3_ACC < 0.82" | tee -a "$MASTER_LOG"
    PASSED+=("$ds_name|$EP1_ACC|$EP3_ACC|$story")
    VERDICTS[$ds_name]="PASS ep3=$EP3_ACC"
  else
    echo "[$ds_name] FAIL saturated (ep3=$EP3_ACC)" | tee -a "$MASTER_LOG"
    VERDICTS[$ds_name]="SATURATED ep3=$EP3_ACC"
    rm -rf "data/$ds_name"
    rm -rf "results/pending_runs/probe_${ds_name}"
    sed -i "s|'$ds_name', ||" src/utils/data_loader.py
  fi
}

# Main
echo "Hunt v2 started $(date)" | tee "$MASTER_LOG"
echo "Branch: $(git branch --show-current)" | tee -a "$MASTER_LOG"

for cand in "${CANDIDATES[@]}"; do
  IFS=':' read -r ds_name tv_class n_classes story <<< "$cand"
  probe_ds "$ds_name" "$tv_class" "$n_classes" "$story"
  if [ ${#PASSED[@]} -ge 4 ]; then
    echo "Got 4 passes, stopping" | tee -a "$MASTER_LOG"
    break
  fi
done

# Final report
{
  echo "# Dataset Hunt v2"
  echo ""
  echo "Started: $(head -1 $MASTER_LOG)"
  echo "Finished: $(date)"
  echo "Branch: dataset-hunt"
  echo ""
  echo "## Verdicts"
  echo ""
  echo "| Dataset | Verdict |"
  echo "|---|---|"
  for cand in "${CANDIDATES[@]}"; do
    IFS=':' read -r ds_name _ _ _ <<< "$cand"
    v="${VERDICTS[$ds_name]:-NOT_TRIED}"
    echo "| $ds_name | $v |"
  done
  echo ""
  echo "## PASSED (HARD datasets kept)"
  echo ""
  if [ ${#PASSED[@]} -eq 0 ]; then
    echo "_None found in this run._"
  else
    echo "| Dataset | ep1 acc | ep3 acc | Story |"
    echo "|---|---|---|---|"
    for p in "${PASSED[@]}"; do
      IFS='|' read -r ds e1 e3 st <<< "$p"
      echo "| $ds | $e1 | $e3 | $st |"
    done
  fi
} > "$REPORT"

echo "Done. ${#PASSED[@]} passed. Report: $REPORT" | tee -a "$MASTER_LOG"
