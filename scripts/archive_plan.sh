#!/usr/bin/env bash
# Archive non-paper sweep directories from results/pending_runs to archive_experiments.
# Each sweep moved with mv (preserves contents, just changes parent dir).
# Cells are NOT deleted — just out of the active experiment tree.

set -euo pipefail
cd ~/OptimizationLoss

ARCHIVE_DIR="archive_experiments"
SRC_DIR="results/pending_runs"
mkdir -p "$ARCHIVE_DIR"

# Sweeps to archive: smoke tests, probes, rejected backbones, non-active datasets,
# cripple/perturbation experiments not referenced in paper, old class-rotation work,
# superseded push-pull cls-exploration sweeps, model search exploration.
SWEEPS=(
  # Smoke tests
  aider_smoke backbone_smoke dtd_smoke eurosat_smoke flowers102_smoke
  octmnist_smoke expansion_smoke derm_smallcnn_smoke
  # Probes
  probe_dtd probe_flowers102 imagewoof_probe gtsrb_probe new_dataset_probes
  _diag_smallcnn
  # Non-active datasets
  octmnist_backbones octmnist_classrotation octmnist_expansion
  pushpull_octmnist_w1 flowers102_tight turing_new_datasets
  # Rejected backbones
  derm_smallcnn_expand derm_smallcnn_full blackwell_new_backbones
  # Cripple/perturbation not in paper
  aider_cripple derm_cripple derm_backbone_weak
  # Class rotation (exploratory, abandoned)
  aider_rotation_full aider_rotation_L30 derm_rotation_full
  tissue_rotation_full class_rotation
  # Cls exploration during the abandoned "majority-class wins" theory
  aider_cls3_backbones aider_cls3_l20_backbones aider_cls3_tight
  derm_cls5_backbones derm_cls5_l20_backbones derm_cls5_tight
  tissue_cls0_tight precision_majority
  # Old/auxiliary
  arch_validation aider_seed_ext expansion_aider_baselines expansion_baselines
  expansion_dermmnist_baselines model_search lr_hp_smoke warmup_confirm
  warmup_probe g4_table_b_backfill
)

moved=0
skipped=0
for sweep in "${SWEEPS[@]}"; do
  if [[ -d "$SRC_DIR/$sweep" ]]; then
    if [[ -e "$ARCHIVE_DIR/$sweep" ]]; then
      echo "SKIP (already in archive): $sweep"
      skipped=$((skipped+1))
    else
      mv "$SRC_DIR/$sweep" "$ARCHIVE_DIR/$sweep"
      echo "MOVED: $sweep"
      moved=$((moved+1))
    fi
  else
    echo "NOT FOUND: $sweep"
    skipped=$((skipped+1))
  fi
done

echo
echo "==========================="
echo "moved:   $moved"
echo "skipped: $skipped"
echo "==========================="
echo
echo "Remaining sweep dirs in $SRC_DIR:"
ls -1 "$SRC_DIR/" | head -30
