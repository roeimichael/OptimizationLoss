#!/usr/bin/env bash
# Conditional, bounded 72 GPU-hour sequence on one physical dsisco02 card.
# Each cell performs its own source/data/GPU/pilot/cost gate and stops on error.
# Usage: bash tools/tabular_weekend_chain.sh IMMUTABLE_RELEASE GPU_INDEX
set -u
[[ $# -eq 2 && $1 =~ ^[0-9a-f]{40}$ && $2 =~ ^[0-9]+$ ]] || exit 2
SHA=$1 GPU=$2
REL=/home/dsi/michaer8/tralo-rebuild/releases/$SHA
RUNS=/tmp/tralo-weekend-michaer8-20261001/runs
[[ -d $REL && -d $RUNS ]] || exit 2
cd "$REL" || exit 2

run_cell() {
  local DATASET=$1 BACKBONE=$2 TAG=$3 ROOT
  ROOT=$RUNS/tabular_${TAG}_6800_6804
  [[ ! -e $ROOT && ! -L $ROOT ]] || {
    echo "Existing cell root $ROOT; inspect, never duplicate" >&2
    return 2
  }
  echo "$(date -u +%FT%TZ) CELL_START $DATASET $BACKBONE $ROOT"
  bash tools/tabular_persistent_queue.sh "$SHA" "$ROOT" "$GPU" "$DATASET" "$BACKBONE"
  local RC=$?
  echo "$(date -u +%FT%TZ) CELL_END $DATASET $BACKBONE EXIT=$RC"
  return "$RC"
}

# Three prospective cells: two datasets and two different modern architectures.
# The independent fMoW queue supplies the third image modality separately.
run_cell isic2020 mobilenet_v3_large isic2020_mnv3 || exit $?
run_cell celeba mobilenet_v3_large celeba_mnv3 || exit $?
run_cell isic2020 vit_b_16 isic2020_vit || exit $?
echo "$(date -u +%FT%TZ) WEEKEND_CHAIN_COMPLETE"
