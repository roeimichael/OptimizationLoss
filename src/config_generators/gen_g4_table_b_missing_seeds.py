"""G4 — Backfill 3 missing post-hoc seeds in Table B asymmetric.

`paper/tables/B_asymmetric_tightness/table_B_phase2_asymmetric_derm.csv` shows n_seeds=1 for:
    L20_G50 / heuristic    L20_G50 / danits_lp
    L50_G20 / heuristic    L50_G20 / danits_lp

These post-hoc methods are deterministic given the same warmup, so the
1-seed row is correct as a point estimate — but the table reports ±std
across 4 seeds for everything else. Re-dispatch seeds 2, 3, 4 to make
the row n consistent with the rest of Table B.

12 cells, ~2 minutes on Blackwell.
"""
from pathlib import Path
import glob, json, os

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/g4_table_b_backfill"

DS_NAME = "dermmnist"
MODEL = "MobileNetV3"
DS_META = {
    "data_dir": "data/dermmnist/slice_1", "num_classes": 7,
    "image_size": 224, "target_column": "label",
    "group_column": "loc_group", "constrained_class": 4,
}
TIGHT_PAIRS = [("L20_G50", (0.20, 0.50)), ("L50_G20", (0.50, 0.20))]
METHODS = ["heuristic", "danits_lp"]
SEEDS = [2, 3, 4]   # seed 1 already exists per existing table_B row

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}


def make_cfg(tight_tag, pair, method, seed):
    hp = {**SHARED_HP, "seed": seed}
    ds_config = dict(DS_META)
    bmid = compute_base_model_id(MODEL, hp, dataset_mode=DS_NAME,
                                 data_dir=DS_META["data_dir"], dataset_config=ds_config)
    return {
        "methodology": method, "model_name": MODEL,
        "constraint": list(pair), "constraint_tag": tight_tag,
        "dataset_mode": DS_NAME, "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "exp_name": f"g4_{method}_{DS_NAME}_{tight_tag}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / tight_tag / method / f"seed_{seed}"
        ),
    }


def build():
    cfgs = [make_cfg(t, p, m, s)
            for (t, p) in TIGHT_PAIRS for m in METHODS for s in SEEDS]
    print(f"G4 backfill: {len(cfgs)} cells.")
    save_configs(cfgs, output_dir=SWEEP_ROOT)


if __name__ == "__main__":
    build()
