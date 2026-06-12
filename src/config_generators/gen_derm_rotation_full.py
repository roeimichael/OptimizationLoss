"""DermMNIST class-rotation test for TraLO advantage mechanism.

Phase 2 of mechanism validation (Phase 1 = AIDER rotation).
Rotates constrained class across 4 derm classes spanning the difficulty range:
- MEL (4): paper-baseline anchor, ~11%, medium-hard
- NV (5): majority ~67%, easy task (warmup nails it)
- BCC (1): ~5%, medium
- DF (3): ~1%, very hard rare class

4 classes x 5 methods x 3 seeds x L30_G30 = 60 cells.

Local group constraint via loc_group. Same paper-baseline HP as headline.
"""
from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

DATASET = "dermmnist"
DATA_DIR = "data/dermmnist/slice_1"
NUM_CLASSES = 7
GROUP_COLUMN = "loc_group"

CLASSES = [
    (4, "MEL_paper_anchor"),
    (5, "NV_majority"),
    (1, "BCC_medium"),
    (3, "DF_rare"),
]
TIGHT = "L30_G30"
SEEDS = [1, 2, 3]
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
    "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "fioretto_step_size": 0.01,
}
TRALO_HP = {
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
    "penalty_mode": "both", "enable_ce_skip": True,
    "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
    "reset_optimizer_at_sat": True,
    "disable_freeze_on_satisfy": False,
}


def _pair(tag):
    p = tag.split("_")
    return (int(p[0][1:]) / 100, int(p[1][1:]) / 100)


def make_cfg(cls_idx, role, seed, method):
    ds_config = {
        "num_classes": NUM_CLASSES, "image_size": 224, "target_column": "label",
        "group_column": GROUP_COLUMN, "constrained_class": cls_idx,
        "data_dir": DATA_DIR,
    }
    hp = {**SHARED_HP, "seed": seed}
    if method == "tralo":
        hp.update(TRALO_HP)
    pair = _pair(TIGHT)
    bmid = compute_base_model_id(
        "MobileNetV3", hp, dataset_mode=DATASET, data_dir=DATA_DIR,
        dataset_config=ds_config,
    )
    cfg_name = f"constrained{cls_idx}_{role}_{TIGHT}"
    return {
        "methodology": method, "model_name": "MobileNetV3",
        "constraint": list(pair), "constraint_tag": TIGHT,
        "dataset_mode": DATASET, "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "experiment_path": (
            f"results/pending_runs/derm_rotation_full/"
            f"MobileNetV3/{cfg_name}/{method}/seed_{seed}"
        ),
    }


def main():
    cfgs = [
        make_cfg(c, r, s, m)
        for (c, r) in CLASSES for s in SEEDS for m in METHODS
    ]
    print(f"Generated {len(cfgs)} configs "
          f"({len(CLASSES)} classes x {len(SEEDS)} seeds x "
          f"{len(METHODS)} methods)")
    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
