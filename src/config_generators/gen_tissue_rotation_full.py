"""TissueMNIST class-rotation test for TraLO advantage mechanism.

Phase 3 of mechanism validation. Rotates constrained class across 4 tissue
classes spanning the difficulty range:
- 4 (GE, 7.1%): paper-baseline anchor, hard task → TraLO wins big in paper
- 0 (32.1%): majority, easy task (warmup likely nails it)
- 6 (23.7%): second-majority, moderately easy
- 2 (3.5%): rarest class, hardest

4 classes x 5 methods x 3 seeds x L30_G30 = 60 cells.
"""
from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

DATASET = "tissuemnist"
DATA_DIR = "data/tissuemnist/slice_1"
NUM_CLASSES = 8
GROUP_COLUMN = "synth_group"

CLASSES = [
    (4, "GE_paper_anchor"),
    (0, "cls0_majority"),
    (6, "cls6_second_majority"),
    (2, "cls2_rarest"),
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
            f"results/pending_runs/tissue_rotation_full/"
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
