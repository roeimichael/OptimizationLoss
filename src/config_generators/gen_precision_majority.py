"""Precision sweep on TraLO's majority-class wins.

Two strongest TraLO wins identified in rotation grids:
- AIDER class 3 (normal, 74% majority): d_F1 = +0.050 vs LP+heuristic at L50_G50
- Derm class 5 (NV, 67% majority): d_F1 = +0.016 vs LP+heuristic at L30_G30

This sweep gives paper-grade stats: 5 seeds, 3 tightness levels, all 5 methods.

Layout:
  AIDER  cls 3 × 5 methods × 5 seeds × {L30, L50, L70} = 75 cells
  Derm   cls 5 × 5 methods × 5 seeds × {L30, L50, L70} = 75 cells

Total: 150 cells. Estimated 3-4 hours parallel on dsisco02 Blackwell.
"""
from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

TIGHTNESS = ["L30_G30", "L50_G50", "L70_G70"]
SEEDS = [1, 2, 3, 4, 5]
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]

DATASETS = [
    {
        "dataset_mode": "aider", "data_dir": "data/aider/slice_1",
        "num_classes": 4, "group_column": "synth_group",
        "constrained_class": 3, "role_tag": "majority_normal_74pct",
    },
    {
        "dataset_mode": "dermmnist", "data_dir": "data/dermmnist/slice_1",
        "num_classes": 7, "group_column": "loc_group",
        "constrained_class": 5, "role_tag": "NV_majority_67pct",
    },
]

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


def make_cfg(ds, tight, seed, method):
    ds_config = {
        "num_classes": ds["num_classes"], "image_size": 224, "target_column": "label",
        "group_column": ds["group_column"], "constrained_class": ds["constrained_class"],
        "data_dir": ds["data_dir"],
    }
    hp = {**SHARED_HP, "seed": seed}
    if method == "tralo":
        hp.update(TRALO_HP)
    pair = _pair(tight)
    bmid = compute_base_model_id(
        "MobileNetV3", hp, dataset_mode=ds["dataset_mode"], data_dir=ds["data_dir"],
        dataset_config=ds_config,
    )
    cfg_name = f"constrained{ds['constrained_class']}_{ds['role_tag']}_{tight}"
    return {
        "methodology": method, "model_name": "MobileNetV3",
        "constraint": list(pair), "constraint_tag": tight,
        "dataset_mode": ds["dataset_mode"], "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "experiment_path": (
            f"results/pending_runs/precision_majority/"
            f"{ds['dataset_mode']}/{cfg_name}/{method}/seed_{seed}"
        ),
    }


def main():
    cfgs = [
        make_cfg(ds, t, s, m)
        for ds in DATASETS for t in TIGHTNESS for s in SEEDS for m in METHODS
    ]
    print(f"Generated {len(cfgs)} configs "
          f"({len(DATASETS)} datasets x {len(TIGHTNESS)} tightness x "
          f"{len(SEEDS)} seeds x {len(METHODS)} methods)")
    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
