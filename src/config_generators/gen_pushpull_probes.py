"""Push-pull probe sweep: warmup_epochs=1 to force CE-active phase 2.

Targets datasets/backbones that AUDIT showed are close to push-pull but where
we don't yet have all 5 methods at warmup=1.

Configs match the existing winning HP recipe (lr, batch_size, etc.) and
differ ONLY in warmup_epochs=1 (the key push-pull lever from HP correlation).
"""
import os
from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SEEDS = [1, 2, 3]
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]

SHARED_HP = {
    "lr": 1e-4,
    "lr_constraint": 5e-6,
    "dropout": 0.3,
    "batch_size": 64,
    "warmup_epochs": 1,            # PUSH-PULL LEVER
    "constraint_epochs": 100,
    "pretrained": True,
    "class_weighted_ce": False,
    "constraint_chunk_size": 256,
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


def make_cfg(sweep_root, dataset_mode, ds_config, backbone, tight, seed,
             method):
    hp = {**SHARED_HP, "seed": seed}
    if method == "tralo":
        hp.update(TRALO_HP)
    lo, hi = (int(tight.split("_")[0][1:]) / 100,
              int(tight.split("_")[1][1:]) / 100)
    bmid = compute_base_model_id(
        backbone, hp, dataset_mode=dataset_mode,
        data_dir=ds_config["data_dir"],
        dataset_config=ds_config,
    )
    return {
        "methodology": method,
        "model_name": backbone,
        "constraint": [lo, hi],
        "constraint_tag": tight,
        "dataset_mode": dataset_mode,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "experiment_path": (
            f"results/pending_runs/{sweep_root}/{backbone}/{dataset_mode}/"
            f"{tight}/{method}/seed_{seed}"
        ),
    }


def main():
    cfgs = []

    # OCT MNIST: never tested with warmup=1. Class 2 = CNV (~37%, largest).
    oct_cfg = {
        "num_classes": 4, "image_size": 224, "target_column": "label",
        "group_column": "synth_group", "constrained_class": 2,
        "data_dir": "data/octmnist/slice_1",
    }
    for backbone in ["MobileNetV2", "RegNetY400MF"]:
        for tight in ["L50_G50"]:
            for seed in SEEDS:
                for method in METHODS:
                    cfgs.append(make_cfg(
                        "pushpull_octmnist_w1", "octmnist", oct_cfg,
                        backbone, tight, seed, method,
                    ))

    # DermMNIST RegNetY400MF L50 cls 4 — we have warmup=1 single seeds in
    # blackwell_validation but want 3 more seeds for stat power. cls 4 = MEL.
    derm_cfg = {
        "num_classes": 7, "image_size": 224, "target_column": "label",
        "group_column": "loc_group", "constrained_class": 4,
        "data_dir": "data/dermmnist/slice_1",
    }
    for backbone in ["RegNetY400MF", "ShuffleNetV2"]:
        for tight in ["L50_G50"]:
            for seed in SEEDS:
                for method in METHODS:
                    cfgs.append(make_cfg(
                        "pushpull_derm_w1", "dermmnist", derm_cfg,
                        backbone, tight, seed, method,
                    ))

    print(f"Generated {len(cfgs)} probe configs")
    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
