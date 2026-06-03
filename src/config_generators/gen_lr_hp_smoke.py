"""LR + HP ablation under contamination.

Goal: can HP tuning recover TraLO advantage when warmup is saturated, or
amplify advantage when in the sweet spot?

Two sub-experiments:

A) LR sweep under contamination:
   3 datasets x sigma=0.20 x L30_G30 x 3 LRs x 5 methods x 2 seeds = 90 cells

B) TraLO HP smoke on derm noise20:
   1 dataset (derm) x sigma=0.20 x L30_G30 x 6 HP variants (tralo only) x 2 seeds = 12 cells

Output: results/pending_runs/lr_hp_smoke/
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/lr_hp_smoke"

DATASETS = {
    "tissuemnist": {"num_classes": 8, "image_size": 224, "target_column": "label",
                    "group_column": "synth_group", "constrained_class": 4},
    "dermmnist":   {"num_classes": 7, "image_size": 224, "target_column": "label",
                    "group_column": "loc_group", "constrained_class": 4},
    "aider":       {"num_classes": 4, "image_size": 224, "target_column": "label",
                    "group_column": "synth_group", "constrained_class": 0},
}

TIGHT = "L30_G30"
SIGMA_INT = 20
SEEDS = [1, 2]
MODEL = "MobileNetV3"
METHODS_LR = ["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]
LRS = [5e-5, 1e-4, 5e-4]                        # 3 LRs

BASE_HP = {
    "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300, "pretrained": True,
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

# B) TraLO HP variants on derm_sigma20 (find a better-than-default config)
HP_VARIANTS = {
    "baseline":         {},
    "rho_target200":    {"rho_target": 200.0},
    "lambda_step005":   {"lambda_step": 0.005},
    "alpha_kl_005":     {"alpha_kl": 0.005},
    "warmup_30":        {"warmup_epochs": 30},
    "warmup_70":        {"warmup_epochs": 70},
}


def _tight_pair(tag):
    parts = tag.split("_")
    return (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)


def make_lr_cfg(dataset, ds_base, lr, method, seed):
    data_dir = f"data/{dataset}_sigma{SIGMA_INT:02d}/slice_1"
    ds_config = {**ds_base, "data_dir": data_dir}
    hp = {**BASE_HP, "lr": lr, "seed": seed}
    if method == "tralo":
        hp.update(TRALO_HP)
    pair = _tight_pair(TIGHT)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode=dataset,
        data_dir=data_dir, dataset_config=ds_config,
    )
    lr_tag = f"lr{int(lr*1e6):04d}"
    return {
        "methodology": method,
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": TIGHT,
        "dataset_mode": dataset,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"lr_{dataset}_s{SIGMA_INT:02d}_{lr_tag}_{method}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / "lr_sweep" / dataset / lr_tag / method / f"seed_{seed}"),
    }


def make_hp_cfg(variant_name, overrides, seed):
    data_dir = f"data/dermmnist_sigma{SIGMA_INT:02d}/slice_1"
    ds_config = {**DATASETS["dermmnist"], "data_dir": data_dir}
    hp = {**BASE_HP, "lr": 1e-4, "seed": seed}
    hp.update(TRALO_HP)
    hp.update(overrides)
    pair = _tight_pair(TIGHT)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode="dermmnist",
        data_dir=data_dir, dataset_config=ds_config,
    )
    return {
        "methodology": "tralo",
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": TIGHT,
        "dataset_mode": "dermmnist",
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"hp_{variant_name}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / "hp_smoke" / variant_name / f"seed_{seed}"),
    }


def build():
    cfgs = []
    for dataset, ds_base in DATASETS.items():
        for lr in LRS:
            for method in METHODS_LR:
                for seed in SEEDS:
                    cfgs.append(make_lr_cfg(dataset, ds_base, lr, method, seed))
    for variant_name, overrides in HP_VARIANTS.items():
        for seed in SEEDS:
            cfgs.append(make_hp_cfg(variant_name, overrides, seed))
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"\nGenerated {len(cfgs)} LR/HP smoke configs -> {SWEEP_ROOT}")


if __name__ == "__main__":
    build()
