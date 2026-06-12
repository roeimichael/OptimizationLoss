"""DermMNIST backbone-weakening axis for headroom-hypothesis validation.

Companion to gen_derm_cripple.py. Where gen_derm_cripple varies the data
(corrupt train+test), this varies the model:
  weak1: ShuffleNetV2 (1.4M params, pretrained) - smaller capacity
  weak2: MobileNetV3 with pretrained=False (cold-start ImageNet weights off)

Both make warmup CE less likely to saturate, so the in-training-vs-
post-hoc ranking should flip in the same direction as the cripple
experiment when the warmup train-acc sits in the [0.70, 0.85] band.

Grid:
  2 weak backbones x 3 tightness (L20, L30, L50) x 4 methods x 2 seeds = 48 cells.

Output: results/pending_runs/derm_backbone_weak/
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/derm_backbone_weak"

VARIANTS = {
    "shuffle":   {"model": "ShuffleNetV2",  "pretrained": True},
    "mnv3_cold": {"model": "MobileNetV3",   "pretrained": False},
}

TIGHTNESS = ["L20_G20", "L30_G30", "L50_G50"]
SEEDS = [1, 2]
CLS = 4
METHODS = ["tralo", "fioretto_ldf", "danits_lp", "heuristic"]

DATASET_CONFIG = {
    "data_dir": "data/dermmnist/slice_1",
    "num_classes": 7, "image_size": 224, "target_column": "label",
    "group_column": "loc_group", "constrained_class": CLS,
}

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
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


def _tight_pair(tag):
    parts = tag.split("_")
    return (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)


def make_cfg(variant_name, variant, method, tight, seed):
    hp = {**SHARED_HP, "pretrained": variant["pretrained"], "seed": seed}
    if method == "tralo":
        hp.update(TRALO_HP)
    pair = _tight_pair(tight)
    bmid = compute_base_model_id(
        variant["model"], hp, dataset_mode="dermmnist",
        data_dir=DATASET_CONFIG["data_dir"], dataset_config=DATASET_CONFIG,
    )
    return {
        "methodology": method,
        "model_name": variant["model"],
        "constraint": list(pair),
        "constraint_tag": tight,
        "dataset_mode": "dermmnist",
        "dataset_config": DATASET_CONFIG,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"derm_bb_{variant_name}_{tight}_{method}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / variant_name / tight / method / f"seed_{seed}"),
    }


def build():
    cfgs = []
    for variant_name, variant in VARIANTS.items():
        for tight in TIGHTNESS:
            for method in METHODS:
                for seed in SEEDS:
                    cfgs.append(make_cfg(variant_name, variant, method, tight, seed))
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"\nGenerated {len(cfgs)} derm-backbone-weak configs -> {SWEEP_ROOT}")


if __name__ == "__main__":
    build()
