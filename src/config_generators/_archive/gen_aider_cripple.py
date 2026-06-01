"""AIDER cripple-training experiment.

Tests the hypothesis: TraLO's F1 edge appears when training CE doesn't
saturate. AIDER normally lets MobileNetV3 hit ~0.9998 train acc by epoch 3,
killing TraLO's edge. Three cripple conditions push warmup train acc
back into the [0.70, 0.85] sweet spot.

Conditions:
  C1_noisy15:   Gaussian noise sigma=0.15 on train+test (mild)
  C2_no_pre:    pretrained=False (cold-start backbone)
  C3_noisy30:   Gaussian noise sigma=0.30 on train+test (heavy)

Grid: 3 conditions x 4 methods x 2 seeds x 1 tightness = 24 cells.
Output: results/pending_runs/aider_cripple/
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/aider_cripple"

CONDITIONS = {
    "C1_noisy15": {"data_dir": "data/aider_noisy15/slice_1", "pretrained": True},
    "C2_no_pre":  {"data_dir": "data/aider/slice_1",         "pretrained": False},
    "C3_noisy30": {"data_dir": "data/aider_noisy30/slice_1", "pretrained": True},
}

TIGHTNESS = "L30_G30"
SEEDS = [1, 2]
MODEL = "MobileNetV3"
CLS = 0  # collapsed_building, matches AIDER headline
METHODS = ["tralo", "fioretto_ldf", "danits_lp", "heuristic"]

DATASET_BASE = {
    "num_classes": 4, "image_size": 224, "target_column": "label",
    "group_column": "synth_group", "constrained_class": CLS,
}

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "fioretto_step_size": 0.01,
}

# Mirror gen_g5_component_ablation's full-TraLO hyperparams so the "tralo"
# row in each condition matches the headline config (only the cripple
# axis changes).
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


def make_cfg(cond_name, cond_overrides, method, seed):
    data_dir = cond_overrides["data_dir"]
    pretrained = cond_overrides["pretrained"]
    ds_config = {**DATASET_BASE, "data_dir": data_dir}
    hp = {**SHARED_HP, "pretrained": pretrained, "seed": seed}
    if method == "tralo":
        hp.update(TRALO_HP)
    pair = _tight_pair(TIGHTNESS)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode="aider",
        data_dir=data_dir, dataset_config=ds_config,
    )
    return {
        "methodology": method,
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": TIGHTNESS,
        "dataset_mode": "aider",
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"cripple_{cond_name}_{method}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / cond_name / method / f"seed_{seed}"),
    }


def build():
    cfgs = []
    for cond_name, cond in CONDITIONS.items():
        for method in METHODS:
            for seed in SEEDS:
                cfgs.append(make_cfg(cond_name, cond, method, seed))
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"\nGenerated {len(cfgs)} aider-cripple configs -> {SWEEP_ROOT}")


if __name__ == "__main__":
    build()
