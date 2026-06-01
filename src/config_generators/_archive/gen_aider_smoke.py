"""AIDER smoke: 1 config to verify dataset integration end-to-end.

MobileNetV3 + tralo (breakthrough) + L30_G30 + seed 1.
Constrained class: 0 = collapsed_building.

Output: results/pending_runs/aider_smoke
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/aider_smoke"

DS_META = {
    "data_dir": "data/aider/slice_1", "num_classes": 4,
    "image_size": 224, "target_column": "label",
    "group_column": "synth_group",
}
MODEL = "MobileNetV3"
CLS = 0  # collapsed_building
SEED = 1

HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
    "penalty_mode": "both", "enable_ce_skip": True,
    "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
    "reset_optimizer_at_sat": True,
    "seed": SEED,
}


def build():
    ds_config = {**DS_META, "constrained_class": CLS}
    bmid = compute_base_model_id(
        MODEL, HP, dataset_mode="aider",
        data_dir=DS_META["data_dir"], dataset_config=ds_config,
    )
    cfg = {
        "methodology": "tralo",
        "model_name": MODEL,
        "constraint": [0.3, 0.3],
        "constraint_tag": "L30_G30",
        "dataset_mode": "aider",
        "dataset_config": ds_config,
        "hyperparams": HP,
        "base_model_id": bmid,
        "exp_name": f"smoke_aider_tralo_L30_G30_seed{SEED}",
        "experiment_path": str(Path(SWEEP_ROOT) / f"seed_{SEED}"),
    }
    save_configs([cfg], output_dir=SWEEP_ROOT)
    print(f"\nGenerated 1 AIDER smoke config -> {SWEEP_ROOT}")


if __name__ == "__main__":
    build()
