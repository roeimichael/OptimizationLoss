"""ViT-on-AIDER hyperparameter probe.

Goal: find a ViT (model, HP) combination whose warmup train-acc on AIDER lands
in the headroom band [0.70, 0.82] at some epoch, so TraLO has room to bite.
Prior failure (docs/REJECTED.md): vit_tiny_patch16_224 default HPs saturated
at ep1=0.83. This script sweeps 5 alternative configs.

Configs:
  A. ViTTiny  + lr=1e-4 + dropout=0.5 + full FT     (dropout intervention)
  B. ViTTiny  + lr=5e-5 + dropout=0.3 + full FT     (LR intervention)
  C. ViTTiny  + lr=5e-5 + dropout=0.5 + full FT     (both)
  D. ViTTiny  + lr=1e-4 + dropout=0.3 + LINEAR probe (frozen backbone)
  E. ViTSmall32 + lr=1e-4 + dropout=0.3 + full FT   (different ViT)

Each config: warmup_epochs=8, constraint_epochs=1, seed=1, AIDER only.
Pass = any ep in [0.70, 0.82] train-acc.

Configs land at results/pending_runs/vit_probe/{config_id}/...
"""
from pathlib import Path
import glob, json, os

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/vit_probe"

DS_NAME = "aider"
DS_META = {
    "data_dir": "data/aider/slice_1", "num_classes": 4,
    "image_size": 224, "target_column": "label",
    "group_column": "synth_group", "constrained_class": 0,
}

TIGHT = "L50_G50"

CONFIGS = [
    # (config_id, model_name, lr, dropout, extra_kwargs_for_model)
    ("A_vitT_drop05",     "ViTTiny",    1e-4, 0.5, {}),
    ("B_vitT_lr5e5",      "ViTTiny",    5e-5, 0.3, {}),
    ("C_vitT_lr5e5_drop05", "ViTTiny",  5e-5, 0.5, {}),
    ("D_vitT_linprobe",   "ViTTiny",    1e-4, 0.3, {"linear_probe": True}),
    ("E_vitS32",          "ViTSmall32", 1e-4, 0.3, {}),
]

BASE_HP = {
    "lr_constraint": 5e-6, "batch_size": 64, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "warmup_epochs": 8, "constraint_epochs": 1,
}

TRALO_HP = {
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
    "penalty_mode": "both", "enable_ce_skip": True,
    "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
    "reset_optimizer_at_sat": True,
}


def make_cfg(config_id, model_name, lr, dropout, model_kwargs):
    hp = {**BASE_HP, **TRALO_HP, "lr": lr, "dropout": dropout, "seed": 1,
          **model_kwargs}
    ds_config = dict(DS_META)
    parts = TIGHT.split("_")
    pair = (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)
    bmid = compute_base_model_id(
        model_name, hp, dataset_mode=DS_NAME,
        data_dir=DS_META["data_dir"], dataset_config=ds_config,
    )
    return {
        "methodology": "tralo",
        "model_name": model_name,
        "constraint": list(pair),
        "constraint_tag": TIGHT,
        "dataset_mode": DS_NAME,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"vitp_{config_id}",
        "experiment_path": str(Path(SWEEP_ROOT) / config_id),
    }


def build():
    cfgs = [make_cfg(*c) for c in CONFIGS]
    print(f"Queueing {len(cfgs)} ViT probes on AIDER.")
    save_configs(cfgs, output_dir=SWEEP_ROOT)


if __name__ == "__main__":
    build()
