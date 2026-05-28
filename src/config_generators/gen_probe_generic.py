"""Generic 3-epoch warmup probe for any imagery dataset.

Usage:
  python -m src.config_generators.gen_probe_generic <ds_name> <n_classes>

Single tralo config, warmup_epochs=3, constraint_epochs=1, seed=1, L50_G50,
constrained_class=0 (placeholder — we just want the warmup curve).
"""
import sys
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

PER_METHOD_TRALO = {
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
    "penalty_mode": "both", "enable_ce_skip": True,
    "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
    "reset_optimizer_at_sat": True,
}

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "pretrained": True, "class_weighted_ce": False, "constraint_chunk_size": 256,
    "warmup_epochs": 3, "constraint_epochs": 1, "seed": 1,
}


def main():
    ds_name = sys.argv[1]
    n_classes = int(sys.argv[2])
    sweep_root = f"results/pending_runs/probe_{ds_name}"
    ds_meta = {
        "data_dir": f"data/{ds_name}/slice_1",
        "num_classes": n_classes,
        "image_size": 224,
        "target_column": "label",
        "group_column": "synth_group",
        "constrained_class": 0,
    }
    hp = {**SHARED_HP, **PER_METHOD_TRALO}
    bmid = compute_base_model_id("MobileNetV3", hp, dataset_mode=ds_name,
                                 data_dir=ds_meta["data_dir"],
                                 dataset_config=ds_meta)
    cfg = {
        "methodology": "tralo", "model_name": "MobileNetV3",
        "constraint": [0.5, 0.5], "constraint_tag": "L50_G50",
        "dataset_mode": ds_name, "dataset_config": ds_meta,
        "hyperparams": hp, "base_model_id": bmid,
        "exp_name": f"probe_{ds_name}_w3",
        "experiment_path": str(Path(sweep_root) / "probe"),
    }
    save_configs([cfg], output_dir=sweep_root)


if __name__ == "__main__":
    main()
