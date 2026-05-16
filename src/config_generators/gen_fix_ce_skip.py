"""Fix-1 test: CE saturation skip restored.

Hardest cell: (1,4,7) L30_G30, MobileNetV3, seed=1, 300 epochs.
Baseline (beta_sweep/beta_0_0): diverges to end_ex ~411, C4 hard ~239.
Hypothesis: CE-skip lets bounded penalty reach E=0 once acc saturates.
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

DATA_DIR = "data/tissuemnist/slice_1"
SWEEP_ROOT = "results/pending_runs/fix_ce_skip"
SEED = 1
MODEL = "MobileNetV3"
PAIR = (0.3, 0.3)
CLASSES = (1, 4, 7)

DS = {
    "target_column": "label", "group_column": "synth_group",
    "num_classes": 8, "image_size": 224, "data_dir": DATA_DIR,
    "constrained_class": list(CLASSES),
}

HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
    "penalty_mode": "both",
    "seed": SEED,
    # Fix 1: CE skip enabled (default True now in train.py)
    "enable_ce_skip": True,
}


def main():
    tag = "ce_skip_on"
    cfg = {
        "methodology": "tralo",
        "model_name": MODEL,
        "constraint": list(PAIR),
        "constraint_tag": constraint_tag(PAIR),
        "dataset_mode": "tissuemnist",
        "dataset_config": dict(DS),
        "hyperparams": dict(HP),
        "base_model_id": compute_base_model_id(
            MODEL, HP, dataset_mode="tissuemnist",
            data_dir=DATA_DIR, dataset_config=DS),
        "exp_name": f"fix_ce_skip_{tag}_seed{SEED}",
        "status": "pending",
        "experiment_path": str(Path(SWEEP_ROOT) / tag),
    }
    save_configs([cfg], output_dir=SWEEP_ROOT)
    print(f"Created 1 config at {SWEEP_ROOT}/{tag}")


if __name__ == "__main__":
    main()
