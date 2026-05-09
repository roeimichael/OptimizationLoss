"""TraLO HP sensitivity sweeps S1-S4.

TissueMNIST L50_G50 class 4, MobileNetV3, 3 seeds (per planner: cheaper grid,
sensitivity is about response shape not cross-dataset generalization).

S1 — lr_constraint in {1e-6, 5e-6 (default), 1e-5, 5e-5}     (skip 5e-6)
S2 — rho_target    in {10, 50, 100 (default), 200, 500}      (skip 100)
S3 — lambda_step   in {0.0005, 0.001, 0.002 (default), 0.005, 0.01}  (skip 0.002)
S4 — constraint_epochs in {50, 100 (default), 200, 300}      (skip 100)

Defaults are skipped because they're already in the main thesis sweep.

Total: (3 + 4 + 4 + 3) x 3 seeds = 42 runs.

Usage:
    python -m src.config_generators.gen_hp_sensitivity
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

SWEEP_ROOT = "results/pending_runs/hp_sensitivity"
SEEDS = [1, 2, 3]
MODEL = "MobileNetV3"
PAIR = (0.5, 0.5)
DATA_DIR = "data/tissuemnist"

DEFAULT_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 100,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
    "penalty_mode": "both",
}

DS = {
    "target_column": "label", "group_column": "synth_group",
    "num_classes": 8, "image_size": 224, "data_dir": DATA_DIR,
    "constrained_class": 4,
}

# (axis_tag, hp_key, value, exp_tag) — defaults skipped
SWEEPS = [
    ("s1_lr_constraint", "lr_constraint", 1e-6, "lr_1e-6"),
    ("s1_lr_constraint", "lr_constraint", 1e-5, "lr_1e-5"),
    ("s1_lr_constraint", "lr_constraint", 5e-5, "lr_5e-5"),
    ("s2_rho_target",    "rho_target",    10,   "rho_10"),
    ("s2_rho_target",    "rho_target",    50,   "rho_50"),
    ("s2_rho_target",    "rho_target",    200,  "rho_200"),
    ("s2_rho_target",    "rho_target",    500,  "rho_500"),
    ("s3_lambda_step",   "lambda_step",   0.0005, "lstep_0p0005"),
    ("s3_lambda_step",   "lambda_step",   0.001,  "lstep_0p001"),
    ("s3_lambda_step",   "lambda_step",   0.005,  "lstep_0p005"),
    ("s3_lambda_step",   "lambda_step",   0.01,   "lstep_0p01"),
    ("s4_constraint_epochs", "constraint_epochs", 50,  "ep_50"),
    ("s4_constraint_epochs", "constraint_epochs", 200, "ep_200"),
    ("s4_constraint_epochs", "constraint_epochs", 300, "ep_300"),
]


def main():
    cfgs = []
    for axis_tag, hp_key, value, exp_tag in SWEEPS:
        for seed in SEEDS:
            hp = dict(DEFAULT_HP)
            hp[hp_key] = value
            hp["seed"] = seed
            cfg = {
                "methodology": "tralo",
                "model_name": MODEL,
                "constraint": list(PAIR),
                "constraint_tag": constraint_tag(PAIR),
                "dataset_mode": "tissuemnist",
                "dataset_config": dict(DS),
                "hyperparams": hp,
                "base_model_id": compute_base_model_id(
                    MODEL, hp, dataset_mode="tissuemnist",
                    data_dir=DATA_DIR, dataset_config=DS),
                "exp_name": f"hp_{exp_tag}_seed{seed}",
                "status": "pending",
                "experiment_path": str(
                    Path(SWEEP_ROOT) / axis_tag / exp_tag / f"seed_{seed}"),
            }
            cfgs.append(cfg)
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Total: {len(cfgs)} HP sensitivity configs across "
          f"{len(set(s[0] for s in SWEEPS))} sweep axes")


if __name__ == "__main__":
    main()
