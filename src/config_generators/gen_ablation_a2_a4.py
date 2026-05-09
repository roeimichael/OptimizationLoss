"""TraLO ablations A2 (lambda ratchet) + A4 (KL anchor) — config-only.

Two datasets (TissueMNIST + So2Sat), MobileNetV3, L50_G50, 5 seeds.

A2 — lambda ratchet ablation:
  default: lambda_step=0.002 (ratchet on)
  ablation: lambda_step=0.0 (ratchet off, lambdas frozen at initial)

A4 — KL anchor sweep:
  alpha_kl in {0.0 (default), 0.1, 0.3, 0.5, 1.0}

A4's alpha_kl=0 already covered by main thesis sweeps; we generate the 4 NEW
values only and skip duplication.

Run on dsisco02, optloss env. Total: 2 datasets x (1 A2 + 4 A4) x 5 seeds = 50 runs.

Usage:
    python -m src.config_generators.gen_ablation_a2_a4
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

SWEEP_ROOT = "results/pending_runs/ablation_tralo"
SEEDS = [1, 2, 3, 4, 5]
MODEL = "MobileNetV3"
PAIR = (0.5, 0.5)

DATASETS = {
    "tissuemnist": {
        "data_dir": "data/tissuemnist",
        "constrained_class": 4,
        "num_classes": 8,
        "group_column": "synth_group",
    },
    "so2sat": {
        "data_dir": "data/so2sat",
        "constrained_class": 7,
        "num_classes": 17,
        "group_column": "city_id",
    },
}

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 100,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "lambda_global": 0.05, "lambda_local": 0.05,
    "initial_rho": 5.0, "rho_target": 100.0,
    "penalty_mode": "both",
}

# A2: lambda_step ablation (just one ablation cell, default=0.002 already in main)
# A4: alpha_kl sweep (4 new values, skip 0 which is in main sweeps)
ABLATIONS = [
    ("a2_no_ratchet",   {"lambda_step": 0.0,   "alpha_kl": 0.0}),
    ("a4_kl_0_1",       {"lambda_step": 0.002, "alpha_kl": 0.1}),
    ("a4_kl_0_3",       {"lambda_step": 0.002, "alpha_kl": 0.3}),
    ("a4_kl_0_5",       {"lambda_step": 0.002, "alpha_kl": 0.5}),
    ("a4_kl_1_0",       {"lambda_step": 0.002, "alpha_kl": 1.0}),
]


def main():
    cfgs = []
    for ds_name, ds_meta in DATASETS.items():
        ds = {
            "target_column": "label", "group_column": ds_meta["group_column"],
            "num_classes": ds_meta["num_classes"], "image_size": 224,
            "data_dir": ds_meta["data_dir"],
            "constrained_class": ds_meta["constrained_class"],
        }
        for tag, override in ABLATIONS:
            for seed in SEEDS:
                hp = dict(SHARED_HP)
                hp.update(override)
                hp["seed"] = seed
                cfg = {
                    "methodology": "tralo",
                    "model_name": MODEL,
                    "constraint": list(PAIR),
                    "constraint_tag": constraint_tag(PAIR),
                    "dataset_mode": ds_name,
                    "dataset_config": dict(ds),
                    "hyperparams": hp,
                    "base_model_id": compute_base_model_id(
                        MODEL, hp, dataset_mode=ds_name,
                        data_dir=ds_meta["data_dir"], dataset_config=ds),
                    "exp_name": f"abl_{ds_name}_{tag}_seed{seed}",
                    "status": "pending",
                    "experiment_path": str(
                        Path(SWEEP_ROOT) / ds_name / tag / f"seed_{seed}"),
                }
                cfgs.append(cfg)
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Total: {len(cfgs)} configs across {len(DATASETS)} datasets x "
          f"{len(ABLATIONS)} ablations x {len(SEEDS)} seeds")


if __name__ == "__main__":
    main()
