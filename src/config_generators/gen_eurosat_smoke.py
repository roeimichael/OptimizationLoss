"""Minimal head-to-head smoke: EuroSAT/MobileNetV3/cls_4 (Industrial)/L50_G50,
TraLO vs Fioretto vs Hounie, 3 seeds each. ~9 runs, ~2h on Blackwell.

Goal: confirm TraLO wins on EuroSAT before committing to the full paper grid.
"""
from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/eurosat_smoke"

DS_CFG = {
    "data_dir": "data/eurosat",
    "num_classes": 10,
    "image_size": 224,
    "target_column": "label",
    "group_column": "synth_group",
}

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}
PER_METHOD = {
    "tralo": {"lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
              "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
              "penalty_mode": "both", "enable_ce_skip": True},
    "fioretto_ldf": {"fioretto_step_size": 0.005},
    "hounie_rcl": {"hounie_eta_lambda": 0.01, "hounie_eta_u": 0.01,
                   "hounie_alpha": 10.0},
}

MODEL = "MobileNetV3"
CLS = 4  # Industrial
TIGHT = (0.5, 0.5)
TIGHT_TAG = "L50_G50"
SEEDS = [1, 2, 3]


def build():
    configs = []
    for method, mhp in PER_METHOD.items():
        for seed in SEEDS:
            hp = {**SHARED_HP, **mhp, "seed": seed}
            ds = {**DS_CFG, "constrained_class": CLS}
            bmid = compute_base_model_id(
                MODEL, hp, "eurosat", DS_CFG["data_dir"], ds)
            exp_name = f"eurosat_smoke_{method}_{MODEL}_cls{CLS}_{TIGHT_TAG}_seed{seed}"
            exp_path = f"{SWEEP_ROOT}/{MODEL}/cls_{CLS}/{TIGHT_TAG}/{method}/seed_{seed}"
            cfg = {
                "methodology": method,
                "model_name": MODEL,
                "constraint": list(TIGHT),
                "constraint_tag": TIGHT_TAG,
                "dataset_mode": "eurosat",
                "dataset_config": ds,
                "hyperparams": hp,
                "base_model_id": bmid,
                "exp_name": exp_name,
                "experiment_path": exp_path,
            }
            configs.append(cfg)
    save_configs(configs, output_dir=SWEEP_ROOT)
    print(f"\nGenerated {len(configs)} configs → {SWEEP_ROOT}")


if __name__ == "__main__":
    build()
