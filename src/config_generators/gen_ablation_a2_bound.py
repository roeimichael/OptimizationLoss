"""Bound the A2 ratchet claim: lambda_initial x ratchet on/off.

Initial A2 result showed ratchet OFF doesn't degrade F1 noticeably when
lambda_initial=0.05 is well-chosen. This sweep tests whether ratchet
recovers performance when lambda_initial is mis-specified.

TissueMNIST L50_G50 class 4, MobileNetV3, 3 seeds.

lambda_initial in {0.001, 0.01, 0.05, 0.2} x ratchet on/off (lambda_step in {0, 0.002}) = 16 cells.
Total: 16 x 3 seeds = 48 runs.

Usage:
    python -m src.config_generators.gen_ablation_a2_bound
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

SWEEP_ROOT = "results/pending_runs/ablation_a2_bound"
SEEDS = [1, 2, 3]
MODEL = "MobileNetV3"
PAIR = (0.5, 0.5)
DATA_DIR = "data/tissuemnist"

LAMBDA_INITS = [0.001, 0.01, 0.05, 0.2]
RATCHET_STEPS = [0.0, 0.002]   # off vs default

DEFAULT_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 100,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
    "penalty_mode": "both",
}

DS = {
    "target_column": "label", "group_column": "synth_group",
    "num_classes": 8, "image_size": 224, "data_dir": DATA_DIR,
    "constrained_class": 4,
}


def main():
    cfgs = []
    for li in LAMBDA_INITS:
        for ls in RATCHET_STEPS:
            ratch = "on" if ls > 0 else "off"
            tag = f"lam_init_{li}_ratchet_{ratch}".replace(".", "p")
            for seed in SEEDS:
                hp = dict(DEFAULT_HP)
                hp.update({
                    "lambda_global": li, "lambda_local": li,
                    "lambda_step": ls, "seed": seed,
                })
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
                    "exp_name": f"a2bound_{tag}_seed{seed}",
                    "status": "pending",
                    "experiment_path": str(Path(SWEEP_ROOT) / tag / f"seed_{seed}"),
                }
                cfgs.append(cfg)
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Total: {len(cfgs)} configs ({len(LAMBDA_INITS)} init × {len(RATCHET_STEPS)} ratchet × {len(SEEDS)} seeds)")


if __name__ == "__main__":
    main()
