"""TraLO long-training convergence test: 1000 constraint epochs.

Hypothesis: with default HP, TraLO never satisfies on TissueMNIST in 100
epochs (raw_excess hovers ~50). Does it converge at 1000? Or hit a true
plateau at the bounded-penalty fixed point?

3 seeds, MobileNetV3, L50_G50 class 4, alpha_kl=0 (default), penalty=both.
~12 min/epoch * 1000 = oh that's wrong. Each epoch is ~5 sec, so 1000 epochs
=~ 5000 sec ≈ 1.4 hours per run, ~4.2 hours total on single GPU.

Track full training_log.csv to plot raw_excess + F1 per 5-epoch tick.

Usage:
    python -m src.config_generators.gen_convergence_test
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

SWEEP_ROOT = "results/pending_runs/convergence_test"
SEEDS = [1, 2, 3]
MODEL = "MobileNetV3"
PAIR = (0.5, 0.5)
DATA_DIR = "data/tissuemnist"

HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 1000,
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


def main():
    cfgs = []
    for seed in SEEDS:
        hp = dict(HP)
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
            "exp_name": f"convtest_tralo_seed{seed}",
            "status": "pending",
            "experiment_path": str(Path(SWEEP_ROOT) / f"seed_{seed}"),
        }
        cfgs.append(cfg)
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Total: {len(cfgs)} convergence-test configs")


if __name__ == "__main__":
    main()
