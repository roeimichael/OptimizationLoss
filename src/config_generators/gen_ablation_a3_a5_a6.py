"""TraLO ablations A3 (lambda freeze), A5 (min-excess restore), A6 (shared lambda).

TissueMNIST L50_G50 class 4, MobileNetV3, 5 seeds. Code-toggle ablations.

Tissue-only because (per planner) ablations only matter in the tight regime;
So2Sat is high-accuracy and ablation effects wash out.

A3 — disable_freeze_on_satisfy: keep ratcheting after first satisfaction
                               (vs default freeze on first sat).
A5 — disable_min_excess_restore: always use final checkpoint
                               (vs default restore best-satisfied / min-excess).
A6 — shared_lambda: increment ALL constrained classes by the same step on any
                    violation (vs default per-class λ).

Total: 3 ablations x 5 seeds = 15 runs.

Usage:
    python -m src.config_generators.gen_ablation_a3_a5_a6
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

SWEEP_ROOT = "results/pending_runs/ablation_tralo"
SEEDS = [1, 2, 3, 4, 5]
MODEL = "MobileNetV3"
PAIR = (0.5, 0.5)
DATA_DIR = "data/tissuemnist"

SHARED_HP = {
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

ABLATIONS = [
    ("a3_no_freeze",   {"disable_freeze_on_satisfy": True}),
    ("a5_no_restore",  {"disable_min_excess_restore": True}),
    ("a6_shared_lam",  {"shared_lambda": True}),
]


def main():
    cfgs = []
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
                "dataset_mode": "tissuemnist",
                "dataset_config": dict(DS),
                "hyperparams": hp,
                "base_model_id": compute_base_model_id(
                    MODEL, hp, dataset_mode="tissuemnist",
                    data_dir=DATA_DIR, dataset_config=DS),
                "exp_name": f"abl_tissuemnist_{tag}_seed{seed}",
                "status": "pending",
                "experiment_path": str(
                    Path(SWEEP_ROOT) / "tissuemnist" / tag / f"seed_{seed}"),
            }
            cfgs.append(cfg)
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Total: {len(cfgs)} configs ({len(ABLATIONS)} ablations x {len(SEEDS)} seeds)")


if __name__ == "__main__":
    main()
