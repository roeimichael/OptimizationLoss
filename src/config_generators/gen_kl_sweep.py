"""KL sweep on hardest cell with fix1 + 3 seeds.

(1,4,7) L30_G30 MobileNetV3 — hardest cell. CE-skip ON.
alpha_kl ∈ {0.0, 0.05, 0.1, 0.3, 0.5} × 3 seeds = 15 TraLO runs.
Goal: find alpha that maximizes acc + F1m without sacrificing satisfaction.

Hypothesis: KL pulls toward warmup (which had high acc, no constraint).
With CE-skip + lambdas frozen, KL provides accuracy anchor while constraint
still drives E -> 0. Best alpha = balance point.
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

DATA_DIR = "data/tissuemnist/slice_1"
SWEEP_ROOT = "results/pending_runs/kl_sweep"
MODEL = "MobileNetV3"
PAIR = (0.3, 0.3)
CLASSES = (1, 4, 7)
ALPHAS = [0.0, 0.05, 0.1, 0.3, 0.5]
SEEDS = [1, 2, 3]

BASE_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0,
    "penalty_mode": "both",
    "enable_ce_skip": True,
}


def main():
    cfgs = []
    for seed in SEEDS:
        for alpha in ALPHAS:
            hp = dict(BASE_HP)
            hp["alpha_kl"] = alpha
            hp["seed"] = seed
            atag = f"akl_{alpha:.2f}".replace(".", "_")
            ds = {
                "target_column": "label", "group_column": "synth_group",
                "num_classes": 8, "image_size": 224, "data_dir": DATA_DIR,
                "constrained_class": list(CLASSES),
            }
            cfgs.append({
                "methodology": "tralo",
                "model_name": MODEL,
                "constraint": list(PAIR),
                "constraint_tag": constraint_tag(PAIR),
                "dataset_mode": "tissuemnist",
                "dataset_config": ds,
                "hyperparams": hp,
                "base_model_id": compute_base_model_id(
                    MODEL, hp, dataset_mode="tissuemnist",
                    data_dir=DATA_DIR, dataset_config=ds),
                "exp_name": f"kl_sweep_{atag}_seed{seed}",
                "status": "pending",
                "experiment_path": str(Path(SWEEP_ROOT) / f"seed_{seed}" / atag),
            })
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Created {len(cfgs)} configs ({len(ALPHAS)} alphas x {len(SEEDS)} seeds)")


if __name__ == "__main__":
    main()
