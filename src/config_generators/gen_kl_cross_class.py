"""KL anchor cross-class validation.

A4 showed alpha_kl=0.1 wins on TissueMNIST class 4. This sweep checks
whether the win generalizes to other constrained classes.

TissueMNIST L50_G50, MobileNetV3, 3 seeds.

Test classes {1, 4, 7} (Phase F multi-class candidates) x alpha_kl {0.0 (default reference), 0.1}.
Total: 3 classes x 2 alpha x 3 seeds = 18 runs (alpha=0 for {1,7} is new since headline used class 4 only).

Usage:
    python -m src.config_generators.gen_kl_cross_class
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

SWEEP_ROOT = "results/pending_runs/kl_cross_class"
SEEDS = [1, 2, 3]
MODEL = "MobileNetV3"
PAIR = (0.5, 0.5)
DATA_DIR = "data/tissuemnist"
TEST_CLASSES = [1, 4, 7]
ALPHA_KLS = [0.0, 0.1]

DEFAULT_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 100,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0,
    "penalty_mode": "both",
}


def main():
    cfgs = []
    for cls in TEST_CLASSES:
        for alpha in ALPHA_KLS:
            tag = f"cls{cls}_alpha_{str(alpha).replace('.', 'p')}"
            for seed in SEEDS:
                hp = dict(DEFAULT_HP)
                hp["alpha_kl"] = alpha
                hp["seed"] = seed
                ds = {
                    "target_column": "label", "group_column": "synth_group",
                    "num_classes": 8, "image_size": 224, "data_dir": DATA_DIR,
                    "constrained_class": cls,
                }
                cfg = {
                    "methodology": "tralo",
                    "model_name": MODEL,
                    "constraint": list(PAIR),
                    "constraint_tag": constraint_tag(PAIR),
                    "dataset_mode": "tissuemnist",
                    "dataset_config": dict(ds),
                    "hyperparams": hp,
                    "base_model_id": compute_base_model_id(
                        MODEL, hp, dataset_mode="tissuemnist",
                        data_dir=DATA_DIR, dataset_config=ds),
                    "exp_name": f"klx_{tag}_seed{seed}",
                    "status": "pending",
                    "experiment_path": str(Path(SWEEP_ROOT) / tag / f"seed_{seed}"),
                }
                cfgs.append(cfg)
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Total: {len(cfgs)} ({len(TEST_CLASSES)} classes × {len(ALPHA_KLS)} alpha × {len(SEEDS)} seeds)")


if __name__ == "__main__":
    main()
