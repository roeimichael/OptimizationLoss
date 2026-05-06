"""Combined-best TraLO HP sweep.

Each of init_lam=0.05, step=0.01, rho_init=50 individually beat the
TraLO baseline on L50_G50 class 4. Test combinations for synergy.

Output: results/pending_runs/tralo_combo/
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

MODEL = "MobileNetV3"
DATA_DIR = "data/tissuemnist/slice_1"
SEED = 1
SWEEP_ROOT = "results/pending_runs/tralo_combo"
PAIR = (0.5, 0.5)
CLASSES = (4,)

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 100,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "seed": SEED,
    "lambda_global": 0.01, "lambda_local": 0.01, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
}

DS = {
    "target_column": "label", "group_column": "synth_group",
    "num_classes": 8, "image_size": 224, "data_dir": DATA_DIR,
    "constrained_class": list(CLASSES) if len(CLASSES) > 1 else CLASSES[0],
}

COMBOS = {
    "init_step_rho":   {"lambda_global": 0.05, "lambda_local": 0.05,
                        "lambda_step": 0.01, "initial_rho": 50.0},
    "init_step":       {"lambda_global": 0.05, "lambda_local": 0.05,
                        "lambda_step": 0.01},
    "init_rho":        {"lambda_global": 0.05, "lambda_local": 0.05,
                        "initial_rho": 50.0},
    "step_rho":        {"lambda_step": 0.01, "initial_rho": 50.0},
    "init_step_rho_kl":{"lambda_global": 0.05, "lambda_local": 0.05,
                        "lambda_step": 0.01, "initial_rho": 50.0,
                        "alpha_kl": 0.1},
    "long_init_step_rho": {"lambda_global": 0.05, "lambda_local": 0.05,
                           "lambda_step": 0.01, "initial_rho": 50.0,
                           "constraint_epochs": 200},
}


def build(name, override):
    hp = dict(SHARED_HP)
    hp.update(override)
    path = Path(SWEEP_ROOT) / name
    return {
        "methodology": "tralo",
        "model_name": MODEL,
        "constraint": list(PAIR),
        "constraint_tag": constraint_tag(PAIR),
        "dataset_mode": "tissuemnist",
        "dataset_config": DS,
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(
            MODEL, hp, dataset_mode="tissuemnist",
            data_dir=DATA_DIR, dataset_config=DS),
        "exp_name": f"tralo_combo_{name}",
        "status": "pending",
        "experiment_path": str(path),
    }


def main():
    cfgs = [build(name, override) for name, override in COMBOS.items()]
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Generated {len(cfgs)} combo configs")


if __name__ == "__main__":
    main()
