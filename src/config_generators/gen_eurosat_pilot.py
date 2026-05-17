"""EuroSAT pilot: 1 heuristic config (warmup + posthoc only) to verify the
integration works and warmup accuracy matches published baselines (~98%).

Constrained class: 3 = Highway (smallest natural class, ~2700 samples).
L50_G50 to mirror existing protocol.

Usage:
    python -m src.config_generators.gen_eurosat_pilot
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

DATA_DIR = "data/eurosat/slice_1"
SWEEP_ROOT = "results/pending_runs/eurosat_pilot"

HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 100,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "seed": 1,
}

DS = {
    "target_column": "label", "group_column": "synth_group",
    "num_classes": 10, "image_size": 224, "data_dir": DATA_DIR,
    "constrained_class": 3,  # Highway
}


def build(methodology, hp_extra=None):
    hp = dict(HP)
    if hp_extra: hp.update(hp_extra)
    return {
        "methodology": methodology,
        "model_name": "MobileNetV3",
        "constraint": [0.5, 0.5],
        "constraint_tag": constraint_tag((0.5, 0.5)),
        "dataset_mode": "eurosat",
        "dataset_config": DS,
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(
            "MobileNetV3", hp, dataset_mode="eurosat",
            data_dir=DATA_DIR, dataset_config=DS),
        "exp_name": f"eurosat_pilot_{methodology}",
        "status": "pending",
        "experiment_path": str(Path(SWEEP_ROOT) / methodology),
    }


def main():
    cfgs = [
        build("heuristic"),  # warmup + posthoc — fastest sanity
        build("tralo", hp_extra={
            "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
            "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
            "penalty_mode": "both"}),
        build("fioretto_ldf", hp_extra={"fioretto_step_size": 0.005}),
    ]
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Generated {len(cfgs)} pilot configs")


if __name__ == "__main__":
    main()
