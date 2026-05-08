"""So2Sat LCZ42 pilot: 4 methods x MobileNetV3 x 1 seed.

Goal: cheap pilot (4 runs) confirming pipeline works, baseline F1 reasonable,
real city groups (10 cities from KMeans-on-coords) actually load.

If pilot looks healthy -> scale to 5 seeds + headline + tightness.
If pilot degenerate -> revert to add-hounie-benchmark and pivot dataset.

17 LCZ classes. Pick a mid-frequency class to constrain
(too-rare classes have K~0; too-frequent have K~0.5*N which is uninformative).

Usage:
    python -m src.config_generators.gen_so2sat_pilot
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

DATA_DIR = "data/so2sat"
SWEEP_ROOT = "results/pending_runs/so2sat_pilot"
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic"]
SEED = 1
MODEL = "MobileNetV3"
CONSTRAINED_CLASS = 1   # LCZ2 Compact mid-rise — pick after class-distribution check
PAIR = (0.5, 0.5)

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 30, "constraint_epochs": 50,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}

PER_METHOD_HP = {
    "tralo": {"lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
              "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
              "penalty_mode": "both"},
    "fioretto_ldf": {"fioretto_step_size": 0.005},
    "hounie_rcl": {"hounie_eta_lambda": 0.01, "hounie_eta_u": 0.01,
                   "hounie_alpha": 10.0},
    "heuristic": {},
}

DS = {
    "target_column": "label", "group_column": "city_id",
    "num_classes": 17, "image_size": 224, "data_dir": DATA_DIR,
    "constrained_class": CONSTRAINED_CLASS,
}


def main():
    cfgs = []
    for method in METHODS:
        hp = dict(SHARED_HP)
        hp.update(PER_METHOD_HP[method])
        hp["seed"] = SEED
        cfg = {
            "methodology": method,
            "model_name": MODEL,
            "constraint": list(PAIR),
            "constraint_tag": constraint_tag(PAIR),
            "dataset_mode": "so2sat",
            "dataset_config": dict(DS),
            "hyperparams": hp,
            "base_model_id": compute_base_model_id(
                MODEL, hp, dataset_mode="so2sat",
                data_dir=DATA_DIR, dataset_config=DS),
            "exp_name": f"so2sat_pilot_{method}_seed{SEED}",
            "status": "pending",
            "experiment_path": str(Path(SWEEP_ROOT) / method),
        }
        cfgs.append(cfg)
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Total: {len(cfgs)} configs")


if __name__ == "__main__":
    main()
