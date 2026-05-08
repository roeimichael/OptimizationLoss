"""So2Sat LCZ42 thesis sweep: headline + tightness, mirroring DermMNIST/TissueMNIST.

Class 7 (LCZ-8 Large low-rise) constrained: 14.1% of data, every city has 11+
samples so K never collapses. 10 real cities as local groups (REAL geography,
not synthetic).

Phases:
  B'' — Headline: 5 methods x 3 models x 5 seeds x L50_G50 (75)
  C'' — Tightness: 5 methods x MobileNetV3 x 5 seeds x {L30_G30, L70_G70} (50)

Total: 125 configs.

Usage:
    python -m src.config_generators.gen_so2sat_thesis
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

DATA_DIR = "data/so2sat"
SWEEP_ROOT = "results/pending_runs/thesis_so2sat"
SEEDS = [1, 2, 3, 4, 5]
MODELS = ["MobileNetV3", "ResNet18", "EfficientNetB0"]
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic", "danits_lp"]
CONSTRAINED_CLASS = 7

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 100,
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
    "danits_lp": {},
}

DS_BASE = {
    "target_column": "label", "group_column": "city_id",
    "num_classes": 17, "image_size": 224, "data_dir": DATA_DIR,
    "constrained_class": CONSTRAINED_CLASS,
}


def _build(methodology, model, seed, pair, axis_path, exp_suffix):
    hp = dict(SHARED_HP)
    hp.update(PER_METHOD_HP[methodology])
    hp["seed"] = seed
    ds = dict(DS_BASE)
    return {
        "methodology": methodology,
        "model_name": model,
        "constraint": list(pair),
        "constraint_tag": constraint_tag(pair),
        "dataset_mode": "so2sat",
        "dataset_config": ds,
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(
            model, hp, dataset_mode="so2sat",
            data_dir=DATA_DIR, dataset_config=ds),
        "exp_name": f"thesis_so2sat_{exp_suffix}_seed{seed}",
        "status": "pending",
        "experiment_path": str(Path(SWEEP_ROOT) / axis_path / f"seed_{seed}"),
    }


def main():
    cfgs = []
    for model in MODELS:
        for method in METHODS:
            for seed in SEEDS:
                cfgs.append(_build(
                    method, model, seed, (0.5, 0.5),
                    axis_path=f"headline/{model}/{method}",
                    exp_suffix=f"headline_{model}_{method}",
                ))
    for pair in [(0.3, 0.3), (0.7, 0.7)]:
        for method in METHODS:
            for seed in SEEDS:
                cfgs.append(_build(
                    method, "MobileNetV3", seed, pair,
                    axis_path=f"tightness/{constraint_tag(pair)}/{method}",
                    exp_suffix=f"tightness_{constraint_tag(pair)}_{method}",
                ))
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Total: {len(cfgs)}")
    print(f"  Headline:  {len(MODELS) * len(METHODS) * len(SEEDS)}")
    print(f"  Tightness: {2 * len(METHODS) * len(SEEDS)}")


if __name__ == "__main__":
    main()
