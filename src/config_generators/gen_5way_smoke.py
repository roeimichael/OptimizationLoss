"""5-way smoke benchmark: tralo, fioretto_ldf, hounie_rcl, heuristic, danits_lp.

One config per methodology, identical scenario otherwise (model, dataset,
constraint, seed). Warmup cache is shared across methodologies (cache key does
not include methodology), so warmup runs at most once.

Usage:
    python -m src.config_generators.gen_5way_smoke
"""

from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)


MODEL = "MobileNetV3"
DATA_DIR = "data/tissuemnist/slice_1"
CONSTRAINT_PAIR = (0.5, 0.5)
CONSTRAINED_CLASS = 4
SEED = 1
SWEEP_ROOT = "results/pending_runs/smoke_5way"

METHODOLOGIES = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic", "danits_lp"]


SHARED_HP = {
    "lr": 1e-4,
    "lr_constraint": 5e-6,
    "dropout": 0.3,
    "batch_size": 64,
    "warmup_epochs": 50,
    "constraint_epochs": 100,
    "use_sum_loss": True,
    "kl_temperature": 1.0,
    "pretrained": True,
    "class_weighted_ce": False,
    "constraint_chunk_size": 256,
    "seed": SEED,
}

# Methodology-specific HPs. Union goes into every config; each methodology only
# reads the keys it cares about. Defaults match each method's hp_defaults.py.
PER_METHOD_HP = {
    "tralo": {
        "lambda_global": 0.01,
        "lambda_local": 0.01,
        "lambda_step": 0.002,
        "initial_rho": 5.0,
        "rho_target": 100.0,
        "alpha_kl": 0.0,
    },
    "fioretto_ldf": {
        "fioretto_step_size": 0.005,
    },
    "hounie_rcl": {
        "hounie_eta_lambda": 0.01,
        "hounie_eta_u": 0.01,
        "hounie_alpha": 10.0,
    },
    "heuristic": {},
    "danits_lp": {},
}

DS_CONFIG = {
    "target_column": "label",
    "group_column": "synth_group",
    "num_classes": 8,
    "image_size": 224,
    "data_dir": DATA_DIR,
    "constrained_class": CONSTRAINED_CLASS,
}


def build(methodology):
    hp = dict(SHARED_HP)
    hp.update(PER_METHOD_HP[methodology])
    ctag = constraint_tag(CONSTRAINT_PAIR)
    path = Path(SWEEP_ROOT) / methodology
    return {
        "methodology": methodology,
        "model_name": MODEL,
        "constraint": list(CONSTRAINT_PAIR),
        "constraint_tag": ctag,
        "dataset_mode": "tissuemnist",
        "dataset_config": DS_CONFIG,
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(
            MODEL, hp, dataset_mode="tissuemnist",
            data_dir=DATA_DIR, dataset_config=DS_CONFIG),
        "exp_name": f"smoke_5way_{methodology}",
        "status": "pending",
        "experiment_path": str(path),
    }


def main():
    cfgs = [build(m) for m in METHODOLOGIES]
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Generated {len(cfgs)} configs: {METHODOLOGIES}")


if __name__ == "__main__":
    main()
