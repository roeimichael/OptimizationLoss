"""Thesis sweep on DermMNIST: headline benchmark + tightness, mirroring TissueMNIST.

DermMNIST classes: 0=AKIEC, 1=BCC, 2=BKL, 3=DF, 4=MEL (constrained), 5=NV, 6=VASC.
Test split (slice_1): 770 NV-heavy. Constrained class 4 (MEL) has 223 samples.
Group column: 'sex' (binary, well-balanced).

Phases:
  B' — Headline: 5 methods × 3 models × 5 seeds × L50_G50 class 4 (75)
  C' — Tightness: 5 methods × MobileNetV3 × 5 seeds × {L30_G30, L70_G70} (50)

Skip penalty ablation (Phase A) here — same conclusion as TissueMNIST.

Usage:
    python -m src.config_generators.gen_dermmnist_thesis
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

DATA_DIR = "data/dermmnist/slice_1"
SWEEP_ROOT = "results/pending_runs/thesis_dermmnist"
SEEDS = [1, 2, 3, 4, 5]
MODELS = ["MobileNetV3", "ResNet18", "EfficientNetB0"]
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic", "danits_lp"]
CONSTRAINED_CLASS = 4

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
    "target_column": "label", "group_column": "sex",
    "num_classes": 7, "image_size": 224, "data_dir": DATA_DIR,
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
        "dataset_mode": "dermmnist",
        "dataset_config": ds,
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(
            model, hp, dataset_mode="dermmnist",
            data_dir=DATA_DIR, dataset_config=ds),
        "exp_name": f"thesis_derm_{exp_suffix}_seed{seed}",
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
