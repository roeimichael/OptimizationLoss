"""Thesis-grade sweep generator: 3 phases for the main paper tables.

Phase A — Penalty form ablation (TraLO methodology paper section):
  rational | quadratic | both  ×  3 models  ×  5 seeds  ×  L50_G50 class 4

Phase B — Headline benchmark (main paper table):
  5 methods  ×  3 models  ×  5 seeds  ×  L50_G50 class 4

Phase C — Tightness sweep (paper subsection):
  5 methods  ×  MobileNetV3  ×  5 seeds  ×  {L30_G30, L70_G70} class 4
  (skip L50_G50 — already in Phase B)

All TissueMNIST. Warmup cache shared across methodology.

Usage:
    python -m src.config_generators.gen_thesis_sweep
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

DATA_DIR = "data/tissuemnist/slice_1"
SWEEP_ROOT = "results/pending_runs/thesis"
SEEDS = [1, 2, 3, 4, 5]
MODELS = ["MobileNetV3", "ResNet18", "EfficientNetB0"]
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic", "danits_lp"]
CONSTRAINED_CLASS = 4

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
    "target_column": "label", "group_column": "synth_group",
    "num_classes": 8, "image_size": 224, "data_dir": DATA_DIR,
    "constrained_class": CONSTRAINED_CLASS,
}


def _build(methodology, model, seed, pair, axis_path, exp_suffix, hp_override=None):
    hp = dict(SHARED_HP)
    hp.update(PER_METHOD_HP[methodology])
    hp["seed"] = seed
    if hp_override:
        hp.update(hp_override)
    ds = dict(DS_BASE)
    return {
        "methodology": methodology,
        "model_name": model,
        "constraint": list(pair),
        "constraint_tag": constraint_tag(pair),
        "dataset_mode": "tissuemnist",
        "dataset_config": ds,
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(
            model, hp, dataset_mode="tissuemnist",
            data_dir=DATA_DIR, dataset_config=ds),
        "exp_name": f"thesis_{exp_suffix}_seed{seed}",
        "status": "pending",
        "experiment_path": str(Path(SWEEP_ROOT) / axis_path / f"seed_{seed}"),
    }


def main():
    cfgs = []

    # Phase A — penalty form ablation (TraLO only)
    for mode in ("rational", "quadratic", "both"):
        for model in MODELS:
            for seed in SEEDS:
                cfgs.append(_build(
                    "tralo", model, seed, (0.5, 0.5),
                    axis_path=f"ablation_penalty/{mode}/{model}",
                    exp_suffix=f"penalty_{mode}_{model}",
                    hp_override={"penalty_mode": mode},
                ))

    # Phase B — headline 5-method benchmark
    for model in MODELS:
        for method in METHODS:
            for seed in SEEDS:
                cfgs.append(_build(
                    method, model, seed, (0.5, 0.5),
                    axis_path=f"headline/{model}/{method}",
                    exp_suffix=f"headline_{model}_{method}",
                ))

    # Phase C — tightness sweep (MobileNetV3 only, exclude L50_G50)
    for pair in [(0.3, 0.3), (0.7, 0.7)]:
        for method in METHODS:
            for seed in SEEDS:
                cfgs.append(_build(
                    method, "MobileNetV3", seed, pair,
                    axis_path=f"tightness/{constraint_tag(pair)}/{method}",
                    exp_suffix=f"tightness_{constraint_tag(pair)}_{method}",
                ))

    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Total configs generated: {len(cfgs)}")
    print(f"  Phase A (penalty ablation): {3 * len(MODELS) * len(SEEDS)}")
    print(f"  Phase B (headline):          {len(MODELS) * len(METHODS) * len(SEEDS)}")
    print(f"  Phase C (tightness):         {2 * len(METHODS) * len(SEEDS)}")


if __name__ == "__main__":
    main()
