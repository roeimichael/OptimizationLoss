"""Thesis sweep extension: more pairs, asymmetric, and multi-class.

MobileNetV3 only (fast, headline-equivalent picture). 3 seeds (1, 2, 3) — drop
to 2/4 if more time later.

Phase D — extended tightness:
  pairs (0.2,0.2), (0.4,0.4), (0.6,0.6), (0.8,0.8)  ×  5 methods × 3 seeds × class 4 = 60

Phase E — extreme asymmetric:
  (0.3, 0.7) and (0.7, 0.3)                          ×  5 methods × 3 seeds × class 4 = 30

Phase F — multi-class:
  classes (4, 1), (3, 4), (1, 4, 7)                  ×  5 methods × 3 seeds × L50_G50 = 45

Total: 135 configs.

Usage:
    python -m src.config_generators.gen_thesis_extended
"""
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

DATA_DIR = "data/tissuemnist/slice_1"
SWEEP_ROOT = "results/pending_runs/thesis_ext"
SEEDS = [1, 2, 3]
MODEL = "MobileNetV3"
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic", "danits_lp"]

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


def _build(methodology, seed, pair, classes, axis_path, exp_suffix):
    hp = dict(SHARED_HP)
    hp.update(PER_METHOD_HP[methodology])
    hp["seed"] = seed
    ds = {
        "target_column": "label", "group_column": "synth_group",
        "num_classes": 8, "image_size": 224, "data_dir": DATA_DIR,
        "constrained_class": list(classes) if len(classes) > 1 else classes[0],
    }
    return {
        "methodology": methodology,
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": constraint_tag(pair),
        "dataset_mode": "tissuemnist",
        "dataset_config": ds,
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(
            MODEL, hp, dataset_mode="tissuemnist",
            data_dir=DATA_DIR, dataset_config=ds),
        "exp_name": f"thesis_ext_{exp_suffix}_seed{seed}",
        "status": "pending",
        "experiment_path": str(Path(SWEEP_ROOT) / axis_path / f"seed_{seed}"),
    }


def main():
    cfgs = []

    # Phase D — extended tightness (single class 4)
    for pair in [(0.2, 0.2), (0.4, 0.4), (0.6, 0.6), (0.8, 0.8)]:
        tag = constraint_tag(pair)
        for method in METHODS:
            for seed in SEEDS:
                cfgs.append(_build(
                    method, seed, pair, (4,),
                    axis_path=f"tightness_ext/{tag}/{method}",
                    exp_suffix=f"tight_{tag}_{method}",
                ))

    # Phase E — extreme asymmetric (single class 4)
    for pair, name in [((0.3, 0.7), "L30_G70"), ((0.7, 0.3), "L70_G30")]:
        for method in METHODS:
            for seed in SEEDS:
                cfgs.append(_build(
                    method, seed, pair, (4,),
                    axis_path=f"asymmetric_ext/{name}/{method}",
                    exp_suffix=f"asym_{name}_{method}",
                ))

    # Phase F — multi-class at L50_G50
    for classes in [(4, 1), (3, 4), (1, 4, 7)]:
        cls_tag = "_".join(str(c) for c in classes)
        for method in METHODS:
            for seed in SEEDS:
                cfgs.append(_build(
                    method, seed, (0.5, 0.5), classes,
                    axis_path=f"multiclass_ext/cls_{cls_tag}/{method}",
                    exp_suffix=f"multi_{cls_tag}_{method}",
                ))

    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"Total: {len(cfgs)}")
    print(f"  Phase D tightness_ext: {4 * len(METHODS) * len(SEEDS)}")
    print(f"  Phase E asymmetric_ext: {2 * len(METHODS) * len(SEEDS)}")
    print(f"  Phase F multiclass_ext: {3 * len(METHODS) * len(SEEDS)}")


if __name__ == "__main__":
    main()
