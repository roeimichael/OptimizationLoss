"""CIFAR-100 experiment configs: harder dataset where constraint training should matter.

100 fine classes, 20 superclasses (used as groups). Constrain specific fine classes
where the model naturally over-predicts. Tight constraints (L30_G50) force heavy
redistribution — the regime where our approach should shine on non-accuracy metrics.

Scenarios:
  single_apple:       constrain class 0 (apple, fruit superclass)
  dual_apple_orange:  constrain [0, 53] (apple + orange, same superclass = competing)
  dual_cross_super:   constrain [8, 48] (bicycle + motorcycle, different superclasses)
  triple_vehicles:    constrain [8, 13, 48] (bicycle, bus, motorcycle)

Each: 3 methods x 5 seeds = 15 runs per scenario.
Total: 4 x 15 = 60 runs (20 tralo + 40 baselines).

Usage:
    python -m danits_research.gen_cifar100_experiments
"""

from __future__ import annotations
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

MODEL = "MobileNetV3"
DATA_DIR = "data/cifar100/slice_1"
ROOT = "results/pending_runs/cifar100"
CONSTRAINT_PAIR = (0.3, 0.5)  # L30_G50

SCENARIOS = {
    "single_apple":     {"cc": 0},
    "dual_apple_orange": {"cc": [0, 53]},
    "dual_cross_super":  {"cc": [8, 48]},
    "triple_vehicles":   {"cc": [8, 13, 48]},
}

BASELINE_HP = {
    "lr": 0.0001, "lr_constraint": 5e-06, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
    "lambda_global": 0.01, "lambda_local": 0.01, "lambda_step": 0.002,
    "use_sum_loss": True, "initial_rho": 5.0, "rho_target": 100.0,
    "alpha_kl": 0.1, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 64,
    "lambda_mode": "ratchet", "diagnostic_level": 2,
}

METHODS = ["tralo", "heuristic", "danits_lp"]
SEEDS = [1, 2, 3, 4, 5]


def _ds(cc):
    return {
        "target_column": "label",
        "group_column": "coarse_label",
        "num_classes": 100,
        "image_size": 224,
        "data_dir": DATA_DIR,
        "constrained_class": cc,
    }


def _build(methodology, scenario_name, seed):
    hp = dict(BASELINE_HP)
    hp["seed"] = seed
    if methodology != "tralo":
        hp.pop("diagnostic_level", None)
        hp.pop("lambda_mode", None)

    cc = SCENARIOS[scenario_name]["cc"]
    ctag = constraint_tag(CONSTRAINT_PAIR)
    variant = f"s{seed}"
    path = Path(ROOT) / scenario_name / ctag / MODEL / methodology / variant
    ds_config = _ds(cc)
    return {
        "methodology": methodology,
        "model_name": MODEL,
        "constraint": list(CONSTRAINT_PAIR),
        "constraint_tag": ctag,
        "dataset_mode": "cifar100",
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(
            MODEL, hp, dataset_mode="cifar100", data_dir=DATA_DIR,
            dataset_config=ds_config),
        "exp_name": f"c100_{scenario_name}_{ctag}_{methodology}_s{seed}",
        "status": "pending",
        "experiment_path": str(path),
    }


def main():
    cfgs = []
    for sc in SCENARIOS:
        for meth in METHODS:
            for seed in SEEDS:
                cfgs.append(_build(meth, sc, seed))

    n_oa = sum(1 for c in cfgs if c["methodology"] == "tralo")
    n_bl = len(cfgs) - n_oa
    print("=" * 70)
    print("CIFAR-100 EXPERIMENTS")
    print("=" * 70)
    print(f"Tier: L30_G50")
    for name, sc in SCENARIOS.items():
        print(f"  {name:25s} constrained_class={sc['cc']}")
    print(f"tralo: {n_oa}  baselines: {n_bl}  total: {len(cfgs)}")
    hashes = sorted({c["base_model_id"] for c in cfgs})
    print(f"Warmup hashes: {len(hashes)}")
    save_configs(cfgs, output_dir=ROOT)


if __name__ == "__main__":
    main()
