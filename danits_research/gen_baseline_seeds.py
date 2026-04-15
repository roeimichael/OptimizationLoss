"""Baseline replicates at seeds 1-5: fair comparison vs our_approach E/F blocks.

Produces heuristic + danits_lp for each of L20_G80 / L50_G50, seeds 1..5.
All share warmup caches already built by the 40-run sweep (seeds 0-5 on disk).
"""

from __future__ import annotations
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

MODEL = "MobileNetV3"
DATA_DIR = "data/tissuemnist/slice_1"
CONSTRAINED_CLASS = 4
SCENARIO = "sweep40_single_GE"  # reuse same scenario dir
TIERS = [(0.2, 0.8), (0.5, 0.5)]
SEEDS = [1, 2, 3, 4, 5]
METHODS = ["heuristic", "danits_lp"]

BASELINE_HP = {
    "lr": 0.0001, "lr_constraint": 5e-06, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
    "lambda_global": 0.01, "lambda_local": 0.01, "lambda_step": 0.002,
    "use_sum_loss": True, "initial_rho": 5.0, "rho_target": 100.0,
    "alpha_kl": 0.1, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 64,
}


def _build(methodology, tier, seed):
    hp = dict(BASELINE_HP)
    hp["seed"] = seed
    ctag = constraint_tag(tier)
    variant = f"slice_1_seed{seed}"
    path = Path("results/pending_runs") / SCENARIO / ctag / MODEL / methodology / variant
    ds = {
        "target_column": "label", "group_column": "synth_group",
        "num_classes": 8, "image_size": 224,
        "data_dir": DATA_DIR, "constrained_class": CONSTRAINED_CLASS,
    }
    return {
        "methodology": methodology, "model_name": MODEL,
        "constraint": list(tier), "constraint_tag": ctag,
        "dataset_mode": "tissuemnist", "dataset_config": ds,
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(
            MODEL, hp, dataset_mode="tissuemnist", data_dir=DATA_DIR),
        "exp_name": f"sweep40_baselines_{ctag}_{methodology}_s{seed}",
        "status": "pending",
        "experiment_path": str(path),
    }


def main():
    cfgs = []
    for tier in TIERS:
        for meth in METHODS:
            for s in SEEDS:
                cfgs.append(_build(meth, tier, s))
    print(f"Total: {len(cfgs)} baseline configs")
    print(f"  methods: {METHODS}")
    print(f"  seeds: {SEEDS}")
    print(f"  tiers: {[constraint_tag(t) for t in TIERS]}")
    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
