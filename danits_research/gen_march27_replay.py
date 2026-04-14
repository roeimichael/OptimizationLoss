"""
Replay the March 27 experiment setting that gave +4% improvement.

Target config: tissuemnist/single_GE/L50_G50/MobileNetV3 (got acc=0.5608 for our_approach)

KEY: uses the MARCH 27 HYPERPARAMS which rely on LEGACY behavior:
  - NO `disable_ce_skip` flag (defaults to False → CE skip active when train_acc saturates)
  - NO `disable_lambda_toggle` flag (defaults to False → lambdas zero on satisfaction)
  - NO `seed` field (non-deterministic, but cache lookup doesn't include seed)

Adds diagnostic_level=2 to observe gradient norms during training.

Generates 3 methods (heuristic, danits_lp, our_approach) for direct comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id,
    constraint_tag,
    save_configs,
)

# Exact March 27 hyperparams (no disable flags, no seed)
MARCH27_HYPERPARAMS = {
    "lr": 0.0001,
    "lr_constraint": 5e-06,
    "dropout": 0.3,
    "batch_size": 64,
    "warmup_epochs": 50,
    "constraint_epochs": 300,
    "lambda_global": 0.01,
    "lambda_local": 0.01,
    "lambda_step": 0.002,
    "use_sum_loss": True,
    "initial_rho": 5.0,
    "rho_target": 100.0,
    "alpha_kl": 0.1,
    "kl_temperature": 1.0,
    "pretrained": True,
    "class_weighted_ce": False,
    "constraint_chunk_size": 64,
    # NO disable_ce_skip, NO disable_lambda_toggle, NO seed
    # But ADD diagnostic_level for our_approach
}

MODEL = "MobileNetV3"
SCENARIO_NAME = "mar27_single_GE"
SCENARIO_CONSTRAINED_CLASS = 4  # GE class index in tissuemnist
SLICE_IDX = 1
PAIR = (0.5, 0.5)  # L50_G50

METHODOLOGIES = ["heuristic", "our_approach", "danits_lp"]


def _build_config(methodology, hp):
    ctag = constraint_tag(PAIR)
    exp_name = f"mar27replay_{SCENARIO_NAME}_{ctag}_{MODEL}_{methodology}"
    data_dir = f"data/tissuemnist/slice_{SLICE_IDX}"
    ds_config = {
        "target_column": "label",
        "group_column": "synth_group",
        "num_classes": 8,
        "image_size": 224,
        "data_dir": data_dir,
        "constrained_class": SCENARIO_CONSTRAINED_CLASS,
    }
    path = (Path("results") / "pending_runs" / SCENARIO_NAME / ctag
            / MODEL / methodology / f"slice_{SLICE_IDX}")
    return {
        "methodology": methodology,
        "model_name": MODEL,
        "constraint": list(PAIR),
        "constraint_tag": ctag,
        "dataset_mode": "tissuemnist",
        "dataset_config": ds_config,
        "hyperparams": hp.copy(),
        "base_model_id": compute_base_model_id(
            MODEL, hp, dataset_mode="tissuemnist", data_dir=data_dir),
        "exp_name": exp_name,
        "status": "pending",
        "experiment_path": str(path),
    }


def main():
    configs = []
    for meth in METHODOLOGIES:
        hp = dict(MARCH27_HYPERPARAMS)
        if meth == "our_approach":
            hp["diagnostic_level"] = 2  # full diagnostics
        configs.append(_build_config(meth, hp))

    print("=" * 60)
    print("MARCH 27 REPLAY")
    print("=" * 60)
    print(f"Model: {MODEL} | Dataset: tissuemnist slice_{SLICE_IDX}")
    print(f"Scenario: {SCENARIO_NAME} (constrained class={SCENARIO_CONSTRAINED_CLASS}, L{int(PAIR[0]*100)}_G{int(PAIR[1]*100)})")
    print(f"Methods: {METHODOLOGIES}")
    print(f"Key flags: NO disable_ce_skip, NO disable_lambda_toggle (legacy behavior)")
    print()
    save_configs(configs, output_dir="results/pending_runs")
    hashes = {c["base_model_id"] for c in configs}
    print(f"Warmup cache hashes: {sorted(hashes)}")


if __name__ == "__main__":
    main()
