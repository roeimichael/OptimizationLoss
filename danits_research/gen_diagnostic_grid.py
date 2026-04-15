"""
Diagnostic experiment grid: small set of challenging experiments with full diagnostics.

Purpose: understand what's happening inside constraint training by generating rich
diagnostic output. Uses settings that SHOULD show improvement based on March 27 data:
  - MobileNetV3 (weakest model = most room for improvement)
  - TissueMNIST (harder dataset)
  - Multiple constrained classes (multi_GE_CST: classes 4, 2)
  - Mix of tight and medium constraints
  - diagnostic_level=2 for full per-sample tracking

Run:
    python -m danits_research.gen_diagnostic_grid
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config_generators.generate_configs import (
    HYPERPARAMS,
    compute_base_model_id,
    constraint_tag,
    save_configs,
)

# --- Experiment design ---

# MobileNetV3: weakest model on TissueMNIST, showed +3.6% in March 27
DEFAULT_MODEL = "MobileNetV3"
NUM_SLICES = 1

# TissueMNIST: 8 classes
TISSUEMNIST_CLASSES = {
    "CDI": 0, "CDS": 1, "CST": 2, "EPI": 3,
    "GE": 4, "PTC": 5, "STR": 6, "TUB": 7,
}

SCENARIOS = {
    # Multi-constraint: GE + CST (gives LP something non-trivial to optimize)
    "multi_GE_CST": {
        "constrained_class": [TISSUEMNIST_CLASSES["GE"], TISSUEMNIST_CLASSES["CST"]],
    },
    # Single constraint for comparison (the setting we've been testing)
    "single_GE": {
        "constrained_class": TISSUEMNIST_CLASSES["GE"],
    },
}

# Constraint pairs: tighter constraints force more reassignment
CONSTRAINT_GRID = {
    "multi_GE_CST": [
        (0.3, 0.3),   # tight: forces heavy reassignment
        (0.5, 0.5),   # medium: moderate reassignment
        (0.3, 0.8),   # asymmetric: tight local, loose global
    ],
    "single_GE": [
        (0.3, 0.3),   # tight: same setting but single class
        (0.5, 0.5),   # medium: baseline comparison point
    ],
}

METHODOLOGIES = ["heuristic", "our_approach", "danits_lp"]

# Hyperparams: March 27 proven values + diagnostics enabled
DIAG_HYPERPARAMS = {
    **HYPERPARAMS,
    "alpha_kl": 0.1,
    "lr_constraint": 5e-6,
    "disable_ce_skip": True,
    "disable_lambda_toggle": True,
    "seed": 42,
    "diagnostic_level": 2,      # FULL diagnostics
}

# For heuristic and danits_lp, diagnostics aren't relevant (no constraint training)
BASELINE_HYPERPARAMS = {
    **HYPERPARAMS,
    "alpha_kl": 0.1,
    "lr_constraint": 5e-6,
    "disable_ce_skip": True,
    "disable_lambda_toggle": True,
    "seed": 42,
    "diagnostic_level": 0,
}


def _build_config(methodology, scenario_name, constrained_class,
                  constraint_pair, model_name, slice_idx, hp):
    ctag = constraint_tag(constraint_pair)
    exp_name = f"diag_{scenario_name}_{ctag}_{model_name}_{methodology}_slice{slice_idx}"
    data_dir = f"data/tissuemnist/slice_{slice_idx}"
    ds_config = {
        "target_column": "label",
        "group_column": "synth_group",
        "num_classes": 8,
        "image_size": 224,
        "data_dir": data_dir,
        "constrained_class": constrained_class,
    }
    path = (Path("results") / "pending_runs" / scenario_name / ctag
            / model_name / methodology / f"slice_{slice_idx}")
    return {
        "methodology": methodology,
        "model_name": model_name,
        "constraint": list(constraint_pair),
        "constraint_tag": ctag,
        "dataset_mode": "tissuemnist",
        "dataset_config": ds_config,
        "hyperparams": hp.copy(),
        "base_model_id": compute_base_model_id(
            model_name, hp, dataset_mode="tissuemnist", data_dir=data_dir),
        "exp_name": exp_name,
        "status": "pending",
        "experiment_path": str(path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    model_name = args.model

    configs = []

    for scenario_name, scenario_def in SCENARIOS.items():
        pairs = CONSTRAINT_GRID[scenario_name]
        for pair in pairs:
            for methodology in METHODOLOGIES:
                # Use diagnostic hyperparams only for our_approach
                hp = DIAG_HYPERPARAMS if methodology == "our_approach" else BASELINE_HYPERPARAMS
                for s in range(1, NUM_SLICES + 1):
                    configs.append(_build_config(
                        methodology=methodology,
                        scenario_name=scenario_name,
                        constrained_class=scenario_def["constrained_class"],
                        constraint_pair=pair,
                        model_name=model_name,
                        slice_idx=s,
                        hp=hp,
                    ))

    print("=" * 70)
    print("DIAGNOSTIC EXPERIMENT GRID")
    print("=" * 70)
    print(f"  Model: {model_name}")
    print(f"  Dataset: tissuemnist")
    print(f"  Scenarios: {list(SCENARIOS.keys())}")
    for sc, pairs in CONSTRAINT_GRID.items():
        print(f"    {sc}: {[constraint_tag(p) for p in pairs]}")
    print(f"  Methods: {METHODOLOGIES}")
    print(f"  Total: {len(configs)} configs")
    print(f"  Diagnostics: level=2 for our_approach, level=0 for baselines")
    print()

    # Breakdown
    our_count = sum(1 for c in configs if c['methodology'] == 'our_approach')
    base_count = len(configs) - our_count
    print(f"  our_approach with diagnostics: {our_count}")
    print(f"  baselines (heuristic + danits_lp): {base_count}")
    print()

    save_configs(configs, output_dir="results/pending_runs")

    hashes = {c["base_model_id"] for c in configs}
    print(f"  Warmup cache hashes: {sorted(hashes)}")


if __name__ == "__main__":
    main()
