"""
Controlled experiment grid to test whether our_approach can improve over heuristic.

Design principles addressing user's concerns:
  1. Argmax should be strong enough that there's room to improve
  2. Constraints should force meaningful redistribution (argmax must violate)
  3. Enough constrained classes that greedy != optimal (LP has something to do)
  4. Full diagnostics (level=2) so we can see gradient dynamics

Test matrix:
  * 1 model: MobileNetV3 (weakest/cheapest, matches March 27 best results)
  * 1 dataset: tissuemnist slice_1 (8 classes)
  * 3 scenarios with increasing # constrained classes:
      - 2class_tight: [CST=2, GE=4] with L30_G30  (baseline, we've seen this is ~flat)
      - 4class_medium: [CDS=1, CST=2, GE=4, PTC=5] with L50_G50
      - 4class_tight:  [CDS=1, CST=2, GE=4, PTC=5] with L30_G30
  * 3 methods: heuristic, danits_lp, our_approach

Total: 3 scenarios * 3 methods = 9 configs

Hyperparams: March 27 proven values + diagnostics enabled:
  alpha_kl=0.1, lr_constraint=5e-6, seed=42,
  disable_ce_skip=True, disable_lambda_toggle=True,
  diagnostic_level=2 (for our_approach)
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

DEFAULT_MODEL = "MobileNetV3"
NUM_SLICES = 1

TISSUEMNIST_CLASSES = {
    "CDI": 0, "CDS": 1, "CST": 2, "EPI": 3,
    "GE": 4, "PTC": 5, "STR": 6, "TUB": 7,
}

SCENARIOS = {
    # Baseline: 2 constrained classes (we've seen this is flat)
    "imp_2class": {
        "constrained_class": [TISSUEMNIST_CLASSES["CST"], TISSUEMNIST_CLASSES["GE"]],
        "pairs": [(0.3, 0.3), (0.5, 0.5)],
    },
    # 4 constrained classes at medium tightness (sweep shows +0.13pp for LP)
    "imp_4class_med": {
        "constrained_class": [
            TISSUEMNIST_CLASSES["CDS"],
            TISSUEMNIST_CLASSES["CST"],
            TISSUEMNIST_CLASSES["GE"],
            TISSUEMNIST_CLASSES["PTC"],
        ],
        "pairs": [(0.5, 0.5)],
    },
    # 4 constrained classes at tight (sweep showed LP +0.25pp here)
    "imp_4class_tight": {
        "constrained_class": [
            TISSUEMNIST_CLASSES["CDS"],
            TISSUEMNIST_CLASSES["CST"],
            TISSUEMNIST_CLASSES["GE"],
            TISSUEMNIST_CLASSES["PTC"],
        ],
        "pairs": [(0.3, 0.3)],
    },
}

METHODOLOGIES = ["heuristic", "our_approach", "danits_lp"]

DIAG_HYPERPARAMS = {
    **HYPERPARAMS,
    "alpha_kl": 0.1,
    "lr_constraint": 5e-6,
    "disable_ce_skip": True,
    "disable_lambda_toggle": True,
    "seed": 42,
    "diagnostic_level": 2,
}

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
    exp_name = f"imp_{scenario_name}_{ctag}_{model_name}_{methodology}_slice{slice_idx}"
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

    configs = []
    print("=" * 70)
    print("IMPROVEMENT TEST GRID")
    print("=" * 70)
    print(f"Model: {args.model} | Dataset: tissuemnist slice_1")
    print()

    for scenario_name, scenario_def in SCENARIOS.items():
        print(f"  {scenario_name}: constrained={scenario_def['constrained_class']}")
        for pair in scenario_def["pairs"]:
            print(f"    {constraint_tag(pair)}")
            for methodology in METHODOLOGIES:
                hp = DIAG_HYPERPARAMS if methodology == "our_approach" else BASELINE_HYPERPARAMS
                configs.append(_build_config(
                    methodology=methodology,
                    scenario_name=scenario_name,
                    constrained_class=scenario_def["constrained_class"],
                    constraint_pair=pair,
                    model_name=args.model,
                    slice_idx=1,
                    hp=hp,
                ))
    print()
    print(f"Total: {len(configs)} configs")
    save_configs(configs, output_dir="results/pending_runs")

    hashes = {c["base_model_id"] for c in configs}
    print(f"Warmup cache hashes: {sorted(hashes)}")


if __name__ == "__main__":
    main()
