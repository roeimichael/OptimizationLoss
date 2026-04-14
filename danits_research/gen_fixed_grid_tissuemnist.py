"""
Full 24-experiment grid generator for TissueMNIST, with all audit fixes.

Mirrors `gen_fixed_grid.py` (same number of experiments, same constraint
pair shape, same model, same slice) but points at `data/tissuemnist/`
and uses TissueMNIST-specific classes:

    single_GE        constrained_class = 4     (GE, the priority cell)
    multi_GE_CST     constrained_class = [4, 2]   (GE + CST)

8 classes total: CDI=0, CDS=1, CST=2, EPI=3, GE=4, PTC=5, STR=6, TUB=7
Group column: `synth_group` (binary 0/1)

Audit fixes applied to every config:
    alpha_kl              0.5  -> 0.0
    lr_constraint         5e-6 -> 5e-5
    disable_ce_skip       False -> True
    disable_lambda_toggle False -> True

After running this, the server's `results/pending_runs/` will contain
24 tissuemnist configs IN ADDITION to any dermmnist configs already
generated (they live under a different scenario name, so no collision).

Run on the server after the dermmnist grid is generated:
    python -m danits_research.gen_fixed_grid_tissuemnist
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

# --- grid definition ---------------------------------------------------

DEFAULT_MODEL = "MobileNetV3"   # overridable via --model
NUM_SLICES = 1                  # slice_1 only, matching gen_fixed_grid.py

SINGLE_SCENARIO = "single_GE"
MULTI_SCENARIO = "multi_GE_CST"

# TissueMNIST class indices (0-indexed, 8 classes)
TISSUEMNIST_CLASSES = {
    "CDI": 0, "CDS": 1, "CST": 2, "EPI": 3,
    "GE":  4, "PTC": 5, "STR": 6, "TUB": 7,
}

SCENARIOS = {
    SINGLE_SCENARIO: {"constrained_class": TISSUEMNIST_CLASSES["GE"]},             # 4
    MULTI_SCENARIO:  {"constrained_class": [TISSUEMNIST_CLASSES["GE"],              # 4
                                             TISSUEMNIST_CLASSES["CST"]]},          # 2
}

# Same constraint pair grid as the dermmnist version so the two grids
# are directly comparable.
SINGLE_PAIRS = [
    (0.3, 0.3),   # tight equal
    (0.5, 0.5),   # medium equal
    (0.8, 0.8),   # loose equal
    (0.3, 0.8),   # Phi tight, Psi loose
    (0.8, 0.3),   # Phi loose, Psi tight
]

MULTI_PAIRS = [
    (0.5, 0.5),
    (0.3, 0.8),
    (0.8, 0.3),
]

METHODOLOGIES = ["heuristic", "our_approach", "danits_lp"]

FIXED_HYPERPARAMS = {
    **HYPERPARAMS,
    # --- audit fixes (revised: March 27 proven values + structural fixes) ---
    "alpha_kl": 0.1,            # March 27 value (0.5 was too strong, 0.0 too unstable)
    "lr_constraint": 5e-6,      # March 27 original (5e-5 was too aggressive)
    "disable_ce_skip": True,    # keep CE training active throughout
    "disable_lambda_toggle": True,  # don't zero lambdas on satisfaction
    "seed": 42,                 # deterministic warmup + training
}


def _build_tissuemnist_config(
    methodology: str,
    scenario_name: str,
    constrained_class,
    constraint_pair: tuple[float, float],
    model_name: str,
    slice_idx: int,
    hp: dict,
) -> dict:
    """Build one pending-run config dict for a tissuemnist experiment."""
    ctag = constraint_tag(constraint_pair)
    exp_name = f"{scenario_name}_{ctag}_{model_name}_{methodology}_slice{slice_idx}"
    data_dir = f"data/tissuemnist/slice_{slice_idx}"
    ds_config = {
        "target_column": "label",
        "group_column":  "synth_group",      # tissuemnist-specific
        "num_classes":   8,                   # tissuemnist-specific
        "image_size":    224,
        "data_dir":      data_dir,
        "constrained_class": constrained_class,
    }
    path = (Path("results") / "pending_runs" / scenario_name / ctag
            / model_name / methodology / f"slice_{slice_idx}")
    return {
        "methodology":     methodology,
        "model_name":      model_name,
        "constraint":      list(constraint_pair),
        "constraint_tag":  ctag,
        "dataset_mode":    "tissuemnist",
        "dataset_config":  ds_config,
        "hyperparams":     hp.copy(),
        # base_model_id depends on dataset_mode + data_dir, so this will
        # NOT collide with any dermmnist cache hash. Two distinct warmups
        # (one per dataset) trained once, shared across all 24 dermmnist
        # configs and all 24 tissuemnist configs respectively.
        "base_model_id":   compute_base_model_id(
            model_name, hp,
            dataset_mode="tissuemnist",
            data_dir=data_dir,
        ),
        "exp_name":        exp_name,
        "status":          "pending",
        "experiment_path": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Backbone model name (default: %(default)s)")
    args = parser.parse_args()
    model_name = args.model

    print("=" * 76)
    print("FIXED TISSUEMNIST GRID (24 configs, all audit fixes applied)")
    print("=" * 76)
    print(f"  model         : {model_name}")
    print(f"  dataset       : tissuemnist, slice_{NUM_SLICES}")
    print(f"  scenarios     : {list(SCENARIOS.keys())}")
    print(f"    {SINGLE_SCENARIO} -> class {SCENARIOS[SINGLE_SCENARIO]['constrained_class']} (GE)")
    print(f"    {MULTI_SCENARIO}  -> classes {SCENARIOS[MULTI_SCENARIO]['constrained_class']} (GE, CST)")
    print(f"  single pairs  : {[constraint_tag(p) for p in SINGLE_PAIRS]}")
    print(f"  multi pairs   : {[constraint_tag(p) for p in MULTI_PAIRS]}")
    print(f"  methodologies : {METHODOLOGIES}")
    print()
    print("  fixes vs corrupted run:")
    print("    alpha_kl              0.5  -> 0.0")
    print("    lr_constraint         5e-6 -> 5e-5")
    print("    disable_ce_skip       False -> True")
    print("    disable_lambda_toggle False -> True")
    print()

    configs: list[dict] = []

    for pair in SINGLE_PAIRS:
        for methodology in METHODOLOGIES:
            for slice_idx in range(1, NUM_SLICES + 1):
                configs.append(_build_tissuemnist_config(
                    methodology=methodology,
                    scenario_name=SINGLE_SCENARIO,
                    constrained_class=SCENARIOS[SINGLE_SCENARIO]["constrained_class"],
                    constraint_pair=pair,
                    model_name=model_name,
                    slice_idx=slice_idx,
                    hp=FIXED_HYPERPARAMS,
                ))

    for pair in MULTI_PAIRS:
        for methodology in METHODOLOGIES:
            for slice_idx in range(1, NUM_SLICES + 1):
                configs.append(_build_tissuemnist_config(
                    methodology=methodology,
                    scenario_name=MULTI_SCENARIO,
                    constrained_class=SCENARIOS[MULTI_SCENARIO]["constrained_class"],
                    constraint_pair=pair,
                    model_name=model_name,
                    slice_idx=slice_idx,
                    hp=FIXED_HYPERPARAMS,
                ))

    save_configs(configs, output_dir="results/pending_runs")

    print(f"  created {len(configs)} tissuemnist configs")
    print(f"    {SINGLE_SCENARIO}   x {len(SINGLE_PAIRS)} pairs x "
          f"{len(METHODOLOGIES)} methods = "
          f"{len(SINGLE_PAIRS) * len(METHODOLOGIES) * NUM_SLICES}")
    print(f"    {MULTI_SCENARIO} x {len(MULTI_PAIRS)} pairs x "
          f"{len(METHODOLOGIES)} methods = "
          f"{len(MULTI_PAIRS) * len(METHODOLOGIES) * NUM_SLICES}")
    print()
    hashes = {c["base_model_id"] for c in configs}
    print("  shared tissuemnist warmup cache hash(es):")
    for h in sorted(hashes):
        print(f"    {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
