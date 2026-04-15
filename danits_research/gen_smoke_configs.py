"""
Generate the 'smoke' experiment grid for a fresh-from-scratch training run.

Single model (MobileNetV3), single slice (slice_1), DermMNIST.

Two scenarios:
    single_MEL        -- 1 constrained class (class 4)
    multi_MEL_BKL     -- 2 constrained classes (classes 4 and 2)

Constraint pairs (5 for single, 3 for multi):
    single:  (0.3, 0.3), (0.5, 0.5), (0.8, 0.8), (0.3, 0.8), (0.8, 0.3)
    multi:   (0.5, 0.5), (0.3, 0.8), (0.8, 0.3)

Methodologies:
    heuristic    -- warmup + project's apply_allocation_heuristic
    our_approach -- warmup + constraint-aware training + targeted_correction
    danits_lp    -- warmup + paper [5] LP (our implementation)

Because `heuristic`, `danits_lp` and `our_approach` share the same
base_model_id hash (it depends only on model + warmup hyperparameters +
data_dir), the warmup model is trained ONCE and reused for every Phase-2
methodology on the same slice. This is the main reason a 24-experiment
grid is only expensive on constraint training, not on warmup.

Usage:
    python -m danits_research.gen_smoke_configs
"""

from __future__ import annotations

from pathlib import Path

from src.config_generators.generate_configs import (
    HYPERPARAMS,
    ROUND1_SCENARIOS,
    constraint_tag,
    generate_configs,
    save_configs,
)

# ---- grid definition ------------------------------------------------

MODEL_NAME = "MobileNetV3"

SINGLE_SCENARIO = "single_MEL"          # class 4
MULTI_SCENARIO = "multi_MEL_BKL"        # classes [4, 2]

SINGLE_PAIRS = [
    (0.3, 0.3),   # tight equal
    (0.5, 0.5),   # medium equal
    (0.8, 0.8),   # loose equal
    (0.3, 0.8),   # local tight, global loose (Phi should bind)
    (0.8, 0.3),   # local loose, global tight (Psi should bind)
]

MULTI_PAIRS = [
    (0.5, 0.5),   # medium equal (2 classes)
    (0.3, 0.8),   # local tight, global loose (Phi binds across 2 classes)
    (0.8, 0.3),   # local loose, global tight (Psi binds across 2 classes)
]

METHODOLOGIES = ["heuristic", "our_approach", "danits_lp"]

NUM_SLICES = 1


def main() -> int:
    # ---- sanity: the scenarios we use must be in round1's table -----
    assert SINGLE_SCENARIO in ROUND1_SCENARIOS, SINGLE_SCENARIO
    assert MULTI_SCENARIO in ROUND1_SCENARIOS, MULTI_SCENARIO

    print(f"Generating smoke grid:")
    print(f"  model          : {MODEL_NAME}")
    print(f"  dataset        : dermmnist  (slice_{NUM_SLICES})")
    print(f"  single scenario: {SINGLE_SCENARIO} "
          f"-> class {ROUND1_SCENARIOS[SINGLE_SCENARIO]['constrained_class']}")
    print(f"  multi scenario : {MULTI_SCENARIO} "
          f"-> classes {ROUND1_SCENARIOS[MULTI_SCENARIO]['constrained_class']}")
    print(f"  single pairs   : "
          + ", ".join(constraint_tag(p) for p in SINGLE_PAIRS))
    print(f"  multi pairs    : "
          + ", ".join(constraint_tag(p) for p in MULTI_PAIRS))
    print(f"  methodologies  : {METHODOLOGIES}")
    print()

    # We call generate_configs twice so we can have different constraint
    # pair sets per scenario (single gets 5, multi gets 3).
    configs_single = generate_configs(
        scenarios=[SINGLE_SCENARIO],
        constraint_pairs=SINGLE_PAIRS,
        model_names=[MODEL_NAME],
        methodologies=METHODOLOGIES,
        num_slices=NUM_SLICES,
        round_name="round1",
    )
    configs_multi = generate_configs(
        scenarios=[MULTI_SCENARIO],
        constraint_pairs=MULTI_PAIRS,
        model_names=[MODEL_NAME],
        methodologies=METHODOLOGIES,
        num_slices=NUM_SLICES,
        round_name="round1",
    )
    all_configs = configs_single + configs_multi

    # Sanity: every single_MEL experiment should share the SAME base_model_id
    # (because base_model_id depends only on model+hp+data_dir, not constraint
    # or methodology). Same for multi_MEL_BKL.
    single_ids = {c["base_model_id"] for c in configs_single}
    multi_ids = {c["base_model_id"] for c in configs_multi}
    assert len(single_ids) == 1, f"single hashes diverged: {single_ids}"
    assert len(multi_ids) == 1, f"multi hashes diverged: {multi_ids}"
    print(f"  warmup cache hashes:")
    print(f"    single_MEL    -> {next(iter(single_ids))}")
    print(f"    multi_MEL_BKL -> {next(iter(multi_ids))}")
    # Since they use the same slice, model, and hyperparameters, they
    # should actually be the same hash:
    if single_ids == multi_ids:
        print("    (same hash -> ONE warmup model trained for the entire grid)")
    else:
        print("    (different hash -> 2 warmup models will be trained)")
    print()

    save_configs(all_configs, output_dir="results/pending_runs")

    print(f"\n  total configs created: {len(all_configs)}")
    print(f"  breakdown:")
    print(f"    single_MEL     x {len(SINGLE_PAIRS)} pairs x "
          f"{len(METHODOLOGIES)} methods = {len(configs_single)}")
    print(f"    multi_MEL_BKL  x {len(MULTI_PAIRS)} pairs x "
          f"{len(METHODOLOGIES)} methods = {len(configs_multi)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
