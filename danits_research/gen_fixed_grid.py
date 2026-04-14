"""
Full 24-experiment grid generator with the 4 pipeline fixes applied.

Mirrors `gen_smoke_configs.py` (same scenarios, same constraint pairs,
same model, same slice), but overrides hyperparams with the fixed values
from the pipeline audit:

    alpha_kl              0.5  -> 0.0       (KL was pulling back to warmup)
    lr_constraint         5e-6 -> 5e-5      (previous LR too small)
    disable_ce_skip       False -> True     (CE skip killed Phase 1)
    disable_lambda_toggle False -> True     (zeroing lambdas killed grad)

The warmup cache (compute_base_model_id) does NOT depend on these fields,
so this grid shares the same warmup hash as the pilot --- meaning the
cached warmup from the pilot run will be reused here. One warmup trained,
24 experiments run on top.

Run after the pilot succeeds:
    python -m danits_research.gen_fixed_grid
    ./run_experiments.sh
"""

from __future__ import annotations

import argparse

from src.config_generators.generate_configs import (
    HYPERPARAMS,
    compute_base_model_id,
    generate_configs,
    save_configs,
)

# Default backbone -- overridable via `--model`. Any entry in
# src.config_generators.generate_configs.MODEL_NAMES works:
# MobileNetV3, EfficientNetB0, ConvNeXtTiny, ResNet18.
DEFAULT_MODEL = "MobileNetV3"
SINGLE_SCENARIO = "single_MEL"
MULTI_SCENARIO = "multi_MEL_BKL"

SINGLE_PAIRS = [
    (0.3, 0.3),   # tight equal
    (0.5, 0.5),   # medium equal  (pilot setting)
    (0.8, 0.8),   # loose equal
    (0.3, 0.8),   # Phi binds
    (0.8, 0.3),   # Psi binds
]

MULTI_PAIRS = [
    (0.5, 0.5),   # medium equal, 2 classes
    (0.3, 0.8),   # Phi binds, 2 classes
    (0.8, 0.3),   # Psi binds, 2 classes
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Backbone model name (default: %(default)s)")
    args = parser.parse_args()
    model_name = args.model

    print("=" * 72)
    print("FIXED-GRID GENERATOR (24 experiments, all audit fixes applied)")
    print("=" * 72)
    print(f"  model         : {model_name}")
    print(f"  single pairs  : {[f'L{int(l*100):02d}_G{int(g*100):02d}' for l,g in SINGLE_PAIRS]}")
    print(f"  multi pairs   : {[f'L{int(l*100):02d}_G{int(g*100):02d}' for l,g in MULTI_PAIRS]}")
    print(f"  methodologies : {METHODOLOGIES}")
    print()
    print("  fixes vs corrupted run:")
    print("    alpha_kl              0.5  -> 0.0")
    print("    lr_constraint         5e-6 -> 5e-5")
    print("    disable_ce_skip       False -> True")
    print("    disable_lambda_toggle False -> True")
    print()

    configs_single = generate_configs(
        scenarios=[SINGLE_SCENARIO],
        constraint_pairs=SINGLE_PAIRS,
        model_names=[model_name],
        methodologies=METHODOLOGIES,
        num_slices=1,
        round_name="round1",
    )
    configs_multi = generate_configs(
        scenarios=[MULTI_SCENARIO],
        constraint_pairs=MULTI_PAIRS,
        model_names=[model_name],
        methodologies=METHODOLOGIES,
        num_slices=1,
        round_name="round1",
    )
    all_configs = configs_single + configs_multi

    # Apply fixed hyperparams + recompute base_model_id
    for cfg in all_configs:
        cfg["hyperparams"] = FIXED_HYPERPARAMS.copy()
        cfg["base_model_id"] = compute_base_model_id(
            cfg["model_name"],
            FIXED_HYPERPARAMS,
            data_dir=cfg["dataset_config"]["data_dir"],
        )

    save_configs(all_configs, output_dir="results/pending_runs")

    print(f"  created {len(all_configs)} configs")
    print(f"    single_MEL     x {len(SINGLE_PAIRS)} pairs x {len(METHODOLOGIES)} methods "
          f"= {len(configs_single)}")
    print(f"    multi_MEL_BKL  x {len(MULTI_PAIRS)} pairs x {len(METHODOLOGIES)} methods "
          f"= {len(configs_multi)}")
    print()
    print("  shared warmup cache hash:")
    hashes = {c["base_model_id"] for c in all_configs}
    for h in sorted(hashes):
        print(f"    {h}")
    print()
    print("  -> ready: ./run_experiments.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
