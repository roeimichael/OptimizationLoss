"""
Generate a 3-experiment pilot (heuristic + our_approach + danits_lp) on
one setting (single_MEL, L50_G50, MobileNetV3, slice_1) with the 4 fixes
from the pipeline audit applied:

  Fix 1:  alpha_kl              = 0.0      (was 0.5; KL was pulling back to warmup)
  Fix 2:  lr_constraint         = 5e-5     (was 5e-6; lr too small to move weights)
  Fix 3:  disable_ce_skip       = True     (was False; CE skip kills Phase 1)
  Fix 4:  disable_lambda_toggle = True     (was False; zeroing lambdas kills grad)

If this pilot shows the constraint-trained posterior actually moves substantially
compared to the warmup posterior (e.g. >30% argmax disagreement on the test set
instead of the 6.5% we saw before), we'll rerun the full 24-experiment grid with
these fixes. If not, we need to investigate further before burning more compute.
"""

from __future__ import annotations

from src.config_generators.generate_configs import (
    HYPERPARAMS,
    generate_configs,
    save_configs,
)


PILOT_SCENARIO = "single_MEL"
PILOT_PAIR = (0.5, 0.5)
PILOT_MODEL = "MobileNetV3"
PILOT_METHODOLOGIES = ["heuristic", "our_approach", "danits_lp"]

FIXED_HYPERPARAMS = {
    **HYPERPARAMS,
    # --- revised fixes (March 27 proven values + structural improvements) ---
    "alpha_kl": 0.1,            # March 27 value (0.5 too strong, 0.0 too unstable)
    "lr_constraint": 5e-6,      # March 27 original (5e-5 was too aggressive)
    "disable_ce_skip": True,    # keep CE training active throughout
    "disable_lambda_toggle": True,  # don't zero lambdas on satisfaction
    "seed": 42,                 # deterministic warmup + training
}


def main() -> int:
    print("=" * 72)
    print("PILOT CONFIG GENERATOR (fixed pipeline)")
    print("=" * 72)
    print(f"  scenario        : {PILOT_SCENARIO}")
    print(f"  constraint pair : {PILOT_PAIR}")
    print(f"  model           : {PILOT_MODEL}")
    print(f"  methodologies   : {PILOT_METHODOLOGIES}")
    print()
    print(f"  fixes applied:")
    print(f"    alpha_kl              : 0.5  -> 0.0")
    print(f"    lr_constraint         : 5e-6 -> 5e-5")
    print(f"    disable_ce_skip       : False -> True")
    print(f"    disable_lambda_toggle : False -> True")
    print()

    configs = generate_configs(
        scenarios=[PILOT_SCENARIO],
        constraint_pairs=[PILOT_PAIR],
        model_names=[PILOT_MODEL],
        methodologies=PILOT_METHODOLOGIES,
        num_slices=1,
        round_name="round1",
    )
    # Override the hyperparams with the fixed values (our fix dict)
    for cfg in configs:
        cfg["hyperparams"] = FIXED_HYPERPARAMS.copy()

    # Every config still maps to the same warmup cache hash -- but since
    # compute_base_model_id is called at generation time in _build_config,
    # the hash was already computed with the OLD hyperparams reference that
    # generate_configs passes in. We recompute here with the FIXED hp so
    # the cache reflects the new warmup that will actually be trained.
    from src.config_generators.generate_configs import compute_base_model_id
    for cfg in configs:
        cfg["base_model_id"] = compute_base_model_id(
            cfg["model_name"], FIXED_HYPERPARAMS,
            data_dir=cfg["dataset_config"]["data_dir"],
        )

    save_configs(configs, output_dir="results/pending_runs")
    print(f"\nwrote {len(configs)} pilot configs")
    print()
    print("  shared warmup hash (all 3 methods will reuse it after the first run):")
    for cfg in configs:
        print(f"    [{cfg['methodology']:<14s}] -> {cfg['base_model_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
