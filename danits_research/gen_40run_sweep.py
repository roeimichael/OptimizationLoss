"""40-run HP sweep on MobileNetV3 + tissuemnist + single_GE.

Two constraint tiers: L20_G80 (easy) + L50_G50 (mid). 20 runs each.

Blocks (per tier):
  A: lambda mode x step grid           (6)  -- mode in {ratchet, proportional}, step in {0.001, 0.002, 0.005}
  B: initial_rho non-baseline          (2)  -- 25, 100 (baseline=5)
  C: alpha_kl non-baseline             (2)  -- 0.0, 0.5 (baseline=0.1)
  D: lr_constraint non-baseline        (2)  -- 1e-6, 2e-5 (baseline=5e-6)
  E: ratchet baseline replicates       (5)  -- seeds 0..4
  F: proportional baseline replicates  (3)  -- seeds 0..2

Plus baselines: heuristic + danits_lp per tier (4 runs total).

All our_approach runs use diagnostic_level=2.
Expected: ~40 * 20 min on Blackwell = ~3.5h across 4 GPUs parallel.
"""

from __future__ import annotations
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

MODEL = "MobileNetV3"
DATA_DIR = "data/tissuemnist/slice_1"
CONSTRAINED_CLASS = 4  # GE
SCENARIO = "sweep40_single_GE"
TIERS = [(0.2, 0.8), (0.5, 0.5)]  # L20_G80, L50_G50

# March 27 baseline HP -- all blocks vary ONE axis from this.
BASELINE_HP = {
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
    "lambda_mode": "ratchet",
    "diagnostic_level": 2,
}


def _build(methodology, tier, variant, hp_overrides=None):
    hp = dict(BASELINE_HP)
    if methodology != "our_approach":
        hp.pop("diagnostic_level", None)
        hp.pop("lambda_mode", None)
    if hp_overrides:
        hp.update(hp_overrides)

    ctag = constraint_tag(tier)
    scenario_tier = f"{SCENARIO}/{ctag}"
    path = Path("results/pending_runs") / SCENARIO / ctag / MODEL / methodology / variant
    ds = {
        "target_column": "label",
        "group_column": "synth_group",
        "num_classes": 8,
        "image_size": 224,
        "data_dir": DATA_DIR,
        "constrained_class": CONSTRAINED_CLASS,
    }
    return {
        "methodology": methodology,
        "model_name": MODEL,
        "constraint": list(tier),
        "constraint_tag": ctag,
        "dataset_mode": "tissuemnist",
        "dataset_config": ds,
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(
            MODEL, hp, dataset_mode="tissuemnist", data_dir=DATA_DIR),
        "exp_name": f"sweep40_{ctag}_{methodology}_{variant}",
        "status": "pending",
        "experiment_path": str(path),
    }


def _tier_configs(tier):
    configs = []

    # Block A: lambda mode x step (6)
    for mode in ["ratchet", "proportional"]:
        for step in [0.001, 0.002, 0.005]:
            ov = {"lambda_mode": mode, "lambda_step": step, "seed": 0}
            if mode == "proportional":
                ov.update({"lambda_max": 0.2, "lambda_k": 20.0, "lambda_ema_alpha": 0.2})
            configs.append(_build("our_approach", tier,
                                  f"A_mode-{mode}_step-{step}_s0", ov))

    # Block B: initial_rho (2)
    for rho in [25.0, 100.0]:
        configs.append(_build("our_approach", tier,
                              f"B_rho-{int(rho)}_s0",
                              {"initial_rho": rho, "seed": 0}))

    # Block C: alpha_kl (2)
    for akl in [0.0, 0.5]:
        configs.append(_build("our_approach", tier,
                              f"C_akl-{akl}_s0",
                              {"alpha_kl": akl, "seed": 0}))

    # Block D: lr_constraint (2)
    for lrc in [1e-6, 2e-5]:
        configs.append(_build("our_approach", tier,
                              f"D_lrc-{lrc}_s0",
                              {"lr_constraint": lrc, "seed": 0}))

    # Block E: ratchet baseline replicates (5)
    # Block A already ran ratchet+step=0.002+seed=0. Use seeds 1..5 here for 5 fresh replicates.
    for s in range(1, 6):
        configs.append(_build("our_approach", tier,
                              f"E_ratchet_s{s}",
                              {"seed": s}))

    # Block F: proportional baseline replicates (3)
    # Block A already ran proportional+step=0.002+seed=0. Use seeds 1..3 for 3 fresh replicates.
    for s in range(1, 4):
        configs.append(_build("our_approach", tier,
                              f"F_prop_s{s}",
                              {
                                  "lambda_mode": "proportional",
                                  "lambda_max": 0.2,
                                  "lambda_k": 20.0,
                                  "lambda_ema_alpha": 0.2,
                                  "seed": s,
                              }))

    # Baselines: heuristic + danits_lp, seed=0.
    for meth in ["heuristic", "danits_lp"]:
        configs.append(_build(meth, tier, "slice_1", {"seed": 0}))

    return configs


def main():
    all_configs = []
    for tier in TIERS:
        all_configs.extend(_tier_configs(tier))

    n_oa = sum(1 for c in all_configs if c["methodology"] == "our_approach")
    n_base = len(all_configs) - n_oa

    print("=" * 70)
    print("40-RUN SWEEP")
    print("=" * 70)
    print(f"Tiers: {[constraint_tag(t) for t in TIERS]}")
    print(f"our_approach configs: {n_oa}")
    print(f"baseline configs: {n_base}")
    print(f"total: {len(all_configs)}")

    hashes = sorted({c["base_model_id"] for c in all_configs})
    print(f"unique warmup cache hashes: {len(hashes)}")
    for h in hashes:
        print(f"  {h}")

    save_configs(all_configs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
