"""
Test excess-proportional lambda mode against March 27 ratchet baseline.

Two variants on the SAME config (tissuemnist/single_GE/L50_G50/MobileNetV3):
  - A: ratchet mode (March 27 baseline, proved 0.5546 earlier)
  - B: proportional mode with lambda_max=0.2, k=20

Plus the 2 baselines (heuristic, danits_lp) for reference.

Expected: proportional should keep grad ratio more balanced (smooth) and ideally
match or beat ratchet's 0.5546.
"""

from __future__ import annotations
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

MARCH27_HP = {
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
}

MODEL = "MobileNetV3"
DATA_DIR = "data/tissuemnist/slice_1"
PAIR = (0.5, 0.5)  # L50_G50


def build(methodology, scenario, hp_extras=None):
    hp = dict(MARCH27_HP)
    if hp_extras:
        hp.update(hp_extras)
    ctag = constraint_tag(PAIR)
    tag = scenario
    path = Path("results/pending_runs") / tag / ctag / MODEL / methodology / "slice_1"
    ds = {
        "target_column": "label",
        "group_column": "synth_group",
        "num_classes": 8,
        "image_size": 224,
        "data_dir": DATA_DIR,
        "constrained_class": 4,
    }
    return {
        "methodology": methodology,
        "model_name": MODEL,
        "constraint": list(PAIR),
        "constraint_tag": ctag,
        "dataset_mode": "tissuemnist",
        "dataset_config": ds,
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(MODEL, hp, dataset_mode="tissuemnist", data_dir=DATA_DIR),
        "exp_name": f"{tag}_{ctag}_{MODEL}_{methodology}",
        "status": "pending",
        "experiment_path": str(path),
    }


def main():
    configs = []
    # A: ratchet mode (baseline — should reproduce 0.5546)
    configs.append(build("heuristic", "prop_A_ratchet"))
    configs.append(build("danits_lp", "prop_A_ratchet"))
    configs.append(build("our_approach", "prop_A_ratchet", {
        "lambda_mode": "ratchet",
        "diagnostic_level": 2,
    }))

    # B: proportional mode
    configs.append(build("heuristic", "prop_B_proportional"))
    configs.append(build("danits_lp", "prop_B_proportional"))
    configs.append(build("our_approach", "prop_B_proportional", {
        "lambda_mode": "proportional",
        "lambda_max": 0.2,
        "lambda_k": 20.0,
        "lambda_ema_alpha": 0.2,
        "diagnostic_level": 2,
    }))

    print("=" * 70)
    print("PROPORTIONAL LAMBDA TEST")
    print("=" * 70)
    print(f"Total: {len(configs)} configs")
    for c in configs:
        mode = c["hyperparams"].get("lambda_mode", "default")
        print(f"  {c['exp_name']} (mode={mode})")

    save_configs(configs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
