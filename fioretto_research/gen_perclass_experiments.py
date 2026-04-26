"""Generate per_class_ratchet experiments to compare against Fioretto LDF.

Same scenarios, seeds, datasets as the Fioretto benchmark.
Uses our_approach with lambda_mode='per_class_ratchet' — each constrained class
gets its own lambda that increments independently.

Usage:
    python -m fioretto_research.gen_perclass_experiments
"""

from __future__ import annotations
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

ROOT = "results/pending_runs/perclass_benchmark"

# ---------- TissueMNIST ----------

TISSUE_MODEL = "MobileNetV3"
TISSUE_DATA_DIR = "data/tissuemnist/slice_1"

TISSUE_SCENARIOS = {
    "single_GE":          {"cc": 4},
    "dual_GE_CST":        {"cc": [4, 2]},
    "triple_GE_CST_PTC":  {"cc": [4, 2, 5]},
    "quad_rare":           {"cc": [4, 2, 5, 7]},
}

TISSUE_HP = {
    "lr": 0.0001, "lr_constraint": 5e-06, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
    "lambda_global": 0.01, "lambda_local": 0.01, "lambda_step": 0.002,
    "use_sum_loss": True, "initial_rho": 5.0, "rho_target": 100.0,
    "alpha_kl": 0.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "lambda_mode": "per_class_ratchet",
}

# ---------- CIFAR-100 ----------

CIFAR_MODEL = "MobileNetV3"
CIFAR_DATA_DIR = "data/cifar100/slice_1"
CIFAR_CONSTRAINT_PAIR = (0.3, 0.5)

CIFAR_SCENARIOS = {
    "single_apple":       {"cc": 0},
    "dual_apple_orange":  {"cc": [0, 53]},
    "dual_cross_super":   {"cc": [8, 48]},
    "triple_vehicles":    {"cc": [8, 13, 48]},
}

CIFAR_HP = {
    "lr": 0.0001, "lr_constraint": 5e-06, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
    "lambda_global": 0.01, "lambda_local": 0.01, "lambda_step": 0.002,
    "use_sum_loss": True, "initial_rho": 5.0, "rho_target": 100.0,
    "alpha_kl": 0.1, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 64,
    "lambda_mode": "per_class_ratchet",
}

CONSTRAINT_PAIR = (0.3, 0.5)  # L30_G50
SEEDS = [1, 2, 3]


def _tissue_ds(cc):
    return {
        "target_column": "label",
        "group_column": "synth_group",
        "num_classes": 8,
        "image_size": 224,
        "data_dir": TISSUE_DATA_DIR,
        "constrained_class": cc,
    }


def _cifar_ds(cc):
    return {
        "target_column": "label",
        "group_column": "coarse_label",
        "num_classes": 100,
        "image_size": 224,
        "data_dir": CIFAR_DATA_DIR,
        "constrained_class": cc,
    }


def _build_tissue(scenario_name, seed):
    hp = dict(TISSUE_HP)
    hp["seed"] = seed
    cc = TISSUE_SCENARIOS[scenario_name]["cc"]
    ctag = constraint_tag(CONSTRAINT_PAIR)
    variant = f"s{seed}"
    path = Path(ROOT) / "tissuemnist" / scenario_name / ctag / TISSUE_MODEL / "our_approach_pcr" / variant
    return {
        "methodology": "our_approach",
        "model_name": TISSUE_MODEL,
        "constraint": list(CONSTRAINT_PAIR),
        "constraint_tag": ctag,
        "dataset_mode": "tissuemnist",
        "dataset_config": _tissue_ds(cc),
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(
            TISSUE_MODEL, hp, dataset_mode="tissuemnist", data_dir=TISSUE_DATA_DIR),
        "exp_name": f"pcr_tissue_{scenario_name}_{ctag}_s{seed}",
        "status": "pending",
        "experiment_path": str(path),
    }


def _build_cifar(scenario_name, seed):
    hp = dict(CIFAR_HP)
    hp["seed"] = seed
    cc = CIFAR_SCENARIOS[scenario_name]["cc"]
    ctag = constraint_tag(CIFAR_CONSTRAINT_PAIR)
    variant = f"s{seed}"
    path = Path(ROOT) / "cifar100" / scenario_name / ctag / CIFAR_MODEL / "our_approach_pcr" / variant
    return {
        "methodology": "our_approach",
        "model_name": CIFAR_MODEL,
        "constraint": list(CIFAR_CONSTRAINT_PAIR),
        "constraint_tag": ctag,
        "dataset_mode": "cifar100",
        "dataset_config": _cifar_ds(cc),
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(
            CIFAR_MODEL, hp, dataset_mode="cifar100", data_dir=CIFAR_DATA_DIR),
        "exp_name": f"pcr_c100_{scenario_name}_{ctag}_s{seed}",
        "status": "pending",
        "experiment_path": str(path),
    }


def main():
    cfgs = []
    for sc in TISSUE_SCENARIOS:
        for seed in SEEDS:
            cfgs.append(_build_tissue(sc, seed))
    for sc in CIFAR_SCENARIOS:
        for seed in SEEDS:
            cfgs.append(_build_cifar(sc, seed))

    n_tissue = sum(1 for c in cfgs if c["dataset_mode"] == "tissuemnist")
    n_cifar = sum(1 for c in cfgs if c["dataset_mode"] == "cifar100")

    print("=" * 70)
    print("PER-CLASS RATCHET BENCHMARK EXPERIMENTS")
    print("=" * 70)
    print(f"TissueMNIST: {n_tissue} ({len(TISSUE_SCENARIOS)} scenarios x {len(SEEDS)} seeds)")
    print(f"CIFAR-100:   {n_cifar} ({len(CIFAR_SCENARIOS)} scenarios x {len(SEEDS)} seeds)")
    print(f"Total:       {len(cfgs)}")
    print(f"lambda_mode: per_class_ratchet")
    print(f"Seeds: {SEEDS}")

    hashes = sorted({c["base_model_id"] for c in cfgs})
    print(f"Warmup hashes: {len(hashes)} (shared with existing experiments)")

    save_configs(cfgs, output_dir=ROOT)
    print(f"\nConfigs saved to {ROOT}/")


if __name__ == "__main__":
    main()
