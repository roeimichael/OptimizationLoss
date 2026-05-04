"""Generate Fioretto LDF benchmark configs matching existing experiment scenarios.

Creates configs for fioretto_ldf methodology on both TissueMNIST and CIFAR-100,
using the same seeds/tiers/scenarios as our_approach experiments for direct comparison.

Step sizes to sweep: 0.001, 0.005, 0.01 (Fioretto's key hyperparameter).
All other HPs match the baseline configs used for our_approach/heuristic/danits_lp.

Usage:
    python -m fioretto_research.gen_fioretto_experiments
"""

from __future__ import annotations
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, constraint_tag, save_configs,
)

ROOT = "results/pending_runs/fioretto_benchmark"

# ---------- TissueMNIST ----------

TISSUE_MODEL = "MobileNetV3"
TISSUE_DATA_DIR = "data/tissuemnist/slice_1"
TISSUE_CONSTRAINT_PAIRS = [(0.3, 0.5)]  # L30_G50

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
}

STEP_SIZES = [0.001, 0.005, 0.01]
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


def _build_tissue(scenario_name, seed, step_size):
    hp = dict(TISSUE_HP)
    hp["seed"] = seed
    hp["fioretto_step_size"] = step_size
    cc = TISSUE_SCENARIOS[scenario_name]["cc"]
    cpair = TISSUE_CONSTRAINT_PAIRS[0]
    ctag = constraint_tag(cpair)
    ss_tag = f"ss{str(step_size).replace('.', '')}"
    variant = f"s{seed}_{ss_tag}"
    path = Path(ROOT) / "tissuemnist" / scenario_name / ctag / TISSUE_MODEL / "fioretto_ldf" / variant
    ds_config = _tissue_ds(cc)
    return {
        "methodology": "fioretto_ldf",
        "model_name": TISSUE_MODEL,
        "constraint": list(cpair),
        "constraint_tag": ctag,
        "dataset_mode": "tissuemnist",
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(
            TISSUE_MODEL, hp, dataset_mode="tissuemnist",
            data_dir=TISSUE_DATA_DIR, dataset_config=ds_config),
        "exp_name": f"fioretto_tissue_{scenario_name}_{ctag}_s{seed}_{ss_tag}",
        "status": "pending",
        "experiment_path": str(path),
    }


def _build_cifar(scenario_name, seed, step_size):
    hp = dict(CIFAR_HP)
    hp["seed"] = seed
    hp["fioretto_step_size"] = step_size
    cc = CIFAR_SCENARIOS[scenario_name]["cc"]
    ctag = constraint_tag(CIFAR_CONSTRAINT_PAIR)
    ss_tag = f"ss{str(step_size).replace('.', '')}"
    variant = f"s{seed}_{ss_tag}"
    path = Path(ROOT) / "cifar100" / scenario_name / ctag / CIFAR_MODEL / "fioretto_ldf" / variant
    ds_config = _cifar_ds(cc)
    return {
        "methodology": "fioretto_ldf",
        "model_name": CIFAR_MODEL,
        "constraint": list(CIFAR_CONSTRAINT_PAIR),
        "constraint_tag": ctag,
        "dataset_mode": "cifar100",
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(
            CIFAR_MODEL, hp, dataset_mode="cifar100",
            data_dir=CIFAR_DATA_DIR, dataset_config=ds_config),
        "exp_name": f"fioretto_c100_{scenario_name}_{ctag}_s{seed}_{ss_tag}",
        "status": "pending",
        "experiment_path": str(path),
    }


def main():
    cfgs = []

    # TissueMNIST
    for sc in TISSUE_SCENARIOS:
        for seed in SEEDS:
            for ss in STEP_SIZES:
                cfgs.append(_build_tissue(sc, seed, ss))

    # CIFAR-100
    for sc in CIFAR_SCENARIOS:
        for seed in SEEDS:
            for ss in STEP_SIZES:
                cfgs.append(_build_cifar(sc, seed, ss))

    n_tissue = sum(1 for c in cfgs if c["dataset_mode"] == "tissuemnist")
    n_cifar = sum(1 for c in cfgs if c["dataset_mode"] == "cifar100")

    print("=" * 70)
    print("FIORETTO LDF BENCHMARK EXPERIMENTS")
    print("=" * 70)
    print(f"TissueMNIST: {n_tissue} ({len(TISSUE_SCENARIOS)} scenarios × "
          f"{len(SEEDS)} seeds × {len(STEP_SIZES)} step_sizes)")
    print(f"CIFAR-100:   {n_cifar} ({len(CIFAR_SCENARIOS)} scenarios × "
          f"{len(SEEDS)} seeds × {len(STEP_SIZES)} step_sizes)")
    print(f"Total:       {len(cfgs)}")
    print(f"\nStep sizes: {STEP_SIZES}")
    print(f"Seeds: {SEEDS}")

    hashes = sorted({c["base_model_id"] for c in cfgs})
    print(f"Warmup hashes: {len(hashes)} (shared with existing experiments)")

    save_configs(cfgs, output_dir=ROOT)
    print(f"\nConfigs saved to {ROOT}/")


if __name__ == "__main__":
    main()
