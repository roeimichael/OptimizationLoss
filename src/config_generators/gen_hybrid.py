"""Hybrid TraLO-Fioretto sweep.

Tests whether mixing TraLO's bounded penalty with Fioretto's linear penalty
beats either alone. Two hybrid modes (single_lambda vs dual_lambda) × 3 mix
levels × 2 constraints × 2 seeds. Anchored by matched vanilla TraLO + vanilla
Fioretto baselines (same code, same seeds, same cells).

Target: tissuemnist L20_G20 + L30_G30 on MobileNetV3.
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/hybrid_v1"
DATASET = "tissuemnist"
DS_META = {
    "data_dir": "data/tissuemnist/slice_1", "num_classes": 8,
    "image_size": 224, "target_column": "label",
    "group_column": "synth_group",
}
MODEL = "MobileNetV3"
CLS = 4
SEEDS = [1, 2]
TIGHTNESS = ["L20_G20", "L30_G30"]

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 150,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}

# Shared TraLO knobs (bounded penalty same as baseline).
TRALO_KNOBS = {
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
    "penalty_mode": "both", "enable_ce_skip": True,
}

# Cell definitions. Each cell becomes (mode, fior_param_value).
CELLS = [
    # single_lambda mode: beta sweep
    ("hybrid_singleL_beta005", "tralo_fioretto",
     {**TRALO_KNOBS, "hybrid_mode": "single_lambda", "fior_beta": 0.05}),
    ("hybrid_singleL_beta020", "tralo_fioretto",
     {**TRALO_KNOBS, "hybrid_mode": "single_lambda", "fior_beta": 0.20}),
    ("hybrid_singleL_beta050", "tralo_fioretto",
     {**TRALO_KNOBS, "hybrid_mode": "single_lambda", "fior_beta": 0.50}),
    # dual_lambda mode: fior_step_size sweep
    ("hybrid_dualL_step001", "tralo_fioretto",
     {**TRALO_KNOBS, "hybrid_mode": "dual_lambda",
      "fior_beta": 0.0, "fior_step_size": 0.001, "fior_lambda_init": 0.0}),
    ("hybrid_dualL_step005", "tralo_fioretto",
     {**TRALO_KNOBS, "hybrid_mode": "dual_lambda",
      "fior_beta": 0.0, "fior_step_size": 0.005, "fior_lambda_init": 0.0}),
    ("hybrid_dualL_step020", "tralo_fioretto",
     {**TRALO_KNOBS, "hybrid_mode": "dual_lambda",
      "fior_beta": 0.0, "fior_step_size": 0.020, "fior_lambda_init": 0.0}),
    # Anchored baselines, same code state, same seeds.
    ("baseline_tralo", "tralo", TRALO_KNOBS),
    ("baseline_fior", "fioretto_ldf", {"fioretto_step_size": 0.005}),
]


def _tight_pair(tag):
    parts = tag.split("_")
    return (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)


def make_cfg(cell_name, method, method_hp, tight_tag, seed):
    hp = {**SHARED_HP, **method_hp, "seed": seed}
    ds_config = {**DS_META, "constrained_class": CLS}
    pair = _tight_pair(tight_tag)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode=DATASET,
        data_dir=DS_META["data_dir"], dataset_config=ds_config,
    )
    return {
        "methodology": method,
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": tight_tag,
        "dataset_mode": DATASET,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"{cell_name}_{tight_tag}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / tight_tag / cell_name / f"seed_{seed}"),
    }


def build():
    cfgs = []
    for tight in TIGHTNESS:
        for cell_name, method, method_hp in CELLS:
            for seed in SEEDS:
                cfgs.append(make_cfg(cell_name, method, method_hp, tight, seed))
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"\nGenerated {len(cfgs)} hybrid_v1 configs -> {SWEEP_ROOT}")


if __name__ == "__main__":
    build()
