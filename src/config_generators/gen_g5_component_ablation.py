"""G5: leave-one-out component ablation backing main.tex lines 305/450/455.

Replaces the older gen_component_ablation.py grid (which used the dropped
eurosat dataset) with the 3 currently-active datasets and proper per-
dataset constrained_class + group_column wiring.

Sweep:
  3 datasets (tissue, derm, aider) x 1 tightness (L30_G30) x
  7 variants (full + 6 leave-one-out) x 3 seeds = 63 cells.

Output: results/pending_runs/g5_component_ablation/
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/g5_component_ablation"

# Per-dataset metadata (cls, group_column matches headline experiments)
DATASETS = {
    "tissuemnist": {
        "data_dir": "data/tissuemnist/slice_1", "num_classes": 8,
        "image_size": 224, "target_column": "label",
        "group_column": "synth_group", "constrained_class": 4,
    },
    "dermmnist": {
        "data_dir": "data/dermmnist/slice_1", "num_classes": 7,
        "image_size": 224, "target_column": "label",
        "group_column": "loc_group", "constrained_class": 4,
    },
    "aider": {
        "data_dir": "data/aider/slice_1", "num_classes": 4,
        "image_size": 224, "target_column": "label",
        "group_column": "synth_group", "constrained_class": 0,
    },
}
TIGHTNESS = ["L30_G30"]
SEEDS = [1, 2, 3]
MODEL = "MobileNetV3"

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}

FULL_TRALO = {
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
    "penalty_mode": "both", "enable_ce_skip": True,
    "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
    "reset_optimizer_at_sat": True,
    "disable_freeze_on_satisfy": False,
}

VARIANTS = {
    "full":         {},
    "no_hinge":     {"hybrid_mode": "bounded_only"},
    "no_reset":     {"reset_optimizer_at_sat": False},
    "no_freeze":    {"disable_freeze_on_satisfy": True},
    "no_ce_skip":   {"enable_ce_skip": False},
    "no_rho_sched": {"rho_target": 5.0},
    "no_warmup":    {"warmup_epochs": 0},
}


def _tight_pair(tag):
    parts = tag.split("_")
    return (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)


def make_cfg(variant_name, overrides, dataset, tight_tag, seed):
    ds_meta = DATASETS[dataset]
    cls = ds_meta["constrained_class"]
    hp = {**SHARED_HP, **FULL_TRALO, **overrides, "seed": seed}
    ds_config = {k: v for k, v in ds_meta.items() if k != "constrained_class"}
    ds_config["constrained_class"] = cls
    pair = _tight_pair(tight_tag)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode=dataset,
        data_dir=ds_meta["data_dir"], dataset_config=ds_config,
    )
    return {
        "methodology": "tralo",
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": tight_tag,
        "dataset_mode": dataset,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"g5_{variant_name}_{dataset}_{tight_tag}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / dataset / tight_tag / variant_name / f"seed_{seed}"),
    }


def build():
    cfgs = []
    for dataset in DATASETS:
        for tight in TIGHTNESS:
            for variant_name, overrides in VARIANTS.items():
                for seed in SEEDS:
                    cfgs.append(make_cfg(variant_name, overrides, dataset, tight, seed))
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"\nGenerated {len(cfgs)} G5 component-ablation configs -> {SWEEP_ROOT}")


if __name__ == "__main__":
    build()
