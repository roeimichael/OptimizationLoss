"""KL anchor ablation on tralo_fix (current full TraLO config).

Tests whether KL regularization (anchor predictions to warmup distribution)
helps or hurts when paired with the bidirectional + Adam-reset setup.
Per memory, KL was a "drift damper" pre-fix. Now that we have bidirectional
penalty + Adam reset, does KL still add value, or is it redundant?

Cells: 2 datasets x 1 tightness (L30, where we saw the cleanest win)
       x 4 alpha_kl values x 2 seeds = 16 runs
Output: results/pending_runs/kl_ablation
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/kl_ablation"
DATASETS = {
    "tissuemnist": {"data_dir": "data/tissuemnist/slice_1", "num_classes": 8,
                    "image_size": 224, "target_column": "label",
                    "group_column": "synth_group"},
    "eurosat": {"data_dir": "data/eurosat/slice_1", "num_classes": 10,
                "image_size": 224, "target_column": "label",
                "group_column": "synth_group"},
}
TIGHTNESS = ["L30_G30"]
ALPHA_KL_VALUES = [0.0, 0.1, 0.3, 1.0]
SEEDS = [1, 2]
MODEL = "MobileNetV3"
CLS = 4

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}

TRALO_FIX = {
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0,
    "penalty_mode": "both", "enable_ce_skip": True,
    "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
    "reset_optimizer_at_sat": True,
}


def _tight_pair(tag):
    parts = tag.split("_")
    return (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)


def make_cfg(dataset, tight_tag, alpha_kl, seed):
    ds_meta = DATASETS[dataset]
    hp = {**SHARED_HP, **TRALO_FIX, "alpha_kl": alpha_kl, "seed": seed}
    ds_config = {**ds_meta, "constrained_class": CLS}
    pair = _tight_pair(tight_tag)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode=dataset,
        data_dir=ds_meta["data_dir"], dataset_config=ds_config,
    )
    cell = f"alphakl{alpha_kl:.1f}".replace(".", "p")
    return {
        "methodology": "tralo",
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": tight_tag,
        "dataset_mode": dataset,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"kl_{cell}_{dataset}_{tight_tag}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / dataset / tight_tag / cell / f"seed_{seed}"),
    }


def build():
    cfgs = []
    for dataset in DATASETS:
        for tight in TIGHTNESS:
            for alpha in ALPHA_KL_VALUES:
                for seed in SEEDS:
                    cfgs.append(make_cfg(dataset, tight, alpha, seed))
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"\nGenerated {len(cfgs)} KL ablation configs -> {SWEEP_ROOT}")


if __name__ == "__main__":
    build()
