"""Contamination grid: 3 datasets x 3 sigmas x 4 tightness x 5 methods x 2 seeds.

Plus split into per-dataset sweep roots for parallel-dispatch on 3 GPUs.

Sigma=0 (clean) NOT generated here — those cells already exist in the
headline sweeps (paper_backbones, asym_tissue_aider) and will be merged
at aggregation time.

Output: results/pending_runs/contamination_{dataset}/
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SIGMAS = [10, 20, 30]                 # encoded as int(sigma*100)
TIGHTNESS = ["L20_G20", "L30_G30", "L50_G50", "L70_G70"]
SEEDS = [1, 2]
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]
MODEL = "MobileNetV3"

DATASETS = {
    "tissuemnist": {
        "num_classes": 8, "image_size": 224, "target_column": "label",
        "group_column": "synth_group", "constrained_class": 4,
    },
    "dermmnist": {
        "num_classes": 7, "image_size": 224, "target_column": "label",
        "group_column": "loc_group", "constrained_class": 4,
    },
    "aider": {
        "num_classes": 4, "image_size": 224, "target_column": "label",
        "group_column": "synth_group", "constrained_class": 0,
    },
}

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
    "fioretto_step_size": 0.01,
}

TRALO_HP = {
    "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
    "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
    "penalty_mode": "both", "enable_ce_skip": True,
    "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
    "reset_optimizer_at_sat": True,
    "disable_freeze_on_satisfy": False,
}


def _tight_pair(tag):
    parts = tag.split("_")
    return (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)


def make_cfg(dataset, ds_base, sigma_int, tight, method, seed):
    data_dir = f"data/{dataset}_sigma{sigma_int:02d}/slice_1"
    ds_config = {**ds_base, "data_dir": data_dir}
    hp = {**SHARED_HP, "seed": seed}
    if method == "tralo":
        hp.update(TRALO_HP)
    pair = _tight_pair(tight)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode=dataset,
        data_dir=data_dir, dataset_config=ds_config,
    )
    sweep_root = f"results/pending_runs/contamination_{dataset}"
    return {
        "methodology": method,
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": tight,
        "dataset_mode": dataset,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"contam_{dataset}_s{sigma_int:02d}_{tight}_{method}_seed{seed}",
        "experiment_path": str(
            Path(sweep_root) / f"sigma{sigma_int:02d}" / tight / method / f"seed_{seed}"),
    }


def build():
    counts = {}
    for dataset, ds_base in DATASETS.items():
        cfgs = []
        for sigma_int in SIGMAS:
            for tight in TIGHTNESS:
                for method in METHODS:
                    for seed in SEEDS:
                        cfgs.append(make_cfg(dataset, ds_base, sigma_int, tight, method, seed))
        sweep_root = f"results/pending_runs/contamination_{dataset}"
        save_configs(cfgs, output_dir=sweep_root)
        counts[dataset] = len(cfgs)
        print(f"  {dataset}: {len(cfgs)} configs -> {sweep_root}")
    total = sum(counts.values())
    print(f"\nTotal: {total} cells")


if __name__ == "__main__":
    build()
