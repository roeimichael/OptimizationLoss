"""DermMNIST heuristic + danits_lp baseline grid for expansion sweep.

Constrained class: 4 = MEL (melanoma) — "oncology referral budget" story.
Real local groups via loc_group (anatomical site).

Grid: 2 methods x 9 tightness x 3 seeds = 54 configs.

Output: results/pending_runs/expansion_dermmnist_baselines
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/expansion_dermmnist_baselines"

DS_META = {
    "data_dir": "data/dermmnist/slice_1", "num_classes": 7,
    "image_size": 224, "target_column": "label",
    "group_column": "loc_group",  # REAL anatomical-site groups
}
TIGHTNESS = [
    "L20_G20", "L30_G30", "L50_G50",
    "L70_G70", "L80_G80",
    "L20_G50", "L50_G20", "L30_G80", "L80_G30",
]
SEEDS = [1, 2, 3]
MODEL = "MobileNetV3"
CLS = 4  # MEL = melanoma (story: oncology referral budget)

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}

PER_METHOD = {
    "heuristic": {},
    "danits_lp": {"danits_cost_preset": "identity"},
}


def _tight_pair(tag):
    parts = tag.split("_")
    return (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)


def make_cfg(method, tight_tag, seed):
    hp = {**SHARED_HP, **PER_METHOD[method], "seed": seed}
    ds_config = {**DS_META, "constrained_class": CLS}
    pair = _tight_pair(tight_tag)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode="dermmnist",
        data_dir=DS_META["data_dir"], dataset_config=ds_config,
    )
    return {
        "methodology": method,
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": tight_tag,
        "dataset_mode": "dermmnist",
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"exp_derm_{method}_{tight_tag}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / tight_tag / method / f"seed_{seed}"),
    }


def build():
    cfgs = []
    for tight in TIGHTNESS:
        for method in PER_METHOD:
            for seed in SEEDS:
                cfgs.append(make_cfg(method, tight, seed))
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"\nGenerated {len(cfgs)} dermmnist baselines -> {SWEEP_ROOT}")


if __name__ == "__main__":
    build()
