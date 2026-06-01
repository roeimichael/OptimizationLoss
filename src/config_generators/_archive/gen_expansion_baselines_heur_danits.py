"""Full heuristic + danits_lp baseline grid for the expansion sweep.

Fills the 2 method holes left after the 332-config sweep: heuristic and
danits_lp were never tested on the post-cleanup pipeline. Both are
non-training methods that operate on the cached warmup; each cell runs
in ~10 seconds.

Grid: 2 datasets x 2 methods x 9 tightness x 3 seeds = 108 configs.

Output: results/pending_runs/expansion_baselines
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/expansion_baselines"

DATASETS = {
    "tissuemnist": {"data_dir": "data/tissuemnist/slice_1", "num_classes": 8,
                    "image_size": 224, "target_column": "label",
                    "group_column": "synth_group"},
    "eurosat": {"data_dir": "data/eurosat/slice_1", "num_classes": 10,
                "image_size": 224, "target_column": "label",
                "group_column": "synth_group"},
}
TIGHTNESS = [
    "L20_G20", "L30_G30", "L50_G50",          # symmetric tight
    "L70_G70", "L80_G80",                      # symmetric loose
    "L20_G50", "L50_G20", "L30_G80", "L80_G30",  # asymmetric
]
SEEDS = [1, 2, 3]
MODEL = "MobileNetV3"
CLS = 4

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


def make_cfg(method, dataset, tight_tag, seed):
    ds_meta = DATASETS[dataset]
    hp = {**SHARED_HP, **PER_METHOD[method], "seed": seed}
    ds_config = {**ds_meta, "constrained_class": CLS}
    pair = _tight_pair(tight_tag)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode=dataset,
        data_dir=ds_meta["data_dir"], dataset_config=ds_config,
    )
    return {
        "methodology": method,
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": tight_tag,
        "dataset_mode": dataset,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"exp_{method}_{dataset}_{tight_tag}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / dataset / tight_tag / method / f"seed_{seed}"),
    }


def build():
    cfgs = []
    for dataset in DATASETS:
        for tight in TIGHTNESS:
            for method in PER_METHOD:
                for seed in SEEDS:
                    cfgs.append(make_cfg(method, dataset, tight, seed))
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"\nGenerated {len(cfgs)} expansion_baselines configs -> {SWEEP_ROOT}")


if __name__ == "__main__":
    build()
