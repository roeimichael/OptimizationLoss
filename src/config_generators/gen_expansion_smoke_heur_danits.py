"""Smoke test: heuristic + danits_lp dispatch post-cleanup.

4 configs (2 datasets x 2 methods x 1 tightness x 1 seed).
If all 4 produce evaluation_metrics.csv, proceed to full grid.

Output: results/pending_runs/expansion_smoke
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/expansion_smoke"

DATASETS = {
    "tissuemnist": {"data_dir": "data/tissuemnist/slice_1", "num_classes": 8,
                    "image_size": 224, "target_column": "label",
                    "group_column": "synth_group"},
    "eurosat": {"data_dir": "data/eurosat/slice_1", "num_classes": 10,
                "image_size": 224, "target_column": "label",
                "group_column": "synth_group"},
}
METHODS = ["heuristic", "danits_lp"]
TIGHTNESS = "L30_G30"
SEED = 1
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


def make_cfg(method, dataset):
    ds_meta = DATASETS[dataset]
    hp = {**SHARED_HP, **PER_METHOD[method], "seed": SEED}
    ds_config = {**ds_meta, "constrained_class": CLS}
    pair = (0.3, 0.3)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode=dataset,
        data_dir=ds_meta["data_dir"], dataset_config=ds_config,
    )
    return {
        "methodology": method,
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": TIGHTNESS,
        "dataset_mode": dataset,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"smoke_{method}_{dataset}_{TIGHTNESS}_seed{SEED}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / dataset / method / f"seed_{SEED}"),
    }


def build():
    cfgs = [make_cfg(m, d) for d in DATASETS for m in METHODS]
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"\nGenerated {len(cfgs)} smoke configs -> {SWEEP_ROOT}")


if __name__ == "__main__":
    build()
