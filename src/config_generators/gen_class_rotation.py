"""Class-rotation search: does the universal TraLO advantage transfer to
different constrained-class choices?

For each of tissue/derm/aider we pick 3 alternate cap classes spanning the
size spectrum (smallest, similar-to-current, larger / different-shape).
Same data, same backbones, same loss; only the constrained_class index
changes.

Output: results/pending_runs/class_rotation/{dataset}/{model}/
        constrained{c}_{tight}/{method}/seed_{n}/config.json

54 cells total: 9 (ds,class) x 3 methods x 2 seeds x 1 tight.
"""
import os
from pathlib import Path
from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

TIGHT = "L50_G50"
SEEDS = [1, 2]
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl"]
MODEL = "MobileNetV3"

# (dataset, num_classes, group_column, alt_cap_classes_to_test)
# Each tuple = class index to test as constrained (not the current paper default).
ROTATIONS = [
    ("tissuemnist", 8, "synth_group",
     [(2, "smallest_3.5pct"), (7, "mid_14.9pct"), (0, "largest_32.1pct")]),
    ("dermmnist",   7, "loc_group",
     [(3, "tiniest_1.1pct"), (0, "small_3.2pct"), (2, "midsize_11.0pct")]),
    ("aider",       4, "synth_group",
     [(1, "alt_smallA_8.7pct"), (2, "alt_smallB_8.8pct"), (3, "majority_73.9pct")]),
]

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
    p = tag.split("_")
    return (int(p[0][1:])/100, int(p[1][1:])/100)


def make_cfg(dataset, n_cls, gcol, cap_class, cap_label, method, seed):
    data_dir = f"data/{dataset}/slice_1"
    ds_config = {
        "num_classes": n_cls, "image_size": 224, "target_column": "label",
        "group_column": gcol, "constrained_class": cap_class,
        "data_dir": data_dir,
    }
    hp = {**SHARED_HP, "seed": seed}
    if method == "tralo":
        hp.update(TRALO_HP)
    pair = _tight_pair(TIGHT)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode=dataset, data_dir=data_dir,
        dataset_config=ds_config,
    )
    sweep_root = "results/pending_runs/class_rotation"
    return {
        "methodology": method,
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": TIGHT,
        "dataset_mode": dataset,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "experiment_path": (
            f"{sweep_root}/{dataset}/{MODEL}/"
            f"constrained{cap_class}_{cap_label}_{TIGHT}/{method}/seed_{seed}"
        ),
    }


def main():
    cfgs = []
    for dataset, n_cls, gcol, alt_classes in ROTATIONS:
        for cap_class, cap_label in alt_classes:
            for method in METHODS:
                for seed in SEEDS:
                    cfgs.append(make_cfg(dataset, n_cls, gcol,
                                          cap_class, cap_label, method, seed))
    print(f"Generated {len(cfgs)} configs")
    save_configs(cfgs, output_dir="results/pending_runs")


if __name__ == "__main__":
    main()
