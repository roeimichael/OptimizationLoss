"""Paper 400-sweep: 1 backbone × all datasets × all constraints × all methods.

Targets a uniform 400-cell grid for the headline paper tables:
  4 datasets × 5 tightness × 5 methods × 4 seeds = 400 configs.

Backbone: MobileNetV3 (workhorse, most existing baselines exist).
Constrained class: 4 (consistent with prior runs).

Already-completed runs are skipped by save_configs based on experiment_path.
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/paper400"

DATASETS = {
    "tissuemnist": {"data_dir": "data/tissuemnist/slice_1", "num_classes": 8,
                    "image_size": 224, "target_column": "label",
                    "group_column": "synth_group"},
    "dermmnist": {"data_dir": "data/dermmnist/slice_1", "num_classes": 7,
                  "image_size": 224, "target_column": "label",
                  "group_column": "sex"},
    "eurosat": {"data_dir": "data/eurosat/slice_1", "num_classes": 10,
                "image_size": 224, "target_column": "label",
                "group_column": "synth_group"},
    "so2sat": {"data_dir": "data/so2sat", "num_classes": 17,
               "image_size": 224, "target_column": "label",
               "group_column": "city_id"},
}

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}

PER_METHOD = {
    "tralo": {"lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
              "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
              "penalty_mode": "both", "enable_ce_skip": True},
    "fioretto_ldf": {"fioretto_step_size": 0.005},
    "hounie_rcl": {"hounie_eta_lambda": 0.01, "hounie_eta_u": 0.01,
                   "hounie_alpha": 10.0},
    "danits_lp": {},
    "heuristic": {},
}

MODEL = "MobileNetV3"
CLS = 4
SEEDS = [1, 2, 3, 4]
TIGHTNESS = ["L20_G20", "L30_G30", "L50_G50", "L70_G70", "L100_G100"]
METHODS = ["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]


def _tight_pair(tag):
    """L20_G20 -> (0.2, 0.2)."""
    parts = tag.split("_")
    return (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)


def make_cfg(method, dataset, tight_tag, seed):
    hp = {**SHARED_HP, **PER_METHOD[method], "seed": seed}
    ds_meta = DATASETS[dataset]
    ds_config = {**ds_meta, "constrained_class": CLS}
    pair = _tight_pair(tight_tag)
    bmid = compute_base_model_id(MODEL, hp, dataset_mode=dataset,
                                 data_dir=ds_meta["data_dir"],
                                 dataset_config=ds_config)
    return {
        "methodology": method,
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": tight_tag,
        "dataset_mode": dataset,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": (f"paper400_{method}_{dataset}_{MODEL}_cls{CLS}"
                     f"_{tight_tag}_seed{seed}"),
        "experiment_path": str(
            Path(SWEEP_ROOT) / dataset / MODEL / f"cls_{CLS}" / tight_tag /
            method / f"seed_{seed}"),
    }


def build():
    cfgs = []
    for dataset in DATASETS:
        for tight in TIGHTNESS:
            for method in METHODS:
                for seed in SEEDS:
                    cfgs.append(make_cfg(method, dataset, tight, seed))
    save_configs(cfgs, output_dir=SWEEP_ROOT)
    print(f"\nGenerated {len(cfgs)} paper400 configs -> {SWEEP_ROOT}")


if __name__ == "__main__":
    build()
