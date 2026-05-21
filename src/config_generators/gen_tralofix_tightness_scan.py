"""Extend the tralofix comparison across more tightness levels, including
asymmetric local/global pairs.

Cells (6 new tightness x 2 datasets x 2 seeds x 4 methods = 96 configs):
  Diagonal-loose:  L70_G70, L80_G80   (cover the loose end)
  Asymmetric:      L20_G50, L50_G20   (loose local, tight global / vice-versa)
                   L30_G80, L80_G30   (extreme asymmetry)

2 seeds for the scan; we can deepen the interesting cells later.
Routes tralo_fioretto -> paper400_tralofix root, others -> paper400_baselines.
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

DATASETS = {
    "tissuemnist": {"data_dir": "data/tissuemnist/slice_1", "num_classes": 8,
                    "image_size": 224, "target_column": "label",
                    "group_column": "synth_group"},
    "eurosat": {"data_dir": "data/eurosat/slice_1", "num_classes": 10,
                "image_size": 224, "target_column": "label",
                "group_column": "synth_group"},
}
NEW_TIGHTNESS = [
    "L70_G70", "L80_G80",                    # diagonal-loose
    "L20_G50", "L50_G20",                    # mild asymmetry
    "L30_G80", "L80_G30",                    # extreme asymmetry
]
SEEDS = [1, 2]
MODEL = "MobileNetV3"
CLS = 4

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}

METHODS = {
    "tralo": {
        "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
        "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
        "penalty_mode": "both", "enable_ce_skip": True,
    },
    "fioretto_ldf": {"fioretto_step_size": 0.005},
    "hounie_rcl": {"hounie_eta_lambda": 0.01, "hounie_eta_u": 0.01,
                   "hounie_alpha": 10.0},
    "tralo_fioretto": {  # the "tralo_fix" cell
        "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
        "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
        "penalty_mode": "both", "enable_ce_skip": True,
        "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
        "reset_optimizer_at_sat": True,
    },
}


def _tight_pair(tag):
    parts = tag.split("_")
    return (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)


def make_cfg(method, dataset, tight_tag, seed):
    ds_meta = DATASETS[dataset]
    hp = {**SHARED_HP, **METHODS[method], "seed": seed}
    ds_config = {**ds_meta, "constrained_class": CLS}
    pair = _tight_pair(tight_tag)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode=dataset,
        data_dir=ds_meta["data_dir"], dataset_config=ds_config,
    )
    if method == "tralo_fioretto":
        root = "results/pending_runs/paper400_tralofix"
        exp_path = Path(root) / dataset / tight_tag / f"seed_{seed}"
        name = f"tralofix_{dataset}_{tight_tag}_seed{seed}"
    else:
        root = "results/pending_runs/paper400_baselines"
        exp_path = Path(root) / dataset / tight_tag / method / f"seed_{seed}"
        name = f"p400base_{method}_{dataset}_{tight_tag}_seed{seed}"
    return {
        "methodology": method,
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": tight_tag,
        "dataset_mode": dataset,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": name,
        "experiment_path": str(exp_path),
    }


def build():
    cfgs = []
    for dataset in DATASETS:
        for tight in NEW_TIGHTNESS:
            for method in METHODS:
                for seed in SEEDS:
                    cfgs.append(make_cfg(method, dataset, tight, seed))
    save_configs(cfgs, output_dir="results/pending_runs/paper400_tralofix")
    print(f"\nGenerated {len(cfgs)} tightness-scan configs")


if __name__ == "__main__":
    build()
