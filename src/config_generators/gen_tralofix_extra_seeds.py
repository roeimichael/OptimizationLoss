"""Extra seeds 5-8 for the 6 paper400_tralofix cells, all 4 methods.

Same HP as gen_paper400_tralofix and gen_paper400_baselines. Doubles
n from 4 to 8 per cell for statistical confidence.

Cells: 2 datasets x 3 tightness x 4 seeds x 4 methods = 96 configs.

Output goes to two sweep dirs so it merges cleanly with the existing
runs:
   paper400_tralofix/<dataset>/<tight>/seed_{5..8}     (tralo_fix)
   paper400_baselines/<dataset>/<tight>/<method>/seed_{5..8}
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
TIGHTNESS = ["L20_G20", "L30_G30", "L50_G50"]
SEEDS = [5, 6, 7, 8]
MODEL = "MobileNetV3"
CLS = 4

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300,
    "use_sum_loss": True, "kl_temperature": 1.0, "pretrained": True,
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
    # Route tralo_fioretto to paper400_tralofix; others to paper400_baselines
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
        for tight in TIGHTNESS:
            for method in METHODS:
                for seed in SEEDS:
                    cfgs.append(make_cfg(method, dataset, tight, seed))
    # save_configs takes a single output_dir; the experiment_path inside
    # each cfg controls where files land, so output_dir is just a default
    # hint. We use one of the two roots; save_configs walks each cfg's
    # experiment_path independently.
    save_configs(cfgs, output_dir="results/pending_runs/paper400_tralofix")
    print(f"\nGenerated {len(cfgs)} extra-seed configs across two sweep dirs")


if __name__ == "__main__":
    build()
