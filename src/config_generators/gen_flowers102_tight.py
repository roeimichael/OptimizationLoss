"""Flowers102 TIGHT smoke: re-test 6 methods at L20_G20 + L10_G10.

Hypothesis: the initial L50_G50 smoke showed 100% satisfaction for ALL methods,
meaning the constraint was too loose to differentiate. Tightening to L20/L10
should make the constraint bite and reveal whether TraLO actually beats
Fioretto in this regime.

Grid: 2 tightness × 6 methods × 4 seeds = 48 cells.
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/flowers102_tight"

DS_NAME = "flowers102"
DS_META = {
    "data_dir": "data/flowers102/slice_1", "num_classes": 102,
    "image_size": 224, "target_column": "label",
    "group_column": "synth_group", "constrained_class": 0,
}
MODEL = "MobileNetV3"
TIGHTNESS = ["L20_G20", "L10_G10"]
SEEDS = [1, 2, 3, 4]

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 3, "constraint_epochs": 100, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}

PER_METHOD = {
    "tralo": {
        "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
        "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
        "penalty_mode": "both", "enable_ce_skip": False,
        "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
        "reset_optimizer_at_sat": True,
    },
    "tralo_bounded": {
        "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
        "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
        "penalty_mode": "both", "enable_ce_skip": False,
    },
    "fioretto_ldf": {"fioretto_step_size": 0.005},
    "hounie_rcl": {"hounie_eta_lambda": 0.01, "hounie_eta_u": 0.01,
                   "hounie_alpha": 10.0},
    "danits_lp": {},
    "heuristic": {},
}
METHODS = list(PER_METHOD.keys())


def make_cfg(tight_tag, method, seed):
    hp = {**SHARED_HP, **PER_METHOD[method], "seed": seed}
    ds_config = dict(DS_META)
    parts = tight_tag.split("_")
    pair = (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)
    bmid = compute_base_model_id(MODEL, hp, dataset_mode=DS_NAME,
                                 data_dir=DS_META["data_dir"],
                                 dataset_config=ds_config)
    return {
        "methodology": method, "model_name": MODEL,
        "constraint": list(pair), "constraint_tag": tight_tag,
        "dataset_mode": DS_NAME, "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "exp_name": f"flowers102_tight_{method}_{tight_tag}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / tight_tag / method / f"seed_{seed}"),
    }


def build():
    cfgs = [make_cfg(t, m, s) for t in TIGHTNESS for m in METHODS for s in SEEDS]
    print(f"Queueing {len(cfgs)} Flowers102 TIGHT configs (2 tight × 6 mthd × 4 seed).")
    save_configs(cfgs, output_dir=SWEEP_ROOT)


if __name__ == "__main__":
    build()
