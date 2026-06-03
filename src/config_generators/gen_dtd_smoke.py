"""DTD smoke: 6 methods × 4 seeds × L50_G50 on MobileNetV3.

DTD (Describable Textures Dataset): 47 texture classes, 1880 train, 1880 test.
Story angle: material quality control / quality assurance — model must not
over-predict defect-like textures.

Same recipe as flowers102_smoke v2 (warmup=3, constraint_epochs=100, ce_skip=False)
to keep TraLO inside its headroom band.
"""
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/dtd_smoke"

DS_NAME = "dtd"
DS_META = {
    "data_dir": "data/dtd/slice_1", "num_classes": 47,
    "image_size": 224, "target_column": "label",
    "group_column": "synth_group", "constrained_class": 0,
}
MODEL = "MobileNetV3"
TIGHT = "L50_G50"
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


def make_cfg(method, seed):
    hp = {**SHARED_HP, **PER_METHOD[method], "seed": seed}
    ds_config = dict(DS_META)
    parts = TIGHT.split("_")
    pair = (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)
    bmid = compute_base_model_id(MODEL, hp, dataset_mode=DS_NAME,
                                 data_dir=DS_META["data_dir"],
                                 dataset_config=ds_config)
    return {
        "methodology": method, "model_name": MODEL,
        "constraint": list(pair), "constraint_tag": TIGHT,
        "dataset_mode": DS_NAME, "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "exp_name": f"dtd_smoke_{method}_seed{seed}",
        "experiment_path": str(Path(SWEEP_ROOT) / method / f"seed_{seed}"),
    }


def build():
    cfgs = [make_cfg(m, s) for m in PER_METHOD for s in SEEDS]
    print(f"Queueing {len(cfgs)} DTD smoke configs (6 methods × 4 seeds).")
    save_configs(cfgs, output_dir=SWEEP_ROOT)


if __name__ == "__main__":
    build()
