"""Imagewoof probe + smoke.

Imagewoof: 10 fine-grained dog breeds, ~10k train, ~3.9k val.
Constrained class 0 has ~390 test samples => K @ L50 = ~195 (well above the
infeasibility cliff that hit flowers/dtd).

Modes:
  python -m src.config_generators.gen_imagewoof_smoke probe
  python -m src.config_generators.gen_imagewoof_smoke smoke
"""
import sys
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

DS_NAME = "imagewoof"
DS_META = {
    "data_dir": "data/imagewoof/slice_1", "num_classes": 10,
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


def make_cfg(method, seed, phase, warmup, constraint_epochs):
    hp = {**SHARED_HP, **PER_METHOD[method], "seed": seed,
          "warmup_epochs": warmup, "constraint_epochs": constraint_epochs}
    ds_config = dict(DS_META)
    parts = TIGHT.split("_")
    pair = (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)
    bmid = compute_base_model_id(MODEL, hp, dataset_mode=DS_NAME,
                                 data_dir=DS_META["data_dir"],
                                 dataset_config=ds_config)
    root = f"results/pending_runs/imagewoof_{phase}"
    leaf = (Path(root) / method / f"seed_{seed}" if phase == "smoke"
            else Path(root) / "probe")
    return {
        "methodology": method, "model_name": MODEL,
        "constraint": list(pair), "constraint_tag": TIGHT,
        "dataset_mode": DS_NAME, "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "exp_name": f"imagewoof_{phase}_{method}_seed{seed}",
        "experiment_path": str(leaf),
    }


def build_probe():
    cfg = make_cfg("tralo", 1, "probe", warmup=3, constraint_epochs=1)
    print("PROBE Imagewoof: 1 config.")
    save_configs([cfg], output_dir="results/pending_runs/imagewoof_probe")


def build_smoke():
    cfgs = [make_cfg(m, s, "smoke", warmup=3, constraint_epochs=100)
            for m in PER_METHOD for s in SEEDS]
    print(f"SMOKE Imagewoof: {len(cfgs)} configs.")
    save_configs(cfgs, output_dir="results/pending_runs/imagewoof_smoke")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if mode == "probe":
        build_probe()
    elif mode == "smoke":
        build_smoke()
    else:
        raise SystemExit("usage: {probe | smoke}")
