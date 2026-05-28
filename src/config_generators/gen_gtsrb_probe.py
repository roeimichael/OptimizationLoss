"""GTSRB warmup probe + smoke test.

Modes:

  python -m src.config_generators.gen_gtsrb_probe probe
      1 config: MobileNetV3 on GTSRB, warmup_epochs=8, constraint_epochs=1,
      tralo, seed 1. Purpose: check whether GTSRB has warmup-headroom for
      MobileNetV3 (target ep1 train-acc in [0.70, 0.82] band).

  python -m src.config_generators.gen_gtsrb_probe smoke EPOCHS
      12 configs: MobileNetV3 x {tralo, fioretto_ldf, hounie_rcl, heuristic,
      danits_lp, tralo_bounded} x seeds {1, 2, 3, 4}? -- actually 3 methods
      x 4 seeds = 12. Wait, user asked TraLO vs Heuristic comparison: 6
      methods x 4 seeds at L50_G50 to match the headline Table A grid.

Configs land at results/pending_runs/gtsrb_probe/...
"""
import sys
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/gtsrb_probe"

DS_NAME = "gtsrb"
DS_META = {
    "data_dir": "data/gtsrb/slice_1", "num_classes": 43,
    "image_size": 224, "target_column": "label",
    "group_column": "synth_group", "constrained_class": 14,  # STOP
}
MODEL = "MobileNetV3"
TIGHT = "L50_G50"
SEEDS = [1, 2, 3, 4]

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "pretrained": True, "class_weighted_ce": False, "constraint_chunk_size": 256,
}

PER_METHOD = {
    "tralo": {
        "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
        "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
        "penalty_mode": "both", "enable_ce_skip": True,
        "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
        "reset_optimizer_at_sat": True,
    },
    "tralo_bounded": {
        "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
        "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
        "penalty_mode": "both", "enable_ce_skip": True,
    },
    "fioretto_ldf": {"fioretto_step_size": 0.005},
    "hounie_rcl": {"hounie_eta_lambda": 0.01, "hounie_eta_u": 0.01,
                   "hounie_alpha": 10.0},
    "danits_lp": {},
    "heuristic": {},
}


def make_cfg(method, seed, warmup, constraint_epochs, phase):
    hp = {**SHARED_HP, **PER_METHOD[method],
          "warmup_epochs": warmup, "seed": seed,
          "constraint_epochs": constraint_epochs}
    ds_config = dict(DS_META)
    parts = TIGHT.split("_")
    pair = (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)
    bmid = compute_base_model_id(MODEL, hp, dataset_mode=DS_NAME,
                                 data_dir=DS_META["data_dir"],
                                 dataset_config=ds_config)
    leaf = (Path(SWEEP_ROOT) / phase / method / f"seed_{seed}"
            if phase == "smoke"
            else Path(SWEEP_ROOT) / phase)
    return {
        "methodology": method, "model_name": MODEL,
        "constraint": list(pair), "constraint_tag": TIGHT,
        "dataset_mode": DS_NAME, "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "exp_name": f"gtsrb_{phase}_{method}_w{warmup}_seed{seed}",
        "experiment_path": str(leaf),
    }


def build_probe():
    cfgs = [make_cfg("tralo", 1, warmup=8, constraint_epochs=1, phase="probe")]
    print("PROBE GTSRB: 1 config (warmup_epochs=8, constraint_epochs=1).")
    save_configs(cfgs, output_dir=SWEEP_ROOT)


def build_smoke(warmup_cap):
    cfgs = []
    for method in PER_METHOD:
        for seed in SEEDS:
            cfgs.append(make_cfg(method, seed, warmup=warmup_cap,
                                 constraint_epochs=100, phase="smoke"))
    print(f"SMOKE GTSRB: {len(cfgs)} configs (6 methods x 4 seeds, "
          f"warmup={warmup_cap}, constraint=100).")
    save_configs(cfgs, output_dir=SWEEP_ROOT)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if mode == "probe":
        build_probe()
    elif mode == "smoke":
        build_smoke(int(sys.argv[2]))
    else:
        raise SystemExit("usage: gen_gtsrb_probe.py {probe | smoke EPOCHS}")
