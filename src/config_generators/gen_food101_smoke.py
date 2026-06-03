"""Food101 probe + smoke generator.

Modes:
  python -m src.config_generators.gen_food101_smoke probe
      1 config: MobileNetV3, warmup=3, constraint=1, seed=1, tralo, L50_G50.
      Verifies headroom (ep3 train-acc in (0.40, 0.82)) before smoke.

  python -m src.config_generators.gen_food101_smoke smoke
      24 configs: 6 methods × 4 seeds × L50_G50 × warmup=3 + ce_skip=False.
      Same recipe as flowers102_smoke / dtd_smoke for direct comparability.

Configs land at results/pending_runs/food101_{probe|smoke}/...

Constrained class = 0 (apple_pie -- placeholder for the paper. With ~250 test
samples per class, K @ L50 = 125 -- well above the infeasibility cliff that
hit flowers102/dtd).
"""
import sys
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

DS_NAME = "food101"
DS_META = {
    "data_dir": "data/food101/slice_1", "num_classes": 101,
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
    leaf = (Path(f"results/pending_runs/food101_{phase}") / method / f"seed_{seed}"
            if phase == "smoke"
            else Path(f"results/pending_runs/food101_{phase}") / "probe")
    return {
        "methodology": method, "model_name": MODEL,
        "constraint": list(pair), "constraint_tag": TIGHT,
        "dataset_mode": DS_NAME, "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "exp_name": f"food101_{phase}_{method}_seed{seed}",
        "experiment_path": str(leaf),
    }


def build_probe():
    cfg = make_cfg("tralo", 1, "probe", warmup=3, constraint_epochs=1)
    print("PROBE Food101: 1 config (warmup=3, constraint=1).")
    save_configs([cfg], output_dir="results/pending_runs/food101_probe")


def build_smoke():
    cfgs = [make_cfg(m, s, "smoke", warmup=3, constraint_epochs=100)
            for m in PER_METHOD for s in SEEDS]
    print(f"SMOKE Food101: {len(cfgs)} configs (6 methods × 4 seeds).")
    save_configs(cfgs, output_dir="results/pending_runs/food101_smoke")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if mode == "probe":
        build_probe()
    elif mode == "smoke":
        build_smoke()
    else:
        raise SystemExit("usage: {probe | smoke}")
