"""Model-search generator for the warmup-headroom iteration.

Two modes:

  python -m src.config_generators.gen_model_search probe MODEL
      1 config: MODEL on derm, warmup_epochs=8, constraint_epochs=1 (cheap),
      tralo, seed 1. Purpose: read the per-epoch warmup train-acc curve from
      the log to find where the model saturates.

  python -m src.config_generators.gen_model_search smoke MODEL E
      9 configs: {tralo, fioretto_ldf, hounie_rcl} x {derm, tissue, aider}
      x seed 1, warmup_epochs=E, constraint_epochs=100. Purpose: test whether,
      capped at E warmup epochs (sub-saturation), TraLO beats both baselines.

Configs land at results/pending_runs/model_search/{MODEL}/{probe|smoke}/...
The whole point is the warmup-headroom regime: a model is worth the smoke only
if it does NOT saturate (train acc ~1.0) within 1 epoch. See
gen_warmup_confirm.py for the MobileNetV3 reference (w1 train-acc 0.746 -> win).
"""
import sys
from pathlib import Path

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/model_search"

DATASETS = {
    "dermmnist": {"data_dir": "data/dermmnist/slice_1", "num_classes": 7,
                  "image_size": 224, "target_column": "label",
                  "group_column": "loc_group", "constrained_class": 4},
    "aider": {"data_dir": "data/aider/slice_1", "num_classes": 4,
              "image_size": 224, "target_column": "label",
              "group_column": "synth_group", "constrained_class": 0},
    "bloodmnist": {"data_dir": "data/bloodmnist/slice_1", "num_classes": 8,
                   "image_size": 224, "target_column": "label",
                   "group_column": "synth_group", "constrained_class": 0},
    "retinamnist": {"data_dir": "data/retinamnist/slice_1", "num_classes": 5,
                    "image_size": 224, "target_column": "label",
                    "group_column": "synth_group", "constrained_class": 2},
}

TIGHT = "L50_G50"

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
    "fioretto_ldf": {"fioretto_step_size": 0.005},
    "hounie_rcl": {"hounie_eta_lambda": 0.01, "hounie_eta_u": 0.01,
                   "hounie_alpha": 10.0},
}


def make_cfg(model, ds_name, warmup, method, seed, constraint_epochs, phase):
    ds_meta = DATASETS[ds_name]
    hp = {**SHARED_HP, **PER_METHOD[method], "warmup_epochs": warmup,
          "seed": seed, "constraint_epochs": constraint_epochs}
    ds_config = dict(ds_meta)
    parts = TIGHT.split("_")
    pair = (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)
    bmid = compute_base_model_id(model, hp, dataset_mode=ds_name,
                                 data_dir=ds_meta["data_dir"],
                                 dataset_config=ds_config)
    leaf = (Path(SWEEP_ROOT) / model / phase / ds_name / method / f"seed_{seed}"
            if phase == "smoke"
            else Path(SWEEP_ROOT) / model / phase / ds_name)
    return {
        "methodology": method, "model_name": model,
        "constraint": list(pair), "constraint_tag": TIGHT,
        "dataset_mode": ds_name, "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "exp_name": f"ms_{model}_{phase}_{ds_name}_{method}_w{warmup}_seed{seed}",
        "experiment_path": str(leaf),
    }


def build_probe(model, ds_name="dermmnist"):
    cfgs = [make_cfg(model, ds_name, 8, "tralo", 1,
                     constraint_epochs=1, phase="probe")]
    print(f"PROBE {model} on {ds_name}: 1 config (warmup_epochs=8, constraint_epochs=1).")
    save_configs(cfgs, output_dir=SWEEP_ROOT)


def build_smoke(model, e, only_ds=None):
    targets = [only_ds] if only_ds else list(DATASETS)
    cfgs = []
    for ds_name in targets:
        for method in PER_METHOD:
            cfgs.append(make_cfg(model, ds_name, e, method, 1,
                                 constraint_epochs=100, phase="smoke"))
    print(f"SMOKE {model}: {len(cfgs)} configs (3 methods x {len(targets)} ds, "
          f"warmup_epochs={e}, ds={targets}).")
    save_configs(cfgs, output_dir=SWEEP_ROOT)


if __name__ == "__main__":
    mode = sys.argv[1]
    model = sys.argv[2]
    if mode == "probe":
        ds = sys.argv[3] if len(sys.argv) > 3 else "dermmnist"
        build_probe(model, ds)
    elif mode == "smoke":
        only = sys.argv[4] if len(sys.argv) > 4 else None
        build_smoke(model, int(sys.argv[3]), only)
    else:
        raise SystemExit("usage: gen_model_search.py {probe MODEL [DS] | smoke MODEL E [DS]}")
