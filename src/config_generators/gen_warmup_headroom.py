"""Warmup-headroom ablation: does limiting warmup (less/no CE pretraining)
restore TraLO's F1 edge on EASY datasets by preventing logit saturation?

Controlled experiment — fix the dataset, vary only warmup_epochs (the headroom
knob). Hypothesis: as warmup_epochs drops, warmup train acc drops, logits stay
unsaturated, the constraint loss keeps gradient, and TraLO's F1 advantage over
Fioretto/Hounie grows (recovering the tissue-regime win on derm/OCT).

Grid:
    {dermmnist(cls4,loc_group), octmnist(cls2,synth_group)}
    x {MobileNetV3, ResNet18, EfficientNetB0}
    x warmup_epochs {0, 3, 10, 50}
    x {tralo, fioretto_ldf, hounie_rcl}
    x 2 seeds x tightness L50  = 144 cells.

warmup_epochs is part of base_model_id, so each warmup level gets its own
fresh warmup cache (shared across the 3 methods at that level). The 50-epoch
(saturated) anchor stays in so the dF1-vs-headroom curve is honest.

Configs land at
    results/pending_runs/warmup_headroom/{ds}/{backbone}/w{warmup}/{method}/seed_{s}/
"""
from pathlib import Path
import glob, json, os

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/warmup_headroom"

DATASETS = {
    "dermmnist": {
        "data_dir": "data/dermmnist/slice_1", "num_classes": 7,
        "image_size": 224, "target_column": "label",
        "group_column": "loc_group", "constrained_class": 4,
    },
    "octmnist": {
        "data_dir": "data/octmnist", "num_classes": 4,
        "image_size": 224, "target_column": "label",
        "group_column": "synth_group", "constrained_class": 2,
    },
}

BACKBONES = ["MobileNetV3", "ResNet18", "EfficientNetB0"]
# Focused on the headroom-transition zone: derm/OCT saturate (~95% train acc)
# by epoch 3, so warmup 10/50 are redundant (all saturated). 0=random head
# (max headroom) ... 3=near-saturated. The known saturated anchor (w=50 = tie)
# lives in the existing paper grid; no need to re-burn slow cells here.
WARMUPS = [0, 1, 2, 3]
TIGHT = "L50_G50"
SEEDS = [1, 2]

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "constraint_epochs": 300, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}

PER_METHOD = {
    "tralo": {
        "lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
        "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
        "penalty_mode": "both", "enable_ce_skip": True,
        "hybrid_mode": "undershoot_hinge",
        "fior_beta": 0.50,
        "reset_optimizer_at_sat": True,
    },
    "fioretto_ldf": {"fioretto_step_size": 0.005},
    "hounie_rcl": {"hounie_eta_lambda": 0.01, "hounie_eta_u": 0.01,
                   "hounie_alpha": 10.0},
}
METHODS = list(PER_METHOD.keys())


def _scan_done_cells():
    done = set()
    for f in glob.glob("results/pending_runs/*/**/config.json", recursive=True):
        ev = f.replace("config.json", "evaluation_metrics.csv")
        if not os.path.exists(ev):
            continue
        try:
            c = json.load(open(f))
        except Exception:
            continue
        done.add((
            c.get("dataset_mode"), c.get("model_name"),
            c.get("dataset_config", {}).get("constrained_class"),
            c.get("dataset_config", {}).get("group_column"),
            c.get("constraint_tag"), c.get("methodology"),
            c.get("hyperparams", {}).get("seed"),
            c.get("hyperparams", {}).get("warmup_epochs"),
        ))
    return done


def make_cfg(ds_name, backbone, warmup, method, seed):
    ds_meta = DATASETS[ds_name]
    hp = {**SHARED_HP, **PER_METHOD[method],
          "warmup_epochs": warmup, "seed": seed}
    ds_config = dict(ds_meta)
    parts = TIGHT.split("_")
    pair = (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)
    bmid = compute_base_model_id(
        backbone, hp, dataset_mode=ds_name,
        data_dir=ds_meta["data_dir"], dataset_config=ds_config,
    )
    return {
        "methodology": method,
        "model_name": backbone,
        "constraint": list(pair),
        "constraint_tag": TIGHT,
        "dataset_mode": ds_name,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"wh_{ds_name}_{backbone}_w{warmup}_{method}_{TIGHT}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / ds_name / backbone / f"w{warmup}" / method
            / f"seed_{seed}"),
    }


def build():
    done = _scan_done_cells()
    print(f"Pre-scan: {len(done)} cells already completed.")
    cfgs, skipped = [], 0
    for ds_name, ds_meta in DATASETS.items():
        for backbone in BACKBONES:
            for warmup in WARMUPS:
                for method in METHODS:
                    for seed in SEEDS:
                        key = (ds_name, backbone, ds_meta["constrained_class"],
                               ds_meta["group_column"], TIGHT, method, seed, warmup)
                        if key in done:
                            skipped += 1
                            continue
                        cfgs.append(make_cfg(ds_name, backbone, warmup, method, seed))
    n_target = (len(DATASETS) * len(BACKBONES) * len(WARMUPS)
                * len(METHODS) * len(SEEDS))
    print(f"Target: {n_target} cells (2 ds x {len(BACKBONES)} bb x "
          f"{len(WARMUPS)} warmup x {len(METHODS)} mthd x {len(SEEDS)} seed). "
          f"Already done: {skipped}. Will queue: {len(cfgs)}.")
    save_configs(cfgs, output_dir=SWEEP_ROOT)


if __name__ == "__main__":
    build()
