"""Confirm the w=1 minimal-warmup sweet spot across backbones + more seeds.

The probe found (derm/MobileNetV3, 2 seeds): at warmup_epochs=1 TraLO beats
BOTH Fioretto (+0.006) and Hounie (+0.016) on F1, both seeds, on a usable
model (acc 0.84) — while w=3 (standard) ties/loses. This run tests whether the
w=1 win is robust across backbones and more seeds.

Grid:
    dermmnist (cls4, loc_group)
    x {MobileNetV3, ResNet18, EfficientNetB0}
    x warmup_epochs {1, 3}   (sweet spot vs standard tie)
    x {tralo, fioretto_ldf, hounie_rcl}
    x 4 seeds x tightness L50  = 72 cells.

constraint_epochs=100 (same as the probe, for apples-to-apples). Frame the
result in TRAIN-ACCURACY terms: record the warmup train acc each (backbone,w)
reached (from the run log) and relate it to TraLO's dF1, to identify the
train-acc threshold at which to break warmup.

Configs land at
    results/pending_runs/warmup_confirm/{backbone}/w{warmup}/{method}/seed_{s}/
"""
from pathlib import Path
import glob, json, os

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/warmup_confirm"

DS_NAME = "dermmnist"
DS_META = {
    "data_dir": "data/dermmnist/slice_1", "num_classes": 7,
    "image_size": 224, "target_column": "label",
    "group_column": "loc_group", "constrained_class": 4,
}

BACKBONES = ["MobileNetV3", "ResNet18", "EfficientNetB0"]
WARMUPS = [1, 3]
TIGHT = "L50_G50"
SEEDS = [1, 2, 3, 4]

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "constraint_epochs": 100, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
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


def make_cfg(backbone, warmup, method, seed):
    hp = {**SHARED_HP, **PER_METHOD[method], "warmup_epochs": warmup, "seed": seed}
    ds_config = dict(DS_META)
    parts = TIGHT.split("_")
    pair = (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)
    bmid = compute_base_model_id(
        backbone, hp, dataset_mode=DS_NAME,
        data_dir=DS_META["data_dir"], dataset_config=ds_config)
    return {
        "methodology": method, "model_name": backbone,
        "constraint": list(pair), "constraint_tag": TIGHT,
        "dataset_mode": DS_NAME, "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "exp_name": f"wconf_{backbone}_w{warmup}_{method}_{DS_NAME}_{TIGHT}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / backbone / f"w{warmup}" / method / f"seed_{seed}"),
    }


def build():
    done = _scan_done_cells()
    print(f"Pre-scan: {len(done)} cells already completed.")
    cfgs, skipped = [], 0
    for backbone in BACKBONES:
        for warmup in WARMUPS:
            for method in METHODS:
                for seed in SEEDS:
                    key = (DS_NAME, backbone, 4, "loc_group", TIGHT,
                           method, seed, warmup)
                    if key in done:
                        skipped += 1
                        continue
                    cfgs.append(make_cfg(backbone, warmup, method, seed))
    n_target = len(BACKBONES) * len(WARMUPS) * len(METHODS) * len(SEEDS)
    print(f"Target: {n_target} cells ({len(BACKBONES)} bb x {len(WARMUPS)} warmup "
          f"x {len(METHODS)} mthd x {len(SEEDS)} seed). Already done: {skipped}. "
          f"Will queue: {len(cfgs)}.")
    save_configs(cfgs, output_dir=SWEEP_ROOT)


if __name__ == "__main__":
    build()
