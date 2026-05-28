"""Backbone smoketest on DermMNIST — does a different backbone give TraLO a
clearer F1 win where MobileNetV3 only ties?

Grid:
    DermMNIST x {ConvNeXtTiny, DenseNet121, RegNetY16GF} x cls=4 (MEL)
    x group=loc_group x {L20, L50} x {tralo, fioretto_ldf, hounie_rcl}
    x 2 seeds = 36 cells.

Picks the two tightness extremes (L20 tight, L50 loose) and the two trained
competitors that tie TraLO on derm with MobileNetV3. TraLO uses the canonical
breakthrough recipe (undershoot_hinge + reset_optimizer_at_sat, alpha_kl=0).

New configs land at
    results/pending_runs/backbone_smoke/{backbone}/{tight}/{method}/seed_{s}/
"""
from pathlib import Path
import glob, json, os

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/backbone_smoke"

DS_META = {
    "data_dir": "data/dermmnist/slice_1", "num_classes": 7,
    "image_size": 224, "target_column": "label",
    "group_column": "loc_group", "constrained_class": 4,
}
DS_NAME = "dermmnist"
# ConvNeXtTiny dropped: reproducibly NaNs at constraint epoch 3 under hardcoded
# BF16 AMP (saturated logits overflow in the constraint two-pass); LR lowering
# didn't help and FP32 would need a pipeline change. DenseNet/RegNet are BN-CNNs
# and BF16-stable.
BACKBONES = ["DenseNet121", "RegNetY16GF"]

TIGHTNESS = ["L20_G20", "L50_G50"]
SEEDS = [1, 2]

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300, "pretrained": True,
    "class_weighted_ce": False, "constraint_chunk_size": 256,
}

# ConvNeXt is LR-sensitive: the MobileNet-tuned 1e-4/5e-6 drives its warmup CE
# to ~0 (overfit) then NaNs at constraint epoch 3. A gentler, architecture-
# appropriate LR keeps it stable. Changing LR also gives it a fresh warmup
# cache (base_model_id includes LR), so the NaN-prone cached warmup is dropped.
PER_BACKBONE_HP = {
    "ConvNeXtTiny": {"lr": 2e-5, "lr_constraint": 1e-6},
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
            c.get("dataset_mode"),
            c.get("model_name"),
            c.get("dataset_config", {}).get("constrained_class"),
            c.get("dataset_config", {}).get("group_column"),
            c.get("constraint_tag"),
            c.get("methodology"),
            c.get("hyperparams", {}).get("seed"),
        ))
    return done


def make_cfg(backbone, tight_tag, method, seed):
    hp = {**SHARED_HP, **PER_BACKBONE_HP.get(backbone, {}),
          **PER_METHOD[method], "seed": seed}
    ds_config = dict(DS_META)
    parts = tight_tag.split("_")
    pair = (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)
    bmid = compute_base_model_id(
        backbone, hp, dataset_mode=DS_NAME,
        data_dir=DS_META["data_dir"], dataset_config=ds_config,
    )
    return {
        "methodology": method,
        "model_name": backbone,
        "constraint": list(pair),
        "constraint_tag": tight_tag,
        "dataset_mode": DS_NAME,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"bbsmoke_{backbone}_{method}_{DS_NAME}_{tight_tag}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / backbone / tight_tag / method / f"seed_{seed}"
        ),
    }


def build():
    done = _scan_done_cells()
    print(f"Pre-scan: {len(done)} cells already completed.")
    cfgs, skipped = [], 0
    for backbone in BACKBONES:
        for tight in TIGHTNESS:
            for method in METHODS:
                for seed in SEEDS:
                    key = (DS_NAME, backbone, 4, "loc_group", tight, method, seed)
                    if key in done:
                        skipped += 1
                        continue
                    cfgs.append(make_cfg(backbone, tight, method, seed))
    n_target = len(BACKBONES) * len(TIGHTNESS) * len(METHODS) * len(SEEDS)
    print(f"Target: {n_target} cells ({len(BACKBONES)} backbones x "
          f"{len(TIGHTNESS)} tight x {len(METHODS)} mthd x {len(SEEDS)} seed). "
          f"Already done: {skipped}. Will queue: {len(cfgs)}.")
    save_configs(cfgs, output_dir=SWEEP_ROOT)


if __name__ == "__main__":
    build()
