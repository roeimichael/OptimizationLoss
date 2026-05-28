"""FAST probe of the warmup-headroom idea before committing to the full grid.

Minimal decisive cut: MobileNetV3 (fastest backbone) x warmup {0 vs 3} (max
headroom vs near-saturated) x both easy datasets x 3 methods x 2 seeds = 24
cells. constraint_epochs capped at 100 (2-3x faster AND better-aligned: the
joint CE+constraint phase re-saturates the model by ~epoch 45-100, so stopping
earlier preserves the headroom we're probing).

Question: does TraLO dF1 vs Fioretto/Hounie go from ~0 (w=3) to POSITIVE (w=0)?
If yes -> expand to ResNet/EfficientNet + full grid. If no -> pivot datasets.

Configs land at
    results/pending_runs/warmup_probe/{ds}/MobileNetV3/w{warmup}/{method}/seed_{s}/
"""
from pathlib import Path
import glob, json, os

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/warmup_probe"

# Fill the unexplored middle: w=0 collapses (majority-class), w>=3 overkills
# derm (~95% train acc). Sweet spot, if any, is w=1 or w=2. Derm only (fast);
# w=0 and w=3 already done in the same dir, so the report shows {0,1,2,3}.
DATASETS = {
    "dermmnist": {
        "data_dir": "data/dermmnist/slice_1", "num_classes": 7,
        "image_size": 224, "target_column": "label",
        "group_column": "loc_group", "constrained_class": 4,
    },
}

BACKBONE = "MobileNetV3"
WARMUPS = [1, 2]
TIGHT = "L50_G50"
SEEDS = [1, 2]

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


def make_cfg(ds_name, warmup, method, seed):
    ds_meta = DATASETS[ds_name]
    hp = {**SHARED_HP, **PER_METHOD[method], "warmup_epochs": warmup, "seed": seed}
    ds_config = dict(ds_meta)
    parts = TIGHT.split("_")
    pair = (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)
    bmid = compute_base_model_id(
        BACKBONE, hp, dataset_mode=ds_name,
        data_dir=ds_meta["data_dir"], dataset_config=ds_config)
    return {
        "methodology": method, "model_name": BACKBONE,
        "constraint": list(pair), "constraint_tag": TIGHT,
        "dataset_mode": ds_name, "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "exp_name": f"wprobe_{ds_name}_{BACKBONE}_w{warmup}_{method}_{TIGHT}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / ds_name / BACKBONE / f"w{warmup}" / method
            / f"seed_{seed}"),
    }


def build():
    done = _scan_done_cells()
    print(f"Pre-scan: {len(done)} cells already completed.")
    cfgs, skipped = [], 0
    for ds_name, ds_meta in DATASETS.items():
        for warmup in WARMUPS:
            for method in METHODS:
                for seed in SEEDS:
                    key = (ds_name, BACKBONE, ds_meta["constrained_class"],
                           ds_meta["group_column"], TIGHT, method, seed, warmup)
                    if key in done:
                        skipped += 1
                        continue
                    cfgs.append(make_cfg(ds_name, warmup, method, seed))
    n_target = len(DATASETS) * len(WARMUPS) * len(METHODS) * len(SEEDS)
    print(f"Target: {n_target} cells (2 ds x 1 bb x {len(WARMUPS)} warmup x "
          f"{len(METHODS)} mthd x {len(SEEDS)} seed). Already done: {skipped}. "
          f"Will queue: {len(cfgs)}.")
    save_configs(cfgs, output_dir=SWEEP_ROOT)


if __name__ == "__main__":
    build()
