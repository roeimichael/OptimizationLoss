"""Phase 2 of paper plan v2 — Table B (asymmetric tightness on dermmnist).

Grid:
    DermMNIST × MobileNetV3 × cls=4 (MEL) × group=loc_group
    × 25 (L,G) pairs (L,G ∈ {0.2, 0.3, 0.5, 0.7, 0.8})
    × 6 methods × 4 seeds = 600 target cells.

Already-completed cells (from Phase 1 / earlier sweeps) are detected by
scanning all of results/pending_runs/ for matching keys and SKIPPED.
New configs land at
    results/pending_runs/paperv2_phase2/{tight}/{method}/seed_{s}/
"""
from pathlib import Path
import glob, json, os

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/paperv2_phase2"

DS_META = {
    "data_dir": "data/dermmnist/slice_1", "num_classes": 7,
    "image_size": 224, "target_column": "label",
    "group_column": "loc_group", "constrained_class": 4,
}
DS_NAME = "dermmnist"
MODEL = "MobileNetV3"

L_VALS = [20, 30, 50, 70, 80]
G_VALS = [20, 30, 50, 70, 80]
SEEDS = [1, 2, 3, 4]

SHARED_HP = {
    "lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
    "warmup_epochs": 50, "constraint_epochs": 300, "pretrained": True,
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


def make_cfg(tight_tag, method, seed):
    hp = {**SHARED_HP, **PER_METHOD[method], "seed": seed}
    ds_config = dict(DS_META)
    parts = tight_tag.split("_")
    pair = (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)
    bmid = compute_base_model_id(
        MODEL, hp, dataset_mode=DS_NAME,
        data_dir=DS_META["data_dir"], dataset_config=ds_config,
    )
    return {
        "methodology": method,
        "model_name": MODEL,
        "constraint": list(pair),
        "constraint_tag": tight_tag,
        "dataset_mode": DS_NAME,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"paperv2_p2_{method}_{DS_NAME}_{tight_tag}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / tight_tag / method / f"seed_{seed}"
        ),
    }


def build():
    done = _scan_done_cells()
    print(f"Pre-scan: {len(done)} cells already completed.")
    cfgs, skipped = [], 0
    for l in L_VALS:
        for g in G_VALS:
            tight = f"L{l}_G{g}"
            for method in METHODS:
                for seed in SEEDS:
                    key = (DS_NAME, MODEL, 4, "loc_group", tight, method, seed)
                    if key in done:
                        skipped += 1
                        continue
                    cfgs.append(make_cfg(tight, method, seed))
    print(f"Target: 600 cells. Already done elsewhere: {skipped}. "
          f"Will queue: {len(cfgs)}.")
    save_configs(cfgs, output_dir=SWEEP_ROOT)


if __name__ == "__main__":
    build()
