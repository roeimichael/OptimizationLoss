"""Phase 5 of paper plan v2 — Table E (group-column ablation on dermmnist).

Grid:
    DermMNIST × MobileNetV3 × cls=4 (MEL) × group=sex
    × 5 symmetric tightness × 6 methods × 4 seeds = 120 target cells.

The loc_group arm of Table E is already covered by Phase 1 (Table A). This
phase adds the `sex` grouping so the ablation can compare a weak-imbalance
group definition (sex) against the strong one (loc_group).

New configs land at
    results/pending_runs/paperv2_phase5/{tight}/{method}/seed_{s}/
"""
from pathlib import Path
import glob, json, os

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/paperv2_phase5"

DS_NAME = "dermmnist"
MODEL = "MobileNetV3"
DS_META = {
    "data_dir": "data/dermmnist/slice_1", "num_classes": 7,
    "image_size": 224, "target_column": "label",
    "group_column": "sex", "constrained_class": 4,
}
TIGHTNESS = ["L20_G20", "L30_G30", "L50_G50", "L70_G70", "L80_G80"]
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
        "methodology": method, "model_name": MODEL,
        "constraint": list(pair), "constraint_tag": tight_tag,
        "dataset_mode": DS_NAME, "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "exp_name": f"paperv2_p5_{method}_{DS_NAME}_sex_{tight_tag}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / tight_tag / method / f"seed_{seed}"
        ),
    }


def build():
    done = _scan_done_cells()
    print(f"Pre-scan: {len(done)} cells already completed.")
    cfgs, skipped = [], 0
    for tight in TIGHTNESS:
        for method in METHODS:
            for seed in SEEDS:
                key = (DS_NAME, MODEL, 4, "sex", tight, method, seed)
                if key in done:
                    skipped += 1
                    continue
                cfgs.append(make_cfg(tight, method, seed))
    print(f"Target: 120 cells (5 tight × 6 mthd × 4 seed). "
          f"Already done: {skipped}. Will queue: {len(cfgs)}.")
    save_configs(cfgs, output_dir=SWEEP_ROOT)


if __name__ == "__main__":
    build()
