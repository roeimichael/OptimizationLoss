"""AIDER asymmetric tightness — mirror of Phase 2 (Table B) but for AIDER.

Goal: validate that TraLO's advantage on AIDER holds across asymmetric
(L, G) tightness configurations, not just symmetric L=G. Mirrors the
derm Table B story for the second active dataset.

Grid: AIDER × MobileNetV3 × cls=0 × group=synth_group
      × 4 off-diagonal asymmetric pairs × 6 methods × 4 seeds = 96 cells.

We use a reduced 4-corner subset of the 5x5 grid (not the full 20-cell
off-diagonal set) to keep compute under 5h on Blackwell:
    (L20,G80) (L80,G20) (L30,G70) (L70,G30)

Configs land at
    results/pending_runs/aider_asym/{tight}/{method}/seed_{s}/
"""
from pathlib import Path
import glob, json, os

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/aider_asym"

DS_META = {
    "data_dir": "data/aider/slice_1", "num_classes": 4,
    "image_size": 224, "target_column": "label",
    "group_column": "synth_group", "constrained_class": 0,
}
DS_NAME = "aider"
MODEL = "MobileNetV3"

ASYM_PAIRS = [("L20_G80"), ("L80_G20"), ("L30_G70"), ("L70_G30")]
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


def _tight_pair(tag):
    parts = tag.split("_")
    return (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)


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
    pair = _tight_pair(tight_tag)
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
        "exp_name": f"aider_asym_{method}_{tight_tag}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / tight_tag / method / f"seed_{seed}"
        ),
    }


def build():
    done = _scan_done_cells()
    print(f"Pre-scan: {len(done)} cells already completed.")
    cfgs, skipped = [], 0
    cls = DS_META["constrained_class"]
    grp = DS_META["group_column"]
    for tight in ASYM_PAIRS:
        for method in METHODS:
            for seed in SEEDS:
                key = (DS_NAME, MODEL, cls, grp, tight, method, seed)
                if key in done:
                    skipped += 1
                    continue
                cfgs.append(make_cfg(tight, method, seed))
    print(f"Target: 96 cells (4 asym × 6 methods × 4 seeds). "
          f"Already done: {skipped}. Will queue: {len(cfgs)}.")
    save_configs(cfgs, output_dir=SWEEP_ROOT)


if __name__ == "__main__":
    build()
