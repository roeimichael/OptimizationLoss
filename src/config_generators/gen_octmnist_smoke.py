"""OCTMNIST hard-dataset smoke: does a genuinely hard task (warmup train acc
that resists memorization) restore TraLO's F1 edge, like TissueMNIST?

OCTMNIST: 4-class retinal OCT (0 CNV, 1 DME, 2 drusen, 3 normal), ~109K pool
subsampled to 20K (prep: data/octmnist/download_data.py). Constrained class =
DRUSEN (2), the minority pathology, with a synthetic binary group.

Grid (smoke):
    octmnist x MobileNetV3 x cls=2 x synth_group x {L20,L50}
    x {tralo, fioretto_ldf, hounie_rcl} x 2 seeds = 12 cells.

TraLO uses the canonical breakthrough recipe (undershoot_hinge + reset at sat,
alpha_kl=0). Same architecture/recipe as the tissue/derm headline so the only
moving part is dataset difficulty.

New configs land at
    results/pending_runs/octmnist_smoke/{tight}/{method}/seed_{s}/
"""
from pathlib import Path
import glob, json, os

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/octmnist_smoke"

DS_META = {
    "data_dir": "data/octmnist", "num_classes": 4,
    "image_size": 224, "target_column": "label",
    "group_column": "synth_group", "constrained_class": 2,
}
DS_NAME = "octmnist"
BACKBONE = "MobileNetV3"

TIGHTNESS = ["L20_G20", "L50_G50"]
SEEDS = [1, 2]

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
        ))
    return done


def make_cfg(tight_tag, method, seed):
    hp = {**SHARED_HP, **PER_METHOD[method], "seed": seed}
    ds_config = dict(DS_META)
    parts = tight_tag.split("_")
    pair = (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)
    bmid = compute_base_model_id(
        BACKBONE, hp, dataset_mode=DS_NAME,
        data_dir=DS_META["data_dir"], dataset_config=ds_config,
    )
    return {
        "methodology": method,
        "model_name": BACKBONE,
        "constraint": list(pair),
        "constraint_tag": tight_tag,
        "dataset_mode": DS_NAME,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"octsmoke_{BACKBONE}_{method}_{DS_NAME}_{tight_tag}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / tight_tag / method / f"seed_{seed}"),
    }


def build():
    done = _scan_done_cells()
    print(f"Pre-scan: {len(done)} cells already completed.")
    cfgs, skipped = [], 0
    for tight in TIGHTNESS:
        for method in METHODS:
            for seed in SEEDS:
                key = (DS_NAME, BACKBONE, 2, "synth_group", tight, method, seed)
                if key in done:
                    skipped += 1
                    continue
                cfgs.append(make_cfg(tight, method, seed))
    n_target = len(TIGHTNESS) * len(METHODS) * len(SEEDS)
    print(f"Target: {n_target} cells (1 backbone x {len(TIGHTNESS)} tight x "
          f"{len(METHODS)} mthd x {len(SEEDS)} seed). Already done: {skipped}. "
          f"Will queue: {len(cfgs)}.")
    save_configs(cfgs, output_dir=SWEEP_ROOT)


if __name__ == "__main__":
    build()
