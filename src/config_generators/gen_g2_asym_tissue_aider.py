"""G2 — Asymmetric tightness on TissueMNIST + AIDER (closes Limitation 2 part 1).

Mirror gen_paperv2_phase2 but for tissue + aider, off-diagonal only:
    {tissue, aider} × MobileNetV3 × 20 off-diag (L,G) × 6 methods × 4 seeds
    = 960 target cells.

Symmetric diagonal (L=G) already covered by Phase 1 (Table A); excluded.
"""
from pathlib import Path
import glob, json, os

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/g2_asym_tissue_aider"

DATASETS = {
    "tissuemnist": {"data_dir": "data/tissuemnist/slice_1", "num_classes": 8,
                    "image_size": 224, "target_column": "label",
                    "group_column": "synth_group", "constrained_class": 4},
    "aider":       {"data_dir": "data/aider/slice_1", "num_classes": 4,
                    "image_size": 224, "target_column": "label",
                    "group_column": "synth_group", "constrained_class": 0},
}
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
    "tralo": {"lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
              "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
              "penalty_mode": "both", "enable_ce_skip": True,
              "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
              "reset_optimizer_at_sat": True},
    "tralo_bounded": {"lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
                      "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
                      "penalty_mode": "both", "enable_ce_skip": True},
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


def make_cfg(ds_name, ds_meta, tight, method, seed):
    hp = {**SHARED_HP, **PER_METHOD[method], "seed": seed}
    ds_config = dict(ds_meta)
    parts = tight.split("_")
    pair = (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)
    bmid = compute_base_model_id(MODEL, hp, dataset_mode=ds_name,
                                 data_dir=ds_meta["data_dir"], dataset_config=ds_config)
    return {
        "methodology": method, "model_name": MODEL,
        "constraint": list(pair), "constraint_tag": tight,
        "dataset_mode": ds_name, "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "exp_name": f"g2_{method}_{ds_name}_{tight}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / ds_name / tight / method / f"seed_{seed}"
        ),
    }


def build():
    done = _scan_done_cells()
    print(f"Pre-scan: {len(done)} cells already completed.")
    cfgs, skipped = [], 0
    for ds_name, ds_meta in DATASETS.items():
        cls = ds_meta["constrained_class"]; grp = ds_meta["group_column"]
        for l in L_VALS:
            for g in G_VALS:
                if l == g:   # diagonal lives in Table A
                    continue
                tight = f"L{l}_G{g}"
                for method in METHODS:
                    for seed in SEEDS:
                        key = (ds_name, MODEL, cls, grp, tight, method, seed)
                        if key in done:
                            skipped += 1; continue
                        cfgs.append(make_cfg(ds_name, ds_meta, tight, method, seed))
    print(f"Target: 960 cells. Already done: {skipped}. Will queue: {len(cfgs)}.")
    save_configs(cfgs, output_dir=SWEEP_ROOT)


if __name__ == "__main__":
    build()
