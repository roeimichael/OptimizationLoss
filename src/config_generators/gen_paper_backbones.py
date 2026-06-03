"""Paper backbone sweep — close MISSING_EXPERIMENTS.md G1 + G5.

Three backbones × three datasets × five symmetric tightness × six methods
× four seeds = 1,080 target cells. Skips cells whose evaluation_metrics.csv
already exists anywhere under results/pending_runs/.

Backbones:
    MobileNetV2     — Tier 1, F1 corroboration (G1, closes Limitation 3).
    RegNetY400MF    — Tier 2, cross-family generality (G5).
    ShuffleNetV2    — Tier 3, breadth check (G5).

HP recipe mirrors gen_paperv2_phase1 verbatim except for model_name.
"""
from pathlib import Path
import glob, json, os

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/paper_backbones"

DATASETS = {
    "tissuemnist": {"data_dir": "data/tissuemnist/slice_1", "num_classes": 8,
                    "image_size": 224, "target_column": "label",
                    "group_column": "synth_group", "constrained_class": 4},
    "dermmnist":   {"data_dir": "data/dermmnist/slice_1", "num_classes": 7,
                    "image_size": 224, "target_column": "label",
                    "group_column": "loc_group", "constrained_class": 4},
    "aider":       {"data_dir": "data/aider/slice_1", "num_classes": 4,
                    "image_size": 224, "target_column": "label",
                    "group_column": "synth_group", "constrained_class": 0},
}

MODELS = ["MobileNetV2", "RegNetY400MF", "ShuffleNetV2"]
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


def make_cfg(model, ds_name, ds_meta, tight_tag, method, seed):
    hp = {**SHARED_HP, **PER_METHOD[method], "seed": seed}
    ds_config = dict(ds_meta)
    pair = _tight_pair(tight_tag)
    bmid = compute_base_model_id(
        model, hp, dataset_mode=ds_name,
        data_dir=ds_meta["data_dir"], dataset_config=ds_config,
    )
    return {
        "methodology": method,
        "model_name": model,
        "constraint": list(pair),
        "constraint_tag": tight_tag,
        "dataset_mode": ds_name,
        "dataset_config": ds_config,
        "hyperparams": hp,
        "base_model_id": bmid,
        "exp_name": f"paper_bb_{model}_{method}_{ds_name}_{tight_tag}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / model / ds_name / tight_tag / method / f"seed_{seed}"
        ),
    }


def build():
    done = _scan_done_cells()
    print(f"Pre-scan: {len(done)} cells already completed across pending_runs/")

    cfgs = []
    skipped = 0
    for model in MODELS:
        for ds_name, ds_meta in DATASETS.items():
            cls = ds_meta["constrained_class"]
            grp = ds_meta["group_column"]
            for tight in TIGHTNESS:
                for method in METHODS:
                    for seed in SEEDS:
                        key = (ds_name, model, cls, grp, tight, method, seed)
                        if key in done:
                            skipped += 1
                            continue
                        cfgs.append(
                            make_cfg(model, ds_name, ds_meta, tight, method, seed)
                        )
    print(f"Target: {len(MODELS)*len(DATASETS)*len(TIGHTNESS)*len(METHODS)*len(SEEDS)} cells. "
          f"Already done: {skipped}. Will queue: {len(cfgs)}.")
    save_configs(cfgs, output_dir=SWEEP_ROOT)


if __name__ == "__main__":
    build()
