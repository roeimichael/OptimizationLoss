"""Phase 3 v3 — Table C backbone robustness, mobile/regnet edition.

Replaces the original Phase 3 (ResNet18+EfficientNetB0) which used backbones
now known to saturate derm at ep1 (see docs/REJECTED.md). The v3 set uses the
two confirmed-winner backbones beyond MobileNetV3 — MobileNetV2 (Blackwell
8-seed paired win on derm+aider) and RegNetY400MF (Blackwell 4-seed win on
aider, Hounie-only on derm) — and spans all three active datasets at the
representative symmetric tightness L50_G50.

Grid: {MobileNetV2, RegNetY400MF} × {tissuemnist, dermmnist, aider}
      × L50_G50 × 6 methods × 4 seeds = 144 cells.

Recipe (warmup_epochs=50, constraint_epochs=300) matches Phase 1 / Table A so
the new rows can be merged directly with MobileNetV3 L50_G50 cells already on
disk. Skips any cell that already has an evaluation_metrics.csv anywhere under
results/pending_runs/.

Configs land at
    results/pending_runs/paperv2_phase3_v3/{backbone}/{ds}/{method}/seed_{s}/
"""
from pathlib import Path
import glob, json, os

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/paperv2_phase3_v3"

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

BACKBONES = ["MobileNetV2", "RegNetY400MF"]
TIGHT = "L50_G50"
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

# Priority ordering for mid-run abort: dispositive cells first.
# RegNet derm + RegNet tissue come early — if TraLO loses both there we drop
# RegNet from the paper and fall back to V3+V2-only.
PRIORITY = [
    ("RegNetY400MF", "dermmnist"),     # dispositive: derm Hounie-only on Turing
    ("RegNetY400MF", "tissuemnist"),   # dispositive: untested
    ("MobileNetV2",  "dermmnist"),     # confirmation: Turing+Blackwell win
    ("MobileNetV2",  "aider"),         # confirmation: Turing+Blackwell win
    ("RegNetY400MF", "aider"),         # confirmation: Turing win
    ("MobileNetV2",  "tissuemnist"),   # confirmation: Turing tie
]


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


def make_cfg(backbone, ds_name, ds_meta, method, seed):
    hp = {**SHARED_HP, **PER_METHOD[method], "seed": seed}
    ds_config = dict(ds_meta)
    pair = _tight_pair(TIGHT)
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
        "exp_name": f"paperv2_p3v3_{backbone}_{method}_{ds_name}_{TIGHT}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / backbone / ds_name / method / f"seed_{seed}"
        ),
    }


def build():
    done = _scan_done_cells()
    print(f"Pre-scan: {len(done)} cells already completed across pending_runs/")
    cfgs, skipped = [], 0
    for backbone, ds_name in PRIORITY:
        ds_meta = DATASETS[ds_name]
        cls = ds_meta["constrained_class"]
        grp = ds_meta["group_column"]
        for method in METHODS:
            for seed in SEEDS:
                key = (ds_name, backbone, cls, grp, TIGHT, method, seed)
                if key in done:
                    skipped += 1
                    continue
                cfgs.append(make_cfg(backbone, ds_name, ds_meta, method, seed))
    print(f"Target: 144 cells (2 bb × 3 ds × 1 tight × 6 mthd × 4 seed). "
          f"Already done: {skipped}. Will queue: {len(cfgs)}.")
    save_configs(cfgs, output_dir=SWEEP_ROOT)


if __name__ == "__main__":
    build()
