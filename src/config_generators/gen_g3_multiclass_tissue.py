"""G3 — Multi-class robustness on TissueMNIST (closes Limitation 2 part 2).

Mirror gen_paperv2_phase4 (Table D) but on tissue. Test prevalences:
    cls 0 (CDI)  ≈  4.5%   cls 1 (CDS)  ≈  6.4%   cls 2 (CST)  ≈ 22.2%
    cls 3 (EPI)  ≈ 16.2%   cls 4 (GE)*  ≈  7.1%   cls 5 (PTC)  ≈ 22.0%
    cls 6 (STR)  ≈  8.5%   cls 7 (TUB)  ≈ 13.2%
    (*) cls 4 is the headline TissueMNIST class — already in Table A; excluded.

Pick three alt classes around the headline prevalence band where K
remains > 100 at L20: cls 2, cls 5, cls 7.

    TissueMNIST × MobileNetV3 × cls ∈ {2, 5, 7} × group=synth_group
    × 5 sym tight × 6 methods × 4 seeds = 360 target cells.
"""
from pathlib import Path
import glob, json, os

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

SWEEP_ROOT = "results/pending_runs/g3_multiclass_tissue"

DS_NAME = "tissuemnist"
MODEL = "MobileNetV3"
DS_BASE = {
    "data_dir": "data/tissuemnist/slice_1", "num_classes": 8,
    "image_size": 224, "target_column": "label",
    "group_column": "synth_group",
}
CLASSES = [2, 5, 7]
TIGHTNESS = ["L20_G20", "L30_G30", "L50_G50", "L70_G70", "L80_G80"]
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


def make_cfg(cls, tight, method, seed):
    hp = {**SHARED_HP, **PER_METHOD[method], "seed": seed}
    ds_config = {**DS_BASE, "constrained_class": cls}
    parts = tight.split("_")
    pair = (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)
    bmid = compute_base_model_id(MODEL, hp, dataset_mode=DS_NAME,
                                 data_dir=DS_BASE["data_dir"], dataset_config=ds_config)
    return {
        "methodology": method, "model_name": MODEL,
        "constraint": list(pair), "constraint_tag": tight,
        "dataset_mode": DS_NAME, "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "exp_name": f"g3_{method}_{DS_NAME}_cls{cls}_{tight}_seed{seed}",
        "experiment_path": str(
            Path(SWEEP_ROOT) / f"cls_{cls}" / tight / method / f"seed_{seed}"
        ),
    }


def build():
    done = _scan_done_cells()
    print(f"Pre-scan: {len(done)} cells already completed.")
    cfgs, skipped = [], 0
    for cls in CLASSES:
        for tight in TIGHTNESS:
            for method in METHODS:
                for seed in SEEDS:
                    key = (DS_NAME, MODEL, cls, "synth_group", tight, method, seed)
                    if key in done:
                        skipped += 1; continue
                    cfgs.append(make_cfg(cls, tight, method, seed))
    print(f"Target: 360 cells (3 cls × 5 tight × 6 mthd × 4 seed). "
          f"Already done: {skipped}. Will queue: {len(cfgs)}.")
    save_configs(cfgs, output_dir=SWEEP_ROOT)


if __name__ == "__main__":
    build()
