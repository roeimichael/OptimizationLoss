"""G2-oct — Asymmetric tightness on OctMNIST (fills the missing regime-table rows).

The master regime table partitions the canonical corpus into constraint-geometry
regimes; the two asymmetric regimes (G<L global-tighter, G>L local-tighter) have
Derm (paperv2_phase2) and Tissue (g2_asym_tissue_aider) cells but NO OctMNIST cells.
This mirrors gen_g2_asym_tissue_aider exactly for octmnist:

    octmnist x MobileNetV3 x 20 off-diag (L,G) x 6 methods x 4 seeds = 480 cells.

Split into TWO sweep roots (one per asym direction) so each can be dispatched on
its own GPU via EXPERIMENT_DIR:
    results/pending_runs/g2_asym_oct_gl   -- G<L (global tighter), 10 tags
    results/pending_runs/g2_asym_oct_lg   -- G>L (local tighter),  10 tags

Frozen paper recipe verbatim (warmup=50, alpha_kl=0, fioretto_step 0.005,
hounie eta 0.01); warmups reuse the paper_final MobileNetV3/octmnist cache via
base_model_id. Run ON THE SERVER so the hash matches the cache environment.
"""
from pathlib import Path
import glob, json, os

from src.config_generators.generate_configs import (
    compute_base_model_id, save_configs,
)

DS_NAME = "octmnist"
DS_META = {
    "data_dir": "data/octmnist/slice_1", "num_classes": 4,
    "image_size": 224, "target_column": "label",
    "group_column": "synth_group", "constrained_class": 2,
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


def _sweep_root(l, g):
    return ("results/pending_runs/g2_asym_oct_gl" if g < l
            else "results/pending_runs/g2_asym_oct_lg")


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


def make_cfg(tight, method, seed, sweep_root):
    hp = {**SHARED_HP, **PER_METHOD[method], "seed": seed}
    ds_config = dict(DS_META)
    parts = tight.split("_")
    pair = (int(parts[0][1:]) / 100, int(parts[1][1:]) / 100)  # [local, global]
    bmid = compute_base_model_id(MODEL, hp, dataset_mode=DS_NAME,
                                 data_dir=DS_META["data_dir"], dataset_config=ds_config)
    return {
        "methodology": method, "model_name": MODEL,
        "constraint": list(pair), "constraint_tag": tight,
        "dataset_mode": DS_NAME, "dataset_config": ds_config,
        "hyperparams": hp, "base_model_id": bmid,
        "exp_name": f"g2oct_{method}_{tight}_seed{seed}",
        "experiment_path": str(
            Path(sweep_root) / DS_NAME / tight / method / f"seed_{seed}"
        ),
    }


def build():
    done = _scan_done_cells()
    print(f"Pre-scan: {len(done)} cells already completed.")
    by_root, skipped = {}, 0
    cls = DS_META["constrained_class"]; grp = DS_META["group_column"]
    for l in L_VALS:
        for g in G_VALS:
            if l == g:   # symmetric diagonal lives in paper_final
                continue
            tight = f"L{l}_G{g}"
            root = _sweep_root(l, g)
            for method in METHODS:
                for seed in SEEDS:
                    key = (DS_NAME, MODEL, cls, grp, tight, method, seed)
                    if key in done:
                        skipped += 1; continue
                    by_root.setdefault(root, []).append(
                        make_cfg(tight, method, seed, root))
    total = sum(len(v) for v in by_root.values())
    print(f"Target: 480 cells (20 asym tags x 6 methods x 4 seeds). "
          f"Already done: {skipped}. Will queue: {total}.")
    for root, cfgs in by_root.items():
        print(f"  {root}: {len(cfgs)}")
        save_configs(cfgs, output_dir=root)


if __name__ == "__main__":
    build()
