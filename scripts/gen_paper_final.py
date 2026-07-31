"""Generate the COMPLETE AAAI paper scope as ONE clean, dated sweep
(results/pending_runs/paper_final/) under the committed code, so every reported number
comes from a single coherent batch with one code_version (no stitching across months).
Canonical dataset_config + exact TraLO_fix recipe + compute_base_model_id (warmups hit the
deterministic cache; cache miss just retrains, still correct). Partitioned into lane0/1/2
subdirs for 3-GPU dispatch; sweep name stays 'paper_final' for every cell.

CONCLUSIVE GRID (1944 cells): 3 backbones x 3 datasets x 9 symmetric levels x 6 methods x 4 seeds.
  {MobileNetV3, RegNetY400MF, ViTB16} x {tissue, derm, oct} x 6 methods
    x {L10,L20,L30,L40,L50,L60,L70,L80,L90} x seeds 1-4 = 3*3*6*9*4 = 1944.
"""
import json
import os
import sys
sys.path.insert(0, os.path.abspath("."))  # repo root (run from ~/OptimizationLoss)
from src.config_generators.generate_configs import compute_base_model_id  # noqa: E402

DS_CFG = {
    "tissuemnist": {"data_dir": "data/tissuemnist/slice_1", "num_classes": 8, "image_size": 224,
                    "target_column": "label", "group_column": "synth_group", "constrained_class": 4},
    "dermmnist": {"data_dir": "data/dermmnist/slice_1", "num_classes": 7, "image_size": 224,
                  "target_column": "label", "group_column": "loc_group", "constrained_class": 4},
    "octmnist": {"data_dir": "data/octmnist/slice_1", "num_classes": 4, "image_size": 224,
                 "target_column": "label", "group_column": "synth_group", "constrained_class": 2},
}
SHARED_HP = {"lr": 1e-4, "lr_constraint": 5e-6, "dropout": 0.3, "batch_size": 64,
             "warmup_epochs": 50, "constraint_epochs": 300, "pretrained": True,
             "class_weighted_ce": False, "constraint_chunk_size": 256}
PER_METHOD = {
    "heuristic": {}, "danits_lp": {},
    "fioretto_ldf": {"fioretto_step_size": 0.005},
    "hounie_rcl": {"hounie_eta_lambda": 0.01, "hounie_eta_u": 0.01, "hounie_alpha": 10.0},
    "tralo_bounded": {"lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
                      "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
                      "penalty_mode": "both", "enable_ce_skip": True},
    "tralo": {"lambda_global": 0.05, "lambda_local": 0.05, "lambda_step": 0.002,
              "initial_rho": 5.0, "rho_target": 100.0, "alpha_kl": 0.0,
              "penalty_mode": "both", "enable_ce_skip": True,
              "hybrid_mode": "undershoot_hinge", "fior_beta": 0.50,
              "reset_optimizer_at_sat": True},
}
ALL_TAGS = ["L10_G10", "L20_G20", "L30_G30", "L40_G40", "L50_G50",
            "L60_G60", "L70_G70", "L80_G80", "L90_G90"]
SEEDS = [1, 2, 3, 4]
DSETS = ["tissuemnist", "dermmnist", "octmnist"]
SCOPE = [  # (model, datasets, tags) — conclusive grid: 3 backbones x 3 ds x 9 levels x 6 methods x 4 seeds
    ("MobileNetV3", DSETS, ALL_TAGS),
    ("RegNetY400MF", DSETS, ALL_TAGS),
    ("ViTB16", DSETS, ALL_TAGS),
]
ROOT = "results/pending_runs/paper_final"
NLANES = 3


def pair(tag):
    p = tag.split("_")
    return [int(p[0][1:]) / 100, int(p[1][1:]) / 100]


cells = []
for model, dsets, tags in SCOPE:
    for ds in dsets:
        for tag in tags:
            for method in PER_METHOD:
                for seed in SEEDS:
                    cells.append((model, ds, tag, method, seed))
# spread ViT (one model) + slow methods across lanes via method/seed-major sort + round-robin
cells.sort(key=lambda c: (c[3], c[4], c[0], c[1], c[2]))

n = 0
lane_counts = [0, 0, 0]
for i, (model, ds, tag, method, seed) in enumerate(cells):
    lane = i % NLANES
    dc = DS_CFG[ds]
    hp = {**SHARED_HP, **PER_METHOD[method], "seed": seed}
    bmid = compute_base_model_id(model, hp, ds, dc["data_dir"], dc)
    ep = f"{ROOT}/lane{lane}/{model}/{ds}/{tag}/{method}/seed_{seed}"
    cfg = {"methodology": method, "model_name": model, "constraint": pair(tag),
           "constraint_tag": tag, "dataset_mode": ds, "dataset_config": dc,
           "hyperparams": hp, "base_model_id": bmid,
           "exp_name": f"paperfinal_{model}_{ds}_{method}_{tag}_seed{seed}",
           "experiment_path": ep}
    os.makedirs(ep, exist_ok=True)
    json.dump(cfg, open(f"{ep}/config.json", "w"), indent=4)
    n += 1
    lane_counts[lane] += 1

print(f"wrote {n} configs -> {ROOT}/lane{{0,1,2}}")
print(f"per-lane counts: {lane_counts}")
print(f"ViT cells per lane: {[sum(1 for j,c in enumerate(cells) if c[0]=='ViTB16' and j%NLANES==L) for L in range(NLANES)]}")
