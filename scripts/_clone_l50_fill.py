"""Fill the 27 missing warmup-50 L50 trained-method cells so the multi-backbone tables
pool over L30/L50/L70 uniformly (no silent 2-vs-3-level bias). Clone each target's existing
L70 config from the paper_backbones sweep -> L50 (constraint [0.5,0.5]), keeping base_model_id
+ full recipe intact so the warmup cache hits and the cell is directly comparable.

Targets (warmup=50, methods fioretto_ldf/hounie_rcl/tralo):
  derm/MobileNetV2  seeds 1-4   (12)
  derm/RegNetY400MF seeds 1-4   (12)
  tissue/MobileNetV2 seed 1     (3)   [seeds 2-4 already present]
Split into two GPU lanes by backbone. Run server-side from repo root.
"""
import json
import os

BASE = "results/pending_runs/paper_backbones"
METHODS = ["fioretto_ldf", "hounie_rcl", "tralo"]
TARGETS = [
    ("MobileNetV2", "dermmnist", [1, 2, 3, 4]),
    ("RegNetY400MF", "dermmnist", [1, 2, 3, 4]),
    ("MobileNetV2", "tissuemnist", [1]),
]
LANES = {0: "results/pending_runs/l50fill_g0", 1: "results/pending_runs/l50fill_g1"}


def lane_of(model):
    return 1 if model == "RegNetY400MF" else 0  # RegNet -> g1, MobileNetV2 -> g0


n, miss = 0, 0
for model, ds, seeds in TARGETS:
    for meth in METHODS:
        for sd in seeds:
            src = f"{BASE}/{model}/{ds}/L70_G70/{meth}/seed_{sd}/config.json"
            if not os.path.exists(src):
                print("MISSING SRC", src)
                miss += 1
                continue
            cfg = json.load(open(src))
            cfg["constraint"] = [0.5, 0.5]
            cfg["constraint_tag"] = "L50_G50"
            for k in ("results", "status", "code_version"):
                cfg.pop(k, None)
            root = LANES[lane_of(model)]
            ep = f"{root}/{model}/{ds}/L50_G50/{meth}/seed_{sd}"
            cfg["experiment_path"] = ep
            cfg["exp_name"] = f"l50fill_{model}_{meth}_{ds}_L50_G50_seed{sd}"
            os.makedirs(ep, exist_ok=True)
            json.dump(cfg, open(f"{ep}/config.json", "w"), indent=4)
            n += 1
print(f"wrote {n} configs ({miss} missing sources)")
for lane, root in LANES.items():
    c = sum(1 for _r, _d, fs in os.walk(root) for f in fs if f == "config.json") if os.path.exists(root) else 0
    print(f"  lane g{lane}: {c} configs -> {root}")
