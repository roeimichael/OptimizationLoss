"""Server-side: clone existing warmup-50 source configs into the missing L40_G40 / L60_G60
dose-response cells listed in scripts/l40l60_targets.csv. Keeps base_model_id + full recipe
intact (warmup cache hits; cells directly comparable). Builds a source index by scanning for
COMPLETED warmup-50 configs (those with a sibling evaluation_metrics.csv -> warmup cache exists),
so it is robust to the differing path layouts across sweeps. Splits into 2 GPU-lane roots.
"""
import glob
import json
import os
import pandas as pd

TARGETS = "scripts/l40l60_targets.csv"
LANES = {0: "results/pending_runs/l40l60_g0", 1: "results/pending_runs/l40l60_g1"}
CONSTRAINT = {"L40_G40": [0.4, 0.4], "L60_G60": [0.6, 0.6]}
SRC_PREF = ["L70_G70", "L50_G50", "L30_G30", "L20_G20", "L80_G80"]

# --- build source index: (model, ds, method, seed) -> {tag: config_path} for completed warmup-50 cells ---
index = {}
scanned = 0
for p in glob.glob("results/pending_runs/**/config.json", recursive=True):
    if not os.path.exists(os.path.join(os.path.dirname(p), "evaluation_metrics.csv")):
        continue  # only completed cells -> warmup cache guaranteed present
    try:
        cfg = json.load(open(p))
    except Exception:
        continue
    hp = cfg.get("hyperparams", {}) or {}
    if hp.get("warmup_epochs") != 50:
        continue
    tag = cfg.get("constraint_tag")
    if tag not in SRC_PREF:
        continue
    seed = hp.get("seed", cfg.get("seed"))
    key = (cfg.get("model_name"), cfg.get("dataset_mode"), cfg.get("methodology"),
           int(seed) if seed is not None else None)
    index.setdefault(key, {})[tag] = p
    scanned += 1
print(f"indexed {scanned} completed warmup-50 source cells; {len(index)} (model,ds,method,seed) keys")


def pick_source(model, ds, method, seed):
    d = index.get((model, ds, method, seed), {})
    for t in SRC_PREF:
        if t in d:
            return d[t]
    return None


df = pd.read_csv(TARGETS)
n, miss = 0, []
for _, r in df.iterrows():
    src = pick_source(r.model, r.dataset, r.method, int(r.seed))
    if not src:
        miss.append((r.model, r.dataset, r.method, int(r.seed)))
        continue
    cfg = json.load(open(src))
    cfg["constraint"] = CONSTRAINT[r.tag]
    cfg["constraint_tag"] = r.tag
    for k in ("results", "status", "code_version"):
        cfg.pop(k, None)
    root = LANES[int(r.lane)]
    ep = f"{root}/{r.model}/{r.dataset}/{r.tag}/{r.method}/seed_{int(r.seed)}"
    cfg["experiment_path"] = ep
    cfg["exp_name"] = f"l4060_{r.model}_{r.dataset}_{r.method}_{r.tag}_seed{int(r.seed)}"
    os.makedirs(ep, exist_ok=True)
    json.dump(cfg, open(f"{ep}/config.json", "w"), indent=4)
    n += 1

print(f"wrote {n} configs; {len(miss)} missing sources")
for m in miss[:30]:
    print("  MISS", m)
for lane, root in LANES.items():
    c = sum(1 for _r, _d, fs in os.walk(root) for f in fs if f == "config.json") if os.path.exists(root) else 0
    print(f"  lane g{lane}: {c} configs -> {root}")
