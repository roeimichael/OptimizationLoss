"""Server-side: fill the MNV3 tissue/derm CANONICAL dose-response gaps (scripts/canon_gap_targets.csv)
exposed once off-class contamination is filtered out. Clones from canonical source configs:
  - same (ds,method,seed) at a different tag  -> keep base_model_id, change tag+constraint
  - else (ds,method) at a different seed       -> set seed, RECOMPUTE base_model_id, set tag+constraint
Warmups are method-agnostic and already cached for seeds 1-4 (trained methods ran them), so all
clones hit the cache. Writes to one GPU-lane root. CANONICAL source = cc==4 + correct group + warmup50
+ has eval + not an ablation sweep.
"""
import glob
import json
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.abspath("."))  # repo root (run from ~/OptimizationLoss)
from src.config_generators.generate_configs import compute_base_model_id  # noqa: E402

CANON = {"tissuemnist": (4, "synth_group"), "dermmnist": (4, "loc_group")}
BLOCK = {"smoke_reorg"}
CONSTRAINT = {"L20_G20": 0.2, "L30_G30": 0.3, "L40_G40": 0.4, "L50_G50": 0.5,
              "L60_G60": 0.6, "L70_G70": 0.7, "L80_G80": 0.8}
ROOT = "results/pending_runs/canon_gap_g2"
TARGETS = "scripts/canon_gap_targets.csv"


def sweep_of(p):
    parts = p.replace("\\", "/").split("/")
    return parts[parts.index("pending_runs") + 1] if "pending_runs" in parts else "?"


# index canonical MNV3 tissue/derm sources: by_exact[(ds,method,seed,tag)]=cfg; by_ms[(ds,method)] = list of cfgs
by_exact, by_ms = {}, {}
for p in glob.glob("results/pending_runs/**/config.json", recursive=True):
    if not os.path.exists(os.path.join(os.path.dirname(p), "evaluation_metrics.csv")):
        continue
    try:
        cfg = json.load(open(p))
    except Exception:
        continue
    if cfg.get("model_name") != "MobileNetV3":
        continue
    ds = cfg.get("dataset_mode")
    if ds not in CANON:
        continue
    dc = cfg.get("dataset_config", {}) or {}
    if (dc.get("constrained_class"), dc.get("group_column")) != CANON[ds]:
        continue
    hp = cfg.get("hyperparams", {}) or {}
    if hp.get("warmup_epochs") != 50:
        continue
    if sweep_of(cfg.get("experiment_path", "")) in BLOCK:
        continue
    method = cfg.get("methodology")
    seed = hp.get("seed")
    tag = cfg.get("constraint_tag")
    by_exact[(ds, method, seed, tag)] = cfg
    by_ms.setdefault((ds, method), []).append(cfg)

print(f"indexed canonical MNV3 sources: {len(by_exact)} cells, {len(by_ms)} (ds,method) groups")


def make_clone(cfg, ds, method, seed, tag, same_seed):
    cfg = json.loads(json.dumps(cfg))  # deep copy
    cfg["constraint"] = [CONSTRAINT[tag], CONSTRAINT[tag]]
    cfg["constraint_tag"] = tag
    for k in ("results", "status", "code_version"):
        cfg.pop(k, None)
    hp = cfg["hyperparams"]
    hp["seed"] = seed
    if not same_seed:  # cross-seed -> recompute base_model_id for the cached target-seed warmup
        cfg["base_model_id"] = compute_base_model_id(
            cfg["model_name"], hp, cfg["dataset_mode"], cfg["dataset_config"]["data_dir"],
            cfg["dataset_config"])
    ep = f"{ROOT}/{cfg['model_name']}/{ds}/{tag}/{method}/seed_{seed}"
    cfg["experiment_path"] = ep
    cfg["exp_name"] = f"canongap_{cfg['model_name']}_{ds}_{method}_{tag}_seed{seed}"
    os.makedirs(ep, exist_ok=True)
    json.dump(cfg, open(f"{ep}/config.json", "w"), indent=4)


df = pd.read_csv(TARGETS)
n, miss = 0, []
for _, r in df.iterrows():
    ds, method, seed, tag = r.dataset, r.method, int(r.seed), r.tag
    # priority: same-seed different-tag (no bmid recompute)
    src = next((by_exact[(ds, method, seed, t)] for t in CONSTRAINT if (ds, method, seed, t) in by_exact), None)
    if src is not None:
        make_clone(src, ds, method, seed, tag, same_seed=True)
        n += 1
        continue
    # else: any (ds,method) source from another seed -> recompute bmid
    pool = by_ms.get((ds, method))
    if pool:
        make_clone(pool[0], ds, method, seed, tag, same_seed=False)
        n += 1
        continue
    miss.append((ds, method, seed, tag))

print(f"wrote {n} canonical-gap clones; {len(miss)} missing sources")
for m in miss[:20]:
    print("  MISS", m)
c = sum(1 for _r, _d, fs in os.walk(ROOT) for f in fs if f == "config.json") if os.path.exists(ROOT) else 0
print(f"  {c} configs -> {ROOT}")
