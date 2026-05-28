"""Aggregate hybrid_v2 sweep results into comparison table."""
import csv
import json
from pathlib import Path
from collections import defaultdict


ROOT = Path("results/pending_runs/hybrid_v2")


def read_config(d):
    with open(d / "config.json") as f:
        return json.load(f)


def read_eval(d):
    p = d / "evaluation_metrics.csv"
    if not p.exists():
        return None
    out = {}
    with open(p) as f:
        rdr = csv.reader(f)
        next(rdr, None)
        for row in rdr:
            if len(row) >= 2:
                out[row[0]] = row[1]
    return out


def _f(d, k, default=0.0):
    v = d.get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def cell_key(cfg):
    method = cfg["methodology"]
    hp = cfg["hyperparams"]
    if method == "tralo_fioretto":
        mode = hp.get("hybrid_mode", "?")
        if mode == "symquad":
            return "symquad"
        if mode == "undershoot_hinge":
            return f"undershoot_b{hp.get('fior_beta', 0):.2f}"
        return f"hybrid_{mode}"
    return f"baseline_{method}"


rows = defaultdict(list)
for cfg_path in ROOT.rglob("config.json"):
    d = cfg_path.parent
    cfg = read_config(d)
    ev = read_eval(d)
    if ev is None:
        continue
    tight = cfg["constraint_tag"]
    cell = cell_key(cfg)
    rows[(cell, tight)].append({
        "f1": _f(ev, "F1 (Macro)"),
        "f1_c": _f(ev, "F1_Class4"),
        "acc": _f(ev, "Accuracy"),
        "ece": _f(ev, "ECE"),
        "flips": int(_f(ev, "Flips Required")),
        "raw_excess": int(_f(ev, "Raw Total Excess")),
        "sat_epoch": ev.get("Satisfaction Epoch", "N/A"),
        "soft_hard_gap": _f(ev, "Soft-Hard Gap Class4"),
    })

order = ["baseline_tralo", "symquad",
         "undershoot_b0.20", "undershoot_b0.50", "undershoot_b1.00"]

print(f"{'CELL':<22} {'TIGHT':<8} {'F1_M':>7} {'F1_C4':>7} {'ACC':>6} "
      f"{'ECE':>6} {'FLIPS':>6} {'RAW_EX':>7} {'GAP_C4':>7} {'SAT_EP':>9}")
print("-" * 96)
for cell in order:
    for tight in ["L20_G20", "L30_G30"]:
        runs = rows.get((cell, tight), [])
        if not runs:
            continue
        def avg(k): return sum(r[k] for r in runs) / len(runs)
        sats = "/".join(str(r["sat_epoch"]) for r in runs)
        print(f"{cell:<22} {tight:<8} {avg('f1'):>7.4f} {avg('f1_c'):>7.4f} "
              f"{avg('acc'):>6.4f} {avg('ece'):>6.4f} "
              f"{avg('flips'):>6.1f} {avg('raw_excess'):>7.1f} "
              f"{avg('soft_hard_gap'):>7.2f} {sats:>9}")
