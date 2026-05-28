"""Aggregate hybrid_v3 results (Adam-state fix variants)."""
import csv
import json
from pathlib import Path
from collections import defaultdict


ROOT = Path("results/pending_runs/hybrid_v3")


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
        fix = "resetAdam" if hp.get("reset_optimizer_at_sat") else ""
        if hp.get("post_sat_optimizer", "adam") != "adam":
            fix = "sgd"
        beta = hp.get("fior_beta", 0)
        if mode == "symquad":
            return f"symquad_{fix}" if fix else "symquad"
        if mode == "undershoot_hinge":
            return f"undershoot_b{beta:.2f}_{fix}" if fix else f"undershoot_b{beta:.2f}"
        return f"hybrid_{mode}"
    return f"baseline_{method}"


rows = defaultdict(list)
for cfg_path in ROOT.rglob("config.json"):
    d = cfg_path.parent
    cfg = json.load(open(cfg_path))
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
        "gap": _f(ev, "Soft-Hard Gap Class4"),
    })

order = [
    "baseline_tralo",
    "symquad_resetAdam",
    "symquad_sgd",
    "undershoot_b0.50_resetAdam",
    "undershoot_b0.50_sgd",
]

print(f"{'CELL':<28} {'TIGHT':<8} {'F1_M':>7} {'F1_C4':>7} {'ACC':>6} "
      f"{'ECE':>6} {'FLIPS':>6} {'RAW_EX':>7} {'GAP':>6} {'SAT_EP':>9}")
print("-" * 100)
for cell in order:
    for tight in ["L20_G20", "L30_G30"]:
        runs = rows.get((cell, tight), [])
        if not runs:
            continue
        def avg(k): return sum(r[k] for r in runs) / len(runs)
        sats = "/".join(str(r["sat_epoch"]) for r in runs)
        print(f"{cell:<28} {tight:<8} {avg('f1'):>7.4f} {avg('f1_c'):>7.4f} "
              f"{avg('acc'):>6.4f} {avg('ece'):>6.4f} "
              f"{avg('flips'):>6.1f} {avg('raw_excess'):>7.1f} "
              f"{avg('gap'):>6.2f} {sats:>9}")
