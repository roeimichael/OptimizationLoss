"""Aggregate hybrid_v1 sweep results into a comparison table.

Per cell: (mode, beta_or_step) x (tightness) x seed -- mean F1 + std, accuracy,
posthoc flips, satisfaction epoch, min_total_excess, restore kind.
"""
import csv
import json
from pathlib import Path
from collections import defaultdict


ROOT = Path("results/pending_runs/hybrid_v1")


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
        next(rdr, None)  # header "Metric,Value"
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
        if mode == "single_lambda":
            return f"hybrid_singleL_b{hp.get('fior_beta', 0):.2f}"
        return f"hybrid_dualL_s{hp.get('fior_step_size', 0):.3f}"
    elif method == "tralo":
        return "baseline_tralo"
    elif method == "fioretto_ldf":
        return "baseline_fior"
    return method


rows = defaultdict(list)
for cfg_path in ROOT.rglob("config.json"):
    d = cfg_path.parent
    cfg = read_config(d)
    ev = read_eval(d)
    if ev is None:
        continue
    tight = cfg["constraint_tag"]
    cell = cell_key(cfg)
    seed = cfg["hyperparams"].get("seed", 0)
    rows[(cell, tight)].append({
        "seed": seed,
        "f1": _f(ev, "F1 (Macro)"),
        "f1_c": _f(ev, "F1_Class4"),
        "acc": _f(ev, "Accuracy"),
        "ece": _f(ev, "ECE"),
        "brier": _f(ev, "Brier Score"),
        "flips": int(_f(ev, "Flips Required")),
        "raw_excess": int(_f(ev, "Raw Total Excess")),
        "sat_epoch": ev.get("Satisfaction Epoch", "N/A"),
        "min_excess": ev.get("Min Total Excess", "N/A"),
        "restore_kind": ev.get("Restore Kind", ""),
    })

# Order cells: baselines first, then single_lambda variants, then dual_lambda.
order = ["baseline_tralo", "baseline_fior",
         "hybrid_singleL_b0.05", "hybrid_singleL_b0.20", "hybrid_singleL_b0.50",
         "hybrid_dualL_s0.001", "hybrid_dualL_s0.005", "hybrid_dualL_s0.020"]

print(f"{'CELL':<24} {'TIGHT':<8} {'F1_M':>7} {'F1_C4':>7} {'ACC':>6} "
      f"{'ECE':>6} {'FLIPS':>6} {'RAW_EX':>7} {'SAT_EP':>9}")
print("-" * 90)
for cell in order:
    for tight in ["L20_G20", "L30_G30"]:
        runs = rows.get((cell, tight), [])
        if not runs:
            continue
        def avg(k):
            xs = [r[k] for r in runs]
            return sum(xs) / len(xs)
        f1_m = avg("f1")
        f1c = avg("f1_c")
        acc = avg("acc")
        ece = avg("ece")
        flips = avg("flips")
        raw_ex = avg("raw_excess")
        sats = [str(r["sat_epoch"]) for r in runs]
        sat_str = "/".join(sats)
        print(f"{cell:<24} {tight:<8} {f1_m:>7.4f} {f1c:>7.4f} {acc:>6.4f} "
              f"{ece:>6.4f} {flips:>6.1f} {raw_ex:>7.1f} {sat_str:>9}")
