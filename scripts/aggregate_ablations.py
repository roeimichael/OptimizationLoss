"""Aggregate KL anchor + component leave-one-out ablations."""
import csv
import json
from pathlib import Path
from collections import defaultdict


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
    v = (d or {}).get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def aggregate(root: Path, group_key_fn, header: str):
    rows = defaultdict(list)
    for cfg_path in root.rglob("config.json"):
        d = cfg_path.parent
        cfg = json.load(open(cfg_path))
        ev = read_eval(d)
        if ev is None:
            continue
        key = group_key_fn(cfg, d)
        rows[key].append({
            "f1": _f(ev, "F1 (Macro)"),
            "f1_c": _f(ev, "F1_Class4"),
            "flips": int(_f(ev, "Flips Required")),
            "rex": int(_f(ev, "Raw Total Excess")),
            "sat": ev.get("Satisfaction Epoch", "N/A"),
        })
    print(header)
    print(f"{'Group':<48} {'F1_M':>7} {'F1_C4':>7} {'FLIPS':>6} {'RAW_EX':>7} {'n':>3}")
    print("-" * 84)
    for key in sorted(rows.keys()):
        runs = rows[key]
        avg = lambda k: sum(r[k] for r in runs) / len(runs)
        print(f"{str(key):<48} {avg('f1'):>7.4f} {avg('f1_c'):>7.4f} "
              f"{avg('flips'):>6.1f} {avg('rex'):>7.1f} {len(runs):>3}")
    print()


# ---- KL ablation ----
KL_ROOT = Path("results/pending_runs/kl_ablation")
if KL_ROOT.exists():
    def kl_key(cfg, d):
        return (cfg["dataset_mode"], cfg["constraint_tag"],
                f"alpha_kl={cfg['hyperparams'].get('alpha_kl', 0):.2f}")
    aggregate(KL_ROOT, kl_key, "=== KL anchor ablation (full TraLO, alpha_kl varied) ===")


# ---- Component leave-one-out ablation ----
COMP_ROOT = Path("results/pending_runs/component_ablation")
if COMP_ROOT.exists():
    def comp_key(cfg, d):
        # variant name is the second-to-last dir component
        variant = d.parent.name
        return (cfg["dataset_mode"], cfg["constraint_tag"], variant)
    aggregate(COMP_ROOT, comp_key, "=== Component leave-one-out ablation ===")
