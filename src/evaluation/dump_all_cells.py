"""Dump every completed cell across all sweeps into one tidy CSV for plotting.

Output: docs/all_cells_raw.csv with columns:
    ds, model, cls, grp, tight, L, G, method, seed,
    f1m, f1w, acc, ece, brier, flips, sat, sat_epoch, phase

Canonical-TraLO filter applied (non-breakthrough tralo variants excluded).
De-duped on (ds, model, cls, grp, tight, method, seed).
This is the single source the paper session should read for all Phase 1-5
graphs.
"""
import csv, glob, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "pending_runs"
OUT = ROOT / "docs" / "all_cells_raw.csv"

METHODS = {"tralo", "tralo_bounded", "fioretto_ldf", "hounie_rcl",
           "danits_lp", "heuristic"}


def load_metrics(p):
    out = {}
    try:
        with open(p) as f:
            for r in csv.reader(f):
                if len(r) == 2:
                    out[r[0]] = r[1]
    except Exception:
        pass
    return out


def fnum(v):
    try:
        x = float(v)
        return "" if x != x else x
    except (TypeError, ValueError):
        return ""


def is_canonical_tralo(hp):
    return (hp.get("hybrid_mode") == "undershoot_hinge"
            and hp.get("reset_optimizer_at_sat") is True
            and hp.get("alpha_kl", 0.0) == 0.0
            and abs(hp.get("fior_beta", 0.0) - 0.5) < 1e-6
            and hp.get("penalty_mode") == "both"
            and hp.get("enable_ce_skip") is True)


def main():
    seen = set()
    rows = []
    for f in glob.glob(str(RUNS / "*/**/config.json"), recursive=True):
        ev = f.replace("config.json", "evaluation_metrics.csv")
        if not os.path.exists(ev):
            continue
        try:
            c = json.load(open(f))
        except Exception:
            continue
        method = c.get("methodology")
        if method not in METHODS:
            continue
        hp = c.get("hyperparams", {})
        if method == "tralo" and not is_canonical_tralo(hp):
            continue
        ds = c.get("dataset_mode")
        model = c.get("model_name")
        cls = c.get("dataset_config", {}).get("constrained_class")
        grp = c.get("dataset_config", {}).get("group_column")
        tight = c.get("constraint_tag")
        seed = hp.get("seed")
        key = (ds, model, cls, grp, tight, method, seed)
        if key in seen:
            continue
        seen.add(key)
        # phase from path
        phase = ""
        for part in Path(f).parts:
            if part.startswith("paperv2_") or part.startswith("paper400") \
               or part.startswith("expansion") or part in (
                   "arch_validation", "component_ablation", "kl_ablation"):
                phase = part
                break
        L = G = ""
        if tight and tight.startswith("L") and "_G" in tight:
            try:
                L = int(tight.split("_")[0][1:])
                G = int(tight.split("_")[1][1:])
            except ValueError:
                pass
        m = load_metrics(ev)
        rows.append({
            "ds": ds, "model": model, "cls": cls, "grp": grp,
            "tight": tight, "L": L, "G": G, "method": method, "seed": seed,
            "f1m": fnum(m.get("F1 (Macro)")),
            "f1w": fnum(m.get("F1 (Weighted)")),
            "acc": fnum(m.get("Accuracy")),
            "ece": fnum(m.get("ECE")),
            "brier": fnum(m.get("Brier Score")),
            "flips": fnum(m.get("Flips Required")),
            "sat": fnum(m.get("Raw All Satisfied")),
            "sat_epoch": fnum(m.get("Satisfaction Epoch")),
            "phase": phase,
        })
    cols = ["ds", "model", "cls", "grp", "tight", "L", "G", "method", "seed",
            "f1m", "f1w", "acc", "ece", "brier", "flips", "sat",
            "sat_epoch", "phase"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (str(r["ds"]), str(r["model"]),
                                             str(r["cls"]), str(r["grp"]),
                                             str(r["tight"]), r["method"],
                                             r["seed"] or 0)):
            w.writerow(r)
    print(f"wrote {OUT} with {len(rows)} cells")


if __name__ == "__main__":
    main()
