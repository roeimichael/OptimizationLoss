"""Cross-dataset class-rotation analysis: test TraLO advantage mechanism.

Reads results from:
  results/pending_runs/aider_rotation_full/
  results/pending_runs/derm_rotation_full/
  results/pending_runs/tissue_rotation_full/

For each (dataset, constrained class):
  - mean F1 + std per method (across 3 seeds)
  - per-class warmup F1 on the constrained class (from final_predictions of warmup-only)
  - paired d_F1 = TraLO - mean(danits_lp, heuristic) [LP/heuristic comparison]
  - paired d_F1 vs Fioretto, Hounie

Hypothesis: d_F1 vs LP/heuristic should DECREASE as the constrained class's
warmup F1 (or warmup-correct-prediction-rate) INCREASES.

Output: CSV + console summary + matplotlib scatter (d_F1 vs warmup_class_F1).
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_CSV = "scripts/_paper_agg/rotation_mechanism.csv"
OUT_PNG = "scripts/_paper_agg/rotation_mechanism.png"

ROOTS = {
    "aider": "results/pending_runs/aider_rotation_full/MobileNetV3",
    "derm": "results/pending_runs/derm_rotation_full/MobileNetV3",
    "tissue": "results/pending_runs/tissue_rotation_full/MobileNetV3",
}


def read_metrics(path):
    if not os.path.exists(path):
        return None
    m = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            m[r["Metric"]] = r["Value"]
    return m


def warmup_class_f1(d, constrained_cls):
    """Compute warmup-only F1 on constrained class.

    Uses the heuristic cell's final_predictions_raw.csv (raw preds = post-warmup,
    before post-hoc adjustment). All methods share the same cached warmup, so
    the raw warmup predictions are identical.
    """
    raw = os.path.join(d, "final_predictions_raw.csv")
    if not os.path.exists(raw):
        return None
    tp = fp = fn = 0
    with open(raw) as f:
        for r in csv.DictReader(f):
            pred = int(r.get("Predicted_Label", r.get("pred_raw", r.get("y_pred", -1))))
            true = int(r.get("True_Label", r.get("y_true", -1)))
            if pred == constrained_cls and true == constrained_cls:
                tp += 1
            elif pred == constrained_cls and true != constrained_cls:
                fp += 1
            elif pred != constrained_cls and true == constrained_cls:
                fn += 1
    if tp == 0:
        return 0.0
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    return 2 * p * r / (p + r) if (p + r) else 0


def scan_dataset(ds, root):
    rows = []
    for cell in sorted(glob.glob(f"{root}/*/*/seed_*")):
        parts = cell.split("/")
        cfg, method, seed = parts[-3], parts[-2], parts[-1]
        # cfg looks like "constrained{N}_{role}_{TIGHT}"
        try:
            cls_idx = int(cfg.split("_")[0].replace("constrained", ""))
        except (ValueError, IndexError):
            continue
        m = read_metrics(os.path.join(cell, "evaluation_metrics.csv"))
        if not m:
            continue
        rows.append({
            "ds": ds, "cls": cls_idx, "cfg": cfg, "method": method, "seed": seed,
            "f1": float(m.get("F1 (Macro)", 0)),
            "flips": float(m.get("Flips Required", 0)),
            "sat": int(m.get("Raw All Satisfied", "0") == "1"),
            "cell": cell,
        })
    return rows


def main():
    all_rows = []
    for ds, root in ROOTS.items():
        if not os.path.exists(root):
            print(f"SKIP {ds}: no data at {root}")
            continue
        all_rows.extend(scan_dataset(ds, root))

    if not all_rows:
        print("No data found. Did the experiments finish?")
        sys.exit(1)

    # Aggregate F1 per (ds, cls, method) across seeds
    by_key = defaultdict(list)
    for r in all_rows:
        by_key[(r["ds"], r["cls"], r["method"])].append(r["f1"])

    # Warmup F1 per (ds, cls): pick first seed's heuristic cell
    warmup_f1 = {}
    for (ds, cls, method), _ in by_key.items():
        if method != "heuristic":
            continue
        key = (ds, cls)
        if key in warmup_f1:
            continue
        cells = [r for r in all_rows
                 if r["ds"] == ds and r["cls"] == cls and r["method"] == "heuristic"]
        if cells:
            f1 = warmup_class_f1(cells[0]["cell"], cls)
            warmup_f1[key] = f1

    # Console summary
    print(f"\n{'ds':6s} {'cls':4s} {'method':14s} {'F1_mean':>8s} {'F1_std':>7s} {'n':>3s}")
    print("-" * 60)
    for (ds, cls, method), f1s in sorted(by_key.items()):
        mean = np.mean(f1s); std = np.std(f1s)
        print(f"{ds:6s} {cls:>4d} {method:14s} {mean:8.4f} {std:7.4f} {len(f1s):>3d}")

    # d_F1 computations
    print(f"\n=== d_F1 (TraLO - baseline) per (ds, cls) ===")
    print(f"{'ds':6s} {'cls':>4s} {'warmup_F1':>10s} {'vs_LP+heur':>12s} {'vs_fio':>8s} {'vs_hou':>8s}")
    print("-" * 60)
    summary_rows = []
    for ds, cls in sorted(set((r["ds"], r["cls"]) for r in all_rows)):
        tr = by_key.get((ds, cls, "tralo"), [])
        if not tr:
            continue
        tr_mean = np.mean(tr)
        d = {}
        for bl in ["fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]:
            bl_vals = by_key.get((ds, cls, bl), [])
            if bl_vals:
                d[bl] = tr_mean - np.mean(bl_vals)
        lp_heur = [d.get("danits_lp"), d.get("heuristic")]
        lp_heur = [v for v in lp_heur if v is not None]
        d_lp = np.mean(lp_heur) if lp_heur else None
        wf1 = warmup_f1.get((ds, cls), None)
        wf1_str = f"{wf1:.4f}" if wf1 is not None else "n/a"
        d_lp_str = f"{d_lp:+.4f}" if d_lp is not None else "n/a"
        print(f"{ds:6s} {cls:>4d} {wf1_str:>10s} {d_lp_str:>12s} "
              f"{d.get('fioretto_ldf', float('nan')):+8.4f} "
              f"{d.get('hounie_rcl', float('nan')):+8.4f}")
        summary_rows.append({
            "ds": ds, "cls": cls, "warmup_class_f1": wf1,
            "d_vs_lp_heur": d_lp,
            "d_vs_fio": d.get("fioretto_ldf"),
            "d_vs_hou": d.get("hounie_rcl"),
            "tralo_f1": tr_mean,
        })

    # Save CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        if summary_rows:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    print(f"\nSaved {OUT_CSV}")

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    markers = {"aider": "o", "derm": "s", "tissue": "^"}
    colors = {"aider": "C0", "derm": "C1", "tissue": "C2"}
    for ax, ykey, ylab in [
        (axes[0], "d_vs_lp_heur", "d_F1 = TraLO - mean(LP, heuristic)"),
        (axes[1], "d_vs_fio", "d_F1 = TraLO - Fioretto"),
    ]:
        for r in summary_rows:
            if r["warmup_class_f1"] is None or r[ykey] is None:
                continue
            ax.scatter(r["warmup_class_f1"], r[ykey],
                       marker=markers[r["ds"]], color=colors[r["ds"]],
                       s=120, alpha=0.8, edgecolor="black",
                       label=r["ds"])
            ax.annotate(f"{r['ds'][0]}{r['cls']}",
                        (r["warmup_class_f1"], r[ykey]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("Warmup F1 on constrained class")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        # dedupe legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    fig.suptitle("TraLO advantage vs warmup quality on constrained class")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
