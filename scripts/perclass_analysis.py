"""Per-class + per-group + per-confidence-bucket analysis.

The macro-F1 story is regime-noisy. This script digs deeper to find
where TraLO actually contributes:

1. **Constrained-class F1**: does TraLO improve the F1 of the class
   we're capping? (TraLO targets this; post-hoc just trims)
2. **Collateral damage**: does TraLO hurt the F1 of unconstrained
   classes? (mechanism explanation for macro-F1 ties)
3. **Per-group F1 on constrained class** (when group_column has structure)
4. **Confidence calibration split** (gap between correct/incorrect)
5. **Constraint cell behavior**: what fraction of samples does TraLO
   re-assign vs post-hoc top-K?

Reads ALL per-cell evaluation_metrics.csv across contamination_clean +
contamination_<ds> + other key sweeps. Aggregates per (dataset, sigma,
tight, method) and emits:
  - paper/HANDOFF/tables/perclass_summary.csv
  - paper/HANDOFF/figures/perclass_v1/*.png
"""
import csv
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_FIG = Path("paper/HANDOFF/figures/perclass_v1")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TBL = Path("paper/HANDOFF/tables/perclass_summary.csv")

METHOD_COLORS = {
    "tralo":        "#1f77b4",
    "fioretto_ldf": "#ff7f0e",
    "hounie_rcl":   "#d62728",
    "danits_lp":    "#2ca02c",
    "heuristic":    "#8c564b",
}
METHOD_ORDER = ["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]
DATASETS = ["tissuemnist", "dermmnist", "aider"]
TIGHTS = ["L20_G20", "L30_G30", "L50_G50", "L70_G70"]

SWEEPS = [
    ("results/pending_runs/contamination_clean",     "clean"),
    ("results/pending_runs/contamination_tissuemnist","contam"),
    ("results/pending_runs/contamination_dermmnist", "contam"),
    ("results/pending_runs/contamination_aider",     "contam"),
]


def read_metrics(p):
    out = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            out[row["Metric"]] = row["Value"]
    return out


def collect():
    """Per-seed rows including per-class F1/Precision/Recall."""
    rows = []
    for root, tag in SWEEPS:
        if not os.path.isdir(root): continue
        for cfg_p in glob.glob(f"{root}/**/config.json", recursive=True):
            try:
                with open(cfg_p) as f: cfg = json.load(f)
                m = read_metrics(cfg_p.replace("config.json","evaluation_metrics.csv"))
            except Exception: continue
            ds = cfg["dataset_mode"]
            tight = cfg["constraint_tag"]
            method = cfg["methodology"]
            seed = cfg["hyperparams"]["seed"]
            cls_cstr = cfg["dataset_config"]["constrained_class"]
            n_cls = cfg["dataset_config"]["num_classes"]
            if tag == "clean":
                sigma = 0.0
            else:
                try:
                    parts = cfg["experiment_path"].split("/")
                    sigma_tag = next(p for p in parts if p.startswith("sigma"))
                    sigma = int(sigma_tag[5:]) / 100.0
                except StopIteration:
                    continue
            try:
                row = {
                    "dataset": ds, "sigma": sigma, "tight": tight,
                    "method": method, "seed": seed,
                    "cls_constrained": cls_cstr,
                    "macro_f1":  float(m["F1 (Macro)"]),
                    "macro_pre": float(m["Precision (Macro)"]),
                    "macro_rec": float(m["Recall (Macro)"]),
                    "acc":       float(m["Accuracy"]),
                    "ece":       float(m["ECE"]),
                    "brier":     float(m["Brier Score"]),
                    "flips":     float(m["Flips Required"]),
                    "sat":       1 if m.get("Raw All Satisfied","0")=="1" else 0,
                    "conf_correct":   float(m.get("Confidence (Correct)","nan")),
                    "conf_incorrect": float(m.get("Confidence (Incorrect)","nan")),
                    "conf_gap":       float(m.get("Confidence Gap","nan")),
                }
            except Exception:
                continue
            # per-class F1/precision/recall
            for c in range(n_cls):
                try:
                    row[f"f1_cls{c}"]  = float(m.get(f"F1_Class{c}","nan"))
                    row[f"pre_cls{c}"] = float(m.get(f"Precision_Class{c}","nan"))
                    row[f"rec_cls{c}"] = float(m.get(f"Recall_Class{c}","nan"))
                except Exception:
                    pass
            row["n_classes"] = n_cls
            rows.append(row)
    return rows


def cell_mean(rows, group_keys, metric):
    out = defaultdict(list)
    for r in rows:
        k = tuple(r[g] for g in group_keys)
        v = r.get(metric)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            out[k].append(v)
    return {k: (np.mean(v), np.std(v), len(v)) for k, v in out.items()}


def main():
    rows = collect()
    print(f"Collected {len(rows)} per-seed rows.")

    # Compute per-cell constrained F1, unconstrained mean F1, collateral damage
    derived = []
    for r in rows:
        cls = r["cls_constrained"]
        f1_c = r.get(f"f1_cls{cls}")
        pre_c = r.get(f"pre_cls{cls}")
        rec_c = r.get(f"rec_cls{cls}")
        uncons_f1 = [r[f"f1_cls{i}"] for i in range(r["n_classes"])
                     if i != cls and f"f1_cls{i}" in r and not np.isnan(r[f"f1_cls{i}"])]
        d = {**r,
             "f1_constrained":   f1_c,
             "pre_constrained":  pre_c,
             "rec_constrained":  rec_c,
             "f1_unconstrained_mean": np.mean(uncons_f1) if uncons_f1 else np.nan,
             "f1_unconstrained_min":  np.min(uncons_f1) if uncons_f1 else np.nan,
        }
        derived.append(d)

    # Write summary CSV per (dataset, sigma, tight, method)
    fields = ["dataset","sigma","tight","method","n",
              "macro_f1","f1_constrained","f1_unconstrained_mean",
              "pre_constrained","rec_constrained",
              "acc","ece","brier",
              "conf_correct","conf_incorrect","conf_gap","flips","sat"]
    by_key = defaultdict(list)
    for r in derived:
        by_key[(r["dataset"], r["sigma"], r["tight"], r["method"])].append(r)
    with open(OUT_TBL, "w", newline="") as f:
        w = csv.writer(f); w.writerow(fields)
        for k, rs in sorted(by_key.items()):
            ds, sig, tight, me = k
            line = [ds, f"{sig:.2f}", tight, me, len(rs)]
            for fname in fields[5:]:
                vals = [r.get(fname) for r in rs if r.get(fname) is not None and not (isinstance(r.get(fname), float) and np.isnan(r.get(fname)))]
                line.append(f"{np.mean(vals):.4f}" if vals else "")
            w.writerow(line)
    print(f"Wrote {OUT_TBL}")

    # =================================================================
    # PLOT 1: constrained-class F1 — does TraLO improve it?
    # =================================================================
    plot_constrained_f1(derived)

    # PLOT 2: collateral damage on unconstrained classes
    plot_collateral_damage(derived)

    # PLOT 3: constrained class precision vs recall (where does TraLO live?)
    plot_pre_rec_constrained(derived)

    # PLOT 4: confidence gap (correct minus incorrect confidence)
    plot_conf_gap(derived)

    # PLOT 5: per-class F1 breakdown bar charts on key cells
    plot_per_class_bars(derived)


def plot_constrained_f1(rows):
    """Constrained-class F1 vs sigma, per dataset+tightness."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("f1_constrained") is None or np.isnan(r["f1_constrained"]):
            continue
        by[(r["dataset"], r["tight"])][(r["sigma"], r["method"])].append(r["f1_constrained"])
    for ds in DATASETS:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for ax, tight in zip(axes, TIGHTS):
            d = by[(ds, tight)]
            sigmas = sorted({s for s, _ in d})
            all_vals = []
            for me in METHOD_ORDER:
                xs, ys, errs = [], [], []
                for s in sigmas:
                    v = d.get((s, me), [])
                    if v:
                        xs.append(s); ys.append(np.mean(v))
                        errs.append(np.std(v)/np.sqrt(len(v)))
                        all_vals.append(np.mean(v))
                if xs:
                    ax.errorbar(xs, ys, yerr=errs, marker="o", label=me,
                                color=METHOD_COLORS[me], capsize=4, lw=2, ms=8)
            if all_vals:
                lo, hi = min(all_vals), max(all_vals)
                m = max((hi-lo)*0.15, 0.005)
                ax.set_ylim(lo-m, hi+m)
            ax.set_title(f"tight={tight}"); ax.set_xlabel("contamination sigma")
            ax.set_ylabel("F1 on CONSTRAINED class"); ax.grid(alpha=0.3)
            ax.set_xticks(sigmas)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper right", ncol=5, bbox_to_anchor=(0.99, 1.03), fontsize=10)
        fig.suptitle(f"{ds.upper()} — F1 on the CONSTRAINED CLASS (the one we're capping)",
                     fontsize=13, y=1.05)
        fig.tight_layout()
        p = OUT_FIG / f"constrained_f1_{ds}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"  {p}")


def plot_collateral_damage(rows):
    """Mean F1 of UNCONSTRAINED classes — TraLO's collateral damage hypothesis."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r.get("f1_unconstrained_mean")
        if v is None or np.isnan(v): continue
        by[(r["dataset"], r["tight"])][(r["sigma"], r["method"])].append(v)
    for ds in DATASETS:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for ax, tight in zip(axes, TIGHTS):
            d = by[(ds, tight)]
            sigmas = sorted({s for s, _ in d})
            all_vals = []
            for me in METHOD_ORDER:
                xs, ys, errs = [], [], []
                for s in sigmas:
                    v = d.get((s, me), [])
                    if v:
                        xs.append(s); ys.append(np.mean(v))
                        errs.append(np.std(v)/np.sqrt(len(v)))
                        all_vals.append(np.mean(v))
                if xs:
                    ax.errorbar(xs, ys, yerr=errs, marker="o", label=me,
                                color=METHOD_COLORS[me], capsize=4, lw=2, ms=8)
            if all_vals:
                lo, hi = min(all_vals), max(all_vals)
                m = max((hi-lo)*0.15, 0.005)
                ax.set_ylim(lo-m, hi+m)
            ax.set_title(f"tight={tight}"); ax.set_xlabel("contamination sigma")
            ax.set_ylabel("F1 on UNCONSTRAINED classes (mean)")
            ax.grid(alpha=0.3); ax.set_xticks(sigmas)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper right", ncol=5, bbox_to_anchor=(0.99, 1.03), fontsize=10)
        fig.suptitle(f"{ds.upper()} — F1 on UNCONSTRAINED classes (collateral damage check)",
                     fontsize=13, y=1.05)
        fig.tight_layout()
        p = OUT_FIG / f"collateral_{ds}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"  {p}")


def plot_pre_rec_constrained(rows):
    """Scatter: precision vs recall on constrained class. Where does TraLO live?"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, ds in zip(axes, DATASETS):
        for me in METHOD_ORDER:
            ps, rs = [], []
            for r in rows:
                if r["dataset"] != ds or r["method"] != me: continue
                p, q = r.get("pre_constrained"), r.get("rec_constrained")
                if p and q and not (np.isnan(p) or np.isnan(q)):
                    ps.append(p); rs.append(q)
            if ps:
                ax.scatter(ps, rs, label=me, color=METHOD_COLORS[me],
                           alpha=0.5, s=40)
        ax.set_xlabel("Precision on constrained class")
        ax.set_ylabel("Recall on constrained class")
        ax.set_title(f"{ds}"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        # f1 contours
        if True:
            xs = np.linspace(0.01, 1, 50); ys = np.linspace(0.01, 1, 50)
            X, Y = np.meshgrid(xs, ys)
            F1 = 2*X*Y/(X+Y+1e-9)
            cs = ax.contour(X, Y, F1, levels=[0.2,0.4,0.5,0.6,0.7,0.8],
                            colors="grey", alpha=0.4, linewidths=0.7)
            ax.clabel(cs, fontsize=7)
    fig.suptitle("Precision vs Recall on the constrained class — does TraLO trade off differently?",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    p = OUT_FIG / "constrained_pre_rec_scatter.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  {p}")


def plot_conf_gap(rows):
    """Confidence gap: confidence on correct - confidence on incorrect.
    Larger gap = better calibrated discrimination."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r.get("conf_gap")
        if v is None or np.isnan(v): continue
        by[r["dataset"]][r["method"]].append(v)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        mvals = [by[ds].get(me, []) for me in METHOD_ORDER]
        ax.boxplot(mvals, tick_labels=METHOD_ORDER, showfliers=False)
        ax.set_title(f"{ds}"); ax.set_ylabel("Confidence gap (correct - incorrect)")
        ax.grid(alpha=0.3, axis="y"); ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Discrimination quality: confidence gap between correct and incorrect predictions",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    p = OUT_FIG / "confidence_gap.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  {p}")


def plot_per_class_bars(rows):
    """For each dataset, at clean sigma=0, L30 tight: bar chart of F1 per class per method."""
    for ds in DATASETS:
        subset = [r for r in rows if r["dataset"]==ds and r["sigma"]==0.0 and r["tight"]=="L30_G30"]
        if not subset: continue
        n_cls = subset[0]["n_classes"]
        cls_cstr = subset[0]["cls_constrained"]
        by_m = defaultdict(lambda: defaultdict(list))
        for r in subset:
            for c in range(n_cls):
                v = r.get(f"f1_cls{c}")
                if v is not None and not np.isnan(v):
                    by_m[r["method"]][c].append(v)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        width = 0.13
        x = np.arange(n_cls)
        for i, me in enumerate(METHOD_ORDER):
            ys = [np.mean(by_m[me].get(c, [np.nan])) for c in range(n_cls)]
            ax.bar(x + i*width - width*2, ys, width, label=me,
                   color=METHOD_COLORS[me])
        ax.set_xticks(x); ax.set_xticklabels([f"cls{c}" + (" (CSTR)" if c==cls_cstr else "")
                                              for c in range(n_cls)], rotation=20)
        ax.set_ylabel("F1"); ax.legend(fontsize=8, loc="best", ncol=2)
        ax.grid(alpha=0.3, axis="y")
        ax.set_title(f"{ds} — per-class F1, clean sigma=0, L30_G30, MobileNetV3")
        fig.tight_layout()
        p = OUT_FIG / f"per_class_bar_{ds}.png"
        fig.savefig(p, dpi=130); plt.close(fig)
        print(f"  {p}")


if __name__ == "__main__":
    main()
