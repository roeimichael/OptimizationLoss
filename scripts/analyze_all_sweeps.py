"""Master analysis script — aggregates every pending sweep into one CSV
and emits a set of staging plots for the paper-writing session.

Run on the server (data lives there). Plots saved to paper/HANDOFF/figures/v3/.
"""
import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("paper/HANDOFF/figures/v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR = Path("paper/HANDOFF/tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SWEEPS = {
    "multiclass_tissue":  "results/pending_runs/g3_multiclass_tissue",
    "asym_tissue_aider":  "results/pending_runs/g2_asym_tissue_aider",
    "component_ablation": "results/pending_runs/g5_component_ablation",
    "tableB_backfill":    "results/pending_runs/g4_table_b_backfill",
    "aider_cripple":      "results/pending_runs/aider_cripple",
    "derm_cripple":       "results/pending_runs/derm_cripple",
    "derm_backbone_weak": "results/pending_runs/derm_backbone_weak",
    "g1_mobilenetv2":     "results/pending_runs/g1_mobilenetv2",
    "paper_backbones":    "results/pending_runs/paper_backbones",
}

METHOD_COLORS = {
    "tralo":         "#1f77b4",
    "tralo_bounded": "#aec7e8",
    "fioretto_ldf":  "#ff7f0e",
    "hounie_rcl":    "#d62728",
    "danits_lp":     "#2ca02c",
    "heuristic":     "#8c564b",
}
METHOD_ORDER = ["tralo", "tralo_bounded", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]


def read_metrics(p):
    out = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            out[row["Metric"]] = row["Value"]
    return out


def collect():
    rows = []
    for sweep_name, root in SWEEPS.items():
        if not os.path.isdir(root):
            continue
        for cfg_p in glob.glob(f"{root}/**/config.json", recursive=True):
            ev_p = cfg_p.replace("config.json", "evaluation_metrics.csv")
            try:
                with open(cfg_p) as f: cfg = json.load(f)
                m = read_metrics(ev_p)
            except Exception:
                continue
            try:
                d = {
                    "sweep": sweep_name,
                    "experiment_path": cfg["experiment_path"],
                    "dataset": cfg.get("dataset_mode", "?"),
                    "model": cfg.get("model_name", "?"),
                    "cls": cfg.get("dataset_config", {}).get("constrained_class", "?"),
                    "tight": cfg.get("constraint_tag", "?"),
                    "method": cfg.get("methodology", "?"),
                    "seed": cfg.get("hyperparams", {}).get("seed", "?"),
                    "pretrained": cfg.get("hyperparams", {}).get("pretrained", True),
                    "data_dir": cfg.get("dataset_config", {}).get("data_dir", "?"),
                    "f1m":   float(m.get("F1 (Macro)", "nan")),
                    "f1w":   float(m.get("F1 (Weighted)", "nan")),
                    "acc":   float(m.get("Accuracy", "nan")),
                    "ece":   float(m.get("ECE", "nan")),
                    "brier": float(m.get("Brier Score", "nan")),
                    "flips": float(m.get("Flips Required", "nan")),
                    "sat":   1 if m.get("Raw All Satisfied", "0") == "1" else 0,
                    "sat_epoch": int(m.get("Satisfaction Epoch", "-1") or "-1"),
                    "warmup_time": float(m.get("Warmup Time", "nan")),
                    "phase2_time": float(m.get("Constraint Train Time", "nan")),
                }
                rows.append(d)
            except Exception as e:
                pass
    return rows


def write_master(rows):
    path = TABLES_DIR / "master_all_sweeps.csv"
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")


def cell_means(rows, group_keys):
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        for m in ("f1m", "f1w", "acc", "ece", "brier", "flips", "sat"):
            v = r[m]
            if isinstance(v, float) and not np.isnan(v):
                agg[key][m].append(v)
    out = {}
    for key, d in agg.items():
        out[key] = {m: (np.mean(v), np.std(v), len(v)) for m, v in d.items() if v}
    return out


def plot_f1_vs_tightness(rows, sweep, dataset_filter=None, savename=None, title=None):
    """Line plot: x=tight, y=F1, one line per method, faceted by dataset."""
    subset = [r for r in rows if r["sweep"] == sweep]
    if dataset_filter:
        subset = [r for r in subset if r["dataset"] in dataset_filter]
    if not subset:
        return
    datasets = sorted({r["dataset"] for r in subset})
    cls_groups = sorted({(r["dataset"], r["cls"]) for r in subset})
    fig, axes = plt.subplots(1, len(cls_groups), figsize=(4*len(cls_groups), 4),
                              sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (ds, cls) in zip(axes, cls_groups):
        cell = cell_means([r for r in subset if r["dataset"]==ds and r["cls"]==cls],
                          ["tight","method"])
        tights = sorted({k[0] for k in cell.keys()},
                        key=lambda t: int(t.split("_")[0][1:]))
        for me in METHOD_ORDER:
            xs, ys, errs = [], [], []
            for t in tights:
                if (t, me) in cell and "f1m" in cell[(t, me)]:
                    mean, std, _n = cell[(t, me)]["f1m"]
                    xs.append(t); ys.append(mean); errs.append(std)
            if xs:
                ax.errorbar(xs, ys, yerr=errs, marker="o", label=me,
                            color=METHOD_COLORS.get(me, "grey"), capsize=3)
        ax.set_title(f"{ds} cls={cls}")
        ax.set_xlabel("tightness"); ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Macro F1")
    fig.suptitle(title or sweep, y=1.02)
    fig.legend(METHOD_ORDER, loc="upper right", bbox_to_anchor=(1.10, 0.95),
               fontsize=8)
    fig.tight_layout()
    p = OUT_DIR / (savename or f"f1_vs_tight_{sweep}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {p}")


def plot_flips_vs_tightness(rows, sweep, savename=None, title=None):
    subset = [r for r in rows if r["sweep"] == sweep]
    if not subset: return
    cls_groups = sorted({(r["dataset"], r["cls"]) for r in subset})
    fig, axes = plt.subplots(1, len(cls_groups), figsize=(4*len(cls_groups), 4),
                              sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (ds, cls) in zip(axes, cls_groups):
        cell = cell_means([r for r in subset if r["dataset"]==ds and r["cls"]==cls],
                          ["tight","method"])
        tights = sorted({k[0] for k in cell.keys()},
                        key=lambda t: int(t.split("_")[0][1:]))
        for me in METHOD_ORDER:
            xs, ys = [], []
            for t in tights:
                if (t, me) in cell and "flips" in cell[(t, me)]:
                    mean, _, _ = cell[(t, me)]["flips"]
                    xs.append(t); ys.append(max(mean, 0.1))
            if xs:
                ax.plot(xs, ys, marker="o", label=me,
                        color=METHOD_COLORS.get(me, "grey"))
        ax.set_title(f"{ds} cls={cls}")
        ax.set_xlabel("tightness")
        ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Flips Required (log)")
    fig.suptitle(title or f"Flips vs tightness — {sweep}", y=1.02)
    fig.legend(METHOD_ORDER, loc="upper right", bbox_to_anchor=(1.10, 0.95),
               fontsize=8)
    fig.tight_layout()
    p = OUT_DIR / (savename or f"flips_vs_tight_{sweep}.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {p}")


def plot_cripple_heatmap(rows, sweep, savename, title):
    """Heatmap: rows=corruption/variant, cols=tight, value=in-training - post-hoc dF1."""
    subset = [r for r in rows if r["sweep"] == sweep]
    if not subset: return
    cell = cell_means(subset, ["experiment_path"])
    # group by condition × tight
    by_ct = defaultdict(lambda: defaultdict(list))
    for r in subset:
        parts = r["experiment_path"].split("/")
        # Sweep root has 3 leading parts; cond is at index 3 in our layout
        cond = parts[3] if len(parts) > 3 else "?"
        by_ct[(cond, r["tight"])][r["method"]].append(r["f1m"])
    conds = sorted({c for (c, _) in by_ct})
    tights = sorted({t for (_, t) in by_ct}, key=lambda t: int(t.split("_")[0][1:]))
    grid = np.full((len(conds), len(tights)), np.nan)
    annot = [[""]*len(tights) for _ in range(len(conds))]
    for i, c in enumerate(conds):
        for j, t in enumerate(tights):
            d = by_ct.get((c, t), {})
            if not d: continue
            in_train = [v for me, vals in d.items() for v in vals
                        if me in ("tralo", "fioretto_ldf", "tralo_bounded", "hounie_rcl")]
            post_hoc = [v for me, vals in d.items() for v in vals
                        if me in ("danits_lp", "heuristic")]
            if in_train and post_hoc:
                diff = np.mean(in_train) - np.mean(post_hoc)
                grid[i, j] = diff
                annot[i][j] = f"{diff:+.3f}"
    fig, ax = plt.subplots(figsize=(1.4*len(tights)+1.5, 0.6*len(conds)+1.5))
    vmax = np.nanmax(np.abs(grid))
    im = ax.imshow(grid, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(tights))); ax.set_xticklabels(tights, rotation=45)
    ax.set_yticks(range(len(conds))); ax.set_yticklabels(conds)
    for i in range(len(conds)):
        for j in range(len(tights)):
            if annot[i][j]:
                ax.text(j, i, annot[i][j], ha="center", va="center",
                        color="black", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="dF1 (in-training - post-hoc)")
    fig.tight_layout()
    p = OUT_DIR / savename
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {p}")


def plot_sat_pct(rows, savename="sat_pct_all_sweeps.png"):
    """Bar chart: mean Sat% per method, faceted by sweep."""
    sweeps_to_show = ["multiclass_tissue", "asym_tissue_aider", "derm_cripple",
                      "derm_backbone_weak", "aider_cripple"]
    rows = [r for r in rows if r["sweep"] in sweeps_to_show]
    by_sm = defaultdict(list)
    for r in rows:
        by_sm[(r["sweep"], r["method"])].append(r["sat"])
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.13
    x = np.arange(len(sweeps_to_show))
    for i, me in enumerate(METHOD_ORDER):
        vals = [np.mean(by_sm.get((s, me), [np.nan])) * 100 for s in sweeps_to_show]
        ax.bar(x + i*width - width*2.5, vals, width, label=me,
               color=METHOD_COLORS[me])
    ax.set_xticks(x); ax.set_xticklabels(sweeps_to_show, rotation=20)
    ax.set_ylabel("Sat% (mean across all cells)")
    ax.set_title("Constraint Satisfaction by method (deployability)")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = OUT_DIR / savename
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  {p}")


def plot_calibration(rows, savename="calibration_all_sweeps.png"):
    """ECE + Brier per method across all in-distribution sweeps."""
    sweeps_to_show = ["multiclass_tissue", "asym_tissue_aider"]
    rows = [r for r in rows if r["sweep"] in sweeps_to_show]
    by_m = defaultdict(lambda: {"ece": [], "brier": []})
    for r in rows:
        if not np.isnan(r["ece"]): by_m[r["method"]]["ece"].append(r["ece"])
        if not np.isnan(r["brier"]): by_m[r["method"]]["brier"].append(r["brier"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    methods = [m for m in METHOD_ORDER if m in by_m]
    a1.boxplot([by_m[m]["ece"] for m in methods], labels=methods)
    a1.set_title("ECE distribution (lower = better)"); a1.grid(alpha=0.3)
    a1.tick_params(axis="x", rotation=30)
    a2.boxplot([by_m[m]["brier"] for m in methods], labels=methods)
    a2.set_title("Brier score distribution (lower = better)"); a2.grid(alpha=0.3)
    a2.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    p = OUT_DIR / savename
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  {p}")


def plot_component_ablation(rows, savename="component_ablation_delta.png"):
    """Bar plot: dF1 per disabled variant vs full TraLO, faceted by dataset."""
    subset = [r for r in rows if r["sweep"] == "component_ablation"]
    if not subset: return
    by_ds_var = defaultdict(list)
    for r in subset:
        var = r["experiment_path"].split("/")[-2]
        by_ds_var[(r["dataset"], var)].append(r["f1m"])
    datasets = sorted({ds for ds, _ in by_ds_var})
    variants = ["full", "no_hinge", "no_reset", "no_freeze",
                "no_ce_skip", "no_rho_sched", "no_warmup"]
    full_mean = {ds: np.mean(by_ds_var.get((ds, "full"), [np.nan])) for ds in datasets}
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.12
    x = np.arange(len(variants[1:]))
    for i, ds in enumerate(datasets):
        vals = [np.mean(by_ds_var.get((ds, v), [np.nan])) - full_mean[ds] for v in variants[1:]]
        ax.bar(x + i*width - width*len(datasets)/2, vals, width, label=ds)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(variants[1:], rotation=30)
    ax.set_ylabel("dF1 vs full TraLO  (negative = component IS essential)")
    ax.set_title("Component ablation — what breaks when we disable each knob")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = OUT_DIR / savename
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  {p}")


def plot_warmup_acc_vs_advantage(rows, savename="headroom_scatter.png"):
    """Scatter: warmup train-acc vs (in-training - post-hoc) F1 advantage.

    Approximate warmup train-acc with test accuracy of the post-hoc methods
    (their final model = warmup model untouched).
    """
    cripple_sweeps = ["aider_cripple", "derm_cripple", "derm_backbone_weak"]
    by_group = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["sweep"] not in cripple_sweeps: continue
        parts = r["experiment_path"].split("/")
        cond = parts[3] if len(parts) > 3 else "?"
        key = (r["sweep"], cond, r["tight"])
        by_group[key][r["method"]].append((r["f1m"], r["acc"]))
    pts = []
    for key, mm in by_group.items():
        if "danits_lp" not in mm: continue
        ph_acc = np.mean([a for _, a in mm["danits_lp"]])
        in_f1 = []
        for me in ("tralo", "fioretto_ldf"):
            if me in mm:
                in_f1.extend([f for f, _ in mm[me]])
        ph_f1 = []
        for me in ("danits_lp", "heuristic"):
            if me in mm:
                ph_f1.extend([f for f, _ in mm[me]])
        if in_f1 and ph_f1:
            pts.append((ph_acc, np.mean(in_f1) - np.mean(ph_f1), key[0], key[1]))
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"aider_cripple":"#1f77b4","derm_cripple":"#2ca02c","derm_backbone_weak":"#d62728"}
    for sw in colors:
        xs = [a for a, d, s, c in pts if s == sw]
        ys = [d for a, d, s, c in pts if s == sw]
        ax.scatter(xs, ys, label=sw, color=colors[sw], alpha=0.7, s=80)
    for a, d, s, c in pts:
        ax.annotate(c[:8], (a, d), fontsize=7, alpha=0.5,
                    xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Post-hoc test accuracy (proxy for warmup quality)")
    ax.set_ylabel("dF1 (in-training - post-hoc)")
    ax.set_title("Headroom hypothesis: lower warmup quality → larger in-training advantage")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = OUT_DIR / savename
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  {p}")


def main():
    print("Collecting from all sweeps...")
    rows = collect()
    print(f"Total rows: {len(rows)}")
    write_master(rows)

    print("\nGenerating plots:")
    plot_f1_vs_tightness(rows, "multiclass_tissue",
        savename="f1_multiclass_tissue.png",
        title="Multi-class TissueMNIST — F1 vs tightness, per alt-class")
    plot_flips_vs_tightness(rows, "multiclass_tissue",
        savename="flips_multiclass_tissue.png")
    plot_f1_vs_tightness(rows, "asym_tissue_aider",
        savename="f1_asym_tissue_aider.png",
        title="Asymmetric L!=G — F1 vs tightness, tissue+aider")

    plot_cripple_heatmap(rows, "derm_cripple",
        savename="cripple_derm_heatmap.png",
        title="DermMNIST cripple — dF1 (in-training - post-hoc) per (corruption, tightness)")
    plot_cripple_heatmap(rows, "derm_backbone_weak",
        savename="cripple_derm_backbone_weak_heatmap.png",
        title="DermMNIST backbone-weak — dF1 per (variant, tightness)")
    plot_cripple_heatmap(rows, "aider_cripple",
        savename="cripple_aider_heatmap.png",
        title="AIDER cripple — dF1 per condition")

    plot_component_ablation(rows)
    plot_warmup_acc_vs_advantage(rows)
    plot_sat_pct(rows)
    plot_calibration(rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
