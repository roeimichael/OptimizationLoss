"""TraLO winning-conditions analyzer.

For every paired (cell, seed) we have:
  - TraLO macro-F1, constrained-class F1, unconstrained-class F1
  - Best post-hoc (Danits LP) ditto
  - Dataset properties (constrained prevalence, num_classes, cap binding)
  - Warmup properties (post-hoc test acc, precision/recall on constrained)
  - Confidence properties (gap, mean entropy)

Question: which features predict TraLO winning ΔF1?

Pulls ALL pending sweeps. Per-seed pairing required for honest paired diffs.
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

OUT_FIG = Path("paper/HANDOFF/figures/winning_conditions")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TBL = Path("paper/HANDOFF/tables")

SWEEPS = [
    "results/pending_runs/contamination_clean",
    "results/pending_runs/contamination_tissuemnist",
    "results/pending_runs/contamination_dermmnist",
    "results/pending_runs/contamination_aider",
    "results/pending_runs/g3_multiclass_tissue",
    "results/pending_runs/g2_asym_tissue_aider",
    "results/pending_runs/g1_mobilenetv2",
    "results/pending_runs/paper_backbones",
    "results/pending_runs/derm_cripple",
    "results/pending_runs/derm_backbone_weak",
    "results/pending_runs/aider_cripple",
    "results/pending_runs/g5_component_ablation",
    "results/pending_runs/lr_hp_smoke",
]


def read_metrics(p):
    out = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            out[row["Metric"]] = row["Value"]
    return out


def collect():
    rows = []
    for root in SWEEPS:
        if not os.path.isdir(root): continue
        sweep_name = root.rsplit("/", 1)[-1]
        for cfg_p in glob.glob(f"{root}/**/config.json", recursive=True):
            try:
                with open(cfg_p) as f: cfg = json.load(f)
                m = read_metrics(cfg_p.replace("config.json","evaluation_metrics.csv"))
            except Exception: continue
            try:
                ds = cfg["dataset_mode"]
                tight = cfg["constraint_tag"]
                method = cfg["methodology"]
                seed = cfg["hyperparams"]["seed"]
                model = cfg["model_name"]
                cls_cstr = cfg["dataset_config"]["constrained_class"]
                n_cls = cfg["dataset_config"]["num_classes"]
                pretrained = cfg["hyperparams"].get("pretrained", True)
                data_dir = cfg["dataset_config"]["data_dir"]
            except Exception:
                continue
            # cell signature for paired comparison
            cell_sig = f"{sweep_name}|{ds}|{model}|{data_dir}|{tight}|{cls_cstr}|{pretrained}|seed{seed}"
            try:
                row = {
                    "sweep": sweep_name, "cell_sig": cell_sig,
                    "dataset": ds, "model": model, "tight": tight, "method": method,
                    "seed": seed, "cls_cstr": cls_cstr, "n_classes": n_cls,
                    "pretrained": pretrained, "data_dir": data_dir,
                    "macro_f1": float(m["F1 (Macro)"]),
                    "acc": float(m["Accuracy"]),
                    "ece": float(m["ECE"]),
                    "brier": float(m["Brier Score"]),
                    "flips": float(m["Flips Required"]),
                    "sat": 1 if m.get("Raw All Satisfied","0")=="1" else 0,
                    "sat_epoch": int(m.get("Satisfaction Epoch", "-1") or "-1"),
                    "conf_correct": float(m.get("Confidence (Correct)", "nan")),
                    "conf_incorrect": float(m.get("Confidence (Incorrect)", "nan")),
                    "conf_gap": float(m.get("Confidence Gap", "nan")),
                    "pct_high_conf": float(m.get("Pct High Confidence", "nan")),
                    "mean_entropy": float(m.get("Mean Entropy", "nan")),
                }
                # constrained-class metrics
                row["f1_cstr"]  = float(m.get(f"F1_Class{cls_cstr}", "nan"))
                row["pre_cstr"] = float(m.get(f"Precision_Class{cls_cstr}", "nan"))
                row["rec_cstr"] = float(m.get(f"Recall_Class{cls_cstr}", "nan"))
                row["sup_cstr"] = float(m.get(f"Support_Class{cls_cstr}", "nan"))
                # unconstrained-class metrics (mean across other classes)
                uf, up, ur = [], [], []
                for c in range(n_cls):
                    if c == cls_cstr: continue
                    f1 = m.get(f"F1_Class{c}")
                    pr = m.get(f"Precision_Class{c}")
                    re = m.get(f"Recall_Class{c}")
                    if f1: uf.append(float(f1))
                    if pr: up.append(float(pr))
                    if re: ur.append(float(re))
                row["f1_uncstr_mean"]  = np.mean(uf) if uf else np.nan
                row["f1_uncstr_min"]   = np.min(uf) if uf else np.nan
                row["pre_uncstr_mean"] = np.mean(up) if up else np.nan
                row["rec_uncstr_mean"] = np.mean(ur) if ur else np.nan
                rows.append(row)
            except Exception:
                continue
    return rows


def paired_diffs(rows, baseline_method="danits_lp"):
    """For each cell, compute (TraLO - baseline) on all metrics."""
    by_cell = defaultdict(dict)
    for r in rows:
        by_cell[r["cell_sig"]][r["method"]] = r
    paired = []
    for sig, cell in by_cell.items():
        if "tralo" not in cell or baseline_method not in cell: continue
        tr = cell["tralo"]; bl = cell[baseline_method]
        p = {
            "sweep": tr["sweep"], "dataset": tr["dataset"], "model": tr["model"],
            "tight": tr["tight"], "seed": tr["seed"],
            "cls_cstr": tr["cls_cstr"], "n_classes": tr["n_classes"],
            "data_dir": tr["data_dir"], "pretrained": tr["pretrained"],
            # baseline (post-hoc) properties
            "ph_acc": bl["acc"],   # warmup quality proxy
            "ph_f1_cstr": bl["f1_cstr"],
            "ph_pre_cstr": bl["pre_cstr"],
            "ph_rec_cstr": bl["rec_cstr"],
            "ph_f1_uncstr_mean": bl["f1_uncstr_mean"],
            "ph_sup_cstr": bl["sup_cstr"],
            "ph_macro_f1": bl["macro_f1"],
            "ph_conf_gap": bl["conf_gap"],
            "ph_mean_entropy": bl["mean_entropy"],
            # TraLO advantage
            "d_macro_f1":  tr["macro_f1"] - bl["macro_f1"],
            "d_f1_cstr":   tr["f1_cstr"] - bl["f1_cstr"],
            "d_f1_uncstr": tr["f1_uncstr_mean"] - bl["f1_uncstr_mean"],
            "d_pre_cstr":  tr["pre_cstr"] - bl["pre_cstr"],
            "d_rec_cstr":  tr["rec_cstr"] - bl["rec_cstr"],
            "d_conf_gap":  tr["conf_gap"] - bl["conf_gap"],
            "d_flips":     tr["flips"] - bl["flips"],
            # binding strength: how far ph_cstr_support is from cap K
            "tight_pct":   int(tr["tight"].split("_")[0][1:]),
        }
        # cap K in samples (approx): tight_pct/100 * sup_cstr
        # binding "tightness" = sup_cstr * tight_pct / 100
        p["binding_gap"] = bl["sup_cstr"] - bl["sup_cstr"] * p["tight_pct"] / 100
        paired.append(p)
    return paired


def report_correlations(paired):
    """Per-dataset Pearson r between TraLO advantage and each predictor."""
    predictors = ["ph_acc", "ph_f1_cstr", "ph_pre_cstr", "ph_rec_cstr",
                  "ph_f1_uncstr_mean", "ph_macro_f1", "ph_conf_gap",
                  "ph_mean_entropy", "tight_pct", "binding_gap"]
    targets = ["d_macro_f1", "d_f1_cstr", "d_f1_uncstr"]
    print(f"\n{'='*70}")
    print("CORRELATIONS: predictors vs TraLO ΔF1 advantage")
    print(f"{'='*70}")
    for ds in ("tissuemnist", "dermmnist", "aider"):
        sub = [p for p in paired if p["dataset"] == ds]
        if len(sub) < 5: continue
        print(f"\n--- {ds} (n={len(sub)}) ---")
        print(f"{'predictor':<22}", end="")
        for t in targets: print(f"  {t:<14}", end="")
        print()
        for pr in predictors:
            xs = np.array([p[pr] for p in sub if not np.isnan(p.get(pr, np.nan))])
            line = f"  {pr:<20}"
            for t in targets:
                ys = np.array([p[t] for p in sub if not np.isnan(p.get(t, np.nan)) and not np.isnan(p.get(pr, np.nan))])
                xs_t = np.array([p[pr] for p in sub if not np.isnan(p.get(pr, np.nan)) and not np.isnan(p.get(t, np.nan))])
                if len(xs_t) > 3 and np.std(xs_t) > 0 and np.std(ys) > 0:
                    r = np.corrcoef(xs_t, ys)[0,1]
                    line += f"  r={r:+.2f}"
                else:
                    line += f"  r=  --"
            print(line)


def report_winning_signatures(paired):
    """For each dataset, find cells where TraLO WON BIG (top quartile of d_macro_f1)
    and characterize them by their predictor values."""
    print(f"\n{'='*70}")
    print("WINNING SIGNATURES: characterize cells where TraLO wins big")
    print(f"{'='*70}")
    for ds in ("tissuemnist", "dermmnist", "aider"):
        sub = [p for p in paired if p["dataset"] == ds]
        if len(sub) < 8: continue
        sorted_ = sorted(sub, key=lambda p: p["d_macro_f1"])
        n = len(sorted_)
        bot = sorted_[:n//4]       # bottom quartile (TraLO loses)
        top = sorted_[3*n//4:]      # top quartile (TraLO wins)
        print(f"\n--- {ds} (n={n}, top vs bottom quartile, each {len(top)}/{len(bot)}) ---")
        feats = ["ph_acc","ph_f1_cstr","ph_pre_cstr","ph_rec_cstr","ph_f1_uncstr_mean",
                 "ph_conf_gap","tight_pct","d_macro_f1","d_f1_cstr","d_f1_uncstr"]
        print(f"  {'feature':<22} {'bot25 mean':>10} {'top25 mean':>10} {'diff':>10}")
        for f in feats:
            bv = [p[f] for p in bot if not np.isnan(p.get(f, np.nan))]
            tv = [p[f] for p in top if not np.isnan(p.get(f, np.nan))]
            if bv and tv:
                bm, tm = np.mean(bv), np.mean(tv)
                print(f"  {f:<22} {bm:>10.3f} {tm:>10.3f} {tm-bm:>+10.3f}")


def plot_top_predictors(paired):
    """For each dataset, scatter of predictor vs d_macro_f1, with regression line."""
    predictors_of_interest = ["ph_f1_uncstr_mean", "ph_f1_cstr",
                              "ph_acc", "ph_conf_gap"]
    fig, axes = plt.subplots(len(predictors_of_interest), 3,
                             figsize=(15, 3.5*len(predictors_of_interest)))
    for i, pr in enumerate(predictors_of_interest):
        for j, ds in enumerate(("tissuemnist","dermmnist","aider")):
            ax = axes[i, j]
            sub = [p for p in paired if p["dataset"]==ds
                   and not np.isnan(p.get(pr, np.nan))
                   and not np.isnan(p.get("d_macro_f1", np.nan))]
            if not sub: continue
            xs = np.array([p[pr] for p in sub])
            ys = np.array([p["d_macro_f1"] for p in sub])
            colors = ["green" if y > 0 else "red" for y in ys]
            ax.scatter(xs, ys, c=colors, alpha=0.5, s=25)
            if len(xs) > 3:
                slope, b = np.polyfit(xs, ys, 1)
                xs_fit = np.linspace(xs.min(), xs.max(), 50)
                ax.plot(xs_fit, slope*xs_fit + b, "k--", lw=1.5)
                r = np.corrcoef(xs, ys)[0,1]
                ax.text(0.05, 0.95, f"r={r:+.2f} slope={slope:+.3f}\nn={len(xs)}",
                        transform=ax.transAxes, va="top", fontsize=8,
                        bbox=dict(boxstyle="round", facecolor="white"))
            ax.axhline(0, color="black", lw=0.5)
            ax.set_title(f"{ds}", fontsize=10)
            if i == len(predictors_of_interest) - 1:
                ax.set_xlabel(pr)
            if j == 0: ax.set_ylabel(f"d_macro_f1 ({pr})")
            ax.grid(alpha=0.3)
    fig.suptitle("Which warmup-side feature predicts TraLO winning?",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    p = OUT_FIG / "predictor_scatter.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  {p}")


def plot_win_loss_split(paired):
    """For each dataset, show win-rate vs binned predictors."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, ds in zip(axes, ("tissuemnist","dermmnist","aider")):
        sub = [p for p in paired if p["dataset"]==ds]
        if not sub: continue
        # bin by ph_f1_uncstr_mean
        xs = np.array([p["ph_f1_uncstr_mean"] for p in sub])
        ys = np.array([1 if p["d_macro_f1"] > 0.001 else (-1 if p["d_macro_f1"] < -0.001 else 0) for p in sub])
        bins = np.linspace(np.nanmin(xs), np.nanmax(xs), 6)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        win, los, tie = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (xs >= lo) & (xs < hi)
            if mask.sum() == 0:
                win.append(0); los.append(0); tie.append(0); continue
            win.append((ys[mask] == 1).sum() / mask.sum())
            los.append((ys[mask] == -1).sum() / mask.sum())
            tie.append((ys[mask] == 0).sum() / mask.sum())
        ax.bar(bin_centers, win, width=(bins[1]-bins[0])*0.8, color="green",
               alpha=0.6, label="TraLO wins")
        ax.bar(bin_centers, [-l for l in los], width=(bins[1]-bins[0])*0.8,
               color="red", alpha=0.6, label="TraLO loses")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("post-hoc unconstrained-class F1 mean")
        ax.set_ylabel("win rate (TraLO win=positive)")
        ax.set_title(f"{ds}"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Win rate vs post-hoc unconstrained-class F1 — does ambiguity predict TraLO win?",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    p = OUT_FIG / "win_rate_vs_uncstr_f1.png"
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  {p}")


def main():
    rows = collect()
    print(f"Total per-seed rows across all sweeps: {len(rows)}")
    paired = paired_diffs(rows, "danits_lp")
    print(f"Paired (cell, seed) TraLO-vs-Danits comparisons: {len(paired)}")
    # write paired CSV
    if paired:
        fields = sorted(paired[0].keys())
        with open(OUT_TBL / "paired_diffs_tralo_vs_danits.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for p in paired:
                w.writerow({k: ("" if isinstance(v, float) and np.isnan(v) else v)
                            for k, v in p.items()})
        print(f"Wrote {OUT_TBL / 'paired_diffs_tralo_vs_danits.csv'}")
    # also vs fioretto and heuristic
    paired_fi = paired_diffs(rows, "fioretto_ldf")
    paired_he = paired_diffs(rows, "heuristic")
    print(f"Paired vs fioretto: {len(paired_fi)}")
    print(f"Paired vs heuristic: {len(paired_he)}")
    report_correlations(paired)
    report_winning_signatures(paired)
    print("\nPlots:")
    plot_top_predictors(paired)
    plot_win_loss_split(paired)


if __name__ == "__main__":
    main()
