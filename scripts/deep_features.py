"""Deep universal features from raw prediction probabilities.

For each cell, we read final_predictions.csv (per-sample probabilities)
and compute warmup-classifier-geometry features that are NOT confounded
with dataset identity:

  borderline_rate_010:   fraction of samples with top1-top2 < 0.10
  borderline_rate_005:   fraction with top1-top2 < 0.05
  high_conf_rate_090:    fraction with top1 > 0.90
  entropy_mean:          mean of -sum(p*log p)
  soft_count_cstr:       sum of P(class=c) — soft count of cap class
  hard_count_cstr:       count of argmax == cap class (warmup behavior)
  soft_minus_hard:       soft_count - hard_count (where TraLO has room)
  binding_ratio:         hard_count / K (>1 = cap binds)
  cstr_prob_mean:        mean of P(class=c) across all samples
  cstr_prob_std:         std of P(class=c)
  cstr_prob_q90:         90th percentile of P(class=c)
  uncstr_prob_mean:      mean of P(not=c)
  per_class_pred_balance: entropy of histogram of argmax labels
                         (lower = warmup over-concentrates on few classes)

We extract from danits_lp's predictions (=warmup model, post-hoc trim only).
This gives us per-paired-cell features describing the WARMUP behavior
without any retraining. Then we test if these predict TraLO ΔF1.
"""
import csv
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_FIG = Path("paper/HANDOFF/figures/deep_v1")
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
]


def read_metrics(p):
    out = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            out[row["Metric"]] = row["Value"]
    return out


def compute_geometry_features(pred_csv, n_classes, cls_cstr, tight_pct, sup_cstr):
    """Compute warmup-classifier-geometry features from final_predictions.csv."""
    probs = []
    labels = []
    with open(pred_csv) as f:
        rdr = csv.reader(f)
        header = next(rdr)
        # locate Prob_Class_i columns and True_Label
        prob_cols = [i for i, h in enumerate(header) if h.startswith("Prob_Class_")]
        try:
            true_col = header.index("True_Label")
        except ValueError:
            true_col = None
        if len(prob_cols) != n_classes:
            return None
        for row in rdr:
            try:
                p = [float(row[i]) for i in prob_cols]
                probs.append(p)
                if true_col is not None:
                    labels.append(int(row[true_col]))
            except (ValueError, IndexError):
                continue
    if not probs:
        return None
    P = np.array(probs, dtype=np.float64)
    n_samples = len(P)
    argmax = P.argmax(axis=1)
    # top1, top2
    sorted_P = -np.sort(-P, axis=1)
    top1 = sorted_P[:, 0]
    top2 = sorted_P[:, 1]
    gap = top1 - top2
    # entropy
    eps = 1e-12
    ent = -np.sum(P * np.log(P + eps), axis=1)
    # cap-class probabilities
    p_c = P[:, cls_cstr]
    # warmup hard count of cap class
    hard_count = int(np.sum(argmax == cls_cstr))
    soft_count = float(p_c.sum())
    # K from cap percentage and support
    K = sup_cstr * tight_pct / 100.0
    # per-class argmax frequencies
    pred_hist = np.bincount(argmax, minlength=n_classes).astype(np.float64)
    pred_hist /= pred_hist.sum() + eps
    pred_balance_entropy = float(-np.sum(pred_hist * np.log(pred_hist + eps)))
    return {
        "n_samples": n_samples,
        "borderline_010": float((gap < 0.10).mean()),
        "borderline_005": float((gap < 0.05).mean()),
        "high_conf_090": float((top1 > 0.90).mean()),
        "high_conf_099": float((top1 > 0.99).mean()),
        "entropy_mean": float(ent.mean()),
        "entropy_std": float(ent.std()),
        "soft_count_cstr": soft_count,
        "hard_count_cstr": float(hard_count),
        "soft_minus_hard_cstr": soft_count - hard_count,
        "cstr_K": float(K),
        "binding_ratio": hard_count / max(K, 1e-9),       # >1 = cap binds
        "soft_binding_ratio": soft_count / max(K, 1e-9),
        "cstr_prob_mean": float(p_c.mean()),
        "cstr_prob_std": float(p_c.std()),
        "cstr_prob_q90": float(np.quantile(p_c, 0.90)),
        "uncstr_prob_mean": float((1.0 - p_c).mean()),
        "pred_balance_entropy": pred_balance_entropy,
    }


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
                model = cfg["model_name"]
                tight = cfg["constraint_tag"]
                method = cfg["methodology"]
                seed = cfg["hyperparams"]["seed"]
                cls = cfg["dataset_config"]["constrained_class"]
                n_cls = cfg["dataset_config"]["num_classes"]
                pretrained = cfg["hyperparams"].get("pretrained", True)
                data_dir = cfg["dataset_config"]["data_dir"]
                tight_pct = int(tight.split("_")[0][1:])
            except Exception: continue
            sig = f"{sweep_name}|{ds}|{model}|{data_dir}|{tight}|{cls}|{pretrained}|seed{seed}"
            try:
                macro = float(m["F1 (Macro)"])
                sup_c = float(m.get(f"Support_Class{cls}", "0"))
            except Exception: continue
            f1c = float(m.get(f"F1_Class{cls}", "nan"))
            uf = [float(m[f"F1_Class{c}"]) for c in range(n_cls)
                  if c != cls and f"F1_Class{c}" in m]
            f1u = float(np.mean(uf)) if uf else float("nan")
            row = {
                "sig": sig, "sweep": sweep_name, "dataset": ds, "model": model,
                "tight": tight, "tight_pct": tight_pct,
                "method": method, "seed": seed,
                "cls_cstr": cls, "n_classes": n_cls,
                "macro": macro, "f1c": f1c, "f1u": f1u,
                "sup_c": sup_c,
                "cfg_path": cfg_p,
            }
            rows.append(row)
    return rows


def attach_geometry_for_baseline(rows, baseline_name):
    """For each cell (signature), compute geometry from baseline's predictions
    once and attach to all method rows for that cell."""
    by_sig = defaultdict(dict)
    for r in rows: by_sig[r["sig"]][r["method"]] = r
    geom_cache = {}
    n_total = len(by_sig)
    n_done = 0
    for sig, cell in by_sig.items():
        n_done += 1
        if baseline_name not in cell: continue
        bl = cell[baseline_name]
        pred_csv = bl["cfg_path"].replace("config.json", "final_predictions.csv")
        if not os.path.isfile(pred_csv): continue
        try:
            g = compute_geometry_features(pred_csv, bl["n_classes"], bl["cls_cstr"],
                                          bl["tight_pct"], bl["sup_c"])
            if g: geom_cache[sig] = g
        except Exception: continue
        if n_done % 500 == 0:
            print(f"    geometry: {n_done}/{n_total} cells processed, {len(geom_cache)} attached")
    return geom_cache


def pearson(xs, ys):
    xs, ys = np.array(xs), np.array(ys)
    mask = ~(np.isnan(xs) | np.isnan(ys) | np.isinf(xs) | np.isinf(ys))
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 5 or np.std(xs) == 0 or np.std(ys) == 0: return np.nan, len(xs)
    return np.corrcoef(xs, ys)[0,1], len(xs)


def spearman(xs, ys):
    xs, ys = np.array(xs), np.array(ys)
    mask = ~(np.isnan(xs) | np.isnan(ys) | np.isinf(xs) | np.isinf(ys))
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 5 or np.std(xs) == 0 or np.std(ys) == 0: return np.nan, len(xs)
    rx = xs.argsort().argsort().astype(float)
    ry = ys.argsort().argsort().astype(float)
    return np.corrcoef(rx, ry)[0,1], len(xs)


GEOM_FEATURES = [
    "borderline_010", "borderline_005", "high_conf_090", "high_conf_099",
    "entropy_mean", "entropy_std",
    "soft_count_cstr", "hard_count_cstr", "soft_minus_hard_cstr",
    "binding_ratio", "soft_binding_ratio",
    "cstr_prob_mean", "cstr_prob_std", "cstr_prob_q90",
    "uncstr_prob_mean", "pred_balance_entropy",
]


def paired_with_geometry(rows, baseline_name, geom_cache):
    by_sig = defaultdict(dict)
    for r in rows: by_sig[r["sig"]][r["method"]] = r
    out = []
    for sig, cell in by_sig.items():
        if "tralo" not in cell or baseline_name not in cell: continue
        if sig not in geom_cache: continue
        tr, bl = cell["tralo"], cell[baseline_name]
        d = {
            "dataset": tr["dataset"], "tight": tr["tight"], "tight_pct": tr["tight_pct"],
            "d_macro": tr["macro"] - bl["macro"],
            "d_f1c":   tr["f1c"]   - bl["f1c"]   if not (np.isnan(tr["f1c"]) or np.isnan(bl["f1c"])) else np.nan,
            "d_f1u":   tr["f1u"]   - bl["f1u"]   if not (np.isnan(tr["f1u"]) or np.isnan(bl["f1u"])) else np.nan,
        }
        d.update(geom_cache[sig])
        out.append(d)
    return out


def print_correlations(paired, baseline):
    print(f"\n{'='*72}")
    print(f"GEOMETRY features vs ΔF1_macro (TraLO vs {baseline})")
    print(f"{'='*72}")
    # Overall
    print(f"\n--- OVERALL ({len(paired)} paired cells) ---")
    print(f"  {'feature':<25}{'P(d_macro)':>12}{'S(d_macro)':>12}{'S(d_f1c)':>12}{'S(d_f1u)':>12}")
    for f in GEOM_FEATURES:
        xs = [p[f] for p in paired]
        for tgt in ["d_macro"]:
            pass
        rp_m, n = pearson(xs, [p["d_macro"] for p in paired])
        rs_m, _ = spearman(xs, [p["d_macro"] for p in paired])
        rs_c, _ = spearman(xs, [p["d_f1c"] for p in paired])
        rs_u, _ = spearman(xs, [p["d_f1u"] for p in paired])
        print(f"  {f:<25}{rp_m:>+12.3f}{rs_m:>+12.3f}{rs_c:>+12.3f}{rs_u:>+12.3f}")
    # Per dataset
    for ds in ("tissuemnist", "dermmnist", "aider"):
        sub = [p for p in paired if p["dataset"] == ds]
        if len(sub) < 20: continue
        print(f"\n--- {ds} (n={len(sub)}) ---")
        print(f"  {'feature':<25}{'P(d_macro)':>12}{'S(d_macro)':>12}{'S(d_f1c)':>12}{'S(d_f1u)':>12}")
        for f in GEOM_FEATURES:
            xs = [p[f] for p in sub]
            rp_m, _ = pearson(xs, [p["d_macro"] for p in sub])
            rs_m, _ = spearman(xs, [p["d_macro"] for p in sub])
            rs_c, _ = spearman(xs, [p["d_f1c"] for p in sub])
            rs_u, _ = spearman(xs, [p["d_f1u"] for p in sub])
            print(f"  {f:<25}{rp_m:>+12.3f}{rs_m:>+12.3f}{rs_c:>+12.3f}{rs_u:>+12.3f}")


def plot_universal_predictors(paired, baseline, outname):
    """Scatter the top 4 features that correlate UNIVERSALLY across datasets."""
    # Rank features by min(|r|) across datasets — we want the most universal predictor
    universal_score = {}
    for f in GEOM_FEATURES:
        rs_per_ds = []
        for ds in ("tissuemnist","dermmnist","aider"):
            sub = [p for p in paired if p["dataset"] == ds]
            if len(sub) < 20: continue
            r, _ = spearman([p[f] for p in sub], [p["d_macro"] for p in sub])
            rs_per_ds.append(r)
        if len(rs_per_ds) == 3 and not any(np.isnan(rs_per_ds)):
            # universal = same SIGN, all non-trivial
            same_sign = (np.sign(rs_per_ds[0]) == np.sign(rs_per_ds[1]) ==
                          np.sign(rs_per_ds[2]))
            min_abs = min(abs(r) for r in rs_per_ds)
            universal_score[f] = (same_sign, min_abs, rs_per_ds)
    ranked = sorted(universal_score.items(),
                    key=lambda kv: (kv[1][0], kv[1][1]), reverse=True)
    print(f"\n=== Universal-predictor ranking vs {baseline} ===")
    print(f"  {'feature':<25} same_sign  min|r|     r_tissue  r_derm  r_aider")
    for f, (same_sign, min_abs, rs) in ranked[:8]:
        print(f"  {f:<25} {str(same_sign):>9}  {min_abs:>6.2f}    {rs[0]:>+6.2f}  {rs[1]:>+6.2f}  {rs[2]:>+6.2f}")
    top4 = ranked[:4]
    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    for i, (f, (same_sign, min_abs, rs)) in enumerate(top4):
        for j, ds in enumerate(("tissuemnist","dermmnist","aider")):
            ax = axes[i, j]
            sub = [p for p in paired if p["dataset"] == ds]
            if not sub: continue
            xs = [p[f] for p in sub]
            ys = [p["d_macro"] for p in sub]
            colors = ["green" if y > 0 else "red" for y in ys]
            ax.scatter(xs, ys, c=colors, alpha=0.5, s=25)
            mask = ~(np.isnan(xs) | np.isnan(ys))
            xs_m, ys_m = np.array(xs)[mask], np.array(ys)[mask]
            if len(xs_m) > 5 and np.std(xs_m) > 0:
                slope, b = np.polyfit(xs_m, ys_m, 1)
                xs_fit = np.linspace(xs_m.min(), xs_m.max(), 50)
                ax.plot(xs_fit, slope*xs_fit + b, "k--", lw=1.5)
                r = np.corrcoef(xs_m, ys_m)[0,1]
                ax.text(0.05, 0.95, f"r={r:+.2f} n={len(xs_m)}",
                        transform=ax.transAxes, va="top", fontsize=8,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            ax.axhline(0, color="black", lw=0.5)
            ax.set_xlabel(f); ax.set_ylabel("d_macro" if j==0 else "")
            ax.set_title(f"{ds}", fontsize=10)
            ax.grid(alpha=0.3)
    fig.suptitle(f"Top universal geometry predictors of TraLO winning (vs {baseline})",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(OUT_FIG / outname, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {OUT_FIG / outname}")


def main():
    rows = collect()
    print(f"Total per-seed rows: {len(rows)}")
    for baseline in ("fioretto_ldf", "hounie_rcl", "danits_lp"):
        print(f"\nComputing geometry features for cells with both tralo + {baseline}...")
        geom = attach_geometry_for_baseline(rows, baseline)
        print(f"  geometry features computed for {len(geom)} unique cells")
        paired = paired_with_geometry(rows, baseline, geom)
        print(f"  paired (tralo,{baseline}) with geometry: {len(paired)}")
        if not paired: continue
        # write CSV
        if paired:
            fields = sorted(paired[0].keys())
            with open(OUT_TBL / f"deep_paired_vs_{baseline}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
                for p in paired:
                    w.writerow({k: ("" if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else v)
                                for k, v in p.items()})
        print_correlations(paired, baseline)
        plot_universal_predictors(paired, baseline,
                                   f"deep_top_predictors_vs_{baseline}.png")


if __name__ == "__main__":
    main()
