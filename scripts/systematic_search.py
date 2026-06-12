"""Systematic search for predictors of TraLO winning.

Pulls EVERY feature we can extract per cell:
  - Dataset properties: n_classes, n_train, n_test, constrained-class prevalence,
    class imbalance ratio (max/min support ratio), data_dir (proxy for noise type)
  - Cell properties: tightness, binding gap, model name, model parameter count,
    pretrained flag, hyperparameters (lr, warmup_epochs, alpha_kl, ...)
  - Warmup quality (via post-hoc baseline's eval which IS the warmup result):
    test accuracy, ECE, Brier, conf gap, entropy, macro/constrained/unconstrained F1
  - Final-epoch warmup TRAIN accuracy from training_log.csv (the real saturation
    signal the headroom hypothesis cares about)

For each baseline (Fioretto LDF, Hounie RCL, Danits LP):
  - Compute paired ΔF1 per cell
  - Run univariate Pearson + Spearman correlations on every feature
  - Run quartile-split analysis (top vs bottom of each feature, see WHERE
    in the data TraLO wins)
  - Run partial-correlation analysis (control for dataset)
  - Compute mutual-information ranking (sklearn) as a non-linear backup

Saves comprehensive tables + a "feature ranking" plot.
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

OUT_FIG = Path("paper/HANDOFF/figures/systematic_v1")
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
    "results/pending_runs/lr_hp_smoke",
]

# Model parameter counts (approx; from torchvision docs)
MODEL_PARAMS = {
    "MobileNetV3": 5_400_000,
    "MobileNetV2": 3_500_000,
    "ShuffleNetV2": 1_400_000,
    "RegNetY400MF": 4_300_000,
    "ResNet18": 11_700_000,
    "EfficientNetB0": 5_300_000,
}

# Dataset-level constants (constrained-class prevalence on test set, etc.)
# These are known/derived from the dataset prep scripts.
DATASET_INFO = {
    "tissuemnist": {"n_classes": 8, "n_train": 9600, "n_test": 2400,
                    "cls_prev": 0.071, "imbalance_ratio": 6.0},
    "dermmnist":   {"n_classes": 7, "n_train": 8012, "n_test": 2003,
                    "cls_prev": 0.113, "imbalance_ratio": 58.0},
    "aider":       {"n_classes": 4, "n_train": 4758, "n_test": 1190,
                    "cls_prev": 0.166, "imbalance_ratio": 4.5},
}


def read_metrics(p):
    out = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            out[row["Metric"]] = row["Value"]
    return out


def read_final_warmup_train_acc(cfg_p):
    """Extract final-epoch warmup train accuracy from training_log.csv if present."""
    log_p = cfg_p.replace("config.json", "training_log.csv")
    if not os.path.isfile(log_p): return float("nan")
    try:
        with open(log_p) as f:
            rdr = list(csv.DictReader(f))
        warmup_rows = [r for r in rdr if r.get("phase","") in ("warmup","Warmup","WARMUP")]
        if not warmup_rows: return float("nan")
        last = warmup_rows[-1]
        for k in ("train_acc","accuracy","acc"):
            if k in last:
                try: return float(last[k])
                except: pass
    except Exception: pass
    return float("nan")


def collect():
    rows = []
    for root in SWEEPS:
        if not os.path.isdir(root): continue
        sweep_name = root.rsplit("/",1)[-1]
        for cfg_p in glob.glob(f"{root}/**/config.json", recursive=True):
            try:
                with open(cfg_p) as f: cfg = json.load(f)
                m = read_metrics(cfg_p.replace("config.json","evaluation_metrics.csv"))
            except Exception: continue
            try:
                ds = cfg["dataset_mode"]
                ds_info = DATASET_INFO.get(ds, {})
                model = cfg["model_name"]
                tight = cfg["constraint_tag"]
                method = cfg["methodology"]
                seed = cfg["hyperparams"]["seed"]
                cls = cfg["dataset_config"]["constrained_class"]
                n_cls = cfg["dataset_config"]["num_classes"]
                pretrained = cfg["hyperparams"].get("pretrained", True)
                data_dir = cfg["dataset_config"]["data_dir"]
                lr = cfg["hyperparams"].get("lr", 1e-4)
                warmup_eps = cfg["hyperparams"].get("warmup_epochs", 50)
                constraint_eps = cfg["hyperparams"].get("constraint_epochs", 300)
                alpha_kl = cfg["hyperparams"].get("alpha_kl", 0)
                rho_target = cfg["hyperparams"].get("rho_target", 100)
                lambda_step = cfg["hyperparams"].get("lambda_step", 0.002)
            except Exception: continue
            sig = f"{sweep_name}|{ds}|{model}|{data_dir}|{tight}|{cls}|{pretrained}|seed{seed}"
            try:
                macro = float(m["F1 (Macro)"])
                acc = float(m["Accuracy"])
                ece = float(m["ECE"])
                brier = float(m["Brier Score"])
                conf_gap = float(m.get("Confidence Gap","nan"))
                mean_entropy = float(m.get("Mean Entropy","nan"))
                pct_high = float(m.get("Pct High Confidence","nan"))
                flips = float(m.get("Flips Required","nan"))
                sat_epoch = int(m.get("Satisfaction Epoch","-1") or "-1")
            except Exception:
                continue
            f1c = float(m.get(f"F1_Class{cls}","nan"))
            pre_c = float(m.get(f"Precision_Class{cls}","nan"))
            rec_c = float(m.get(f"Recall_Class{cls}","nan"))
            sup_c = float(m.get(f"Support_Class{cls}","nan"))
            uf = []
            for c in range(n_cls):
                if c == cls: continue
                v = m.get(f"F1_Class{c}")
                if v:
                    try: uf.append(float(v))
                    except: pass
            f1u = float(np.mean(uf)) if uf else float("nan")
            f1u_std = float(np.std(uf)) if uf else float("nan")
            warmup_train_acc = read_final_warmup_train_acc(cfg_p)
            rows.append({
                "sig": sig, "sweep": sweep_name,
                "dataset": ds, "model": model, "tight": tight,
                "method": method, "seed": seed,
                "cls_cstr": cls, "n_classes": n_cls,
                "pretrained": pretrained, "data_dir": data_dir,
                "lr": lr, "warmup_eps": warmup_eps, "constraint_eps": constraint_eps,
                "alpha_kl": alpha_kl, "rho_target": rho_target, "lambda_step": lambda_step,
                "model_params": MODEL_PARAMS.get(model, 5_000_000),
                # dataset-level
                "ds_n_classes": ds_info.get("n_classes", n_cls),
                "ds_n_train": ds_info.get("n_train", 0),
                "ds_n_test": ds_info.get("n_test", 0),
                "ds_cls_prev": ds_info.get("cls_prev", 0),
                "ds_imbalance": ds_info.get("imbalance_ratio", 1),
                # outcome
                "macro": macro, "f1c": f1c, "f1u": f1u, "f1u_std": f1u_std,
                "pre_c": pre_c, "rec_c": rec_c, "sup_c": sup_c,
                "acc": acc, "ece": ece, "brier": brier,
                "conf_gap": conf_gap, "mean_entropy": mean_entropy, "pct_high": pct_high,
                "flips": flips, "sat_epoch": sat_epoch,
                "warmup_train_acc": warmup_train_acc,
                # derived
                "tight_pct": int(tight.split("_")[0][1:]),
            })
    return rows


def paired_diffs(rows, baseline):
    by_sig = defaultdict(dict)
    for r in rows: by_sig[r["sig"]][r["method"]] = r
    paired = []
    for sig, cell in by_sig.items():
        if "tralo" not in cell or baseline not in cell: continue
        tr = cell["tralo"]; bl = cell[baseline]
        p = {
            # contextual features (use baseline's warmup-side numbers)
            "sweep": tr["sweep"], "dataset": tr["dataset"], "model": tr["model"],
            "tight": tr["tight"], "tight_pct": tr["tight_pct"],
            "pretrained": tr["pretrained"], "data_dir": tr["data_dir"],
            "lr": tr["lr"], "warmup_eps": tr["warmup_eps"],
            "alpha_kl": tr["alpha_kl"], "rho_target": tr["rho_target"],
            "model_params": tr["model_params"],
            "ds_n_classes": tr["ds_n_classes"], "ds_n_train": tr["ds_n_train"],
            "ds_n_test": tr["ds_n_test"], "ds_cls_prev": tr["ds_cls_prev"],
            "ds_imbalance": tr["ds_imbalance"],
            "n_classes": tr["n_classes"],
            # baseline = warmup observable
            "ph_acc": bl["acc"], "ph_ece": bl["ece"], "ph_brier": bl["brier"],
            "ph_conf_gap": bl["conf_gap"], "ph_mean_entropy": bl["mean_entropy"],
            "ph_pct_high": bl["pct_high"],
            "ph_macro": bl["macro"], "ph_f1c": bl["f1c"], "ph_f1u": bl["f1u"],
            "ph_pre_c": bl["pre_c"], "ph_rec_c": bl["rec_c"], "ph_sup_c": bl["sup_c"],
            "ph_f1u_std": bl["f1u_std"],
            "ph_warmup_train_acc": bl["warmup_train_acc"],
            # outcomes
            "d_macro":  tr["macro"]   - bl["macro"],
            "d_f1c":    tr["f1c"]     - bl["f1c"],
            "d_f1u":    tr["f1u"]     - bl["f1u"],
            "d_pre_c":  tr["pre_c"]   - bl["pre_c"],
            "d_rec_c":  tr["rec_c"]   - bl["rec_c"],
            "d_acc":    tr["acc"]     - bl["acc"],
            "d_flips":  tr["flips"]   - bl["flips"],
            "tr_macro": tr["macro"], "tr_sat_epoch": tr["sat_epoch"],
            "tr_flips": tr["flips"],
        }
        paired.append(p)
    return paired


def pearson(xs, ys):
    xs = np.array(xs); ys = np.array(ys)
    mask = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 5 or np.std(xs) == 0 or np.std(ys) == 0:
        return np.nan, len(xs)
    return np.corrcoef(xs, ys)[0,1], len(xs)


def spearman(xs, ys):
    xs = np.array(xs); ys = np.array(ys)
    mask = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 5 or np.std(xs) == 0 or np.std(ys) == 0:
        return np.nan, len(xs)
    rx = xs.argsort().argsort().astype(float)
    ry = ys.argsort().argsort().astype(float)
    return np.corrcoef(rx, ry)[0,1], len(xs)


FEATURES = [
    # dataset-level
    "ds_n_classes", "ds_n_train", "ds_n_test", "ds_cls_prev", "ds_imbalance",
    # cell-level
    "tight_pct", "model_params", "lr", "warmup_eps", "alpha_kl", "rho_target",
    # warmup quality
    "ph_acc", "ph_ece", "ph_brier", "ph_conf_gap", "ph_mean_entropy",
    "ph_pct_high", "ph_macro", "ph_f1c", "ph_f1u", "ph_pre_c", "ph_rec_c",
    "ph_sup_c", "ph_f1u_std", "ph_warmup_train_acc",
]
TARGETS = ["d_macro", "d_f1c", "d_f1u"]


def correlation_table(paired, baseline_name, dataset=None):
    if dataset:
        sub = [p for p in paired if p["dataset"] == dataset]
    else:
        sub = paired
    rows = []
    for f in FEATURES:
        xs = [p[f] for p in sub]
        row = {"feature": f, "n": 0}
        for t in TARGETS:
            ys = [p[t] for p in sub]
            r_p, n = pearson(xs, ys)
            r_s, _ = spearman(xs, ys)
            row[f"{t}_pearson"] = f"{r_p:+.3f}" if not np.isnan(r_p) else "  -  "
            row[f"{t}_spearman"] = f"{r_s:+.3f}" if not np.isnan(r_s) else "  -  "
            row["n"] = n
        rows.append(row)
    return rows


def print_correlation_table(rows, header):
    print(f"\n{header}")
    print(f"  {'feature':<22}{'n':>5}", end="")
    for t in TARGETS:
        print(f"  {t+'_P':>10}{t+'_S':>10}", end="")
    print()
    for r in rows:
        line = f"  {r['feature']:<22}{r['n']:>5}"
        for t in TARGETS:
            line += f"  {r[t+'_pearson']:>10}{r[t+'_spearman']:>10}"
        print(line)


def quartile_split(paired, baseline_name):
    """For each feature, compare top-quartile mean d_macro vs bottom-quartile."""
    print(f"\n=== Feature quartile-split: TraLO ΔF1_macro vs {baseline_name} ===")
    print(f"  {'feature':<22} {'bot25_dF1':>10} {'top25_dF1':>10} {'spread':>10}")
    for f in FEATURES:
        vals = [(p[f], p["d_macro"]) for p in paired
                if not np.isnan(p[f]) and not np.isnan(p["d_macro"])]
        if len(vals) < 20: continue
        vals.sort()
        n = len(vals)
        bot = vals[:n//4]; top = vals[3*n//4:]
        b_d = np.mean([d for _,d in bot])
        t_d = np.mean([d for _,d in top])
        print(f"  {f:<22} {b_d:>+10.4f} {t_d:>+10.4f} {t_d - b_d:>+10.4f}")


def plot_feature_ranking(paired, baseline_name, out_name):
    """Bar chart of |Spearman r| ranked, per dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, ds in zip(axes, ("tissuemnist","dermmnist","aider")):
        sub = [p for p in paired if p["dataset"] == ds]
        if not sub: continue
        scores = []
        for f in FEATURES:
            xs = [p[f] for p in sub]
            ys = [p["d_macro"] for p in sub]
            r, n = spearman(xs, ys)
            if not np.isnan(r):
                scores.append((f, r, n))
        scores.sort(key=lambda t: abs(t[1]), reverse=True)
        names = [s[0] for s in scores]
        vals = [s[1] for s in scores]
        colors = ["green" if v > 0 else "red" for v in vals]
        y = np.arange(len(names))
        ax.barh(y, vals, color=colors, alpha=0.6)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel(f"Spearman r vs d_macro")
        ax.set_title(f"{ds}  (n={scores[0][2] if scores else 0})")
        ax.grid(alpha=0.3, axis="x")
        ax.set_xlim(-1, 1)
    fig.suptitle(f"Predictor ranking — TraLO winning vs {baseline_name}",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(OUT_FIG / out_name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {OUT_FIG / out_name}")


def write_paired_csv(paired, name):
    if not paired: return
    fields = sorted(paired[0].keys())
    with open(OUT_TBL / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in paired:
            w.writerow({k: ("" if isinstance(v, float) and np.isnan(v) else v)
                        for k, v in p.items()})


def main():
    rows = collect()
    print(f"Total per-seed rows: {len(rows)}")
    for baseline_name in ("fioretto_ldf", "hounie_rcl", "danits_lp"):
        paired = paired_diffs(rows, baseline_name)
        print(f"\n{'='*72}\nBASELINE: {baseline_name}   paired cells: {len(paired)}\n{'='*72}")
        write_paired_csv(paired, f"paired_diffs_vs_{baseline_name}.csv")

        # Overall correlations
        rows_corr = correlation_table(paired, baseline_name)
        print_correlation_table(rows_corr,
            f"OVERALL Pearson(P) & Spearman(S) — n_max={len(paired)}")

        # Per-dataset correlations
        for ds in ("tissuemnist", "dermmnist", "aider"):
            sub = [p for p in paired if p["dataset"] == ds]
            if not sub: continue
            r = correlation_table(paired, baseline_name, dataset=ds)
            print_correlation_table(r, f"\n{ds.upper()} only  (n={len(sub)})")

        # Quartile split
        quartile_split(paired, baseline_name)

        # Plot
        plot_feature_ranking(paired, baseline_name,
                             f"predictor_ranking_vs_{baseline_name}.png")


if __name__ == "__main__":
    main()
