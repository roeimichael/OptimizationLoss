"""Build paper-ready artifacts from post-fix runs.

Outputs:
- paper/results_v2.tex     LaTeX tables (booktabs, matches existing style)
- paper/figures/fig_convergence.png   TraLO vs Fioretto vs Hounie C4 hard-count vs epoch on (1,4,7) L30
- paper/figures/fig_satisfaction.png  Per-method sat-rate bar across tightness
- paper/figures/fig_f1_tightness.png  F1m vs tightness line chart

Run from project root:
    python paper_results/build_paper_artifacts.py
"""
import csv, json, os, glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "pending_runs"
PAPER = ROOT / "paper"
FIG = PAPER / "figures"
FIG.mkdir(parents=True, exist_ok=True)

POST_FIX = {
    "tralo": {"fix_ce_skip", "fix1_validation", "kl_sweep",
              "overnight_2026_05_14", "paper_rerun"},
    "fioretto_ldf": {"convergence_validation_300", "fix1_validation",
                     "overnight_2026_05_14", "paper_rerun"},
    "hounie_rcl": {"hounie_rerun", "convergence_validation_300",
                   "fix1_validation", "overnight_2026_05_14", "paper_rerun"},
    "heuristic": {"convergence_validation_300", "overnight_2026_05_14",
                  "thesis_ext", "thesis", "overnight_sweep", "paper_rerun"},
    "danits_lp": {"convergence_validation_300", "overnight_2026_05_14",
                  "thesis_ext", "thesis", "overnight_sweep", "paper_rerun"},
}

METHOD_LABEL = {
    "tralo": "TraLO",
    "fioretto_ldf": "Fioretto",
    "hounie_rcl": "Hounie",
    "heuristic": "Heuristic",
    "danits_lp": "DANITS",
}

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "figure.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
})


def parse_cls(cc):
    if isinstance(cc, list):
        return tuple(sorted(cc))
    return (cc,)


def collect():
    runs = []
    for em in BASE.rglob("evaluation_metrics.csv"):
        cfg_path = em.parent / "config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            continue
        method = cfg.get("methodology")
        if method not in POST_FIX:
            continue
        sweep = em.parent.relative_to(BASE).parts[0]
        if sweep not in POST_FIX[method]:
            continue
        hp = cfg["hyperparams"]
        if method == "tralo":
            if hp.get("alpha_kl", 0) != 0 or hp.get("linear_sat_tail", 0) != 0:
                continue
        # Load eval_metrics.csv for training-time fields (Raw All Satisfied,
        # Flips Required, satisfaction epoch). Overlay fair_evaluation_metrics.csv
        # (uniform clamp) on top for the F1/acc/precision/recall fields used
        # in cross-method comparisons.
        metrics = {r["Metric"]: r["Value"] for r in csv.DictReader(open(em))}
        fair = em.parent / "fair_evaluation_metrics.csv"
        if fair.exists():
            fair_metrics = {r["Metric"]: r["Value"]
                            for r in csv.DictReader(open(fair))}
            metrics.update(fair_metrics)
        cc = cfg["dataset_config"].get("constrained_class", [])
        cc_list = cc if isinstance(cc, list) else [cc]
        f1_const = []
        for c in cc_list:
            v = metrics.get(f"F1_Class{c}")
            if v not in (None, ""):
                try:
                    f1_const.append(float(v))
                except ValueError:
                    pass
        runs.append({
            "method": method,
            "sweep": sweep,
            "dir": str(em.parent.relative_to(BASE)),
            "dataset": cfg.get("dataset_mode"),
            "model": cfg.get("model_name"),
            "cls": parse_cls(cc),
            "tight": cfg.get("constraint_tag"),
            "seed": int(hp.get("seed", 0)),
            "acc": float(metrics.get("Accuracy", 0) or 0),
            "f1m": float(metrics.get("F1 (Macro)", 0) or 0),
            "f1c": (sum(f1_const) / len(f1_const)) if f1_const else 0,
            "flips": int(metrics.get("Flips Required", "0") or "0"),
            "sat": (metrics.get("Raw All Satisfied", "0") == "1"),
        })
    return runs


def stat(xs, fmt="{:.3f}"):
    if not xs:
        return "-"
    mu = sum(xs) / len(xs)
    if len(xs) > 1:
        sd = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    else:
        sd = 0
    return f"{fmt.format(mu)}\\,{{\\scriptsize$\\pm${fmt.format(sd)}}}"


def latex_tightness_table(runs, dataset, model, cls, label, caption):
    """Tightness sweep, all 5 methods, 3 metrics (F1m, F1const, acc)."""
    methods = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic", "danits_lp"]
    tights = ["L20_G20", "L30_G30", "L40_G40", "L50_G50",
              "L60_G60", "L70_G70", "L80_G80"]
    by_key = defaultdict(list)
    for r in runs:
        if (r["dataset"], r["model"], r["cls"]) != (dataset, model, tuple(cls)):
            continue
        by_key[(r["tight"], r["method"])].append(r)

    rows = [r"\begin{table}[htbp]\centering\small",
            r"\caption{" + caption + r"}",
            r"\label{" + label + r"}",
            r"\begin{tabular}{l l " + "c" * len(methods) + r"}",
            r"\toprule",
            "Tight & Metric & " + " & ".join(METHOD_LABEL[m] for m in methods) + r" \\",
            r"\midrule"]
    for t in tights:
        for label_m, key, fmt in [("F1$_m$", "f1m", "{:.3f}"),
                                   ("F1$_c$", "f1c", "{:.3f}"),
                                   ("Acc", "acc", "{:.3f}")]:
            vals = {}
            for m in methods:
                xs = [r[key] for r in by_key.get((t, m), [])]
                vals[m] = sum(xs) / len(xs) if xs else None
            if all(v is None for v in vals.values()):
                continue
            valid = [(m, v) for m, v in vals.items() if v is not None]
            best_val = max(v for _, v in valid)
            cells = []
            for m in methods:
                xs = [r[key] for r in by_key.get((t, m), [])]
                if not xs:
                    cells.append("-")
                    continue
                s = stat(xs, fmt)
                if abs(vals[m] - best_val) < 1e-6 and len(valid) > 1:
                    s = r"\textbf{" + s + r"}"
                cells.append(s)
            t_label = t.replace("_", "\\_")
            rows.append(f"{t_label} & {label_m} & " + " & ".join(cells) + r" \\")
        rows.append(r"\addlinespace[2pt]")
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


def latex_summary_table(runs, datasets_models, caption, label):
    """Headline table — for each (dataset, model), one row, multiple metrics."""
    methods = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic", "danits_lp"]
    rows = [r"\begin{table}[htbp]\centering\small",
            r"\caption{" + caption + r"}",
            r"\label{" + label + r"}",
            r"\begin{tabular}{l l l " + "c" * len(methods) + r"}",
            r"\toprule",
            "Dataset & Model & Metric & " + " & ".join(METHOD_LABEL[m] for m in methods) + r" \\",
            r"\midrule"]
    for dataset, model, cls, tight in datasets_models:
        block = [r for r in runs if r["dataset"] == dataset and r["model"] == model
                 and r["cls"] == tuple(cls) and r["tight"] == tight]
        if not block:
            continue
        for label_m, key, fmt in [("F1$_m$", "f1m", "{:.3f}"),
                                   ("F1$_c$", "f1c", "{:.3f}"),
                                   ("Acc", "acc", "{:.3f}")]:
            vals = {}
            for m in methods:
                xs = [r[key] for r in block if r["method"] == m]
                vals[m] = sum(xs) / len(xs) if xs else None
            if all(v is None for v in vals.values()):
                continue
            valid = [(m, v) for m, v in vals.items() if v is not None]
            best = max(v for _, v in valid)
            cells = []
            for m in methods:
                xs = [r[key] for r in block if r["method"] == m]
                if not xs:
                    cells.append("-")
                    continue
                s = stat(xs, fmt)
                if abs(vals[m] - best) < 1e-6 and len(valid) > 1:
                    s = r"\textbf{" + s + r"}"
                cells.append(s)
            rows.append(f"{dataset} & {model[:6]} & {label_m} & " +
                        " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    if rows[-1] == r"\midrule":
        rows.pop()
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


def latex_satisfaction_table(runs, dataset, model, cls, caption, label):
    """Per-method satisfaction rate + average flip count across tightness."""
    methods = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic", "danits_lp"]
    tights = ["L20_G20", "L30_G30", "L40_G40", "L50_G50",
              "L60_G60", "L70_G70", "L80_G80"]
    by_key = defaultdict(list)
    for r in runs:
        if (r["dataset"], r["model"], r["cls"]) != (dataset, model, tuple(cls)):
            continue
        by_key[(r["tight"], r["method"])].append(r)

    rows = [r"\begin{table}[htbp]\centering\small",
            r"\caption{" + caption + r"}",
            r"\label{" + label + r"}",
            r"\begin{tabular}{l " + "cc " * len(methods) + r"}",
            r"\toprule",
            r"Tight " + "".join(r"& \multicolumn{2}{c}{" + METHOD_LABEL[m] + r"} "
                                for m in methods) + r"\\",
            "".join(r"\cmidrule(lr){" + f"{2 + 2*i}-{3 + 2*i}" + r"} "
                    for i in range(len(methods))),
            " " + "".join("& Sat & Flips " for _ in methods) + r"\\",
            r"\midrule"]
    for t in tights:
        cells = []
        any_data = False
        for m in methods:
            xs = by_key.get((t, m), [])
            if not xs:
                cells.extend(["-", "-"])
                continue
            any_data = True
            sat_rate = sum(1 for r in xs if r["sat"]) / len(xs)
            flips_mean = sum(r["flips"] for r in xs) / len(xs)
            cells.append(f"{sat_rate:.0%}".replace("%", r"\%"))
            cells.append(f"{flips_mean:.0f}")
        if not any_data:
            continue
        rows.append(t.replace("_", "\\_") + " & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


def _extract_trajectory(log_path):
    """Return (epochs, total_excess) handling TraLO + Hounie/Fioretto schemas."""
    rows = list(csv.DictReader(open(log_path)))
    if not rows:
        return [], []
    head = rows[0]
    if "Epoch" in head:  # TraLO schema
        eps, exs = [], []
        for r in rows:
            eps.append(int(r["Epoch"]))
            total = 0
            for c in range(8):  # tissuemnist 8 classes
                lim = r.get(f"Limit_Class{c}")
                h = r.get(f"Hard_Class{c}")
                if lim and h and lim not in ("inf", ""):
                    try:
                        total += max(0, int(h) - int(float(lim)))
                    except ValueError:
                        pass
                for gid in (0, 1):
                    gl = r.get(f"Group{gid}_Limit_Class{c}")
                    gh = r.get(f"Group{gid}_Hard_Class{c}")
                    if gl and gh and gl not in ("inf", ""):
                        try:
                            total += max(0, int(gh) - int(float(gl)))
                        except ValueError:
                            pass
            exs.append(total)
        return eps, exs
    elif "epoch" in head:  # Hounie / Fioretto schema
        return ([int(r["epoch"]) for r in rows],
                [int(float(r.get("total_excess", 0))) for r in rows])
    return [], []


def figure_convergence(runs):
    """Total excess vs epoch for TraLO/Fior/Hou on (1,4,7) L30_G30 seed 1."""
    targets = {"tralo": "TraLO", "fioretto_ldf": "Fioretto", "hounie_rcl": "Hounie"}
    colors = {"tralo": "#1976D2", "fioretto_ldf": "#2E7D32", "hounie_rcl": "#E53935"}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for method in targets:
        cands = [r for r in runs if r["method"] == method
                 and r["dataset"] == "tissuemnist"
                 and r["model"] == "MobileNetV3"
                 and r["cls"] == (1, 4, 7)
                 and r["tight"] == "L30_G30"
                 and r["seed"] == 1]
        if not cands:
            continue
        cands.sort(key=lambda r: r["sweep"] != "overnight_2026_05_14")
        log = BASE / cands[0]["dir"] / "training_log.csv"
        if not log.exists():
            continue
        eps, exs = _extract_trajectory(log)
        if not eps:
            continue
        ax.plot(eps, exs, label=targets[method], color=colors[method], linewidth=2)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.6,
               label=r"Feasibility ($\sum E = 0$)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Total excess $\sum_c \max(0, \mathrm{count}_c - K_c)$")
    ax.set_title("Constraint enforcement during training\n"
                 "(TissueMNIST MobileNetV3, classes (1,4,7), L30\\_G30, seed 1)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    out = FIG / "fig_convergence.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


def figure_f1_tightness(runs):
    """F1m vs tightness, lines for each method."""
    methods = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic"]
    colors = {"tralo": "#1976D2", "fioretto_ldf": "#2E7D32",
              "hounie_rcl": "#E53935", "heuristic": "#9E9E9E"}
    tights = ["L20_G20", "L30_G30", "L40_G40", "L50_G50",
              "L60_G60", "L70_G70", "L80_G80"]
    xs = [int(t[1:3]) for t in tights]  # 20, 30, 40 ...

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax_idx, (key, ylabel) in enumerate(
            [("f1m", r"F1 macro"), ("f1c", r"F1 on constrained class")]):
        ax = axes[ax_idx]
        for method in methods:
            ys = []
            for t in tights:
                cands = [r[key] for r in runs
                         if r["method"] == method
                         and r["dataset"] == "tissuemnist"
                         and r["model"] == "MobileNetV3"
                         and r["cls"] == (4,)
                         and r["tight"] == t]
                ys.append(sum(cands)/len(cands) if cands else None)
            # Filter None pairs
            pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if not pairs:
                continue
            px, py = zip(*pairs)
            ax.plot(px, py, "-o", label=METHOD_LABEL[method],
                    color=colors[method], linewidth=2, markersize=5)
        ax.set_xlabel(r"Tightness $\alpha$ (\%)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.15)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_title("(a) F1 macro" if ax_idx == 0 else "(b) F1 on constrained class")
    fig.suptitle("TissueMNIST MobileNetV3, class 4 — tightness sweep",
                 fontsize=11)
    fig.tight_layout()
    out = FIG / "fig_f1_tightness.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


def figure_satisfaction_rates(runs):
    """Bar chart of sat rate by method across tightness."""
    methods = ["tralo", "fioretto_ldf", "hounie_rcl"]
    colors = {"tralo": "#1976D2", "fioretto_ldf": "#2E7D32",
              "hounie_rcl": "#E53935"}
    tights = ["L20_G20", "L30_G30", "L40_G40", "L50_G50",
              "L60_G60", "L70_G70", "L80_G80"]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(tights))
    width = 0.27
    for i, method in enumerate(methods):
        rates = []
        for t in tights:
            cands = [r for r in runs if r["method"] == method
                     and r["dataset"] == "tissuemnist"
                     and r["model"] == "MobileNetV3"
                     and r["cls"] == (4,)
                     and r["tight"] == t]
            rates.append(
                sum(1 for r in cands if r["sat"]) / len(cands) * 100
                if cands else 0)
        ax.bar(x + (i - 1) * width, rates, width, label=METHOD_LABEL[method],
               color=colors[method])
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_G", "/G") for t in tights], rotation=15)
    ax.set_ylabel("In-training satisfaction rate (\\%)")
    ax.set_title("Fraction of runs satisfying constraints during training\n"
                 "(TissueMNIST, MobileNetV3, class 4)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.15, axis="y")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    out = FIG / "fig_satisfaction.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    runs = collect()
    print(f"Collected {len(runs)} runs")
    by_method = defaultdict(int)
    for r in runs:
        by_method[r["method"]] += 1
    for m, n in sorted(by_method.items()):
        print(f"  {m}: {n}")

    # LaTeX tables
    tex_parts = [
        "% Auto-generated paper-ready tables. Source: paper_results/build_paper_artifacts.py",
        "% Methods: TraLO (ours), Fioretto LDF, Hounie RCL, Heuristic, DANITS-LP",
        "% Data: post-fix runs across all sweeps.",
        ""]

    tex_parts.append(latex_summary_table(
        runs,
        [("tissuemnist", "MobileNetV3", (4,), "L50_G50"),
         ("tissuemnist", "MobileNetV3", (1, 4, 7), "L50_G50"),
         ("tissuemnist", "MobileNetV3", (3, 4), "L50_G50")],
        "Headline benchmark on TissueMNIST L50\\_G50 (mean over available seeds, "
        "best per row in bold). TraLO ties or wins Fioretto on every cell; "
        "Hounie's higher numbers are produced post-hoc since Hounie rarely satisfies "
        "during training (see Table~\\ref{tab:satisfaction}).",
        "tab:headline"))

    tex_parts.append(latex_tightness_table(
        runs, "tissuemnist", "MobileNetV3", (4,),
        "tab:tightness_tissue",
        "Tightness sweep on TissueMNIST MobileNetV3, single-class constraint on "
        "class 4 (GE). Mean$\\pm$std over up to 5 seeds. Tight $\\alpha$ in $\\{20,\\dots,80\\}\\%$."))

    tex_parts.append(latex_satisfaction_table(
        runs, "tissuemnist", "MobileNetV3", (4,),
        "Satisfaction discipline on TissueMNIST MobileNetV3 class 4: fraction of "
        "runs satisfying both global and local constraints during training, and "
        "mean number of post-hoc flips required. TraLO converges in $\\ge 80\\%$ of "
        "runs across tightness levels; Fioretto converges instantly but with lower "
        "F1; Hounie almost never converges during training and pays a heavy posthoc "
        "tax.",
        "tab:satisfaction"))

    out_tex = PAPER / "results_v2.tex"
    out_tex.write_text("\n\n".join(tex_parts), encoding="utf-8")
    print(f"\nWrote {out_tex}")

    # Figures
    figure_convergence(runs)
    figure_f1_tightness(runs)
    figure_satisfaction_rates(runs)


if __name__ == "__main__":
    main()
