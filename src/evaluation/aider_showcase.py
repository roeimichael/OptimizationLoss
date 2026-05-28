"""Aider showcase — frozen-dataset analysis for thesis-advisor review.

Aider runs are frozen (no further experiments) because the warmup model
saturates at 99.98 % train accuracy, so the constraint cap fully dictates
F1 on the constrained class and the trained methods (TraLO/Fior/Hounie)
incur unavoidable collateral damage on unconstrained classes that the
post-hoc heuristic does not.

Outputs into docs/aider_results/:
    aider_per_seed.csv      one row per (tight, seed) × 6 methods × 8 metrics
    aider_head_to_head.csv  one row per tightness, TraLO vs best baseline
    aider_per_class.csv     mean per-class F1/P/R per (tight, method)
    README.md               narrative + summary table

Run from project root:
    python -m src.evaluation.aider_showcase
"""
import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "pending_runs"
OUT_DIR = ROOT / "docs" / "aider_results"

CLS_NAMES = ["collapsed_building", "fire", "flooded_areas", "normal"]
MODEL = "MobileNetV3"
GROUP = "synth_group"
CONSTRAINED_CLS = 0
TIGHTNESS = ["L20_G20", "L30_G30", "L50_G50", "L70_G70", "L80_G80"]
METHODS = ["tralo", "tralo_bounded", "fioretto_ldf", "hounie_rcl",
           "danits_lp", "heuristic"]
METHOD_LABEL = {
    "tralo": "TraLO (ours)", "tralo_bounded": "TraLO-bounded",
    "fioretto_ldf": "Fioretto LDF", "hounie_rcl": "Hounie RCL",
    "danits_lp": "Danits LP", "heuristic": "Heuristic",
}
METRICS = [
    ("F1 (Macro)",            "F1m",   True),
    ("Accuracy",              "Acc",   True),
    ("ECE",                   "ECE",   False),
    ("Brier Score",           "Brier", False),
    ("Flips Required",        "Flips", False),
    ("Raw All Satisfied",     "Sat%",  True),
    ("Satisfaction Epoch",    "SatEp", False),
    ("Constraint Train Time", "Time",  False),
]


def _load_metrics(path):
    out = {}
    try:
        with open(path) as f:
            for r in csv.reader(f):
                if len(r) == 2:
                    out[r[0]] = r[1]
    except Exception:
        pass
    return out


def _to_float(v):
    try:
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def _is_canonical_tralo(hp):
    return (
        hp.get("hybrid_mode") == "undershoot_hinge"
        and hp.get("reset_optimizer_at_sat") is True
        and hp.get("alpha_kl", 0.0) == 0.0
        and abs(hp.get("fior_beta", 0.0) - 0.5) < 1e-6
        and hp.get("penalty_mode") == "both"
        and hp.get("enable_ce_skip") is True
    )


def collect():
    """Per-cell rows: includes scalar metrics + per-class F1/P/R."""
    rows = []
    seen = set()
    for f in glob.glob(str(RUNS / "*/**/config.json"), recursive=True):
        ev = f.replace("config.json", "evaluation_metrics.csv")
        if not os.path.exists(ev):
            continue
        try:
            cfg = json.load(open(f))
        except Exception:
            continue
        if cfg.get("dataset_mode") != "aider":
            continue
        if cfg.get("model_name") != MODEL:
            continue
        ds_cfg = cfg.get("dataset_config", {})
        if ds_cfg.get("constrained_class") != CONSTRAINED_CLS:
            continue
        if ds_cfg.get("group_column") != GROUP:
            continue
        tight = cfg.get("constraint_tag")
        if tight not in TIGHTNESS:
            continue
        method = cfg.get("methodology")
        if method not in METHODS:
            continue
        hp = cfg.get("hyperparams", {})
        if method == "tralo" and not _is_canonical_tralo(hp):
            continue
        seed = hp.get("seed")
        dedupe_key = (tight, method, seed)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        m = _load_metrics(ev)
        row = {"tight": tight, "method": method, "seed": seed}
        for csv_k, short, _ in METRICS:
            row[short] = _to_float(m.get(csv_k))
        for c in range(4):
            row[f"F1_c{c}"] = _to_float(m.get(f"F1_Class{c}"))
            row[f"P_c{c}"] = _to_float(m.get(f"Precision_Class{c}"))
            row[f"R_c{c}"] = _to_float(m.get(f"Recall_Class{c}"))
        rows.append(row)
    return rows


def write_per_seed_csv(rows, path):
    cols = ["tight", "seed"]
    for m in METHODS:
        for _, s, _ in METRICS:
            cols.append(f"{m}__{s}")
    by_idx = defaultdict(dict)
    for r in rows:
        by_idx[(r["tight"], r["seed"])][r["method"]] = r
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for (tight, seed) in sorted(by_idx.keys()):
            row = [tight, seed]
            for m in METHODS:
                rr = by_idx[(tight, seed)].get(m, {})
                for _, s, _ in METRICS:
                    v = rr.get(s)
                    row.append("" if v is None else f"{v:.6g}")
            w.writerow(row)


def aggregate(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["tight"], r["method"])].append(r)
    agg = {}
    for k, items in groups.items():
        out = {"n": len(items)}
        for _, s, _ in METRICS:
            v = [r[s] for r in items if r[s] is not None]
            out[s] = (mean(v) if v else None,
                      stdev(v) if len(v) > 1 else 0.0,
                      len(v))
        for c in range(4):
            for k2 in (f"F1_c{c}", f"P_c{c}", f"R_c{c}"):
                v = [r[k2] for r in items if r[k2] is not None]
                out[k2] = (mean(v) if v else None,
                           stdev(v) if len(v) > 1 else 0.0,
                           len(v))
        agg[k] = out
    return agg


def write_head_to_head(agg, path):
    cols = ["tight", "tralo_n",
            "tralo_F1m", "best_baseline_F1m", "best_baseline_method", "tralo_gap_F1m",
            "tralo_Flips", "best_baseline_Flips", "best_baseline_method_Flips", "tralo_gap_Flips",
            "tralo_ECE", "best_baseline_ECE", "best_baseline_method_ECE", "tralo_gap_ECE"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for tight in TIGHTNESS:
            tr = agg.get((tight, "tralo"))
            if not tr:
                continue
            row = {"tight": tight, "tralo_n": tr["n"]}
            # F1m: higher is better
            tr_f1, _, _ = tr["F1m"]
            base_f1 = [(agg[(tight, m)]["F1m"][0], m) for m in METHODS
                       if m != "tralo" and (tight, m) in agg
                       and agg[(tight, m)]["F1m"][0] is not None]
            best_f1, best_f1_m = max(base_f1)
            row["tralo_F1m"] = f"{tr_f1:.4f}"
            row["best_baseline_F1m"] = f"{best_f1:.4f}"
            row["best_baseline_method"] = best_f1_m
            row["tralo_gap_F1m"] = f"{tr_f1 - best_f1:+.4f}"
            # Flips: lower is better
            tr_fl, _, _ = tr["Flips"]
            base_fl = [(agg[(tight, m)]["Flips"][0], m) for m in METHODS
                       if m != "tralo" and (tight, m) in agg
                       and agg[(tight, m)]["Flips"][0] is not None]
            best_fl, best_fl_m = min(base_fl)
            row["tralo_Flips"] = f"{tr_fl:.2f}"
            row["best_baseline_Flips"] = f"{best_fl:.2f}"
            row["best_baseline_method_Flips"] = best_fl_m
            row["tralo_gap_Flips"] = f"{best_fl - tr_fl:+.2f}"
            # ECE: lower is better
            tr_e, _, _ = tr["ECE"]
            base_e = [(agg[(tight, m)]["ECE"][0], m) for m in METHODS
                      if m != "tralo" and (tight, m) in agg
                      and agg[(tight, m)]["ECE"][0] is not None]
            best_e, best_e_m = min(base_e)
            row["tralo_ECE"] = f"{tr_e:.4f}"
            row["best_baseline_ECE"] = f"{best_e:.4f}"
            row["best_baseline_method_ECE"] = best_e_m
            row["tralo_gap_ECE"] = f"{best_e - tr_e:+.4f}"
            w.writerow(row)


def write_per_class_csv(agg, path):
    cols = ["tight", "method"]
    for c in range(4):
        cls = CLS_NAMES[c]
        cols += [f"F1_c{c}_{cls}", f"P_c{c}_{cls}", f"R_c{c}_{cls}"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for tight in TIGHTNESS:
            for m in METHODS:
                a = agg.get((tight, m))
                if not a:
                    continue
                row = [tight, m]
                for c in range(4):
                    for k in (f"F1_c{c}", f"P_c{c}", f"R_c{c}"):
                        v = a[k][0]
                        row.append("" if v is None else f"{v:.4f}")
                w.writerow(row)


def fmt(stat, spec):
    m, _, _ = stat
    return "—" if m is None else ("{:" + spec + "}").format(m)


def write_readme(agg, path):
    lines = []
    lines.append("# Aider — frozen dataset experiments\n")
    lines.append("**Status:** FROZEN as of 2026-05-24. No further aider experiments will be run.\n")
    lines.append(
        "**Why frozen:** the warmup model on aider saturates at 99.98 % "
        "training accuracy after only 2–3 epochs (out of 50). The trained "
        "constraint-satisfying methods (TraLO, TraLO-bounded, Fioretto LDF, "
        "Hounie RCL) cannot improve on this base model without disturbing "
        "the already-correct predictions on the unconstrained classes. The "
        "post-hoc heuristic, which leaves the model untouched and only flips "
        "the top-K most confident over-predictions on the constrained class, "
        "is therefore close to optimal on F1 macro.\n"
    )
    lines.append("---\n\n## Dataset setup\n")
    lines.append("| Property | Value |")
    lines.append("|---|---|")
    lines.append("| Source | Aerial disaster imagery (4 classes) |")
    lines.append("| Classes | collapsed_building, fire, flooded_areas, normal |")
    lines.append("| Class balance | 8.6 % / 8.7 % / 8.8 % / 73.9 % (test) |")
    lines.append("| Constrained class | 0 = collapsed_building (8.6 %) — rescue-triage framing |")
    lines.append("| Group column | `synth_group` (binary, near-balanced 7.5 % / 9.7 %) |")
    lines.append("| Backbone | MobileNetV3 |")
    lines.append("| Warmup epochs | 50 |")
    lines.append("| Constraint epochs | 300 |")
    lines.append("| Seeds | 1, 2, 3, 4 |")
    lines.append("| Tightness | symmetric L20/L30/L50/L70/L80 |\n")

    lines.append("## Structural finding: F1 on the constrained class is identical across methods\n")
    lines.append(
        "With warmup saturated, the trained methods cannot find a better set of "
        "collapsed_building predictions than the post-hoc heuristic. Every "
        "constraint-satisfying method ends up predicting the same top-K most "
        "confident collapsed_building instances:\n"
    )
    lines.append("| Tight | TraLO F1_c0 | Fior F1_c0 | Hounie F1_c0 | Danits F1_c0 | Heuristic F1_c0 |")
    lines.append("|---|---|---|---|---|---|")
    for tight in TIGHTNESS:
        cells = [tight]
        for m in ["tralo", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]:
            a = agg.get((tight, m))
            cells.append(fmt(a["F1_c0"], ".3f") if a else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "The differences across methods in macro F1 come entirely from "
        "collateral damage on the **unconstrained** classes (fire, flooded, "
        "normal). Trained methods perturb features used by adjacent classes; "
        "heuristic doesn't.\n"
    )

    lines.append("## Headline summary: TraLO vs best baseline\n")
    lines.append("| Tight | TraLO F1m | Best-base F1m | F1m gap | TraLO Flips | Best-base Flips | Flips gap | TraLO ECE | Best-base ECE | ECE gap |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for tight in TIGHTNESS:
        tr = agg.get((tight, "tralo"))
        if not tr:
            continue
        base_f1 = [(agg[(tight, m)]["F1m"][0], m) for m in METHODS
                   if m != "tralo" and (tight, m) in agg
                   and agg[(tight, m)]["F1m"][0] is not None]
        best_f1, _ = max(base_f1)
        base_fl = [(agg[(tight, m)]["Flips"][0], m) for m in METHODS
                   if m != "tralo" and (tight, m) in agg
                   and agg[(tight, m)]["Flips"][0] is not None]
        best_fl, _ = min(base_fl)
        base_e = [(agg[(tight, m)]["ECE"][0], m) for m in METHODS
                  if m != "tralo" and (tight, m) in agg
                  and agg[(tight, m)]["ECE"][0] is not None]
        best_e, _ = min(base_e)
        tr_f1 = tr["F1m"][0]
        tr_fl = tr["Flips"][0]
        tr_e = tr["ECE"][0]
        lines.append(
            f"| {tight} | {tr_f1:.4f} | {best_f1:.4f} | {tr_f1 - best_f1:+.4f} "
            f"| {tr_fl:.1f} | {best_fl:.1f} | {best_fl - tr_fl:+.1f} "
            f"| {tr_e:.4f} | {best_e:.4f} | {best_e - tr_e:+.4f} |"
        )
    lines.append("")
    lines.append(
        "**Interpretation:** TraLO ties or loses 0.003–0.010 on F1m, "
        "but wins the Flips comparison by 2–8 (and by 22–84 against the "
        "post-hoc baselines danits_lp / heuristic, since those need to flip "
        "every over-predicted collapsed instance after training). On ECE, "
        "post-hoc methods win because they leave the well-calibrated "
        "warmup model untouched.\n"
    )

    lines.append("## Possible paper framings (TBD with thesis advisor)\n")
    lines.append(
        "1. **Easy-task regime ablation.** Keep aider as evidence that on "
        "easy tasks the heuristic is hard to beat on F1m, but TraLO is "
        "still the only method that produces a deployable constraint-aware "
        "model end-to-end. The Flips gap is the deployability claim.\n"
    )
    lines.append(
        "2. **Switch constrained class.** Try fire (cls=1) or flooded "
        "(cls=2) to see whether they trigger the same saturation. Most "
        "likely yes — the warmup is the bottleneck, not the class.\n"
    )
    lines.append(
        "3. **Reconstruct aider as a harder benchmark** (reduce warmup "
        "epochs, add noise, real groups). Changes the experimental "
        "contract, so probably not.\n"
    )

    lines.append("## Files in this folder\n")
    lines.append("- `aider_per_seed.csv` — 6 methods × 8 metrics, one row per (tightness, seed)")
    lines.append("- `aider_head_to_head.csv` — TraLO vs best-baseline gap per tightness")
    lines.append("- `aider_per_class.csv` — per-class F1, precision, recall for every (tight, method)")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect()
    agg = aggregate(rows)
    write_per_seed_csv(rows, OUT_DIR / "aider_per_seed.csv")
    write_head_to_head(agg, OUT_DIR / "aider_head_to_head.csv")
    write_per_class_csv(agg, OUT_DIR / "aider_per_class.csv")
    write_readme(agg, OUT_DIR / "README.md")
    print(f"rows={len(rows)} groups={len(agg)}")
    for name in ("README.md", "aider_per_seed.csv",
                 "aider_head_to_head.csv", "aider_per_class.csv"):
        print(f"wrote {OUT_DIR/name}")


if __name__ == "__main__":
    main()
