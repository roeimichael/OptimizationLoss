"""Table A summary aggregator — paper plan v2.

Scans all completed cells in results/pending_runs/ that map onto Table A:
    3 datasets × MobileNetV3 × story-(cls,grp) × 5 symmetric tightness
    × 6 methods × 4 seeds.

Outputs:
    docs/table_a_raw.csv         one row per (ds, tight, method, seed) cell
    docs/table_a_summary.csv     one row per (ds, tight, method): mean, std, n
    docs/table_a_summary.md      paper-ready markdown with winner highlight

Metrics reported (mean ± std across seeds):
    F1m  (F1 Macro)             primary
    Acc  (Accuracy)
    ECE                         (lower is better)
    Brier
    Flips    (Posthoc flips required, lower is better)
    Sat%     (Raw All Satisfied)
    SatEp    (Satisfaction Epoch)
    Time     (Constraint Train Time, sec)

Winner highlight per (ds, tight, metric): bold on highest F1m/Acc/Sat%
and lowest ECE/Brier/Flips/SatEp/Time.
"""
import csv, glob, json, os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "pending_runs"
OUT_DIR = ROOT / "docs"

STORY = {
    "tissuemnist": {"cls": 4, "group": "synth_group", "label": "TissueMNIST · GE / synth_group"},
    "dermmnist":   {"cls": 4, "group": "loc_group",   "label": "DermMNIST · MEL / loc_group"},
    "aider":       {"cls": 0, "group": "synth_group", "label": "Aider · collapsed_building / synth_group"},
}
MODEL = "MobileNetV3"
TIGHTNESS = ["L20_G20", "L30_G30", "L50_G50", "L70_G70", "L80_G80"]
METHODS = ["tralo", "tralo_bounded", "fioretto_ldf", "hounie_rcl",
           "danits_lp", "heuristic"]
METHOD_LABEL = {
    "tralo": "TraLO (ours)", "tralo_bounded": "TraLO-bounded",
    "fioretto_ldf": "Fioretto LDF", "hounie_rcl": "Hounie RCL",
    "danits_lp": "Danits LP", "heuristic": "Heuristic",
}
# (key in csv, label, higher_is_better)
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


def load_metrics_csv(path):
    out = {}
    try:
        with open(path) as f:
            for r in csv.reader(f):
                if len(r) == 2:
                    out[r[0]] = r[1]
    except Exception:
        pass
    return out


def to_float(v):
    if v is None or v == "" or v == "N/A":
        return None
    try:
        x = float(v)
        if x != x:
            return None
        return x
    except ValueError:
        return None


def _is_canonical_tralo(hp):
    """Headline TraLO recipe — the breakthrough HP set from
    project_tralofix_breakthrough.md. Reject ablation/variant runs."""
    return (
        hp.get("hybrid_mode") == "undershoot_hinge"
        and hp.get("reset_optimizer_at_sat") is True
        and hp.get("alpha_kl", 0.0) == 0.0
        and abs(hp.get("fior_beta", 0.0) - 0.5) < 1e-6
        and hp.get("penalty_mode") == "both"
        and hp.get("enable_ce_skip") is True
    )


def collect():
    rows = []
    for f in glob.glob(str(RUNS / "*/**/config.json"), recursive=True):
        ev_path = f.replace("config.json", "evaluation_metrics.csv")
        if not os.path.exists(ev_path):
            continue
        try:
            cfg = json.load(open(f))
        except Exception:
            continue
        ds = cfg.get("dataset_mode")
        if ds not in STORY:
            continue
        if cfg.get("model_name") != MODEL:
            continue
        if cfg.get("dataset_config", {}).get("constrained_class") != STORY[ds]["cls"]:
            continue
        if cfg.get("dataset_config", {}).get("group_column") != STORY[ds]["group"]:
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
        seed = cfg.get("hyperparams", {}).get("seed")
        m = load_metrics_csv(ev_path)
        row = {"ds": ds, "tight": tight, "method": method, "seed": seed}
        for csv_key, short, _ in METRICS:
            row[short] = to_float(m.get(csv_key))
        rows.append(row)
    return rows


def aggregate(rows):
    """Return {(ds, tight, method): {metric_short: (mean, std, n)}}.

    Dedupes rows so a (ds, tight, method, seed) cell only counts once
    even if it appears under multiple sweep paths."""
    seen = {}
    for r in rows:
        key = (r["ds"], r["tight"], r["method"], r["seed"])
        if key not in seen:
            seen[key] = r
    rows = list(seen.values())
    groups = defaultdict(list)
    for r in rows:
        groups[(r["ds"], r["tight"], r["method"])].append(r)
    agg = {}
    for key, items in groups.items():
        out = {"n": len(items)}
        for _, short, _ in METRICS:
            vals = [r[short] for r in items if r[short] is not None]
            if not vals:
                out[short] = (None, None, 0)
            elif len(vals) == 1:
                out[short] = (vals[0], 0.0, 1)
            else:
                out[short] = (mean(vals), stdev(vals), len(vals))
        agg[key] = out
    return agg


def fmt_cell(stat, fmt_spec):
    m, s, n = stat
    if m is None:
        return "—"
    if n == 1:
        return ("{:" + fmt_spec + "}").format(m)
    return ("{:" + fmt_spec + "}±{:" + fmt_spec + "}").format(m, s)


def fmt_for_metric(short):
    if short in ("F1m", "Acc", "Sat%"):
        return ".3f"
    if short == "ECE" or short == "Brier":
        return ".3f"
    if short == "Flips":
        return ".0f"
    if short == "SatEp":
        return ".0f"
    if short == "Time":
        return ".0f"
    return ".3f"


def write_raw_csv(rows, path):
    cols = ["ds", "tight", "method", "seed"] + [s for _, s, _ in METRICS]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["ds"], r["tight"], r["method"], r["seed"] or 0)):
            w.writerow({c: r.get(c, "") for c in cols})


def write_summary_csv(agg, path):
    cols = ["ds", "tight", "method", "n"]
    for _, s, _ in METRICS:
        cols += [f"{s}_mean", f"{s}_std"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for (ds, tight, method), out in sorted(agg.items()):
            row = {"ds": ds, "tight": tight, "method": method, "n": out["n"]}
            for _, s, _ in METRICS:
                m, sd, _ = out[s]
                row[f"{s}_mean"] = "" if m is None else f"{m:.6g}"
                row[f"{s}_std"] = "" if sd is None else f"{sd:.6g}"
            w.writerow(row)


def write_markdown(agg, path):
    short_keys = [s for _, s, _ in METRICS]
    higher = {s: hi for _, s, hi in METRICS}
    lines = []
    lines.append("# Table A — Headline method comparison (paper plan v2)\n")
    lines.append(
        "Backbone: MobileNetV3. Each cell shows mean ± std across seeds. "
        "**Bold** marks the winner per (dataset, tightness, metric) "
        "(highest F1m/Acc/Sat%; lowest ECE/Brier/Flips/SatEp/Time). "
        "`n` is the seed count actually completed. — = no data.\n"
    )
    for ds in STORY:
        lines.append(f"\n## {STORY[ds]['label']}\n")
        for tight in TIGHTNESS:
            present = [(m, agg.get((ds, tight, m))) for m in METHODS]
            present = [(m, a) for m, a in present if a]
            if not present:
                lines.append(f"### {tight}\n_(no data yet)_\n")
                continue
            lines.append(f"### {tight}\n")
            # find winner per metric
            best = {}
            for s in short_keys:
                vals = []
                for m, a in present:
                    mn, _, _ = a[s]
                    if mn is not None:
                        vals.append((mn, m))
                if not vals:
                    continue
                if higher[s]:
                    best[s] = max(vals)[1]
                else:
                    best[s] = min(vals)[1]
            header = ["Method", "n"] + short_keys
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join(["---"] * len(header)) + "|")
            for m, a in present:
                cells = [METHOD_LABEL[m], str(a["n"])]
                for s in short_keys:
                    txt = fmt_cell(a[s], fmt_for_metric(s))
                    if m == best.get(s) and txt != "—":
                        txt = f"**{txt}**"
                    cells.append(txt)
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    lines.append("\n## Per-method overall summary (across all (ds, tight))\n")
    overall = defaultdict(lambda: defaultdict(list))
    for (ds, tight, method), out in agg.items():
        for s in short_keys:
            mn, _, n = out[s]
            if mn is not None and n > 0:
                overall[method][s].append(mn)
    header = ["Method", "cells"] + short_keys
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for m in METHODS:
        if m not in overall:
            continue
        cells = [METHOD_LABEL[m]]
        any_metric = next(iter(overall[m].values()))
        cells.append(str(len(any_metric)))
        for s in short_keys:
            vals = overall[m].get(s, [])
            if not vals:
                cells.append("—")
            else:
                cells.append(("{:" + fmt_for_metric(s) + "}").format(mean(vals)))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _dedupe_rows(rows):
    """Drop duplicate (ds, tight, method, seed) entries (canonical-tralo
    filter can still leave dupes if the cell exists in two sweep paths)."""
    seen = {}
    for r in rows:
        k = (r["ds"], r["tight"], r["method"], r["seed"])
        if k not in seen:
            seen[k] = r
    return list(seen.values())


def write_per_seed_wide_csv(rows, path):
    """One row per (ds, tight, seed), one column per (method, metric).
    Lets the user eyeball seed-by-seed comparisons across methods."""
    short_keys = [s for _, s, _ in METRICS]
    by_idx = defaultdict(dict)
    for r in rows:
        key = (r["ds"], r["tight"], r["seed"])
        for s in short_keys:
            by_idx[key][(r["method"], s)] = r[s]
    cols = ["ds", "tight", "seed"]
    for m in METHODS:
        for s in short_keys:
            cols.append(f"{m}__{s}")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for (ds, tight, seed) in sorted(by_idx.keys()):
            row = [ds, tight, seed]
            for m in METHODS:
                for s in short_keys:
                    v = by_idx[(ds, tight, seed)].get((m, s))
                    row.append("" if v is None else f"{v:.6g}")
            w.writerow(row)


def write_head_to_head_csv(agg, path):
    """Per (ds, tight): TraLO vs best baseline, with explicit gaps.
    Baselines considered = all non-tralo methods.
    Reports gaps as TraLO_value - best_baseline_value, sign flipped for
    metrics where lower is better so the sign of the gap is always
    'positive = TraLO better'."""
    short_keys = [s for _, s, _ in METRICS]
    higher = {s: hi for _, s, hi in METRICS}
    BASELINES = [m for m in METHODS if m != "tralo"]

    rows = []
    for ds in STORY:
        for tight in TIGHTNESS:
            tr = agg.get((ds, tight, "tralo"))
            if not tr or tr["n"] == 0:
                continue
            row = {"ds": ds, "tight": tight, "tralo_n": tr["n"]}
            for s in short_keys:
                tr_mean, _, _ = tr[s]
                row[f"tralo_{s}"] = "" if tr_mean is None else f"{tr_mean:.4f}"
                base_vals = []
                for b in BASELINES:
                    bg = agg.get((ds, tight, b))
                    if bg:
                        mb, _, _ = bg[s]
                        if mb is not None:
                            base_vals.append((mb, b))
                if not base_vals or tr_mean is None:
                    row[f"best_baseline_{s}"] = ""
                    row[f"best_baseline_method_{s}"] = ""
                    row[f"tralo_gap_{s}"] = ""
                    continue
                if higher[s]:
                    best_v, best_m = max(base_vals)
                    gap = tr_mean - best_v
                else:
                    best_v, best_m = min(base_vals)
                    gap = best_v - tr_mean
                row[f"best_baseline_{s}"] = f"{best_v:.4f}"
                row[f"best_baseline_method_{s}"] = best_m
                row[f"tralo_gap_{s}"] = f"{gap:+.4f}"
            rows.append(row)

    cols = ["ds", "tight", "tralo_n"]
    for s in short_keys:
        cols += [f"tralo_{s}", f"best_baseline_{s}",
                 f"best_baseline_method_{s}", f"tralo_gap_{s}"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _dedupe_rows(collect())
    agg = aggregate(rows)
    write_raw_csv(rows, OUT_DIR / "table_a_raw.csv")
    write_summary_csv(agg, OUT_DIR / "table_a_summary.csv")
    write_markdown(agg, OUT_DIR / "table_a_summary.md")
    write_per_seed_wide_csv(rows, OUT_DIR / "table_a_per_seed.csv")
    write_head_to_head_csv(agg, OUT_DIR / "table_a_head_to_head.csv")
    print(f"rows={len(rows)} groups={len(agg)}")
    for name in ("table_a_raw.csv", "table_a_summary.csv", "table_a_summary.md",
                 "table_a_per_seed.csv", "table_a_head_to_head.csv"):
        print(f"wrote {OUT_DIR/name}")
    target = len(STORY) * len(TIGHTNESS) * len(METHODS) * 4
    have = sum(min(out["n"], 4) for out in agg.values())
    print(f"completion: {have}/{target} cells "
          f"({100*have/target:.1f}%) toward Table A target")


if __name__ == "__main__":
    main()
