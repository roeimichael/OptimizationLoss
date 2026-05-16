"""Auto-regenerate paper-ready tables from valid post-fix runs.

Reads every `results/pending_runs/**/evaluation_metrics.csv` whose parent
sweep is in POST_FIX (per method), merges with the new `paper_rerun` sweep,
and writes `paper_results/PAPER_TABLES_v2.md`.

Numbers are mean ± std over the available seeds. Best per row in **bold**.

Run after a batch finishes:
    python paper_results/regenerate_tables.py
"""
import csv, json, os, glob
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "results" / "pending_runs"
OUT = Path(__file__).resolve().parent / "PAPER_TABLES_v2.md"

# Trusted post-fix sweeps per method
POST_FIX = {
    "tralo": {"fix_ce_skip", "fix1_validation", "kl_sweep",
              "overnight_2026_05_14", "paper_rerun"},
    "fioretto_ldf": {"convergence_validation_300", "fix1_validation",
                     "overnight_2026_05_14", "paper_rerun"},
    "hounie_rcl": {"hounie_rerun", "convergence_validation_300",
                   "fix1_validation", "overnight_2026_05_14", "paper_rerun"},
    # Heuristic + danits are warmup+posthoc only — no convergence bug applies
    "heuristic": {"convergence_validation_300", "overnight_2026_05_14",
                  "thesis_ext", "thesis", "thesis_dermmnist", "thesis_eurosat",
                  "thesis_so2sat", "overnight_sweep", "paper_rerun"},
    "danits_lp": {"convergence_validation_300", "overnight_2026_05_14",
                  "thesis_ext", "thesis", "thesis_dermmnist", "thesis_eurosat",
                  "thesis_so2sat", "overnight_sweep", "paper_rerun"},
}

METHOD_LABEL = {
    "tralo": "TraLO (ours)",
    "fioretto_ldf": "Fioretto LDF",
    "hounie_rcl": "Hounie RCL",
    "heuristic": "heuristic",
    "danits_lp": "danits_lp",
}


def parse_cls(cc):
    if isinstance(cc, list):
        return tuple(sorted(cc))
    return (cc,)


def collect_runs():
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
        # For TraLO, only canonical baseline (no KL, no beta)
        hp = cfg["hyperparams"]
        if method == "tralo":
            if hp.get("alpha_kl", 0) != 0 or hp.get("linear_sat_tail", 0) != 0:
                continue
        m = {r["Metric"]: r["Value"] for r in csv.DictReader(open(em))}
        runs.append({
            "method": method,
            "dataset": cfg.get("dataset_mode"),
            "model": cfg.get("model_name"),
            "cls": parse_cls(cfg["dataset_config"].get("constrained_class", [])),
            "tight": cfg.get("constraint_tag"),
            "seed": int(hp.get("seed", 0)),
            "acc": float(m.get("Accuracy", 0) or 0),
            "f1m": float(m.get("F1 (Macro)", 0) or 0),
            "f1_const_avg": _avg_constrained_f1(m, cfg),
            "ece": float(m.get("ECE", 0) or 0),
            "brier": float(m.get("Brier Score", 0) or 0),
            "flips": int(m.get("Flips Required", "0") or "0"),
            "sat": (m.get("Raw All Satisfied", "0") == "1"),
        })
    return runs


def _avg_constrained_f1(metrics, cfg):
    """Mean F1 over the constrained classes."""
    cc = cfg["dataset_config"].get("constrained_class", [])
    if isinstance(cc, int):
        cc = [cc]
    vals = []
    for c in cc:
        v = metrics.get(f"F1_Class{c}")
        if v is not None and v != "":
            try:
                vals.append(float(v))
            except ValueError:
                pass
    return sum(vals) / len(vals) if vals else 0.0


def stat(xs, fmt="{:.4f}"):
    if not xs:
        return "—"
    mu = sum(xs) / len(xs)
    if len(xs) > 1:
        sd = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    else:
        sd = 0
    return f"{fmt.format(mu)} ± {fmt.format(sd)}"


def render_table(runs, dataset, model, cls, tight_list, methods, title):
    by_key = defaultdict(list)
    for r in runs:
        if r["dataset"] != dataset or r["model"] != model or r["cls"] != tuple(cls):
            continue
        if r["tight"] not in tight_list:
            continue
        by_key[(r["tight"], r["method"])].append(r)
    rows = []
    rows.append(f"## {title}\n")
    header = "| tight | metric |" + "|".join(METHOD_LABEL[m] for m in methods) + "|"
    sep = "|---" * (2 + len(methods)) + "|"
    rows.append(header)
    rows.append(sep)
    for t in tight_list:
        for label, key, fmt in [
            ("F1 macro", "f1m", "{:.4f}"),
            ("F1 const", "f1_const_avg", "{:.4f}"),
            ("acc", "acc", "{:.4f}"),
        ]:
            vals = {}
            for m in methods:
                xs = [r[key] for r in by_key.get((t, m), [])]
                vals[m] = (sum(xs)/len(xs)) if xs else None
            if all(v is None for v in vals.values()):
                continue
            # Find best (max for f1/acc, ignoring None)
            valid = [(m, v) for m, v in vals.items() if v is not None]
            best = max(v for _, v in valid)
            cells = []
            for m in methods:
                xs = [r[key] for r in by_key.get((t, m), [])]
                s = stat(xs, fmt)
                if vals[m] == best and len(valid) > 1:
                    s = f"**{s}**"
                cells.append(s)
            rows.append(f"| {t} | {label} | " + " | ".join(cells) + " |")
    rows.append("")
    return "\n".join(rows)


def main():
    runs = collect_runs()
    print(f"Collected {len(runs)} runs across {len(POST_FIX)} methods")
    by_method = defaultdict(int)
    for r in runs:
        by_method[r["method"]] += 1
    for m, n in sorted(by_method.items()):
        print(f"  {m:<14} {n}")

    out = ["# Paper tables v2 (auto-regenerated)\n",
           f"Generated from {len(runs)} valid post-fix runs. "
           "Best per row in **bold** when ≥2 methods have data.\n",
           "## Methods\n",
           "All numbers are **mean ± std over available seeds** (target N=5 for "
           "headline, N=3 for ablations).\n"]

    methods_main = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic", "danits_lp"]
    methods_constr = ["tralo", "fioretto_ldf", "hounie_rcl"]

    # Table 2: TissueMNIST L50 cls 4 across backbones
    for backbone in ("MobileNetV3", "ResNet18", "EfficientNetB0"):
        title = f"Table — TissueMNIST {backbone} L50_G50 class 4"
        out.append(render_table(runs, "tissuemnist", backbone, (4,),
                                ["L50_G50"], methods_main, title))

    # Table 3: TissueMNIST tightness sweep, MobileNetV3
    title = "Table — TissueMNIST MobileNetV3 class 4, tightness sweep"
    out.append(render_table(runs, "tissuemnist", "MobileNetV3", (4,),
                            ["L20_G20", "L30_G30", "L40_G40", "L50_G50",
                             "L60_G60", "L70_G70", "L80_G80"],
                            methods_main, title))

    # Table 4: TissueMNIST multi-class
    for cls in [(4,), (3, 4), (1, 4), (1, 4, 7)]:
        title = f"Table — TissueMNIST MobileNetV3 L50_G50 cls {cls}"
        out.append(render_table(runs, "tissuemnist", "MobileNetV3", cls,
                                ["L50_G50"], methods_main, title))

    # Other datasets if data exists
    for ds in ("dermmnist", "eurosat", "so2sat"):
        for backbone in ("MobileNetV3", "ResNet18", "EfficientNetB0"):
            cls = (5,) if ds == "eurosat" else (7,) if ds == "so2sat" else (4,)
            title = f"Table — {ds} {backbone} L50_G50 cls {cls[0]}"
            tbl = render_table(runs, ds, backbone, cls, ["L50_G50"],
                               methods_main, title)
            # Skip empty tables
            if "—" not in tbl or any("0." in line for line in tbl.split("\n")):
                out.append(tbl)

    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
