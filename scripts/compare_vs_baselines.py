"""Compare TraLO vs Fioretto LDF, Hounie RCL, AND Danits LP separately.

The earlier winning_conditions.py used Danits LP (post-hoc, structurally
easier) as the baseline. This is the correct head-to-head: TraLO vs the
SOTA trained baselines.
"""
import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

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
TABLES = Path("paper/HANDOFF/tables")


def read_metrics(p):
    out = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            out[row["Metric"]] = row["Value"]
    return out


def collect():
    rows = []
    for root in SWEEPS:
        if not os.path.isdir(root):
            continue
        sweep_name = root.rsplit("/", 1)[-1]
        for cfg_p in glob.glob(f"{root}/**/config.json", recursive=True):
            try:
                with open(cfg_p) as f: cfg = json.load(f)
                m = read_metrics(cfg_p.replace("config.json","evaluation_metrics.csv"))
            except Exception:
                continue
            try:
                ds = cfg["dataset_mode"]
                model = cfg["model_name"]
                tight = cfg["constraint_tag"]
                method = cfg["methodology"]
                seed = cfg["hyperparams"]["seed"]
                cls = cfg["dataset_config"]["constrained_class"]
                n = cfg["dataset_config"]["num_classes"]
                pretrained = cfg["hyperparams"].get("pretrained", True)
                data_dir = cfg["dataset_config"]["data_dir"]
            except Exception:
                continue
            sig = f"{sweep_name}|{ds}|{model}|{data_dir}|{tight}|{cls}|{pretrained}|seed{seed}"
            try:
                macro = float(m["F1 (Macro)"])
            except Exception:
                continue
            f1c = float(m.get(f"F1_Class{cls}", "nan"))
            uf = []
            for c in range(n):
                if c == cls: continue
                v = m.get(f"F1_Class{c}")
                if v:
                    try: uf.append(float(v))
                    except: pass
            f1u = float(np.mean(uf)) if uf else float("nan")
            flips = float(m.get("Flips Required", "nan"))
            sat = 1 if m.get("Raw All Satisfied","0") == "1" else 0
            rows.append({"sig": sig, "sweep": sweep_name, "ds": ds, "model": model,
                         "tight": tight, "method": method, "seed": seed,
                         "macro": macro, "f1c": f1c, "f1u": f1u,
                         "flips": flips, "sat": sat,
                         "tight_pct": int(tight.split("_")[0][1:])})
    return rows


def paired_report(rows, baselines):
    by_sig = defaultdict(dict)
    for r in rows:
        by_sig[r["sig"]][r["method"]] = r
    for base in baselines:
        print(f"\n{'='*70}")
        print(f"TraLO vs {base}")
        print(f"{'='*70}")
        paired = [(c["tralo"], c[base]) for c in by_sig.values()
                  if "tralo" in c and base in c]
        print(f"Paired (cell, seed) comparisons: {len(paired)}")
        if not paired: continue

        by_ds = defaultdict(list)
        for tr, bl in paired:
            by_ds[tr["ds"]].append((tr, bl))

        print(f"\n  {'dataset':<14}{'n':>5}{'macro W/L/T':>14}{'dF1_macro':>12}{'dF1_cstr':>11}{'dF1_uncstr':>12}")
        for ds in ("tissuemnist", "dermmnist", "aider"):
            pairs = by_ds.get(ds, [])
            if not pairs: continue
            dmac = np.array([t["macro"] - b["macro"] for t, b in pairs])
            dcst = np.array([t["f1c"] - b["f1c"] for t, b in pairs
                              if not np.isnan(t["f1c"]) and not np.isnan(b["f1c"])])
            duns = np.array([t["f1u"] - b["f1u"] for t, b in pairs
                              if not np.isnan(t["f1u"]) and not np.isnan(b["f1u"])])
            w = int(np.sum(dmac > 1e-4))
            l = int(np.sum(dmac < -1e-4))
            t_ = len(dmac) - w - l
            print(f"  {ds:<14}{len(dmac):>5} {w}/{l}/{t_:<8}  {np.mean(dmac):+10.4f} {np.mean(dcst):+10.4f} {np.mean(duns):+11.4f}")

        # Per-tightness breakdown
        print(f"\n  Per (dataset, tightness) macro-F1 W/L/T")
        print(f"  {'ds':<14}{'tight':<10}{'n':>5}{'W/L/T':>10}{'dF1':>10}")
        for ds in ("tissuemnist", "dermmnist", "aider"):
            for tight in ("L20_G20","L30_G30","L50_G50","L70_G70","L80_G80"):
                pairs = [(t, b) for t, b in by_ds.get(ds, []) if t["tight"] == tight]
                if not pairs: continue
                dmac = np.array([t["macro"] - b["macro"] for t, b in pairs])
                w = int(np.sum(dmac > 1e-4))
                l = int(np.sum(dmac < -1e-4))
                t_ = len(dmac) - w - l
                print(f"  {ds:<14}{tight:<10}{len(dmac):>5}  {w}/{l}/{t_:<7} {np.mean(dmac):+8.4f}")


def main():
    rows = collect()
    print(f"Total per-seed rows: {len(rows)}")
    paired_report(rows, ["fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"])


if __name__ == "__main__":
    main()
