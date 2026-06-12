"""Summarize new_dataset_probes results: paired TraLO vs Fioretto/Hounie
per dataset, plus class_rotation."""
import csv
import glob
import json
import os
from collections import defaultdict
import numpy as np


def read_metrics(p):
    out = {}
    try:
        with open(p) as f:
            for r in csv.DictReader(f):
                out[r["Metric"]] = r["Value"]
    except Exception: return None
    return out


def collect(root):
    rows = []
    for p in glob.glob(f"{root}/**/config.json", recursive=True):
        try:
            with open(p) as f: cfg = json.load(f)
            m = read_metrics(p.replace("config.json","evaluation_metrics.csv"))
            if m is None: continue
            ds = cfg["dataset_mode"]
            method = cfg["methodology"]
            tight = cfg["constraint_tag"]
            seed = cfg["hyperparams"]["seed"]
            cls = cfg["dataset_config"]["constrained_class"]
            macro = float(m["F1 (Macro)"])
            f1c = float(m.get(f"F1_Class{cls}", "nan"))
            flips = float(m.get("Flips Required", "nan"))
            sat = int(m.get("Raw All Satisfied", "0") == "1")
            train_acc = m.get("Train Accuracy", "")
            rows.append({"ds": ds, "method": method, "tight": tight,
                         "seed": seed, "cls": cls,
                         "macro": macro, "f1c": f1c,
                         "flips": flips, "sat": sat, "train_acc": train_acc})
        except Exception as e:
            print(f"skip {p}: {e}")
    return rows


def paired(rows, baseline):
    by_key = defaultdict(dict)
    for r in rows:
        k = (r["ds"], r["tight"], r["seed"], r["cls"])
        by_key[k][r["method"]] = r
    out = []
    for k, by_m in by_key.items():
        if "tralo" in by_m and baseline in by_m:
            out.append((by_m["tralo"], by_m[baseline]))
    return out


def main():
    for root in ("results/pending_runs/new_dataset_probes",
                 "results/pending_runs/class_rotation"):
        if not os.path.isdir(root):
            print(f"missing {root}"); continue
        rows = collect(root)
        print(f"\n{'='*72}")
        print(f"{root}  (n={len(rows)} cells)")
        print(f"{'='*72}")
        if not rows: continue
        # By (ds, method) mean
        by_dm = defaultdict(list)
        for r in rows: by_dm[(r["ds"], r["method"])].append(r["macro"])
        for k, mfs in sorted(by_dm.items()):
            ds, me = k
            print(f"  {ds:<14} {me:<14} n={len(mfs):>2}  macro_f1 mean={np.mean(mfs):.4f}")
        # Paired comparisons
        for bl in ("fioretto_ldf", "hounie_rcl"):
            pairs = paired(rows, bl)
            if not pairs: continue
            by_ds = defaultdict(list)
            for t, b in pairs: by_ds[t["ds"]].append(t["macro"] - b["macro"])
            print(f"\n  --- TraLO vs {bl} (macro_f1 paired) ---")
            for ds, ds_diffs in sorted(by_ds.items()):
                d = np.array(ds_diffs)
                w = (d>1e-4).sum(); l = (d<-1e-4).sum()
                print(f"  {ds:<14} n={len(d):>2}  W/L/T={w}/{l}/{len(d)-w-l}  "
                      f"mean_d={np.mean(d):+.4f}  median_d={np.median(d):+.4f}")
        # flips for fioretto
        pairs_f = paired(rows, "fioretto_ldf")
        if pairs_f:
            print(f"\n  --- TraLO vs Fioretto: flips + sat ---")
            by_ds = defaultdict(list)
            for t, b in pairs_f: by_ds[t["ds"]].append(
                (t["flips"]-b["flips"], t["sat"], b["sat"]))
            for ds, items in sorted(by_ds.items()):
                d = np.array([x[0] for x in items])
                ts = np.mean([x[1] for x in items]); fs = np.mean([x[2] for x in items])
                print(f"  {ds:<14} n={len(d):>2}  d_flips mean={np.mean(d):+.1f}  "
                      f"tralo_sat={ts:.0%}  fioretto_sat={fs:.0%}")


if __name__ == "__main__":
    main()
