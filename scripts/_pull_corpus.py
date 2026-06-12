"""Broad corpus export: EVERY dataset x backbone x method (not just the 3 active), so we
can pick the 3 datasets + 3 backbones that show TraLO best. One row per run; keeps sweep,
warmup, constrained_class, tightness. Excludes incompatible sweeps. Pull /tmp output local.
"""
import csv
import glob
import json
import os
import pandas as pd

ROOTS = ["results", "archive"]
BADSWEEP = ("contamination", "kl_ablation", "component", "dynamics")
OUTP = "/tmp/corpus_full.csv"


def load_eval(p):
    d = {}
    with open(p) as f:
        for r in csv.reader(f):
            if len(r) == 2:
                d[r[0]] = r[1]
    return d


def num(ev, k):
    try:
        return float(ev[k])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def sweep_of(expp, fb):
    parts = str(expp).replace("\\", "/").split("/")
    for anchor in ("pending_runs", "server_only_sweeps"):
        if anchor in parts:
            i = parts.index(anchor)
            if i + 1 < len(parts):
                return parts[i + 1]
    return fb


rows, nf = [], 0
for root in ROOTS:
    for p in glob.glob(f"{root}/**/evaluation_metrics.csv", recursive=True):
        nf += 1
        cfgp = os.path.join(os.path.dirname(p), "config.json")
        if not os.path.exists(cfgp):
            continue
        try:
            cfg = json.load(open(cfgp))
        except (json.JSONDecodeError, OSError):
            continue
        dc = cfg.get("dataset_config", {}) or {}
        dataset = cfg.get("dataset_mode") or dc.get("dataset_mode")
        model = cfg.get("model_name")
        method = cfg.get("methodology")
        tag = cfg.get("constraint_tag")
        K = dc.get("constrained_class", cfg.get("constrained_class"))
        hp = cfg.get("hyperparams", {}) or {}
        warm = hp.get("warmup_epochs", cfg.get("warmup_epochs"))
        seed = hp.get("seed", cfg.get("seed"))
        sweep = sweep_of(cfg.get("experiment_path", ""), root)
        if None in (dataset, model, method) or K is None:
            continue
        if any(b in sweep for b in BADSWEEP):
            continue
        try:
            Ki = int(K)
        except (ValueError, TypeError):
            continue
        ev = load_eval(p)
        rows.append(dict(
            sweep=sweep, dataset=dataset, model=model, method=method, constraint_tag=tag,
            constrained_class=Ki, group_column=dc.get("group_column"), warmup_epochs=warm, seed=seed,
            acc=num(ev, "Accuracy"), f1_macro=num(ev, "F1 (Macro)"),
            cc_f1=num(ev, f"F1_Class{Ki}"), cc_rec=num(ev, f"Recall_Class{Ki}"),
            cc_prec=num(ev, f"Precision_Class{Ki}"),
            flips=num(ev, "Flips Required"), sat=num(ev, "Raw All Satisfied"),
        ))

df = pd.DataFrame(rows)
print(f"scanned {nf} eval files -> {len(df)} runs")
print("\n=== runs per dataset ===")
print(df.dataset.value_counts().to_string())
print("\n=== runs per (dataset, model) [datasets with >=100 runs] ===")
big = df.dataset.value_counts()
big = big[big >= 100].index
g = df[df.dataset.isin(big)].groupby(["dataset", "model"]).size()
print(g.to_string())
df.to_csv(OUTP, index=False)
print(f"\nwrote {OUTP}")
