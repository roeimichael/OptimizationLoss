"""Provenance manifest: for EVERY completed eval cell on the server, record full source
provenance (path, sweep, code_version, status, file mtimes) + the recipe flags that matter
for correctness + key metrics. Lets us audit exactly which runs feed the paper tables/figures
and whether any are stale / out-of-sync / pre-bugfix / recipe-drifted. Output /tmp/provenance.csv.
"""
import csv
import glob
import json
import os
import pandas as pd

ROOTS = ["results", "archive"]
BADSWEEP = ("contamination", "kl_ablation", "component", "dynamics")
OUT = "/tmp/provenance.csv"


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
    for a in ("pending_runs", "server_only_sweeps"):
        if a in parts:
            i = parts.index(a)
            if i + 1 < len(parts):
                return parts[i + 1]
    return fb


rows = []
for root in ROOTS:
    for p in glob.glob(f"{root}/**/evaluation_metrics.csv", recursive=True):
        cfgp = os.path.join(os.path.dirname(p), "config.json")
        if not os.path.exists(cfgp):
            continue
        try:
            cfg = json.load(open(cfgp))
        except (json.JSONDecodeError, OSError):
            continue
        dc = cfg.get("dataset_config", {}) or {}
        hp = cfg.get("hyperparams", {}) or {}
        ds = cfg.get("dataset_mode") or dc.get("dataset_mode")
        model = cfg.get("model_name")
        method = cfg.get("methodology")
        K = dc.get("constrained_class", cfg.get("constrained_class"))
        if None in (ds, model, method) or K is None:
            continue
        sweep = sweep_of(cfg.get("experiment_path", ""), root)
        if any(b in sweep for b in BADSWEEP):
            continue
        try:
            Ki = int(K)
        except (ValueError, TypeError):
            continue
        ev = load_eval(p)
        rows.append(dict(
            sweep=sweep, dataset=ds, model=model, method=method,
            constraint_tag=cfg.get("constraint_tag"), constrained_class=Ki,
            warmup_epochs=hp.get("warmup_epochs", cfg.get("warmup_epochs")),
            seed=hp.get("seed", cfg.get("seed")),
            base_model_id=cfg.get("base_model_id"),
            code_version=cfg.get("code_version"), status=cfg.get("status"),
            group_column=dc.get("group_column"), data_dir=dc.get("data_dir"),
            hybrid_mode=hp.get("hybrid_mode"), reset_opt=hp.get("reset_optimizer_at_sat"),
            alpha_kl=hp.get("alpha_kl"), penalty_mode=hp.get("penalty_mode"),
            enable_ce_skip=hp.get("enable_ce_skip"),
            disable_lambda_toggle=hp.get("disable_lambda_toggle"),
            lambda_global=hp.get("lambda_global"), fior_beta=hp.get("fior_beta"),
            fioretto_step=hp.get("fioretto_step_size"), hounie_alpha=hp.get("hounie_alpha"),
            cfg_mtime=os.path.getmtime(cfgp), eval_mtime=os.path.getmtime(p),
            f1_macro=num(ev, "F1 (Macro)"), cc_f1=num(ev, f"F1_Class{Ki}"),
            flips=num(ev, "Flips Required"), sat=num(ev, "Raw All Satisfied"),
            path=os.path.dirname(p),
        ))

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"{len(df)} completed cells -> {OUT}")
print("\n=== code_version distribution ===")
print(df.code_version.value_counts(dropna=False).head(12).to_string())
print(f"\n=== distinct sweeps: {df.sweep.nunique()} ===")
print("\n=== disable_lambda_toggle values (must be None/False) ===")
print(df.disable_lambda_toggle.value_counts(dropna=False).to_string())
