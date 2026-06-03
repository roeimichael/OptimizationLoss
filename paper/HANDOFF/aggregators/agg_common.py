"""Shared helpers for G1–G4 aggregators.

Each gap aggregator scans a single sweep root, reads every
`evaluation_metrics.csv` + its sibling `config.json`, and emits a
flat per-(cell, seed) CSV in the same schema as `docs/all_cells_raw.csv`:

    ds,model,cls,grp,tight,L,G,method,seed,
    f1m,f1w,acc,ece,brier,flips,sat,sat_epoch,phase
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
COLUMNS = ["ds", "model", "cls", "grp", "tight", "L", "G", "method", "seed",
           "f1m", "f1w", "acc", "ece", "brier", "flips", "sat", "sat_epoch", "phase"]


def _first(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def collect_sweep(sweep_root: str, phase_tag: str) -> list[dict]:
    """Return flat rows for every completed cell under `sweep_root`."""
    root = REPO / sweep_root
    rows: list[dict] = []
    for ev_path in glob.glob(str(root / "**" / "evaluation_metrics.csv"), recursive=True):
        cfg_path = Path(ev_path).with_name("config.json")
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        with open(ev_path) as f:
            r = list(csv.DictReader(f))
        if not r:
            continue
        ev = r[0]
        tight = cfg.get("constraint_tag", "")
        try:
            L = int(tight.split("_")[0][1:])
            G = int(tight.split("_")[1][1:])
        except Exception:
            L = G = 0
        row = {
            "ds": cfg.get("dataset_mode"),
            "model": cfg.get("model_name"),
            "cls": cfg.get("dataset_config", {}).get("constrained_class"),
            "grp": cfg.get("dataset_config", {}).get("group_column"),
            "tight": tight, "L": L, "G": G,
            "method": cfg.get("methodology"),
            "seed": cfg.get("hyperparams", {}).get("seed"),
            "f1m":   _first(ev, "macro_f1", "f1_macro"),
            "f1w":   _first(ev, "weighted_f1", "f1_weighted"),
            "acc":   _first(ev, "accuracy", "acc"),
            "ece":   _first(ev, "ece", "expected_calibration_error"),
            "brier": _first(ev, "brier_score", "brier"),
            "flips": _first(ev, "posthoc_flips_total", "post_hoc_flips", "flips"),
            "sat":   _first(ev, "satisfied", "in_train_sat", "sat"),
            "sat_epoch": _first(ev, "satisfaction_epoch", "sat_epoch"),
            "phase": phase_tag,
        }
        rows.append(row)
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})


def summarize(rows: list[dict], group_keys: list[str]) -> list[dict]:
    """Mean/std across seeds, keyed by (group_keys + method)."""
    from collections import defaultdict
    bins = defaultdict(list)
    for r in rows:
        key = tuple(r.get(k) for k in group_keys + ["method"])
        bins[key].append(r)
    out = []
    for key, items in bins.items():
        kv = dict(zip(group_keys + ["method"], key))
        kv["n_seeds"] = len(items)
        for col in ("f1m", "f1w", "acc", "ece", "brier", "flips", "sat"):
            vals = [float(r[col]) for r in items if r.get(col) not in (None, "")]
            if not vals:
                continue
            kv[f"{col}_mean"] = sum(vals) / len(vals)
            if len(vals) > 1:
                m = kv[f"{col}_mean"]
                kv[f"{col}_std"] = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
            else:
                kv[f"{col}_std"] = 0.0
        out.append(kv)
    return out
