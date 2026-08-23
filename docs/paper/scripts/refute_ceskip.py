"""Independent re-derivation of the 'noceskip never disabled the gate for the duals' claim.

Read-only. Rebuilds the config census from raw config.json files and checks the
training logs for the CE-saturation signature, per CELL (never pooled).
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = "results/headroom"
CAMPAIGNS = ["headroom_b30", "headroom_b30_lrc0.0001",
             "headroom_b30_lrc0.0001_noceskip", "headroom_b30_lrc5e-05",
             "headroom_b30_lrc0.0001_noceskip_full"]
CLAIM_CENSUS = ["headroom_b30_lrc0.0001_noceskip", "headroom_b30_lrc5e-05",
                "headroom_b30_lrc0.0001_noceskip_full"]


def cellof(cfg, path):
    return (cfg.get("dataset_mode"), cfg.get("model_name"),
            cfg.get("constraint_tag"))


def load_cfgs():
    rows = []
    for camp in CAMPAIGNS:
        for p in glob.glob(os.path.join(ROOT, camp, "**", "config.json"),
                           recursive=True):
            with open(p) as f:
                cfg = json.load(f)
            hp = cfg.get("hyperparams", {})
            # look for the key ANYWHERE in the config, not just hyperparams
            anywhere = []
            def walk(o, pre=""):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if k == "enable_ce_skip":
                            anywhere.append((pre + "/" + k, v))
                        walk(v, pre + "/" + k)
                elif isinstance(o, list):
                    for i, v in enumerate(o):
                        walk(v, pre + "[%d]" % i)
            walk(cfg)
            rows.append({
                "campaign": camp,
                "dir": os.path.dirname(p),
                "method": cfg.get("methodology"),
                "dataset": cfg.get("dataset_mode"),
                "backbone": cfg.get("model_name"),
                "cap": cfg.get("constraint_tag"),
                "seed": hp.get("seed"),
                "status": cfg.get("status"),
                "warmup": hp.get("warmup_epochs"),
                "cepochs": hp.get("constraint_epochs"),
                "lr": hp.get("lr"),
                "lrc": hp.get("lr_constraint"),
                "stable_thr": hp.get("stable_count_threshold"),
                "ce_skip": hp.get("enable_ce_skip", "UNSET"),
                "ce_skip_anywhere": ";".join("%s=%s" % kv for kv in anywhere) or "NONE",
            })
    return pd.DataFrame(rows)


def numeric(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def log_signature(rundir):
    """Return (max_epoch, n_rows, n_nan_ce, first_nan_epoch, ce_col, ep_col)."""
    p = os.path.join(rundir, "training_log.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    ep_col = cols.get("epoch")
    ce_col = None
    for cand in ("ce_loss", "ce loss", "train_ce", "l_ce", "ce"):
        if cand in cols:
            ce_col = cols[cand]
            break
    if ep_col is None:
        return None
    ep = pd.to_numeric(df[ep_col], errors="coerce")
    ok = ep.notna()                       # drops repeated header rows
    ep = ep[ok]
    if ce_col is None:
        return dict(max_epoch=float(ep.max()), n_rows=int(ok.sum()),
                    n_nan_ce=None, first_nan_epoch=None,
                    ce_col=None, ep_col=ep_col, cols=list(df.columns))
    ce_raw = df[ce_col][ok]
    ce = pd.to_numeric(ce_raw, errors="coerce")
    # A header row was already dropped by the epoch filter; remaining NaN in CE
    # is a genuine literal 'nan'/'' written by np.mean([]).
    isnan = ce.isna()
    first = float(ep[isnan].min()) if isnan.any() else None
    return dict(max_epoch=float(ep.max()), n_rows=int(ok.sum()),
                n_nan_ce=int(isnan.sum()), first_nan_epoch=first,
                ce_col=ce_col, ep_col=ep_col, cols=list(df.columns))


def main():
    df = load_cfgs()
    print("=" * 78)
    print("A. CONFIG COUNTS PER CAMPAIGN x METHOD")
    print("=" * 78)
    print(pd.crosstab(df["campaign"], df["method"]).to_string())

    print()
    print("=" * 78)
    print("B. enable_ce_skip VALUE (from hyperparams) BY CAMPAIGN x METHOD")
    print("=" * 78)
    print(pd.crosstab([df["campaign"], df["method"]], df["ce_skip"]).to_string())

    print()
    print("--- key found ANYWHERE in config.json (not just hyperparams) ---")
    print(pd.crosstab([df["method"]], df["ce_skip_anywhere"]).to_string())

    print()
    print("=" * 78)
    print("C. THE CLAIM'S EXACT CENSUS: 3 campaigns %s" % CLAIM_CENSUS)
    print("=" * 78)
    sub = df[df["campaign"].isin(CLAIM_CENSUS)]
    c = Counter(zip(sub["method"], sub["ce_skip"]))
    for k in sorted(c, key=lambda x: (x[0], str(x[1]))):
        print("  %-28s : %d" % (str(k), c[k]))
    print("  TOTAL configs in those 3 campaigns: %d" % len(sub))

    print()
    print("=" * 78)
    print("D. PER-CELL COUNTING (dataset,backbone,cap) -- noceskip campaign")
    print("   does 'UNSET for both duals' hold in EVERY cell, or only on average?")
    print("=" * 78)
    nc = df[df["campaign"] == "headroom_b30_lrc0.0001_noceskip"]
    for method in ["tralo", "fioretto_ldf", "hounie_rcl"]:
        m = nc[nc["method"] == method]
        cells = defaultdict(list)
        for _, r in m.iterrows():
            cells[(r["dataset"], r["backbone"], r["cap"])].append(r["ce_skip"])
        agree = sum(1 for v in cells.values()
                    if len(set(map(str, v))) == 1)
        vals = set()
        for v in cells.values():
            vals.update(map(str, v))
        print("  %-14s cells=%d  uniform-within-cell=%d  values seen=%s  "
              "runs/cell=%s" % (method, len(cells), agree, sorted(vals),
                                sorted({len(v) for v in cells.values()})))

    print()
    print("=" * 78)
    print("E. LOG FORENSICS -- CE-saturation signature (nan CE) per run")
    print("   traps honoured: 'Epoch' vs 'epoch', repeated headers, sparse TraLO")
    print("=" * 78)
    target = ("results/headroom/headroom_b30_lrc0.0001_noceskip/lane0/"
              "MobileNetV3/dermmnist/L30_G30/hounie_rcl/seed_1")
    print("  named-example run: %s" % target)
    sig = log_signature(target)
    print("   ", sig)
    if sig:
        p = os.path.join(target, "training_log.csv")
        raw = pd.read_csv(p)
        print("    raw head of the log (first 30 lines, ce column):")
        print(raw.to_string(max_rows=32))

    print()
    print("  --- all runs in the noceskip campaign, per method ---")
    recs = []
    for _, r in nc.iterrows():
        s = log_signature(r["dir"])
        if s is None:
            recs.append(dict(method=r["method"], dataset=r["dataset"],
                             backbone=r["backbone"], cap=r["cap"],
                             seed=r["seed"], ce_skip=r["ce_skip"],
                             max_epoch=np.nan, n_rows=np.nan, n_nan=np.nan,
                             first_nan=np.nan, missing=True))
            continue
        recs.append(dict(method=r["method"], dataset=r["dataset"],
                         backbone=r["backbone"], cap=r["cap"], seed=r["seed"],
                         ce_skip=r["ce_skip"], max_epoch=s["max_epoch"],
                         n_rows=s["n_rows"], n_nan=s["n_nan_ce"],
                         first_nan=s["first_nan_epoch"],
                         ce_col=s["ce_col"], missing=False))
    L = pd.DataFrame(recs)
    L.to_csv("paper/scripts/out_refute_ceskip_runs.csv", index=False)
    for method in ["tralo", "fioretto_ldf", "hounie_rcl"]:
        m = L[L["method"] == method]
        fired = m["n_nan"].fillna(0) > 0
        print("  %-14s n=%3d  ce_col=%s  runs_with_nan_CE=%d/%d  "
              "median_first_nan_epoch=%s  max_epoch range=[%s,%s]"
              % (method, len(m), sorted(set(m["ce_col"].dropna())),
                 int(fired.sum()), len(m),
                 m.loc[fired, "first_nan"].median() if fired.any() else "-",
                 m["max_epoch"].min(), m["max_epoch"].max()))

    print()
    print("  --- PER CELL: fraction of the 4 seeds where the gate left a nan-CE"
          " trace ---")
    for method in ["tralo", "fioretto_ldf", "hounie_rcl"]:
        m = L[L["method"] == method]
        print("  [%s]" % method)
        for key, g in m.groupby(["dataset", "backbone", "cap"]):
            fired = (g["n_nan"].fillna(0) > 0).sum()
            print("     %-12s %-14s %-8s  gate-fired %d/%d seeds   "
                  "first_nan_epochs=%s  nan_rows=%s  max_epochs=%s"
                  % (key[0], key[1], key[2], fired, len(g),
                     sorted(g["first_nan"].dropna().astype(int).tolist()),
                     sorted(g["n_nan"].dropna().astype(int).tolist()),
                     sorted(g["max_epoch"].dropna().astype(int).tolist())))

    print()
    print("=" * 78)
    print("F. CONTROL: same signature in the OTHER campaigns")
    print("=" * 78)
    for camp in CAMPAIGNS:
        cc = df[df["campaign"] == camp]
        out = []
        for method in ["tralo", "fioretto_ldf", "hounie_rcl"]:
            mm = cc[cc["method"] == method]
            if not len(mm):
                continue
            n_fired = 0
            n_tot = 0
            for _, r in mm.iterrows():
                s = log_signature(r["dir"])
                if s is None or s["n_nan_ce"] is None:
                    continue
                n_tot += 1
                if s["n_nan_ce"] > 0:
                    n_fired += 1
            skset = sorted({str(x) for x in mm["ce_skip"]})
            out.append("%s cfg=%s gate-trace %d/%d" %
                       (method, skset, n_fired, n_tot))
        print("  %-40s %s" % (camp, " | ".join(out)))


if __name__ == "__main__":
    main()
