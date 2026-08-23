"""Independent re-derivation of 'max consecutive satisfied epochs' per run.

Deliberately does NOT import taxonomy.py -- everything is re-parsed from the raw
training_log.csv / evaluation_metrics.csv / config.json so the prior agent's
reconstruction assumptions are re-tested, not inherited.

    python paper/scripts/refute_satstate.py --root results/headroom/headroom_b30_lrc0.0001_noceskip
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

TRAINED = ["tralo", "fioretto_ldf", "hounie_rcl"]
CELL = ["dataset", "model", "cap"]


def num(s):
    return pd.to_numeric(s, errors="coerce")


def read_log(path):
    t = pd.read_csv(path, dtype=str, low_memory=False)
    key = t.columns[0]
    t = t[t[key] != key]            # trap 4: headers repeat mid-file
    return t


def maxrun(s):
    best = cur = 0
    for v in s:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def evalcsv(d):
    p = os.path.join(d, "evaluation_metrics.csv")
    if not os.path.exists(p):
        return {}
    t = pd.read_csv(p)
    return dict(zip(t["Metric"].astype(str), t["Value"]))


def fnum(v):
    try:
        x = float(v)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/headroom/headroom_b30_lrc0.0001_noceskip")
    ap.add_argument("--out", default="paper/scripts/out_refute_satstate.csv")
    args = ap.parse_args()

    rows = []
    for cfgp in sorted(glob.glob(args.root + "/**/config.json", recursive=True)):
        cfg = json.load(open(cfgp))
        m = cfg.get("methodology")
        if m not in TRAINED:
            continue
        d = os.path.dirname(cfgp)
        lg = os.path.join(d, "training_log.csv")
        if not os.path.exists(lg):
            continue
        hp = cfg["hyperparams"]
        NCE = int(hp.get("constraint_epochs", 29))
        cls = cfg["dataset_config"]["constrained_class"]
        cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)  # trap 5

        t = read_log(lg)
        if m == "tralo":                                             # trap 2
            ep = num(t["Epoch"])
            t = t[ep.notna()]
            ep = ep[ep.notna()]
            e = (ep - 1).astype(int).to_numpy()                       # common index 1..29
            gs = num(t["Global_Satisfied"]).fillna(0).astype(int).to_numpy()
            ls = num(t["Local_Satisfied"]).fillna(0).astype(int).to_numpy()
            sat = ((gs == 1) & (ls == 1)).astype(int)
            ce = num(t["L_CE"]).to_numpy(float)
            cnt = num(t["Hard_Class%d" % cls]).to_numpy(float)
            cstr = (num(t["L_Global"]).fillna(0) + num(t["L_Local"]).fillna(0)).to_numpy(float)
        else:
            ep = num(t["epoch"])
            t = t[ep.notna()]
            ep = ep[ep.notna()]
            e = (ep + 1).astype(int).to_numpy()
            sat = num(t["all_satisfied"]).fillna(0).astype(int).to_numpy()
            ce = num(t["ce_loss"]).to_numpy(float)                    # trap: nan == CE skipped
            cnt = np.full(len(e), np.nan)                             # trap 3
            cstr = num(t["constraint_loss"]).to_numpy(float)

        max_e = int(e.max())
        satmap = dict(zip(e.tolist(), sat.tolist()))
        cemap = dict(zip(e.tolist(), ce.tolist()))
        cstrmap = dict(zip(e.tolist(), cstr.tolist()))

        # -- literal, no-inference reconstruction (trap 1: never use len(df)) --
        E_lit = max_e
        s_lit = np.array([satmap.get(k, 0) for k in range(1, E_lit + 1)], int)

        # -- prior agent's inference: tralo's 5th consecutive sat epoch trips the
        #    break BEFORE the log block, so it is never written. Add it back.
        inferred = False
        E_inf, s_inf = E_lit, s_lit.copy()
        if m == "tralo":
            tail4 = [satmap.get(max_e - k, 0) for k in range(4)]
            if max_e < NCE and all(tail4):
                E_inf = max_e + 1
                s_inf = np.append(s_inf, 1)
                inferred = True

        # -- did the run actually take an optimizer step on its satisfied epochs? --
        sat_e = [k for k in sorted(satmap) if satmap[k] == 1]
        n_frozen_ce = sum(1 for k in sat_e if not np.isfinite(cemap.get(k, np.nan)))
        n_zero_cstr = sum(1 for k in sat_e if (cstrmap.get(k, np.nan) == 0.0))
        n_fully_frozen = sum(1 for k in sat_e
                             if (not np.isfinite(cemap.get(k, np.nan)))
                             and cstrmap.get(k, np.nan) == 0.0)
        # first epoch at which CE stopped being trained anywhere in the run
        ce_all = np.array([cemap.get(k, np.nan) for k in sorted(cemap)])
        ks = sorted(cemap)
        bad = [ks[i] for i in range(len(ks)) if not np.isfinite(ce_all[i])]
        ce_skip_e = bad[0] if bad else None

        ev = evalcsv(d)
        rows.append({
            "path": d, "dataset": cfg["dataset_mode"], "model": cfg["model_name"],
            "cap": cfg["constraint_tag"], "method": m, "seed": hp.get("seed"),
            "NCE": NCE, "max_e_logged": max_e, "rows": len(e),
            "E_lit": E_lit, "maxrun_lit": maxrun(s_lit), "n_sat_lit": int(s_lit.sum()),
            "E_inf": E_inf, "maxrun_inf": maxrun(s_inf), "n_sat_inf": int(s_inf.sum()),
            "inferred_tail": inferred,
            "early_stop": max_e < NCE,
            "first_sat": (int(np.argmax(s_lit) + 1) if s_lit.sum() else None),
            "sat_epoch_csv": fnum(ev.get("Satisfaction Epoch")),
            "final_sat_csv": fnum(ev.get("Raw All Satisfied")),
            "flips": fnum(ev.get("Flips Required")),
            "enable_ce_skip_cfg": hp.get("enable_ce_skip", "ABSENT->default True"),
            "n_sat_epochs_logged": len(sat_e),
            "sat_epochs_with_CE_off": n_frozen_ce,
            "sat_epochs_with_zero_cstr": n_zero_cstr,
            "sat_epochs_fully_frozen": n_fully_frozen,
            "ce_skip_first_e": ce_skip_e,
            "cnt_min": np.nanmin(cnt) if np.isfinite(cnt).any() else np.nan,
            "cnt_max": np.nanmax(cnt) if np.isfinite(cnt).any() else np.nan,
        })

    D = pd.DataFrame(rows)
    D.to_csv(args.out, index=False)
    P = lambda x: print(x.to_string())  # noqa: E731
    pd.set_option("display.width", 220)
    print("runs=%d  -> %s" % (len(D), args.out))
    print("\n" + "=" * 100)
    print("1. THE CLAIMED CROSSTAB, re-derived. rows=method, cols=max consecutive satisfied epochs")
    print("=" * 100)
    print("\n  (a) LITERAL, no tail inference:")
    P(pd.crosstab(D.method, D.maxrun_lit))
    print("\n  (b) WITH the tralo break-epoch inference (prior agent's number):")
    P(pd.crosstab(D.method, D.maxrun_inf))

    print("\n" + "=" * 100)
    print("2. WAS THE MODEL EVEN BEING UPDATED ON ITS 'SATISFIED' EPOCHS?")
    print("   CE off  == ce_loss is NaN == np.mean([]) == the CE batch loop was skipped")
    print("   cstr==0 == no constraint backward/step was taken that epoch")
    print("=" * 100)
    P(D.groupby("method").agg(
        runs=("path", "size"),
        cfg_ce_skip=("enable_ce_skip_cfg", lambda x: sorted(set(map(str, x)))),
        sat_epochs=("n_sat_epochs_logged", "sum"),
        sat_ep_CE_off=("sat_epochs_with_CE_off", "sum"),
        sat_ep_zero_cstr=("sat_epochs_with_zero_cstr", "sum"),
        sat_ep_FULLY_FROZEN=("sat_epochs_fully_frozen", "sum"),
        runs_that_ever_stopped_CE=("ce_skip_first_e", lambda x: int(x.notna().sum()))))

    print("\n  runs reaching maxrun_inf>=5, and how many of those epochs were frozen:")
    hi = D[D.maxrun_inf >= 5]
    P(hi.groupby("method").agg(runs=("path", "size"),
                               sat_ep=("n_sat_epochs_logged", "sum"),
                               fully_frozen=("sat_epochs_fully_frozen", "sum"),
                               CE_off=("sat_epochs_with_CE_off", "sum")))

    print("\n" + "=" * 100)
    print("3. SANITY vs evaluation_metrics.csv (independent source for first satisfaction)")
    print("=" * 100)
    D["sat_norm"] = np.where(D.method == "tralo", D.sat_epoch_csv - 1, D.sat_epoch_csv)
    ok = D.dropna(subset=["sat_epoch_csv"])
    print("  runs with a recorded Satisfaction Epoch : %d" % len(ok))
    print("  of those, reconstructed first_sat agrees: %d" % int((ok.sat_norm == ok.first_sat).sum()))
    nev = D[D.sat_epoch_csv.isna()]
    print("  runs with NO recorded Satisfaction Epoch: %d ; reconstructed n_sat==0 for %d"
          % (len(nev), int((nev.n_sat_lit == 0).sum())))
    print("\n  never-satisfied count by method (two independent sources):")
    P(D.groupby("method").agg(
        never_from_log=("n_sat_lit", lambda x: int((x == 0).sum())),
        never_from_evalcsv=("sat_epoch_csv", lambda x: int(x.isna().sum())),
        n=("path", "size")))

    print("\n" + "=" * 100)
    print("4. EPOCH BUDGET ACTUALLY RUN (trap 1: max(Epoch), never len(df))")
    print("=" * 100)
    P(D.groupby("method").agg(rows_in_csv=("rows", "mean"),
                              max_e=("max_e_logged", "mean"),
                              E_used=("E_inf", "mean"),
                              early_stopped=("early_stop", "sum"),
                              tail_inferred=("inferred_tail", "sum")))

    print("\n" + "=" * 100)
    print("5. COUNT THE CELLS, DO NOT POOL. per (dataset,model,cap): #runs of 4 with maxrun>=2")
    print("=" * 100)
    piv = D.pivot_table(index=CELL, columns="method", values="maxrun_inf",
                        aggfunc=lambda x: "%d/%d" % (int((x >= 2).sum()), len(x)))
    P(piv)
    print("\n  per-cell MEAN maxrun_inf:")
    P(D.pivot_table(index=CELL, columns="method", values="maxrun_inf", aggfunc="mean"))
    print("\n  per-cell #runs that NEVER satisfy (of 4):")
    P(D.pivot_table(index=CELL, columns="method", values="n_sat_lit",
                    aggfunc=lambda x: int((x == 0).sum())))
    print("\n  cells (of 12) where tralo's mean maxrun is strictly below BOTH duals:")
    mp = D.pivot_table(index=CELL, columns="method", values="maxrun_inf", aggfunc="mean")
    lose = ((mp["tralo"] < mp["fioretto_ldf"]) & (mp["tralo"] < mp["hounie_rcl"])).sum()
    print("    %d of %d" % (int(lose), len(mp)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
