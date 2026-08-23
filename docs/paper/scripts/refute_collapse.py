"""INDEPENDENT re-derivation of the 'derm duals collapse / oct duals never
reach feasibility' claim.  Reads RAW files only; does not import dualdiag.py
or collapse.py.  Counts CELLS, never pools.

Schema traps handled explicitly:
  * column case ('epoch' vs 'Epoch')
  * repeated mid-file headers -> to_numeric(errors='coerce') + dropna
  * TraLO's log is SPARSE -> epochs from df[epochcol].max(), never len(df)
  * constrained_class may be a list -> [0]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.utils.constants import UNLIMITED                                  # noqa: E402
from src.training.constraints import (compute_global_constraints,          # noqa: E402
                                      compute_local_constraints)

DUALS = ["fioretto_ldf", "hounie_rcl"]
TRAINED = ["tralo"] + DUALS
CELL = ["dataset", "model", "cap"]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def scan(root):
    """One row per run, everything recomputed from raw files."""
    rows = []
    for cfgp in sorted(glob.glob(root + "/**/config.json", recursive=True)):
        try:
            cfg = json.load(open(cfgp))
        except Exception:
            continue
        d = os.path.dirname(cfgp)
        raw = os.path.join(d, "final_predictions_raw.csv")
        fin = os.path.join(d, "final_predictions.csv")
        logp = os.path.join(d, "training_log.csv")
        if not os.path.exists(raw):
            continue
        t = pd.read_csv(raw)
        dc = cfg.get("dataset_config") or {}
        cls = dc.get("constrained_class")
        cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)
        ncls = int(dc.get("num_classes"))
        y = t["True_Label"].to_numpy(int)
        rawp = t["Predicted_Label"].to_numpy(int)
        g = (t["Group_ID"].to_numpy(int) if "Group_ID" in t.columns
             else np.zeros(len(t), int))
        lp, gp = cfg["constraint"]
        df = pd.DataFrame({"label": y, "grp": g})
        G = compute_global_constraints(df, "label", gp, constrained_class=[cls],
                                       num_classes=ncls)
        L = compute_local_constraints(df, "label", lp, "grp",
                                      constrained_class=[cls], num_classes=ncls)
        if G[cls] >= UNLIMITED:
            continue
        K = int(G[cls])
        cnt_raw = int((rawp == cls).sum())

        # excess of the FINAL (possibly checkpoint-restored) model, exactly the
        # quantity the duals' own restore rule computes.
        fin_exc = max(0, cnt_raw - K)
        for gid, bounds in (L or {}).items():
            if bounds[cls] < UNLIMITED:
                gc = int(((rawp == cls) & (g == gid)).sum())
                fin_exc += max(0, gc - int(bounds[cls]))

        r = {
            "dataset": cfg.get("dataset_mode"), "model": cfg.get("model_name"),
            "cap": cfg.get("constraint_tag"), "method": cfg.get("methodology"),
            "seed": (cfg.get("hyperparams") or {}).get("seed"),
            "K": K, "pool": len(t), "n_true": int((y == cls).sum()),
            "count_raw": cnt_raw, "fill": cnt_raw / K,
            "final_excess_recomputed": fin_exc,
            "final_feasible": int(fin_exc == 0),
        }
        if os.path.exists(fin):
            r["count_adj"] = int((pd.read_csv(fin)["Predicted_Label"]
                                  .to_numpy(int) == cls).sum())

        # ---- training log, robustly ----
        if os.path.exists(logp):
            lg = pd.read_csv(logp)
            ecol = "epoch" if "epoch" in lg.columns else (
                "Epoch" if "Epoch" in lg.columns else None)
            if ecol is not None:
                ep = _num(lg[ecol])
                keep = ep.notna()
                ep = ep[keep].to_numpy(float)
                r["log_rows"] = int(keep.sum())
                r["log_epoch_max"] = float(ep.max()) if len(ep) else np.nan
                if "total_excess" in lg.columns:
                    exc = _num(lg["total_excess"])[keep].to_numpy(float)
                    ok = ~np.isnan(exc)
                    if ok.any():
                        r["log_exc_last"] = float(exc[ok][-1])
                        r["log_exc_min"] = float(np.nanmin(exc))
                        r["ever_feasible_intrain"] = int((exc[ok] == 0).any())
                        r["first_feas_ep"] = (float(ep[ok][exc[ok] == 0][0])
                                              if (exc[ok] == 0).any() else np.nan)
                        r["n_feas_ep"] = int((exc[ok] == 0).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def hdr(s):
    print("\n" + "=" * 108)
    print(s)
    print("=" * 108)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="results/headroom/headroom_b30_lrc0.0001_noceskip")
    ap.add_argument("--out", default="paper/scripts/out_refute_collapse.csv")
    args = ap.parse_args()
    pd.set_option("display.width", 260)
    fl = lambda x: "%.3f" % x  # noqa: E731

    d = scan(args.root)
    d = d[d.method.isin(TRAINED)].copy()
    print("scanned %d trained runs from %s" % (len(d), args.root))
    d.to_csv(args.out, index=False)

    du = d[d.method.isin(DUALS)]

    hdr("A. THE CLAIM'S OWN ROLLUP, re-derived (in-training feasibility, 16 runs/ds/method)")
    for ds, g in du.groupby("dataset"):
        print("  %-12s %2d/%2d dual runs EVER feasible in training | mean epoch_max %.1f "
              "| mean LAST-ROW excess %.1f | mean MIN excess %.1f"
              % (ds, g.ever_feasible_intrain.sum(), len(g), g.log_epoch_max.mean(),
                 g.log_exc_last.mean(), g.log_exc_min.mean()))

    hdr("B. SAME QUESTION ASKED OF THE MODEL THAT WAS ACTUALLY SCORED\n"
        "   (duals restore a checkpoint at the end: best-satisfied, else min-excess.\n"
        "    final_predictions_raw.csv comes from the RESTORED model, not the last epoch.)")
    for ds, g in du.groupby("dataset"):
        print("  %-12s %2d/%2d dual runs FEASIBLE AS SCORED | mean excess of scored model %.1f "
              "| (last-row excess was %.1f)"
              % (ds, g.final_feasible.sum(), len(g),
                 g.final_excess_recomputed.mean(), g.log_exc_last.mean()))
    print()
    print("  per method:")
    for (ds, m), g in du.groupby(["dataset", "method"]):
        print("    %-12s %-14s in-train %2d/16  as-scored %2d/16  "
              "last-row exc %7.1f  scored exc %6.1f"
              % (ds, m, g.ever_feasible_intrain.sum(), g.final_feasible.sum(),
                 g.log_exc_last.mean(), g.final_excess_recomputed.mean()))

    hdr("C. FILL = own raw count / K, PER CELL (4 seeds), ALL THREE DATASETS")
    piv = d.pivot_table(index=CELL, columns="method", values="fill").reset_index()
    cr = d.pivot_table(index=CELL, columns="method", values="count_raw").reset_index()
    KK = d.groupby(CELL)["K"].first().reset_index()
    m = piv.merge(cr, on=CELL, suffixes=("_fill", "_cnt")).merge(KK, on=CELL)
    print(m.to_string(index=False, float_format=fl))

    hdr("D. COUNT THE CELLS, do not average them.  'collapsed' = mean fill < 0.75")
    for ds, g in m.groupby("dataset"):
        line = "  %-12s" % ds
        for meth in TRAINED:
            c = meth + "_fill"
            line += "  %s %d/%d cells<0.75" % (meth, int((g[c] < 0.75).sum()), len(g))
        print(line)

    hdr("E. WHICH DUAL IS THE COMPARATOR?  The headline delta is tralo - MAX(duals).\n"
        "   If the collapsed dual is not the max, its collapse cannot produce the win.")
    fills = d.pivot_table(index=CELL + ["seed"], columns="method", values="fill")
    print("  per-cell mean fill of the two duals:")
    sub = m[["dataset", "model", "cap", "fioretto_ldf_fill", "hounie_rcl_fill",
             "tralo_fill"]]
    print(sub.to_string(index=False, float_format=fl))
    print()
    print("  (fills object shape %s)" % (fills.shape,))
    return 0


if __name__ == "__main__":
    sys.exit(main())
