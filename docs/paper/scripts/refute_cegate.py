"""INDEPENDENT re-derivation of the "CE gate fires" claim.  Read-only.

Does not import hounie_dyn.  Parses every training_log.csv from scratch with
the documented schema traps handled explicitly:
  * epoch column is "epoch" for the duals and "Epoch" for TraLO
  * headers repeat mid-file -> coerce everything, drop rows whose epoch is NaN
  * TraLO's log is SPARSE -> never use len(df) as an epoch count
  * ce_loss NaN is SIGNAL for the duals (np.mean([]) when the CE loop is skipped)

    python paper/scripts/refute_cegate.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")

CELL = ["dataset", "model", "cap"]
NOCE = "results/headroom/headroom_b30_lrc0.0001_noceskip"
GATEON = "results/headroom/headroom_b30_lrc0.0001"
LOWLR = "results/headroom/headroom_b30_lrc5e-05"


def load_log(path):
    """Robust load: coerce all columns, drop repeated-header rows."""
    df = pd.read_csv(path, dtype=str, on_bad_lines="skip")
    epcol = "epoch" if "epoch" in df.columns else ("Epoch" if "Epoch" in df.columns else None)
    if epcol is None:
        return None, None
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    out = out[out[epcol].notna()].reset_index(drop=True)
    return out, epcol


def scan(root):
    rows = []
    for cfg_path in sorted(glob.glob(root + "/**/config.json", recursive=True)):
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            continue
        d = os.path.dirname(cfg_path)
        lp = os.path.join(d, "training_log.csv")
        if not os.path.exists(lp):
            continue
        log, epcol = load_log(lp)
        if log is None or log.empty:
            continue
        hp = cfg.get("hyperparams") or {}
        m = cfg.get("methodology")
        ep = log[epcol].to_numpy(float)
        r = dict(dataset=cfg.get("dataset_mode"), model=cfg.get("model_name"),
                 cap=cfg.get("constraint_tag"), method=m, seed=hp.get("seed"),
                 nrows=len(log),
                 max_ep0=int(ep.max()),           # raw value in the file
                 ce_skip_in_cfg=hp.get("enable_ce_skip", "ABSENT->trainer default True"),
                 path=d)
        # ---- epochs actually executed -----------------------------------
        # duals: 0-indexed epoch of the constraint phase, dense (one row/epoch)
        # tralo: 1-indexed and includes warm-up, SPARSE
        if m in ("fioretto_ldf", "hounie_rcl"):
            r["epochs_run"] = int(ep.max()) + 1
            r["dense"] = (len(log) == r["epochs_run"])
            ce = log["ce_loss"].to_numpy(float)
            nan = np.isnan(ce)
            r["n_ce_off"] = int(nan.sum())
            r["ce_off_ep"] = int(ep[nan][0]) + 1 if nan.any() else None
            # is the NaN block a clean suffix?  the gate never un-fires, so a
            # non-suffix pattern would mean something OTHER than the gate.
            r["nan_suffix"] = bool(nan.any() and nan[np.argmax(nan):].all())
            r["frac_ce_off"] = r["n_ce_off"] / r["epochs_run"]
            sat = log["all_satisfied"].to_numpy(float)
            r["first_sat"] = int(ep[sat == 1][0]) + 1 if (sat == 1).any() else None
        else:  # tralo
            r["epochs_run"] = int(ep.max())       # already 1-indexed incl. warm-up
            r["dense"] = False
            r["n_ce_off"] = 0                     # gate disabled; L_CE always written
            r["ce_off_ep"] = None
            r["nan_suffix"] = False
            r["frac_ce_off"] = 0.0
            if "Train_Acc" in log.columns:
                ta = log["Train_Acc"].to_numpy(float)
                r["train_acc_max"] = float(np.nanmax(ta))
                r["train_acc_final"] = float(ta[-1])
                r["tralo_would_gate"] = bool(np.nanmax(ta) >= 0.995)
            gs = log.get("Global_Satisfied")
            if gs is not None:
                g = gs.to_numpy(float)
                r["first_sat"] = int(ep[g == 1][0]) if (g == 1).any() else None
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 300)

    d = scan(NOCE)
    print("=" * 118)
    print("A. RUN INVENTORY  %s" % NOCE)
    print("=" * 118)
    print(d.groupby(["method"]).agg(n=("seed", "size"),
                                    dense_all=("dense", "all")).to_string())
    print("\nenable_ce_skip as it appears in the CONFIG, by method:")
    print(d.groupby(["method", "ce_skip_in_cfg"]).size().to_string())

    du = d[d.method.isin(["fioretto_ldf", "hounie_rcl"])]
    print()
    print("=" * 118)
    print("B. PER-RUN CE-OFF, BOTH DUALS (16 runs per dataset per method)")
    print("=" * 118)
    for m, g in du.groupby("method"):
        fired = g[g.n_ce_off > 0]
        print("\n--- %s ---" % m)
        print("  runs with ANY CE-off epoch, by dataset: %s"
              % g.groupby("dataset")["n_ce_off"].apply(lambda s: "%d/%d" % ((s > 0).sum(), len(s))).to_dict())
        print("  NaN block is a clean suffix in every fired run: %s"
              % (bool(fired["nan_suffix"].all()) if len(fired) else "n/a"))
        print(g.sort_values(["dataset", "model", "cap", "seed"])
              [["dataset", "model", "cap", "seed", "epochs_run", "dense",
                "ce_off_ep", "n_ce_off", "frac_ce_off", "first_sat"]]
              .to_string(index=False, float_format=lambda x: "%.3f" % x))

    print()
    print("=" * 118)
    print("C. PER-CELL (never pooled): seeds fired / mean first-fire / mean n_ce_off / mean epochs_run")
    print("=" * 118)
    for m, g in du.groupby("method"):
        agg = g.groupby(CELL).apply(lambda s: pd.Series({
            "nseed": len(s),
            "seeds_fired": int((s.n_ce_off > 0).sum()),
            "ce_off_ep": s.ce_off_ep.mean(),
            "n_ce_off": s.n_ce_off.mean(),
            "frac_ce_off": s.frac_ce_off.mean(),
            "epochs_run": s.epochs_run.mean(),
            "first_sat": s.first_sat.mean(),
        }), include_groups=False)
        print("\n--- %s ---" % m)
        print(agg.to_string(float_format=lambda x: "%.4g" % x))

    tr = d[d.method == "tralo"]
    print()
    print("=" * 118)
    print("D. WOULD TRALO'S OWN GATE HAVE FIRED?  (its Train_Acc, gate disabled by config)")
    print("=" * 118)
    if "train_acc_max" in tr.columns:
        print(tr.groupby(CELL).apply(lambda s: pd.Series({
            "nseed": len(s),
            "train_acc_max": s.train_acc_max.mean(),
            "seeds_over_0.995": int((s.train_acc_max >= 0.995).sum()),
            "epochs_run": s.epochs_run.mean(),
            "logged_rows": s.nrows.mean(),
        }), include_groups=False).to_string(float_format=lambda x: "%.4g" % x))

    d.to_csv("paper/scripts/out_refute_cegate_perrun.csv", index=False)
    print("\nwrote paper/scripts/out_refute_cegate_perrun.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
