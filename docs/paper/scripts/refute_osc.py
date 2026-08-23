"""Adversarial re-derivation of the hounie_osc.py "phase transition at the CE gate" claim.

Checks, in order:
  1. Reproduce the headline table (sanity).
  2. Is the CE-off segment contiguous / how much of it is NaN-dropped?
  3. Does the trajectory actually TURN at the gate?  (gate epoch vs peak epoch)
  4. TIME-MATCHED PLACEBO: apply the same statistic to a fake gate placed at the
     same epoch fraction in runs that never gated, and to the pre-gate tail.
  5. Same statistic on the DIRECTLY LOGGED total_excess (hard counts), which is
     not a reconstruction at all.
  6. COUNT CELLS, do not average them.
  7. "essentially zero net drift" -- what is the net count movement in counts?

    python paper/scripts/refute_osc.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import hounie_dyn as H  # noqa: E402

ROOTS = {"lrc1e-4": "results/headroom/headroom_b30_lrc0.0001_noceskip",
         "lrc5e-5": "results/headroom/headroom_b30_lrc5e-05",
         "lrc1e-4_gateon": "results/headroom/headroom_b30_lrc0.0001"}


def seg(s):
    """EXACTLY hounie_osc.seg -- do not change."""
    s = np.asarray(s, float)
    s = s[np.isfinite(s)]
    if len(s) < 3:
        return None
    d = np.diff(s)
    step = float(np.mean(np.abs(d)))
    drift = float((s[-1] - s[0]) / len(d))
    return step, drift, abs(drift) / step if step > 0 else np.nan, len(s)


def collect():
    runs = []
    for tag, root in ROOTS.items():
        for p in sorted(glob.glob(root + "/**/config.json", recursive=True)):
            cfg = json.load(open(p))
            if cfg.get("methodology") != "hounie_rcl":
                continue
            r = H.load_run(p)
            if r is None:
                continue
            r["campaign"] = tag
            runs.append(r)
    return runs


def main():
    runs = collect()
    print("hounie_rcl runs loaded:", len(runs))
    rows = []
    for r in runs:
        T = H.hounie_traj(r)
        log = r["log"]
        ep = log["epoch"].to_numpy(int)
        ce = log["ce_loss"].to_numpy(float)
        exc = log["total_excess"].to_numpy(float)
        sat = log["all_satisfied"].to_numpy(int)
        soft = T["soft"]
        lam = T["lam"]
        off = np.isnan(ce)
        n = len(ep)
        gate_ep = int(ep[off][0]) if off.any() else None

        d = dict(campaign=r["campaign"], dataset=r["dataset"], model=r["model"],
                 cap=r["cap"], seed=r["seed"], K=r["K"], N=r["N"], n_rows=n,
                 last_ep=int(ep.max()), gate_ep=gate_ep,
                 n_off=int(off.sum()), n_on=int((~off).sum()),
                 n_nan_soft=int(np.isnan(soft).sum()),
                 n_nan_soft_off=int(np.isnan(soft[off]).sum()) if off.any() else 0,
                 lam_min=float(np.nanmin(lam)), lam_final=float(lam[-1]),
                 first_sat=(int(ep[sat == 1][0]) if sat.any() else None),
                 n_sat=int(sat.sum()), exc_final=float(exc[-1]),
                 soft_final=float(soft[-1]) if np.isfinite(soft[-1]) else np.nan)

        # --- the claim's own statistic ---
        for lbl, mask in [("on", ~off), ("off", off)]:
            v = seg(soft[mask])
            if v is not None:
                d["step_" + lbl], d["drift_" + lbl], d["ratio_" + lbl], d["npts_" + lbl] = v
                d["netcounts_" + lbl] = v[1] * (v[3] - 1)

        # --- (3) does it TURN at the gate? peak of the reconstructed soft ---
        fin = np.isfinite(soft)
        if fin.sum() >= 3:
            idx = np.where(fin)[0]
            pk = idx[int(np.nanargmax(soft[fin]))]
            d["peak_ep_soft"] = int(ep[pk])
            # last epoch at which soft rises (end of the non-monotone region)
            ds = np.diff(soft[fin])
            up = np.where(ds > 0)[0]
            d["last_rise_ep_soft"] = int(ep[idx[up[-1] + 1]]) if len(up) else int(ep[idx[0]])
        # peak of the DIRECTLY LOGGED hard excess
        d["peak_ep_exc"] = int(ep[int(np.argmax(exc))])
        dexc = np.diff(exc)
        upe = np.where(dexc > 0)[0]
        d["last_rise_ep_exc"] = int(ep[upe[-1] + 1]) if len(upe) else int(ep[0])

        # --- (5) same statistic on the raw logged excess ---
        for lbl, mask in [("on", ~off), ("off", off)]:
            v = seg(exc[mask])
            if v is not None:
                d["excratio_" + lbl] = v[2]
                d["excstep_" + lbl] = v[0]
                d["excdrift_" + lbl] = v[1]

        # --- (4) TIME-MATCHED PLACEBO ---
        # split every run at the same epoch index, whether or not it gated
        for cut in (10, 12, 15):
            v = seg(soft[ep >= cut])
            if v is not None:
                d["ratio_late%d" % cut] = v[2]
                d["npts_late%d" % cut] = v[3]
            v = seg(soft[ep < cut])
            if v is not None:
                d["ratio_early%d" % cut] = v[2]
        # tail of the SAME length as the CE-off segment but taken from CE-ON rows
        if off.any() and (~off).sum() >= 3:
            L = int(off.sum())
            on_idx = np.where(~off)[0]
            v = seg(soft[on_idx[-min(L, len(on_idx)):]])
            if v is not None:
                d["ratio_on_tail"] = v[2]
                d["npts_on_tail"] = v[3]
        rows.append(d)

    d = pd.DataFrame(rows)
    d.to_csv("paper/scripts/out_refute_osc.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 100)
    f = lambda x: "%.3f" % x  # noqa: E731

    print()
    print("=" * 118)
    print("(1) REPRODUCTION of hounie_osc (soft, pooled by dataset -- the claim's own view)")
    print("=" * 118)
    m = d.melt(id_vars=["campaign", "dataset", "model", "cap", "seed"],
               value_vars=["ratio_on", "ratio_off"], var_name="phase", value_name="ratio")
    print(m.dropna().groupby(["campaign", "dataset", "phase"]).agg(
        runs=("ratio", "size"), ratio=("ratio", "mean")).to_string(float_format=f))

    print()
    print("=" * 118)
    print("(2) HOW MUCH DATA IS THE 'CE off' RATIO BUILT FROM?  (per campaign x dataset)")
    print("=" * 118)
    g = d.groupby(["campaign", "dataset"]).agg(
        runs=("seed", "size"),
        runs_that_gated=("gate_ep", lambda s: int(s.notna().sum())),
        runs_with_usable_off_seg=("ratio_off", lambda s: int(s.notna().sum())),
        mean_off_pts=("npts_off", "mean"), min_off_pts=("npts_off", "min"),
        mean_on_pts=("npts_on", "mean"))
    print(g.to_string(float_format=f))

    print()
    print("=" * 118)
    print("(3) DOES THE TRAJECTORY TURN AT THE GATE?  gated runs only")
    print("    lag = (epoch the count stops rising) - (gate epoch).  0 => transition AT the gate.")
    print("=" * 118)
    gt = d[d.gate_ep.notna()].copy()
    gt["lag_soft_peak"] = gt.peak_ep_soft - gt.gate_ep
    gt["lag_soft_lastrise"] = gt.last_rise_ep_soft - gt.gate_ep
    gt["lag_exc_peak"] = gt.peak_ep_exc - gt.gate_ep
    gt["lag_exc_lastrise"] = gt.last_rise_ep_exc - gt.gate_ep
    print(gt.groupby(["campaign", "dataset"]).agg(
        runs=("seed", "size"), gate_ep=("gate_ep", "mean"),
        peak_soft=("peak_ep_soft", "mean"), lag_soft_peak=("lag_soft_peak", "mean"),
        lag_soft_lastrise=("lag_soft_lastrise", "mean"),
        peak_exc=("peak_ep_exc", "mean"), lag_exc_peak=("lag_exc_peak", "mean"),
        lag_exc_lastrise=("lag_exc_lastrise", "mean")).to_string(float_format=f))
    print()
    print("  runs where the count was STILL RISING after the gate (lag>0):  soft %d/%d   excess %d/%d"
          % (int((gt.lag_soft_lastrise > 0).sum()), len(gt),
             int((gt.lag_exc_lastrise > 0).sum()), len(gt)))
    print("  correlation across runs of gate_ep vs peak_ep_soft: r = %.3f  (n=%d)"
          % (gt[["gate_ep", "peak_ep_soft"]].corr().iloc[0, 1], len(gt)))
    print("  correlation across runs of gate_ep vs peak_ep_exc : r = %.3f"
          % gt[["gate_ep", "peak_ep_exc"]].corr().iloc[0, 1])

    print()
    print("=" * 118)
    print("(4) TIME-MATCHED PLACEBO: same statistic on the LATE part of every run,")
    print("    including runs that NEVER gated (tissue).  If 'late' alone reproduces the")
    print("    effect, the gate explains nothing.")
    print("=" * 118)
    cols = ["ratio_early10", "ratio_late10", "ratio_early12", "ratio_late12",
            "ratio_early15", "ratio_late15", "ratio_on", "ratio_off", "ratio_on_tail"]
    cols = [c for c in cols if c in d.columns]
    print(d.groupby(["campaign", "dataset"])[cols].mean().to_string(float_format=f))
    print()
    print("  never-gated runs only (the 'CE always on' population):")
    ng = d[d.gate_ep.isna()]
    if len(ng):
        print(ng.groupby(["campaign", "dataset"])[
            [c for c in cols if c in ng.columns]].agg(["mean", "size"]).iloc[:, :8]
            .to_string(float_format=f))

    print()
    print("=" * 118)
    print("(5) SAME STATISTIC ON THE DIRECTLY LOGGED total_excess (hard counts, NOT reconstructed)")
    print("=" * 118)
    ec = [c for c in ["excratio_on", "excratio_off", "excstep_on", "excstep_off",
                      "excdrift_on", "excdrift_off"] if c in d.columns]
    print(d.groupby(["campaign", "dataset"])[ec].mean().to_string(float_format=f))

    print()
    print("=" * 118)
    print("(6) COUNT CELLS, DO NOT AVERAGE THEM.  cell = (dataset, backbone, cap)")
    print("=" * 118)
    cell = d.groupby(["campaign", "dataset", "model", "cap"]).agg(
        seeds=("seed", "size"),
        gated=("gate_ep", lambda s: int(s.notna().sum())),
        off_seg=("ratio_off", lambda s: int(s.notna().sum())),
        r_on=("ratio_on", "mean"), r_off=("ratio_off", "mean"),
        r_late12=("ratio_late12", "mean"),
        er_on=("excratio_on", "mean"), er_off=("excratio_off", "mean"))
    print(cell.to_string(float_format=f))
    print()
    for camp in sorted(d.campaign.unique()):
        c = cell.loc[camp]
        tot = len(c)
        have = int(c.off_seg.gt(0).sum())
        strong = int(((c.r_off > 0.9) & (c.r_on < 0.3)).sum())
        print("  %-16s cells=%d | cells with ANY CE-off segment: %d | cells showing "
              "the claimed transition (r_off>0.9 AND r_on<0.3): %d"
              % (camp, tot, have, strong))

    print()
    print("=" * 118)
    print("(7) 'ESSENTIALLY ZERO NET DRIFT' -- net movement of the count, IN COUNTS")
    print("=" * 118)
    nd = d.groupby(["campaign", "dataset"]).agg(
        K=("K", "mean"),
        net_on=("netcounts_on", "mean"), net_off=("netcounts_off", "mean"),
        pts_on=("npts_on", "mean"), pts_off=("npts_off", "mean"),
        ratio_on=("ratio_on", "mean"), ratio_off=("ratio_off", "mean"))
    nd["net_on_over_K"] = nd.net_on / nd.K
    print(nd.to_string(float_format=f))
    print("\nwrote paper/scripts/out_refute_osc.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
