"""Backbone interaction, step 3: decompose the edge against the plain-CE floor,
then test what actually predicts it.

d_vs_bestdual = tralo - max(duals) is a DIFFERENCE OF TWO EFFECTS. It rises
either because TraLO gains or because the duals lose. The plain-CE clipper arm
(headroom_b30, warmup 30 / constraint 0, same total epochs) is the untreated
control that separates them, so every trained arm is also scored as
(method - best clipper) in the same cell and the same seed.

Correlations are reported twice: raw across the 12 cells, and after removing the
dataset mean, because dataset identity is the dominant factor (derm +4/4,
oct -4/4) and a raw r over 12 cells is mostly reading dataset labels.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402

CAMP = "results/headroom/headroom_b30_lrc0.0001_noceskip"
CLIPROOT = "results/headroom/headroom_b30"
CELL = ["dataset", "model", "cap"]
DUALS = ["fioretto_ldf", "hounie_rcl"]
CLIP = ["heuristic", "danits_lp"]


def corr(x, y, lab):
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 4:
        print("    %-34s n<4, skipped" % lab)
        return
    r = np.corrcoef(x[ok], y[ok])[0, 1]
    rs = pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman")
    n = int(ok.sum())
    # two-sided p for pearson via t, no scipy dependency
    t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
    print("    %-34s n=%2d  r=%+0.3f  r2=%.3f  spearman=%+0.3f  t=%+0.2f"
          % (lab, n, r, r * r, rs, t))


def main():
    tr = A.rows_for(CAMP)
    cl = A.rows_for(CLIPROOT)
    cl = cl[cl.method.isin(CLIP)]
    print("trained runs %d   clipper runs %d" % (len(tr), len(cl)))

    M = "ccF1eq"
    rows = []
    for metric in ["ccF1eq", "AP", "macroEq"]:
        pt = tr.pivot_table(index=CELL + ["seed"], columns="method", values=metric)
        pc = cl.pivot_table(index=CELL + ["seed"], columns="method", values=metric)
        clipbest = pc.max(axis=1)
        s = pt.copy()
        s["clip"] = clipbest.reindex(s.index)
        s["bestdual"] = s[DUALS].max(axis=1)
        s["tralo_m_clip"] = s["tralo"] - s["clip"]
        s["bestdual_m_clip"] = s["bestdual"] - s["clip"]
        s["tralo_m_bestdual"] = s["tralo"] - s["bestdual"]
        s["fio_m_clip"] = s["fioretto_ldf"] - s["clip"]
        s["hou_m_clip"] = s["hounie_rcl"] - s["clip"]
        s = s.reset_index()
        s["metric"] = metric
        rows.append(s)
    S = pd.concat(rows, ignore_index=True)
    S.to_csv("paper/scripts/out_bb_decomp_perseed.csv", index=False)

    for metric in ["ccF1eq", "AP"]:
        print("\n" + "=" * 122)
        print("DECOMPOSITION  metric=%s   (cell means over 4 seeds, paired within seed)" % metric)
        print("clip = best of {heuristic, danits_lp}, the untreated 30-epoch CE control")
        print("=" * 122)
        g = S[S.metric == metric].groupby(CELL).agg(
            clip=("clip", "mean"), tralo=("tralo", "mean"),
            fio=("fioretto_ldf", "mean"), hou=("hounie_rcl", "mean"),
            T_minus_C=("tralo_m_clip", "mean"),
            D_minus_C=("bestdual_m_clip", "mean"),
            F_minus_C=("fio_m_clip", "mean"),
            H_minus_C=("hou_m_clip", "mean"),
            T_minus_D=("tralo_m_bestdual", "mean"),
            nseed_TgtD=("tralo_m_bestdual", lambda v: int((v > 0).sum())),
        ).reset_index()
        print(g.to_string(index=False, float_format=lambda x: "%+.4f" % x))

    # ---------------- what predicts the edge ------------------------------
    fb = pd.read_csv("paper/scripts/out_factbase.csv")
    fb = fb[fb.campaign == "lrc0.0001_noceskip"]
    tj = pd.read_csv("paper/scripts/out_bb_traj.csv")

    cc = S[S.metric == "ccF1eq"].groupby(CELL).agg(
        T_minus_D=("tralo_m_bestdual", "mean"),
        T_minus_C=("tralo_m_clip", "mean"),
        D_minus_C=("bestdual_m_clip", "mean")).reset_index()

    ft = fb[fb.method == "tralo"][CELL + ["K", "clip_raw", "n_pool", "count_raw"]]
    ft = ft.rename(columns={"count_raw": "tralo_count_raw"})
    cc = cc.merge(ft, on=CELL)
    cc["overshoot_ratio"] = cc.clip_raw / cc.K
    cc["overshoot_abs"] = cc.clip_raw - cc.K
    cc["cut_frac"] = 1 - cc.K / cc.clip_raw

    # collapse + satisfaction of the duals, and TraLO's own satisfaction
    dc = fb[fb.method.isin(DUALS)].groupby(CELL).agg(
        dual_collapsed=("n_collapsed", "sum"),
        dual_count_raw=("count_raw", "mean")).reset_index()
    cc = cc.merge(dc, on=CELL)

    tt = tj[tj.method == "tralo"].groupby(CELL).agg(
        tralo_nsat=("n_sat_rows", "mean"),
        tralo_hard_last=("hard_last", "mean"),
        tralo_lam_last=("lam_last", "mean")).reset_index()
    dd = tj[tj.method.isin(DUALS)].groupby(CELL).agg(
        dual_excess_last=("excess_last", "mean"),
        dual_excess_min=("excess_min", "mean"),
        dual_first_sat=("first_sat_cep", "mean"),
        dual_nsat=("n_sat_rows", "mean")).reset_index()
    cc = cc.merge(tt, on=CELL).merge(dd, on=CELL)
    cc["tralo_over_K"] = cc.tralo_hard_last / cc.K
    cc["dual_over_K"] = cc.dual_count_raw / cc.K
    cc["is_regnet"] = (cc.model == "RegNetY400MF").astype(float)

    print("\n" + "=" * 122)
    print("PREDICTOR TABLE, one row per cell")
    print("=" * 122)
    show = ["dataset", "model", "cap", "K", "clip_raw", "overshoot_ratio",
            "tralo_hard_last", "tralo_over_K", "tralo_nsat", "tralo_lam_last",
            "dual_count_raw", "dual_collapsed", "dual_first_sat", "dual_nsat",
            "T_minus_D", "T_minus_C", "D_minus_C"]
    print(cc[show].sort_values(["dataset", "model", "cap"])
          .to_string(index=False, float_format=lambda x: "%.3f" % x))

    print("\n" + "=" * 122)
    print("CORRELATIONS over the 12 cells (raw), target = TraLO minus best dual (ccF1eq)")
    print("=" * 122)
    y = cc.T_minus_D.to_numpy(float)
    for c in ["overshoot_ratio", "overshoot_abs", "cut_frac", "clip_raw", "K",
              "tralo_over_K", "tralo_nsat", "tralo_lam_last", "dual_collapsed",
              "dual_over_K", "dual_first_sat", "dual_nsat", "is_regnet"]:
        corr(cc[c].to_numpy(float), y, c)

    print("\n  same, target = TraLO minus plain-CE clipper (ccF1eq)")
    y2 = cc.T_minus_C.to_numpy(float)
    for c in ["overshoot_ratio", "cut_frac", "tralo_over_K", "tralo_nsat",
              "is_regnet"]:
        corr(cc[c].to_numpy(float), y2, c)

    print("\n  same, target = best dual minus plain-CE clipper (ccF1eq)")
    y3 = cc.D_minus_C.to_numpy(float)
    for c in ["overshoot_ratio", "cut_frac", "dual_collapsed", "dual_over_K",
              "dual_nsat", "is_regnet"]:
        corr(cc[c].to_numpy(float), y3, c)

    print("\n" + "=" * 122)
    print("DATASET-DEMEANED (within-dataset) correlations: dataset identity is")
    print("the dominant factor, so remove it before reading any of the above.")
    print("=" * 122)
    dm = cc.copy()
    for c in ["T_minus_D", "T_minus_C", "D_minus_C", "overshoot_ratio", "cut_frac",
              "tralo_over_K", "tralo_nsat", "dual_collapsed", "dual_over_K",
              "dual_first_sat", "dual_nsat"]:
        dm[c] = dm[c] - dm.groupby("dataset")[c].transform("mean")
    yd = dm.T_minus_D.to_numpy(float)
    for c in ["overshoot_ratio", "cut_frac", "tralo_over_K", "tralo_nsat",
              "dual_collapsed", "dual_over_K", "dual_first_sat", "dual_nsat"]:
        corr(dm[c].to_numpy(float), yd, c + " (demeaned)")

    print("\n" + "=" * 122)
    print("SEED-LEVEL (n=48) correlation, overshoot is a CELL constant so it")
    print("cannot gain power; the per-seed clipper count CAN.")
    print("=" * 122)
    ps = S[S.metric == "ccF1eq"][CELL + ["seed", "tralo_m_bestdual", "tralo_m_clip"]]
    cr = cl[cl.method == "heuristic"][CELL + ["seed", "count_raw", "K"]].rename(
        columns={"count_raw": "clip_raw_seed"})
    ps = ps.merge(cr, on=CELL + ["seed"], how="left")
    ps["overshoot_seed"] = ps.clip_raw_seed / ps.K
    print("  matched seed-level rows: %d" % ps.overshoot_seed.notna().sum())
    corr(ps.overshoot_seed.to_numpy(float), ps.tralo_m_bestdual.to_numpy(float),
         "overshoot_seed vs T-D")
    d2 = ps.copy()
    for c in ["overshoot_seed", "tralo_m_bestdual"]:
        d2[c] = d2[c] - d2.groupby("dataset")[c].transform("mean")
    corr(d2.overshoot_seed.to_numpy(float), d2.tralo_m_bestdual.to_numpy(float),
         "overshoot_seed vs T-D (demeaned)")
    cc.to_csv("paper/scripts/out_bb_predictors.csv", index=False)
    print("\nwrote paper/scripts/out_bb_predictors.csv, out_bb_decomp_perseed.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
