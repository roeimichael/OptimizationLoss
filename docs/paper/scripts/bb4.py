"""Backbone interaction, step 4: error bars, quantisation, replication.

Three things the cell means hide.

(1) ccF1eq is F1 of ONE class evaluated at a FIXED budget K, so it is quantised.
    equalize() always labels exactly K samples, so precision = TP/K and
    recall = TP/n_true, giving F1 = 2TP/(K+n_true): one extra true positive
    inside the top-K moves ccF1eq by exactly 2/(K+n_true) and nothing else can.
    Every reported gap should be read in that unit before it is called an effect.

(2) The cells are 4 seeds. A +0.02 mean that holds on 2 of 4 seeds is not the
    same object as a +0.02 mean that holds on 4 of 4.

(3) The sibling campaigns re-run the same cells under a different constraint LR
    and with the CE-saturation gate on. If a backbone split is mechanism it
    should survive at least one of them.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402

CELL = ["dataset", "model", "cap"]
DUALS = ["fioretto_ldf", "hounie_rcl"]
CLIP = ["heuristic", "danits_lp"]
ROOTS = {
    "lrc0.0001_noceskip": "results/headroom/headroom_b30_lrc0.0001_noceskip",
    "lrc0.0001":          "results/headroom/headroom_b30_lrc0.0001",
    "lrc5e-05":           "results/headroom/headroom_b30_lrc5e-05",
}


def paired_table(root, metric="ccF1eq"):
    d = A.rows_for(root)
    if d.empty:
        return None
    p = d.pivot_table(index=CELL + ["seed"], columns="method", values=metric)
    have = [m for m in DUALS if m in p.columns]
    if "tralo" not in p.columns or not have:
        return None
    p = p.dropna(subset=["tralo"]).copy()
    p["bestdual"] = p[have].max(axis=1)
    p["d"] = p["tralo"] - p["bestdual"]
    return p.reset_index()


def main():
    # ---------- quantisation unit -----------------------------------------
    fb = pd.read_csv("paper/scripts/out_factbase.csv")
    q = fb[fb.campaign == "lrc0.0001_noceskip"].groupby(CELL).agg(
        K=("K", "first"), n_true=("n_true_cls", "first")).reset_index()
    q["ccF1_quantum"] = 2.0 / (q.K + q.n_true)

    p = paired_table(ROOTS["lrc0.0001_noceskip"])
    st = p.groupby(CELL)["d"].agg(["mean", "std", "count",
                                   lambda v: int((v > 0).sum()),
                                   "min", "max"]).reset_index()
    st.columns = CELL + ["mean", "sd", "n", "n_pos", "min", "max"]
    st["se"] = st.sd / np.sqrt(st.n)
    st["t"] = st["mean"] / st.se
    st = st.merge(q, on=CELL)
    st["TP_per_seed"] = st["mean"] / st.ccF1_quantum

    print("=" * 126)
    print("PER-CELL PAIRED DIFFERENCE  tralo - best dual  (ccF1eq), 4 seeds each")
    print("ccF1_quantum = 2/(K+n_true) = the ccF1eq change caused by ONE extra")
    print("true positive inside the equal-budget top-K.  TP_per_seed = mean/quantum.")
    print("=" * 126)
    print(st[CELL + ["K", "n_true", "ccF1_quantum", "mean", "sd", "se", "t",
                     "n_pos", "min", "max", "TP_per_seed"]]
          .sort_values(["dataset", "model", "cap"])
          .to_string(index=False, float_format=lambda x: "%.4f" % x))

    print("\n  distinct ccF1eq values actually observed on tissuemnist L30_G30")
    print("  (should be multiples of the quantum, confirming the unit):")
    d0 = A.rows_for(ROOTS["lrc0.0001_noceskip"])
    v = np.sort(d0[(d0.dataset == "tissuemnist") &
                   (d0.cap == "L30_G30")].ccF1eq.unique())
    qq = 2.0 / (51 + 171)
    print("   quantum=%.6f  values/quantum = %s" % (qq, np.round(v / qq, 3)[:18]))

    # ---------- backbone contrast, with seed noise -------------------------
    print("\n" + "=" * 126)
    print("BACKBONE CONTRAST per (dataset,cap): mean_d(MNV3) - mean_d(RegNet),")
    print("against the pooled seed-level sd of d.  8 seeds per contrast.")
    print("=" * 126)
    for (ds, cap), g in p.groupby(["dataset", "cap"]):
        a = g[g.model == "MobileNetV3"]["d"].to_numpy(float)
        b = g[g.model == "RegNetY400MF"]["d"].to_numpy(float)
        sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        se = sp * np.sqrt(1 / len(a) + 1 / len(b))
        t = (a.mean() - b.mean()) / se if se > 0 else np.nan
        print("  %-12s %-8s  MNV3 %+0.4f (sd %.4f)  RegNet %+0.4f (sd %.4f)"
              "  diff %+0.4f  t=%+0.2f (df=%d)"
              % (ds, cap, a.mean(), a.std(ddof=1), b.mean(), b.std(ddof=1),
                 a.mean() - b.mean(), t, len(a) + len(b) - 2))

    # ---------- replication across sibling campaigns -----------------------
    print("\n" + "=" * 126)
    print("REPLICATION: same cells, different campaign")
    print("=" * 126)
    reps = {}
    for lab, root in ROOTS.items():
        if not os.path.isdir(root):
            print("  MISSING %s" % root)
            continue
        pp = paired_table(root)
        if pp is None:
            print("  %s: not scorable" % lab)
            continue
        t = pp.groupby(CELL)["d"].agg(["mean", "count",
                                       lambda v: int((v > 0).sum())]).reset_index()
        t.columns = CELL + ["mean", "n", "n_pos"]
        reps[lab] = t
        print("  %-22s cells=%d  seeds=%d" % (lab, len(t), pp.shape[0]))
    if reps:
        wide = None
        for lab, t in reps.items():
            tt = t.rename(columns={"mean": lab, "n_pos": lab + "_pos",
                                   "n": lab + "_n"})
            wide = tt if wide is None else wide.merge(tt, on=CELL, how="outer")
        print()
        cols = CELL + [c for lab in reps for c in (lab, lab + "_pos", lab + "_n")]
        print(wide[cols].sort_values(CELL)
              .to_string(index=False, float_format=lambda x: "%+.4f" % x))
        print("\n  backbone split on tissuemnist, per campaign"
              " (mean d, MNV3 vs RegNet):")
        for lab in reps:
            t = reps[lab]
            tm = t[(t.dataset == "tissuemnist")]
            if tm.empty:
                continue
            a = tm[tm.model == "MobileNetV3"]["mean"]
            b = tm[tm.model == "RegNetY400MF"]["mean"]
            print("    %-22s MNV3 %s   RegNet %s"
                  % (lab, np.round(a.to_numpy(), 4).tolist(),
                     np.round(b.to_numpy(), 4).tolist()))

    # ---------- within-dataset overshoot ordering vs backbone --------------
    print("\n" + "=" * 126)
    print("DIRECT TEST of the overshoot story WITHIN each dataset: does the")
    print("backbone with the LARGER overshoot get the LARGER TraLO edge?")
    print("=" * 126)
    ov = fb[(fb.campaign == "lrc0.0001_noceskip") & (fb.method == "tralo")][
        CELL + ["clip_raw", "K", "d_vs_bestdual"]].copy()
    ov["overshoot"] = ov.clip_raw / ov.K
    for (ds, cap), g in ov.groupby(["dataset", "cap"]):
        g = g.set_index("model")
        try:
            om, orn = g.loc["MobileNetV3", "overshoot"], g.loc["RegNetY400MF", "overshoot"]
            em, ern = g.loc["MobileNetV3", "d_vs_bestdual"], g.loc["RegNetY400MF", "d_vs_bestdual"]
        except KeyError:
            continue
        agree = "AGREES" if np.sign(om - orn) == np.sign(em - ern) else "CONTRADICTS"
        print("  %-12s %-8s  overshoot MNV3 %.2f vs RegNet %.2f (%s bigger); "
              "edge MNV3 %+0.4f vs RegNet %+0.4f (%s bigger)  -> %s"
              % (ds, cap, om, orn, "MNV3" if om > orn else "RegNet", em, ern,
                 "MNV3" if em > ern else "RegNet", agree))

    # ---------- base capacity by backbone ---------------------------------
    print("\n" + "=" * 126)
    print("BASE CAPACITY: the untreated 30-epoch CE model, per backbone")
    print("=" * 126)
    cl = A.rows_for("results/headroom/headroom_b30")
    cl = cl[cl.method == "heuristic"]
    b = cl.groupby(["dataset", "model"]).agg(
        n=("seed", "size"), AP=("AP", "mean"), macroEq=("macroEq", "mean"),
        ccF1eq=("ccF1eq", "mean"), count_raw=("count_raw", "mean"),
        raw_sd=("count_raw", "std")).reset_index()
    print(b.to_string(index=False, float_format=lambda x: "%.4f" % x))
    return 0


if __name__ == "__main__":
    sys.exit(main())
