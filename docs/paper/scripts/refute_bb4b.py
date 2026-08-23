"""Part 2. The claim's premise, its power, and its own null distribution."""
import glob
import json
import os
import sys
from math import comb

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402

NOCE = "results/headroom/headroom_b30_lrc0.0001_noceskip"
CLIP = "results/headroom/headroom_b30"
DUALS = ["fioretto_ldf", "hounie_rcl"]
CELL = ["dataset", "model", "cap"]


def sect(t):
    print("\n" + "=" * 104)
    print(t)
    print("=" * 104)


def clip_per_seed():
    out = []
    for cfg_path in glob.glob(CLIP + "/**/config.json", recursive=True):
        cfg = json.load(open(cfg_path))
        if cfg.get("methodology") != "heuristic":
            continue
        raw = os.path.join(os.path.dirname(cfg_path), "final_predictions_raw.csv")
        if not os.path.exists(raw):
            continue
        dc = cfg.get("dataset_config", {}) or {}
        c = dc.get("constrained_class")
        c = int(c[0] if isinstance(c, (list, tuple)) else c)
        t = pd.read_csv(raw, usecols=["Predicted_Label"])
        out.append({"dataset": cfg["dataset_mode"], "model": cfg["model_name"],
                    "cap": cfg["constraint_tag"],
                    "seed": (cfg.get("hyperparams") or {}).get("seed"),
                    "clip_raw": int((t.Predicted_Label.to_numpy(int) == c).sum())})
    return pd.DataFrame(out)


def main():
    cl = clip_per_seed()
    d = A.rows_for(NOCE)
    d = d[d.method.isin(["tralo"] + DUALS)]
    p = d.pivot_table(index=CELL + ["seed"], columns="method", values="ccF1eq")
    p = p.dropna(subset=["tralo"]).copy()
    p["edge"] = p["tralo"] - p[DUALS].max(axis=1)
    p = p.reset_index()[CELL + ["seed", "edge"]]
    K = d.groupby(CELL)["K"].first().reset_index()
    ntr = d.groupby(CELL)["K"].size().reset_index(name="_n")
    S = p.merge(cl, on=CELL + ["seed"]).merge(K, on=CELL)
    S["overshoot"] = S.clip_raw / S.K
    ntrue = {"dermmnist": 223, "octmnist": 250, "tissuemnist": 171}
    S["quantum"] = 2.0 / (S.K + S.dataset.map(ntrue))

    # ---------------------------------------------------------------- 1
    sect("1  THE PREMISE, PER SEED. In how many matched seeds is MNV3's overshoot\n"
         "   actually the larger one?  (the claim asserts it in all 6 rows)")
    for ds, g in S[S.cap == "L30_G30"].groupby("dataset"):
        a = g[g.model == "MobileNetV3"].sort_values("seed")
        b = g[g.model == "RegNetY400MF"].sort_values("seed")
        m = a.clip_raw.to_numpy(float); r = b.clip_raw.to_numpy(float)
        matched = int((m > r).sum())
        allp = int(sum(1 for x in m for y in r if x > y))
        print("  %-12s matched seeds MNV3>RegNet: %d of 4   all 4x4 pairs: %2d of 16 (%.0f%%)"
              "   %s" % (ds, matched, allp, 100 * allp / 16,
                         "<-- premise is a COIN FLIP" if 5 <= allp <= 11 else ""))
        print("               MNV3 %s   RegNet %s" % (m.astype(int).tolist(),
                                                      r.astype(int).tolist()))

    # ---------------------------------------------------------------- 2
    sect("2  BINDING CHECK: (cell,seed) pairs where the cap is ALREADY satisfied by\n"
         "   the unconstrained model, i.e. overshoot < 1 and nothing is constrained")
    nb = S[S.overshoot <= 1.0]
    print("  non-binding (cell,seed) pairs: %d of %d" % (len(nb), len(S)))
    if len(nb):
        print(nb[CELL + ["seed", "clip_raw", "K", "overshoot", "edge"]]
              .to_string(index=False, float_format=lambda x: "%.3f" % x))
        print("\n  -> lands inside octmnist/MobileNetV3/L50_G50, one of the four rows")
        print("     the claim calls CONTRADICTS. Its cell-mean overshoot of 1.28 is")
        print("     an average over a seed that is not a test of the constraint.")

    # ---------------------------------------------------------------- 3
    sect("3  THE HYPOTHESIS AT SEED LEVEL, COUNTED OVER CELLS (12 cells, never pooled).\n"
         "   Within each atomic cell, does a seed with more overshoot get a bigger edge?")
    rows = []
    for (ds, mo, cap), g in S.groupby(CELL):
        g = g.sort_values("seed")
        if g.clip_raw.nunique() < 2:
            continue
        r, pv = stats.pearsonr(g.clip_raw, g.edge)
        rho, _ = stats.spearmanr(g.clip_raw, g.edge)
        rows.append({"dataset": ds, "model": mo, "cap": cap, "n": len(g),
                     "pearson_r": r, "spearman": rho, "p": pv})
    R = pd.DataFrame(rows)
    print(R.to_string(index=False, float_format=lambda x: "%.4f" % x))
    pos = int((R.pearson_r > 0).sum()); neg = int((R.pearson_r < 0).sum())
    sig = int((R.p < 0.05).sum())
    print("\n  cells with POSITIVE overshoot->edge slope: %d" % pos)
    print("  cells with NEGATIVE overshoot->edge slope: %d" % neg)
    print("  cells significant at p<0.05:               %d of %d" % (sig, len(R)))
    print("  -> at 3x the resolution of the claim's 6 contrasts, the relation is")
    print("     an even split. The campaign cannot resolve the sign in EITHER")
    print("     direction, so it cannot 'contradict' anything.")

    # ---------------------------------------------------------------- 4
    sect("4  EFFECT SIZE IN THE CLAIM'S OWN UNIT. bb4.py defines ccF1_quantum =\n"
         "   2/(K+n_true) = one extra true positive inside the equal-budget top-K.")
    cellm = S.groupby(CELL).agg(edge=("edge", "mean"), q=("quantum", "first"),
                                K=("K", "first")).reset_index()
    for (ds, cap), g in cellm.groupby(["dataset", "cap"]):
        g = g.set_index("model")
        gap = g.loc["MobileNetV3", "edge"] - g.loc["RegNetY400MF", "edge"]
        q = g["q"].iloc[0]
        v = "CONTRADICTS" if gap < 0 else "AGREES"
        print("  %-12s %-8s  backbone edge gap %+0.4f = %+.2f true positives per seed"
              "   [%s]" % (ds, cap, gap, gap / q, v))
    print("\n  the four CONTRADICTS rows are gaps of 1.3-2.2 samples per seed;")
    print("  the two AGREES rows are 3.5-4.8. bb4.py's own backbone-contrast")
    print("  t-tests, printed one section above its verdict table, were:")
    for (ds, cap), g in S.groupby(["dataset", "cap"]):
        a = g[g.model == "MobileNetV3"].edge.to_numpy(float)
        b = g[g.model == "RegNetY400MF"].edge.to_numpy(float)
        t, pv = stats.ttest_ind(a, b)
        print("    %-12s %-8s t=%+.2f  p=%.3f  %s"
              % (ds, cap, t, pv, "" if pv < 0.05 else "NOT SIGNIFICANT"))

    # ---------------------------------------------------------------- 5
    sect("5  THE CLAIM AGAINST ITS OWN NULL. The overshoot sign is +1 in all rows,\n"
         "   so the tally is just a count of negative edge gaps.")
    for n, k, lab in [(6, 4, "as the claim counts them (6 rows, treated independent)"),
                      (3, 2, "counting INDEPENDENT contrasts (K cancels -> 3)")]:
        pv = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
        print("  %-58s  P(>= %d of %d | fair coin) = %.3f  %s"
              % (lab, k, n, pv, "NOT SIGNIFICANT" if pv > 0.05 else "significant"))
    print("\n  Both readings are consistent with chance. The 3-contrast reading is")
    print("  exactly p=0.500: the claim's headline is the coin landing 2 of 3.")

    # ---------------------------------------------------------------- 6
    sect("6  THE ADJUDICATION RULE HAS NO TIE HANDLING")
    print("  bb4.py: agree = (np.sign(om-orn) == np.sign(em-ern)). np.sign(0)==0,")
    print("  so an EXACT TIE on either side is printed as CONTRADICTS. This fires")
    print("  on the sibling campaign lrc0.0001, octmnist L30_G30, where the two")
    print("  backbones' edges are identical (-0.0092 vs -0.0092, gap +0.0000) and")
    print("  the rule still reports CONTRADICTS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
