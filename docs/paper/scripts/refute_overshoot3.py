"""Part 3: fragility of every encoding that clears the 0.15 bar, plus the
within-cap strata the pooled r cannot see."""
import math

import numpy as np
import pandas as pd

CELL = ["dataset", "model", "cap"]


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return pearson(pd.Series(x).rank(), pd.Series(y).rank())


def main():
    tr = pd.read_csv("paper/scripts/out_refute_trained.csv")
    cl = pd.read_csv("paper/scripts/out_refute_clip.csv")
    piv = tr.pivot_table(index=CELL + ["seed"], columns="method", values="ccF1eq")
    pc = cl[cl.method.isin(["heuristic", "danits_lp"])].pivot_table(
        index=CELL + ["seed"], columns="method", values="ccF1eq")
    s = piv.copy()
    s["T_D"] = s["tralo"] - s[["fioretto_ldf", "hounie_rcl"]].max(axis=1)
    s = s.reset_index()
    c = s.groupby(CELL)["T_D"].mean().reset_index()
    kt = tr.groupby(CELL).agg(K=("K", "first"), n_pool=("n_pool", "first"),
                              n_true=("n_true", "first")).reset_index()
    nat = cl[cl.method == "heuristic"].groupby(CELL)["count_raw"].mean() \
            .rename("clip_raw").reset_index()
    c = c.merge(kt, on=CELL).merge(nat, on=CELL).sort_values(["dataset", "cap", "model"])
    c["ratio"] = c.clip_raw / c.K
    c["absov"] = c.clip_raw - c.K
    c["natrate"] = c.clip_raw / c.n_pool

    print("=" * 104)
    print("F. LEAVE-ONE-CELL-OUT jackknife of every encoding that cleared r2>=0.15")
    print("=" * 104)
    tests = {
        "overshoot ratio (raw)": (c.ratio.to_numpy(float), c.T_D.to_numpy(float)),
        "overshoot abs (raw)": (c.absov.to_numpy(float), c.T_D.to_numpy(float)),
    }
    for nm, col in [("overshoot abs (-dataset)", "absov"),
                    ("overshoot ratio (-dataset)", "ratio"),
                    ("natural_rate (-dataset)", "natrate")]:
        x = (c[col] - c.groupby("dataset")[col].transform("mean")).to_numpy(float)
        y = (c.T_D - c.groupby("dataset")["T_D"].transform("mean")).to_numpy(float)
        tests[nm] = (x, y)
    tests["natural_rate (raw)"] = (c.natrate.to_numpy(float), c.T_D.to_numpy(float))
    for nm, (x, y) in tests.items():
        full = pearson(x, y)
        jk = [pearson(np.delete(x, i), np.delete(y, i)) for i in range(len(x))]
        signflip = sum(1 for v in jk if np.sign(v) != np.sign(full))
        print("  %-28s r=%+0.3f  jackknife range [%+0.3f, %+0.3f]  "
              "sign flips on %d/12 deletions" % (nm, full, min(jk), max(jk), signflip))
        worst = int(np.argmax([abs(full - v) for v in jk]))
        print("      most influential cell: %s %s %s (drop -> r=%+0.3f)"
              % (c.iloc[worst].dataset, c.iloc[worst].model, c.iloc[worst].cap, jk[worst]))

    print("\n" + "=" * 104)
    print("G. LEAVE-ONE-DATASET-OUT for the only p<0.05 encoding")
    print("=" * 104)
    for drop in sorted(c.dataset.unique()):
        sub = c[c.dataset != drop]
        x = (sub.natrate - sub.groupby("dataset")["natrate"].transform("mean")).to_numpy(float)
        y = (sub.T_D - sub.groupby("dataset")["T_D"].transform("mean")).to_numpy(float)
        print("  drop %-12s n=%d  natural_rate(-dataset) r=%+0.3f" % (drop, len(sub), pearson(x, y)))
    print("  (within a dataset the clipper count is CAP-INVARIANT, so this")
    print("   'encoding' has only 6 distinct values across 12 cells and its")
    print("   within-dataset contrast is exactly one backbone difference each.)")
    print("  within-dataset backbone contrasts of clip_raw:")
    for ds, g in c.groupby("dataset"):
        gg = g.groupby("model").agg(clip=("clip_raw", "first"), T=("T_D", "mean"))
        print("    %-12s %s" % (ds, gg.to_dict()))

    print("\n" + "=" * 104)
    print("H. WITHIN-CAP strata (6 cells each): does overshoot order them?")
    print("=" * 104)
    for cap, g in c.groupby("cap"):
        print("  %s  r(ratio,T_D)=%+0.3f  rho=%+0.3f  r(abs)=%+0.3f"
              % (cap, pearson(g.ratio, g.T_D), spear(g.ratio, g.T_D),
                 pearson(g.absov, g.T_D)))
        print("      %s" % g[["dataset", "model", "ratio", "T_D"]]
              .to_string(index=False, float_format=lambda x: "%.3f" % x).replace("\n", "\n      "))

    print("\n" + "=" * 104)
    print("I. WITHIN-BACKBONE and WITHIN-DATASET strata")
    print("=" * 104)
    for mo, g in c.groupby("model"):
        print("  %-14s n=%d r(ratio,T_D)=%+0.3f rho=%+0.3f" % (mo, len(g),
              pearson(g.ratio, g.T_D), spear(g.ratio, g.T_D)))
    for ds, g in c.groupby("dataset"):
        print("  %-12s n=%d r(ratio,T_D)=%+0.3f  (cells: %s)"
              % (ds, len(g), pearson(g.ratio, g.T_D),
                 ", ".join("%.2f->%+.4f" % (a, b) for a, b in zip(g.ratio, g.T_D))))

    print("\n" + "=" * 104)
    print("J. the actual driver, for contrast")
    print("=" * 104)
    D = pd.get_dummies(c.dataset, drop_first=True).to_numpy(float)
    y = c.T_D.to_numpy(float)
    A = np.hstack([np.ones((12, 1)), D])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print("  dataset identity alone: R2=%.3f  (overshoot ratio alone: R2=%.3f)"
          % (r2, pearson(c.ratio, y) ** 2))
    inter = pd.get_dummies(c.dataset + "|" + c.model, drop_first=True).to_numpy(float)
    A2 = np.hstack([np.ones((12, 1)), inter])
    b2, *_ = np.linalg.lstsq(A2, y, rcond=None)
    r22 = 1 - ((y - A2 @ b2) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print("  dataset x backbone identity: R2=%.3f (11 params on 12 points, so it")
    print("  is not a model -- it only says the variance lives in the labels)" % r22)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
