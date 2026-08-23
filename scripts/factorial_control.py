"""Is an unseen group's novelty REAL, or does the screen's baseline invent it?

FRAMEWORK 2(n) gives an unseen test group the GLOBAL training prevalence as its
baseline, on the argument that a model which never saw camera 501 holds no
prior for it. That argument is exactly right for an ATOMIC group -- a camera, a
hospital, a trap -- and `dataset_screen` is sound there.

🛑 IT IS TOO GENEROUS FOR A GROUP BUILT AS A PRODUCT OF FACTORS THAT BOTH
APPEAR IN TRAINING. A model that has seen (head/neck, 60s) and (upper
extremity, 70s) can interpolate a prior for (head/neck, 70s) far better than
the global one, so the "novelty" the screen credits is partly information the
training set already carries -- which is the one thing 2(n) exists to exclude.

This matters in items, not in principle. On the ISIC 2019 slice grouped by
(body site x age band) over two hospitals, NET is +2169 under the global
baseline and +380 once the model is credited with interpolating the marginals:
**82% of the headline was the baseline, not the dataset.**

THE CONTROL. Re-measure NET with each unseen group's baseline replaced by the
independence (raking) estimate  p(c|f0) * p(c|f1) / p(c),  renormalised, with
both marginals taken from TRAINING. Whatever survives is novelty the factor
structure does not already supply.

✅ IT PASSES ITS NEGATIVE CONTROL. Every atomic-group dataset must be
unaffected, because there are no factors to interpolate from, and all five are:

    iwildcam 100.1%   cct 99.9%   idaho 100.0%   wcs 99.8%   serengeti 100.0%
    isic (site x age, BCN+HAM)  17.6%      isic (src x site)  72.7%

⚠️ READ THE ABSOLUTE ITEMS, NOT ONLY THE RATIO. The raking estimate is itself
fitted on training data, so on a small training set it is noisy and can be
WORSE than the global prior -- one ISIC/BCN seed returns 128%. The ratio is a
direction, the surviving item count is the number.

    python -m scripts.factorial_control <slice-dir> ...      # no images, no GPU
"""
import argparse
import os

import numpy as np
import pandas as pd

from scripts.dataset_screen import _dev


def control(path, sep="|", n_null=200, seed=0):
    tr = pd.read_csv(os.path.join(path, "train_meta.csv"))
    te = pd.read_csv(os.path.join(path, "test_meta.csv"))
    rng = np.random.default_rng(seed)
    classes = sorted(set(tr["label"]) | set(te["label"]))
    idx = {c: i for i, c in enumerate(classes)}

    def cc(frame):
        out = np.zeros(len(classes))
        for c, k in frame["label"].value_counts().items():
            out[idx[c]] = k
        return out

    n_tr, n_te = len(tr), len(te)
    p_glob = cc(tr) / n_tr
    for f in (tr, te):
        s = f["location"].astype(str)
        f["_f0"] = s.str.split(sep, regex=False).str[0]
        f["_f1"] = s.str.split(sep, regex=False).str[-1]
    p_f0 = {k: cc(g) / len(g) for k, g in tr.groupby("_f0")}
    p_f1 = {k: cc(g) / len(g) for k, g in tr.groupby("_f1")}

    units_glob, units_add, n_unseen = [], [], 0
    for g in sorted(te["location"].unique()):
        te_g = te[te["location"] == g]
        tr_g = tr[tr["location"] == g]
        if len(tr_g):
            unit = (cc(te_g), cc(tr_g) / len(tr_g), len(te_g))
            units_glob.append(unit)
            units_add.append(unit)
            continue
        n_unseen += 1
        f0, f1 = te_g["_f0"].iloc[0], te_g["_f1"].iloc[0]
        units_glob.append((cc(te_g), p_glob, len(te_g)))
        if f0 in p_f0 and f1 in p_f1 and f0 != f1:
            q = np.divide(p_f0[f0] * p_f1[f1], p_glob,
                          out=np.zeros_like(p_glob), where=p_glob > 0)
            q = q / q.sum() if q.sum() > 0 else p_glob
        else:
            # ATOMIC group (no separator) or a factor itself unseen: the global
            # prior IS the best the model can do, so the screen was right.
            q = p_glob
        units_add.append((cc(te_g), q, len(te_g)))

    shift = np.divide(cc(te) / n_te, p_glob, out=np.ones_like(p_glob),
                      where=p_glob > 0)

    def net_expect(p, n):
        q = p * shift
        tot = q.sum()
        return (q / tot * n) if tot > 0 else p * n

    def net(units):
        obs = sum(_dev(o, net_expect(p, n)) for o, p, n in units)
        null = np.array([sum(_dev(rng.multinomial(n, net_expect(p, n) / n),
                                  net_expect(p, n)) for _, p, n in units)
                         for _ in range(n_null)])
        ex = obs - float(null.mean())
        sd = float(null.std(ddof=1))
        return ex, (ex / sd if sd > 0 else float("nan"))

    a, az = net(units_glob)
    b, bz = net(units_add)
    return {"path": path, "unseen": n_unseen, "net_global": a, "z_global": az,
            "net_additive": b, "z_additive": bz,
            "survives": (100 * b / a) if a else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="slice dirs with train/test_meta.csv")
    ap.add_argument("--sep", default="|", help="factor separator in `location`")
    args = ap.parse_args()
    print("FACTORIAL-GROUP CONTROL -- how much unseen-group novelty survives")
    print("once the model is credited with interpolating the two factors?")
    print("An ATOMIC group must return ~100%: there is nothing to interpolate.")
    print("")
    for p in args.paths:
        r = control(p, sep=args.sep)
        name = os.path.basename(os.path.dirname(p.rstrip("/\\"))) or p
        print("  %-24s unseen=%2d  NET(global) %+7.0f z=%5.1f   "
              "NET(additive) %+7.0f z=%5.1f   survives %6.1f%%"
              % (name[-24:], r["unseen"], r["net_global"], r["z_global"],
                 r["net_additive"], r["z_additive"], r["survives"]))


if __name__ == "__main__":
    main()
