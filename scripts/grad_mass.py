"""How much gradient does the soft count actually HAVE, and who carries it?

The constraint reaches the weights only through d(soft count)/d(theta) =
sum_i d p_i(c)/d theta, and each item's contribution scales with p_i(1-p_i).
A saturated model has p near 0 or 1 everywhere, so that factor collapses --
and because `constraint_grad_mode: normalize` rescales the summed gradient to a
FIXED norm, whatever tiny, noisy subset still carries mass ends up choosing the
whole direction. This measures that collapse directly.

Reported per capped class: mean p(1-p), the total, and the share of the total
carried by the top 1% and top 5% of items -- concentration is the thing that
turns a weak signal into a random one.

MEASURED on fm2_mn3, against a per-item maximum of 0.25:

  clip / tralo / tralo_null / alm   mean 0.007-0.023, top 1% carries 34% (class 1)
  focal_clip                        mean 0.015-0.045, top 1% carries 5-16%

CAVEAT, stated because it bounds the claim: these are the FINAL model's
probabilities, not the ones present at each constraint step. Train accuracy is
>= 0.95 from epoch 4 of 30 on this cell, so the endpoint stands in for epochs
4-30 and not for the first three.
"""
import glob, json, os, sys, collections
import numpy as np, pandas as pd


def main(globs):
    per = collections.defaultdict(list)
    for g in globs:
        for d in glob.glob(g):
            fp, cj = os.path.join(d, "final_predictions_raw.csv"), os.path.join(d, "config.json")
            if not (os.path.exists(fp) and os.path.exists(cj)):
                continue
            cfg = json.load(open(cj))
            df = pd.read_csv(fp)
            for c in cfg["dataset_config"]["constrained_class"]:
                p = df["Prob_Class_%d" % c].to_numpy()
                w = p * (1.0 - p)
                s = np.sort(w)[::-1]
                tot = s.sum()
                n = len(s)
                per[(cfg["model_name"], cfg["arm"], c)].append(
                    (w.mean(), tot,
                     s[:max(1, n // 100)].sum() / max(tot, 1e-12),
                     s[:max(1, n // 20)].sum() / max(tot, 1e-12)))
    if not per:
        print("no runs matched")
        return 1
    print("gradient weight p(1-p) on the capped classes -- 0.25 is the maximum per item")
    print("  %-12s %-12s %4s %4s %10s %10s %9s %9s" %
          ("backbone", "arm", "cls", "n", "mean w", "total w", "top1%", "top5%"))
    for k in sorted(per):
        v = np.array(per[k])
        print("  %-12s %-12s %4d %4d %10.4f %10.1f %8.0f%% %8.0f%%" %
              (k[0], k[1], k[2], len(v), v[:, 0].mean(), v[:, 1].mean(),
               100 * v[:, 2].mean(), 100 * v[:, 3].mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
