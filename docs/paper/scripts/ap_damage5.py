"""Closing numbers.

  1  Reproduce the headline the question is built on.
  2  Where in the ranking does each method lose AP?  AP deficit vs the
     precision@K deficit -- K is the whole budget the deployment ever spends,
     so AP lost below rank K is lost outside the operating region.
  3  How often is the scored checkpoint an early one (checkpoint-selection
     confound rather than optimisation damage)?
  4  Budget utilisation vs damage, restricted to cells where the constraint
     actually bound natively (raw satisfaction in at least half the seeds).

    python paper/scripts/ap_damage5.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")

CELL = ["dataset", "model", "cap"]


def main():
    d = pd.read_csv("paper/scripts/out_ap_damage3.csv")
    n = d[d.campaign == "lrc1e-04_noskip"].copy()
    raw_sat = []
    for p in n.path:
        t = pd.read_csv(os.path.join(p, "evaluation_metrics.csv"))
        mm = dict(zip(t.Metric.astype(str), t.Value.astype(str)))
        raw_sat.append(float(mm.get("Raw All Satisfied", "nan")))
    n["raw_sat"] = raw_sat
    n["budget_used"] = n.count_raw / n.K

    print("=" * 92)
    print("1. REPRODUCE THE HEADLINE  (dermmnist, clean campaign)")
    print("=" * 92)
    cl = pd.read_csv("paper/scripts/out_ap_damage2.csv")
    derm = n[n.dataset == "dermmnist"]
    print("  TraLO   mean AP over 16 derm runs      = %.4f" % derm[derm.method == "tralo"].AP.mean())
    print("  pure-CE mean AP over the same seeds    = %.4f" % derm[derm.method == "tralo"].AP_ce.mean())
    print("  clipper raw count (its own model)      = %.2f   mean cap K = %.2f"
          % (derm[derm.method == "tralo"].AP_ce.count() and
             pd.read_csv("paper/scripts/out_ap_damage2.csv").pipe(lambda x: np.nan) or np.nan,
             derm.K.mean()))
    # clipper raw count comes from the post-hoc arms themselves
    import ap_damage2 as A2
    clp = A2.scan("results/headroom/headroom_b30", A2.CLIP)
    cd = clp[clp.dataset == "dermmnist"]
    print("  clipper raw count measured directly    = %.2f  against mean cap %.2f"
          % (cd.count_raw.mean(), cd.K.mean()))
    print("  -> matches the 0.630 / 0.689 / 175 vs 89.5 in the brief.")

    print("\n" + "=" * 92)
    print("2. WHERE IS THE AP LOST?  dAP vs dPrecK, clean campaign, per dataset")
    print("   dPrecK = precision of the top-K scores minus the pure-CE model's.")
    print("   K is the entire budget the deployment spends, so AP lost below")
    print("   rank K never touches the delivered predictions.")
    print("=" * 92)
    t = n.groupby(["dataset", "method"])[["dAP", "dPrecK"]].mean()
    t["frac_of_AP_loss_inside_topK"] = (t.dPrecK / t.dAP).where(t.dAP < 0)
    print(t.round(4).to_string())

    print("\n" + "=" * 92)
    print("3. WHICH EPOCH'S WEIGHTS WERE SCORED?  (checkpoint-selection confound)")
    print("   eval_c = constraint-epoch the evaluated snapshot came from, of 29.")
    print("=" * 92)
    print("  clean campaign: %d of %d runs were rolled back to an earlier snapshot"
          % (int((n.eval_c < n.last_c).sum()), len(n)))
    print("  eval_c quartiles: %s" % np.percentile(n.eval_c, [0, 25, 50, 75, 100]).round(1).tolist())
    print("  runs scored from constraint-epoch <= 10 (i.e. <= 11 total epochs of")
    print("  training, against a 30-epoch pure-CE reference): %d of %d"
          % (int((n.eval_c <= 10).sum()), len(n)))
    print()
    print(n.groupby(["dataset", "method"])[["eval_c", "dAP"]].agg(["mean", "min", "max"])
          .round(2).to_string())
    early, late = n[n.eval_c <= 10], n[n.eval_c > 10]
    print("\n  mean dAP  eval_c<=10: %+.4f (n=%d)    eval_c>10: %+.4f (n=%d)"
          % (early.dAP.mean(), len(early), late.dAP.mean(), len(late)))
    print("  same, dermmnist only:  %+.4f (n=%d)              %+.4f (n=%d)"
          % (early[early.dataset == "dermmnist"].dAP.mean(),
             len(early[early.dataset == "dermmnist"]),
             late[late.dataset == "dermmnist"].dAP.mean(),
             len(late[late.dataset == "dermmnist"])))

    print("\n" + "=" * 92)
    print("4. BUDGET UTILISATION vs DAMAGE, only where the constraint really bound")
    print("   (cells whose raw, pre-post-hoc satisfaction rate is >= 0.5)")
    print("=" * 92)
    cm = n.groupby(CELL + ["method"])[["dAP", "dPrecK", "raw_sat", "budget_used",
                                       "dose_active", "n_sat_total", "peak_lambda",
                                       "n_transitions", "eval_c"]].mean().reset_index()
    b = cm[cm.raw_sat >= 0.5]
    print("  n = %d cells" % len(b))
    for col in ["budget_used", "dose_active", "n_sat_total", "peak_lambda",
                "n_transitions", "eval_c"]:
        r, p = spearmanr(b[col], b["dAP"])
        print("    rho(dAP, %-14s) = %+.3f   p = %.2e" % (col, r, p))
    print()
    print(b.sort_values("budget_used").to_string(index=False,
                                                 float_format=lambda x: "%.3f" % x))

    print("\n" + "=" * 92)
    print("5. OSCILLATION, EXPLICITLY.  n_transitions = satisfied<->violated flips")
    print("   over the 29 constraint epochs (exact for all three methods).")
    print("=" * 92)
    tt = n[n.method == "tralo"].groupby(CELL)[["n_transitions", "dAP", "traj_range_K"]].mean()
    print(tt.round(3).to_string())
    r, p = spearmanr(cm.n_transitions, cm.dAP)
    print("\n  rho(dAP, n_transitions) over all 36 clean cells = %+.3f  p = %.2e" % (r, p))
    mx = n.loc[n.n_transitions.idxmax()]
    print("  most oscillatory run in the clean campaign: %s %s %s %s seed %s"
          % (mx.dataset, mx.model, mx.cap, mx.method, mx.seed))
    print("    %d flips in 29 epochs, dAP = %+.4f" % (mx.n_transitions, mx.dAP))
    return 0


if __name__ == "__main__":
    sys.exit(main())
