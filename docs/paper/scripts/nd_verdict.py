"""Early stop/continue verdict for one research arm. RUN ON THE SERVER.

Each arm gets one dedicated GPU and a 24-run smoke (2 datasets x 2 backbones x
2 caps x 3 seeds). This scores whatever has finished so far against the
baselines and says whether the direction is worth its GPU.

The comparison is paired within (dataset, cap, model, seed) against the arm's
own campaign-matched baselines, so a partial smoke is still meaningful: cells
that both arms have finished are compared, and cells only the arm has finished
are ignored rather than averaged in.

DECISION RULE, fixed in advance so it cannot be rationalised after the fact:

  KILL      >= MIN_RUNS finished AND the paired ccF1eq gap versus TraLO is
            below -0.005 (a real regression, outside the +/-0.005 noise band)
            OR macro-F1 has regressed by more than 0.01 (guard breach).
  CONTINUE  the gap is positive and the arm has not finished its smoke.
  PROMOTE   smoke complete AND ccF1eq beats BOTH the incumbent TraLO and the
            post-hoc clipper. This is the bar the project actually needs:
            TraLO-variant > clipper >= duals.
  INCONCLUSIVE otherwise -- reported honestly rather than rounded into a verdict.

    python paper/scripts/nd_verdict.py --arm topk [--min-runs 20]
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402

CELL = ["dataset", "cap", "model", "seed"]
BASE_TRALO = "results/headroom/headroom_b30_lrc0.0001_noceskip"
BASE_CLIP = "results/headroom/headroom_b30"
BAND = 0.005
GUARD = 0.01


def paired_gap(arm, base, metric, base_methods):
    """Arm minus baseline on cells BOTH have finished."""
    b = base[base.method.isin(base_methods)]
    if b.empty or arm.empty:
        return None
    b = b.sort_values("method").groupby(CELL, as_index=False).first()
    j = arm.merge(b, on=CELL, suffixes=("_a", "_b"))
    if j.empty:
        return None
    g = j[metric + "_a"] - j[metric + "_b"]
    return {"n": len(g), "mean": float(g.mean()),
            "won": int((g > 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--min-runs", type=int, default=20)
    ap.add_argument("--root", default=None)
    args = ap.parse_args()

    root = args.root or ("newdirections/arm_%s/results" % args.arm)
    if not os.path.isdir(root):
        print("VERDICT: NO-RUNS   (%s does not exist)" % root)
        return 0
    arm = A.rows_for(root)
    if arm.empty:
        print("VERDICT: NO-RUNS   (nothing scorable under %s)" % root)
        return 0

    print("arm %-12s finished runs: %d" % (args.arm, len(arm)))
    print(arm.groupby("dataset")[["AP", "ccF1eq", "macroEq", "count", "K"]]
          .mean().round(4).to_string())

    tralo = A.rows_for(BASE_TRALO)
    clip = A.rows_for(BASE_CLIP)

    res = {}
    for lab, base, methods in [("vs TraLO", tralo, ["tralo"]),
                               ("vs clipper", clip, A.CLIP),
                               ("vs duals", tralo, ["fioretto_ldf", "hounie_rcl"])]:
        for metric in ["ccF1eq", "AP", "macroEq"]:
            r = paired_gap(arm, base, metric, methods)
            if r:
                res[(lab, metric)] = r
    print("\npaired gaps (arm minus comparator, matched cells)")
    for (lab, metric), r in sorted(res.items()):
        print("  %-12s %-8s %+0.4f   (%d/%d cells won)"
              % (lab, metric, r["mean"], r["won"], r["n"]))

    t = res.get(("vs TraLO", "ccF1eq"))
    c = res.get(("vs clipper", "ccF1eq"))
    mg = res.get(("vs TraLO", "macroEq"))
    n = len(arm)

    verdict, why = "INCONCLUSIVE", "not enough matched cells yet"
    if t and n >= args.min_runs:
        if mg and mg["mean"] < -GUARD:
            verdict = "KILL"
            why = ("macro-F1 guard breached: %+0.4f (limit -%0.3f)"
                   % (mg["mean"], GUARD))
        elif t["mean"] < -BAND:
            verdict = "KILL"
            why = ("ccF1eq %+0.4f vs incumbent TraLO, below the -%0.3f band "
                   "over %d cells" % (t["mean"], BAND, t["n"]))
        elif c and t["mean"] > BAND and c["mean"] > BAND and n >= 24:
            verdict = "PROMOTE"
            why = ("beats incumbent (%+0.4f) AND clipper (%+0.4f) on ccF1eq"
                   % (t["mean"], c["mean"]))
        elif t["mean"] > 0:
            verdict = "CONTINUE"
            why = "ccF1eq %+0.4f and smoke incomplete (%d runs)" % (t["mean"], n)
    elif t:
        verdict = "CONTINUE"
        why = "only %d runs finished, need %d to judge" % (n, args.min_runs)

    print("\nVERDICT: %s   -- %s" % (verdict, why))
    if verdict == "KILL":
        print("   bash paper/scripts/nd_fire.sh %s --dry-run" % args.arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
