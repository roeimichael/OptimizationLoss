"""How far is the DECISION BOUNDARY from the CUT? The regime reversal, measured.

WHY THIS EXISTS. Every count penalty this project ships puts its per-item
gradient at or near the DECISION BOUNDARY. `sum`'s weight is `p(1-p)`, which
peaks at p = 0.5, i.e. where an item is one step from flipping; `margin`
windows that point explicitly, via `sigma'(m/T)` at `m = p_ic - max_{c'!=c}
p_ic'`. But deployed quality is decided at the CUT: the post-hoc allocator
emits exactly K items by probability rank, whatever the model's hard count is.

Those are two different items, and `hard_count - K` is the distance between
them -- between the point the penalty can push and the point the metric reads.

⚠️ STATUS: AN UNREFUTED ACCOUNT THAT THIS DESIGN CANNOT DISCRIMINATE.
Read this before quoting the geometry as a cause. Tested 2026-08-30 over the
five campaigns below, and the test was mostly negative:

  * `gap`, `slope_K` and `K/n` are NOT three hypotheses. Within one warm-up
    model the hard count is constant across every cap tag (verified, 40/40
    groups), so `gap = hard - K` is an exact decreasing affine function of K:
    measured `rho(gap, K) = -1.0000`. `K/n` is exactly increasing in K and
    `slope_K` near-deterministically so (mean rho +0.937). Inside a model they
    are ONE VARIABLE IN THREE COSTUMES and cannot be separated at all.
  * The only variation that could pry them apart is between backbones at a
    fixed cap: 10 strata of n=4, where a PERFECT ordering reaches at best
    p=0.083. Stratified permutation, 20k draws: gap~delta p=0.30,
    slope_K~delta p=0.55. Null.
  * Both `gap` and `slope_K` REVERSE SIGN once the cap is held fixed
    (`rho(gap, delta)` pooled -0.291, within TIGHT **+0.590**, p=0.002), so
    their pooled correlations are between-cap artefacts. Only `K/n` orders the
    effect robustly -- and `K/n` is a restatement of the cap, not a mechanism.
  * ⛔ **THE SHARP PREDICTION FAILED.** This account predicted
    `tralo_uniform` should order OPPOSITELY in `gap`. It does not: the two
    arms' slopes have the SAME sign at every level, and uniform's one nominally
    significant result dies against its own floor (+0.451 p=0.012 raw ->
    +0.126 p=0.492 floor-corrected).

✅ WHAT IS SOLID, AND IT IS THE PART WORTH KEEPING. The regime step itself is
real and cleanly attributable. Paired on the **12 CNN warm-up models that
appear in BOTH regimes** -- identical base weights, so no backbone, host or amp
confound -- `tralo` moves **+6.24 items from tight to loose in 12 of 12**
(sign p=0.00049, the exact floor at n=12), while **the reseed floor does not
move** (5/12, p=0.77). Floor-corrected it is +5.30 items, 12/12. The geometry
below is a plausible and unrefuted account OF that step; it is not a measured
cause of it, and it must not be cited as one.

🛑 TO ACTUALLY TEST IT you must vary `gap` at FIXED `K/n`, which this design
cannot do -- the hard count is constant within a model. That needs a dataset or
backbone whose calibration differs far more than the 3-126 item spread here, or
a deliberate miscalibration arm.

MEASURED GEOMETRY (the numbers themselves are not in doubt; their causal
reading is). Over 5 campaigns, 4 backbones, 2 capped classes, 4 seeds:

    regime          cap          gap (items)   p at the cut   slope_bd/slope_K
    LOOSE  K/n=0.90  L90_G95           3 -  79   0.587-0.986        1.9 -   6.6
    LOOSE  K/n=0.80  L80_G95          40 - 126   0.718-0.999        2.8 -  47.3
    TIGHT  K/n=0.30  L30/L50         198 - 397   0.997-1.000       79   - 32387
    TIGHT  K/n=0.20  L20_G50         235 - 442   0.999-1.000      420   - 81926

At a tight cap the cut sits at p = 0.9999, where `p(1-p)` is 0.0001. That is a
fact about where the gradient is, and it is compatible with -- but does not
establish -- the claim that misplacement is WHY the tight regime fails.

⚠️ TWO STRUCTURAL FACTS THIS TOOL WILL NOT TELL YOU, and both change the n:
  * `dom1`'s L80_G95 and L90_G95 cells are **byte-identical to `loose1`'s**
    (80/80 `final_predictions.csv`). `dom1` contributes only L95_G80.
  * The CNN warm-ups are **shared across campaigns**: one warm-up per
    (model, seed) spans `uniform1` + `loose1` + `dom1`. Only **20 distinct
    warm-up models exist** across all five, 12 of them in both regimes. Only
    ViTB16 has separate warm-ups per regime.
  ⇒ count units by WARM-UP, not by campaign or by cell.

K is taken from the arm's own `final_predictions.csv`, which emits exactly K,
so this needs no cap arithmetic and makes no assumption about which scope
binds. The geometry is measured on the `tralo_null` twin -- the model before
any constraint acted -- so it belongs to the cap, not to the arm.

    python -m scripts.cut_gap <campaign-root> [<campaign-root> ...]
    python -m scripts.cut_gap --self-test
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# Below this the cut sits where p(1-p) has effectively vanished and no count
# function differentiating `sum_i p_ic` can move it. Calibrated on the tight
# campaigns above, so treat it as a flag rather than a law.
DEAD_SLOPE = 0.005


def per_group_cut(p, alloc_pred, groups, cls):
    """The cut the ALLOCATOR actually makes, which is PER GROUP.

    `np.argsort(-p)[K-1]` is the globally K-th ranked item. The allocator never
    emits that set: it takes top-k_g WITHIN each group, so the real cut is one
    probability PER GROUP and the global one is an upper bound on every one of
    them -- the global top-K is by construction the selection that maximises
    the minimum selected probability.

    ⚠️ ONLY THE MINIMUM IS ORDERABLE AGAINST THE GLOBAL READING, and I got
    this wrong first. The global top-K maximises the MINIMUM selected
    probability, so `min_g(cut_g) <= p_K_global` always. The budget-weighted
    MEAN has no such relation -- it is a mean against a minimum, and where the
    budgets roughly track group difficulty it comes out ABOVE the global value.
    Measured on a synthetic tree with per-group budgets at 60% of each group's
    own argmax count: mean 0.690 vs global 0.681, i.e. the opposite direction
    from the one asserted. Both are returned; the min is the one with a proof
    attached. FRAMEWORK 2(z64).

    Returns (budget-weighted mean cut, mean slope, groups cut, min cut,
    max slope over groups). `max slope` is the reachability answer the global
    reading structurally could not give: whether ANY group's cut sits where
    `p(1-p)` is large, rather than whether the average one does.

    THE SLOPE IS AVERAGED, NOT RECOMPUTED FROM THE AVERAGED p. `p(1-p)` is
    concave, so by Jensen `pbar(1-pbar) >= mean(p_g(1-p_g))`: taking the slope
    of the mean cut would overstate the per-item gradient and replace one bias
    with another one pointing the other way.
    """
    if groups is None:
        return None, None, None, None, None
    num = den = slope = 0.0
    ncut, lo, hi = 0, 1.0, 0.0
    for gg in np.unique(groups):
        idx = np.where(groups == gg)[0]
        k_g = int((alloc_pred[idx] == cls).sum())
        if k_g == 0:                      # K=0 ceiling: no cut exists here
            continue
        cut = float(np.sort(p[idx])[-k_g])
        num += cut * k_g
        slope += cut * (1.0 - cut) * k_g
        den += k_g
        ncut += 1
        lo = min(lo, cut)
        hi = max(hi, cut * (1.0 - cut))
    if den == 0:
        return None, None, None, None, None
    return num / den, slope / den, ncut, lo, hi


def measure(root, arm="tralo_null"):
    """Per (cap, class, seed) geometry for one campaign root. Returns a frame."""
    rows = []
    camp = os.path.basename(os.path.normpath(root))
    for cfg_path in glob.glob(os.path.join(root, "*", "*", "*", "*",
                                           "seed_*", "config.json")):
        d = os.path.dirname(cfg_path)
        raw_p = os.path.join(d, "final_predictions_raw.csv")
        alloc_p = os.path.join(d, "final_predictions.csv")
        if not (os.path.exists(raw_p) and os.path.exists(alloc_p)):
            continue
        cfg = json.load(open(cfg_path))
        if cfg.get("status") != "completed" or cfg.get("arm") != arm:
            continue
        parts = os.path.normpath(d).split(os.sep)
        model, cap, seed = parts[-5], parts[-3], parts[-1]
        classes = (cfg.get("dataset_config") or {}).get("constrained_class") or []

        raw, alloc = pd.read_csv(raw_p), pd.read_csv(alloc_p)
        y = raw["True_Label"].to_numpy()
        groups = (raw["Group_ID"].to_numpy() if "Group_ID" in raw.columns
                  else None)
        pcols = sorted((int(c[len("Prob_Class_"):]), c) for c in raw.columns
                       if c.startswith("Prob_Class_"))
        if not pcols:
            continue
        P = np.column_stack([raw[c].to_numpy() for _, c in pcols])
        raw_pred = P.argmax(1)
        acol = ("Predicted_Label" if "Predicted_Label" in alloc.columns
                else [c for c in alloc.columns if "redict" in c][0])
        alloc_pred = alloc[acol].to_numpy()

        for cls in classes:
            cls = int(cls)
            K = int((alloc_pred == cls).sum())
            hard = int((raw_pred == cls).sum())
            if not (0 < K <= len(P) and 0 < hard <= len(P)):
                continue
            order = np.argsort(-P[:, cls])
            # THE GLOBAL READING, KEPT ONLY SO THE PUBLISHED TABLE IN THIS
            # DOCSTRING STAYS REPRODUCIBLE. It is not the cut the allocator
            # makes; `p_K` below is. FRAMEWORK 2(z64).
            pk_g = float(P[order[K - 1], cls])
            pb = float(P[order[hard - 1], cls])
            pk, sl, ncut, pk_lo, sl_hi = per_group_cut(
                P[:, cls], alloc_pred, groups, cls)
            if pk is None:      # no Group_ID column: no per-group cut exists
                pk, sl, ncut = pk_g, pk_g * (1 - pk_g), 0
                pk_lo, sl_hi = pk_g, pk_g * (1 - pk_g)
            rows.append(dict(campaign=camp, model=model, cap=cap, seed=seed,
                             cls=cls, K=K, hard=hard,
                             n_pos=int((y == cls).sum()), gap=hard - K,
                             p_K=pk, p_K_global=pk_g, cut_groups=ncut,
                             p_K_min=pk_lo, slope_max=sl_hi, p_bd=pb,
                             slope_K=sl, slope_bd=pb * (1 - pb)))
    return pd.DataFrame(rows)


def summarise(df, by_model=True):
    """Per (campaign, model, cap, class). MODEL IS IN THE KEY, deliberately.

    This grouped by (campaign, cap, cls) when it was written, which POOLS
    ACROSS BACKBONES -- a direct violation of the project's own rule that the
    atomic cell is (dataset, backbone, cap, method) and that nothing is ever
    averaged across backbones. Every number the first version printed was a
    backbone-average wearing a per-cap label. The geometry reproduces once
    disaggregated, but the table was wrong and a per-cell claim could not have
    been checked against it. Caught 2026-08-30.
    """
    keys = ["campaign", "model", "cap", "cls"] if by_model else ["campaign", "cap", "cls"]
    g = (df.groupby(keys)
           .agg(seeds=("seed", "nunique"), K=("K", "mean"), hard=("hard", "mean"),
                n_pos=("n_pos", "mean"), gap=("gap", "mean"), p_K=("p_K", "mean"),
                p_K_glob=("p_K_global", "mean"), cut_grp=("cut_groups", "mean"),
                p_Kmin=("p_K_min", "mean"), slope_max=("slope_max", "mean"),
                slope_K=("slope_K", "mean"), slope_bd=("slope_bd", "mean"))
           .reset_index())
    g["K_over_n"] = g.K / g.n_pos
    g["ratio"] = g.slope_bd / g.slope_K.replace(0, np.nan)
    return g.sort_values(["cls", "K_over_n"] + (["model"] if by_model else [])
                         + ["campaign"])


def report(g, out=sys.stdout):
    pd.set_option("display.width", 240)
    print("BOUNDARY-TO-CUT GAP  (on the lambda=0 twin: the model before any "
          "constraint)\n", file=out)
    print(g.to_string(index=False, float_format=lambda v: "%9.4f" % v), file=out)
    print("\ngap   = hard_count - K, in ITEMS. The penalty pushes at the boundary;",
          file=out)
    print("        the metric reads the cut. This is how far apart they are.", file=out)
    print("ratio = p(1-p) at the boundary / at the cut. How many times more pull",
          file=out)
    print("        the penalty has where the metric does NOT look.\n", file=out)
    print("p_K   = the PER-GROUP cut, budget-weighted over the `cut_grp` groups",
          file=out)
    print("        that hold a nonzero ceiling. p_K_glob is the old GLOBAL rank-K",
          file=out)
    print("        reading, kept only so the docstring's table reproduces.",
          file=out)
    print("p_Kmin/slope_max = the DEEPEST group's cut. Only this one is ordered",
          file=out)
    print("        against p_K_glob (the global top-K maximises the minimum",
          file=out)
    print("        selected probability); the MEAN can fall either side, so do",
          file=out)
    print("        NOT read a p_K vs p_K_glob comparison as a direction.\n",
          file=out)
    if "slope_max" in g.columns:
        n_live = int((g.slope_max >= DEAD_SLOPE).sum())
        print("    %d of %d rows have at least ONE group whose cut carries live"
              % (n_live, len(g)), file=out)
        print("    gradient (slope_max >= %.3f), against %d by the averaged cut."
              % (DEAD_SLOPE, int((g.slope_K >= DEAD_SLOPE).sum())), file=out)
        print("    The global reading could not report this column at all.\n",
              file=out)

    dead = g[g.slope_K < DEAD_SLOPE]
    if len(dead):
        print("*** %d of %d (campaign, cap, class) points have the CUT in a dead "
              "zone" % (len(dead), len(g)), file=out)
        print("    (p(1-p) < %.3f at rank K). No count differentiating `sum_i "
              "p_ic`" % DEAD_SLOPE, file=out)
        print("    can move those cuts at any dose or shape. Their K/n is %s."
              % ", ".join("%.2f" % v for v in sorted(set(round(x, 2)
                                                         for x in dead.K_over_n))),
              file=out)
    live = g[g.slope_K >= DEAD_SLOPE]
    if len(live):
        print("    The %d live points sit at K/n = %s." % (
            len(live), ", ".join("%.2f" % v for v in
                                 sorted(set(round(x, 2) for x in live.K_over_n)))),
              file=out)
    if len(dead) and len(live):
        print("\n    => the split is by CAP FRACTION, not by dataset or backbone.",
              file=out)


def self_test(out=sys.stdout):
    """Does the instrument report the geometry it claims to, and can it say NO?

    Three synthetic models with a KNOWN answer. The gate is that the tight case
    and the loose case come out on opposite sides, and that a perfectly
    calibrated model -- hard count exactly K -- reports a gap of zero rather
    than something merely small.
    """
    rng = np.random.default_rng(0)
    n, n_pos = 3000, 400
    # a score distribution with a realistic saturated head
    p = np.clip(rng.beta(0.35, 0.35, n), 1e-6, 1 - 1e-6)
    srt = np.sort(p)[::-1]
    hard = int((p > 0.5).sum())

    ok = True
    print("SELF-TEST -- synthetic scores, n=%d, hard count %d\n" % (n, hard), file=out)
    print("  %-10s %6s %6s %10s %10s" % ("case", "K", "gap", "p_at_K", "slope_K"),
          file=out)
    seen = {}
    for name, K in (("tight", int(0.20 * n_pos)), ("loose", int(0.95 * n_pos)),
                    ("exact", hard)):
        pk = float(srt[K - 1])
        seen[name] = (hard - K, pk, pk * (1 - pk))
        print("  %-10s %6d %6d %10.4f %10.6f"
              % (name, K, hard - K, pk, pk * (1 - pk)), file=out)

    # ---- THE CUT IS PER GROUP, NOT GLOBAL (FRAMEWORK 2(z64)) ----------------
    # A fixture where the two rules give DIFFERENT answers, which is the only
    # kind that gates anything: group 0 is confident and gets a budget of 1,
    # group 1 is not and gets 2, so the allocator is forced down to p=0.30
    # while a global top-3 never goes below 0.97.
    pg = np.array([0.99, 0.98, 0.97, 0.60, 0.30, 0.05])
    grp = np.array([0, 0, 0, 1, 1, 1])
    sel = np.array([1, 0, 0, 1, 1, 0])          # k_0 = 1, k_1 = 2, K = 3
    gcut = float(np.sort(pg)[::-1][int(sel.sum()) - 1])
    pk_pg, sl_pg, ncut_pg, lo_pg, hi_pg = per_group_cut(pg, sel, grp, 1)
    # the same items under ONE group: the two rules must then COINCIDE
    pk_one, sl_one, _, lo_one, _ = per_group_cut(pg, sel, np.zeros(6, int), 1)
    # a fully-satisfied K=0 group must be SKIPPED, not averaged in
    pg3 = np.concatenate([pg, [0.999, 0.999, 0.999]])
    pk_k0, sl_k0, ncut_k0, _, _ = per_group_cut(
        pg3, np.concatenate([sel, [0, 0, 0]]),
        np.concatenate([grp, [2, 2, 2]]), 1)
    jensen = pk_pg * (1 - pk_pg)                # slope OF the mean cut
    # THE ONE ORDERING WITH A PROOF: the global top-K is the selection that
    # maximises the minimum selected probability, so no per-group cut set can
    # have a higher minimum. A SECOND fixture, budgets tracking difficulty,
    # where the MEAN goes the other way -- the case the e2e run caught.
    pg2 = np.array([0.99, 0.97, 0.40, 0.95, 0.93, 0.35])
    sel2 = np.array([1, 1, 0, 1, 1, 0])         # k_0 = 2, k_1 = 2, K = 4
    g2cut = float(np.sort(pg2)[::-1][3])
    pk2, _, _, lo2, _ = per_group_cut(pg2, sel2, grp, 1)

    print("\n  per-group cut %.4f (slope %.6f, %d groups, min %.4f) vs GLOBAL "
          "rank-K %.4f (slope %.6f)" % (pk_pg, sl_pg, ncut_pg, lo_pg, gcut,
                                        gcut * (1 - gcut)), file=out)

    checks = [
        ("the per-group MINIMUM cut never exceeds the global rank-K one",
         lo_pg <= gcut + 1e-12 and lo2 <= g2cut + 1e-12),
        ("...and the MEAN has no such ordering: fixture 2 puts it ABOVE",
         pk2 > g2cut + 1e-9),
        ("the two rules disagree at all -- fixture 1 separates them",
         abs(pk_pg - gcut) > 1e-9),
        ("a deeper cut carries MORE per-item gradient, not less",
         sl_pg > gcut * (1 - gcut) + 1e-9),
        ("the slope is the MEAN OF SLOPES, not the slope of the mean (Jensen)",
         abs(sl_pg - jensen) > 1e-6 and sl_pg < jensen),
        ("slope_max reports the DEEPEST group's cut, not the average one",
         hi_pg >= sl_pg - 1e-12 and abs(hi_pg - 0.30 * 0.70) < 1e-9),
        ("NEGATIVE CONTROL: with ONE group the two rules coincide exactly",
         abs(pk_one - gcut) < 1e-12 and abs(lo_one - gcut) < 1e-12
         and abs(sl_one - gcut * (1 - gcut)) < 1e-12),
        ("NEGATIVE CONTROL: a K=0 group is skipped, not averaged in",
         abs(pk_k0 - pk_pg) < 1e-12 and ncut_k0 == ncut_pg),
        ("NEGATIVE CONTROL: no Group_ID yields no per-group cut, not a guess",
         per_group_cut(pg, sel, None, 1) == (None, None, None, None, None)),
        ("a tighter cap gives a strictly larger gap",
         seen["tight"][0] > seen["loose"][0]),
        ("a tighter cap puts the cut nearer p=1",
         seen["tight"][1] > seen["loose"][1]),
        ("the tight cut carries less per-item gradient than the loose one",
         seen["tight"][2] < seen["loose"][2]),
        ("a PERFECTLY calibrated model reports gap EXACTLY zero",
         seen["exact"][0] == 0),
        ("the flag fires on the tight cut and not the loose one",
         seen["tight"][2] < DEAD_SLOPE <= seen["loose"][2]),
    ]
    print(file=out)
    for label, good in checks:
        print("  %-4s %s" % ("OK" if good else "FAIL", label), file=out)
        ok = ok and good
    print("\nSELF-TEST %s" % ("PASSED" if ok else "FAILED"), file=out)
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", nargs="*")
    ap.add_argument("--arm", default="tralo_null",
                    help="the lambda=0 twin; the geometry belongs to the cap")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.roots:
        ap.error("give one or more campaign roots, or --self-test")

    frames = [measure(r, args.arm) for r in args.roots]
    frames = [f for f in frames if len(f)]
    if not frames:
        print("no completed `%s` runs with both prediction files under: %s"
              % (args.arm, ", ".join(args.roots)))
        return 1
    report(summarise(pd.concat(frames, ignore_index=True)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
