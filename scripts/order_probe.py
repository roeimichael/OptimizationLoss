"""Did the constraint REORDER the capped class, or only move the cut?

WHY THIS EXISTS. The allocator is a top-K over the capped class's scores, so it
reads exactly one thing: the ORDER of the test items. Two models with the same
order produce the same predictions at every budget, no matter how different
their probabilities are.

Every count penalty this project has shipped has the form

    L_pen = f( sum_i p_ic )

so its gradient on item i's logit is  f'(S) * p_ic(1 - p_ic)  -- a function of
`p_ic` ALONE. On the direct (logit) channel that is a monotone map on p, and a
monotone map cannot change the order. It moves the cut; it cannot move an item
across one. The only way such a penalty reorders is through the SHARED WEIGHTS,
i.e. the representation channel -- which `iwc1`/`iwc2` measured and found
NEGATIVE (AP -0.031 / -0.094 against the arm's own lambda=0 twin).

If that is right, then no tuning of a `sum_i f(p_ic)` penalty can win, and the
fix has to be a per-item weight that is NOT a function of `p_ic` alone. The
margin `m_ic = p_ic - max_{c'!=c} p_ic'` is the cheapest such quantity: it reads
the whole row, so two items with equal `p_ic` but different runner-ups get
different gradients.

WHAT IS MEASURED, per (model, cap, seed, capped class):

  rho(arm, null)      Spearman of the capped-class score, arm vs its OWN
                      lambda=0 twin -- same warm-up, same allocator, same seed.
                      ~1.0 means "did not reorder".
  rho(reseed, null)   THE CONTROL, and the whole point. `tralo_reseed` is that
                      same null with one extra draw from the RNG and nothing
                      else, so it is how much the order moves for FREE. A
                      constraint that reorders LESS than a reseed has not
                      reordered.
  topK Jaccard        set overlap at the arm's own budget, which is what the
                      allocator actually consumes.
  rho on the BAND     the same, restricted to the contested items (ranks
                      K/2 .. 2K). Global rho is dominated by the easy mass and
                      will read ~1.0 for any two models at all; the band is
                      where the cut lives.

A NULL HERE IS A MEASUREMENT, not silence, because the reseed arm bounds what
"moved" means. Read the SIGN of (arm - reseed), never rho alone.

    python -m scripts.order_probe --campaign results/iwc2
"""
import argparse
import glob
import json
import math
import os
import io
import sys

import numpy as np
import pandas as pd
# !! IMPORT THE SYMBOL, NOT THE MODULE. This file defines its own
# `capped_classes(run_dir)` at module level, which REBINDS the name and
# made `capped_classes.assert_single_dataset` an AttributeError on every
# `--campaign` invocation from 2026-09-09 (`2dd84549`) until 2026-09-10.
# It parsed, imported and passed every AST gate -- the same shape as the
# three scorers that shipped `quarantine.` with no import. Gated by
# `test_no_module_import_is_shadowed_by_a_local_definition`.
from scripts.capped_classes import assert_single_dataset  # noqa: E402


from scripts import quarantine
from scripts.family_split import null_of

RAW = "final_predictions_raw.csv"
DEPLOYED = "final_predictions.csv"


def spearman(a, b):
    """Rank correlation without scipy (the env has it, but this file is a
    read-only probe and must run anywhere the artefacts do)."""
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def sign_test(k, n):
    """Two-sided exact binomial p for k successes in n, under p=0.5.

    WHY THIS EXISTS. The verdict below used to branch on the bare POOLED MEAN
    of (rho_arm - rho_reseed), with no test at all, so a mean of -0.0076 on a
    27/48 split -- a coin flip -- printed "the constraint reordered MORE than a
    reseed. The order-preservation argument does NOT hold here." Measured on
    `results/loose1` 2026-08-28, that fired for `tralo` (27/48, p=0.47) AND for
    `tralo_uniform` (26/48, p=0.66).

    `tralo_uniform` is this probe's built-in NEGATIVE CONTROL: its per-item
    gradient is constant in log-odds, so on the direct channel it is a pure
    bias shift and CANNOT reorder (configs/protocol.yml says so at its
    definition). A verdict that calls it "reordered MORE" is a verdict that
    fails its own control, which is exactly the failure this project keeps
    finding elsewhere -- a tie read as an effect because nothing gated it.
    """
    n, k = int(n), int(k)
    if n <= 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1))
    return min(1.0, 2.0 * tail / float(2 ** n))


def _points_needed(dd, power_const=7.85):
    """Points at 80% power to call the observed mean against its own spread.

    Same constant `scripts/paired_noise.py` prices seeds with, so a tie here
    reports which of the two things it is -- no effect, or not enough points.
    """
    e, s = abs(float(dd.mean())), float(dd.std(ddof=1))
    if not e or e != e or s != s:
        return -1
    return int(math.ceil(power_const * (s / e) ** 2))


def paired_sd_items(arm, ctrl):
    """(sd, n_cells) of (arm - ctrl) net items, pooled the way paired_noise is.

    RMS of the WITHIN-cell sds, never one sd over the flattened set: a
    cell-to-cell mean shift is a real difference between cells, not noise, and
    pooling it in inflates the sd and hides an effect.
    """
    key = ["model", "cap", "seed", "cls"]
    if arm.empty or ctrl.empty:
        return float("nan"), 0
    m = arm.merge(ctrl, on=key, suffixes=("_a", "_b"))
    if m.empty:
        return float("nan"), 0
    m["d"] = m.net_items_a - m.net_items_b
    sds = m.groupby(["model", "cap", "cls"]).d.std(ddof=1).dropna()
    if not len(sds):
        return float("nan"), 0
    return float(np.sqrt((sds ** 2).mean())), int(len(sds))


def load(run_dir):
    f = os.path.join(run_dir, RAW)
    if not os.path.exists(f):
        return None
    return pd.read_csv(f)


def capped_classes(run_dir):
    try:
        c = json.load(open(os.path.join(run_dir, "config.json"),
                           encoding="utf-8"))
    except (OSError, ValueError):
        return []
    dc = c.get("dataset_config") or {}
    cc = dc.get("constrained_class")
    if cc is None:
        cc = c.get("constrained_classes")
    if isinstance(cc, int):
        cc = [cc]
    return list(cc or [])


def budget_for(df, cls):
    """How many items the arm actually emitted for this class -- the real K the
    allocator used, read off the arm's own output rather than recomputed."""
    return int((df["Predicted_Label"] == cls).sum())


def load_deployed(run_dir):
    """The AS-DEPLOYED predictions -- what the allocator actually emitted.

    `load` reads `final_predictions_raw.csv`, which carries the model's
    argmax (`raw_pred` in `src/pipeline/eval.py:106`). That is the right
    source for PROBABILITIES -- both files carry the same `y_proba` -- and the
    WRONG source for a BUDGET: `budget_for` on the raw frame returns the HARD
    COUNT, not K. On iwildcam at L20 the lambda=0 hard count is ~336 against a
    deployed K of ~74, so a band of `K//2 .. 2K` taken off the raw frame spans
    ranks 168-672 and does not even CONTAIN the cut. FRAMEWORK 2(z80).
    """
    f = os.path.join(run_dir, DEPLOYED)
    if not os.path.exists(f):
        return None
    return pd.read_csv(f)


def group_budgets(dep, cls):
    """(K, {group: k_g}) for one class, read off the DEPLOYED file.

    The allocator emits top-`k_g` WITHIN each group, so `k_g` is the only cut
    that exists. `K` is their sum and is kept for the global reading only.
    """
    sel = dep["Predicted_Label"] == cls
    if "Group_ID" not in dep.columns:
        raise SystemExit(
            "order_probe: `%s` has no Group_ID column, so the per-group cut "
            "cannot be found. REFUSING rather than falling back to a global "
            "sort -- that fallback is FRAMEWORK 2(z80), the eighth site of "
            "the same substitution. Re-run the campaign, or score a campaign "
            "whose runs wrote groups." % DEPLOYED)
    by = dep.loc[sel, "Group_ID"].value_counts()
    return int(sel.sum()), {g: int(n) for g, n in by.items()}


def band_per_group(p, groups, k_by_group, lo=0.5, hi=2.0):
    """Ranks `lo*k_g .. hi*k_g` WITHIN each group, unioned.

    The global form -- `argsort(-p)[K//2:2K]` over the whole test set -- is
    what this replaces. A globally-ranked item in that window may sit safely
    inside a large-`k_g` group or in one of the 7 of 14 iwildcam groups whose
    ceiling is K=0 and which can never select it.
    """
    idx = []
    for g in np.unique(groups):
        where = np.flatnonzero(groups == g)
        kg = int(k_by_group.get(g, k_by_group.get(str(g), 0)))
        if kg < 1:
            continue                      # a K=0 ceiling has no cut to be near
        order = where[np.argsort(-p[where])]
        a, b = max(1, int(lo * kg)), min(len(order), int(hi * kg))
        if b > a:
            idx.extend(order[a:b].tolist())
    return np.array(sorted(set(idx)), dtype=int)


def deployed_set(dep, cls):
    """The exact item set the allocator emitted for this class.

    No `argsort` and no re-derivation: `jac` used to compare two GLOBAL top-K
    sets rebuilt from the probabilities, which is neither arm's selection.
    Both arms emit exactly K items -- K comes from the labels and the cap
    policy, not from the model -- so the two sets are equal-size by
    construction and the Jaccard is still the right statistic.
    """
    return set(np.flatnonzero((dep["Predicted_Label"] == cls).to_numpy())
               .tolist())


def self_test(out=sys.stdout):
    """The gate. This probe produced the sharpest NEGATIVE in the project --
    "the constraint re-ranks exactly as much as a coin flip" -- so every rung
    of its ladder has to be reachable, or the negative is an artefact of a
    verdict function that can only say one thing.

    Four rungs, on synthetic differences: nothing to test, coin flip, reorders
    MORE than a reseed, reorders LESS. Pure arithmetic, no campaign.
    """
    rng = np.random.default_rng(0)
    n = 48
    cases = [
        ("zero points", np.zeros(n), "NOTHING TO TEST", ("TIE", "monotone map")),
        ("coin flip", rng.normal(size=n) * 1e-4, "TIE", ("NOTHING TO TEST",)),
        ("reordered MORE", np.array([-1.0] * 40 + [1.0] * 8),
         "reordered MORE than a reseed, and it CLEARS",
         ("TIE", "NOTHING TO TEST")),
        ("preserved order", np.array([1.0] * 40 + [-1.0] * 8),
         "rho_arm >= rho_reseed, and it CLEARS",
         ("TIE", "NOTHING TO TEST")),
    ]
    ok = True
    w = out.write
    w("SELF-TEST -- can the ladder reach all four rungs?" + chr(10) + chr(10))
    for name, dd, want, forbid in cases:
        buf = io.StringIO()
        verdict(dd, dd, out=buf)
        txt = buf.getvalue()
        bad = [f for f in forbid if f in txt]
        if want not in txt or bad:
            w("  FAIL  %-14s expected %r, forbade %r, got:%s%s"
              % (name, want, bad or list(forbid), chr(10), txt))
            ok = False
        else:
            w("  PASS  %-14s -> %s" % (name, want) + chr(10))

    # --- THE CUT IS PER GROUP (FRAMEWORK 2(z80)) ---------------------------
    def chk(cond, msg):
        w(("  PASS  " if cond else "  FAIL  ") + msg + chr(10))
        return cond

    # Two groups of 100. Group A is EASY (high scores), group B is hard. The
    # deployed budget is 10 in each, so the two cuts sit at within-group rank
    # 10 -- global rank 10 and global rank ~110. A global band centred on
    # K=20 can only ever see the first of them.
    pg = np.concatenate([np.linspace(0.99, 0.60, 100),
                         np.linspace(0.40, 0.01, 100)])
    gg = np.array(["A"] * 100 + ["B"] * 100)
    kg = {"A": 10, "B": 10}
    b_pg = band_per_group(pg, gg, kg)
    b_gl = np.argsort(-pg)[max(1, 20 // 2):min(len(pg), 2 * 20)]
    ok &= chk(len(set(b_pg.tolist()) & set(range(100, 200))) > 0,
              "the per-group band reaches the HARD group's cut")
    ok &= chk(len(set(b_gl.tolist()) & set(range(100, 200))) == 0,
              "NEGATIVE CONTROL: the global band NEVER reaches it "
              "(%d of its %d items are in group B)"
              % (len(set(b_gl.tolist()) & set(range(100, 200))), len(b_gl)))

    # NEGATIVE CONTROL: a K=0 ceiling has no cut, so it contributes nothing.
    b_zero = band_per_group(pg, gg, {"A": 10, "B": 0})
    ok &= chk(not (set(b_zero.tolist()) & set(range(100, 200))),
              "NEGATIVE CONTROL: a group with k_g=0 contributes NO band items")

    # NEGATIVE CONTROL: with ONE group holding the whole budget the fix is a
    # NO-OP. If it changed the answer there it would be a different statistic,
    # not a corrected one.
    one = np.array(["A"] * 200)
    ok &= chk(sorted(band_per_group(pg, one, {"A": 20}).tolist())
              == sorted(np.argsort(-pg)[10:40].tolist()),
              "NEGATIVE CONTROL: one group, k_g=K -> per-group band EQUALS "
              "the global band")

    dep = pd.DataFrame({"Predicted_Label": [2] * 20 + [0] * 180,
                        "Group_ID": gg})
    K_d, kg_d = group_budgets(dep, 2)
    ok &= chk(K_d == 20 and sum(kg_d.values()) == K_d,
              "group_budgets: the per-group budgets sum to K (%d)" % K_d)
    ok &= chk(deployed_set(dep, 2) == set(range(20)),
              "deployed_set is the allocator's OWN selection, not a re-sort")

    # NEGATIVE CONTROL: no Group_ID must REFUSE, never fall back -- and it
    # must refuse ON PURPOSE. A KeyError five frames down reads as a bug in
    # the caller, which is the trap `pred_integrity` exists for, so the check
    # pins the EXCEPTION TYPE and not merely that something went wrong.
    try:
        group_budgets(dep.drop(columns=["Group_ID"]), 2)
        got = "returned a value"
    except SystemExit:
        got = "SystemExit"
    except Exception as exc:
        got = type(exc).__name__
    ok &= chk(got == "SystemExit",
              "NEGATIVE CONTROL: no Group_ID REFUSES deliberately (got %s)"
              % got)

    # `sign_test` is what separates rung 2 from rungs 3-4, so pin its two ends.
    if not (sign_test(24, 48) > 0.9 and sign_test(40, 48) < 0.001):
        w("  FAIL  sign_test: 24/48 must be a coin (%.3f) and 40/48 must clear "
          "(%.5f)" % (sign_test(24, 48), sign_test(40, 48)) + chr(10))
        ok = False
    else:
        w("  PASS  sign_test 24/48 p=%.3f, 40/48 p=%.5f"
          % (sign_test(24, 48), sign_test(40, 48)) + chr(10))
    w("SELF-TEST %s%s" % ("PASSED" if ok else "FAILED", chr(10)))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--campaign")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--arm", default="tralo")
    # NOT a fixed "tralo_null". That is correct for `tralo`, `tralo_uniform`
    # and `tralo_head` -- which share one twin -- and quietly WRONG for
    # `--arm fioretto` on a cross-family campaign, where the twin actually run
    # is `fioretto_null`. Resolved from the campaign the same way
    # `family_split` does: dedicated null if this campaign ran one, shared
    # `null_sibling` otherwise. An explicit --null always wins.
    ap.add_argument("--null", default=None,
                    help="lambda=0 twin; default resolves from --arm")
    ap.add_argument("--reseed", default="tralo_reseed")
    ap.add_argument("--out", default=None,
                    help="write the per-(run, class) rows to a CSV. "
                         "Both the per-group and the global band "
                         "columns are dumped, so a published figure "
                         "can be traced to the reading it came from.")
    ap.add_argument("--evictions", action="store_true",
                    help="which items did it move, and were they the right ones")
    ap.add_argument("--allow-quarantined", action="store_true",
                    help="probe a campaign `scripts.quarantine` marked dead")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    if not args.campaign:
        ap.error("--campaign is required, or --self-test")

    # 🛑 THE QUARANTINE GATE. Audited 2026-09-04: this tool had NONE, so a
    # marker on a dead campaign prevented nothing here -- and this is the tool
    # that produced "the constraint evicts the CORRECT items", a headline
    # number. No fallback import -- if the gate cannot load, the tool must
    # break.
    from scripts.quarantine import gate
    blocked, dead = gate([args.campaign], args.allow_quarantined, "probe")
    if blocked:
        return 1

    # THIS TOOL KEYS ITS CELLS (model, cap, class) WITH NO `dataset`,
    # so it is correct only on a single-dataset root -- and nothing
    # checked. `gen_campaign --datasets` is `nargs="+"`. Harmless while
    # iwildcam was the only runnable dataset; bcn and fmow ended that.
    # FRAMEWORK 2(z65).
    assert_single_dataset(args.campaign, "order_probe")
    # A PARTIAL marker names arms whose contrasts are disqualified. Every read
    # below is `--arm` against `--null` and against `--reseed`, so the
    # enforcement is a FILTER on the enumerated arm directories:
    # `drop_dead_runs` resolves the arm off each path, which is exactly the
    # granularity these globs work at. PER CAMPAIGN, never a union.
    here = dead.for_path(args.campaign) if hasattr(dead, "for_path") else dead
    if args.null is None:
        present = {os.path.basename(os.path.dirname(d))
                   for d in glob.glob(os.path.join(args.campaign,
                                                   "*", "*", "*", "*", "seed_*"))}
        args.null = null_of(args.arm, present)
        print("  --null not given; resolved %s -> %s" % (args.arm, args.null))

    if args.evictions:
        e = evictions(args.campaign, args.arm, args.null, args.reseed, here)
        # THE CONTROL, and without it the arm's number means nothing. EVICTED
        # items were in the twin's top-K by construction, so they necessarily
        # have higher p and are necessarily more often true positives than the
        # ADMITTED ones. ANY perturbation scores negative on this statistic.
        # `tralo_reseed` is the twin with one extra RNG draw and no constraint,
        # so it is what a perturbation of no consequence costs. Read the
        # DIFFERENCE.
        c = evictions(args.campaign, args.reseed, args.null, args.reseed, here)
        if e.empty:
            print("no moved items -- the arm and its twin agree at every budget")
            return 0
        print("=" * 84)
        print("WHICH ITEMS DID THE CONSTRAINT MOVE, AND WERE THEY THE RIGHT ONES?")
        print("=" * 84)
        print("  At the arm's own budget K, against its lambda=0 twin's top-K:")
        print("    EVICTED  = twin had it, arm dropped it")
        print("    ADMITTED = arm added it, twin did not have it")
        print("  A method that helps evicts FALSE positives and admits TRUE ones,")
        print("  so `admitted_tp` must exceed `evicted_tp`. net_items is the")
        print("  swap's yield in the only unit that counts.")
        print()
        g = e.groupby(["model", "cap", "cls"])[
            ["K", "moved", "evicted_tp", "admitted_tp", "net_items"]].mean()
        print(g.round(4).to_string())
        print()
        print("-" * 84)
        ev, ad = e["evicted_tp"].mean(), e["admitted_tp"].mean()
        print("POOLED over %d (cell, class, seed) points" % len(e))
        print("   items moved per cell        %.1f of K=%.0f" % (e["moved"].mean(), e["K"].mean()))
        print("   precision of what it EVICTED  %.4f" % ev)
        print("   precision of what it ADMITTED %.4f" % ad)
        print("   NET items                     %+.2f per cell (%d/%d cells negative)"
              % (e["net_items"].mean(), int((e["net_items"] < 0).sum()), len(e)))
        print()
        print("-" * 84)
        print("THE CONTROL: the same statistic for `%s`, which is the twin with" % args.reseed)
        print("one extra RNG draw and NO constraint. Evicted items outrank admitted ones")
        print("by construction, so any perturbation scores negative here. Only the")
        print("DIFFERENCE is attributable to the constraint.")
        print()
        if c.empty:
            print("   *** NO CONTROL AVAILABLE -- the arm's number above is NOT")
            print("       interpretable on its own. Do not quote it.")
            return 1
        cev, cad = c["evicted_tp"].mean(), c["admitted_tp"].mean()
        print("   %-26s %10s %10s" % ("", args.arm, args.reseed))
        print("   %-26s %10.1f %10.1f" % ("items moved per cell", e["moved"].mean(), c["moved"].mean()))
        print("   %-26s %10.4f %10.4f" % ("precision EVICTED", ev, cev))
        print("   %-26s %10.4f %10.4f" % ("precision ADMITTED", ad, cad))
        print("   %-26s %+10.2f %+10.2f" % ("NET items per cell", e["net_items"].mean(), c["net_items"].mean()))
        print()
        print("   WHERE THE CUT SITS (from the twin's own scores)")
        print("     p at each group's cut     %.4f   below 0.5 in %d/%d cells"
              % (e["p_cut"].mean(), int(e["cut_below_half"].sum()), len(e)))
        print("     deepest group's cut       %.4f" % e["p_cut_min"].mean())
        print("     p at the GLOBAL K-th      %.4f   <- the pre-2026-09-10"
              % e["p_cut_glob"].mean())
        print("                                          reading, retained so")
        print("                                          old figures reproduce.")
        print("                                          The gap is NOT a")
        print("                                          direction (2(z64)).")
        print("     deployed K vs HARD count  %.1f vs %.1f  <- the old K was the"
              % (e["K"].mean(), e["K_raw"].mean()))
        print("                                          HARD one, off the RAW")
        print("                                          frame (2(z84)).")
        print("     mean p of EVICTED items   %.4f" % e["p_evicted"].mean())
        print("     mean p of ADMITTED items  %.4f" % e["p_admitted"].mean())
        print()
        d_net = e["net_items"].mean() - c["net_items"].mean()
        d_gap = (ev - ad) - (cev - cad)
        print("   ATTRIBUTABLE TO THE CONSTRAINT")
        print("     net items      %+.2f per cell   (arm %+.2f minus control %+.2f)"
              % (d_net, e["net_items"].mean(), c["net_items"].mean()))
        print("     precision gap  %+.1f pp" % (100 * d_gap))
        print()
        sd_i, n_cells = paired_sd_items(e, c)
        need = (int(math.ceil(7.85 * (sd_i / abs(d_net)) ** 2))
                if d_net and sd_i == sd_i and sd_i else -1)
        print("   RESOLUTION -- can this contrast see what it is reporting?")
        print("     paired seed sd %6.2f items  (within cell, pooled over %d "
              "cell(s))" % (sd_i, n_cells))
        if need > 0:
            print("     to detect %+.2f items at 80%% power needs ~%d seeds "
                  "per cell: %s" % (d_net, need,
                                    "OK" if need <= 4 else "UNDERPOWERED"))
        print()
        print("   !! THIS IS A GLOBAL TOP-K, NOT THE ALLOCATOR THAT RAN.")
        print("      The sets above are argsort(-p)[:K] on the raw class "
              "column. The deployed")
        print("      allocator is LP/greedy under PER-GROUP ceilings, and on "
              "iwildcam 7 of 14")
        print("      local ceilings are K=0 -- so it cannot take the global "
              "top-K and does not.")
        print("      MEASURED on results/loose1 2026-08-28: this block said "
              "+16.50 items")
        print("      attributable where `full_panel --control tralo_null` "
              "said tralo +9.24 and")
        print("      tralo_reseed +6.71, i.e. +2.53 attributable -- this "
              "number was 6.5x too")
        print("      large. Use it to read WHICH items moved and WHY. Quote "
              "`full_panel` for")
        print("      HOW MANY.")
        print()
        if d_net < -1.0:
            print("   => the constraint's swaps are WORSE than a pointless reseed's by")
            print("      %.1f items per cell. It is not merely failing to help; it is" % abs(d_net))
            print("      selecting which items to drop, and selecting badly.")
        elif d_net > 1.0:
            print("   => the constraint's swaps are BETTER than a reseed's by %.1f items." % d_net)
        else:
            print("   => indistinguishable from a reseed (%+.2f items). The eviction cost is" % d_net)
            print("      GEOMETRY, not method: any perturbation of this size pays it, so the")
            print("      lever is the SIZE of the perturbation, not its direction.")
        return 0

    rows, skipped_nodep = [], 0
    pat = os.path.join(args.campaign, "*", "*", "*", args.arm, "seed_*")
    for arm_dir in quarantine.drop_dead_runs(sorted(glob.glob(pat)), here,
                                             label="arm run"):
        parts = arm_dir.split(os.sep)
        model, dataset, cap, seed = parts[-5], parts[-4], parts[-3], parts[-1]
        base = os.sep.join(parts[:-2])
        a = load(arm_dir)
        n = load(os.path.join(base, args.null, seed))
        r = load(os.path.join(base, args.reseed, seed))
        # The BUDGET comes from the deployed file, the PROBABILITIES from the
        # raw one. Reading both off the raw frame is what centred the band on
        # the hard count -- FRAMEWORK 2(z80).
        a_dep = load_deployed(arm_dir)
        n_dep = load_deployed(os.path.join(base, args.null, seed))
        r_dep = load_deployed(os.path.join(base, args.reseed, seed))
        if a is None or n is None:
            continue
        for cls in capped_classes(arm_dir):
            col = "Prob_Class_%d" % cls
            if col not in a.columns or col not in n.columns:
                continue
            if a_dep is None or n_dep is None:
                skipped_nodep += 1
                continue
            pa, pn = a[col].to_numpy(), n[col].to_numpy()
            K_raw = budget_for(a, cls)      # the HARD count -- the old centre
            K, kg = group_budgets(n_dep, cls)   # the DEPLOYED budget, per group
            if K < 3:
                continue
            groups = n["Group_ID"].to_numpy()

            # THE CUT IS PER GROUP. `band` is that; `band_glob` is the reading
            # every figure before 2026-09-10 was taken at, kept beside it so no
            # published number silently changes meaning. FRAMEWORK 2(z80).
            band = band_per_group(pn, groups, kg)
            order_n = np.argsort(-pn)
            band_glob = order_n[max(1, K_raw // 2):min(len(pn), 2 * K_raw)]

            # The Jaccard is now the two AS-DEPLOYED sets -- no re-derivation.
            sel_a, sel_n = deployed_set(a_dep, cls), deployed_set(n_dep, cls)
            ta_g = set(np.argsort(-pa)[:K_raw].tolist())
            tn_g = set(np.argsort(-pn)[:K_raw].tolist())

            row = {"model": model, "cap": cap, "seed": seed, "cls": cls,
                   "K": K, "K_raw": K_raw,
                   "n_band": len(band), "n_band_glob": len(band_glob),
                   "rho_arm": spearman(pa, pn),
                   "rho_arm_band": spearman(pa[band], pn[band]),
                   "rho_arm_band_glob": spearman(pa[band_glob], pn[band_glob]),
                   "jac_arm": len(sel_a & sel_n) / max(1, len(sel_a | sel_n)),
                   "jac_arm_glob": len(ta_g & tn_g) / max(1, len(ta_g | tn_g))}
            if r is not None and col in r.columns and r_dep is not None:
                pr = r[col].to_numpy()
                sel_r = deployed_set(r_dep, cls)
                tr_g = set(np.argsort(-pr)[:K_raw].tolist())
                row["rho_reseed"] = spearman(pr, pn)
                row["rho_reseed_band"] = spearman(pr[band], pn[band])
                row["rho_reseed_band_glob"] = spearman(pr[band_glob],
                                                       pn[band_glob])
                row["jac_reseed"] = (len(sel_r & sel_n)
                                     / max(1, len(sel_r | sel_n)))
                row["jac_reseed_glob"] = (len(tr_g & tn_g)
                                          / max(1, len(tr_g | tn_g)))
            rows.append(row)

    if not rows:
        print("no paired runs found under %s" % args.campaign)
        return 1

    d = pd.DataFrame(rows)
    if args.out:
        d.to_csv(args.out, index=False)
        print("wrote %d rows to %s" % (len(d), args.out))
    print("=" * 88)
    print("DID THE CONSTRAINT REORDER THE CAPPED CLASS?   %s" % args.campaign)
    print("  arm=%s   null=%s   control=%s" % (args.arm, args.null, args.reseed))
    print("=" * 88)
    print("  rho = Spearman of the capped-class score against the arm's OWN")
    print("  lambda=0 twin. 1.0 = identical order = the allocator cannot tell")
    print("  them apart. The reseed column is the SAME twin with one extra RNG")
    print("  draw, so it is how much the order moves for free.")
    print()
    have_ctrl = "rho_reseed" in d.columns
    cols = ["rho_arm", "rho_arm_band", "rho_arm_band_glob", "jac_arm",
            "jac_arm_glob"]
    if have_ctrl:
        cols += ["rho_reseed", "rho_reseed_band", "rho_reseed_band_glob",
                 "jac_reseed", "jac_reseed_glob"]
    if skipped_nodep:
        print("   !! %d (run, class) points had no `%s` and were SKIPPED."
              % (skipped_nodep, DEPLOYED))
    print("   !! THE `_band` COLUMNS ARE THE PER-GROUP CUT (fixed 2026-09-10).")
    print("      `_band_glob` is the old global reading, kept beside them so a")
    print("      published figure can be traced. The allocator emits top-k_g")
    print("      WITHIN each group; on iwildcam 7 of 14 local ceilings are")
    print("      K=0. The old band was also centred on the HARD count from")
    print("      `%s`, not on K: at L20 that is ~336 against a" % RAW)
    print("      deployed K of ~74, so it did not contain the cut at all.")
    print("      DIRECTION UNKNOWN -- do not read the pair as one. 2(z80).")
    print("      band items: per-group %.0f, global %.0f (mean)"
          % (d["n_band"].mean(), d["n_band_glob"].mean()))
    print()
    g = d.groupby(["model", "cap", "cls"])[cols].mean()
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(g.round(4).to_string())
    print()
    print("-" * 88)
    print("POOLED over %d (cell, class) points, %d seeds each"
          % (len(g), d.groupby(["model", "cap", "cls"]).size().max()))
    for c in cols:
        print("   %-18s %.4f" % (c, d[c].mean()))
    if have_ctrl:
        print()
        d2 = d.dropna(subset=["rho_arm", "rho_reseed"])
        verdict(d2["rho_arm"] - d2["rho_reseed"],
                (d2["rho_arm_band"] - d2["rho_reseed_band"]).dropna())
    return 0


def verdict(dd, db, out=None, alpha=0.05):
    """Print the reorder verdict, GATED on an exact sign test.

    Split out of `main` so the gate can run on a synthetic split with no
    campaign: `tests/` drives it with a 27/48 coin flip and asserts it refuses
    to call a direction. See `sign_test` for what it used to do instead.
    """
    out = out or sys.stdout
    n_g, k_g = int((dd != 0).sum()), int((dd < 0).sum())
    n_b, k_b = int((db != 0).sum()), int((db < 0).sum())
    p_g, p_b = sign_test(k_g, n_g), sign_test(k_b, n_b)
    w = out.write
    w("VERDICT -- the SIGN, and whether the sign is DISTINGUISHABLE from a "
      "coin\n")
    w("   rho_arm - rho_reseed        %+.4f   (%d/%d points reordered MORE "
      "than a reseed, sign p=%.3f)\n" % (dd.mean(), k_g, n_g, p_g))
    w("   band                        %+.4f   (%d/%d, sign p=%.3f)\n"
      % (db.mean() if n_b else float("nan"), k_b, n_b, p_b))
    w("\n")
    if not n_g:
        # NOT a tie. `n_g` counts points where the arm and its reseed differ
        # AT ALL, so zero of them means nothing was compared -- an inert arm
        # whose predictions are byte-identical, or an empty input. Printing
        # TIE here also printed the monotone-map paragraph below it, which
        # reads as a CONFIRMED MECHANISM for a run that produced no data.
        w("   => NOTHING TO TEST. Not one point differs between the arm and "
          "its reseed,\n")
        w("      so there is no sign to test and no effect size to bound. "
          "That is a\n")
        w("      statement about the INPUT, not about reordering: either the "
          "arm is inert\n")
        w("      (md5 its `final_predictions_raw.csv` against the twin -- "
          "CLAUDE.md rule 3,\n")
        w("      five occurrences) or the glob matched nothing usable.\n")
        return
    if p_g >= alpha:
        w("   => TIE. The constraint is INDISTINGUISHABLE from a pure RNG "
          "reseed (p=%.3f at\n" % p_g)
        w("      %d points; %d/%d is a coin flip). What order movement there "
          "is, is SEED.\n" % (n_g, k_g, n_g))
        w("      This is the EXPECTED reading for any `sum_i f(p_ic)` "
          "penalty: its per-item\n")
        w("      gradient is a function of `p_ic` ALONE, hence a monotone map "
          "on the logit\n")
        w("      channel, and a monotone map cannot move an item across "
          "another. It moves the\n")
        w("      CUT, not the RANKING.\n")
        w("      A tie is 'no effect' OR 'not enough points' -- to call the "
          "observed %+.4f at\n" % dd.mean())
        w("      80%% power would need ~%d points, against the %d here.\n"
          % (_points_needed(dd), n_g))
    elif dd.mean() >= 0:
        w("   => rho_arm >= rho_reseed, and it CLEARS the coin (p=%.3f): the "
          "constraint\n" % p_g)
        w("      preserved the order at least as well as doing NOTHING and "
          "reseeding. It moved\n")
        w("      the cut, not the ranking -- which is what `sum_i f(p_ic)` is "
          "mathematically able\n")
        w("      to do, and all it is able to do.\n")
    else:
        w("   => the constraint reordered MORE than a reseed, and it CLEARS "
          "the coin (p=%.3f).\n" % p_g)
        w("      The order-preservation argument does NOT hold here; the "
          "representation\n")
        w("      channel is doing something. Say how much.\n")
    # THE BAND IS THE INFORMATIVE STATISTIC AND IT IS NOT WHAT THE VERDICT
    # ABOVE BRANCHES ON. Global rho is taken over the whole capped class, which
    # is dominated by the easy mass and reads ~1.0 for any two models; the band
    # is ranks K/2..2K, where the cut actually falls. So the case that matters
    # -- global TIE, band CLEARS -- would otherwise be printed and missed.
    # Measured on `results/uniform1` (tight caps) 2026-08-28: global -0.0100 at
    # 41/72 p=0.289, band -0.0454 at 45/72 p=0.044.
    if n_b and p_b < alpha and (p_g >= alpha or not n_g):
        w("\n")
        w("   !! BUT THE BAND CLEARS WHILE THE GLOBAL TIES (band p=%.3f vs "
          "global p=%.3f).\n" % (p_b, p_g))
        w("      Global rho is diluted by the easy mass; the band is ranks "
          "K/2..2K, where the\n")
        w("      cut falls. On the items the allocator actually contests this "
          "arm reordered\n")
        w("      %s a reseed. That is the representation channel, and it is "
          "the\n" % ("MORE than" if db.mean() < 0 else "LESS than"))
        w("      one channel a `sum_i f(p_ic)` penalty CAN reach.\n")
        w("      One test among several: quote it with its multiplicity, not "
          "as a standalone p.\n")
    return p_g



# ---------------------------------------------------------------------------
# WHICH items did it move? -- run with --evictions
#
# The band result says the constraint DOES reorder where the cut falls, and
# more than a reseed does (16/16 on iwc2). So "it cannot reorder" is refuted.
# The remaining question is whether the items it moves are the right ones.
#
# At the arm's own budget K, against its lambda=0 twin's top-K:
#   EVICTED  = in the twin's top-K, not in the arm's   (the constraint pushed out)
#   ADMITTED = in the arm's top-K, not in the twin's   (the constraint pulled in)
# A method that helps evicts false positives and admits true ones. Precision on
# those two sets is the whole story, and it is denominated in ITEMS, which is
# the only unit this project trusts.
# ---------------------------------------------------------------------------
def evictions(campaign, arm_name, null_name, reseed_name, dead=()):
    """`dead` is the arm set a PARTIAL quarantine marker disqualifies.

    A PARAMETER rather than a global: the gate runs in `main` and the
    enumeration runs here, and reading it off the enclosing scope is the exact
    NameError three scorers shipped on 2026-09-04.
    """
    rows = []
    pat = os.path.join(campaign, "*", "*", "*", arm_name, "seed_*")
    for arm_dir in quarantine.drop_dead_runs(sorted(glob.glob(pat)), dead,
                                             label="eviction run"):
        parts = arm_dir.split(os.sep)
        model, cap, seed = parts[-5], parts[-3], parts[-1]
        base = os.sep.join(parts[:-2])
        a = load(arm_dir)
        n = load(os.path.join(base, null_name, seed))
        # !! SITE 1 OF THE GLOBAL-TOP-K DEFECT, DISCLOSED 2026-08-28 AND FIXED
        # 2026-09-10 (2(z84)). It carried BOTH halves of what site 8 carried:
        #   (a) `budget_for(a, cls)` on the RAW frame, which holds the model's
        #       ARGMAX, so K was the HARD COUNT and not the allocator's budget;
        #   (b) `argsort(-p)[:K]` over the whole test set, while the allocator
        #       emits top-`k_g` WITHIN each group.
        # The fix needs no reconstruction: `final_predictions.csv` IS the
        # allocator's output, so each arm's emitted set is `Predicted_Label ==
        # cls`, exactly. The RAW frame is still what supplies the
        # probabilities, because this probe is about WHERE the moved items sat
        # in the model's own ranking.
        a_dep = load_deployed(arm_dir)
        n_dep = load_deployed(os.path.join(base, null_name, seed))
        if a is None or n is None or a_dep is None or n_dep is None:
            continue
        for cls in capped_classes(arm_dir):
            col = "Prob_Class_%d" % cls
            if col not in a.columns:
                continue
            y = a["True_Label"].to_numpy()
            K_raw = budget_for(a, cls)          # the HARD count -- the old K
            K, kg = group_budgets(n_dep, cls)   # the DEPLOYED budget, per group
            if K < 3:
                continue
            pn = n[col].to_numpy()
            ta, tn = deployed_set(a_dep, cls), deployed_set(n_dep, cls)
            ev, ad = sorted(tn - ta), sorted(ta - tn)
            if not ev and not ad:
                continue
            # WHERE the cut sits decides the mechanism. d p / d logit under
            # this penalty is -eta*[p(1-p)]^2, which peaks at p=0.5 and
            # vanishes at both extremes -- so if the cut sits BELOW 0.5 the
            # items just above it move down FASTER than the items just below,
            # and the order inverts exactly at the cut. Measured, not assumed.
            pn_ev = float(np.mean(pn[ev])) if ev else np.nan
            pn_ad = float(np.mean(pn[ad])) if ad else np.nan
            # THE CUT IS PER GROUP. `p_cut` is the budget-weighted mean of each
            # group's own lowest SELECTED probability; `p_cut_glob` is the old
            # global-K-th reading, retained so the 2026-08-28 figures
            # reproduce. NEVER read the gap between them as a direction --
            # only the MINIMUM is ordered against the global value, and 2(z64)
            # is the record of asserting otherwise, fixturing it, passing
            # mutation testing and then being refuted.
            gn = n_dep["Group_ID"].to_numpy()
            cuts = []
            for g, k in kg.items():
                pg = np.sort(pn[gn == g])[::-1]
                if k >= 1 and len(pg) >= k:
                    cuts.append((k, float(pg[k - 1])))
            p_cut = (float(sum(k * v for k, v in cuts) / sum(k for k, _ in cuts))
                     if cuts else np.nan)
            p_cut_min = min((v for _k, v in cuts), default=np.nan)
            p_cut_glob = float(np.sort(pn)[::-1][K - 1])
            rows.append({
                "model": model, "cap": cap, "seed": seed, "cls": cls, "K": K,
                "K_raw": K_raw, "n_groups": len(kg),
                "p_cut": p_cut, "p_cut_min": p_cut_min,
                "p_cut_glob": p_cut_glob,
                "p_evicted": pn_ev, "p_admitted": pn_ad,
                "cut_below_half": p_cut < 0.5,
                "moved": len(ev),
                "evicted_tp": float(np.mean(y[ev] == cls)) if ev else np.nan,
                "admitted_tp": float(np.mean(y[ad] == cls)) if ad else np.nan,
                "net_items": (float(np.sum(y[ad] == cls))
                              - float(np.sum(y[ev] == cls))),
            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    raise SystemExit(main())
