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
import os

import numpy as np
import pandas as pd

RAW = "final_predictions_raw.csv"


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--arm", default="tralo")
    ap.add_argument("--null", default="tralo_null")
    ap.add_argument("--reseed", default="tralo_reseed")
    ap.add_argument("--evictions", action="store_true",
                    help="which items did it move, and were they the right ones")
    args = ap.parse_args()

    if args.evictions:
        e = evictions(args.campaign, args.arm, args.null, args.reseed)
        # THE CONTROL, and without it the arm's number means nothing. EVICTED
        # items were in the twin's top-K by construction, so they necessarily
        # have higher p and are necessarily more often true positives than the
        # ADMITTED ones. ANY perturbation scores negative on this statistic.
        # `tralo_reseed` is the twin with one extra RNG draw and no constraint,
        # so it is what a perturbation of no consequence costs. Read the
        # DIFFERENCE.
        c = evictions(args.campaign, args.reseed, args.null, args.reseed)
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
        print("     p at the K-th item        %.4f   below 0.5 in %d/%d cells"
              % (e["p_cut"].mean(), int(e["cut_below_half"].sum()), len(e)))
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

    rows = []
    pat = os.path.join(args.campaign, "*", "*", "*", args.arm, "seed_*")
    for arm_dir in sorted(glob.glob(pat)):
        parts = arm_dir.split(os.sep)
        model, dataset, cap, seed = parts[-5], parts[-4], parts[-3], parts[-1]
        base = os.sep.join(parts[:-2])
        a = load(arm_dir)
        n = load(os.path.join(base, args.null, seed))
        r = load(os.path.join(base, args.reseed, seed))
        if a is None or n is None:
            continue
        for cls in capped_classes(arm_dir):
            col = "Prob_Class_%d" % cls
            if col not in a.columns or col not in n.columns:
                continue
            pa, pn = a[col].to_numpy(), n[col].to_numpy()
            K = budget_for(a, cls)
            if K < 3:
                continue

            # the contested band: where the cut actually falls
            order_n = np.argsort(-pn)
            lo, hi = max(1, K // 2), min(len(pn), 2 * K)
            band = order_n[lo:hi]

            top_a = set(np.argsort(-pa)[:K].tolist())
            top_n = set(np.argsort(-pn)[:K].tolist())
            jac = len(top_a & top_n) / max(1, len(top_a | top_n))

            row = {"model": model, "cap": cap, "seed": seed, "cls": cls,
                   "K": K,
                   "rho_arm": spearman(pa, pn),
                   "rho_arm_band": spearman(pa[band], pn[band]),
                   "jac_arm": jac}
            if r is not None and col in r.columns:
                pr = r[col].to_numpy()
                top_r = set(np.argsort(-pr)[:K].tolist())
                row["rho_reseed"] = spearman(pr, pn)
                row["rho_reseed_band"] = spearman(pr[band], pn[band])
                row["jac_reseed"] = len(top_r & top_n) / max(1, len(top_r | top_n))
            rows.append(row)

    if not rows:
        print("no paired runs found under %s" % args.campaign)
        return 1

    d = pd.DataFrame(rows)
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
    cols = ["rho_arm", "rho_arm_band", "jac_arm"]
    if have_ctrl:
        cols += ["rho_reseed", "rho_reseed_band", "jac_reseed"]
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
        dd = d2["rho_arm"] - d2["rho_reseed"]
        db = (d2["rho_arm_band"] - d2["rho_reseed_band"]).dropna()
        print("VERDICT -- read the SIGN, never rho alone")
        print("   rho_arm - rho_reseed        %+.4f   (%d/%d points where the "
              "constraint reordered MORE than a reseed)"
              % (dd.mean(), int((dd < 0).sum()), len(dd)))
        print("   band                        %+.4f   (%d/%d)"
              % (db.mean(), int((db < 0).sum()), len(db)))
        print()
        if dd.mean() >= 0:
            print("   => rho_arm >= rho_reseed: the constraint preserved the "
                  "order at least as well as")
            print("      doing NOTHING and reseeding. It moved the cut, not "
                  "the ranking -- which is what")
            print("      `sum_i f(p_ic)` is mathematically able to do, and all "
                  "it is able to do.")
        else:
            print("   => the constraint reordered MORE than a reseed. The "
                  "order-preservation argument")
            print("      does NOT hold here; the representation channel is "
                  "doing something. Say how much.")
    return 0



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
def evictions(campaign, arm_name, null_name, reseed_name):
    rows = []
    pat = os.path.join(campaign, "*", "*", "*", arm_name, "seed_*")
    for arm_dir in sorted(glob.glob(pat)):
        parts = arm_dir.split(os.sep)
        model, cap, seed = parts[-5], parts[-3], parts[-1]
        base = os.sep.join(parts[:-2])
        a = load(arm_dir)
        n = load(os.path.join(base, null_name, seed))
        if a is None or n is None:
            continue
        for cls in capped_classes(arm_dir):
            col = "Prob_Class_%d" % cls
            if col not in a.columns:
                continue
            y = a["True_Label"].to_numpy()
            K = budget_for(a, cls)
            if K < 3:
                continue
            pn = n[col].to_numpy()
            ta = set(np.argsort(-a[col].to_numpy())[:K].tolist())
            tn = set(np.argsort(-pn)[:K].tolist())
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
            p_cut = float(np.sort(pn)[::-1][K - 1])
            rows.append({
                "model": model, "cap": cap, "seed": seed, "cls": cls, "K": K,
                "p_cut": p_cut, "p_evicted": pn_ev, "p_admitted": pn_ad,
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
