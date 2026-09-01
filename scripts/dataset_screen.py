"""Can a count constraint POSSIBLY help on this dataset? Ask before downloading it.

THE ONE CRITERION, derived from every null this project has measured.

Post-hoc top-K allocation is provably optimal for expected TP *given the
probabilities*, so training can only win by producing BETTER PROBABILITIES. The
constraint's only contribution is information: it states counts. That
information can improve the probabilities only if it is information the TRAINING
SET DOES NOT ALREADY CARRY.

On all three current datasets it does not, and two measurements say so:

  * FRAMEWORK 2(j): top-K is invariant to a GLOBAL prior shift. One multiplier
    per class is monotone, so it cannot reorder any two items -- a 1000x
    correction moved fewer items than an RNG reseed. A global count cap
    therefore cannot help on ANY dataset, however chosen. That route is shut.
  * FRAMEWORK 2(m): a PER-GROUP multiplier is not monotone over the full set and
    CAN reorder across groups, which is the one live route. But on dermmnist the
    residual per-group miscalibration is 1.68x and only 6 items moved, because
    the model has already learned each group's prior from training.

So the live route needs per-group counts the model CANNOT have learned. That
happens when the test groups are not the training groups, or when a group's
class distribution genuinely shifts between train and test. Everything below
measures exactly that, from labels and metadata alone -- no images, no model, no
GPU.

THE UNIT IS ITEMS, because that is the unit every effect in this project is
quoted in, and the scale is unforgiving: the paired seed sd is ~2.7 items and
the whole `clip`-to-perfect headroom is 1.9-9.9 items. A cap whose novelty is
"about one item" cannot be beaten out of, whatever the method.

    python -m scripts.dataset_screen data/dermmnist/slice_1 data/octmnist/slice_1
"""
import argparse
import os

import numpy as np
import pandas as pd

# ⚠️ THIS DIVISOR IS FROM A REMOVED DATASET, AND EVERY BOUNDARY ABOVE RESTS ON
# IT. 2.7 is the paired seed sd on dermmnist x MobileNetV3, and dermmnist is
# removed and was leaked. The live figure this project measures on iwildcam is
# 4.75 to 27.83 items (`ceiling_screen.IWILDCAM_CURVE`), so every threshold here
# is 1.8x to 10x too generous. `--noise` overrides it; the verdicts print which
# value they used, because a PASS at 2.7 can be a DEAD at 27.8.
SEED_NOISE_ITEMS = 2.7      # dermmnist x MobileNetV3, paired seed sd
GROUP_CANDIDATES = ("loc_group", "synth_group", "group", "domain", "site",
                    "location", "hospital", "region", "group_id")


def _group_column(df):
    for c in GROUP_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _dev(counts_obs, counts_exp):
    return float(np.abs(counts_obs - counts_exp).sum())


GENERIC_SLICE_DIRS = ("oodslice", "slice_1", "shift_1", "data", ".")


def slice_label(path):
    """A name that IDENTIFIES the slice, not the convention it was cut with.

    Every candidate slice is written to `<dataset>/oodslice`, so the bare
    basename is the same string for all of them. Screening the 21-candidate
    inventory printed `oodslice` on all 21 rows and was unreadable -- which
    matters because this tool exists to be run on MANY slices at once and its
    whole output is the comparison between them. Walk up until the component
    says something.
    """
    parts = [p for p in os.path.normpath(path).replace("\\", "/").split("/") if p]
    keep = []
    for p in reversed(parts):
        keep.insert(0, p)
        if p not in GENERIC_SLICE_DIRS:
            break
    return "/".join(keep) if keep else path


def novelty_items(train, test, gcol, label="label", n_null=200, seed=0):
    """How many items does knowing the TEST counts buy over predicting them
    from TRAINING prevalence -- IN EXCESS OF SAMPLING NOISE?

    🛑 THE RAW DEVIATION IS NOT THE ANSWER, and reading it as one scored
    dermmnist at "62x the seed noise" when dermmnist is measured to null. Even
    under ZERO shift, drawing n_g labels from the training distribution gives
    E|X - np| ~ sqrt(2 n p (1-p) / pi) per cell, and summing that over classes
    and groups produces a large positive number out of nothing. A group of 218
    manufactures tens of "novel" items by binomial noise alone.

    So the null is simulated explicitly: resample each group's test labels from
    ITS OWN training distribution, recompute the same statistic, and report the
    EXCESS over that null in items, with a z against the null spread. A dataset
    is only a candidate if the excess survives.
    """
    rng = np.random.default_rng(seed)
    classes = sorted(set(train[label]) | set(test[label]))
    idx = {c: i for i, c in enumerate(classes)}
    n_tr, n_te = len(train), len(test)

    def cell_counts(frame):
        out = np.zeros(len(classes))
        for c, k in frame[label].value_counts().items():
            out[idx[c]] = k
        return out

    p_glob = cell_counts(train) / n_tr
    glob_obs = _dev(cell_counts(test), p_glob * n_te)
    glob_null = np.array([_dev(rng.multinomial(n_te, p_glob), p_glob * n_te)
                          for _ in range(n_null)])

    units, unseen_groups, unseen_items = [], [], 0
    if gcol is not None:
        for g in sorted(test[gcol].unique()):
            te_g = test[test[gcol] == g]
            tr_g = train[train[gcol] == g]
            if len(tr_g) == 0:
                # 🛑 AN UNSEEN GROUP IS THE STRONGEST CASE, NOT A MISSING ONE.
                # Skipping it (the first version did) reports novelty 0 for a
                # fully held-out-domain split -- the exact design the criterion
                # asks for -- because no unit survives to be summed. A model
                # that has never seen this group holds no group-specific prior,
                # so the best it can do is fall back to the GLOBAL training
                # prevalence. That is the honest baseline, and the deviation
                # from it is precisely what the cap would be telling it.
                unseen_groups.append(g)
                unseen_items += len(te_g)
                units.append((cell_counts(te_g), p_glob, len(te_g)))
                continue
            units.append((cell_counts(te_g), cell_counts(tr_g) / len(tr_g),
                          len(te_g)))
    else:
        units = []

    # THE NET IS THE ONLY REORDERABLE PART, and this decomposition is the whole
    # point. If every group shifts by the SAME factor, the per-group correction
    # is one global multiplier wearing three hats -- monotone, unable to reorder
    # anything, dead by 2(j). What can reorder is the DIFFERENTIAL: group A up
    # while group B is down. So the net expectation rescales each group's
    # training prevalence by the observed GLOBAL shift first, and measures what
    # is left over.
    shift = np.divide(cell_counts(test) / n_te, p_glob,
                      out=np.ones_like(p_glob), where=p_glob > 0)

    def net_expect(p, n):
        q = p * shift
        tot = q.sum()
        return (q / tot * n) if tot > 0 else p * n

    loc_obs = sum(_dev(obs, p * n) for obs, p, n in units)
    net_obs = sum(_dev(obs, net_expect(p, n)) for obs, p, n in units)
    net_null = np.array([
        sum(_dev(rng.multinomial(n, net_expect(p, n) / n), net_expect(p, n))
            for _, p, n in units)
        for _ in range(n_null)]) if units else np.zeros(n_null)
    loc_null = np.array([
        sum(_dev(rng.multinomial(n, p), p * n) for _, p, n in units)
        for _ in range(n_null)]) if units else np.zeros(n_null)

    def summarise(obs, null):
        sd = float(null.std(ddof=1)) if len(null) > 1 else 0.0
        excess = obs - float(null.mean())
        return excess, (excess / sd if sd > 0 else float("nan"))

    g_ex, g_z = summarise(glob_obs, glob_null)
    l_ex, l_z = summarise(loc_obs, loc_null)
    n_ex, n_z = summarise(net_obs, net_null)
    return {"net_items": n_ex, "net_z": n_z, "net_raw": net_obs,
            "net_null": float(net_null.mean()),
            "global_items": g_ex, "global_z": g_z, "global_raw": glob_obs,
            "global_null": float(glob_null.mean()),
            "local_items": l_ex, "local_z": l_z, "local_raw": loc_obs,
            "local_null": float(loc_null.mean()),
            "unseen_groups": unseen_groups, "unseen_items": unseen_items}


def heterogeneity_items(test, gcol, label="label"):
    """How far the per-group class distribution is from proportional, in items.

    A PRECONDITION, not a result: if every group holds the same class mix, a
    local cap is just the global cap divided up and carries nothing extra. But
    dermmnist has a 5.4x prevalence spread across groups and still nulls, so
    heterogeneity alone is not sufficient -- read it beside `local_items`.
    """
    if gcol is None:
        return 0.0
    n = len(test)
    out = 0.0
    for c in sorted(test[label].unique()):
        n_c = int((test[label] == c).sum())
        for g in sorted(test[gcol].unique()):
            te_g = test[test[gcol] == g]
            expected = n_c * len(te_g) / n
            out += abs(int((te_g[label] == c).sum()) - expected)
    return out


def screen(path):
    tr = pd.read_csv(os.path.join(path, "train_meta.csv"))
    te = pd.read_csv(os.path.join(path, "test_meta.csv"))
    gcol = _group_column(te)
    counts = te["label"].value_counts().sort_index()
    ratio = counts.max() / max(counts.min(), 1)
    nov = novelty_items(tr, te, gcol)
    return {"path": path, "n_train": len(tr), "n_test": len(te),
            "n_classes": int(te["label"].nunique()), "gcol": gcol,
            "n_groups": int(te[gcol].nunique()) if gcol else 0,
            "counts": counts.to_dict(), "imbalance": float(ratio),
            "rarest": int(counts.min()),
            "heterogeneity": heterogeneity_items(te, gcol), **nov}


def verdict_lines(r, name, noise=None):
    """The verdict ladder, as data rather than as prints.

    Extracted so it can be gated. It decided which datasets this project
    would spend a campaign on, and it lived inside `main()` where nothing
    could reach it -- which is why an undefined z fell through DEAD into
    STAGE 1 PASS for as long as it did.
    """
    noise = SEED_NOISE_ITEMS if noise is None else float(noise)
    out = []
    if r["gcol"] is None:
        out.append("  %-22s NO GROUP COLUMN -- the local scope does not exist here."
                  % name)
    elif not np.isfinite(r["net_z"]):
            # 🛑 AN UNDEFINED SIGNIFICANCE TEST USED TO **UPGRADE** THE VERDICT.
            # `summarise` returns nan for z when the null spread is 0, and
            # `nan < 2.0` is False, so the DEAD branch was skipped entirely and
            # the slice fell through to MARGINAL or STAGE 1 PASS on its LOCAL
            # number. An absent measurement must never read as a pass.
        out.append("  %-22s UNDECIDABLE: the sampling-noise null has zero "
                  "spread, so z is" % name)
        out.append("  %-22s   undefined and NOTHING was tested. This is not a "
                  "pass. Usually it" % "")
        out.append("  %-22s   means one group, or identical groups -- check "
                  "the group column." % "")
    elif r["net_z"] < 2.0:
        out.append("  %-22s DEAD: NET per-group novelty %+.0f items is within "
                  "sampling noise (z=%.1f)." % (name, r["net_items"], r["net_z"]))
    elif r["net_items"] < noise:
            # ⚠️ GATE ON `net`, REPORT `net`. This branch used to print
            # `local_items` while testing `net_items`, so a slice with net=1
            # and local=500 printed "DEAD: local novelty 500 items is BELOW
            # the 2.7-item seed noise", which contradicts itself on its own
            # line. LOCAL includes the global shift replicated across groups,
            # which is one multiplier in disguise; NET is the reorderable part.
        out.append("  %-22s DEAD: NET novelty %.0f items is BELOW the %.1f-item "
                  "seed noise (local reads %.0f, but that includes the global "
                  "shift)." % (name, r["net_items"], noise,
                               r["local_items"]))
    elif r["net_items"] < 3 * noise:
        out.append("  %-22s MARGINAL: NET novelty %.0f items against %.1f-item "
                  "noise (local %.0f)." % (name, r["net_items"],
                                           noise, r["local_items"]))
    else:
        out.append("  %-22s STAGE 1 PASS (necessary, not sufficient): NET "
                  "novelty %.0f items, %.0fx seed noise (local %.0f)."
                  % (name, r["net_items"], r["net_items"] / noise,
                     r["local_items"]))
    return out


def main():
    global SEED_NOISE_ITEMS
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="slice dirs with train/test_meta.csv")
    ap.add_argument("--noise", type=float, default=SEED_NOISE_ITEMS,
                    help="paired seed sd in items, the divisor every verdict "
                         "below is scaled by. The default %.1f is dermmnist x "
                         "MobileNetV3, and dermmnist is REMOVED and was leaked; "
                         "iwildcam measures 4.75 to 27.83. Pass the number for "
                         "the dataset and backbone you actually intend to run."
                         % SEED_NOISE_ITEMS)
    args = ap.parse_args()
    SEED_NOISE_ITEMS = float(args.noise)

    print("DATASET SCREEN -- can a count constraint carry information here?")
    print("Everything is in ITEMS. Every verdict below is scaled by a paired "
          "seed sd of")
    print("%.2f items%s. On iwildcam the measured range is 4.75 to 27.83, so a "
          "PASS at 2.7"
          % (SEED_NOISE_ITEMS,
             " (the default: dermmnist x MobileNetV3, a REMOVED dataset)"
             if abs(SEED_NOISE_ITEMS - 2.7) < 1e-9 else " (--noise)"))
    print("can be a DEAD at 27.8. Pass --noise to price it for the dataset you "
          "will run.")
    print("")

    rows = [screen(p) for p in args.paths]
    print("  %-34s %7s %6s %7s %7s %9s"
          % ("dataset", "n_test", "cls", "groups", "imbal", "rarest"))
    for r in rows:
        print("  %-34s %7d %6d %7d %7.1fx %9d"
              % (slice_label(r["path"])[-34:], r["n_test"], r["n_classes"], r["n_groups"],
                 r["imbalance"], r["rarest"]))

    print("")
    print("  NOVELTY = observed deviation MINUS the sampling-noise null, in "
          "items.")
    print("  %-30s %9s %6s %9s %6s %9s %6s %7s"
          % ("dataset", "NET ex", "z", "LOCAL ex", "z", "GLOBAL ex", "z",
             "unseen"))
    for r in rows:
        print("  %-30s %+9.0f %6.1f %+9.0f %6.1f %+9.0f %6.1f %7d"
              % (slice_label(r["path"])[-30:], r["net_items"], r["net_z"],
                 r["local_items"], r["local_z"], r["global_items"],
                 r["global_z"], len(r["unseen_groups"])))

    print("")
    print("  !! STAGE 1 ONLY. Passing here is NECESSARY, NOT SUFFICIENT.")
    print("  dermmnist scores +69 items (z=3.4) of real per-group novelty and")
    print("  still nulls: FRAMEWORK 2(m) fed a model the TRUE per-group counts")
    print("  and only 6 items moved. Information existing is not the same as it")
    print("  being convertible into ORDERING. Stage 2 needs one trained model:")
    print("  `scripts.scope_probe --calibrate`, which measures how many items a")
    print("  per-group correction of the observed size actually moves.")
    print("")
    print("  WHAT SEPARATES THE TWO. A per-group multiplier only flips items")
    print("  whose scores sit within its ratio of the cut. On dermmnist the")
    print("  residual factor between groups is 1.68x and the top-K items are")
    print("  further apart than that. So stage 2 wants groups whose class")
    print("  distributions differ by ORDERS OF MAGNITUDE, not the 5.4x here --")
    print("  and its strongest form is a test group ABSENT from training, where")
    print("  the model holds no prior at all and the cap is the only source.")
    print("  Every dataset below has `unseen = 0`. None of them has that.")
    print("")
    print("  READ `NET` FIRST -- it is the DIFFERENTIAL per-group shift, the")
    print("  only part that can reorder. LOCAL includes the global shift")
    print("  replicated across groups, which is one multiplier in disguise.")
    print("  GLOBAL novelty cannot help at any size: FRAMEWORK 2(j) showed top-K")
    print("  is invariant to a global prior shift, because one multiplier per")
    print("  class is monotone and cannot reorder. Heterogeneity is a")
    print("  PRECONDITION only -- dermmnist has a 5.4x spread and still nulls.")
    print("")
    for r in rows:
        name = slice_label(r["path"])
        for line in verdict_lines(r, name):
            print(line)
        if r["unseen_groups"]:
            print("  %-22s   and %d test group(s) are ABSENT from train (%d "
                  "items) -- training carries no prior for them at all."
                  % ("", len(r["unseen_groups"]), r["unseen_items"]))


if __name__ == "__main__":
    main()
