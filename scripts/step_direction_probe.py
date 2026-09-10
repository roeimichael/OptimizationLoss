"""Do the count functions deliver DIFFERENT parameter steps? They do not.

Every count function in this project has the form `S_c = sum_i phi(p_ic)`, so
its gradient w.r.t. the capped class's head weights is

    dS/dw_c = sum_i g_i * f_i ,        g_i = phi'(p_ic) * dp_ic/dz_ic

which is a g-WEIGHTED MEAN OF THE TEST FEATURES. The only thing a new count
function can change is the weighting g. Under `constraint_grad_mode: normalize`
the magnitude is discarded (FRAMEWORK: the delivered step is exactly lr*clip),
so a new count function can only matter if it changes the DIRECTION.

This measures that directly, on the stored `test_embeddings.npz`. If the
directions are collinear, the count-function family is one arm and no member of
it can behave differently from any other -- which closes the entire family with
one number instead of one campaign each.

\u26a0\ufe0f HEAD-ONLY, and deliberately so. It bounds what the count function can do
through the LINEAR HEAD, where the effect is exactly computable. The backbone
adds a further channel, measured separately and separately negative
(iwc1/iwc2, AP -0.031 / -0.094 vs the twin). A collinearity here does not say
the arms are identical end-to-end; it says the count function is not the thing
that differentiates them.
"""
import argparse
import csv
import glob
import os
import sys
from scripts import capped_classes                  # noqa: E402

import numpy as np

BAND = 20      # half-width in ITEMS of the band straddling the cut


def sech2(x):
    """Stable and precise: `1/cosh(x)**2` overflows, the sigmoid form underflows."""
    a = np.exp(-2.0 * np.abs(x))
    return 4.0 * a / (1.0 + a) ** 2


def group_tau(z, groups, k_by_group):
    """Per-ITEM cut logit: the cut of the group that item sits in.

    A single scalar `tau` is the GLOBAL K-th logit and is not a cut the
    allocator ever makes. An item in a confident group is far above its own
    cut while sitting near the global one, and vice versa.
    """
    tau = np.empty_like(z)
    for g in np.unique(groups):
        where = np.flatnonzero(groups == g)
        kg = int(k_by_group.get(g if isinstance(g, str) else str(g), 0))
        zs = np.sort(z[where])[::-1]
        tau[where] = zs[min(max(kg, 1), len(zs)) - 1]
    return tau


def weightings(p, z, K=None, n_items=40, tau=None):
    """Per-item gradient weights. `cut_window` needs a CUT and is skipped
    without one.

    `margin_sech2` is centred on the DECISION BOUNDARY (|z| = 0) and
    `cut_window` on the CUT. Those are the two things CLAUDE.md rule 3 warns
    against conflating, and on tight caps they are ~300 items apart.

    !! `tau` WAS THE GLOBAL K-TH LOGIT UNTIL 2026-09-10, so the candidate
    weighting (z12) proposed as ITS OWN FIX was aimed at a cut the allocator
    never makes -- the same substitution as the diagnosis it was answering.
    Pass a per-item `tau` from `group_tau` to aim it where the emission
    actually happens. The scalar path is kept so (z12)'s published table
    regenerates. FRAMEWORK 2(z79).
    """
    m = np.abs(z)
    w = {
        "uniform": np.ones_like(p),
        "sum_p(1-p)": p * (1.0 - p),
        "margin_sech2": sech2(m / 0.5),
        "one_minus_p": 1.0 - p,
        "p": p.copy(),
        "linear_z": z - z.min(),
    }
    if tau is None and K is not None and 1 <= K < len(z):
        tau = np.sort(z)[::-1][K - 1]
    if tau is not None:
        d = np.abs(z - tau)
        T = max(float(np.sort(d)[min(n_items, len(d)) - 1]), 1e-9)
        w["cut_window"] = sech2((z - tau) / T)
    return w


def probs(run_dir, cls):
    """(probabilities for `cls`, K deployed, group ids, per-group k_g).

    !! IT REFUSES A PREDICTIONS FILE WITH NO `Group_ID` RATHER THAN FALLING
    BACK TO THE GLOBAL READING (2026-09-10). That is the fix `paired_noise`
    got in 2(z63), and the refusal is the point: a silent global fallback is
    exactly how this substitution survived in five other tools. `Group_ID` is
    what `src/training/logging.py` writes.

    `k_g` is COUNTED off the deployed labels, not derived from a cap policy,
    so it is what the allocator actually emitted in that group.
    """
    path = os.path.join(run_dir, "final_predictions.csv")
    col = "Prob_Class_%d" % cls
    out, gids, K = [], [], 0
    with open(path) as f:
        rdr = csv.DictReader(f)
        if "Group_ID" not in (rdr.fieldnames or []):
            raise SystemExit(
                "%s has no `Group_ID` column, so the cut cannot be taken where "
                "the allocator takes it. Refusing rather than reading a GLOBAL "
                "top-K -- FRAMEWORK 2(z63), 2(z79)." % path)
        for row in rdr:
            out.append(float(row[col]))
            gids.append(row["Group_ID"])
            K += int(int(row["Predicted_Label"]) == cls)
    g = np.array(gids)
    kg = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if int(row["Predicted_Label"]) == cls:
                kg[row["Group_ID"]] = kg.get(row["Group_ID"], 0) + 1
    return np.array(out), K, g, kg


def cut_band(z, K, groups=None, k_by_group=None, half=BAND):
    """Indices of the items straddling THE CUT -- per group when told how.

    The allocator emits the top `k_g` inside each group, so the items whose
    movement can change the emitted set are the ones near each GROUP's own
    cut. A single global sort at rank K instead counts high-scoring items in
    confident groups that can never flip, and misses items sitting at the cut
    in hard ones.

    !! THIS IS THE SIXTH SITE OF THAT SUBSTITUTION IN THIS REPO. The cap
    screen (2(z28)), the task window (2(z16)), the fmow window (2(z59)),
    `paired_noise` (2(z63)) and `cut_gap` (2(z64)) were the first five, and
    each of them CHANGED AN ANSWER. FRAMEWORK 2(z79).

    !! AND DO NOT ATTACH A DIRECTION TO THE DIFFERENCE. 2(z64) records the
    author asserting that the global reading understates the gradient at the
    cut, building a fixture that showed it, mutation-testing it 2/2 green, and
    being refuted by the end-to-end run: the maximin property orders the
    global value against the MINIMUM group cut, never against the mean. The
    two readings are both printed here for exactly that reason.
    """
    if groups is None or k_by_group is None:
        return np.argsort(-z)[max(0, K - half):K + half]
    idx = []
    for g in np.unique(groups):
        where = np.flatnonzero(groups == g)
        kg = k_by_group.get(g if isinstance(g, str) else str(g), 0)
        order = where[np.argsort(-z[where])]
        idx.extend(order[max(0, kg - half):kg + half].tolist())
    return np.array(sorted(set(idx)), dtype=int)


def directions(feats, p, K=None, n_items=40, tau=None):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p) - np.log1p(-p)
    D = {}
    for k, g in weightings(p, z, K, n_items, tau).items():
        v = (g[:, None] * feats).sum(axis=0)
        n = np.linalg.norm(v)
        D[k] = v / n if n else v
    fb = feats.mean(axis=0)
    D["_fbar"] = fb / np.linalg.norm(fb)
    return D


def self_test(out=sys.stdout):
    """Synthetic features with a KNOWN answer in both directions."""
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print("  %-62s %s" % (name, "PASS" if cond else "FAIL"), file=out)

    rng = np.random.default_rng(0)
    n, d = 1500, 32

    # (a) features with a LARGE mean -> every weighting collapses onto fbar
    F = rng.normal(size=(n, d)) + 4.0
    p = rng.uniform(0.02, 0.98, size=n)
    D = directions(F, p)
    c = float(D["uniform"] @ D["sum_p(1-p)"])
    check("large-mean features: uniform and sum are collinear (>0.99)",
          c > 0.99)
    check("large-mean features: both align with fbar (>0.99)",
          float(D["uniform"] @ D["_fbar"]) > 0.99
          and float(D["sum_p(1-p)"] @ D["_fbar"]) > 0.99)

    # (b) NEGATIVE CONTROL: a geometry the weighting CAN steer. Items are
    #     mapped to ORTHOGONAL directions by p-bucket, so a flat weighting and
    #     a |p-0.5|-peaked one MUST point somewhere different. Note what it
    #     takes: orthogonal geometry and three separated buckets. With
    #     non-negative features and non-negative weights every direction lies
    #     in the positive orthant and a high cosine is FORCED -- that is the
    #     mechanism this probe measures, not an artefact of it.
    buckets = [(0.01, 0.05), (0.45, 0.55), (0.95, 0.99)]
    per = 500
    p2 = np.concatenate([rng.uniform(lo, hi, per) for lo, hi in buckets])
    F2 = np.zeros((per * len(buckets), len(buckets)))
    for b in range(len(buckets)):
        F2[b * per:(b + 1) * per, b] = 1.0
    D2 = directions(F2, p2)
    c2 = abs(float(D2["uniform"] @ D2["sum_p(1-p)"]))
    check("NEGATIVE CONTROL: a steerable geometry gives cosine < 0.9 "
          "(got %.3f)" % c2, c2 < 0.9)
    check("NEGATIVE CONTROL: so the probe CAN report non-collinearity",
          c2 < 0.99)

    # (c) the cut window appears ONLY when K is given, and must actually
    #     concentrate at the cut -- otherwise the column means nothing.
    F3 = rng.normal(size=(n, d)) + 4.0
    p3 = np.clip(rng.uniform(0.02, 0.98, size=n), 1e-6, 1 - 1e-6)
    K3 = 300
    check("cut_window is ABSENT without K (never a silent zero column)",
          "cut_window" not in directions(F3, p3))
    check("cut_window is PRESENT with K",
          "cut_window" in directions(F3, p3, K3))
    z3 = np.log(p3) - np.log1p(-p3)
    W3 = weightings(p3, z3, K3)
    band = np.argsort(-z3)[K3 - 20:K3 + 20]
    mc = W3["cut_window"][band].sum() / W3["cut_window"].sum()
    ms = W3["sum_p(1-p)"][band].sum() / W3["sum_p(1-p)"].sum()
    check("cut_window puts >10x the shipped count's mass at the cut "
          "(%.3f vs %.3f)" % (mc, ms), mc > 10 * ms)

    # ---- (d) THE CUT IS TAKEN WHERE THE ALLOCATOR TAKES IT --------------
    # 2(z79). Two groups whose difficulty differs: A is easy (high scores,
    # gets most of the budget), B is hard. A GLOBAL rank-K window lands almost
    # entirely inside A and never sees B's cut at all.
    zA = np.linspace(6.0, 4.0, 100)          # easy group, confident
    zB = np.linspace(0.4, -0.4, 100)         # hard group, at the boundary
    zz = np.concatenate([zA, zB])
    gg = np.array(["A"] * 100 + ["B"] * 100)
    kgs = {"A": 60, "B": 20}                 # the allocator's own emission
    K = sum(kgs.values())

    bg = cut_band(zz, K, gg, kgs, half=5)
    bl = cut_band(zz, K, half=5)
    inA = lambda b: int(np.sum(b < 100))
    inB = lambda b: int(np.sum(b >= 100))
    check("per-group band straddles BOTH groups' cuts (A %d, B %d)"
          % (inA(bg), inB(bg)), inA(bg) > 0 and inB(bg) > 0)
    # NEGATIVE CONTROL: the global reading must MISS the hard group, or the
    # fixture does not separate the two rules and the test proves nothing.
    check("NEGATIVE CONTROL: the GLOBAL band misses group B entirely "
          "(A %d, B %d)" % (inA(bl), inB(bl)), inB(bl) == 0)
    # ...and the two must actually differ as index sets.
    check("NEGATIVE CONTROL: the two bands are not the same items",
          set(bg.tolist()) != set(bl.tolist()))
    # The per-group band must sit AT each group's k_g, not at rank 0 or the end.
    ordA = np.argsort(-zA)
    check("the A-side band is centred on k_g=60, not on rank K",
          abs(float(np.median(bg[bg < 100])) - 60) <= 6)

    # ...and with no groups supplied it must reproduce the OLD reading exactly,
    # so (z12)'s published table stays regenerable.
    check("with no groups it reproduces the global window byte for byte",
          np.array_equal(cut_band(zz, K, half=5),
                         np.argsort(-zz)[max(0, K - 5):K + 5]))

    # ---- (e) THE PRESCRIPTION IS AIMED WHERE THE DIAGNOSIS LOOKS ---------
    # `cut_window` is the weighting (z12) proposes as its own fix, and it was
    # centred on the GLOBAL K-th logit -- the same substitution as the defect
    # it answers. A per-item tau puts it on each group's own cut.
    tg = group_tau(zz, gg, kgs)
    tglob = np.sort(zz)[::-1][K - 1]
    check("per-item tau takes TWO values here, one per group (%d)"
          % len(np.unique(tg)), len(np.unique(tg)) == 2)
    check("the hard group's cut is far from the global one (%.2f vs %.2f)"
          % (float(tg[150]), float(tglob)), abs(tg[150] - tglob) > 1.0)
    wg = weightings(1 / (1 + np.exp(-zz)), zz, K, 40, tg)
    wl = weightings(1 / (1 + np.exp(-zz)), zz, K, 40)
    inB_g = float(wg["cut_window"][100:].sum() / wg["cut_window"].sum())
    inB_l = float(wl["cut_window"][100:].sum() / wl["cut_window"].sum())
    check("aimed cut_window puts real mass on the HARD group (%.3f)" % inB_g,
          inB_g > 0.2)
    # NEGATIVE CONTROL: the globally-aimed one must NOT, or the fixture does
    # not separate the two and the check above proves nothing.
    check("NEGATIVE CONTROL: the globally-aimed one nearly ignores it (%.4f)"
          % inB_l, inB_l < 0.05)

    print("\n%s" % ("ALL PASS" if ok else "FAILURES ABOVE"), file=out)
    return 0 if ok else 1


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--runs", nargs="+")
    a.add_argument("--glob")
    a.add_argument("--classes", nargs="+", type=int,
                   default=None, help="capped classes. DEFAULT: read from the campaign's own config.json. A pair that CONTRADICTS the config is REFUSED -- the old default was iwildcam's [2, 7], wrong on bcn (0, 2) and fmow (3, 5), and it printed a plausible number rather than raising.")
    a.add_argument("--n-items", type=int, default=40,
                   help="items inside the cut window (T is derived from this)")
    a.add_argument("--limit", type=int, default=0,
                   help="cap the number of run dirs (0 = no cap). "
                        "Prints what it dropped.")
    a.add_argument("--self-test", action="store_true")
    args = a.parse_args(argv)
    if args.self_test:
        return self_test()
    # THE CAPPED CLASSES COME FROM THE CAMPAIGN, NOT FROM A DEFAULT.
    # This defaulted to iwildcam's [2, 7] and every caller consumed
    # it, so on a bcn or fmow campaign it scored two classes the
    # experiment never constrained -- silently, with a plausible
    # number attached. FRAMEWORK 2(z60).
    args.classes = capped_classes.resolve(args.glob, args.classes)

    # 🛑 THIS WAS `[:12]`, AND IT WAS A SILENT CAP. `sorted()` is alphabetical,
    # so on a 24-run cell it kept the first two or three ARMS and dropped the
    # rest -- a subset that looks like a campaign and is not one. It produced
    # the first version of the FRAMEWORK 2(z12) table, which read 24 pairs when
    # the campaign held far more. No silent caps: unlimited by default, and if
    # a limit is asked for, say what it dropped.
    runs = args.runs or sorted(glob.glob(args.glob))
    if not runs:
        raise SystemExit("no runs")
    if args.limit and args.limit < len(runs):
        print("!! --limit %d applied: %d of %d run dir(s) DROPPED. `sorted()` "
              "is alphabetical, so this biases toward whichever arms sort "
              "first -- do not read a cross-arm number off a limited run."
              % (args.limit, len(runs) - args.limit, len(runs)))
        runs = runs[:args.limit]

    keys = None
    acc = {}
    mass = {}          # per-GROUP cut band -- what the allocator actually cuts
    mass_glob = {}     # the old global rank-K window, kept so (z12) regenerates
    nband = {}
    nrun = 0
    for rd in runs:
        fp = os.path.join(rd, "test_embeddings.npz")
        if not os.path.exists(fp):
            continue
        F = np.load(fp)["features"].astype(np.float64)
        for c in args.classes:
            try:
                p, K, gids, kg = probs(rd, c)
            except SystemExit:
                raise
            except Exception:
                continue
            if len(p) != len(F):
                continue
            pc0 = np.clip(p, 1e-6, 1 - 1e-6)
            zc0 = np.log(pc0) - np.log1p(-pc0)
            taug = group_tau(zc0, gids, kg)
            D = directions(F, p, K, args.n_items, taug)
            if keys is None:
                keys = [k for k in D if k != "_fbar"]
            for i, x in enumerate(keys):
                for y in keys[i:]:
                    acc.setdefault((x, y), []).append(float(D[x] @ D[y]))
                acc.setdefault((x, "_fbar"), []).append(float(D[x] @ D["_fbar"]))

            # the z12(b) table: where each weighting's gradient actually lands.
            # BOTH readings, because the entry's published figures are the
            # global one and must stay reproducible -- 2(z79).
            pc = np.clip(p, 1e-6, 1 - 1e-6)
            zc = np.log(pc) - np.log1p(-pc)
            grp = cut_band(zc, K, gids, kg)
            glb = cut_band(zc, K)
            nband.setdefault("per_group", []).append(len(grp))
            nband.setdefault("global", []).append(len(glb))
            wg = weightings(pc, zc, K, args.n_items, taug)   # cut_window aimed
            wl = weightings(pc, zc, K, args.n_items)         # ...and the old one
            for k in wg:
                tg, tl = float(wg[k].sum()), float(wl[k].sum())
                mass.setdefault(k, []).append(
                    float(wg[k][grp].sum()) / tg if tg else 0.0)
                mass_glob.setdefault(k, []).append(
                    float(wl[k][glb].sum()) / tl if tl else 0.0)
            nrun += 1
    if not nrun:
        raise SystemExit("no usable (run, class) pairs")

    print("STEP-DIRECTION PROBE -- cosine between count functions' parameter "
          "steps")
    print("%d (run, class) pairs, REAL stored features\n" % nrun)
    print("%-14s %s" % ("", "".join("%13s" % k[:12] for k in keys)))
    for x in keys:
        cells = []
        for y in keys:
            v = acc.get((x, y)) or acc.get((y, x))
            cells.append("%13.4f" % np.mean(v))
        print("%-14s %s" % (x[:14], "".join(cells)))
    gb = float(np.mean(nband["per_group"]))
    lb = float(np.mean(nband["global"]))
    print("\nFRACTION OF TOTAL GRADIENT MASS at the cut -- the only items whose")
    print("movement can change the emitted top-K set. BOTH readings, because")
    print("the allocator cuts top-k_g WITHIN each group and this tool read a")
    print("GLOBAL rank-K window until 2026-09-10 (FRAMEWORK 2(z79), the SIXTH")
    print("site of that substitution in this repo).")
    print("   band size: per_group %.0f items, global %.0f -- so compare the"
          % (gb, lb))
    print("   per-ITEM column, not the raw fractions.\n")
    print("   %-16s %10s %10s %12s %12s"
          % ("weighting", "PER-GROUP", "global", "per-item pg", "per-item gl"))
    for k in sorted(mass, key=lambda k: -float(np.mean(mass[k]))):
        v, w = mass[k], mass_glob[k]
        print("   %-16s %10.4f %10.4f %12.2e %12.2e"
              % (k, np.mean(v), np.mean(w),
                 np.mean(v) / gb if gb else 0.0,
                 np.mean(w) / lb if lb else 0.0))
        print("   %-16s   [pg min %.4f max %.4f]  [gl min %.4f max %.4f]"
              % ("", np.min(v), np.max(v), np.min(w), np.max(w)))
    print("\n!! DO NOT READ THE PER-GROUP/GLOBAL GAP AS A DIRECTION. 2(z64)")
    print("   records that claim being asserted, fixtured, mutation-tested 2/2")
    print("   green, and REFUTED end to end: the maximin property orders the")
    print("   global value against the MINIMUM group cut, not the mean.")
    print("\ncosine with the PLAIN MEAN FEATURE f_bar (mean over pairs, "
          "min in brackets):")
    for x in keys:
        v = acc[(x, "_fbar")]
        print("   %-16s %8.4f   [min %.4f]" % (x, np.mean(v), np.min(v)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
