"""Is an unseen group's novelty REAL, or does the screen's baseline invent it?

FRAMEWORK 2(n) gives an unseen test group the GLOBAL training prevalence as its
baseline, on the argument that a model which never saw camera 501 holds no
prior for it. That argument is exactly right for an ATOMIC group -- a camera, a
hospital, a trap -- and `dataset_screen` is sound there.

IT IS TOO GENEROUS FOR A GROUP BUILT AS A PRODUCT OF FACTORS THAT BOTH APPEAR
IN TRAINING. A model that has seen (head/neck, 60s) and (upper extremity, 70s)
can interpolate a prior for (head/neck, 70s) far better than the global one, so
the "novelty" the screen credits is partly information the training set already
carries -- which is the one thing 2(n) exists to exclude.

THE CONTROL. Re-measure NET with each unseen group's baseline replaced by the
independence (raking) estimate  p(c|f0) * p(c|f1) / p(c),  renormalised, with
both marginals taken from TRAINING. Whatever survives is novelty the factor
structure does not already supply.

TWO DEFECTS FOUND 2026-09-01. Both pushed `survives` toward a reassuring 100%,
which is the direction that keeps a bad slice alive.

1. THE PUBLISHED NEGATIVE CONTROL WAS AN ARITHMETIC IDENTITY. It used to print

       iwildcam 100.1%   cct 99.9%   idaho 100.0%   wcs 99.8%   serengeti 100.0%

   as five atomic datasets PASSING. Those are not five measurements. When the
   separator does not occur in the group label, `str.split(sep)[0]` and `[-1]`
   are both the WHOLE string, so `f0 == f1`, every unseen group falls back to
   `q = p_glob`, and `units_add` is element-wise EQUAL to `units_glob`. The two
   arms then differ only by the null draw -- the 0.1-0.2% scatter in that row
   IS that draw. On a synthetic atomic slice three different separators return
   the same 99.6% with 0 of 6 groups factorised. A control that cannot fail is
   not a control, and a WRONG `--sep` on a genuinely factorial slice returned
   the same reassuring ~100%. The percentage is now GATED on how many unseen
   groups actually received a raked baseline; zero of them prints NOT A CONTROL.

2. THE RATIO WAS DILUTED BY THE SEEN GROUPS. `survives` was
   `NET(additive) / NET(global)` over the WHOLE slice, but the two arms differ
   only on the UNSEEN units -- every seen group contributes the same items to
   both. So the ratio was dragged toward 100% in proportion to how much of the
   test set is seen. Measured on a slice built so that raking is EXACTLY right
   (the held-out cell drawn from the product of the observed training
   marginals), 6 seeds per row:

       unseen share of test    shipped ratio    unseen-only ratio
              65.2%                87.1%              81.4%
              20.0%                47.5%              19.6%
               7.0%                76.0%              21.2%
               2.4%                91.9%              26.1%

   The truth is the right-hand column and it is flat, as it must be. The
   shipped column is a reading of the unseen SHARE, not of the factor
   structure. `survives` is now the unseen-only ratio; the diluted whole-slice
   figure is still printed, labelled, because that is the one `dataset_screen`
   inherits.

   Direction of the old bias: it OVERSTATED survival, so the ISIC headline
   (+2169 global -> +380 additive, "82% of it was the baseline") is a
   conservative reading and gets stronger, not weaker, under the fix. Every
   atomic ~100% is void in both directions.

CAVEAT THAT CANNOT BE FIXED HERE. `net_expect` applies the GLOBAL test-vs-train
label shift to both baselines, by `dataset_screen`'s definition. When one
unseen cell is most of the test set, that shift is largely computed FROM the
cell, so it absorbs the cell's own novelty and the raked baseline is corrected
twice: measured, the raking estimate sits 0.014 in L1 from the truth and
0.126 after the shift. Read `unseen share` before the ratio -- above ~50% the
comparison degrades from that side.

    python -m scripts.factorial_control <slice-dir> ...      # no images, no GPU
    python -m scripts.factorial_control --self-test
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

from scripts.dataset_screen import _dev


def control(path, sep="|", n_null=200, seed=0):
    tr = pd.read_csv(os.path.join(path, "train_meta.csv"))
    te = pd.read_csv(os.path.join(path, "test_meta.csv"))
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
        f["_ntok"] = s.str.count(re.escape(sep)) + 1
    p_f0 = {k: cc(g) / len(g) for k, g in tr.groupby("_f0")}
    p_f1 = {k: cc(g) / len(g) for k, g in tr.groupby("_f1")}

    seen_units, unseen_glob, unseen_add = [], [], []
    n_raked = no_sep = unseen_factor = multi_tok = n_unseen_items = 0
    for g in sorted(te["location"].unique()):
        te_g = te[te["location"] == g]
        tr_g = tr[tr["location"] == g]
        if len(tr_g):
            # Identical in both arms, so it can only DILUTE the ratio. Kept out
            # of `survives` and carried only for the whole-slice figure.
            seen_units.append((cc(te_g), cc(tr_g) / len(tr_g), len(te_g)))
            continue
        n_unseen_items += len(te_g)
        f0, f1 = te_g["_f0"].iloc[0], te_g["_f1"].iloc[0]
        unseen_glob.append((cc(te_g), p_glob, len(te_g)))
        if f0 == f1:
            # The separator does not occur in this label. Either the group is
            # genuinely ATOMIC or `--sep` is wrong -- and NOTHING here can tell
            # those apart, so neither may be reported as a passed control.
            no_sep += 1
            q = p_glob
        elif f0 not in p_f0 or f1 not in p_f1:
            unseen_factor += 1
            q = p_glob
        else:
            n_raked += 1
            multi_tok += int(te_g["_ntok"].iloc[0] > 2)
            q = np.divide(p_f0[f0] * p_f1[f1], p_glob,
                          out=np.zeros_like(p_glob), where=p_glob > 0)
            q = q / q.sum() if q.sum() > 0 else p_glob
        unseen_add.append((cc(te_g), q, len(te_g)))

    shift = np.divide(cc(te) / n_te, p_glob, out=np.ones_like(p_glob),
                      where=p_glob > 0)

    def net_expect(p, n):
        q = p * shift
        tot = q.sum()
        return (q / tot * n) if tot > 0 else p * n

    def net(units):
        # Its OWN rng, so the two arms face the SAME null draws and an un-raked
        # slice returns exactly 100.000% rather than a plausible 99.6%.
        if not units:
            return 0.0, float("nan")
        rng = np.random.default_rng(seed)
        obs = sum(_dev(o, net_expect(p, n)) for o, p, n in units)
        null = np.array([sum(_dev(rng.multinomial(n, net_expect(p, n) / n),
                                  net_expect(p, n)) for _, p, n in units)
                         for _ in range(n_null)])
        ex = obs - float(null.mean())
        sd = float(null.std(ddof=1))
        return ex, (ex / sd if sd > 0 else float("nan"))

    a, az = net(unseen_glob)
    b, bz = net(unseen_add)
    all_a, _ = net(seen_units + unseen_glob)
    all_b, _ = net(seen_units + unseen_add)
    return {"path": path, "sep": sep, "unseen": len(unseen_glob),
            "raked": n_raked, "no_sep": no_sep,
            "unseen_factor": unseen_factor, "multi_tok": multi_tok,
            "unseen_share": (n_unseen_items / n_te) if n_te else float("nan"),
            "net_global": a, "z_global": az, "net_additive": b, "z_additive": bz,
            "survives": (100 * b / a) if (a and n_raked) else float("nan"),
            "net_all_global": all_a, "net_all_additive": all_b,
            "survives_diluted": ((100 * all_b / all_a)
                                 if (all_a and n_raked) else float("nan"))}


def report(r, name):
    """One dataset. Prints a percentage ONLY when the control actually ran."""
    head = ("  %-24s unseen=%2d (%4.1f%% of test)  raked=%2d  "
            "NET(global) %+7.0f z=%5.1f"
            % (name[-24:], r["unseen"], 100 * r["unseen_share"], r["raked"],
               r["net_global"], r["z_global"]))
    if not r["raked"]:
        why = ("the separator %r does not occur in any unseen group label"
               % r["sep"] if r["no_sep"] == r["unseen"] else
               "every unseen group has a factor level that is itself unseen")
        return [head,
                "      *** NOT A CONTROL: 0 of %d unseen groups were factorised."
                % r["unseen"],
                "          %s," % why,
                "          so the additive baseline IS the global one and the two",
                "          arms are the same arm. This says nothing about whether",
                "          the group is atomic -- an ATOMIC slice and a WRONG",
                "          --sep are indistinguishable here. Re-run with the real",
                "          separator, or read `dataset_screen` unadjusted."]
    lines = [head + "   NET(additive) %+7.0f z=%5.1f   survives %6.1f%%"
             % (r["net_additive"], r["z_additive"], r["survives"])]
    lines.append("      (unseen groups only. Over the WHOLE slice it reads "
                 "%.1f%%, diluted by the seen groups, which are identical in "
                 "both arms)" % r["survives_diluted"])
    if r["unseen_share"] > 0.5:
        lines.append("      *** %.0f%% of the test set is unseen, so the GLOBAL "
                     "shift is largely computed from these very groups and "
                     "corrects the raked baseline twice. Read the item counts, "
                     "not the ratio." % (100 * r["unseen_share"]))
    if r["raked"] < r["unseen"]:
        lines.append("      (%d of %d unseen groups kept the GLOBAL baseline: "
                     "%d no separator, %d unseen factor level)"
                     % (r["unseen"] - r["raked"], r["unseen"],
                        r["no_sep"], r["unseen_factor"]))
    if r["multi_tok"]:
        lines.append("      (%d raked group(s) have >2 tokens; only the FIRST "
                     "and LAST are used, the middle is dropped)" % r["multi_tok"])
    return lines


def _synthetic(tmp, sep="|", n_class=4, seed=0, n_tr=900, n_unseen=600,
               n_seen=1200):
    """A slice on which raking is EXACTLY right: the held-out cell is drawn
    from the product of the marginals the tool will actually observe. A correct
    control must absorb most of its novelty; the shipped diluted ratio must
    not."""
    rng = np.random.default_rng(seed)
    cells = {(0, 0): rng.dirichlet(np.ones(n_class) * 0.7),
             (0, 1): rng.dirichlet(np.ones(n_class) * 0.7),
             (1, 0): rng.dirichlet(np.ones(n_class) * 0.7)}
    tr = []
    for (s, a), p in cells.items():
        tr += [{"location": "s%d%sa%d" % (s, sep, a), "label": int(c)}
               for c in rng.choice(n_class, size=n_tr, p=p)]
    trf = pd.DataFrame(tr)

    def cc(f):
        o = np.zeros(n_class)
        for c, k in f["label"].value_counts().items():
            o[int(c)] = k
        return o

    p_glob = cc(trf) / len(trf)
    f0 = trf["location"].str.split(sep, regex=False).str[0]
    f1 = trf["location"].str.split(sep, regex=False).str[-1]
    q = ((cc(trf[f0 == "s1"]) / (f0 == "s1").sum())
         * (cc(trf[f1 == "a1"]) / (f1 == "a1").sum())
         / np.where(p_glob > 0, p_glob, 1))
    q = q / q.sum()
    te = [{"location": "s1%sa1" % sep, "label": int(c)}
          for c in rng.choice(n_class, size=n_unseen, p=q)]
    for cell in ((0, 0), (1, 0)):
        te += [{"location": "s%d%sa%d" % (cell[0], sep, cell[1]),
                "label": int(c)}
               for c in rng.choice(n_class, size=n_seen, p=cells[cell])]
    os.makedirs(tmp, exist_ok=True)
    trf.to_csv(os.path.join(tmp, "train_meta.csv"), index=False)
    pd.DataFrame(te).to_csv(os.path.join(tmp, "test_meta.csv"), index=False)
    return tmp


def self_test():
    import tempfile
    root = tempfile.mkdtemp()
    ok = True

    live = [control(_synthetic(os.path.join(root, "f%d" % s), seed=s), sep="|")
            for s in range(6)]
    print("\n".join(report(live[0], "liveness")))
    surv = float(np.mean([r["survives"] for r in live]))
    dil = float(np.mean([r["survives_diluted"] for r in live]))
    if any(r["raked"] != r["unseen"] or not r["raked"] for r in live):
        print("FAIL (a): the right separator did not rake every unseen group")
        ok = False
    elif surv >= 50.0:
        print("FAIL (a): raking absorbed only %.0f%% of a slice built so that "
              "raking is exact" % (100 - surv))
        ok = False
    else:
        print("  PASS (a) LIVENESS: the control RUNS and measures -- survives "
              "%.1f%% over 6 seeds on a slice where raking is exact" % surv)

    # (b) the shipped ratio must be the UNDILUTED one. On this slice the two
    #     differ by ~25 points, so a regression to the whole-slice figure is
    #     caught rather than looking like noise.
    if not (dil > surv + 10.0):
        print("FAIL (b): expected the whole-slice figure (%.1f%%) to sit well "
              "above the unseen-only one (%.1f%%)" % (dil, surv))
        ok = False
    else:
        print("  PASS (b) DILUTION: whole-slice %.1f%% vs unseen-only %.1f%% -- "
              "the headline is the undiluted number" % (dil, surv))

    dead = control(os.path.join(root, "f0"), sep="@")
    txt = "\n".join(report(dead, "wrong-sep"))
    print(txt)
    if dead["raked"] != 0:
        print("FAIL (c): a wrong separator still raked %d group(s)"
              % dead["raked"])
        ok = False
    elif not np.isnan(dead["survives"]) or "NOT A CONTROL" not in txt:
        print("FAIL (c): a wrong separator printed a survival figure")
        ok = False
    else:
        print("  PASS (c) REFUSAL: the wrong separator is REFUSED, not reported "
              "as ~100%")

    # (d) the retracted identity, kept as a gate so it cannot return as a
    #     `passed control`. Same null draws now, so this is exact.
    if dead["net_global"] != dead["net_additive"]:
        print("FAIL (d): expected the un-raked arms to coincide EXACTLY, got "
              "%+.4f vs %+.4f" % (dead["net_global"], dead["net_additive"]))
        ok = False
    else:
        print("  PASS (d) the un-raked arms are bit-identical (%+.0f items "
              "both), which is why ~100%% could never fail" % dead["net_global"])
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="slice dirs with train/test_meta.csv")
    ap.add_argument("--sep", default="|", help="factor separator in `location`")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    if not args.paths:
        ap.error("give at least one slice dir, or --self-test")
    print("FACTORIAL-GROUP CONTROL -- how much unseen-group novelty survives")
    print("once the model is credited with interpolating the two factors?")
    print("READ `raked` FIRST: with raked=0 the two baselines are the SAME")
    print("baseline and the percentage is arithmetic, not a measurement.")
    print("")
    for p in args.paths:
        r = control(p, sep=args.sep)
        name = os.path.basename(os.path.dirname(p.rstrip("/\\"))) or p
        for line in report(r, name):
            print(line)


if __name__ == "__main__":
    main()
