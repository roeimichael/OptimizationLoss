"""IS THE LOCAL SCOPE A TIER STRUCTURE, OR JUST A SPARSITY PATTERN?

`dataset_screen` asks whether per-group LABEL SHIFT exists. That is necessary
and it is not sufficient, and iwildcam is the proof: it scores z = 96.3, the
best of 21 candidates, and 6 of its 8 classes still cannot carry a local
constraint at all.

THE PROPERTY THAT MATTERS. A per-group count cap is a real allocation decision
only when the class could plausibly appear in more than one group -- the way a
hospital's gold/silver/bronze tiers are a real decision because any patient
could land in any tier. If a class occurs at exactly ONE group, then

    "at most K of class c in each group"   ==   "at most K of class c overall"

and the LOCAL scope has silently collapsed onto the GLOBAL one. The experiment
still runs, still reports numbers, and measures a global cap twice.

MEASURED ON iwildcam/oodslice (2026-09-07): 8 classes, 7 groups. Classes 0, 3,
4 and 5 each live at ONE camera; 1 and 6 are 92% and 81% concentrated. Only
classes 2 and 7 are genuinely spread (max-group share 43% and 50%) -- which is
why the protocol caps exactly those two, and why 7 of the 14 per-group ceilings
are K = 0. The single largest group (camera 218, 1657 items = 56% of the test
set) contains ZERO of both capped classes, so it contributes two ceilings that
are satisfied before training starts and can never change an emitted item.

WHAT THIS REPORTS, per slice:
  density     fraction of (class, group) cells that are non-empty. 1.00 is a
              tier structure; iwildcam is the sparse end.
  usable      classes at >= MIN_GROUPS groups AND below MAX_CONCENTRATION at
              their biggest one. These are the only classes a local cap can
              constrain independently of the global one.
  zero_ceil   at cap fraction f, the share of (group, class) ceilings that come
              out K = 0. This is the defect in its most direct form: a K = 0
              ceiling on a group holding none of that class is satisfied at
              initialisation and forever.
  dead_share  fraction of TEST ITEMS sitting in groups that hold none of the
              usable classes -- items the local scope can never act on.
  off_prop    total-variation distance between the observed (group x class)
              matrix and the product of its marginals. ZERO means every group
              is a scaled copy of the global label mix, so a per-group cap IS
              the global cap divided by group size and the local scope adds
              nothing at all. 🛑 DENSITY CANNOT SEE THIS: `domainnet` reads
              density 1.00, 8/8 usable, 0% zero ceilings -- and 2.4%
              off-proportional with per-group label shift at z = 1.2, i.e.
              dead. Found by the dataset audit 2026-09-08, after this tool
              had already passed it.

    python -m scripts.tier_viability <slice-dir> [more ...] [--cap 0.7]
    python -m scripts.tier_viability --self-test
"""
import argparse
import glob
import os
import sys

MIN_GROUPS = 2          # a class at one group has no local scope at all
MAX_CONCENTRATION = 0.80  # >80% at one group is local-in-name-only

GROUP_COLS = ("location", "group", "group_id", "country", "site", "institution",
              "source", "camera", "skin_type", "synth_group")


def read_meta(path):
    import pandas as pd
    df = pd.read_csv(path)
    gcol = next((c for c in GROUP_COLS if c in df.columns), None)
    if gcol is None:
        cand = [c for c in df.columns
                if c not in ("label", "class_name", "filename")
                and 1 < df[c].nunique() < max(2, len(df) // 4)]
        gcol = cand[0] if cand else None
    if gcol is None or "label" not in df.columns:
        return None, None
    return df, gcol


def analyse(df, gcol, cap):
    labels = sorted(df["label"].unique())
    groups = sorted(df[gcol].unique())
    counts = {}
    for c in labels:
        sub = df[df["label"] == c]
        vc = sub[gcol].value_counts()
        counts[c] = (len(sub), vc)

    cells = len(labels) * len(groups)
    nonempty = sum(len(vc) for _, vc in counts.values())

    usable = []
    for c in labels:
        n, vc = counts[c]
        if n == 0 or len(vc) < MIN_GROUPS:
            continue
        if vc.iloc[0] / float(n) > MAX_CONCENTRATION:
            continue
        usable.append(c)

    # ceilings at this cap, over the USABLE classes -- what a campaign would build
    zero, total = 0, 0
    for c in usable:
        _n, vc = counts[c]
        for g in groups:
            total += 1
            if int(round(cap * int(vc.get(g, 0)))) == 0:
                zero += 1

    # OFF-PROPORTIONAL: total-variation distance between the observed
    # (group x class) matrix and the product of its marginals, as a share of
    # items. Zero means each group is a scaled copy of the global label
    # distribution -- so a per-group cap IS the global cap divided by group
    # size and the local scope adds nothing. Density cannot see this:
    # `domainnet` reads density 1.00 and off_proportional 2.4%.
    n_tot = float(len(df))
    off = 0.0
    if n_tot:
        gsz = {g: float((df[gcol] == g).sum()) for g in groups}
        csz = {c: float((df["label"] == c).sum()) for c in labels}
        for g in groups:
            sub = df[df[gcol] == g]
            for c in labels:
                obs = float((sub["label"] == c).sum())
                exp = gsz[g] * csz[c] / n_tot
                off += abs(obs - exp)
        off = off / 2.0 / n_tot

    dead_items = 0
    for g in groups:
        sub = df[df[gcol] == g]
        if not any((sub["label"] == c).any() for c in usable):
            dead_items += len(sub)

    return {
        "classes": len(labels), "groups": len(groups),
        "density": nonempty / float(cells) if cells else 0.0,
        "usable": usable,
        "zero_ceil": (zero / float(total)) if total else float("nan"),
        "dead_share": dead_items / float(len(df)) if len(df) else 0.0,
        "off_prop": off,
    }


# Below this, each group is close enough to a scaled copy of the global label
# distribution that a per-group cap is the global cap divided by group size.
# `domainnet` sits at 0.024 with density 1.00, which is why density alone is
# not a sufficient gate.
MIN_OFF_PROPORTIONAL = 0.05


def verdict(r):
    if len(r["usable"]) < 2:
        return "DEAD      fewer than 2 classes can carry a local cap"
    if r.get("off_prop", 1.0) < MIN_OFF_PROPORTIONAL:
        return ("DEAD      groups are proportional copies of the global mix "
                "(off_prop %.1f%%)" % (100 * r["off_prop"]))
    if r["zero_ceil"] >= 0.50:
        return "WEAK      half the ceilings are K=0 before training"
    if r["zero_ceil"] >= 0.25 or r["density"] < 0.50:
        return "PARTIAL   local scope is sparse, not a tier structure"
    return "TIER-LIKE every usable class is spread across groups"


def run(paths, cap, out=sys.stdout):
    rows = []
    for p in paths:
        meta = (p if p.endswith(".csv")
                else os.path.join(p, "test_meta.csv"))
        if not os.path.exists(meta):
            hits = glob.glob(os.path.join(p, "**", "test_meta.csv"), recursive=True)
            if not hits:
                continue
            meta = hits[0]
        df, gcol = read_meta(meta)
        if df is None:
            out.write("  %-26s NO USABLE label/group columns\n" % os.path.basename(p))
            continue
        r = analyse(df, gcol, cap)
        r["name"] = os.path.basename(os.path.dirname(meta)) or os.path.basename(meta)
        if r["name"] in ("oodslice", "slice_1"):
            r["name"] = os.path.basename(os.path.dirname(os.path.dirname(meta)))
        r["gcol"] = gcol
        rows.append(r)

    rows.sort(key=lambda r: (-len(r["usable"]), r["zero_ceil"]))
    out.write("=" * 100 + "\n")
    out.write("TIER VIABILITY -- can a LOCAL cap say anything a GLOBAL cap cannot?"
              "   (cap fraction %.2f)\n" % cap)
    out.write("  usable = class at >=%d groups and <%d%% at its biggest one\n"
              % (MIN_GROUPS, int(100 * MAX_CONCENTRATION)))
    out.write("=" * 100 + "\n")
    out.write("%-22s %-11s %4s %4s %8s %7s %9s %6s %6s  %s\n"
              % ("slice", "group by", "cls", "grp", "density", "usable",
                 "zero_ceil", "dead", "offprp", "verdict"))
    out.write("-" * 100 + "\n")
    for r in rows:
        out.write("%-22s %-11s %4d %4d %7.2f %7d %8.0f%% %5.0f%% %5.1f%%  %s\n"
                  % (r["name"][:22], str(r["gcol"])[:11], r["classes"], r["groups"],
                     r["density"], len(r["usable"]),
                     100 * r["zero_ceil"] if r["zero_ceil"] == r["zero_ceil"] else 0,
                     100 * r["dead_share"], 100 * r["off_prop"], verdict(r)))
    return rows


def self_test():
    import pandas as pd
    ok = []

    def check(name, cond):
        ok.append((name, bool(cond)))

    # a TIER structure: every class in every group, evenly
    tier = pd.DataFrame({"label": [c for c in range(4) for _ in range(40)],
                         "location": [g for _ in range(4) for g in range(4)
                                      for _ in range(10)]})
    r = analyse(tier, "location", 0.7)
    check("tier: density 1.0", abs(r["density"] - 1.0) < 1e-9)
    check("tier: all 4 usable", len(r["usable"]) == 4)
    check("tier: no zero ceilings", r["zero_ceil"] == 0.0)
    # NB: `tier` is dense AND proportional, so its verdict is DEAD on the
    # off_prop gate -- asserted below. Density and usability still read right.

    # NEGATIVE CONTROL: the iwildcam shape -- each class at exactly one group
    diag = pd.DataFrame({"label": [c for c in range(4) for _ in range(40)],
                         "location": [c for c in range(4) for _ in range(40)]})
    r = analyse(diag, "location", 0.7)
    check("diagonal: density 0.25", abs(r["density"] - 0.25) < 1e-9)
    check("diagonal: ZERO usable classes", len(r["usable"]) == 0)
    check("diagonal: verdict DEAD", verdict(r).startswith("DEAD"))

    # NEGATIVE CONTROL: spread but 90% concentrated -> local in name only
    conc = pd.DataFrame({"label": [0] * 100 + [1] * 100,
                         "location": ["a"] * 90 + ["b"] * 10 + ["a"] * 90 + ["b"] * 10})
    r = analyse(conc, "location", 0.7)
    check("concentrated: 0 usable at 90%", len(r["usable"]) == 0)

    # NEGATIVE CONTROL: a big group holding none of the usable classes
    dead = pd.DataFrame({
        "label": [0] * 20 + [1] * 20 + [2] * 160,
        "location": (["a"] * 10 + ["b"] * 10) + (["a"] * 10 + ["b"] * 10)
                    + ["z"] * 160})
    r = analyse(dead, "location", 0.7)
    check("dead group detected", r["dead_share"] > 0.7)
    check("dead group makes ceilings zero", r["zero_ceil"] > 0.3)

    # NEGATIVE CONTROL: dense AND proportional -> the local scope is the global
    # one divided by group size. `tier` above is exactly this shape.
    r = analyse(tier, "location", 0.7)
    check("proportional: off_prop ~ 0", r["off_prop"] < 0.01)
    check("proportional: verdict DEAD", verdict(r).startswith("DEAD"))

    # POSITIVE: same density, but the groups carry DIFFERENT label mixes
    shifted = pd.DataFrame({
        "label": ([0] * 30 + [1] * 10) + ([0] * 10 + [1] * 30),
        "location": ["a"] * 40 + ["b"] * 40})
    r = analyse(shifted, "location", 0.7)
    check("shifted: off_prop large", r["off_prop"] > 0.20)
    check("shifted: density 1.0", abs(r["density"] - 1.0) < 1e-9)
    check("shifted: NOT dead", not verdict(r).startswith("DEAD"))

    # a cap so small every ceiling rounds to zero
    r = analyse(tier, "location", 0.01)
    check("tiny cap -> all ceilings zero", r["zero_ceil"] == 1.0)

    for name, good in ok:
        print("  %s %s" % ("ok  " if good else "FAIL", name))
    bad = [n for n, g in ok if not g]
    print("%d checks, %d failed" % (len(ok), len(bad)))
    return 1 if bad else 0


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("slices", nargs="*", help="slice dirs or test_meta.csv paths")
    a.add_argument("--cap", type=float, default=0.7,
                   help="cap fraction used to count K=0 ceilings (default 0.7)")
    a.add_argument("--self-test", action="store_true")
    args = a.parse_args()
    if args.self_test:
        return self_test()
    if not args.slices:
        a.error("give at least one slice directory")
    rows = run(args.slices, args.cap)
    return 0 if any(len(r["usable"]) >= 2 for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
