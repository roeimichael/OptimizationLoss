"""THE METRIC PANEL, ACROSS OPERATING POINTS -- one number cannot decide this.

`deployed_h2h` and `full_panel` each report one family. This reports four side
by side for the same runs, because they answer different questions and this
project has repeatedly read one as if it were another:

  ALLOCATION   capTP, ccF1. What is actually deployed. The outcome.
  RANKING      gAP, gAUROC (per-GROUP) and capAP, capAUROC (global).
               🔑 RANKING IS THE ONLY FAMILY THAT CAN CHANGE A TOP-K SET.
               Post-hoc allocation is optimal GIVEN the probabilities and that
               optimality is distribution-free, so the ONLY way an arm moves a
               deployed item is by re-ranking. An allocation win with no
               ranking win is an allocator artefact, not a method effect.
  CALIBRATION  negBrier, negNLL. Allocation-FREE: monotone recalibration
               cannot reorder, so these can never WIN an allocation. They
               price a cost.
  COLLATERAL   macroF1, uncapF1. What the constraint costs in the classes
               nobody capped -- structurally invisible to every capped-class
               metric.

🛑 gAP IS THE RANKING METRIC THAT MATCHES THE ALLOCATOR, capAP IS NOT.
The allocator takes top-K_gc WITHIN each group and never ranks an item against
one in another group. A global AP therefore measures an ordering the system
never uses. Both are printed because they can disagree, and this repo has paid
for the global/per-group confusion twice already (the cap screen that counted
a global top-K and overstated the prize 4.25x, and `task_window`'s p@K).

🛑 THE SIGN CONVENTION IS FORCED: higher is better in EVERY column. Brier and
NLL are negated at source and named negBrier / negNLL. A table mixing
directions is how a cost gets read as a win.

⚠️ ENSEMBLE SIZE IS AN OPERATING POINT, NOT A DETAIL. `--k 1` scores each seed
alone, paired by seed -- the protocol, and what `deployed_h2h` sees. `--k 3`
builds the C(4,3) leave-one-out ensembles of 3, paired by which seed was
dropped, and every arm ensembles its OWN seeds so it stays equal compute
ACROSS ARMS. MEASURED ON bcn1mn3: at k=1 every arm-vs-arm contrast reads 2/4,
a coin flip; at k=3 stable orderings appear. A finding present at k=3 and
absent at k=1 is a statement about variance, not about the method -- SAY WHICH.

⚠️ AND n/n IS STABILITY, NOT SIGNIFICANCE. At k=3 the replicates share two of
three members, so "4/4 positive" means no single seed carries the sign. It is
NOT p = 0.0625 and must never be quoted as one.
"""
import argparse
import collections
import csv
import glob
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METRICS = [("capTP", "ALLOC", 1.0), ("ccF1", "ALLOC", 1000.0),
           ("gAP", "RANK-g", 1000.0), ("gAUROC", "RANK-g", 1000.0),
           ("capAP", "rank-GL", 1000.0), ("capAUROC", "rank-GL", 1000.0),
           ("negBrier", "CALIB", 1000.0), ("negNLL", "CALIB", 1000.0),
           ("macroF1", "COLLAT", 1000.0), ("uncapF1", "COLLAT", 1000.0)]


def load(d):
    rows = list(csv.DictReader(io.open(
        os.path.join(d, "final_predictions_raw.csv"), encoding="utf-8",
        newline="")))
    cols = sorted(int(c.rsplit("_", 1)[1]) for c in rows[0]
                  if c.startswith("Prob_Class_"))
    y = np.array([int(float(r["True_Label"])) for r in rows])
    P = np.array([[float(r["Prob_Class_%d" % c]) for c in cols] for r in rows])
    g = np.array([r.get("Group_ID", "_") for r in rows])
    return y, P, g


def panel(y, P, g, classes, lpct):
    """Every metric from one (y, P, g). Higher is better throughout."""
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
    out = {}
    uncapped = [c for c in range(P.shape[1]) if c not in classes]

    # ---- ALLOCATION: per-group top-K, exactly what is deployed
    tp_tot, f1s = 0.0, []
    for c in classes:
        tp_c, denom = 0, 0
        for gg in np.unique(g):
            idx = np.where(g == gg)[0]
            n = int((y[idx] == c).sum())
            K = int(round(n * lpct))
            if K <= 0:
                continue
            top = idx[np.argsort(-P[idx, c])[:K]]
            tp_c += int((y[top] == c).sum())
            denom += K + n
        tp_tot += tp_c
        if denom:
            f1s.append(2.0 * tp_c / denom)
    out["capTP"] = tp_tot
    out["ccF1"] = float(np.mean(f1s)) if f1s else float("nan")

    # ---- RANKING, GLOBAL. Printed for continuity with the older scorers; it
    #      measures an ordering across groups that the allocator never uses.
    aps, aucs = [], []
    for c in classes:
        yc = (y == c).astype(int)
        if 0 < yc.sum() < len(yc):
            aps.append(average_precision_score(yc, P[:, c]))
            aucs.append(roc_auc_score(yc, P[:, c]))
    out["capAP"] = float(np.mean(aps)) if aps else float("nan")
    out["capAUROC"] = float(np.mean(aucs)) if aucs else float("nan")

    # ---- RANKING, PER GROUP. The one that matches the allocator. Weighted by
    #      POSITIVES: an unweighted mean lets a group holding two positives
    #      outvote one holding two hundred.
    apg, aucg = [], []
    for c in classes:
        na, nu, den = 0.0, 0.0, 0.0
        for gg in np.unique(g):
            idx = np.where(g == gg)[0]
            yc = (y[idx] == c).astype(int)
            npos = int(yc.sum())
            if npos == 0 or npos == len(idx):
                continue
            na += average_precision_score(yc, P[idx, c]) * npos
            nu += roc_auc_score(yc, P[idx, c]) * npos
            den += npos
        if den:
            apg.append(na / den)
            aucg.append(nu / den)
    out["gAP"] = float(np.mean(apg)) if apg else float("nan")
    out["gAUROC"] = float(np.mean(aucg)) if aucg else float("nan")

    # ---- CALIBRATION: cannot reorder, so cannot win an allocation. A COST.
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    out["negBrier"] = -float(np.mean(np.sum((P - onehot) ** 2, axis=1)))
    out["negNLL"] = -float(np.mean(-np.log(np.clip(
        P[np.arange(len(y)), y], 1e-12, None))))

    # ---- COLLATERAL: what the constraint costs where it is not looking
    pred = P.argmax(axis=1)
    out["macroF1"] = float(f1_score(y, pred, average="macro", zero_division=0))
    if uncapped:
        out["uncapF1"] = float(f1_score(
            y, pred, labels=uncapped, average="macro", zero_division=0))
    return out


def units_for(seeds, k):
    """The (label, seeds-used) list for an operating point.

    k == 1 is one seed per unit, paired by seed. Otherwise leave-one-out
    ensembles of len(seeds) - 1, paired by which seed was dropped.
    """
    if k == 1:
        return [(s, [s]) for s in seeds]
    return [(d, [s for s in seeds if s != d]) for d in seeds]


def collect(root, arms):
    cells = collections.defaultdict(lambda: collections.defaultdict(dict))
    meta = {}
    for d in sorted(glob.glob(os.path.join(root, "*", "*", "*", "*", "seed_*"))):
        if not os.path.exists(os.path.join(d, "final_predictions_raw.csv")):
            continue
        try:
            cfg = json.load(open(os.path.join(d, "config.json")))
        except (ValueError, OSError):
            continue
        if cfg.get("status") != "completed":
            continue
        arm = os.path.basename(os.path.dirname(d))
        if arms and arm not in arms:
            continue
        p = os.path.normpath(d).split(os.sep)
        key = (p[-5], p[-3])
        meta[key] = ((cfg.get("dataset_config") or {}).get("constrained_class"),
                     cfg["constraint"][0])
        cells[key][arm][int(p[-1].split("_")[1])] = d
    return cells, meta


def report(root, arms, base, k, out=sys.stdout):
    from scripts import quarantine
    quarantine.gate([root])
    cells, meta = collect(root, arms)
    w = out.write
    w("ensemble k=%d. %s. Higher is better in every column; Brier and NLL are\n"
      % (k, "each seed scored alone, paired by seed" if k == 1
         else "leave-one-out ensembles, paired by dropped seed"))
    w("negated. ccF1 onward are x1000; capTP is ITEMS. n/n is STABILITY, not\n")
    w("significance -- at k>1 the replicates share members.\n")
    for key in sorted(cells):
        classes, lpct = meta[key]
        present = [a for a in arms if cells[key].get(a)] or sorted(cells[key])
        seeds = sorted(set.intersection(*[set(cells[key][a]) for a in present]))
        if len(seeds) < max(2, k):
            continue
        units = units_for(seeds, k)
        vals = collections.defaultdict(dict)
        for arm in present:
            for label, use in units:
                acc, y0, g0 = None, None, None
                for s in use:
                    y, P, g = load(cells[key][arm][s])
                    acc = P.copy() if acc is None else acc + P
                    y0, g0 = y, g
                vals[arm][label] = panel(y0, acc / len(use), g0, classes, lpct)
        w("\n" + "=" * 112 + "\n")
        w("%s   %s   seeds %s   k=%d\n" % (key[0], key[1], seeds, k))
        w("=" * 112 + "\n")
        w("%-16s" % "arm" + "".join(" %10s" % m for m, _, _ in METRICS) + "\n")
        w("%-16s" % "" + "".join(" %10s" % f for _, f, _ in METRICS) + "\n")
        w("-" * 112 + "\n")
        for arm in present:
            line = "%-16s" % arm
            for m, _, sc in METRICS:
                line += " %10.1f" % np.mean(
                    [vals[arm][u].get(m, float("nan")) * sc for u, _ in units])
            w(line + "\n")
        if base not in vals:
            continue
        w("\nCONTRASTS vs %s, paired   (mean, n/%d positive)\n"
          % (base, len(units)))
        w("-" * 112 + "\n")
        for b in present:
            if b == base:
                continue
            line = "%-16s" % ("%s-%s" % (base, b))[:16]
            for m, _, sc in METRICS:
                dd = [(vals[base][u].get(m, float("nan"))
                       - vals[b][u].get(m, float("nan"))) * sc
                      for u, _ in units]
                line += " %6.1f %d/%d" % (np.mean(dd),
                                          sum(1 for x in dd if x > 0), len(dd))
            w(line + "\n")
    return 0


def self_test(out=sys.stdout):
    checks = []
    rng = np.random.RandomState(0)

    # Two groups, one capped class. Group A is easy, group B is hard.
    y = np.array([1] * 20 + [0] * 20 + [1] * 20 + [0] * 20)
    g = np.array(["A"] * 40 + ["B"] * 40)

    def probs(sep_a, sep_b):
        p1 = np.concatenate([
            rng.rand(20) * 0.4 + sep_a, rng.rand(20) * 0.4,
            rng.rand(20) * 0.4 + sep_b, rng.rand(20) * 0.4])
        p1 = np.clip(p1, 0.01, 0.99)
        return np.stack([1 - p1, p1], axis=1)

    good = probs(0.55, 0.55)
    bad = probs(0.05, 0.05)
    rg = panel(y, good, g, [1], 0.5)
    rb = panel(y, bad, g, [1], 0.5)
    checks.append(("a better-separated arm wins gAP", rg["gAP"] > rb["gAP"]))
    checks.append(("...and wins deployed capTP", rg["capTP"] > rb["capTP"]))

    # NEGATIVE CONTROL: identical inputs must give an exactly zero contrast.
    r2 = panel(y, good.copy(), g, [1], 0.5)
    checks.append(("NEGATIVE CONTROL: identical probabilities -> every metric "
                   "identical",
                   all(abs(rg[m] - r2[m]) < 1e-12 for m in rg)))

    # 🛑 THE CONTROL FOR gAP ITSELF. Swap the two groups' score RANGES so the
    # global ordering is destroyed while every WITHIN-group ordering is
    # untouched. gAP must not move; capAP must fall. A gAP that tracks capAP
    # here is just capAP under another name.
    shifted = good.copy()
    b = np.arange(40, 80)
    shifted[b, 1] = shifted[b, 1] * 0.02          # group B pushed far below A
    shifted[b, 0] = 1 - shifted[b, 1]
    rs = panel(y, shifted, g, [1], 0.5)
    checks.append(("gAP is INVARIANT to a between-group shift that preserves "
                   "within-group order", abs(rs["gAP"] - rg["gAP"]) < 1e-9))
    checks.append(("NEGATIVE CONTROL: the GLOBAL capAP DOES move on that same "
                   "shift", rs["capAP"] < rg["capAP"] - 1e-6))
    checks.append(("...and deployed capTP is invariant too (allocation is "
                   "per-group)", rs["capTP"] == rg["capTP"]))

    # Sign convention: a deliberately worse-calibrated arm must read LOWER.
    mushy = np.full_like(good, 0.5)
    rm = panel(y, mushy, g, [1], 0.5)
    checks.append(("negBrier is a COST: the confident-correct arm reads higher",
                   rg["negBrier"] > rm["negBrier"]))
    checks.append(("negNLL likewise", rg["negNLL"] > rm["negNLL"]))

    # Operating points
    checks.append(("k=1 gives one unit per seed",
                   units_for([1, 2, 3, 4], 1) == [(1, [1]), (2, [2]),
                                                  (3, [3]), (4, [4])]))
    u3 = units_for([1, 2, 3, 4], 3)
    checks.append(("k=3 gives C(4,3)=4 leave-one-out units of 3",
                   len(u3) == 4 and all(len(x[1]) == 3 for x in u3)
                   and all(x[0] not in x[1] for x in u3)))

    # K rounding must match the pipeline: a cap rounding a nonzero count to 0
    # contributes nothing rather than emitting a phantom item.
    ytiny = np.array([1, 0, 0, 0])
    gtiny = np.array(["A"] * 4)
    rt = panel(ytiny, np.stack([np.full(4, 0.5)] * 2, axis=1), gtiny, [1], 0.2)
    checks.append(("a cap rounding K to 0 contributes no TP",
                   rt["capTP"] == 0))

    print("", file=out)
    for label, ok in checks:
        print("  %-70s %s" % (label[:70], "PASS" if ok else "FAIL"), file=out)
    bad_n = [c for c, ok in checks if not ok]
    print("", file=out)
    print("SELF-TEST PASSED" if not bad_n else "FAILED: %d" % len(bad_n),
          file=out)
    return 1 if bad_n else 0


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("root", nargs="?")
    a.add_argument("--arms", nargs="*", default=[])
    a.add_argument("--base", default="tralo",
                   help="arm every contrast is taken against (default tralo)")
    a.add_argument("--k", type=int, default=1,
                   help="ensemble size. 1 = the protocol operating point, one "
                        "seed per unit. >1 = leave-one-out ensembles.")
    a.add_argument("--self-test", action="store_true")
    args = a.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.root:
        a.error("give a campaign root (or --self-test)")
    return report(args.root, args.arms, args.base, args.k)


if __name__ == "__main__":
    sys.exit(main())
