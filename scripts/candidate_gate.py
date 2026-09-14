"""ONE gate, every condition at once. Labels + metadata only, no images, no GPU.

Four separate tools each answered one condition -- dataset_screen (label shift),
tier_viability (group structure), task_window (does the cap bind), the split
audit (label integrity). A candidate passed one and failed another, and nothing
ever returned a single yes/no, which is why the same datasets kept coming back.

Conditions, all measured here except INTEGRITY which needs the source archive:

  C1 CLASSES   >= 8 classes in the slice
  C2 GROUPS    >= 8 held-out groups
  C3 SPREAD    the capped pair each live in >= half the groups
  C4 DENSITY   fraction of (group, class) cells that are non-empty
  C5 DEAD      share of test items in groups holding NEITHER capped class
  C6 ZEROCEIL  share of per-group ceilings that are K=0 at the cap fraction
  C7 PRIZE     items a cap actually evicts -- "is it hard enough" from labels
  C8 BALANCE   rarest/commonest class support ratio (a class too rare to cap)

C7 is the labels-only half of hardness. The other half (are there ERRORS inside
K, and is p@K < 0.99) needs a finished unconstrained run and cannot be had here.
"""
import csv, os, sys, collections

CAPFRAC = 0.80
MIN_EVICT = 10


def load(meta):
    rows = list(csv.DictReader(open(meta)))
    if not rows:
        raise SystemExit("empty: " + meta)
    ycol = next(c for c in rows[0] if c.lower() in ("label", "y", "target"))
    gcol = next(c for c in rows[0] if c.lower() in ("location", "group", "group_id", "country_code"))
    return [(int(float(r[ycol])), r[gcol]) for r in rows]


def screen(name, meta, capped=None):
    data = load(meta)
    groups = sorted({g for _, g in data})
    classes = sorted({y for y, _ in data})
    cell = collections.Counter(data)
    n_by_c = collections.Counter(y for y, _ in data)
    # spread = in how many groups does the class appear, and how concentrated
    spread = {}
    for c in classes:
        present = [g for g in groups if cell[(c, g)] > 0]
        top = max((cell[(c, g)] for g in groups), default=0) / max(1, n_by_c[c])
        spread[c] = (len(present), top)
    # pick the capped pair if not declared: the two most spread, least concentrated
    if capped is None:
        capped = [c for c in sorted(classes, key=lambda c: (-spread[c][0], spread[c][1]))[:2]]
    density = sum(1 for c in classes for g in groups if cell[(c, g)] > 0) / max(1, len(classes) * len(groups))
    ceilings = [(c, g) for c in capped for g in groups]
    zero = sum(1 for c, g in ceilings if int(cell[(c, g)] * CAPFRAC) == 0)
    dead = [g for g in groups if all(cell[(c, g)] == 0 for c in capped)]
    deaditems = sum(1 for _, g in data if g in dead)
    # C7: at a CAPFRAC cap, how many items would a perfectly-predicting model be
    # forced to drop? This is the labels-only floor on what the cap can move.
    evict = sum(max(0, cell[(c, g)] - int(cell[(c, g)] * CAPFRAC)) for c, g in ceilings)
    binding = sum(1 for c, g in ceilings
                  if cell[(c, g)] - int(cell[(c, g)] * CAPFRAC) >= MIN_EVICT)
    bal = min(n_by_c.values()) / max(1, max(n_by_c.values()))
    checks = [
        ("C1 classes>=8", len(classes) >= 8, "%d" % len(classes)),
        ("C2 groups>=8", len(groups) >= 8, "%d" % len(groups)),
        ("C3 spread>=half", all(spread[c][0] >= len(groups) / 2.0 for c in capped),
         "%s of %d" % ("/".join(str(spread[c][0]) for c in capped), len(groups))),
        ("C4 density>=.50", density >= 0.50, "%.2f" % density),
        ("C5 dead<=10%%", deaditems <= 0.10 * len(data), "%.0f%%" % (100.0 * deaditems / len(data))),
        ("C6 zeroceil<=25%%", zero <= 0.25 * len(ceilings), "%d/%d" % (zero, len(ceilings))),
        ("C7 binding ceils", binding >= max(2, len(ceilings) // 4), "%d/%d evict>=%d" % (binding, len(ceilings), MIN_EVICT)),
        ("C8 balance>=.25", bal >= 0.25, "%.2f" % bal),
    ]
    npass = sum(1 for _, ok, _ in checks if ok)
    print("")
    print("%-22s  %d items, capped %s   %s" %
          (name, len(data), list(capped), "PASS ALL" if npass == len(checks) else "%d/%d" % (npass, len(checks))))
    for label, ok, detail in checks:
        print("    %-18s %-4s %s" % (label, "ok" if ok else "FAIL", detail))
    return npass == len(checks)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", nargs="+", required=True,
                    help="test_meta.csv of each candidate slice")
    ap.add_argument("--capped", default=None,
                    help="comma-separated capped classes; default picks the most spread pair")
    ap.add_argument("--name", nargs="+", default=None,
                    help="display name per --meta, in the same order")
    args = ap.parse_args(argv)
    cap = [int(x) for x in args.capped.split(",")] if args.capped else None
    names = args.name or [os.path.basename(os.path.dirname(m)) or m for m in args.meta]
    if len(names) != len(args.meta):
        raise SystemExit("--name must give one name per --meta")
    ok = [screen(n, m, cap) for n, m in zip(names, args.meta)]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
