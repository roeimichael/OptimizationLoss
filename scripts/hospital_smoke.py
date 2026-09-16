"""The hospital smoke test: does a cap express an ALLOCATION POLICY?

THE PROBLEM THIS EXISTS TO CATCH. Shifman et al. (2025) Eq. (2) takes
Phi[lambda][i] -- the ceiling for class i inside group lambda -- as a GIVEN
bound. The worked example is a hospital: 100 normal beds and 200 special beds,
split across membership tiers 10/30/60 and 20/60/120. Those numbers are a
policy. They are not a property of who happens to have applied.

Until 2026-09-16 this pipeline could only derive Phi as a fraction of each
group's OWN true count, so the ceiling was proportional to prevalence and the
policy was inexpressible. This script builds a pool where the two derivations
give DIFFERENT answers and prints them side by side, so the difference is
visible rather than argued.

The pool is built so the tiers are near-equal in size (334/333/333) while the
policy deliberately favours the tier with the FEWEST candidates -- which is
exactly the case a prevalence-derived cap gets backwards.

    python scripts/hospital_smoke.py                 # print the tables
    python scripts/hospital_smoke.py --json out.json # machine-readable
    python scripts/hospital_smoke.py --write DIR     # emit meta + labels
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.constraints import (  # noqa: E402
    compute_global_constraints,
    compute_local_constraints,
)
from src.utils.constants import UNLIMITED  # noqa: E402

CLASSES = {0: "outpatient", 1: "normal_bed", 2: "special_bed"}
CAPPED = [1, 2]
# Integer-coded on purpose: `_encode_groups` passes an already-integer column
# through unchanged, so these ids survive into every cached artefact. A string
# column is factorised by sorted unique value, which would make "gold" group 0.
TIERS = {0: "silver", 1: "gold", 2: "premium"}
# THE POLICY: share of each class's bed budget, by tier. This is the input the
# paper treats as given, and the thing prevalence cannot reproduce.
SHARES = {0: 0.10, 1: 0.30, 2: 0.60}
# tier -> class -> number of patients. Anti-correlated with SHARES on purpose:
# the tier entitled to 60% of the beds contributes the fewest candidates.
POOL = {
    0: {0: 14, 1: 120, 2: 200},
    1: {0: 163, 1: 50, 2: 120},
    2: {0: 223, 1: 30, 2: 80},
}
GROUP_COL = "tier"


def build_pool():
    rows = []
    for tier, by_class in POOL.items():
        for cls, n in by_class.items():
            rows.extend([(cls, tier)] * n)
    frame = pd.DataFrame(rows, columns=["label", GROUP_COL])
    return frame.sort_values(["label", GROUP_COL], kind="stable").reset_index(drop=True)


def k(value):
    return "-" if value == UNLIMITED else str(int(value))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local", type=float, default=0.50, help="L, the local percentage")
    ap.add_argument("--global", dest="glob", type=float, default=0.50, help="G")
    ap.add_argument("--json", help="write the measured tables here")
    ap.add_argument("--write", help="emit test_meta.csv + test_labels.npy here")
    a = ap.parse_args()

    frame = build_pool()
    kw = dict(constrained_class=CAPPED, num_classes=len(CLASSES))

    glob_con = compute_global_constraints(frame, "label", a.glob, **kw)
    policy = compute_local_constraints(
        frame, "label", a.local, GROUP_COL, group_budget_shares=SHARES, **kw
    )
    legacy = compute_local_constraints(frame, "label", a.local, GROUP_COL, **kw)

    print("")
    print("HOSPITAL SMOKE TEST   L%d_G%d" % (round(a.local * 100), round(a.glob * 100)))
    print("=" * 78)
    print("pool: %d patients, %d tiers, %d classes (%d capped)"
          % (len(frame), len(TIERS), len(CLASSES), len(CAPPED)))
    print("")
    print("  %-10s %8s   %s" % ("tier", "patients", "  ".join("%12s" % CLASSES[c] for c in sorted(CLASSES))))
    for g in sorted(TIERS):
        counts = [int(((frame["label"] == c) & (frame[GROUP_COL] == g)).sum()) for c in sorted(CLASSES)]
        print("  %-10s %8d   %s" % (TIERS[g], int((frame[GROUP_COL] == g).sum()),
                                    "  ".join("%12d" % v for v in counts)))
    print("  %-10s %8d   %s" % ("TOTAL", len(frame),
                                "  ".join("%12d" % int((frame["label"] == c).sum()) for c in sorted(CLASSES))))

    print("")
    print("GLOBAL ceiling Psi(i)  --  the beds that exist")
    print("-" * 78)
    for c in CAPPED:
        print("  %-14s %s" % (CLASSES[c], k(glob_con[c])))

    print("")
    print("LOCAL ceilings Phi[tier][i]  --  who the beds are reserved for")
    print("-" * 78)
    print("  %-26s %s" % ("", "  ".join("%12s" % TIERS[g] for g in sorted(TIERS))))
    ok = True
    for c in CAPPED:
        want_total = int(glob_con[c])
        pol = [int(policy[g][c]) for g in sorted(TIERS)]
        leg = [int(legacy[g][c]) for g in sorted(TIERS)]
        print("  %-26s %s" % ("%s  POLICY (shares)" % CLASSES[c],
                              "  ".join("%12d" % v for v in pol)))
        print("  %-26s %s   sum %d vs global %d  %s"
              % ("", "", sum(pol), want_total, "OK" if sum(pol) == want_total else "MISMATCH"))
        print("  %-26s %s" % ("%s  legacy (prevalence)" % CLASSES[c],
                              "  ".join("%12d" % v for v in leg)))
        print("  %-26s %s   sum %d" % ("", "", sum(leg)))
        ok = ok and sum(pol) == want_total

    print("")
    print("EXPECTED, from the policy alone")
    print("-" * 78)
    expected = {}
    for c in CAPPED:
        total = int(glob_con[c])
        expected[c] = {g: int(round(total * SHARES[g])) for g in sorted(TIERS)}
        got = {g: int(policy[g][c]) for g in sorted(TIERS)}
        match = got == expected[c]
        ok = ok and match
        print("  %-14s want %s  got %s  %s"
              % (CLASSES[c],
                 "/".join(str(expected[c][g]) for g in sorted(TIERS)),
                 "/".join(str(got[g]) for g in sorted(TIERS)),
                 "OK" if match else "MISMATCH"))

    print("")
    print("VERDICT: %s" % ("PASS" if ok else "FAIL"))

    if a.json:
        payload = dict(
            local_pct=a.local, global_pct=a.glob,
            classes=CLASSES, capped=CAPPED, tiers=TIERS, shares=SHARES,
            counts={str(g): {str(c): int(((frame["label"] == c) & (frame[GROUP_COL] == g)).sum())
                             for c in sorted(CLASSES)} for g in sorted(TIERS)},
            psi={str(c): int(glob_con[c]) for c in CAPPED},
            phi_policy={str(g): {str(c): int(policy[g][c]) for c in CAPPED} for g in sorted(TIERS)},
            phi_legacy={str(g): {str(c): int(legacy[g][c]) for c in CAPPED} for g in sorted(TIERS)},
            passed=bool(ok),
        )
        with open(a.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print("wrote %s" % a.json)

    if a.write:
        os.makedirs(a.write, exist_ok=True)
        frame.to_csv(os.path.join(a.write, "test_meta.csv"), index=False)
        np.save(os.path.join(a.write, "test_labels.npy"),
                frame["label"].to_numpy(dtype=np.int64))
        print("wrote %s" % a.write)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
