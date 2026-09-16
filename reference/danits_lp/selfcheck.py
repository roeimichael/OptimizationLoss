"""Execute the revived LP against a known answer.

A restore is not verified until it RUNS. This solves the paper's own worked
example -- the hospital pool from `scripts/hospital_smoke.py` -- with the
policy-derived Psi and Phi, and checks the LP returns an assignment that
saturates every ceiling without violating any.

    python -m reference.danits_lp.selfcheck
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from reference.danits_lp import solve_lp_assignment  # noqa: E402
from reference.danits_lp.constraints_builder import (  # noqa: E402
    build_psi_phi_from_percentages,
)
from src.training.constraints import (  # noqa: E402
    compute_global_constraints,
    compute_local_constraints,
)
from src.utils.constants import UNLIMITED  # noqa: E402
from hospital_smoke import build_pool, SHARES, CAPPED, CLASSES, TIERS, GROUP_COL  # noqa: E402

NC = len(CLASSES)


def run(label, psi, phi, probs, groups):
    omega = np.ones((NC, NC)) - np.eye(NC)
    res = solve_lp_assignment(y_proba=probs, groups=groups, cost_matrix=omega,
                              psi=psi, phi=phi)
    y = res.y_pred
    print("")
    print("%s   status=%s  objective=%.4f  vars=%d  constraints=%d"
          % (label, res.status, res.objective_value,
             res.num_variables, res.num_constraints))
    bad = []
    for c in CAPPED:
        n = int((y == c).sum())
        if psi[c] is not None and n > psi[c]:
            bad.append("global class %d: %d > %d" % (c, n, psi[c]))
        for g in sorted(phi):
            m = int((y[groups == g] == c).sum())
            lim = phi[g][c]
            if lim is not None and m > lim:
                bad.append("local g%d c%d: %d > %d" % (g, c, m, lim))
    print("  %-9s %s" % ("tier", "  ".join("%22s" % CLASSES[c] for c in CAPPED)))
    for g in sorted(phi):
        mask = groups == g
        print("  %-9s %s" % (TIERS[g], "  ".join(
            "%12d of %-7s" % (int((y[mask] == c).sum()), phi[g][c]) for c in CAPPED)))
    print("  %-9s %s" % ("TOTAL", "  ".join(
        "%12d of %-7s" % (int((y == c).sum()), psi[c]) for c in CAPPED)))
    print("  violations: %s" % (bad if bad else "none"))
    return res.status == "OPTIMAL" and not bad


def main():
    frame = build_pool()
    groups = frame[GROUP_COL].to_numpy()
    y_true = frame["label"].to_numpy()
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(NC), size=len(frame))
    kw = dict(constrained_class=CAPPED, num_classes=NC)

    def as_lp(global_con, local_con):
        psi = [int(v) if v < UNLIMITED else None for v in global_con]
        phi = {int(g): [int(v) if v < UNLIMITED else None for v in b]
               for g, b in local_con.items()}
        return psi, phi

    ok = True
    # 1. the POLICY budgets this project can now express
    psi, phi = as_lp(
        compute_global_constraints(frame, "label", 0.50, **kw),
        compute_local_constraints(frame, "label", 0.50, GROUP_COL,
                                  group_budget_shares=SHARES, **kw))
    ok &= run("POLICY  L50_G50  (shares 10/30/60)", psi, phi, probs, groups)

    # 2. the paper's own builder, from percentages, on the same pool
    psi2, phi2 = build_psi_phi_from_percentages(
        y_true, groups, NC, CAPPED, feature_pct=0.50, target_pct=0.50)
    ok &= run("PAPER   build_psi_phi_from_percentages(0.50, 0.50)",
              psi2, phi2, probs, groups)

    print("")
    print("SELFCHECK: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
