"""TraLO and ALM weight the SAME scope-state in OPPOSITE directions.

Both methods reduce to a per-scope scalar A_S that multiplies p_ic(1-p_ic) at
every logit. Under `constraint_grad_mode: normalize` the step norm is fixed, so
the RATIOS of A_S across scopes are the entire degree of freedom.

    ALM     src/methodologies/fioretto_alm/train.py:244,253
            loss += w_S * proba[:, c].sum(),   w_S = lambda_S + mu_t * r_S
            => A_S = lambda_S + mu_t * r_S,    r_S = soft_S - K_S  IN RAW ITEMS
            No division by K anywhere. Big-K scopes that are many items over
            get proportionally more weight.

    TraLO   src/losses/transductive_loss.py:420-428
            pen = E/(E+s) + rho*e^2/(1+e^2),   E = relu(soft-K), s = max(K,1), e = E/s
            => A_S = lambda_S * [ s/(E+s)^2 + rho*2e/(s*(1+e^2)^2) ]
            The excess is made DIMENSIONLESS by dividing by max(K,1). A scope
            with K=0 therefore measures its excess in ABSOLUTE items while a
            scope with K=333 measures it in PERCENT, and the small-K scope wins
            by orders of magnitude.

`s = max(K,1)` exists to stop the K=0 penalty pinning at its own bound with zero
gradient (see the _penalty docstring). It is the identity for every K >= 1, so
it is invisible on a dataset with no zero ceilings. On iwildcam SEVEN of the
FOURTEEN per-group ceilings are K = 0.

⚠️ ALL THREE SHIPPED SHAPES CARRY THE SAME DENOMINATOR. `linear` returns e and
`squared` returns e**2, both of which are E/scale. So the penalty-shape ablation
already in the rejected ledger could not have caught this: it varied the
numerator and never the denominator.

This script evaluates BOTH formulas on the SAME logged scope-states and reports
where each would send the step.

⚠️ WHAT THIS IS NOT: ALM's own lambda is path-dependent and ALM trained a
different model, so this is not a replay of ALM's run. It is a counterfactual
WEIGHTING of one fixed set of scope-states by two rules, which is exactly the
comparison the `normalize` algebra makes decisive.

    python scale_inversion.py <campaign-root> [more roots ...]
"""
import collections
import csv
import glob
import json
import os
import re
import sys

GLOBAL_RE = re.compile(r"^(Limit|Hard|Soft)_Class(\d+)$")
LOCAL_RE = re.compile(r"^Group(\d+)_(Hard|Soft|Limit)_Class(\d+)$")
# NOT re-derived here. `metrics.py` once declared a local 1e9 against the
# rest of the codebase's 1e10, so a constraint set to UNLIMITED was skipped
# by the loss and counted as ACTIVE by the metric layer. Four analysis
# scripts then reintroduced the same literal. gated in test_pipeline.py.
from src.utils.constants import UNLIMITED


def scopes_of(header):
    out = collections.defaultdict(dict)
    for col in header:
        m = GLOBAL_RE.match(col)
        if m:
            out[("global", None, int(m.group(2)))][m.group(1).lower()] = col
            continue
        m = LOCAL_RE.match(col)
        if m:
            out[("local", int(m.group(1)), int(m.group(3)))][m.group(2).lower()] = col
    return {k: v for k, v in out.items() if {"limit", "hard", "soft"} <= set(v)}


def tralo_slope(soft, K, rho, item_scale=False):
    E = max(0.0, soft - K)
    if E <= 0.0:
        return 0.0
    s = K if K >= 1 else 1.0
    e = E / s
    sl = s / ((E + s) ** 2) + rho * (2.0 * e) / (s * ((1.0 + e * e) ** 2))
    return sl * s if item_scale else sl


def fnum(row, col):
    try:
        return float(row[col])
    except (KeyError, TypeError, ValueError):
        return None


def main(roots):
    # (K-class) -> summed weight under each rule
    tralo_w = collections.Counter()
    fixed_w = collections.Counter()
    st_w = collections.Counter()
    both_w = collections.Counter()
    alm_w = collections.Counter()
    n_ep = collections.Counter()
    examples = {}
    runs = 0

    for root in roots:
        for run in sorted(glob.glob(os.path.join(root, "*", "*", "*", "*", "seed_*"))):
            arm = os.path.normpath(run).split(os.sep)[-2]
            if arm != "tralo":
                continue
            try:
                cfg = json.load(open(os.path.join(run, "config.json")))
            except (IOError, ValueError):
                continue
            hp = cfg.get("hyperparams", {})
            if str(hp.get("lambda_ratchet_mode", "constant")) != "constant":
                continue
            lam_g0 = float(hp.get("lambda_global", 0.0))
            lam_l0 = float(hp.get("lambda_local", 0.0))
            lam_step = float(hp.get("lambda_step", 0.0))
            rho0 = float(hp.get("initial_rho", 0.0))
            rho_step = float(hp.get("rho_step", 0.0))
            rho_tgt = float(hp.get("rho_target", 0.0))
            # ALM's own knobs, read from an ALM sibling if present
            eta, mu0, mu_step = 0.01, 0.1, 0.1
            alm_dir = os.path.join(os.path.dirname(os.path.dirname(run)), "alm")
            for sib in glob.glob(os.path.join(alm_dir, "seed_*", "config.json")):
                try:
                    ahp = json.load(open(sib)).get("hyperparams", {})
                except (IOError, ValueError):
                    continue
                eta = float(ahp.get("alm_eta", ahp.get("fioretto_step_size", eta)))
                mu0 = float(ahp.get("alm_mu0", mu0))
                mu_step = float(ahp.get("alm_mu_step", mu_step))
                break

            path = os.path.join(run, "training_log.csv")
            if not os.path.exists(path):
                continue
            with open(path, newline="") as fh:
                rd = csv.DictReader(fh)
                if not rd.fieldnames:
                    continue
                scopes = scopes_of(rd.fieldnames)
                rows = list(rd)
            if not scopes:
                continue
            runs += 1

            viol = collections.Counter()
            alm_lam = collections.Counter()
            rho, frozen = rho0, False
            for t, row in enumerate(rows):
                mu = mu0 + mu_step * t
                for scope, cols in scopes.items():
                    K = fnum(row, cols["limit"])
                    hard = fnum(row, cols["hard"])
                    soft = fnum(row, cols["soft"])
                    if K is None or hard is None or soft is None or K >= UNLIMITED:
                        continue
                    r = soft - K
                    lam0 = lam_g0 if scope[0] == "global" else lam_l0
                    lam = lam0 + lam_step * viol[scope]
                    at = lam * tralo_slope(soft, K, rho)
                    ft = lam * tralo_slope(soft, K, rho, item_scale=True)
                    # `straight_through` swaps the penalty's ARGUMENT to the
                    # hard count (tralo/train.py:443-459), so relu(hard-K) is
                    # exactly 0 for a K=0 scope at hard 0 -- and also for every
                    # hard-feasible K>=1 scope, which is why it deletes 71% of
                    # the pull rather than re-aiming it. FRAMEWORK 2(z54) 5b.
                    st = lam * tralo_slope(hard, K, rho)
                    bt = lam * tralo_slope(hard, K, rho, item_scale=True)
                    alm_lam[scope] = max(0.0, alm_lam[scope] + eta * r)
                    aa = alm_lam[scope] + mu * max(0.0, r) if r > 0 else 0.0
                    key = "K=0" if K < 1 else ("K=1..9" if K < 10 else
                                               ("K=10..99" if K < 100 else "K>=100"))
                    tralo_w[key] += at
                    fixed_w[key] += ft
                    st_w[key] += st
                    both_w[key] += bt
                    alm_w[key] += aa
                    n_ep[key] += 1
                    if key not in examples and r > 0:
                        examples[key] = (K, soft, hard, at, aa)
                    if hard > K:
                        viol[scope] += 1
                if not frozen:
                    gs = str(row.get("Global_Satisfied", "")).strip().lower()
                    ls = str(row.get("Local_Satisfied", "")).strip().lower()
                    if gs in ("true", "1") and ls in ("true", "1"):
                        frozen = True
                    else:
                        rho = min(rho + rho_step, rho_tgt) if rho_tgt else rho + rho_step

    order = ["K=0", "K=1..9", "K=10..99", "K>=100"]
    tt = sum(tralo_w[k] for k in order) or 1.0
    ta = sum(alm_w[k] for k in order) or 1.0
    tf = sum(fixed_w[k] for k in order) or 1.0
    ts = sum(st_w[k] for k in order) or 1.0
    tb = sum(both_w[k] for k in order) or 1.0
    print("=" * 92)
    print("WHERE EACH RULE SENDS THE FIXED-NORM STEP  (%d tralo runs, same scope-states)"
          % runs)
    print("=" * 92)
    print("%-10s %9s %8s %9s %8s %8s %8s" %
          ("budget K", "scope-ep", "shipped", "ITEMSCALE", "st", "both", "ALM"))
    print("-" * 92)
    for k in order:
        if not n_ep[k]:
            continue
        print("%-10s %9d %7.1f%% %8.1f%% %7.1f%% %7.1f%% %7.1f%%" %
              (k, n_ep[k], 100.0 * tralo_w[k] / tt, 100.0 * fixed_w[k] / tf,
               100.0 * st_w[k] / ts, 100.0 * both_w[k] / tb,
               100.0 * alm_w[k] / ta))
    print("")
    print("  TOTAL PULL RETAINED -- read this before the shares. A knob that")
    print("  DELETES work rather than re-aiming it shows up here and nowhere else:")
    for lab, v in (("shipped", tt), ("itemscale", tf), ("st", ts), ("both", tb)):
        print("    %-10s %12.1f  (%.2fx shipped)" % (lab, v, v / tt))
    print("")
    print("A SINGLE SCOPE-EPOCH FROM EACH BUCKET, to show the inversion concretely:")
    print("%-10s %8s %10s %8s %14s %14s" %
          ("budget K", "K", "soft", "hard", "TraLO A_S", "ALM A_S"))
    print("-" * 92)
    for k in order:
        if k in examples:
            K, soft, hard, at, aa = examples[k]
            print("%-10s %8.0f %10.2f %8.0f %14.5f %14.5f" % (k, K, soft, hard, at, aa))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
