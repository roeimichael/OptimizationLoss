"""WHERE DOES TraLO'S FIXED-NORM CONSTRAINT STEP ACTUALLY GO?

Under `constraint_grad_mode: normalize` the constraint GRADIENT is rescaled to
norm exactly `constraint_grad_clip` before the optimizer sees it, so a global
scalar on the constraint loss divides out and the whole degree of freedom is
the DIRECTION -- i.e. the RATIOS between scopes.

WARNING: `p.grad`, NOT the weight step. The delivered WEIGHT step is lr*clip
only under `constraint_step_rule: sgd`; the protocol runs `shared`, which hands
the pinned gradient to the CE Adam. The ratio argument is unaffected (the
rescale is upstream of the optimizer) but `lr_constraint` and
`constraint_grad_clip` are NOT magnitude-inert -- they are held EQUAL ACROSS
ARMS by `check_parity`, which is a different guarantee. This asks which
scopes those ratios favour, and in particular how much goes to scopes with no
prize left.

The established per-logit gradient is

    dL/dz_ic = A_S * p_ic (1 - p_ic),      A_S = lambda_S * d(pen_S)/d(soft_S)

so A_S is exactly the per-scope pull weight, and share(A_S) is the split of the
step across scopes up to the per-scope item counts (see CAVEAT).

Both factors are reconstructible from `training_log.csv` + `config.json`:

  pen(soft, K) = E/(E+s) + rho * e^2/(1+e^2),   E = relu(soft-K), s = max(K,1), e = E/s
  d pen/d soft = s/(E+s)^2 + rho * 2e / (s (1+e^2)^2)          for E > 0, else 0
  lambda_S(t)  = lambda_0 + lambda_step * #{epochs < t : hard_S > K_S}   (constant ratchet)
  rho(t)       = initial_rho + rho_step * t, frozen on the first all-satisfied epoch

THE QUESTION. On iwildcam SEVEN of the fourteen per-group ceilings are K = 0.
A K = 0 scope whose HARD count is already 0 is fully compliant -- the allocator
emits nothing there and no item is at stake -- yet `relu(soft - 0) > 0` for any
softmax, so the term never switches off. Every unit of A_S it carries is a unit
the fixed-norm step does NOT spend on a scope that still has items to win.

CAVEAT, stated because it bounds the claim: A_S weights each scope's OWN item
set, and group sizes differ. share(A_S) is the split of the per-item pull, not
of the summed gradient norm. It is exact when groups are equal-sized and an
approximation otherwise. The log carries no per-group item counts, so this is
the honest ceiling of what the log can say.

    python budget_share.py <campaign-root> [more roots ...]
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


def penalty_and_slope(soft, K, rho):
    E = max(0.0, soft - K)
    if E <= 0.0:
        return 0.0, 0.0
    s = K if K >= 1 else 1.0
    e = E / s
    pen = E / (E + s) + rho * (e * e) / (1.0 + e * e)
    slope = s / ((E + s) ** 2) + rho * (2.0 * e) / (s * ((1.0 + e * e) ** 2))
    return pen, slope


def hp_of(run_dir):
    try:
        with open(os.path.join(run_dir, "config.json")) as fh:
            cfg = json.load(fh)
    except (IOError, ValueError):
        return None
    hp = cfg.get("hyperparams", cfg.get("hyperparameters", cfg))
    return {
        "lambda_global": float(hp.get("lambda_global", 0.0)),
        "lambda_local": float(hp.get("lambda_local", 0.0)),
        "lambda_step": float(hp.get("lambda_step", 0.0)),
        "initial_rho": float(hp.get("initial_rho", 0.0)),
        "rho_step": float(hp.get("rho_step", 0.0)),
        "rho_target": float(hp.get("rho_target", 0.0)),
        "mode": str(hp.get("lambda_ratchet_mode", "constant")),
    }


def rows_of(run_dir):
    path = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(path):
        return None, None
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        if not rd.fieldnames:
            return None, None
        return scopes_of(rd.fieldnames), list(rd)


def fnum(row, col):
    try:
        return float(row[col])
    except (KeyError, TypeError, ValueError):
        return None


def bucket(kind, K, hard):
    if K < 1:
        return "K0_won" if hard <= 0 else "K0_live"
    if hard > K:
        return "over"
    return "under"


def analyse(run_dir):
    hp = hp_of(run_dir)
    scopes, rows = rows_of(run_dir)
    if not hp or not scopes or not rows:
        return None
    if hp["mode"] != "constant":
        return None                      # lambda not reconstructible; skip loudly

    viol_count = collections.Counter()    # scope -> epochs violated so far
    share = collections.Counter()         # bucket -> summed A_S
    scope_eps = collections.Counter()
    rho_frozen = False
    rho = hp["initial_rho"]

    for t, row in enumerate(rows):
        # rho ramps until the first all-satisfied epoch, then freezes
        gs = str(row.get("Global_Satisfied", "")).strip().lower()
        ls = str(row.get("Local_Satisfied", "")).strip().lower()
        for scope, cols in scopes.items():
            K = fnum(row, cols["limit"])
            hard = fnum(row, cols["hard"])
            soft = fnum(row, cols["soft"])
            if K is None or hard is None or soft is None or K >= UNLIMITED:
                continue
            lam0 = hp["lambda_global"] if scope[0] == "global" else hp["lambda_local"]
            lam = lam0 + hp["lambda_step"] * viol_count[scope]
            _pen, slope = penalty_and_slope(soft, K, rho)
            a = lam * slope
            b = bucket(scope[0], K, hard)
            share[b] += a
            scope_eps[b] += 1
            if hard > K:
                viol_count[scope] += 1
        if not rho_frozen:
            if gs in ("true", "1") and ls in ("true", "1"):
                rho_frozen = True
            else:
                rho = min(rho + hp["rho_step"], hp["rho_target"]) \
                    if hp["rho_target"] else rho + hp["rho_step"]
    return share, scope_eps


def main(roots):
    per_arm_share = collections.defaultdict(collections.Counter)
    per_arm_eps = collections.defaultdict(collections.Counter)
    runs = collections.Counter()
    skipped = collections.Counter()

    for root in roots:
        for run in sorted(glob.glob(os.path.join(root, "*", "*", "*", "*", "seed_*"))):
            arm = os.path.normpath(run).split(os.sep)[-2]
            if not arm.startswith("tralo"):
                continue
            got = analyse(run)
            if got is None:
                skipped[arm] += 1
                continue
            share, eps = got
            per_arm_share[arm].update(share)
            per_arm_eps[arm].update(eps)
            runs[arm] += 1

    BUCKETS = ("K0_won", "K0_live", "over", "under")
    LABEL = {
        "K0_won":  "K=0, hard=0   NOTHING TO WIN",
        "K0_live": "K=0, hard>0   winnable",
        "over":    "K>=1, over    winnable",
        "under":   "K>=1, under   soft-only",
    }
    print("=" * 92)
    print("SHARE OF THE PER-ITEM PULL WEIGHT  A_S = lambda_S * d(pen)/d(soft)")
    print("Under `normalize` the step norm is fixed, so this IS the split of the step.")
    print("=" * 92)
    for arm in sorted(per_arm_share):
        tot = sum(per_arm_share[arm][b] for b in BUCKETS)
        if tot <= 0:
            continue
        print("\n%s   (%d runs%s)" %
              (arm, runs[arm],
               ", %d skipped" % skipped[arm] if skipped[arm] else ""))
        print("  %-28s %12s %8s %12s" % ("bucket", "sum A_S", "share", "scope-ep"))
        for b in BUCKETS:
            print("  %-28s %12.4f %7.1f%% %12d"
                  % (LABEL[b], per_arm_share[arm][b],
                     100.0 * per_arm_share[arm][b] / tot, per_arm_eps[arm][b]))
        won = 100.0 * per_arm_share[arm]["K0_won"] / tot
        print("  --> %.1f%% of the fixed-norm step is aimed at scopes with NO ITEMS AT STAKE"
              % won)
    if skipped:
        print("\n!! skipped (non-constant ratchet, lambda not reconstructible): %s"
              % dict(skipped))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
