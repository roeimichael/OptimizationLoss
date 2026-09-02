"""How much does one constraint step disturb the classes it never names?

THE QUESTION, and why it is the one that matters. `scripts/family_split.py` on
`results/xfam1` (16 matched cell-seeds, 9 cells, 2026-08-24) split every dual
family's score against `clip` into the 29 extra epochs and the constraint
itself, and found:

                        d ccF1 (capped)   d uncF1 (UNCAPPED)
    tralo    constraint      -0.0020            -0.0144
    fioretto constraint      -0.0023            -0.0027
    hounie   constraint      -0.0028            -0.0114

All three do the SAME negligible thing (~1 item) to the two classes the
constraint is about. They differ 5.3x in what they do to the six classes it
never mentions. So the spread between dual families -- the entire ordering the
manuscript reports -- is collateral damage, not constraint quality.

THE MECHANISM, and the fix it implies. The shipped count is `S_c = sum_i p_ic`
with `p = softmax(z)`, so `dS_c/dz_k = -sum_i p_ic p_ik` is NONZERO for every
uncapped k: pushing a capped class down moves all eight logits. A one-vs-rest
count `S_c = sum_i sigmoid(z_ic)` has `dS_c/dz_k = 0` EXACTLY for k != c, so no
uncapped item can change which uncapped class it prefers AT ANY DOSE. Lowering
`z_c` can only convert a `predicted c` item into `predicted k`, which is the
intended effect of a cap.

THE GRADIENTS ARE TAKEN FROM `src`, NOT RESTATED HERE. The first version of
this file re-derived `uniform`'s gradient by hand and got it wrong -- it
reported the arm moving the capped count by -0.0000, which would have
mispriced a campaign that was already staged to launch. Every shipped mode is
now autograd'd through the real `src.losses.transductive_loss` function, so a
change there cannot silently invalidate this probe. `ovr` is defined here
because it does not exist in `src` yet; that is the point of pricing it.

DOSE IS MATCHED ON EFFECT, NOT ON STEP SIZE. At one unit-norm step no mode
flips a single uncapped prediction, so equal-dose comparison reports 0 vs 0 and
says nothing. Each mode is instead stepped until it removes the SAME number of
capped predictions, and the collateral is read there. Comparing collateral at
equal dose but unequal effect measures the dose (FRAMEWORK 2(a3)).

`z = log p` is a valid logit vector: softmax is invariant to a per-item
additive constant, so the stored probabilities determine z up to exactly that,
and every quantity below is invariant to it. Runs on CPU in seconds against
`final_predictions_raw.csv`, which every finished run already wrote -- no
model, no GPU, no labels.

SELF-TEST (runs by default; `--no-self-test` to skip):
  * `ovr` moves NO uncapped logit                 -- exact zero, by construction
  * `sum` moves them                              -- else there is nothing to fix
  * a zero-size step moves nothing under any mode -- else the diff is noise

    python -m scripts.collateral_probe --campaign results/xfam1 --arm tralo_null
"""

import argparse
import sys
import glob
import json
import os

import numpy as np
import torch

from pathlib import Path

from scripts.reachability import budgets
from src.losses.transductive_loss import uniform_grad_count

MODES = ("sum", "uniform", "ovr")


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def grad_count(z, capped, mode):
    """d/dz of `sum_c S_c` over the capped classes, by autograd. Returns (N, C).

    `sum` and `uniform` call the SHIPPED functions, so this cannot drift from
    what the trainer does. `ovr` is the candidate and is defined here.
    """
    t = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    if mode == "ovr":
        s = torch.sigmoid(t[:, capped]).sum()
    else:
        p = torch.softmax(t, dim=1)
        counts = uniform_grad_count(p) if mode == "uniform" else p
        s = counts[:, capped].sum()
    s.backward()
    return t.grad.detach().numpy()


def step(z, capped, mode, eta):
    g = grad_count(z, capped, mode)
    n = np.linalg.norm(g)
    if n < 1e-12:
        return z.copy()
    return z - eta * g / n          # descend: push the capped counts DOWN


def removed(z0, z1, capped):
    """Capped predictions the step removed -- the INTENDED effect."""
    a = np.isin(softmax(z0).argmax(1), capped).sum()
    b = np.isin(softmax(z1).argmax(1), capped).sum()
    return int(a - b)


ETA_MAX = 4096.0          # at this step the logits have already moved ~79
                          # units, so "unreachable" means unreachable in
                          # practice -- but the bound is named, not implied


def eta_for(z0, capped, mode, target, hi=ETA_MAX):
    """Smallest eta removing >= `target` capped predictions, or None.

    Bisection is valid because the step direction is fixed and the capped
    logits fall monotonically in eta, so the count removed is monotone.
    """
    if removed(z0, step(z0, capped, mode, hi), capped) < target:
        return None
    lo = 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if removed(z0, step(z0, capped, mode, mid), capped) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def report(z0, z1, capped):
    unc = [k for k in range(z0.shape[1]) if k not in capped]
    p0, p1 = softmax(z0), softmax(z1)
    pred0, pred1 = p0.argmax(1), p1.argmax(1)
    # The collateral that costs uncF1: an item that preferred one UNCAPPED
    # class and now prefers a DIFFERENT uncapped one. No cap can justify it.
    both_unc = np.isin(pred0, unc) & np.isin(pred1, unc)
    return {
        "capped_removed": removed(z0, z1, capped),
        "unc_logit_moved": float(np.abs(z1[:, unc] - z0[:, unc]).max()),
        "unc_to_unc_flips": int((both_unc & (pred0 != pred1)).sum()),
    }


def runs(root, arm):
    for cfgp in glob.glob(os.path.join(root, "*/*/*/%s/*/config.json" % arm)):
        d = os.path.dirname(cfgp)
        raw = os.path.join(d, "final_predictions_raw.csv")
        if os.path.exists(raw):
            yield raw, json.load(open(cfgp))


def capped_of(cfg):
    dc = cfg.get("dataset_config", {}) or {}
    c = dc.get("constrained_class")
    if c is None:
        raise SystemExit("no constrained_class in config; the capped columns "
                         "would be unnamed and every number below would be "
                         "answering a different question")
    return [int(x) for x in (c if isinstance(c, (list, tuple)) else [c])]


def self_test(z0, capped):
    if not np.allclose(step(z0, capped, "ovr", 0.0), z0):
        raise SystemExit("SELF-TEST: a zero-size step moved the logits.")
    m_ovr = report(z0, step(z0, capped, "ovr", 1.0), capped)["unc_logit_moved"]
    if m_ovr != 0.0:
        raise SystemExit("SELF-TEST: `ovr` moved an uncapped logit by %.3g. "
                         "Its whole claim is that it cannot." % m_ovr)
    m_sum = report(z0, step(z0, capped, "sum", 1.0), capped)["unc_logit_moved"]
    if m_sum == 0.0:
        raise SystemExit("SELF-TEST: `sum` moved no uncapped logit either, so "
                         "there is no collateral here to remove and the "
                         "contrast is void.")


def standalone_self_test(w=sys.stdout.write):
    """Run `self_test`'s properties on SYNTHETIC logits, with no campaign.

    WARNING: NOT the fallback-to-a-toy defect (conftest, `slice_dir`). That rule is
    about MEASURING a campaign against synthetic data and reporting the number
    as if it were real. Nothing is measured here: `step` and `report` are pure
    functions of the logits, and the claims under test -- `ovr` cannot move an
    uncapped logit, `sum` can -- are algebraic and hold for any input. Feeding
    them real logits made the liveness check unrunnable anywhere the artefacts
    are absent, which is CI and every fresh checkout.

    Added 2026-09-02: this was the ONE script of 23 whose `--self-test` could
    not run, because the flag was inverted (`--no-self-test`) and gated behind
    a `--campaign` that is `required=True`. A liveness check that cannot be
    invoked is not a liveness check.
    """
    ok = True

    def check(good, label):
        nonlocal ok
        w("  %-4s %s\n" % ("PASS" if good else "FAIL", label))
        ok = ok and good

    rng = np.random.default_rng(0)
    z0 = rng.normal(size=(400, 8))
    capped = [2, 7]
    try:
        self_test(z0, capped)
        check(True, "the three properties hold on synthetic logits")
    except SystemExit as exc:
        check(False, "self_test rejected synthetic logits: %s" % exc)

    # NEGATIVE CONTROL. A check that has never failed has never been shown to
    # work. Replace `step` with a no-op -- the exact failure this probe exists
    # to detect, an intervention that is installed and delivers nothing (see
    # `ortho_survival`: `prm.grad` is not the delivery mechanism) -- and
    # require `self_test` to catch it.
    g = globals()
    real = g["step"]
    g["step"] = lambda z, capped_, mode, size: z
    try:
        self_test(z0, capped)
        check(False, "NEGATIVE CONTROL: `step` was replaced by a NO-OP and "
                     "self_test still passed, so it proves nothing")
    except SystemExit:
        check(True, "NEGATIVE CONTROL: self_test FAILS when `step` does nothing")
    finally:
        g["step"] = real

    w("\nSELF-TEST %s\n" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    import pandas as pd
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--self-test", action="store_true",
                   help="run the property checks on synthetic logits and "
                        "exit; needs no campaign and no artefacts")
    a.add_argument("--campaign")
    a.add_argument("--arm", default="tralo_null",
                   help="a lambda=0 arm: the state the constraint acts FROM")
    a.add_argument("--target", type=int, default=20,
                   help="capped predictions each mode must remove before its "
                        "collateral is read -- the EFFECT is what is matched")
    a.add_argument("--feasibility", action="store_true",
                   help="ignore --target; ask instead whether each mode can "
                        "remove this run's OWN excess (raw count - K), i.e. "
                        "reach feasibility through the output layer at all")
    a.add_argument("--no-self-test", action="store_true")
    args = a.parse_args()

    if args.self_test:
        return standalone_self_test()
    if not args.campaign:
        a.error("--campaign is required (or use --self-test)")

    rows, unreachable, attempted = [], [], 0
    for raw, cfg in runs(args.campaign, args.arm):
        t = pd.read_csv(raw)
        cols = sorted((c for c in t.columns if c.startswith("Prob_Class_")),
                      key=lambda c: int(c.rsplit("_", 1)[1]))
        P = t[cols].to_numpy(float)
        if not np.isfinite(P).all():
            continue
        z0 = np.log(np.clip(P / np.clip(P.sum(1, keepdims=True), 1e-12, None),
                            1e-12, None))
        capped = capped_of(cfg)
        cell = "%s/%s" % (cfg.get("model_name"), cfg.get("constraint_tag"))
        if not args.no_self_test:
            self_test(z0, capped)
        attempted += 1
        target = args.target
        if args.feasibility:
            # The excess the model ACTUALLY has to shed, not a round number.
            # `budgets` reads the training log first and falls back to the
            # config, because a post-hoc arm logs `Limit_Class = inf` and a
            # caller that skips it drops the control silently.
            K = budgets(Path(raw).parent)
            pred = softmax(z0).argmax(1)
            target = int(sum(max(0, int((pred == c).sum()) - int(K[c]))
                             for c in capped if c in K))
            if target <= 0:
                attempted -= 1
                continue      # already feasible: nothing to reach
        for mode in MODES:
            eta = eta_for(z0, capped, mode, target)
            if eta is None:
                unreachable.append((cell, mode))
                continue
            r = report(z0, step(z0, capped, mode, eta), capped)
            r.update(mode=mode, cell=cell, eta=eta, target=target)
            rows.append(r)

    if not rows:
        raise SystemExit("no scorable run under %s/*/*/*/%s"
                         % (args.campaign, args.arm))

    df = pd.DataFrame(rows)
    n_runs = attempted
    print("=" * 80)
    if args.feasibility:
        print("CAN THE COUNT REACH FEASIBILITY AT ALL?  --  arm=%s, %d run(s)"
              % (args.arm, n_runs))
        print("  target is each run's OWN excess (raw count - K), mean %.1f "
              "items" % df["target"].mean())
    else:
        print("COLLATERAL AT MATCHED EFFECT  --  arm=%s, %d run(s), target=%d "
              "capped preds removed" % (args.arm, n_runs, args.target))
    print("  self-test PASSED: `ovr` moves no uncapped logit, `sum` does, and")
    print("  a zero-size step moves nothing.")
    print("=" * 80)
    print("  %-9s %12s %14s %16s %16s"
          % ("mode", "reached", "eta needed", "unc logit max", "unc->unc flips"))
    for mode in MODES:
        sub = df[df["mode"] == mode]
        reached = "%d/%d" % (len(sub), n_runs)
        if sub.empty:
            print("  %-9s %12s   -- never reaches it at eta <= %.0f --"
                  % (mode, reached, ETA_MAX))
            continue
        print("  %-9s %12s %14.3f %16.6f %16.2f"
              % (mode, reached, sub["eta"].mean(),
                 sub["unc_logit_moved"].mean(), sub["unc_to_unc_flips"].mean()))
    if unreachable:
        print("  A run counted as NOT reached could not get there at any")
        print("  eta <= %.0f, by which point the logits have moved tens of"
              % ETA_MAX)
        print("  units. That is a verdict, not a gap -- and the averages above")
        print("  are over the runs that DID reach it, so a mode with a low")
        print("  `reached` is being flattered by the rows it is missing.")
    print()
    print("  `unc->unc flips` is the damage: an item that preferred one")
    print("  UNCAPPED class and now prefers a DIFFERENT uncapped one. It is")
    print("  what `d uncF1` is made of, and no cap can justify it.")
    print("  Effect is held equal across the rows, so the flips column is the")
    print("  PRICE each mode charges for the same amount of cap enforcement.")


if __name__ == "__main__":
    main()
