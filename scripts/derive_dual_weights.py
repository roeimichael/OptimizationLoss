"""The receipt for FRAMEWORK section 2's dual-weight table.

That table is the evidence for "do not claim these are three distinct dual
baselines": at protocol.yml's step sizes the three duals put wildly different
effective weights on d(soft_count)/d(theta), and two of them land so far above
the unit-norm clip that they take a bit-identical update.

It was prose-only, and its ALM row was WRONG ONCE -- 701.2 / 701, which
reproduces only under `lambda = max(0, lambda + eta*r) + mu_t*r+` STORED BACK
into lambda, the form the current code documents as the bug it fixed. Under the
rule the code actually runs it is 67.86, a 10.3x overstatement that changed the
row's conclusion from "ALM is 31x Fioretto" to "ALM is 3x Fioretto". A number
that only exists in prose is a number that can stay wrong, so:

    python -m scripts.derive_dual_weights

Everything comes from configs/protocol.yml. Nothing is hardcoded except the
regime the table is stated for (N, K, and a soft count held constant across the
29 constraint epochs), which is printed. No data, no GPU.
"""
import argparse
import io
import os
import sys

import yaml

# The regime FRAMEWORK section 2 states the table for: dermmnist slice_1 at
# L30_G30, with the soft count held at its warm-up value for all 29 epochs.
# Holding it constant is what makes this a closed form rather than a rerun --
# and it is the charitable assumption, since a falling count would only shrink
# every multiplier.
N_TEST, K, SOFT = 2003, 67, 223
EPOCHS = 29


def _assert_rule(path, needle, what):
    """Refuse to report a weight derived from a rule the code no longer runs."""
    src = io.open(os.path.join("src", "methodologies", path, "train.py"),
                  encoding="utf-8").read()
    if needle not in src:
        raise SystemExit(
            "%s no longer contains %r, so its %s is not the rule this script "
            "models. Re-derive before quoting the table." % (path, needle, what))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default=os.path.join("configs", "protocol.yml"))
    args = ap.parse_args(argv)

    blocks = yaml.safe_load(io.open(args.protocol, encoding="utf-8"))["blocks"]
    f_step = float(blocks["fioretto"]["fioretto_step_size"])
    alm = blocks["alm"]
    hou = blocks["hounie"]

    # Each arm's update rule, checked against the source before it is used.
    _assert_rule("fioretto_ldf", "lambda_g[c] += step_size * viol", "dual update")
    _assert_rule("fioretto_alm", "lambda_g[c] = max(0.0, lambda_g[c] + eta * r)",
                 "dual update")
    _assert_rule("fioretto_alm", "aug_g[c] = mu_t * max(0.0, r)",
                 "augmentation (added at USE time, never stored back)")
    _assert_rule("fioretto_alm", "mu_t = mu0 + mu_step * epoch", "mu schedule")
    _assert_rule("hounie_rcl", "hounie_eta_lambda", "dual step size")
    print("all four update rules match the source")

    r = SOFT - K                      # raw residual, the fioretto/ALM convention
    r_norm = r / N_TEST               # hounie divides its primal term by n_test
    print("\nregime: N=%d, K=%d, soft count %d held constant over %d epochs"
          % (N_TEST, K, SOFT, EPOCHS))
    print("        raw residual r = %d,  normalized r/N = %.5f\n" % (r, r_norm))

    # fioretto_ldf: pure positive-part subgradient accumulation
    lam_f = EPOCHS * f_step * r
    eff_f = lam_f

    # fioretto_alm: the multiplier follows the SIGNED residual with a
    # nonnegativity projection, and mu_t*r+ is added to the PRIMAL weight at use
    # time. mu_t is read at the last epoch INDEX, which is EPOCHS-1 because the
    # loop is `for epoch in range(constraint_epochs)`. Storing the augmentation
    # back into lambda instead is exactly the 701.2 error.
    lam_a = EPOCHS * float(alm["alm_eta"]) * r
    mu_t = float(alm["alm_mu0"]) + float(alm["alm_mu_step"]) * (EPOCHS - 1)
    aug = mu_t * r
    eff_a = lam_a + aug

    # hounie_rcl: normalized residual, and the primal term is divided by n_test
    # again, so the weight on d(soft_count)/d(theta) picks up 1/N twice.
    lam_h = EPOCHS * float(hou["hounie_eta_lambda"]) * r_norm
    eff_h = lam_h / N_TEST

    print("%-10s %14s %20s" % ("arm", "lambda @ep29", "effective weight"))
    print("%-10s %14.4f %20.4f" % ("fioretto", lam_f, eff_f))
    print("%-10s %14.4f %20.4f   (+ mu_t=%.2f x r = %.2f)"
          % ("alm", lam_a, eff_a, mu_t, aug))
    print("%-10s %14.4f %20.3e" % ("hounie", lam_h, eff_h))

    print("\nfioretto / hounie = %.3g" % (eff_f / eff_h))
    print("alm / fioretto    = %.2fx" % (eff_a / eff_f))
    print("\nthe WRONG rule (augmentation stored back into lambda, compounding")
    print("every epoch) gives, for comparison:")
    lam_bad = 0.0
    for e in range(EPOCHS):
        mu_e = float(alm["alm_mu0"]) + float(alm["alm_mu_step"]) * e
        lam_bad = max(0.0, lam_bad + float(alm["alm_eta"]) * r) + mu_e * r
    print("  alm lambda = %.1f  <-- the retracted 701.2" % lam_bad)

    print("\nBoth fioretto and ALM blow past the unit-norm constraint_grad_clip")
    print("for any plausible ||dS/dtheta||, so the clip renormalizes them to the")
    print("same norm-1 step: with a single active constraint the two arms take a")
    print("BIT-IDENTICAL update. Hounie never reaches norm 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
