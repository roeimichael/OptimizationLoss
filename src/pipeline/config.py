"""Maintained runtime hyperparameter schema; unknown keys fail before training."""

CORE = {
    "lr",
    "dropout",
    "batch_size",
    "pretrained",
    "seed",
    "warmup_epochs",
    "constraint_epochs",
    # Absent by default, so `compute_base_model_id` -- which includes only the
    # identity keys PRESENT in hp -- leaves every existing warm-up digest
    # unchanged. Declared here so a config that DOES carry it is accepted.
    "augment",
}
# The focal warm-up objective was accepted for `heuristic` only, so the arm the
# user asked for -- focal loss UNDER the constraint -- could not be expressed.
# `make_ce_criterion` already honours `warmup_loss` and is what tralo/train.py
# and dual_common build their task criterion with, so this schema line is the
# whole change. Measured reason it is worth having: the constraint reaches the
# weights only through d(soft count)/d(theta), whose per-item weight is
# p(1-p); on a saturated model that mean is 0.007-0.023 against a maximum of
# 0.25, with the top 1% of items carrying 34% of it. Under focal the same
# number is 0.015-0.045 and the top 1% carries 5-16% -- a gradient the
# constraint can actually steer with instead of a few dozen borderline items.
FOCAL = {"warmup_loss", "focal_alpha", "focal_gamma"}
STEP = {
    "constraint_grad_clip",
    "constraint_grad_mode",
    "constraint_fp32",
    "lr_constraint",
    "constraint_chunk_size",
}
TRALO = {"lambda_step", "lambda_global", "lambda_local", "initial_rho", "rho_target"}
# Per-item weighting of the transductive soft count. tralo only -- no other
# methodology builds a soft count we control.
WEIGHT = {"constraint_weight", "constraint_weight_k", "constraint_weight_floor"}
# Observation only, and deliberately NOT a warm-up identity key: it changes what
# is recorded, never what is trained, so it must not invalidate a cached warm-up
# or split code_version from an otherwise identical run. tralo ONLY -- no other
# methodology reads it, and `audit_config` rejects a config key that no runtime
# reads, which is the guard that caught it sitting in `core`.
DIAG = {"epoch_trace"}
METHOD_KEYS = {
    "tralo": CORE | STEP | TRALO | FOCAL | WEIGHT | DIAG,
    "fioretto_ldf": CORE | STEP | FOCAL | {"fioretto_step_size", "fioretto_lambda_init"},
    "hounie_rcl": CORE | STEP | FOCAL | {"hounie_eta_lambda", "hounie_eta_u", "hounie_alpha"},
    "fioretto_alm": CORE
    | STEP
    | FOCAL
    | {"alm_eta", "alm_mu0", "alm_mu_step", "fioretto_lambda_init"},
    "heuristic": CORE | FOCAL,
}


def validate_hyperparams(methodology, hp):
    if methodology not in METHOD_KEYS:
        raise ValueError("Unknown methodology: %s" % methodology)
    unknown = set(hp) - METHOD_KEYS[methodology]
    if unknown:
        raise ValueError(
            "Unknown hyperparameter(s) for %s: %s"
            % (methodology, ", ".join(sorted(unknown)))
        )
    if hp.get("warmup_loss", "ce") not in ("ce", "focal"):
        raise ValueError("warmup_loss must be ce or focal")
    for key in ("constraint_fp32", "pretrained", "epoch_trace"):
        if key in hp and not isinstance(hp[key], bool):
            raise ValueError("%s must be a bool" % key)
    if "constraint_grad_mode" in hp and hp["constraint_grad_mode"] not in (
        "clip",
        "normalize",
    ):
        raise ValueError("constraint_grad_mode must be clip or normalize")
