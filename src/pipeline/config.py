"""Maintained runtime hyperparameter schema; unknown keys fail before training."""

CORE = {
    "lr",
    "dropout",
    "batch_size",
    "pretrained",
    "class_weighted_ce",
    "seed",
    "warmup_epochs",
    "constraint_epochs",
}
STEP = {
    "constraint_grad_clip",
    "constraint_grad_mode",
    "constraint_fp32",
    "lr_constraint",
    "constraint_chunk_size",
}
TRALO = {"lambda_step", "lambda_global", "lambda_local", "initial_rho", "rho_target"}
METHOD_KEYS = {
    "tralo": CORE | STEP | TRALO,
    "fioretto_ldf": CORE | STEP | {"fioretto_step_size", "fioretto_lambda_init"},
    "hounie_rcl": CORE | STEP | {"hounie_eta_lambda", "hounie_eta_u", "hounie_alpha"},
    "fioretto_alm": CORE
    | STEP
    | {"alm_eta", "alm_mu0", "alm_mu_step", "fioretto_lambda_init"},
    "heuristic": CORE
    | {"warmup_loss", "focal_alpha", "focal_gamma", "inference_chunk_size"},
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
    for key in ("constraint_fp32", "pretrained", "class_weighted_ce"):
        if key in hp and not isinstance(hp[key], bool):
            raise ValueError("%s must be a bool" % key)
    if "constraint_grad_mode" in hp and hp["constraint_grad_mode"] not in (
        "clip",
        "normalize",
    ):
        raise ValueError("constraint_grad_mode must be clip or normalize")
