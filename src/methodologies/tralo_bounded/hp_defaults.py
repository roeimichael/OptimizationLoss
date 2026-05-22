"""TraLO hyperparameter defaults (informational; live values come from config)."""

DEFAULTS = {
    "lr_constraint": 1e-5,
    "lambda_global": 0.01,
    "lambda_local": 0.01,
    "lambda_step": 0.002,
    "initial_rho": 0.5,
    "rho_target": 100.0,
    "alpha_kl": 0.0,
    "constraint_chunk_size": 256,
}
