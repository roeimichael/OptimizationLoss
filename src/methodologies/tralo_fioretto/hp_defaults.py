"""TraLO-Fioretto hybrid hyperparameter defaults (informational)."""

DEFAULTS = {
    "lr_constraint": 1e-5,
    "lambda_global": 0.01,
    "lambda_local": 0.01,
    "lambda_step": 0.002,
    "initial_rho": 0.5,
    "rho_target": 100.0,
    "alpha_kl": 0.0,
    "constraint_chunk_size": 256,
    # Hybrid-specific
    "hybrid_mode": "single_lambda",   # "single_lambda" | "dual_lambda"
    "fior_beta": 0.2,                  # mix coefficient (single_lambda mode)
    "fior_step_size": 0.005,           # subgradient step for lambda_F (dual_lambda mode)
    "fior_lambda_init": 0.0,           # initial value of lambda_F (dual_lambda mode)
}
