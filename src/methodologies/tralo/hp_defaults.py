"""TraLO hyperparameter defaults (informational).

Only knobs that survived measurement live here. See docs/FRAMEWORK.md section 2
for what was removed and why -- in short: penalty shape, step count, step
direction and the KL anchor were all measured and all made things worse or
nothing at all.
"""

DEFAULTS = {
    "lr_constraint": 1e-4,      # MUST equal lr; unequal LR fabricated a -16.7pp finding
    "lambda_global": 0.01,
    "lambda_local": 0.01,
    "lambda_step": 0.05,
    "initial_rho": 0.5,
    "rho_target": 100.0,        # rho_step is DERIVED from this; a config rho_step is ignored
    "constraint_chunk_size": 256,
}
