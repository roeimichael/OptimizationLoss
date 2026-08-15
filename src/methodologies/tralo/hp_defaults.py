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
    "hybrid_mode": "undershoot_hinge",  # "bounded_only" | "undershoot_hinge"
    "fior_beta": 0.50,                  # undershoot hinge weight (paper recipe)
}

# ---- GEOM arm: the cut objective (see newdirections/arm_geom/README.md) ----
DEFAULTS.update({
    # P1  cut-margin objective.  "off" reproduces the incumbent exactly.
    "cut_loss": "off",        # off | hinge | otce (Asano-style OT control)
    "cut_gamma": 1.0,         # margin demanded at the cut, in MAD units
    "cut_weight": 1.0,
    "cut_scope": "global",    # global | both (adds the per-group caps)
    # P2  count what verification counts.
    "soft_count_mode": "prob",   # prob (incumbent, sum_i p_ic) | sigmoid
    "count_tau": 0.25,           # nats, only used when soft_count_mode=sigmoid
})
