"""fioretto_alm hyperparameter defaults (B3: augmented-Lagrangian baseline).

Only the dual update differs from fioretto_ldf. `alm_eta` is the projected
dual-ascent step on the RAW residual (can shrink the multiplier when slack);
`alm_mu0` / `alm_mu_step` are the augmentation penalty coefficient and its
per-epoch linear growth. Constraint-phase HPs (constraint_epochs,
lr_constraint, ...) are inherited from the cloned Fioretto/TraLO config so the
ALM vs Fioretto comparison is apples-to-apples (same budget, same early stop).
"""

DEFAULTS = {
    "alm_eta": 0.005,      # dual-ascent step on the raw residual (S_c - K_c)
    "alm_mu0": 0.01,       # initial augmentation penalty coefficient
    "alm_mu_step": 0.01,   # linear growth of mu per epoch
    "lr_constraint": 1e-5,
}
