"""hounie_rh hyperparameter defaults.

Host (hounie_rcl) values are inherited verbatim from the paper_final
configs of the matched cell; only the two graft knobs are new.
"""

DEFAULTS = {
    "lr_constraint": 1e-5,
    "hounie_eta_lambda": 0.01,
    "hounie_eta_u": 0.01,
    "hounie_alpha": 10.0,
    # graft knobs (TraLO's shipped values)
    "fior_beta": 0.5,
    "reset_optimizer_at_sat": True,
}
