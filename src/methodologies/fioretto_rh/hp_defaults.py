"""fioretto_rh hyperparameter defaults.

Host (fioretto_ldf) values are inherited verbatim from the paper_final
configs of the matched cell; only the two graft knobs are new.
"""

DEFAULTS = {
    "fioretto_step_size": 0.005,
    "lr_constraint": 1e-5,
    # graft knobs (TraLO's shipped values)
    "fior_beta": 0.5,
    "reset_optimizer_at_sat": True,
}
