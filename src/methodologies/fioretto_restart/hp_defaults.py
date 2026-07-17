"""fioretto_restart hyperparameter defaults.

Host (fioretto_ldf) values are inherited verbatim from the paper_final
configs of the matched cell. The dual restart has no tunable knob: at
hard-count feasibility all multipliers are zeroed (Gallego-Posada et al.
2022 dual-restart rule).
"""

DEFAULTS = {
    "fioretto_step_size": 0.005,
    "lr_constraint": 1e-5,
}
