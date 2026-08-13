"""alm_rh hyperparameter defaults.

Host values are inherited verbatim from the paper_final config of the matched
cell. The ALM knobs match src/config_generators/gen_alm_full.py so this arm is
paired against fioretto_alm at identical settings; the graft knobs match
fioretto_rh so it is paired against that arm too.
"""

DEFAULTS = {
    "fioretto_step_size": 0.005,
    "lr_constraint": 1e-5,
    # ALM dual knobs (gen_alm_full.py ALM_HP)
    "alm_eta": 0.005,
    "alm_mu0": 0.01,
    "alm_mu_step": 0.01,
    # graft knobs (TraLO's shipped values)
    "fior_beta": 0.5,
    "reset_optimizer_at_sat": True,
}
