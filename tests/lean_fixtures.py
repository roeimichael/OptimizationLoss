"""Correctness-only rival null configurations, never public campaign arms."""

from copy import deepcopy
from configs.gen_campaign import load_protocol


def protocol_with_nulls():
    p = load_protocol()
    for parent, zero in {
        "fioretto": {"fioretto_step_size": 0.0, "fioretto_lambda_init": 0.0},
        "hounie": {"hounie_eta_lambda": 0.0},
        "alm": {
            "alm_eta": 0.0,
            "alm_mu0": 0.0,
            "alm_mu_step": 0.0,
            "fioretto_lambda_init": 0.0,
        },
    }.items():
        name = parent + "_null"
        p["blocks"][name] = {**p["blocks"][parent], **zero}
        p["arms"][name] = deepcopy(p["arms"][parent])
        p["arms"][name]["blocks"] = ["constraint_phase", name]
    return p
