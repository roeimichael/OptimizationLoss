import math

import torch

from tralo.cutpair_gate import gate_counts


def test_tau_is_the_logit_of_the_cap_th_largest_development_p3_and_bands_count_correctly():
    dev = torch.tensor([0.9, 0.8, 0.5, 0.2, 0.1])
    tau = math.log(0.5 / 0.5)                                   # cap 3 -> third largest p3 = 0.5
    s = torch.tensor([-5.0, -0.5, 0.5, 2.0, -3.5, -0.9, 0.2])
    y = torch.tensor([0, 0, 0, 3, 3, 3, 3])
    out = gate_counts(dev, s, y, 3)
    assert abs(out['tau'] - tau) < 1e-12
    assert out['n_act'] == 2                # negatives with s > tau - 1: -0.5, 0.5
    assert out['p_act'] == 2                # positives in (tau - 3, tau + 1): -0.9, 0.2
    assert out['positives_above'] == 1      # 2.0
    assert out['positives_far_below'] == 1  # -3.5
    assert out['negatives_far_below'] == 1  # -5.0
