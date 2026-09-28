import copy

import pytest
import torch

from tralo.fmow_local import RECIPE, budgets, snapshot_side_steps, validate


def test_fixed_recipe_and_unlabeled_country_budgets():
    config = dict(RECIPE, seed=6099, snapshot_steps=False)
    validate(config)
    with pytest.raises(ValueError):
        validate(dict(config, lr=2e-4))
    with pytest.raises(ValueError):
        validate(dict(config, seed=6100))
    groups = ["IRQ"] * 500 + ["NLD"] * 400 + ["DZA"] * 300 + ["PHL"] * 250 + ["TUR"] * 223
    quotas = budgets(groups)
    assert quotas["10"]["global_cap"] == 167
    assert quotas["10"]["local_total"] == sum(quotas["10"]["local_caps"].values()) == 209
    assert quotas["20"]["global_cap"] == 83
    assert quotas["20"]["local_total"] == sum(quotas["20"]["local_caps"].values()) == 104
    assert quotas == budgets(list(reversed(groups)))


def test_fixed_dose_config_is_distinct_from_failed_feasibility_study():
    config = dict(RECIPE, seed=6199, snapshot_steps=True,
                  study="local_fixed_dose_v1", step_radius=0.1)
    validate(config)
    with pytest.raises(ValueError):
        validate(dict(config, step_radius=0.2))
    with pytest.raises(ValueError):
        validate(dict(config, seed=6099))


def test_side_snapshots_leave_pto_weights_and_rng_unchanged(tmp_path):
    torch.manual_seed(11)
    model = torch.nn.Linear(4, 3)
    with torch.no_grad():
        model.weight.zero_()
        model.bias[:] = torch.tensor([0.0, 0.05, 0.0])
    chunks = [torch.randn(5, 4), torch.randn(5, 4)]
    groups = ["A"] * 5 + ["B"] * 5
    before = copy.deepcopy(model.state_dict())
    random_state = torch.get_rng_state().clone()
    steps = snapshot_side_steps(model, chunks, groups,
                                {"global_cap": 4, "local_caps": {"A": 2, "B": 3}},
                                6099, 1, tmp_path)
    assert all(torch.equal(model.state_dict()[key], value) for key, value in before.items())
    assert torch.equal(torch.get_rng_state(), random_state)
    assert all((tmp_path / f"epoch01_{arm}.pt").exists()
               for arm in ("joint", "global_dose", "sham"))
    assert all(steps[arm]["radius"] == steps["joint"]["radius"]
               for arm in ("joint", "global_dose", "sham"))
    assert steps["joint"]["hard_after_global"] <= 4
    assert steps["joint"]["hard_after_local"]["A"] <= 2
