import copy
import json

import pytest
import torch

from tralo.fmow_local import RECIPE, budgets, run, snapshot_side_steps, validate
from tralo.knee_end_to_end import infer


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


def test_boundary_study_config_has_its_own_fixed_seed_block():
    config = dict(RECIPE, seed=6400, snapshot_steps=True,
                  study="local_boundary_v1", step_radius=0.1, alm_rho=0.5)
    validate(config)
    for seed in range(6401, 6413):
        validate(dict(config, seed=seed))
    for invalid in (6099, 6199, 6300, 6413):
        with pytest.raises(ValueError):
            validate(dict(config, seed=invalid))
    for key, value in (("step_radius", 0.2), ("alm_rho", 0.4),
                       ("study", "local_boundary_v2")):
        with pytest.raises(ValueError):
            validate(dict(config, **{key: value}))
    with pytest.raises(ValueError):
        validate(dict(config, snapshot_steps=False, seed=6401))
    validate(dict(config, snapshot_steps=False))
    validate(dict(RECIPE, seed=6199, snapshot_steps=True,
                  study="local_fixed_dose_v1", step_radius=0.1))
    validate(dict(RECIPE, seed=6300, snapshot_steps=True,
                  study="local_alm_direction_v1", step_radius=0.1, alm_rho=0.5))


def test_boundary_run_requests_no_development_labels(tmp_path, monkeypatch):
    config = dict(RECIPE, seed=6400, snapshot_steps=False,
                  study="local_boundary_v1", step_radius=0.1, alm_rho=0.5)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    seen = []

    class ReachedLoader(Exception):
        pass

    def preflight_loader(_root, *, include_pool_labels):
        seen.append(include_pool_labels)
        raise ReachedLoader

    monkeypatch.setattr("tralo.fmow_local.cuda_setup", lambda: None)
    monkeypatch.setattr("tralo.fmow_local.load", preflight_loader)
    with pytest.raises(ReachedLoader):
        run(tmp_path, path, tmp_path / "output")
    assert seen == [False]


class _BiasOnly(torch.nn.Module):
    def __init__(self, bias=0.2):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0.0))
        self.bias = bias
        self.bn = torch.nn.BatchNorm1d(1)

    def forward(self, images):
        one = self.bias + self.theta + self.bn(images)[:, 0] * 0.0
        return torch.stack([torch.zeros_like(one), one], dim=1)


@pytest.mark.parametrize("cap,local_caps,expect_joint", [
    (2, {"A": 1, "B": 1}, True),
    (10, {"A": 5, "B": 5}, False),
])
def test_boundary_snapshot_matches_controls_and_preserves_pto(
        tmp_path, cap, local_caps, expect_joint):
    torch.manual_seed(41)
    model = _BiasOnly()
    model.train()
    pool = [torch.zeros(10, 1)]
    groups = ["A"] * 5 + ["B"] * 5
    original_state = copy.deepcopy(model.state_dict())
    original_rng = torch.get_rng_state().clone()
    original_predictions = infer(model, pool)
    phr_state = {"dual": torch.zeros(3)}
    steps = snapshot_side_steps(
        model, pool, groups, {"global_cap": cap, "local_caps": local_caps},
        6400, 1, tmp_path, fixed_radius=0.1, phr_state=phr_state,
        phr_rho=0.5, boundary_calibrated=True,
        pto_probabilities=original_predictions)

    assert model.training
    assert all(torch.equal(model.state_dict()[key], value)
               for key, value in original_state.items())
    assert torch.equal(torch.get_rng_state(), original_rng)
    assert steps["joint"]["applied"] == expect_joint
    assert "boundary_policy" in steps["joint"]
    assert "boundary_policy" in steps["phr_local"]
    assert set(steps) == {"joint", "global_dose", "sham", "phr_local"}
    if expect_joint:
        assert steps["joint"]["radius"] <= 0.1
        assert steps["phr_local"]["applied"]
        assert 0 < steps["phr_local"]["radius"] <= 0.1
        assert steps["phr_local"]["boundary_policy"]["applied"]
        assert steps["joint"]["radius"] == steps["global_dose"]["radius"]
        assert steps["joint"]["radius"] == steps["sham"]["radius"]
        assert steps["global_dose"]["applied"] and steps["sham"]["applied"]
        assert len(steps["joint"]["tensor_displacement_norms"]) == len(
            steps["sham"]["tensor_displacement_norms"])
        assert all(abs(a - b) <= 1e-5 for a, b in zip(
            steps["joint"]["tensor_displacement_norms"],
            steps["sham"]["tensor_displacement_norms"]))
    else:
        assert all(not steps[arm]["applied"] for arm in
                   ("joint", "global_dose", "sham", "phr_local"))
        assert all(steps[arm]["radius"] == 0.0 for arm in
                   ("joint", "global_dose", "sham", "phr_local"))
        assert all(torch.equal(torch.load(tmp_path / f"epoch01_{arm}.pt",
                                          weights_only=True), original_predictions)
                   for arm in ("joint", "global_dose", "sham", "phr_local"))
        assert torch.equal(phr_state["dual"], torch.zeros(3))
    assert (tmp_path / "epoch01_phr_local.pt").exists()


def test_boundary_skip_rejects_a_false_pto_reference(tmp_path):
    model = _BiasOnly()
    pool = [torch.zeros(10, 1)]
    false_reference = infer(model, pool).clone()
    false_reference[0, 1] = -1
    with pytest.raises(RuntimeError, match="differs from PTO probabilities"):
        snapshot_side_steps(
            model, pool, ["A"] * 5 + ["B"] * 5,
            {"global_cap": 10, "local_caps": {"A": 5, "B": 5}},
            6400, 1, tmp_path, fixed_radius=0.1,
            phr_state={"dual": torch.zeros(3)}, phr_rho=0.5,
            boundary_calibrated=True, pto_probabilities=false_reference)


def test_boundary_sham_matches_two_nonzero_parameter_tensor_doses(tmp_path):
    model = torch.nn.Linear(4, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias[:] = torch.tensor([0.0, 0.2])
    pool = [torch.ones(10, 4)]
    steps = snapshot_side_steps(
        model, pool, ["A"] * 5 + ["B"] * 5,
        {"global_cap": 2, "local_caps": {"A": 1, "B": 1}},
        6400, 1, tmp_path, fixed_radius=0.1,
        phr_state={"dual": torch.zeros(3)}, phr_rho=0.5,
        boundary_calibrated=True, pto_probabilities=infer(model, pool))
    assert steps["joint"]["applied"]
    assert all(x > 0 for x in steps["joint"]["tensor_displacement_norms"])
    assert all(abs(joint - sham) <= 1e-5 for joint, sham in zip(
        steps["joint"]["tensor_displacement_norms"],
        steps["sham"]["tensor_displacement_norms"]))


@pytest.mark.parametrize("cap,local_caps", [
    (2, {"A": 1, "B": 1}),
    (10, {"A": 5, "B": 5}),
])
def test_boundary_runner_records_pass_independent_side_audit(
        tmp_path, cap, local_caps):
    from analysis.score_fmow_boundary import _audit_side

    model = _BiasOnly()
    model.train()
    pool = [torch.zeros(10, 1)]
    groups = ["A"] * 5 + ["B"] * 5
    quota = {"global_cap": cap, "local_caps": local_caps}
    pto = infer(model, pool).cpu()
    records = snapshot_side_steps(
        model, pool, groups, quota, 6400, 1, tmp_path,
        fixed_radius=0.1, phr_state={"dual": torch.zeros(3)},
        phr_rho=0.5, boundary_calibrated=True, pto_probabilities=pto)
    dual = [0.0, 0.0, 0.0]
    for arm in ("joint", "global_dose", "sham", "phr_local"):
        side = torch.load(tmp_path / f"epoch01_{arm}.pt", weights_only=True)
        dual = _audit_side(records[arm], pto, side, groups, quota, arm, dual)
