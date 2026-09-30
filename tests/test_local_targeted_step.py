import copy

import pytest
import torch

from tralo.local_targeted_step import local_count_logit_gradient, local_targeted_step


def fixture(seed=3, bias=0.05):
    torch.manual_seed(seed)
    model = torch.nn.Linear(4, 3)
    with torch.no_grad():
        model.weight.zero_()
        model.bias[:] = torch.tensor([0.0, bias, 0.0])
    chunks = [torch.randn(5, 4), torch.randn(5, 4)]
    groups = ["A"] * 5 + ["B"] * 5
    return model, chunks, groups


def test_joint_logit_gradient_matches_independent_autograd():
    logits = torch.tensor([[0.1, 1.0, -0.2], [0.3, 0.8, 0.5],
                           [0.0, 0.2, 0.7]], dtype=torch.double, requires_grad=True)
    groups = ["A", "A", "B"]
    p = logits.softmax(1)
    objective = p[:, 1].sum() / 2 + p[:2, 1].sum() / 1
    expected = torch.autograd.grad(objective, logits)[0]
    actual = local_count_logit_gradient(p.detach(), groups, 1, 2,
                                        {"A": 1, "B": 2}, True, {"A"})
    assert torch.allclose(actual, expected, atol=1e-12)


def test_chunked_parameter_gradient_matches_full_autograd_and_finite_difference():
    torch.manual_seed(23)
    model = torch.nn.Linear(2, 3, dtype=torch.double)
    images = torch.tensor([[0.2, 0.7], [-0.4, 0.1], [0.8, -0.3]], dtype=torch.double)
    groups = ["A", "B", "A"]
    params = list(model.parameters())

    def objective():
        p = model(images).softmax(1)[:, 1]
        return p.sum() / 2 + (p[0] + p[2]) / 1

    expected = torch.autograd.grad(objective(), params)
    probabilities = model(images).softmax(1).detach()
    upstream = local_count_logit_gradient(probabilities, groups, 1, 2,
                                           {"A": 1, "B": 2}, True, {"A"})
    model.zero_grad(set_to_none=True)
    for start, stop in ((0, 1), (1, 3)):
        logits = model(images[start:stop])
        logits.backward(upstream[start:stop])
    for actual, reference in zip((p.grad for p in params), expected):
        assert torch.allclose(actual, reference, atol=1e-12)
    with torch.no_grad():
        original = model.weight[1, 0].item()
        h = 1e-6
        model.weight[1, 0] = original + h
    plus = objective().item()
    with torch.no_grad():
        model.weight[1, 0] = original - h
    minus = objective().item()
    with torch.no_grad():
        model.weight[1, 0] = original
    assert abs((plus - minus) / (2 * h) - expected[0][1, 0].item()) < 1e-9


def test_joint_step_makes_all_hard_counts_feasible_and_sham_matches_dose():
    real, chunks, groups = fixture()
    sham = copy.deepcopy(real)
    caps = {"A": 2, "B": 3}
    r = local_targeted_step(real, chunks, groups, 1, 4, caps)
    s = local_targeted_step(sham, chunks, groups, 1, 4, caps,
                            sham_generator=torch.Generator().manual_seed(7),
                            fixed_radius=r["radius"])
    assert r["applied"] and r["hard_after_global"] <= 4
    assert all(r["hard_after_local"][g] <= caps[g] for g in caps)
    assert s["radius"] == r["radius"]
    assert abs(s["displacement"] - r["displacement"]) < 1e-5
    assert s["hard_before_local"] == r["hard_before_local"]
    global_control, _, _ = fixture()
    g = local_targeted_step(global_control, chunks, groups, 1, 4, caps,
                            fixed_radius=r["radius"], global_only_direction=True)
    assert g["radius"] == r["radius"] and abs(g["displacement"] - r["displacement"]) < 1e-5
    assert g["active_global"] and not g["active_local"]


def test_no_step_if_all_scopes_feasible():
    model, chunks, groups = fixture()
    before = copy.deepcopy(model.state_dict())
    out = local_targeted_step(model, chunks, groups, 1, 10, {"A": 5, "B": 5})
    assert not out["applied"]
    assert all(torch.equal(model.state_dict()[k], value) for k, value in before.items())


def test_invalid_group_mapping_and_failed_search_restore_weights():
    model, chunks, groups = fixture()
    with pytest.raises(ValueError):
        local_targeted_step(model, chunks, groups, 1, 4, {"A": 2})
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError):
        local_targeted_step(model, chunks, groups, 1, 4, {"A": 2, "B": 3},
                            max_doublings=0)
    assert all(torch.equal(model.state_dict()[k], value) for k, value in before.items())


def test_opposing_country_gradients_are_rejected_and_weights_restored():
    class Conflict(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            one = 2.0 + self.theta * x[:, 0]
            return torch.stack([torch.zeros_like(one), one], dim=1)

    model = Conflict()
    before = model.theta.detach().clone()
    chunks = [torch.tensor([[1.0], [1.0], [-1.0]])]
    with pytest.raises(RuntimeError, match="scope descent"):
        local_targeted_step(model, chunks, ["A", "A", "B"], 1, 3,
                            {"A": 0, "B": 0})
    assert torch.equal(model.theta, before)


def test_fixed_dose_records_conflicting_scopes_without_abort():
    class Conflict(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            one = 2.0 + self.theta * x[:, 0]
            return torch.stack([torch.zeros_like(one), one], dim=1)

    model = Conflict()
    out = local_targeted_step(model, [torch.tensor([[1.0], [1.0], [-1.0]])],
                              ["A", "A", "B"], 1, 3, {"A": 0, "B": 0},
                              fixed_radius=0.1, require_common_descent=False)
    assert out["applied"]
    assert out["gradient_norm"] > 0
    assert abs(out["displacement"] - 0.1) < 1e-6
    assert any(value >= 0 for value in out["scope_directional_derivatives"].values())


def test_fixed_radius_ceiling_fails_without_changing_model():
    model, chunks, groups = fixture(bias=2.0)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="displacement ceiling"):
        local_targeted_step(model, chunks, groups, 1, 4, {"A": 2, "B": 3},
                            max_radius=0.1)
    assert all(torch.equal(model.state_dict()[k], value) for k, value in before.items())


def test_fixed_dose_reports_raw_infeasibility_without_aborting():
    model, chunks, groups = fixture(bias=2.0)
    out = local_targeted_step(model, chunks, groups, 1, 4, {"A": 2, "B": 3},
                              fixed_radius=0.1, require_common_descent=False)
    assert out["applied"]
    assert abs(out["radius"] - 0.1) < 1e-12
    assert abs(out["displacement"] - 0.1) < 1e-5
    assert out["hard_after_global"] > 4


class BiasOnly(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0.0))
        self.bias = bias
        self.bn = torch.nn.BatchNorm1d(1)

    def forward(self, images):
        one = self.bias + self.theta + self.bn(images)[:, 0] * 0.0
        return torch.stack([torch.zeros_like(one), one], dim=1)


@pytest.mark.parametrize("bias,expected_radius,expected_probes", [
    (0.2, 0.1, 1),
    (0.06, 0.05, 2),
])
def test_boundary_accepts_first_safe_radius_or_backtracks(
        bias, expected_radius, expected_probes):
    model = BiasOnly(bias)
    model.train()
    chunks = [torch.zeros(10, 1)]
    groups = ["A"] * 5 + ["B"] * 5
    rng_before = torch.random.get_rng_state().clone()
    buffers_before = {key: value.detach().clone()
                      for key, value in model.named_buffers()}
    out = local_targeted_step(model, chunks, groups, 1, 2,
                              {"A": 1, "B": 1}, boundary_calibrated=True)
    assert out["applied"]
    assert model.training
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert all(torch.equal(value, buffers_before[key])
               for key, value in model.named_buffers())
    assert out["radius"] == pytest.approx(expected_radius)
    assert out["displacement"] == pytest.approx(expected_radius, abs=1e-7)
    assert out["hard_after_global"] == 10
    assert len(out["boundary_policy"]["probes"]) == expected_probes
    assert out["boundary_policy"]["probes"][-1]["accepted"]
    q = torch.sigmoid(torch.tensor(bias)).item()
    expected_derivative = -5 * q * (1 - q)
    assert out["scope_directional_derivatives"]["pooled"] == pytest.approx(
        expected_derivative, abs=1e-6)
    assert out["scope_directional_derivatives"]["local:A"] == pytest.approx(
        expected_derivative, abs=1e-6)
    assert all(set(record["local_hard"]) == {"A", "B"}
               for record in out["boundary_policy"]["probes"])
    if expected_probes == 2:
        assert out["boundary_policy"]["probes"][0]["pooled_hard"] == 0
        assert "pooled_hard_floor" in out["boundary_policy"]["probes"][0]["rejections"]


def test_boundary_checks_soft_violated_scope_even_when_hard_inactive():
    class CountryConflict(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            one = 0.2 + self.theta * x[:, 0] - 0.4 * x[:, 1]
            return torch.stack([torch.zeros_like(one), one], dim=1)

    model = CountryConflict()
    original = model.theta.detach().clone()
    chunks = [torch.tensor([[1.0, 0.0]] * 3 + [[-1.0, 1.0]] * 4)]
    groups = ["A"] * 3 + ["B"] * 4
    out = local_targeted_step(model, chunks, groups, 1, 2,
                              {"A": 1, "B": 1}, boundary_calibrated=True)
    assert out["hard_before_local"]["B"] == 0
    assert out["active_local"] == ["A"]
    assert out["soft_before_local"]["B"] > 1
    assert out["scope_directional_derivatives"]["local:B"] > 0
    assert out["boundary_policy"]["reason"] == "conflicting_direction"
    assert "local:B" in out["boundary_policy"]["conflicting_scopes"]
    assert not out["applied"] and out["evaluations"] == 1
    assert torch.equal(model.theta, original)


def test_boundary_no_positive_soft_violation_skips_and_preserves_old_path():
    boundary, chunks, groups = fixture()
    reference = copy.deepcopy(boundary)
    # Hard calls exceed these caps, but the soft probabilities do not.
    skipped = local_targeted_step(boundary, chunks, groups, 1, 4,
                                  {"A": 2, "B": 3}, boundary_calibrated=True)
    assert skipped["active_global"]
    assert skipped["boundary_policy"]["reason"] == "no_positive_violation"
    assert not skipped["applied"]
    assert all(torch.equal(boundary.state_dict()[key], value)
               for key, value in reference.state_dict().items())
    old = local_targeted_step(reference, chunks, groups, 1, 4, {"A": 2, "B": 3})
    assert old["applied"] and old["hard_after_global"] <= 4
    assert "boundary_policy" not in old


def test_boundary_skips_soft_only_violation_without_a_hard_active_direction():
    model = BiasOnly(-0.2)
    original = copy.deepcopy(model.state_dict())
    out = local_targeted_step(model, [torch.zeros(10, 1)],
                              ["A"] * 5 + ["B"] * 5, 1, 2,
                              {"A": 1, "B": 1}, boundary_calibrated=True)
    assert out["hard_before_global"] == 0
    assert out["soft_before_global"] > 2
    assert out["skip_reason"] == "no_hard_active_scope"
    assert not out["boundary_policy"]["probes"]
    assert all(torch.equal(model.state_dict()[key], value)
               for key, value in original.items())


def test_boundary_rejects_incompatible_controls_without_model_mutation():
    model = BiasOnly(0.2)
    original = model.theta.detach().clone()
    with pytest.raises(ValueError, match="joint real direction"):
        local_targeted_step(model, [torch.zeros(10, 1)], ["A"] * 10, 1,
                            2, {"A": 1}, boundary_calibrated=True,
                            fixed_radius=0.05)
    assert torch.equal(model.theta, original)
