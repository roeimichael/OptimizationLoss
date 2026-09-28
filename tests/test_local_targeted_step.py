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


def test_fixed_radius_ceiling_fails_without_changing_model():
    model, chunks, groups = fixture(bias=2.0)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="displacement ceiling"):
        local_targeted_step(model, chunks, groups, 1, 4, {"A": 2, "B": 3},
                            max_radius=0.1)
    assert all(torch.equal(model.state_dict()[k], value) for k, value in before.items())
