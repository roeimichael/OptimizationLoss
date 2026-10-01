"""Independent checks for a genuine pooled-plus-tabular TraLO objective."""

import pytest

torch = pytest.importorskip("torch")

from tralo.local_bounded_penalty import (
    bounded_local_count_penalty, bounded_local_logit_gradient)
from tralo.global_constraint import bounded_count_penalty


def test_hand_computed_two_scope_objective_and_slack_group():
    logits = torch.tensor([[2.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
                          dtype=torch.double, requires_grad=True)
    groups = ["A", "A", "B"]
    q = logits.softmax(1)[:, 0]
    global_e = torch.relu(q.sum() - 1.0)
    a_e = torch.relu(q[:2].sum() - 1.0)
    expected = (2.0 * (global_e / (1 + global_e) +
                       0.5 * global_e.square() / (1 + global_e.square())) +
                3.0 * (a_e / (1 + a_e) +
                       0.5 * a_e.square() / (1 + a_e.square())))
    actual = bounded_local_count_penalty(logits, groups, 0, 1,
                                         {"A": 1, "B": 1},
                                         {"global": 2.0, "A": 3.0, "B": 7.0},
                                         0.5)
    assert torch.allclose(actual, expected)
    assert q[-1] < 1  # B is slack and contributes no coefficient.
    gradient = torch.autograd.grad(actual, logits)[0]
    analytic = bounded_local_logit_gradient(logits.detach().softmax(1), groups,
                                            0, 1, {"A": 1, "B": 1},
                                            {"global": 2.0, "A": 3.0,
                                             "B": 7.0}, 0.5)
    assert torch.allclose(gradient, analytic, atol=1e-10, rtol=1e-10)


def test_finite_difference_matches_true_saturating_logit_derivative():
    logits = torch.tensor([[1.6, -0.2, 0.1], [0.9, 0.3, -0.2],
                           [0.3, 0.4, 0.2], [1.0, -0.1, 0.0]],
                          dtype=torch.double, requires_grad=True)
    groups = ["A", "A", "B", "B"]
    caps = {"A": 1, "B": 1}
    multipliers = {"global": 1.2, "A": 0.7, "B": 2.1}
    value = bounded_local_count_penalty(logits, groups, 0, 2, caps,
                                        multipliers, 0.4)
    autograd = torch.autograd.grad(value, logits)[0]
    analytic = bounded_local_logit_gradient(logits.detach().softmax(1), groups,
                                            0, 2, caps, multipliers, 0.4)
    assert torch.allclose(autograd, analytic, atol=1e-10, rtol=1e-10)
    for row in range(len(logits)):
        for col in range(logits.shape[1]):
            plus, minus = logits.detach().clone(), logits.detach().clone()
            plus[row, col] += 1e-6
            minus[row, col] -= 1e-6
            numerical = (bounded_local_count_penalty(plus, groups, 0, 2, caps,
                         multipliers, 0.4) - bounded_local_count_penalty(
                         minus, groups, 0, 2, caps, multipliers, 0.4)) / 2e-6
            assert analytic[row, col] == pytest.approx(float(numerical), abs=1e-8)


def test_all_slack_has_zero_gradient_and_metadata_is_required():
    logits = torch.tensor([[0.1, 2.0], [0.2, 1.8]], dtype=torch.double,
                          requires_grad=True)
    multipliers = {"global": 1., "female": 1., "male": 1.}
    args = (["female", "male"], 0, 2, {"female": 1, "male": 1},
            multipliers, 0.5)
    loss = bounded_local_count_penalty(logits, *args)
    assert loss.item() == 0.0
    assert torch.equal(torch.autograd.grad(loss, logits)[0],
                       torch.zeros_like(logits))
    with pytest.raises(ValueError, match="one nonempty metadata group"):
        bounded_local_count_penalty(logits.detach(), ["female"], *args[1:])
    with pytest.raises(ValueError, match="observed local caps"):
        bounded_local_count_penalty(logits.detach(), args[0], 0, 2,
                                    {"ghost": 1},
                                    {"global": 1., "ghost": 1.}, 0.5)


def test_missing_metadata_group_stays_in_global_pool_without_tiny_local_quota():
    logits = torch.tensor([[1.5, 0.0], [1.3, 0.1], [1.0, 0.2]],
                          dtype=torch.double, requires_grad=True)
    groups = ["female", "female", "missing"]
    caps = {"female": 1}
    multipliers = {"global": 1.4, "female": 0.7}
    loss = bounded_local_count_penalty(logits, groups, 0, 1, caps,
                                       multipliers, 0.5)
    actual = torch.autograd.grad(loss, logits)[0]
    analytic = bounded_local_logit_gradient(logits.detach().softmax(1), groups,
                                            0, 1, caps, multipliers, 0.5)
    assert torch.allclose(actual, analytic, atol=1e-10, rtol=1e-10)
    assert analytic[-1].abs().sum() > 0  # Missing metadata remains pooled.


def test_global_component_is_exact_original_tralo_penalty():
    logits = torch.tensor([[1.4, -0.1], [0.8, 0.0], [0.2, -0.4]],
                          dtype=torch.double, requires_grad=True)
    original = bounded_count_penalty(logits, [1, None],
                                     torch.tensor([1.7, 0.0], dtype=torch.double),
                                     0.35)
    extended = bounded_local_count_penalty(
        logits, ["female", "female", "male"], 0, 1,
        {"female": 2, "male": 1},
        {"global": 1.7, "female": 0.0, "male": 0.0}, 0.35)
    assert torch.allclose(original, extended, atol=1e-12, rtol=0)
    original_grad = torch.autograd.grad(original, logits, retain_graph=True)[0]
    extended_grad = torch.autograd.grad(extended, logits)[0]
    assert torch.allclose(original_grad, extended_grad, atol=1e-12, rtol=0)
