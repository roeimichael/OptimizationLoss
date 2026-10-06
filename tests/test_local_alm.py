import pytest
import torch

from tralo.alm import augmented_penalty
from tralo.local_alm import projected_dual, residuals_and_gradient, snapshot_phr_step


def test_signed_residuals_penalty_and_logit_gradient_match_autograd():
    logits = torch.tensor([[0.1, 1.0, -0.2], [0.3, 0.8, 0.5],
                           [0.0, 0.2, 0.7]], dtype=torch.double, requires_grad=True)
    groups = ["B", "A", "B"]
    caps = {"A": 0, "B": 1}
    dual = torch.tensor([0.2, 0.3, 0.0], dtype=torch.double)
    rho = 0.7
    probabilities = logits.softmax(1)
    residuals, penalty, analytic = residuals_and_gradient(
        probabilities.detach(), groups, 1, 2, caps, dual, rho)
    q = probabilities[:, 1]
    reference_residuals = torch.stack([(q.sum() - 2) / 2, q[1:2].sum(),
                                       q[[0, 2]].sum() - 1])
    reference = augmented_penalty(reference_residuals, dual, rho)
    gradient = torch.autograd.grad(reference, logits)[0]
    assert torch.allclose(residuals, reference_residuals.detach(), atol=1e-12)
    assert torch.allclose(penalty, reference.detach(), atol=1e-12)
    assert torch.allclose(analytic, gradient, atol=1e-12)


def test_slack_projection_and_no_step_dual_update():
    p = torch.tensor([[0.8, 0.2], [0.7, 0.3]], dtype=torch.double)
    dual = torch.zeros(3, dtype=torch.double)
    residuals, penalty, gradient = residuals_and_gradient(
        p, ["A", "B"], 1, 2, {"A": 1, "B": 1}, dual, 0.5)
    assert torch.all(residuals < 0)
    assert penalty.item() == 0
    assert torch.count_nonzero(gradient) == 0
    assert torch.equal(projected_dual(residuals, dual, 0.5), dual)


def test_group_order_and_sample_permutation():
    p = torch.tensor([[0.1, 0.9], [0.6, 0.4], [0.3, 0.7]], dtype=torch.double)
    groups = ["Z", "A", "Z"]
    caps = {"Z": 1, "A": 0}
    dual = torch.tensor([0.0, 0.2, 0.4], dtype=torch.double)
    first = residuals_and_gradient(p, groups, 1, 1, caps, dual, 0.5)
    order = [2, 0, 1]
    second = residuals_and_gradient(p[order], [groups[i] for i in order], 1,
                                    1, caps, dual, 0.5)
    assert torch.allclose(first[0], second[0])
    assert torch.allclose(first[1], second[1])
    assert torch.allclose(first[2][order], second[2])


def test_analytic_gradient_matches_central_difference():
    logits = torch.tensor([[0.3, 0.8], [-0.4, 1.1]], dtype=torch.double)
    dual = torch.tensor([0.1, 0.2, 0.0], dtype=torch.double)
    groups, caps = ["A", "B"], {"A": 0, "B": 0}

    def value(z):
        return residuals_and_gradient(z.softmax(1), groups, 1, 1,
                                      caps, dual, 0.5)[1].item()

    analytic = residuals_and_gradient(logits.softmax(1), groups, 1, 1,
                                      caps, dual, 0.5)[2]
    h = 1e-6
    for row in range(2):
        for column in range(2):
            plus, minus = logits.clone(), logits.clone()
            plus[row, column] += h
            minus[row, column] -= h
            assert abs((value(plus) - value(minus)) / (2 * h) -
                       analytic[row, column].item()) < 1e-9


def test_chunked_model_replay_matches_full_phr_autograd():
    torch.manual_seed(31)
    model = torch.nn.Linear(3, 2, dtype=torch.double)
    images = torch.tensor([[0.2, 0.1, -0.3], [-0.4, 0.7, 0.5],
                           [0.8, -0.1, 0.2]], dtype=torch.double)
    groups = ["A", "B", "A"]
    caps = {"A": 1, "B": 0}
    dual = torch.tensor([0.1, 0.0, 0.2], dtype=torch.double)
    logits = model(images)
    p = logits.softmax(1)
    residuals = torch.stack([(p[:, 1].sum() - 1),
                             (p[[0, 2], 1].sum() - 1), p[1, 1]])
    expected = torch.autograd.grad(augmented_penalty(residuals, dual, 0.5),
                                   tuple(model.parameters()))
    upstream = residuals_and_gradient(p.detach(), groups, 1, 1,
                                      caps, dual, 0.5)[2]
    model.zero_grad(set_to_none=True)
    for start, stop in ((0, 1), (1, 3)):
        model(images[start:stop]).backward(upstream[start:stop])
    for parameter, reference in zip(model.parameters(), expected):
        assert torch.allclose(parameter.grad, reference, atol=1e-12)


@pytest.mark.parametrize("bad", [
    lambda p, g, c, d: (p, g, c, d[:1]),
    lambda p, g, c, d: (p, ["A", "C"], c, d),
    lambda p, g, c, d: (p, g, {"A": -1, "B": 1}, d),
    lambda p, g, c, d: (p * 2, g, c, d),
])
def test_invalid_inputs_rejected(bad):
    p = torch.tensor([[0.8, 0.2], [0.7, 0.3]])
    q, groups, caps, dual = bad(p, ["A", "B"], {"A": 1, "B": 1}, torch.zeros(3))
    with pytest.raises(ValueError):
        residuals_and_gradient(q, groups, 1, 2, caps, dual, 0.5)


def test_nonfinite_dual_residual_rejected():
    with pytest.raises(ValueError, match="finite"):
        projected_dual(torch.tensor([float("nan")]), torch.zeros(1), 0.5)


def test_snapshot_step_has_fixed_dose_and_neutral_buffers_rng():
    torch.manual_seed(41)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2),
                                torch.nn.Linear(2, 2))
    model.train()
    with torch.no_grad():
        model[2].bias[:] = torch.tensor([0.0, 2.0])
    images = [torch.tensor([[1.0, 0.2], [0.3, -0.1]]),
              torch.tensor([[-0.6, 0.4], [0.5, 0.8]])]
    groups = ["A", "A", "B", "B"]
    original = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    record, dual = snapshot_phr_step(model, images, groups, 1, 1,
                                     {"A": 0, "B": 1}, torch.zeros(3))
    assert record["applied"]
    assert abs(record["displacement"] - 0.1) < 1e-5
    assert torch.any(dual > 0)
    assert record["scope_derivative_schema"] == "pooled-local-v1"
    assert set(record["scope_directional_derivatives"]) == {"pooled", "local:A", "local:B"}
    coefficients = [(lam + 0.5 * g) for lam, g in
                    zip(record["dual_before"], record["residuals_before"])]
    weighted = sum(max(0, coefficient) * record["scope_directional_derivatives"][name]
                    for coefficient, name in zip(coefficients, ("pooled", "local:A", "local:B")))
    assert abs(weighted + record["gradient_norm"]) < 1e-4
    assert record["hard_before_global"] == 4
    assert sum(record["hard_after_local"].values()) == record["hard_after_global"]
    assert model.training
    assert torch.equal(torch.get_rng_state(), rng)
    for key in ("1.running_mean", "1.running_var", "1.num_batches_tracked"):
        assert torch.equal(model.state_dict()[key], original[key])


def test_slack_snapshot_does_not_move_model_or_dual():
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias[:] = torch.tensor([2.0, 0.0])
    before = {key: value.clone() for key, value in model.state_dict().items()}
    images = [torch.zeros(2, 2)]
    record, dual = snapshot_phr_step(model, images, ["A", "B"], 1, 2,
                                     {"A": 1, "B": 1}, torch.zeros(3))
    assert not record["applied"] and record["gradient_norm"] == 0
    assert torch.equal(dual, torch.zeros(3))
    assert all(torch.equal(model.state_dict()[key], value) for key, value in before.items())


def test_boundary_phr_step_records_real_probe_and_preserves_buffers_rng():
    torch.manual_seed(41)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2),
                                torch.nn.Linear(2, 2))
    model.train()
    with torch.no_grad():
        model[2].bias[:] = torch.tensor([0.0, 1.0])
    images = [torch.tensor([[1.0, 0.2], [0.3, -0.1]]),
              torch.tensor([[-0.6, 0.4], [0.5, 0.8]])]
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    record, dual = snapshot_phr_step(
        model, images, ["A", "A", "B", "B"], 1, 1,
        {"A": 0, "B": 1}, torch.zeros(3), boundary_calibrated=True)
    policy = record["boundary_policy"]
    assert record["applied"] and policy["applied"]
    assert 0 < record["radius"] <= 0.1
    assert abs(record["displacement"] - record["radius"]) < 1e-5
    assert policy["probes"] and policy["probes"][-1]["accepted"]
    assert record["activation_reason"] == "boundary_accepted"
    assert torch.any(dual > 0)
    assert model.training and torch.equal(torch.get_rng_state(), rng)
    for key in ("1.running_mean", "1.running_var", "1.num_batches_tracked"):
        assert torch.equal(model.state_dict()[key], before[key])


def test_boundary_phr_slack_does_not_move_model_or_dual():
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias[:] = torch.tensor([2.0, 0.0])
    before = {key: value.clone() for key, value in model.state_dict().items()}
    record, dual = snapshot_phr_step(
        model, [torch.zeros(2, 2)], ["A", "B"], 1, 2,
        {"A": 1, "B": 1}, torch.zeros(3), boundary_calibrated=True)
    assert not record["applied"] and record["radius"] == 0.0
    assert record["boundary_policy"]["reason"] == "zero_phr_gradient"
    assert torch.equal(dual, torch.zeros(3))
    assert all(torch.equal(model.state_dict()[key], value) for key, value in before.items())


def test_boundary_phr_rejects_a_final_replay_mismatch(monkeypatch):
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias[:] = torch.tensor([0.0, 1.0])
    original = {key: value.clone() for key, value in model.state_dict().items()}

    def forged_policy(*args, **kwargs):
        return dict(applied=True, radius=0.05, reason="accepted", probes=[
            dict(pooled_hard=999, pooled_soft=999.0,
                 local_soft={"A": 999.0, "B": 0.0})])

    monkeypatch.setattr("tralo.local_alm.choose_boundary_step", forged_policy)
    with pytest.raises(RuntimeError, match="did not replay exactly"):
        snapshot_phr_step(model, [torch.zeros(2, 2)], ["A", "B"], 1, 1,
                          {"A": 0, "B": 0}, torch.zeros(3),
                          boundary_calibrated=True)
    assert all(torch.equal(model.state_dict()[key], value)
               for key, value in original.items())


def test_boundary_phr_restores_a_rejected_probe(monkeypatch):
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias[:] = torch.tensor([0.0, 1.0])
    original = {key: value.clone() for key, value in model.state_dict().items()}

    def rejecting_policy(*args, **kwargs):
        probe = args[7]
        probe(0.1)
        return dict(applied=False, radius=0.0, reason="no_acceptable_probe",
                    probes=[dict(radius=0.1, accepted=False)])

    monkeypatch.setattr("tralo.local_alm.choose_boundary_step", rejecting_policy)
    record, _ = snapshot_phr_step(
        model, [torch.zeros(2, 2)], ["A", "B"], 1, 1,
        {"A": 0, "B": 0}, torch.zeros(3), boundary_calibrated=True)
    assert not record["applied"] and record["radius"] == 0.0
    assert record["activation_reason"] == "boundary_no_acceptable_probe"
    assert all(torch.equal(model.state_dict()[key], value)
               for key, value in original.items())


def test_boundary_phr_rejects_and_restores_mutable_eval_buffer():
    class MutableEval(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.register_buffer("calls", torch.zeros((), dtype=torch.long))

        def forward(self, images):
            self.calls.add_(1)
            return self.linear(images)

    model = MutableEval()
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias[:] = torch.tensor([0.0, 1.0])
    original = model.calls.clone()
    with pytest.raises(RuntimeError, match="changed model buffers"):
        snapshot_phr_step(model, [torch.zeros(2, 2)], ["A", "B"], 1, 1,
                          {"A": 0, "B": 0}, torch.zeros(3),
                          boundary_calibrated=True)
    assert torch.equal(model.calls, original)
