"""Two-pass image-pool gradients match direct TraLO and PHR autograd."""

import copy

import pytest

torch = pytest.importorskip("torch")

from tralo.local_bounded_penalty import bounded_local_count_penalty
from tralo.tabular_constraint_gradient import (
    apply_fixed_correction, streaming_parameter_gradient)


def _setup():
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 2, bias=False).double().eval()
    images = torch.tensor([[1.0, 0.2], [0.8, 0.4], [0.4, 0.8], [0.1, 1.0]],
                          dtype=torch.double)
    groups = ["female", "female", "male", "male"]
    quota = {"global_cap": 1, "local_caps": {"female": 1, "male": 1}}
    def batches():
        yield images[:2], ["a", "b"]
        yield images[2:], ["c", "d"]
    return model, images, groups, quota, batches


def test_streaming_tralo_matches_exact_full_pool_autograd():
    model, images, groups, quota, batches = _setup()
    multipliers = {"global": 1.2, "female": 0.8, "male": 0.4}
    direct = copy.deepcopy(model)
    value = bounded_local_count_penalty(direct(images), groups, 1, 1,
                                        quota["local_caps"], multipliers, 0.5)
    value.backward()
    report = streaming_parameter_gradient(model, batches, groups, quota, "tralo",
                                          multipliers=multipliers, rho=0.5)
    assert report["active"] and report["fixed_weight_probability_max_abs_gap"] == 0
    assert report["sample_count"] == 4 and report["next_dual"] is None
    assert torch.allclose(model.weight.grad, direct.weight.grad, atol=1e-12, rtol=1e-12)


def test_streaming_phr_matches_direct_augmented_objective_and_dual_update():
    model, images, groups, quota, batches = _setup()
    dual = {"global": 0.3, "female": 0.1, "male": 0.2}
    rho = 0.7
    direct = copy.deepcopy(model)
    q = direct(images).softmax(1)[:, 1]
    residuals = {"global": (q.sum() - 1), "female": (q[:2].sum() - 1),
                 "male": (q[2:].sum() - 1)}
    value = sum((torch.relu(dual[scope] + rho * residual).square() -
                 dual[scope] ** 2) / (2 * rho)
                for scope, residual in residuals.items())
    value.backward()
    report = streaming_parameter_gradient(model, batches, groups, quota, "phr",
                                          dual=dual, rho=rho)
    assert torch.allclose(model.weight.grad, direct.weight.grad, atol=1e-12, rtol=1e-12)
    for scope, residual in residuals.items():
        assert report["next_dual"][scope] == pytest.approx(
            max(0., dual[scope] + rho * float(residual)))


def test_second_pass_order_change_is_rejected_before_update_claim():
    model, images, groups, quota, _ = _setup()
    count = 0
    def bad_batches():
        nonlocal count
        count += 1
        ids = ["a", "b", "c", "d"] if count == 1 else ["b", "a", "c", "d"]
        yield images, ids
    with pytest.raises(RuntimeError, match="order changed"):
        streaming_parameter_gradient(model, bad_batches, groups, quota, "tralo",
            multipliers={"global": 1., "female": 1., "male": 1.})


def test_fixed_correction_preserves_loss_magnitude_and_dose_limit():
    model, images, groups, quota, batches = _setup()
    streaming_parameter_gradient(model, batches, groups, quota, "tralo",
        multipliers={"global": 1., "female": 1., "male": 1.})
    initial = model.weight.detach().clone()
    gradient = model.weight.grad.detach().clone()
    refused = apply_fixed_correction(model, step_size=1000., max_displacement=0.01)
    assert refused["reason"] == "dose_limit"
    assert torch.equal(model.weight, initial)
    applied = apply_fixed_correction(model, step_size=0.001, max_displacement=0.1)
    assert applied["applied"]
    assert torch.allclose(model.weight, initial - 0.001 * gradient,
                          atol=1e-12, rtol=1e-12)
    assert applied["actual_displacement_norm"] == pytest.approx(
        float((model.weight - initial).double().norm()))


def test_slack_pool_reports_no_correction_without_failing_training():
    model, _, groups, _, batches = _setup()
    quota = {"global_cap": 4, "local_caps": {"female": 2, "male": 2}}
    report = streaming_parameter_gradient(model, batches, groups, quota, "tralo",
        multipliers={"global": 1., "female": 1., "male": 1.})
    assert not report["active"] and report["parameter_gradient_norm"] == 0
    before = model.weight.detach().clone()
    correction = apply_fixed_correction(model, step_size=0.01,
                                        max_displacement=0.1)
    assert correction["reason"] == "slack" and not correction["applied"]
    assert torch.equal(model.weight, before)
