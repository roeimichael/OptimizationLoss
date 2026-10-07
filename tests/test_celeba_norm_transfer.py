"""New norm logging cases: exact reduction and bounded scalar extraction."""
import math

import pytest
import torch

from tralo.celeba_snapshot_core import _norm


def fixed_model(gradients):
    model = torch.nn.ParameterList([
        torch.nn.Parameter(torch.zeros_like(g)) for g in gradients
    ])
    for parameter, gradient in zip(model, gradients):
        parameter.grad = gradient.clone()
    model.append(torch.nn.Parameter(torch.zeros(1)))
    return model


def test_fixed_norm_uses_at_most_one_scalar_extraction():
    # A transfer once for every parameter violates the bounded logging contract.
    model = fixed_model([
        torch.tensor([3., 0.]), torch.tensor([4.]),
        torch.tensor([0., 0., 0.]), torch.tensor([0.]),
    ])
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        value = _norm(model)
    extractions = sum(e.count for e in profile.key_averages()
                      if e.key == 'aten::_local_scalar_dense')
    assert value == 5.
    assert extractions <= 1


def test_norm_preserves_original_float64_python_reduction_and_states():
    gradients = [torch.tensor([3., 4.]),
                 torch.tensor([1e-17, -2e-17, 0.]),
                 torch.tensor([1e10, -1e10]),
                 torch.arange(12, dtype=torch.float32).reshape(3, 4).T]
    model = fixed_model(gradients)
    model.train()
    rng = torch.get_rng_state().clone()
    parameters = [p.detach().clone() for p in model]
    before_gradients = [None if p.grad is None else p.grad.clone() for p in model]
    # This is the pinned existing reduction on new explicit gradient tensors.
    expected = math.sqrt(sum(float(g.double().square().sum()) for g in gradients))
    assert _norm(model).hex() == expected.hex()
    assert model.training and torch.equal(rng, torch.get_rng_state())
    for parameter, before, gradient in zip(model, parameters, before_gradients):
        assert torch.equal(parameter.detach(), before)
        assert (parameter.grad is None) == (gradient is None)
        if gradient is not None:
            assert torch.equal(parameter.grad, gradient)


@pytest.mark.parametrize('value', [float('inf'), float('nan')])
def test_nonfinite_norm_remains_visible_to_callers(value):
    result = _norm(fixed_model([torch.tensor([value])]))
    assert not math.isfinite(result)


def test_missing_gradients_keep_zero_norm():
    assert _norm(torch.nn.Linear(2, 1)) == 0.
