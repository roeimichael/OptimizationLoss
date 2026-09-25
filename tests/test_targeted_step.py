import copy
import math

import pytest
import torch

from tralo.targeted_step import targeted_step

CAPS = [None, None, None, 20, None]


def fixture(bias=1.0, seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(6, 16), torch.nn.ReLU(), torch.nn.Linear(16, 5))
    with torch.no_grad():
        model[2].bias[3] += bias      # grade 3 over-called: the hard cap of 20 is violated
    chunks = [torch.randn(30, 6), torch.randn(25, 6)]
    return model, chunks


def hard(model, chunks):
    with torch.no_grad():
        return int((torch.cat([model(x) for x in chunks]).argmax(1) == 3).sum())


def soft_count_unit_descent(model, chunks):
    """Independent autograd reference: unit steepest-descent direction of the grade-3 soft count."""
    model.zero_grad()
    torch.cat([model(x).softmax(1) for x in chunks])[:, 3].sum().backward()
    grads = [p.grad.detach().clone() for p in model.parameters()]
    norm = math.sqrt(sum(float(g.double().square().sum()) for g in grads))
    model.zero_grad()
    return [-g / norm for g in grads]


def moved_to(model, direction, r):
    out = copy.deepcopy(model)
    with torch.no_grad():
        for p, d in zip(out.parameters(), direction):
            p.add_(r * d)
    return out


def test_fixture_violates_the_hard_cap():
    model, chunks = fixture()
    assert hard(model, chunks) > CAPS[3]


def test_targeted_step_lands_on_the_cap_with_the_smallest_radius_along_the_soft_count_gradient():
    model, chunks = fixture()
    start = copy.deepcopy(model)
    out = targeted_step(model, chunks, CAPS)
    assert out['applied'] and out['hard_after'] <= CAPS[3] and hard(model, chunks) == out['hard_after']
    assert abs(out['displacement'] - out['radius']) / out['radius'] < 1e-4
    unit = soft_count_unit_descent(start, chunks)
    # the applied step IS radius x the independent reference direction
    for p, s, u in zip(model.parameters(), start.parameters(), unit):
        assert torch.allclose(p - s, out['radius'] * u, atol=1e-6)
    # and it is minimal: just short of the radius, the cap is still violated
    assert hard(moved_to(start, unit, out['radius_violating']), chunks) > CAPS[3]
    assert (out['radius'] - out['radius_violating']) / out['radius'] < 1e-4


def test_no_step_when_the_hard_count_already_meets_the_cap():
    model, chunks = fixture(bias=-3.0)
    assert hard(model, chunks) <= CAPS[3]
    start = copy.deepcopy(model)
    out = targeted_step(model, chunks, CAPS)
    assert not out['applied'] and out['displacement'] == 0.0
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), start.parameters()))


def test_triggers_on_the_hard_count_even_when_the_soft_count_is_under_the_cap():
    """Audit D2: a soft-count trigger misses a hard violation. Find such a state and require a step."""
    for bias in [x / 20 for x in range(-40, 40)]:
        model, chunks = fixture(bias=bias)
        with torch.no_grad():
            soft = float(torch.cat([model(x).softmax(1) for x in chunks])[:, 3].sum())
        if hard(model, chunks) > CAPS[3] and soft <= CAPS[3]:
            break
    else:
        pytest.skip('fixture has no hard-violating, soft-feasible state')
    out = targeted_step(model, chunks, CAPS)
    assert out['applied'] and out['hard_after'] <= CAPS[3]


def test_sham_moves_the_same_radius_with_the_same_per_tensor_norms_in_a_random_direction():
    real, chunks = fixture()
    sham = copy.deepcopy(real)
    start = copy.deepcopy(real)
    r = targeted_step(real, chunks, CAPS)
    s = targeted_step(sham, chunks, CAPS, sham_generator=torch.Generator().manual_seed(5))
    assert s['radius'] == r['radius']
    assert abs(s['displacement'] - r['displacement']) / r['displacement'] < 1e-4
    for a, b, z in zip(real.parameters(), sham.parameters(), start.parameters()):
        da, db = (a - z).double().flatten(), (b - z).double().flatten()
        assert abs(float(db.norm()) - float(da.norm())) <= 1e-5 * max(float(da.norm()), 1e-12) + 1e-7
        if da.numel() >= 8 and float(da.norm()) > 0:
            assert abs(float(torch.dot(da, db) / (da.norm() * db.norm()))) < 0.9


def test_sham_is_deterministic_given_its_generator():
    def run(seed):
        model, chunks = fixture()
        targeted_step(model, chunks, CAPS, sham_generator=torch.Generator().manual_seed(seed))
        return torch.cat([p.detach().flatten() for p in model.parameters()])
    assert torch.equal(run(5), run(5))
    assert not torch.equal(run(5), run(6))


def test_rejects_anything_but_one_capped_class():
    model, chunks = fixture()
    with pytest.raises(ValueError):
        targeted_step(model, chunks, [None, 10, None, 20, None])
