import copy

import pytest
import torch

from tralo.knee_e2e_v3 import ARMS, train_one, validate

CONFIG = dict(seed=1801, epochs=4, warmup_epochs=2, batch_size=8, task_lr=1e-3,
              constraint_lr=3e-3, caps=[None, None, None, 5, None], lambda_initial=0.01,
              lambda_step=0.05, rho_initial=0.5, rho_target=0.5, development_batch_size=6)


def fixture():
    torch.manual_seed(0)
    train = [(torch.randn(6), int(c)) for c in torch.randint(0, 5, (40,))]
    val = [torch.randn(6, 6) for _ in range(3)]
    model = torch.nn.Sequential(torch.nn.Linear(6, 12), torch.nn.ReLU(), torch.nn.Linear(12, 5))
    with torch.no_grad():
        model[2].bias[3] += 3.0      # grade 3 over-called: the hard cap of 5 binds
    return model, train, val


@pytest.fixture(scope='module')
def results():
    out = {}
    for arm in ARMS:
        model, train, val = fixture()
        out[arm] = train_one(copy.deepcopy(model), train, val, CONFIG, arm, lambda row: None, lambda *a: None)
    return out


def test_every_arm_shares_warmup_batch_order_and_task_dose(results):
    assert len({(r['warmup_sha256'], r['batch_sha256']) for r in results.values()}) == 1
    assert {r['task_updates_applied'] for r in results.values()} == {4 * 5}


def test_targeted_arms_land_every_applied_step_on_the_hard_cap(results):
    for arm in ('tralo_target', 'sham_target'):
        steps = results[arm]['targeted_steps']
        assert len(steps) == 2
        assert any(s['applied'] for s in steps)
    for s in results['tralo_target']['targeted_steps']:
        if s['applied']:
            assert s['hard_before'] > 5 and s['hard_after'] <= 5


def test_sham_first_step_has_the_target_radius_and_differs_in_effect(results):
    t, s = results['tralo_target']['targeted_steps'][0], results['sham_target']['targeted_steps'][0]
    assert t['applied'] and s['applied'] and t['radius'] == s['radius']
    assert abs(t['displacement'] - s['displacement']) / t['displacement'] < 1e-4
    assert not torch.allclose(results['tralo_target']['final_probabilities'],
                              results['sham_target']['final_probabilities'])


def test_unconstrained_arms_take_no_constraint_step(results):
    for arm in ('clipper', 'tralo_null'):
        assert results[arm]['constraint_updates_applied'] == 0 and results[arm]['targeted_steps'] == []


def test_validate_rejects_seeds_outside_the_v3_preregistration():
    good = dict(CONFIG, epochs=10, warmup_epochs=5, caps=[None, None, None, 76, None])
    validate(good)
    with pytest.raises(ValueError):
        validate(dict(good, seed=1701))
    validate(dict(good, seed=1901, caps=[None, None, None, 50, None]))
    with pytest.raises(ValueError):
        validate(dict(good, seed=1901))           # cap 76 on a cap-50 seed
    with pytest.raises(ValueError):
        validate(dict(good, caps=[None, None, None, 50, None]))   # cap 50 on a cap-76 seed
