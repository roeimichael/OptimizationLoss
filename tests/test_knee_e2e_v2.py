import copy

import pytest
import torch

from tralo.global_comparison import _state_hash
from tralo.knee_e2e_v2 import ARMS, train_one, validate

CONFIG = dict(seed=1701, epochs=4, warmup_epochs=2, batch_size=8, task_lr=1e-3,
              constraint_lr=3e-3, caps=[None, None, None, 5, None], lambda_initial=0.01,
              lambda_step=0.05, rho_initial=0.5, rho_target=0.5, development_batch_size=6)


def fixture():
    torch.manual_seed(0)
    train = [(torch.randn(6), int(c)) for c in torch.randint(0, 5, (40,))]
    # every development item leans to grade 3, so the cap of 5 binds on the hard count
    val = [torch.randn(6, 6) + 0.0 for _ in range(3)]
    model = torch.nn.Sequential(torch.nn.Linear(6, 12), torch.nn.ReLU(), torch.nn.Linear(12, 5))
    with torch.no_grad():
        model[2].bias[3] += 3.0
    return model, train, val


def run_arm(arm):
    model, train, val = fixture()
    events = []
    out = train_one(copy.deepcopy(model), train, val, CONFIG, arm, events.append, lambda *a: None)
    return out, events


@pytest.fixture(scope='module')
def results():
    return {arm: run_arm(arm) for arm in ARMS}


def test_every_arm_shares_warmup_batch_order_and_task_dose(results):
    ids = {(r['warmup_sha256'], r['batch_sha256']) for r, _ in results.values()}
    assert len(ids) == 1
    assert {r['task_updates_applied'] for r, _ in results.values()} == {4 * 5}


def test_only_constrained_arms_take_constraint_steps(results):
    for arm in ('clipper', 'tralo_null'):
        assert results[arm][0]['constraint_updates_applied'] == 0
        assert results[arm][0]['parameter_displacements'] == []
    for arm in ('tralo_adam', 'tralo_sgd', 'sham_sgd'):
        assert results[arm][0]['constraint_updates_applied'] >= 1


def test_the_cap_binds_on_the_hard_count_and_it_is_recorded(results):
    for r, _ in results.values():
        assert r['cap_binds_on_hard_count'] is True
        assert r['first_constraint_hard_counts'][3] > 5


def test_sham_first_step_has_exactly_the_tralo_sgd_dose(results):
    """Same model at the first constraint step, same calibration: same displacement norm."""
    a = results['tralo_sgd'][0]['parameter_displacements'][0]
    b = results['sham_sgd'][0]['parameter_displacements'][0]
    assert abs(a - b) / a < 1e-4


def test_sgd_first_step_matches_the_adam_first_step_dose(results):
    a = results['tralo_adam'][0]['parameter_displacements'][0]
    b = results['tralo_sgd'][0]['parameter_displacements'][0]
    assert abs(a - b) / a < 1e-5


def test_sham_moves_the_model_differently_from_tralo(results):
    a = results['tralo_sgd'][0]['final_probabilities']
    b = results['sham_sgd'][0]['final_probabilities']
    assert not torch.allclose(a, b)


def test_validate_rejects_seeds_and_caps_outside_the_preregistration():
    good = dict(CONFIG, epochs=10, warmup_epochs=5, caps=[None, None, None, 76, None])
    validate(good)
    with pytest.raises(ValueError):
        validate(dict(good, seed=1301))
    with pytest.raises(ValueError):
        validate(dict(good, caps=[None, None, None, 82, 16]))
