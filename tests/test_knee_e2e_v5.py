import copy
import math

import pytest
import torch

import tralo.knee_e2e_v5 as v5
from tralo.band_consistency import log_odds
from tralo.cutpair_gate import gate_counts
from tralo.global_clipper import allocate
from tralo.knee_e2e_v4 import train_one as train_v4
from tralo.knee_e2e_v5 import (ARMS, CUTPAIR, DOSE_RATIO, active_sets, cut_hinge, cut_logit, dosed_gradient,
                               shift_ranks, train_one, validate)

CAP = 6
CONFIG = dict(seed=2701, epochs=4, warmup_epochs=2, batch_size=8, task_lr=1e-3,
              constraint_lr=3e-5, caps=[None, None, None, CAP, None], lambda_initial=0.01,
              lambda_step=0.05, rho_initial=0.5, rho_target=0.5, development_batch_size=10)


def fixture():
    """Per-image channel offsets and a sharpened head spread the log-odds, so the anchor matters:
    with a near-uniform model every item is active and the hinge gradient is blind to tau."""
    torch.manual_seed(0)
    train = [(torch.randn(3, 32, 32) + 2 * torch.randn(3, 1, 1), int(c)) for c in torch.randint(0, 5, (40,))]
    val = [torch.randn(10, 3, 32, 32) + 2 * torch.randn(10, 3, 1, 1) for _ in range(4)]
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.BatchNorm2d(4), torch.nn.ReLU(),
                                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 5))
    with torch.no_grad():
        model[5].weight.mul_(4.0)
    return model, train, val


def run_arm(arm, module_train=train_one):
    model, train, val = fixture()
    rows, snaps = [], {}
    rng = torch.get_rng_state()
    result = module_train(copy.deepcopy(model), train, val, CONFIG, arm, rows.append,
                          lambda epoch, phase, values: snaps.__setitem__((epoch, phase), values.clone()))
    result['global_rng_untouched'] = torch.equal(rng, torch.get_rng_state())
    return dict(result=result, rows=[r for r in rows if r['event'] == 'epoch'], snaps=snaps)


@pytest.fixture(scope='module')
def results():
    return {arm: run_arm(arm) for arm in ARMS}


def post_warmup(v):
    return v['rows'][CONFIG['warmup_epochs']:]


def test_every_arm_shares_warmup_batch_order_task_dose_and_tta_draws(results):
    r = [v['result'] for v in results.values()]
    assert len({(x['warmup_sha256'], x['batch_sha256'], x['tta_draws_sha256']) for x in r}) == 1
    assert {x['task_updates_applied'] for x in r} == {4 * 5}
    assert all(x['global_rng_untouched'] for x in r)


def test_non_cutpair_arms_are_bit_identical_to_v4(results):
    for arm in ('clipper', 'tralo_null', 'aug_clip'):
        old = run_arm(arm, train_v4)['result']
        new = results[arm]['result']
        assert torch.equal(old['final_probabilities'], new['final_probabilities'])
        assert torch.equal(old['tta_probabilities'], new['tta_probabilities'])
        assert old['batch_sha256'] == new['batch_sha256'] and old['warmup_sha256'] == new['warmup_sha256']


def test_non_cutpair_arms_never_reach_the_dosed_gradient(monkeypatch):
    def refuse(*args):
        raise AssertionError('dose rule reached by a non-cutpair arm')
    monkeypatch.setattr(v5, 'dosed_gradient', refuse)
    for arm in ('clipper', 'tralo_null', 'aug_clip'):
        run_arm(arm)


def test_tau_is_the_capped_first_boundary_logit(results):
    ids = ['id%02d' % i for i in range(40)]
    for r in post_warmup(results['cutpair_aug']):
        probs = results['cutpair_aug']['snaps'][(r['epoch'], 'before_constraint')]
        chosen = [i for i, c in enumerate(allocate(probs.tolist(), CONFIG['caps'], ids, 'capped_first')) if c == 3]
        p = min(float(probs[i, 3]) for i in chosen)
        assert len(chosen) == CAP and r['anchor_rank'] == CAP
        assert abs(r['tau'] - math.log(p / (1 - p))) < 1e-9


def test_shift_arm_draws_its_anchor_from_its_own_generator(results):
    assert shift_ranks(76) == (25, 228) and shift_ranks(CAP) == (2, 18)
    g = torch.Generator().manual_seed(CONFIG['seed'] + 29)
    for r in post_warmup(results['cutpair_aug_shift']):
        expected = shift_ranks(CAP)[int(torch.randint(0, 2, (1,), generator=g))]
        probs = results['cutpair_aug_shift']['snaps'][(r['epoch'], 'before_constraint')]
        assert r['anchor_rank'] == expected != CAP
        assert r['tau'] == cut_logit(probs[:, 3], expected)


def test_active_sets_match_a_hand_computation():
    s = torch.tensor([-5.0, -0.5, 0.5, 2.0, -3.5, -0.9, 0.2, 1.0, -3.0], dtype=torch.float64)
    y = torch.tensor([0, 0, 1, 3, 3, 3, 3, 3, 3])
    neg, pos = active_sets(s, y, 0.0)
    assert neg.tolist() == [False, True, True, False, False, False, False, False, False]
    # positives strictly inside (-3, 1): -0.9 and 0.2; 1.0 and -3.0 sit on the open boundaries
    assert pos.tolist() == [False, False, False, False, False, True, True, False, False]


def test_runner_active_sets_come_from_a_clean_bank_pass_that_touches_no_state(monkeypatch):
    seen = []
    real = v5.training_log_odds

    def spy(model, images):
        buffers = [b.clone() for b in model.buffers()]
        rng, mode = torch.get_rng_state(), model.training
        s, y = real(model, images)
        assert all(torch.equal(a, b) for a, b in zip(buffers, model.buffers()))
        assert torch.equal(rng, torch.get_rng_state()) and model.training == mode
        # an independent eval-mode reference for the scores
        ref = copy.deepcopy(model).eval()
        with torch.no_grad():
            expected = log_odds(ref(torch.stack([x for x, _ in images]))).double()
        assert torch.allclose(s, expected, atol=1e-5)
        seen.append((s, y))
        return s, y
    monkeypatch.setattr(v5, 'training_log_odds', spy)
    v = run_arm('cutpair_aug')
    rows = post_warmup(v)
    assert len(seen) == len(rows) == 2
    for (s, y), r in zip(seen, rows):
        probs = v['snaps'][(r['epoch'], 'before_constraint')]
        hand_neg = sum(1 for a, b in zip(s.tolist(), y.tolist()) if b != 3 and a > r['tau'] - 1)
        hand_pos = sum(1 for a, b in zip(s.tolist(), y.tolist()) if b == 3 and r['tau'] - 3 < a < r['tau'] + 1)
        gate = gate_counts(probs[:, 3], s, y, CAP)
        assert r['n_act'] == hand_neg == gate['n_act'] and r['p_act'] == hand_pos == gate['p_act']
        assert abs(gate['tau'] - r['tau']) < 1e-12


def test_hinge_value_and_gradient_match_an_independent_reference():
    s = torch.tensor([0.3, -1.5, 2.0, -0.2, 0.9, -4.0], requires_grad=True)
    neg = torch.tensor([True, True, False, False, False, False])
    pos = torch.tensor([False, False, False, True, True, False])
    tau = 0.5
    loss = cut_hinge(s, neg, pos, tau)
    loss.backward()
    # negatives: relu(s - tau + 1) = 0.8, relu(-1.0) = 0; positives: relu(tau + 1 - s) = 1.7, 0.6
    assert abs(float(loss) - (0.8 + 0.0 + 1.7 + 0.6) / 4) < 1e-6
    assert torch.allclose(s.grad, torch.tensor([0.25, 0.0, 0.0, -0.25, -0.25, 0.0]))
    assert cut_hinge(s, torch.zeros(6, dtype=torch.bool), torch.zeros(6, dtype=torch.bool), tau) is None


def _batch():
    model, train, _ = fixture()
    images = torch.stack([x for x, _ in train[:8]])
    labels = torch.tensor([y for _, y in train[:8]])
    return model.train(), images, labels


def _flat_grads(model):
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def test_dosed_gradient_matches_an_independent_autograd_reference_and_moves_batchnorm_once():
    model, images, labels = _batch()
    ce_only = copy.deepcopy(model)
    ref_ce, ref_cut = copy.deepcopy(model), copy.deepcopy(model)
    neg = torch.tensor([True, False, True, False, False, True, False, False])
    pos = torch.tensor([False, True, False, False, True, False, False, False])
    tau = 0.0
    logits = model(images)
    dose = dosed_gradient(model, torch.nn.functional.cross_entropy(logits, labels),
                          cut_hinge(log_odds(logits), neg, pos, tau), DOSE_RATIO)
    ce_only(images)
    assert all(torch.equal(a, b) for a, b in zip(model.buffers(), ce_only.buffers()))
    torch.nn.functional.cross_entropy(ref_ce(images), labels).backward()
    z = log_odds(ref_cut(images))
    terms = [torch.relu(z[i] - tau + 1) if neg[i] else torch.relu(tau + 1 - z[i]) for i in range(8) if neg[i] or pos[i]]
    (sum(terms) / len(terms)).backward()
    a, b = _flat_grads(ref_ce).double(), _flat_grads(ref_cut).double()
    assert float(b.norm()) > 0 and abs(float(a.norm() / b.norm()) - 1) > 0.05
    expected = a + DOSE_RATIO * (a.norm() / b.norm()) * b
    assert torch.allclose(_flat_grads(model).double(), expected, rtol=1e-5, atol=1e-7)
    assert abs(dose['realised_ratio'] - DOSE_RATIO) < 1e-6


def test_no_active_item_falls_back_to_ce():
    model, images, labels = _batch()
    ref = copy.deepcopy(model)
    dose = dosed_gradient(model, torch.nn.functional.cross_entropy(model(images), labels), None, DOSE_RATIO)
    torch.nn.functional.cross_entropy(ref(images), labels).backward()
    assert dose['cut_grad_norm'] == 0.0 and dose['realised_ratio'] == 0.0
    assert torch.equal(_flat_grads(model), _flat_grads(ref))


def test_cutpair_arms_log_the_dose_and_active_sets_and_others_do_not(results):
    for arm, v in results.items():
        logs = v['result']['cut_logs']
        if arm in CUTPAIR:
            assert v['result']['dose_ratio'] == DOSE_RATIO == 0.1 and len(logs) == 2
            assert sum(log['active_batches'] for log in logs) > 0
            for log in logs:
                assert 0 < log['n_act'] < 31 and 0 < log['p_act'] < 9      # partial: the sets select
                if log['active_batches']:
                    assert abs(log['realised_ratio_mean'] - DOSE_RATIO) < 1e-6
                    assert abs(log['realised_ratio_max'] - DOSE_RATIO) < 1e-6
        else:
            assert v['result']['dose_ratio'] is None and logs == []
            assert not any('tau' in r for r in v['rows'])
        assert [r['natural_count'] for r in post_warmup(v)] == v['result']['natural_counts_start']


def test_arms_differ_where_they_should(results):
    final = {arm: v['result']['final_probabilities'] for arm, v in results.items()}
    assert not torch.allclose(final['cutpair_aug'], final['aug_clip'])
    assert not torch.allclose(final['cutpair_aug'], final['cutpair_aug_shift'])


def test_validate_rejects_seeds_and_caps_outside_the_v5_preregistration():
    good = dict(CONFIG, epochs=10, warmup_epochs=5, development_batch_size=16, caps=[None, None, None, 76, None])
    for seed in (2700, 2701, 2724):
        validate(dict(good, seed=seed))
    for bad in (dict(good, seed=2699), dict(good, seed=2725), dict(good, seed=2401), dict(good, seed=2000),
                dict(good, caps=[None, None, None, 50, None]), dict(good, seed=2700, caps=[None, None, None, 50, None]),
                dict(good, extra=1)):
        with pytest.raises(ValueError):
            validate(bad)


def test_batchnorm_sees_exactly_one_train_mode_forward_per_batch_in_every_augmented_arm():
    for arm in ('aug_clip',) + CUTPAIR:
        model, train, val = fixture()
        calls = []
        model[1].register_forward_pre_hook(lambda module, args: calls.append(module.training))
        out = train_one(model, train, val, CONFIG, arm, lambda row: None, lambda *a: None)
        assert sum(calls) == out['task_updates_applied'] == 4 * 5


def test_cutpair_without_a_hinge_is_bit_identical_to_aug_clip(results, monkeypatch):
    """Pins the shared augmentation schedule, generator (seed+11) and the CE fallback of the dose rule."""
    monkeypatch.setattr(v5, 'cut_hinge', lambda *args: None)
    base = results['aug_clip']['result']
    for arm in CUTPAIR:
        out = run_arm(arm)['result']
        assert torch.equal(out['final_probabilities'], base['final_probabilities'])
        assert torch.equal(out['tta_probabilities'], base['tta_probabilities'])
        assert (out['warmup_sha256'], out['batch_sha256']) == (base['warmup_sha256'], base['batch_sha256'])
        assert all(log['active_batches'] == log['dosed_batches'] == 0 for log in out['cut_logs'])


def test_batch_masks_are_the_bank_masks_at_the_shuffled_training_indices(monkeypatch):
    banks, masks = [], []
    real_sets, real_hinge = v5.active_sets, v5.cut_hinge

    def sets_spy(s, y, tau):
        out = real_sets(s, y, tau)
        banks.append(out)
        return out

    def hinge_spy(s, neg, pos, tau):
        masks.append((neg.clone(), pos.clone()))
        return real_hinge(s, neg, pos, tau)
    monkeypatch.setattr(v5, 'active_sets', sets_spy)
    monkeypatch.setattr(v5, 'cut_hinge', hinge_spy)
    run_arm('cutpair_aug')
    g = torch.Generator().manual_seed(CONFIG['seed'] + 1)
    orders = [torch.randperm(40, generator=g) for _ in range(CONFIG['epochs'])][CONFIG['warmup_epochs']:]
    assert len(banks) == len(orders) == 2 and len(masks) == 2 * 5
    for e, (order, (neg, pos)) in enumerate(zip(orders, banks)):
        assert not torch.equal(order, torch.arange(40))
        for b, start in enumerate(range(0, 40, CONFIG['batch_size'])):
            index = order[start:start + CONFIG['batch_size']]
            got_neg, got_pos = masks[e * 5 + b]
            assert torch.equal(got_neg, neg[index]) and torch.equal(got_pos, pos[index])
        # the positional slice would be a different selection in this fixture
        assert any(not torch.equal(neg[order[s:s + 8]], neg[s:s + 8]) for s in range(0, 40, 8))


def test_logs_separate_active_from_dosed_batches_and_record_the_anchor_p3(results):
    for arm in CUTPAIR:
        v = results[arm]
        for log in v['result']['cut_logs']:
            probs = v['snaps'][(log['epoch'], 'before_constraint')]
            assert log['anchor_p3'] == float(torch.sort(probs[:, 3].double(), descending=True).values[log['anchor_rank'] - 1])
            assert abs(log['tau'] - math.log(log['anchor_p3'] / (1 - log['anchor_p3']))) < 1e-9
            assert 0 < log['dosed_batches'] <= log['active_batches'] <= log['batches'] == 5


def test_an_active_hinge_with_zero_gradient_is_counted_active_but_not_dosed(results, monkeypatch):
    """With m > 0 an active item always has a positive hinge term, so active == dosed in practice;
    the two counts must still be kept apart, and a zero-gradient hinge must fall back to CE exactly."""
    real = v5.cut_hinge

    def flat(s, neg, pos, tau):
        loss = real(s, neg, pos, tau)
        return None if loss is None else loss * 0.0
    monkeypatch.setattr(v5, 'cut_hinge', flat)
    out = run_arm('cutpair_aug')['result']
    assert all(log['active_batches'] > 0 and log['dosed_batches'] == 0 and log['realised_ratio_mean'] is None
               for log in out['cut_logs'])
    assert torch.equal(out['final_probabilities'], results['aug_clip']['result']['final_probabilities'])
