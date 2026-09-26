import copy

import pytest
import torch

import tralo.knee_e2e_v4 as v4
from tralo.band_consistency import band_indices, consistency_loss, log_odds, random_band
from tralo.knee_e2e_v4 import ARMS, BANDCONS, DOSE_RATIO, U, dual_backward, half_width, train_one, validate

CAP = 12
CONFIG = dict(seed=2001, epochs=4, warmup_epochs=2, batch_size=8, task_lr=1e-3,
              constraint_lr=3e-5, caps=[None, None, None, CAP, None], lambda_initial=0.01,
              lambda_step=0.05, rho_initial=0.5, rho_target=0.5, development_batch_size=10)


def fixture():
    torch.manual_seed(0)
    train = [(torch.randn(3, 32, 32), int(c)) for c in torch.randint(0, 5, (40,))]
    val = [torch.randn(10, 3, 32, 32) for _ in range(4)]
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.BatchNorm2d(4), torch.nn.ReLU(),
                                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 5))
    with torch.no_grad():
        model[5].bias[3] += 1.0      # grade 3 over-called: the natural centre sits away from the cap
    return model, train, val


@pytest.fixture(scope='module')
def results():
    out = {}
    for arm in ARMS:
        model, train, val = fixture()
        rows, snaps = [], {}
        rng = torch.get_rng_state()
        result = train_one(copy.deepcopy(model), train, val, CONFIG, arm, rows.append,
                           lambda epoch, phase, values: snaps.__setitem__((epoch, phase), values.clone()))
        result['global_rng_untouched'] = torch.equal(rng, torch.get_rng_state())
        out[arm] = dict(result=result, rows=[r for r in rows if r['event'] == 'epoch'], snaps=snaps)
    return out


def test_every_arm_shares_warmup_batch_order_task_dose_and_tta_draws(results):
    r = [v['result'] for v in results.values()]
    assert len({(x['warmup_sha256'], x['batch_sha256']) for x in r}) == 1
    assert {x['task_updates_applied'] for x in r} == {4 * 5}
    assert len({x['tta_draws_sha256'] for x in r}) == 1


def test_no_arm_touches_the_global_rng(results):
    assert all(v['result']['global_rng_untouched'] for v in results.values())


def test_only_bandcons_arms_log_consistency(results):
    for arm, v in results.items():
        logged = [r for r in v['rows'] if 'consistency_loss_mean' in r]
        if arm in BANDCONS:
            assert len(logged) == 2 and len(v['result']['band_logs']) == 2
            per_epoch = 5 * 1          # one U-item term per labeled batch
            assert all(r['consistency_terms'] == per_epoch and r['band_size'] == 2 * half_width(CAP)
                       and r['band_disagreement_start'] > 0 for r in logged)
        else:
            assert logged == [] and v['result']['band_logs'] == []
            assert not any('band' in r for r in v['rows'])


def test_bands_come_from_start_of_epoch_probabilities(results):
    w = half_width(CAP)
    starts = {}
    rand_generator = torch.Generator().manual_seed(CONFIG['seed'] + 17)
    for arm in BANDCONS:
        v = results[arm]
        for r in v['rows'][CONFIG['warmup_epochs']:]:
            probs = v['snaps'][(r['epoch'], 'before_constraint')]
            if arm == 'bandcons':
                assert r['band_center'] == CAP
                assert r['band'] == band_indices(probs[:, 3], CAP, w).tolist()
            elif arm == 'bandcons_unc':
                count = int((probs.argmax(1) == 3).sum())
                assert r['band_center'] == min(max(count, w), 40 - w) != CAP
                assert r['band'] == band_indices(probs[:, 3], r['band_center'], w).tolist()
            else:
                assert r['band_center'] is None
                assert r['band'] == random_band(probs[:, 3], CAP, w, rand_generator).tolist()
                assert not set(r['band']) & set(band_indices(probs[:, 3], CAP, w).tolist())
            starts.setdefault(arm, []).append(r['band'])
    # the start of the first consistency epoch is the same state in every arm
    first = {arm: results[arm]['snaps'][(CONFIG['warmup_epochs'] + 1, 'before_constraint')] for arm in ARMS}
    assert all(torch.equal(first['clipper'], p) for p in first.values())
    assert starts['bandcons_rand'][0] != starts['bandcons_rand'][1]


def test_every_post_warmup_epoch_has_both_snapshots_and_end_equals_the_final_probabilities(results):
    for v in results.values():
        assert set(v['snaps']) == {(e, p) for e in (3, 4) for p in ('before_constraint', 'after_constraint')}
        assert torch.equal(v['snaps'][(3, 'after_constraint')], v['snaps'][(4, 'before_constraint')])
        assert torch.equal(v['snaps'][(4, 'after_constraint')], v['result']['final_probabilities'])


def test_arms_differ_where_they_should(results):
    final = {arm: v['result']['final_probabilities'] for arm, v in results.items()}
    assert not torch.allclose(final['aug_clip'], final['tralo_null'])
    assert not torch.allclose(final['bandcons'], final['bandcons_rand'])
    assert not torch.allclose(final['bandcons'], final['tralo_null'])
    assert not torch.allclose(final['clipper'], final['tralo_null'])
    tta = {arm: v['result']['tta_probabilities'] for arm, v in results.items()}
    assert not torch.allclose(tta['bandcons'], final['bandcons'])


def test_band_constants_are_fixed():
    assert U == 8 and half_width(50) == 12 and half_width(76) == 19


def test_validate_rejects_seeds_and_caps_outside_the_v4_preregistration():
    good = dict(CONFIG, epochs=10, warmup_epochs=5, development_batch_size=16, caps=[None, None, None, 50, None])
    validate(good)
    validate(dict(good, seed=2000))
    validate(dict(good, seed=2124, caps=[None, None, None, 76, None]))
    validate(dict(good, seed=2400))
    validate(dict(good, seed=2424))
    validate(dict(good, seed=2501, caps=[None, None, None, 76, None]))
    validate(dict(good, seed=2524, caps=[None, None, None, 76, None]))
    for bad in (dict(good, seed=1801), dict(good, seed=2025), dict(good, seed=2100),
                dict(good, seed=2399), dict(good, seed=2425), dict(good, seed=2500), dict(good, seed=2525),
                dict(good, seed=2400, caps=[None, None, None, 76, None]),
                dict(good, seed=2412, caps=[None, None, None, 76, None]),
                dict(good, seed=2501), dict(good, seed=2524),
                dict(good, caps=[None, None, None, 76, None]),                 # cap 76 on a cap-50 seed
                dict(good, seed=2000, caps=[None, None, None, 76, None]),
                dict(good, seed=2101),                                          # cap 50 on a cap-76 seed
                dict(good, extra=1)):
        with pytest.raises(ValueError):
            validate(bad)


def _batch():
    model, train, val = fixture()
    images = torch.stack([x for x, _ in train[:8]])
    labels = torch.tensor([y for _, y in train[:8]])
    return model, images, labels, val[0][:U]


def _flat_grads(model):
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def test_dual_backward_matches_an_independent_autograd_reference():
    model, images, labels, dev = _batch()
    ref = copy.deepcopy(model)
    model.train()
    ce = torch.nn.functional.cross_entropy(model(images), labels)
    _, dose = dual_backward(model, ce, lambda: consistency_loss(model, dev, torch.Generator().manual_seed(4)),
                            DOSE_RATIO)
    ref.train()
    params = list(ref.parameters())
    g_ce = torch.autograd.grad(torch.nn.functional.cross_entropy(ref(images), labels), params)
    g = torch.Generator().manual_seed(4)
    from tralo.band_consistency import strong_view, weak_view
    weak, strong = weak_view(dev, g), strong_view(dev, g)
    ref.eval()
    z = log_odds(ref(torch.cat([weak, strong])))
    g_cons = torch.autograd.grad(torch.nn.functional.smooth_l1_loss(z[U:], z[:U].detach(), beta=1.0), params)
    a, b = torch.cat([x.flatten() for x in g_ce]).double(), torch.cat([x.flatten() for x in g_cons]).double()
    assert float(b.norm()) > 0 and abs(float(a.norm()) / float(b.norm()) - 1) > 0.05   # the rescale is not a no-op
    expected = a + DOSE_RATIO * (a.norm() / b.norm()) * b
    assert torch.allclose(_flat_grads(model).double(), expected, rtol=1e-5, atol=1e-7)
    assert abs(dose['realised_ratio'] - DOSE_RATIO) < 1e-6
    assert abs(dose['ce_grad_norm'] - float(a.norm())) < 1e-5 * float(a.norm())
    assert abs(dose['consistency_grad_norm'] - float(b.norm())) < 1e-5 * float(b.norm())


def test_dual_backward_moves_batchnorm_statistics_only_through_the_ce_forward():
    model, images, labels, dev = _batch()
    ce_only = copy.deepcopy(model).train()
    ce_only(images)
    model.train()
    ce = torch.nn.functional.cross_entropy(model(images), labels)
    dual_backward(model, ce, lambda: consistency_loss(model, dev, torch.Generator().manual_seed(4)), DOSE_RATIO)
    assert model.training
    assert all(torch.equal(a, b) for a, b in zip(model.buffers(), ce_only.buffers()))


def test_zero_consistency_gradient_falls_back_to_ce():
    model, images, labels, _ = _batch()
    ref = copy.deepcopy(model).train()
    model.train()
    ce = torch.nn.functional.cross_entropy(model(images), labels)
    _, dose = dual_backward(model, ce, lambda: (model(images) * 0.0).sum(), DOSE_RATIO)
    torch.nn.functional.cross_entropy(ref(images), labels).backward()
    assert dose['consistency_grad_norm'] == 0.0 and dose['realised_ratio'] == 0.0
    assert torch.equal(_flat_grads(model), _flat_grads(ref))


def test_the_dose_rule_is_the_same_in_every_bandcons_arm(results):
    for arm in BANDCONS:
        r = results[arm]['result']
        assert r['dose_ratio'] == DOSE_RATIO == 0.1
        for log in r['band_logs']:
            assert log['dose_ratio'] == DOSE_RATIO
            assert abs(log['realised_ratio_mean'] - DOSE_RATIO) < 1e-6
            assert abs(log['realised_ratio_max'] - DOSE_RATIO) < 1e-6
            assert 0 < log['ce_grad_norm_mean'] <= log['ce_grad_norm_max']
            assert 0 < log['consistency_grad_norm_mean'] <= log['consistency_grad_norm_max']
    for arm in ('clipper', 'tralo_null', 'aug_clip'):
        assert results[arm]['result']['dose_ratio'] is None


def _refusing_runs(monkeypatch):
    calls = []

    def refuse(model, ce, consistency, ratio):
        calls.append(1)
        raise AssertionError('dual backward reached by a non-bandcons arm')
    monkeypatch.setattr(v4, 'dual_backward', refuse)
    out = {}
    for arm in ('clipper', 'tralo_null', 'aug_clip'):
        model, train, val = fixture()
        out[arm] = train_one(copy.deepcopy(model), train, val, CONFIG, arm, lambda row: None, lambda *a: None)
    return out, calls


def test_non_bandcons_arms_never_reach_the_dual_backward(monkeypatch):
    assert _refusing_runs(monkeypatch)[1] == []


def test_non_bandcons_arms_are_unchanged_by_the_dose_rule(results, monkeypatch):
    """With the dual backward made unreachable they reproduce the module run bit for bit."""
    out, _ = _refusing_runs(monkeypatch)
    for arm, r in out.items():
        assert torch.equal(r['final_probabilities'], results[arm]['result']['final_probabilities'])
        assert torch.equal(r['tta_probabilities'], results[arm]['result']['tta_probabilities'])


def test_bandcons_takes_one_dual_backward_per_post_warmup_batch(monkeypatch):
    calls = []

    def spy(*args):
        calls.append(1)
        return dual_backward(*args)
    monkeypatch.setattr(v4, 'dual_backward', spy)
    model, train, val = fixture()
    out = train_one(copy.deepcopy(model), train, val, CONFIG, 'bandcons', lambda row: None, lambda *a: None)
    assert len(calls) == 2 * 5 and out['task_updates_applied'] == 4 * 5


def test_every_arm_logs_the_natural_count_at_each_epoch_start(results):
    for v in results.values():
        counts = [r['natural_count'] for r in v['rows'][CONFIG['warmup_epochs']:]]
        expected = [int((v['snaps'][(e, 'before_constraint')].argmax(1) == 3).sum()) for e in (3, 4)]
        assert counts == expected == v['result']['natural_counts_start']
