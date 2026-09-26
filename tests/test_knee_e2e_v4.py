import copy

import pytest
import torch

from tralo.band_consistency import band_indices, random_band
from tralo.knee_e2e_v4 import ARMS, BANDCONS, U, half_width, train_one, validate

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
    for bad in (dict(good, seed=1801), dict(good, seed=2025), dict(good, seed=2100),
                dict(good, caps=[None, None, None, 76, None]),                 # cap 76 on a cap-50 seed
                dict(good, seed=2000, caps=[None, None, None, 76, None]),
                dict(good, seed=2101),                                          # cap 50 on a cap-76 seed
                dict(good, extra=1)):
        with pytest.raises(ValueError):
            validate(bad)
