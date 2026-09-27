import copy
import math

import pytest
import torch
from PIL import Image

from tralo.knee_yuval import (CAPPED, Images, carve, custom_loss, f_and_derivative, train_run, validate)

CONFIG = dict(seed=4099, cap=76, max_epochs=6, patience=2, batch_size=8, lr=1e-3, weight_decay=1e-4,
              decay_epoch=5, decay_factor=0.8, mu=8 / 600, b=100.0, development_batch_size=6)


def test_f_matches_yuval_and_its_derivative_matches_finite_differences():
    assert f_and_derivative(76, 76, 100.0) == (0.0, 0.0)
    assert f_and_derivative(76, 60, 100.0)[0] < 1e-5          # under the cap: the loop stops
    f, df = f_and_derivative(76, 126, 100.0)
    assert f == pytest.approx(2 * 50 ** 2) and df == pytest.approx(4 * 50)
    for count in (75.9, 76.004, 76.02, 80.0):
        h = 1e-7
        numeric = (f_and_derivative(76, count + h, 100.0)[0] - f_and_derivative(76, count - h, 100.0)[0]) / (2 * h)
        assert f_and_derivative(76, count, 100.0)[1] == pytest.approx(numeric, rel=1e-4, abs=1e-6)


def test_custom_loss_is_cross_entropy_when_c_is_one():
    torch.manual_seed(0)
    logits, labels = torch.randn(64, 5), torch.randint(0, 5, (64,))
    loss, gate = custom_loss(logits, labels, torch.ones(5))
    assert float(loss) == pytest.approx(float(torch.nn.functional.cross_entropy(logits, labels)), abs=1e-5)
    assert gate == pytest.approx(float((logits.argmax(1) != CAPPED).float().mean()))


def test_custom_loss_weights_only_predicted_capped_items_by_their_true_class():
    logits = torch.tensor([[0., 0., 0., 4., 0.],      # predicted 3, truly 2: weight C[2]
                           [0., 0., 0., 4., 0.],      # predicted 3, truly 3: weight C[3] = 1
                           [0., 0., 4., 0., 0.]])     # predicted 2, truly 1: plain CE
    labels = torch.tensor([2, 3, 1])
    C = torch.tensor([5., 5., 5., 1., 5.])
    ce = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    for i, weight in enumerate((5., 1., 1.)):
        loss, _ = custom_loss(logits[i:i + 1], labels[i:i + 1], C)
        assert float(loss) == pytest.approx(weight * float(ce[i]), rel=1e-5)


def test_carve_keeps_subjects_whole_and_holds_out_about_a_tenth():
    rows = [dict(split='train', subject=f'{9000000 + i}', sample_id=f'{9000000 + i}{s}', path='x', label=0, sha256='')
            for i in range(2000) for s in 'LR'] + [dict(split='val', subject='1', sample_id='1L')]
    train, stop = carve(rows)
    assert len(train) + len(stop) == 4000 and 0.07 < len(stop) / 4000 < 0.13
    assert not {r['subject'] for r in train} & {r['subject'] for r in stop}
    assert carve(rows) == (train, stop)


def test_validate_rejects_other_seeds_caps_and_keys():
    validate(CONFIG)
    validate(dict(CONFIG, seed=4000))
    validate(dict(CONFIG, seed=4123, backbone='efficientnet_b5'))
    validate(dict(CONFIG, seed=4199, backbone='efficientnet_b5'))
    for bad in (dict(CONFIG, seed=4024), dict(CONFIG, cap=50), dict(CONFIG, extra=1),
                dict(CONFIG, patience=0), dict(CONFIG, mu=-1.0), dict(CONFIG, seed=4100),
                dict(CONFIG, backbone='efficientnet_b5'), dict(CONFIG, seed=4124, backbone='efficientnet_b5')):
        with pytest.raises(ValueError):
            validate(bad)


class _Fake(Images):
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.labels = [int(c) for c in torch.randint(0, 5, (n,), generator=g)]
        self.images = [Image.fromarray((torch.rand(32, 32, generator=g) * 60 + 40 * y).byte().numpy()).convert('RGB')
                       for y in self.labels]


def _setup():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 5, stride=8), torch.nn.ReLU(),
                                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 5))
    with torch.no_grad():
        model[4].bias[CAPPED] += 2.0      # grade 3 over-called, so C has predicted-3 items to weight
    data, held = _Fake(40, 1), _Fake(12, 2)
    from tralo.knee_yuval import transforms_for
    _, eval_tf = transforms_for()
    stop = [held.batch(list(range(i, min(i + 8, 12))), eval_tf) for i in range(0, 12, 8)]
    pool = [held.batch(list(range(i, i + 6)), eval_tf)[0] for i in (0, 6)]
    return model, data, stop, pool


def test_retrains_share_order_and_augmentation_and_restore_the_best_epoch():
    runs = []
    for draws, C in ((0, torch.ones(5)), (7, torch.tensor([3., 3., 3., 1., 3.]))):
        model, data, stop, pool = _setup()
        torch.rand(draws)                         # retrains start from different global RNG states
        rows, snaps = [], {}
        result = train_run(model, data, stop, pool, CONFIG, C, rows.append, lambda e, v: snaps.__setitem__(e, v))
        from tralo.knee_end_to_end import infer
        assert torch.equal(infer(model, pool), snaps[result['best_epoch']])
        best = min(rows, key=lambda r: r['stop_loss'])
        assert best['epoch'] == result['best_epoch']
        assert result['epochs_run'] == len(rows) <= CONFIG['max_epochs']
        assert result['task_updates'] == result['epochs_run'] * math.ceil(40 / 8)
        mean_c = float(C.mean())
        for row in rows:                          # train.py's per-batch LR: base / mean(C) where the gate is 0
            assert row['base_lr'] / mean_c - 1e-12 <= row['last_lr'] <= row['base_lr'] + 1e-12
        if mean_c > 1:
            assert any(row['last_lr'] < row['base_lr'] for row in rows)
        runs.append((result, snaps))
    (a, sa), (b, sb) = runs
    assert (a['first_order_sha256'], a['first_batch_sha256']) == (b['first_order_sha256'], b['first_batch_sha256'])
    assert not torch.equal(sa[1], sb[1])          # C changes training from the first epoch


def test_balanced_weights_equalise_expected_class_mass():
    data = _Fake(200, 3)
    w = data.weights()
    mass = [float(w[[i for i, y in enumerate(data.labels) if y == c]].sum()) for c in range(5)]
    assert max(mass) == pytest.approx(min(mass))


def test_early_stopping_restores_an_earlier_best_epoch():
    model, data, _, pool = _setup()
    held = _Fake(12, 2)
    held.labels = [(y + 2) % 5 for y in held.labels]      # the train signal hurts this split
    from tralo.knee_yuval import transforms_for
    from tralo.knee_end_to_end import infer
    _, eval_tf = transforms_for()
    stop = [held.batch(list(range(i, min(i + 8, 12))), eval_tf) for i in range(0, 12, 8)]
    snaps = {}
    result = train_run(model, data, stop, pool, dict(CONFIG, lr=1e-2), torch.ones(5), lambda row: None,
                       lambda e, v: snaps.__setitem__(e, v))
    assert result['best_epoch'] < result['epochs_run'] == result['best_epoch'] + CONFIG['patience']
    assert torch.equal(infer(model, pool), snaps[result['best_epoch']])
    assert not torch.equal(snaps[result['best_epoch']], snaps[result['epochs_run']])


def test_efficientnet_b5_has_a_fresh_five_way_head():
    pytest.importorskip('timm')
    from tralo.knee_yuval import efficientnet_b5
    model = efficientnet_b5(pretrained=False)
    assert model.classifier.out_features == 5 and model.classifier.in_features == 2048
