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


FACTORIAL = dict(CONFIG, seed=4299, max_retrains=1, augment=True, balanced=True, early_stop=True)


def test_validate_admits_the_recipe_factorial_only_on_its_seeds():
    validate(FACTORIAL)
    validate(dict(FACTORIAL, seed=4200, augment=False, balanced=False, early_stop=False))
    for bad in (dict(FACTORIAL, max_retrains=2), dict(FACTORIAL, augment=1), dict(FACTORIAL, seed=4224),
                {k: v for k, v in FACTORIAL.items() if k != 'early_stop'}, dict(FACTORIAL, seed=4000),
                dict(CONFIG, augment=False), dict(FACTORIAL, backbone='efficientnet_b5')):
        with pytest.raises(ValueError):
            validate(bad)


SMALL = dict(CONFIG, seed=4399, backbone='mobilenet_v3_large', max_retrains=1)


def test_validate_admits_the_small_backbone_blocks_only_with_their_backbone_and_one_retrain():
    from tralo.knee_yuval import backbone_for, make_backbone
    for seed, backbone, cls in ((4300, 'mobilenet_v3_large', 'MobileNetV3'), (4323, 'mobilenet_v3_large', 'MobileNetV3'),
                                (4399, 'mobilenet_v3_large', 'MobileNetV3'), (4400, 'regnet_y_400mf', 'RegNet'),
                                (4423, 'regnet_y_400mf', 'RegNet'), (4499, 'regnet_y_400mf', 'RegNet'),
                                (4000, 'resnet18', 'ResNet'), (4200, 'resnet18', 'ResNet')):
        assert backbone_for(seed) == backbone
        model = make_backbone(backbone_for(seed), pretrained=False)
        assert type(model).__name__ == cls and model(torch.zeros(2, 3, 64, 64)).shape == (2, 5)
    for seed, backbone in ((4300, 'mobilenet_v3_large'), (4399, 'mobilenet_v3_large'), (4423, 'regnet_y_400mf'),
                           (4499, 'regnet_y_400mf')):
        validate(dict(SMALL, seed=seed, backbone=backbone))
    assert (backbone_for(4000), backbone_for(4100), backbone_for(4200)) == ('resnet18', 'efficientnet_b5', 'resnet18')
    for bad in (dict(SMALL, max_retrains=2), dict(SMALL, max_retrains=True), dict(SMALL, max_retrains=1.0),
                {k: v for k, v in SMALL.items() if k != 'max_retrains'},
                {k: v for k, v in SMALL.items() if k != 'backbone'}, dict(SMALL, backbone='regnet_y_400mf'),
                dict(SMALL, seed=4324), dict(SMALL, seed=4424, backbone='regnet_y_400mf'),
                dict(SMALL, augment=True), dict(CONFIG, seed=4000, max_retrains=1)):
        with pytest.raises(ValueError):
            validate(bad)


STEPENS = dict(CONFIG, seed=4500, max_retrains=1, snapshot_steps=True)


def test_validate_admits_the_step_ensemble_study_only_on_its_seeds_and_pilot():
    from tralo.knee_yuval import backbone_for
    for seed in (4500, 4571, 4000):                       # 4000: the pilot reruns a stored ResNet18 seed
        validate(dict(STEPENS, seed=seed))
    rgy = dict(STEPENS, seed=4600, backbone='regnet_y_400mf')
    for seed in (4600, 4671, 4400):                       # the RegNetY replication; 4400 reruns a stored RegNetY seed
        validate(dict(rgy, seed=seed))
    validate(dict(SMALL, seed=4400, backbone='regnet_y_400mf'))   # without snapshot_steps it is still the small block
    assert [backbone_for(s) for s in (4499, 4500, 4571, 4599, 4600, 4671, 4699)] == [
        'regnet_y_400mf', 'resnet18', 'resnet18', 'resnet18', 'regnet_y_400mf', 'regnet_y_400mf', 'regnet_y_400mf']
    for bad in (dict(STEPENS, snapshot_steps=False), dict(STEPENS, snapshot_steps=1), dict(STEPENS, max_retrains=2),
                dict(STEPENS, max_retrains=True), dict(STEPENS, seed=4572), dict(STEPENS, seed=4001),
                dict(CONFIG, seed=4500), dict(CONFIG, seed=4500, max_retrains=1),
                {k: v for k, v in STEPENS.items() if k != 'max_retrains'}, dict(STEPENS, backbone='mobilenet_v3_large'),
                dict(STEPENS, augment=True), dict(FACTORIAL, snapshot_steps=True), dict(SMALL, snapshot_steps=True),
                dict(rgy, seed=4672), dict(rgy, seed=4599), dict(rgy, seed=4401), dict(rgy, seed=4499),
                dict(rgy, backbone='resnet18'), {k: v for k, v in rgy.items() if k != 'backbone'},
                dict(rgy, seed=4000), dict(rgy, max_retrains=2), dict(rgy, snapshot_steps=False),
                dict(CONFIG, seed=4600, backbone='regnet_y_400mf'),
                dict(CONFIG, seed=4600, backbone='regnet_y_400mf', max_retrains=1),
                dict(rgy, seed=4400, max_retrains=2), dict(rgy, seed=4400, backbone='resnet18')):
        with pytest.raises(ValueError):
            validate(bad)


def test_validate_admits_the_dsisco02_blocks_and_their_steps_off_references():
    from tralo.knee_yuval import backbone_for
    mn3 = dict(STEPENS, seed=4700, backbone='mobilenet_v3_large')
    b5 = dict(STEPENS, seed=4800, backbone='efficientnet_b5')
    for config in (mn3, dict(mn3, seed=4771), dict(mn3, seed=4300), b5, dict(b5, seed=4847), dict(b5, seed=4100),
                   dict(mn3, seed=4300, snapshot_steps=False), dict(b5, seed=4100, snapshot_steps=False)):
        validate(config)
    assert [backbone_for(s) for s in (4699, 4700, 4771, 4799, 4800, 4847, 4899, 4900)] == [
        'regnet_y_400mf', 'mobilenet_v3_large', 'mobilenet_v3_large', 'mobilenet_v3_large',
        'efficientnet_b5', 'efficientnet_b5', 'efficientnet_b5', 'resnet18']
    for bad in (dict(mn3, seed=4772), dict(b5, seed=4848), dict(mn3, snapshot_steps=False), dict(b5, snapshot_steps=False),
                dict(mn3, seed=4771, snapshot_steps=False), dict(b5, seed=4101), dict(b5, seed=4101, snapshot_steps=False),
                dict(mn3, seed=4301), dict(mn3, seed=4301, snapshot_steps=False), dict(mn3, backbone='resnet18'),
                dict(b5, backbone='mobilenet_v3_large'), dict(b5, seed=4100, snapshot_steps=False, max_retrains=2),
                dict(b5, seed=4100, snapshot_steps=None), dict(b5, seed=4100, snapshot_steps=0),
                {k: v for k, v in b5.items() if k != 'backbone'}, dict(CONFIG, seed=4700, backbone='mobilenet_v3_large'),
                dict(mn3, seed=4300, snapshot_steps=False, max_retrains=1.0), dict(mn3, seed=4300, cap=50)):
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


def test_train_run_uses_the_transforms_and_the_capped_class_it_is_given():
    from tralo.knee_yuval import transforms_for
    _, eval_tf = transforms_for()
    model, data, stop, pool = _setup()
    seen = []

    def train_tf(image):
        seen.append(image.size)
        return eval_tf(image)
    rows, snaps = [], {}
    config = dict(CONFIG, max_epochs=2, patience=5)
    train_run(model, data, stop, pool, config, torch.ones(5), rows.append, lambda e, v: snaps.__setitem__(e, v),
              capped=1, transforms=(train_tf, eval_tf))
    assert len(seen) == 2 * 40                   # every training draw went through the transform passed in
    for row in rows:                              # the logged soft count is the capped class's, here class 1
        assert row['soft_count_capped'] == float(snaps[row['epoch']][:, 1].sum())
        assert len(row['hard_counts']) == 5


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


def test_recipe_switches_remove_augmentation_balancing_and_early_stopping():
    import hashlib
    from tralo.knee_yuval import SAMPLER_OFFSET, transforms_for
    from tralo.knee_end_to_end import infer
    _, eval_tf = transforms_for()
    held = _Fake(12, 2)
    held.labels = [(y + 2) % 5 for y in held.labels]      # early stopping would stop before max_epochs
    stop = [held.batch(list(range(i, min(i + 8, 12))), eval_tf) for i in range(0, 12, 8)]
    config = dict(CONFIG, lr=1e-2, augment=False, balanced=False, early_stop=False)
    model, data, _, pool = _setup()
    snaps = {}
    result = train_run(model, data, stop, pool, config, torch.ones(5), lambda row: None,
                       lambda e, v: snaps.__setitem__(e, v))
    order = torch.randperm(40, generator=torch.Generator().manual_seed(CONFIG['seed'] + SAMPLER_OFFSET))
    assert result['first_order_sha256'] == hashlib.sha256(order.numpy().tobytes()).hexdigest()
    first = data.batch(order[:CONFIG['batch_size']].tolist(), eval_tf)[0]
    assert result['first_batch_sha256'] == hashlib.sha256(first.numpy().tobytes()).hexdigest()
    assert result['epochs_run'] == result['best_epoch'] == CONFIG['max_epochs']
    assert torch.equal(infer(model, pool), snaps[CONFIG['max_epochs']])
    model, data, _, pool = _setup()
    stopped = train_run(model, data, stop, pool, dict(config, early_stop=True), torch.ones(5), lambda row: None,
                        lambda e, v: None)
    assert stopped['best_epoch'] < CONFIG['max_epochs']   # so it is the switch that keeps the last epoch


def test_snapshot_steps_step_side_copies_and_leave_the_trajectory_byte_identical(tmp_path, monkeypatch):
    import tralo.knee_yuval as ky
    from tralo.knee_end_to_end import infer
    caps = [None, None, None, 3, None]                    # the fixture over-calls grade 3 on a 12-item pool
    runs = []
    for steps in (False, True):
        model, data, stop, pool = _setup()
        snaps, records = {}, {}

        def snapshot(e, v):
            snaps[e] = v
            if steps:
                records[e] = ky.snapshot_steps(model, pool, caps, CONFIG['seed'], e, tmp_path)

        result = train_run(model, data, stop, pool, CONFIG, torch.ones(5), lambda row: None, snapshot)
        runs.append((result, snaps, [p.detach().clone() for p in model.parameters()]))
    (a, sa, pa), (b, sb, pb) = runs
    assert a == b and sa.keys() == sb.keys() and all(torch.equal(sa[e], sb[e]) for e in sa)
    assert all(torch.equal(x, y) for x, y in zip(pa, pb))
    applied = 0
    for e, v in sb.items():
        tralo = torch.load(tmp_path / f'epoch{e:02d}_tralo.pt', weights_only=True)
        sham = torch.load(tmp_path / f'epoch{e:02d}_sham.pt', weights_only=True)
        t, s = records[e]['tralo'], records[e]['sham']
        assert t['hard_before'] == s['hard_before'] == int((v.argmax(1) == CAPPED).sum())
        if t['applied']:
            applied += 1
            assert s['applied'] and t['radius'] == s['radius'] and t['hard_after'] <= 3
            assert int((tralo.argmax(1) == CAPPED).sum()) <= 3 and not torch.equal(tralo, v) and not torch.equal(sham, v)
            assert not torch.equal(sham, tralo)           # a random direction, not the targeted one
        else:
            assert torch.equal(tralo, v) and torch.equal(sham, v)
    assert applied > 0
    real = ky.targeted_step                               # a step that draws from the global RNG must not leak it

    def drawing(*args, **kwargs):
        torch.rand(5)
        return real(*args, **kwargs)
    monkeypatch.setattr(ky, 'targeted_step', drawing)
    model, _, _, pool = _setup()
    torch.manual_seed(3)
    ky.snapshot_steps(model, pool, caps, CONFIG['seed'], 1, tmp_path)
    after = torch.rand(4)
    torch.manual_seed(3)
    assert torch.equal(after, torch.rand(4))
    assert torch.equal(infer(model, pool), infer(_setup()[0], pool))   # the stepped model itself is untouched


def test_efficientnet_b5_has_a_fresh_five_way_head():
    pytest.importorskip('timm')
    from tralo.knee_yuval import efficientnet_b5
    model = efficientnet_b5(pretrained=False)
    assert model.classifier.out_features == 5 and model.classifier.in_features == 2048
