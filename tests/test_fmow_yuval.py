import hashlib
import csv

import numpy as np
import pytest
import torch

from tralo.fmow_yuval import (CLASSES, MEAN, ArrayImages, make_model, pool_chunks, roles, transforms_for, validate)
import tralo.fmow_yuval as fmow_yuval

CONFIG = dict(seed=5000, backbone='mobilenet_v3_large', capped_class=1, cap_divisor=10, max_epochs=75, patience=5,
              batch_size=32, lr=1e-4, weight_decay=1e-4, decay_epoch=5, decay_factor=0.8, development_batch_size=16,
              snapshot_steps=True)


def test_unlabeled_load_does_not_materialize_pool_labels(tmp_path, monkeypatch):
    for split in ('train', 'test'):
        np.save(tmp_path / f'{split}_images.npy', np.zeros((1, 224, 224, 3), dtype=np.uint8))
        with (tmp_path / f'{split}_meta.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['location', 'label'])
            writer.writeheader()
            writer.writerow({'location': split, 'label': 0})
    np.save(tmp_path / 'train_labels.npy', np.array([0]))
    np.save(tmp_path / 'test_labels.npy', np.array([7]))
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in tmp_path.iterdir()}
    monkeypatch.setattr(fmow_yuval, 'FILES', files)
    monkeypatch.setattr(fmow_yuval, 'COUNTS', {'train': 1, 'test': 1})
    monkeypatch.setattr(fmow_yuval, 'CLASSES', 1)
    monkeypatch.setattr(fmow_yuval, 'roles', lambda *_: dict(
        train=[0], stop=[], dev=[0], stop_countries=[], dev_countries=['test'],
        reserved_countries=[]))
    real_load = np.load

    def checked_load(path, *args, **kwargs):
        if str(path).endswith('test_labels.npy'):
            raise AssertionError('training path read development labels')
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, 'load', checked_load)
    _, _, rows, _ = fmow_yuval.load(tmp_path, include_pool_labels=False)
    assert rows == [{'split': 'val', 'sample_id': 'test0', 'location': 'test'}]


def test_validate_admits_the_study_seeds_and_the_pilot_reference_only():
    for config in (CONFIG, dict(CONFIG, seed=5047), dict(CONFIG, seed=5099), dict(CONFIG, seed=5099, snapshot_steps=False),
                   dict(CONFIG, cap_divisor=20)):
        validate(config)
    for bad in (dict(CONFIG, seed=5048), dict(CONFIG, seed=4999), dict(CONFIG, seed=5098), dict(CONFIG, snapshot_steps=False),
                dict(CONFIG, snapshot_steps=1), dict(CONFIG, seed=5099, snapshot_steps=None), dict(CONFIG, backbone='resnet18'),
                dict(CONFIG, capped_class=3), dict(CONFIG, cap_divisor=5), dict(CONFIG, cap_divisor=10.0),
                dict(CONFIG, cap_divisor=True), dict(CONFIG, patience=0), dict(CONFIG, lr=-1.0), dict(CONFIG, extra=1),
                {k: v for k, v in CONFIG.items() if k != 'cap_divisor'}, dict(CONFIG, max_epochs=2.0)):
        with pytest.raises(ValueError):
            validate(bad)


def _meta(countries, sizes):
    return [dict(location=c) for c, n in zip(countries, sizes) for _ in range(n)]


def test_roles_carve_countries_by_hash_and_split_the_test_countries_in_half():
    train_countries = [f'T{i:02d}' for i in range(40)]
    train = _meta(train_countries, [3] * 40)
    test = _meta(['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF'], [5, 1, 4, 2, 6, 3])
    r = roles(train, test)
    h = lambda c: int(hashlib.sha256(c.encode()).hexdigest(), 16)
    assert r['stop_countries'] == sorted(c for c in train_countries if h(c) % 10 == 0) and r['stop_countries']
    assert sorted(r['train'] + r['stop']) == list(range(len(train)))
    assert not {train[i]['location'] for i in r['train']} & set(r['stop_countries'])
    order = sorted({'AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF'}, key=lambda c: hashlib.sha256(c.encode()).hexdigest())
    assert r['dev_countries'] == order[:3] and r['reserved_countries'] == order[3:]
    assert r['dev'] == [i for i, row in enumerate(test) if row['location'] in order[:3]]
    with pytest.raises(RuntimeError, match='both train and test'):
        roles(train + _meta(['AAA'], [1]), test)


def test_array_images_decode_rows_in_order_with_their_own_labels():
    g = np.random.default_rng(0)
    array = g.integers(0, 256, size=(10, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(10) % CLASSES
    data = ArrayImages(array, [7, 2, 5], labels)
    _, evaluation = transforms_for()
    images, y = data.batch([2, 0], evaluation)
    assert images.shape == (2, 3, 224, 224) and y.tolist() == [labels[5], labels[7]]
    chunks = pool_chunks(array, [7, 2, 5], evaluation, 2)
    assert [c.shape[0] for c in chunks] == [2, 1]
    assert torch.equal(chunks[0][0], data.batch([0], evaluation)[0][0]) and torch.equal(chunks[1][0], images[0])
    assert torch.allclose(data.weights(), torch.tensor([1.0, 1.0, 1.0], dtype=torch.double))
    assert MEAN == [0.485, 0.456, 0.406]


def test_model_has_a_fresh_eight_way_head():
    model = make_model(pretrained=False)
    assert type(model).__name__ == 'MobileNetV3' and model(torch.zeros(2, 3, 64, 64)).shape == (2, CLASSES)


def test_vit_model_has_a_fresh_eight_way_head_at_fixed_image_size():
    model = make_model(pretrained=False, backbone='vit_b_16')
    assert type(model).__name__ == 'VisionTransformer'
    assert model.heads.head.out_features == CLASSES
    with torch.no_grad():
        assert model.eval()(torch.zeros(1, 3, 224, 224)).shape == (1, CLASSES)
    with pytest.raises(ValueError, match='unsupported fmow2 backbone'):
        make_model(pretrained=False, backbone='unknown')
