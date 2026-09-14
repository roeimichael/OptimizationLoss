"""Cache identity is mandatory; digests describe the payload actually loaded."""
import copy
import hashlib
import os

import torch

from src.training import model_cache as mc


def cache_config():
    return dict(model_name='fixture', hyperparams={'dropout': 0.0},
                cache_identity={'release_id': 'new-release', 'data_id': 'six-file-data'},
                data_fingerprint='labels-groups')


def test_cache_requires_release_data_and_runtime_identity(tmp_path, monkeypatch):
    monkeypatch.setenv('OPTLOSS_MODEL_CACHE', str(tmp_path))
    monkeypatch.setattr(mc, 'get_model', lambda *a, **k: torch.nn.Linear(2, 2))
    cfg = cache_config()
    model = torch.nn.Linear(2, 2)
    mc.save_to_cache(model, 'same', cfg)
    loaded = mc.load_from_cache('same', cfg, 2, torch.device('cpu'))
    assert loaded is not None
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    assert cfg['warmup_checkpoint']['state_sha256'] == mc.state_digest(model.state_dict())
    assert cfg['warmup_checkpoint']['artifact_sha256']
    for changed in ({}, {'release_id': 'old', 'data_id': 'six-file-data'},
                    {'release_id': 'new-release', 'data_id': 'changed'}):
        other = copy.deepcopy(cfg)
        other['cache_identity'] = changed
        assert mc.load_from_cache('same', other, 2, torch.device('cpu')) is None
    other = copy.deepcopy(cfg)
    other.pop('data_fingerprint')
    assert mc.load_from_cache('same', other, 2, torch.device('cpu')) is None


def test_state_digest_includes_names_shapes_dtypes_buffers_and_is_order_independent():
    base = {'weight': torch.tensor([1., 2.]), 'buffer': torch.tensor(3)}
    hashed = mc.state_digest(base)
    assert hashed == mc.state_digest(dict(reversed(list(base.items()))))
    for changed in ({'renamed': base['weight'], 'buffer': base['buffer']},
                    {**base, 'weight': base['weight'].reshape(1, 2)},
                    {**base, 'weight': base['weight'].double()},
                    {**base, 'buffer': torch.tensor(4)}):
        assert mc.state_digest(changed) != hashed


def test_cache_digest_binds_opened_payload_even_when_path_replaced(tmp_path, monkeypatch):
    monkeypatch.setenv('OPTLOSS_MODEL_CACHE', str(tmp_path))
    monkeypatch.setattr(mc, 'get_model', lambda *a, **k: torch.nn.Linear(2, 2))
    cfg = cache_config()
    first = torch.nn.Linear(2, 2)
    mc.save_to_cache(first, 'same', cfg)
    path = mc.get_cache_path('same', cfg)
    first_bytes = path.read_bytes()
    alternate = path.with_suffix('.replacement')
    alternate.write_bytes(b'atomically replaced payload')
    original_load = torch.load

    def replace_then_load(stream, **kwargs):
        os.replace(alternate, path)
        return original_load(stream, **kwargs)

    monkeypatch.setattr(torch, 'load', replace_then_load)
    loaded = mc.load_from_cache('same', cfg, 2, torch.device('cpu'))
    assert cfg['warmup_checkpoint']['artifact_sha256'] == hashlib.sha256(first_bytes).hexdigest()
    assert cfg['warmup_checkpoint']['state_sha256'] == mc.state_digest(first.state_dict())
    torch.testing.assert_close(loaded.weight, first.weight, rtol=0, atol=0)


def test_missing_cache_payload_metadata_refuses_reuse(tmp_path, monkeypatch):
    monkeypatch.setenv('OPTLOSS_MODEL_CACHE', str(tmp_path))
    cfg = cache_config()
    model = torch.nn.Linear(2, 2)
    mc.save_to_cache(model, 'same', cfg)
    path = mc.get_cache_path('same', cfg)
    torch.save({'model_state_dict': model.state_dict()}, path)
    assert mc.load_from_cache('same', cfg, 2, torch.device('cpu')) is None
