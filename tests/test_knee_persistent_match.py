import copy
import json
from pathlib import Path

import pytest
import torch

from tralo import knee_persistent_match as match


CONFIG = dict(seed=6700, backbone='mobilenet_v3_large', caps=[76],
              max_epochs=75, patience=5, batch_size=32, lr=1e-4,
              weight_decay=0, decay_epoch=5, decay_factor=0.8,
              mu=8 / 600, b=100.0, development_batch_size=16,
              max_retrains=8)


def test_config_fixes_seed_cap_backbone_and_comparable_recipe():
    match.validate(CONFIG)
    match.validate(dict(CONFIG, seed=6701, caps=[54, 86], backbone='vit_b_16'))
    match.validate(dict(CONFIG, seed=6712, caps=[54, 86], backbone='efficientnet_b5'))
    match.validate(dict(CONFIG, seed=6713, backbone='vit_b_16'))
    for changed in (dict(seed=6713), dict(seed=6701), dict(caps=[54, 86]),
                    dict(backbone='resnet18'), dict(weight_decay=1e-4),
                    dict(max_retrains=9), dict(batch_size=16), dict(extra=1)):
        with pytest.raises(ValueError):
            match.validate(dict(CONFIG, **changed))


def test_all_fixed_pilot_and_full_configs_exist_and_validate():
    root = Path(__file__).resolve().parents[1] / 'experiments' / 'configs'
    files = sorted(root.glob('knee_persistent_*.json'))
    assert len(files) == 40
    observed = set()
    for path in files:
        config = json.loads(path.read_text())
        match.validate(config)
        assert path.name == f"knee_persistent_{config['backbone']}_{config['seed']}.json"
        observed.add((config['backbone'], config['seed']))
    assert observed == ({(backbone, seed) for backbone in match.BACKBONES
                         for seed in (6700,) + match.SEEDS_STUDY}
                        | {('vit_b_16', 6713)})


def test_vit_attention_replay_uses_same_path_with_and_without_grad(monkeypatch):
    if not hasattr(torch, '_native_multi_head_attention'):
        pytest.skip('native MHA unavailable')
    previous = torch.backends.mha.get_fastpath_enabled()
    attention = torch.nn.MultiheadAttention(8, 2, batch_first=True).eval()
    calls = []
    original = torch._native_multi_head_attention

    def counted(*args, **kwargs):
        calls.append('native')
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, '_native_multi_head_attention', counted)
    try:
        torch.backends.mha.set_fastpath_enabled(True)
        assert match.disable_vit_mha_fastpath() is False

        class SmallAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = attention
                self.head = torch.nn.Linear(8, 5)

            def forward(self, values):
                encoded = self.attention(values, values, values,
                                         need_weights=False)[0]
                return self.head(encoded[:, 0])

        result = match.vit_attention_replay(SmallAttention().eval(),
                                            torch.randn(2, 4, 8))
        assert result['passed'] and result['max_tolerance_ratio'] <= 1
        assert calls == []
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def test_vit_attention_replay_rejects_a_grad_only_prediction_change():
    class Shift(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, 5) * 0.01)

        def forward(self, values):
            logits = values @ self.weight
            if torch.is_grad_enabled():
                logits = logits + torch.tensor([0.0, 0.001, 0.0, 0.0, 0.0])
            return logits

    previous = torch.backends.mha.get_fastpath_enabled()
    try:
        match.disable_vit_mha_fastpath()
        result = match.vit_attention_replay(Shift().eval(), torch.ones(2, 4))
        assert not result['passed'] and result['max_tolerance_ratio'] > 1
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def test_target_and_sham_hooks_match_each_tensor_dose_without_sharing_direction(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(),
                                torch.nn.Linear(4, 5))
    initial = copy.deepcopy(model.state_dict())

    def fixed_target(current, pool, caps):
        with torch.no_grad():
            current[0].weight.add_(0.01)
            current[2].bias.add_(0.02)
        return dict(applied=True, hard_after=caps[3], hard_before=caps[3] + 1,
                    radius=0.01, displacement=0.01)

    monkeypatch.setattr(match, 'targeted_step', fixed_target)
    doses = []
    target = match._target_hook([], 76, doses)(1, model)
    assert target['intervention'] == 'tralo' and target['target_step']['hard_after'] == 76
    assert len(doses) == 1 and len(doses[0]) == len(list(model.parameters()))

    sham_model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(),
                                     torch.nn.Linear(4, 5))
    sham_model.load_state_dict(initial)
    sham = match._sham_hook(doses, 6700)(1, sham_model)
    assert sham['intervention'] == 'sham'
    for expected, actual in zip(doses[0], sham['tensor_displacement_norms']):
        assert actual == pytest.approx(expected, abs=2e-6, rel=2e-3)
    assert not torch.equal(sham_model[0].weight, model[0].weight)


def test_sham_rejects_nonmatching_parameter_structure():
    model = torch.nn.Linear(2, 3)
    with pytest.raises(RuntimeError, match='parameter tensors differ'):
        match._sham_hook([[0.1]], 6700)(1, model)


def test_null_hook_measures_development_predictions_without_changing_parameters():
    model = torch.nn.Linear(2, 5)
    before = copy.deepcopy(model.state_dict())
    row = match._null_hook([torch.ones(3, 2)])(1, model)
    assert row['null_hook'] and sum(row['pre_hook_hard_counts']) == 3
    assert 0 <= row['pre_hook_soft_count_capped'] <= 3
    assert all(torch.equal(model.state_dict()[key], value) for key, value in before.items())


def test_preflight_requests_label_free_audit_and_keeps_receipt_exclusive(monkeypatch, tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(CONFIG))
    rows = [dict(split=split, sample_id=str(index), sha256=str(index),
                 pixel_sha256=str(index), subject=str(index), **({'label': 3} if split == 'train' else {}))
            for index, split in enumerate(('train', 'val', 'test'))]
    manifest = dict(rows=rows, counts={'train': 5778, 'val': 826, 'test': 1656},
                    cross_split_subject_overlap=0, cross_split_pixel_overlap=0)

    def label_free_audit(root, include_val_labels=True):
        assert not include_val_labels
        return manifest

    monkeypatch.setattr(match, 'audit', label_free_audit)
    monkeypatch.setattr(match, 'carve', lambda value: (value[:1], value[:1]))
    monkeypatch.setattr(match, '_weights', lambda backbone: ('weights', 'weight-hash'))
    monkeypatch.setattr(match, 'make_model', lambda backbone: torch.nn.Linear(2, 5))
    monkeypatch.setattr(match, 'source', lambda: {'knee_persistent_match.py': 'source-hash'})
    output = tmp_path / 'preflight.json'
    receipt = match.preflight('images', config, output)
    assert receipt['scope'] == 'knee_persistent_match_label_blind_preflight'
    assert receipt['pretrained_weight_sha256'] == 'weight-hash'
    assert json.loads(output.read_text()) == receipt
    with pytest.raises(FileExistsError):
        match.preflight('images', config, output)


def test_vit_and_mobilenet_heads_are_five_class_without_pretrained_download():
    for backbone in ('vit_b_16', 'mobilenet_v3_large'):
        model = match.make_model(backbone, pretrained=False)
        head = model.heads.head if backbone == 'vit_b_16' else model.classifier[-1]
        assert head.out_features == 5
