import copy

import pytest
import torch

from tralo.knee_e2e_v3 import ARMS, BACKBONES, build_model, train_one, validate

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


def test_validate_accepts_the_backbone_blocks_and_rejects_unknown_or_mismatched_backbones():
    good = dict(CONFIG, epochs=10, warmup_epochs=5, caps=[None, None, None, 76, None])
    validate(dict(good, backbone='resnet18'))
    validate(dict(good, seed=3000, backbone='mobilenet_v3_large'))
    validate(dict(good, seed=3124, backbone='regnet_y_400mf'))
    for bad in (dict(good, backbone='resnet50'), dict(good, backbone='mobilenet_v3_large'),
                dict(good, seed=3000), dict(good, seed=3100, backbone='mobilenet_v3_large'),
                dict(good, seed=3025, backbone='mobilenet_v3_large'),
                dict(good, seed=3000, backbone='mobilenet_v3_large', caps=[None, None, None, 50, None])):
        with pytest.raises(ValueError):
            validate(bad)


def _cached(backbone):
    from pathlib import Path
    from torchvision import models
    url = models.get_weight(BACKBONES[backbone][1]).url
    return (Path(torch.hub.get_dir()) / 'checkpoints' / url.rsplit('/', 1)[-1]).exists()


@pytest.mark.parametrize('pretrained', [False, True])
def test_default_backbone_is_the_original_resnet18_construction_bit_for_bit(pretrained):
    from torchvision import models
    from tralo.global_comparison import _state_hash
    if pretrained and not _cached('resnet18'):
        pytest.skip('resnet18 ImageNet weights not cached')
    torch.manual_seed(1801)       # the pre-backbone v3 construction, verbatim
    base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    base.fc = torch.nn.Linear(base.fc.in_features, 5)
    expected, expected_draw = _state_hash(base), torch.rand(4)
    backbone = dict(CONFIG).get('backbone', 'resnet18')     # a config without the key
    torch.manual_seed(1801)
    model = build_model(backbone, pretrained=pretrained)
    assert _state_hash(model) == expected
    assert torch.equal(torch.rand(4), expected_draw)      # same RNG consumption


@pytest.mark.parametrize('backbone', ['mobilenet_v3_large', 'regnet_y_400mf'])
def test_new_backbones_build_with_a_five_way_head_and_run_forward(backbone):
    options = [False] + ([True] if _cached(backbone) else [])
    for pretrained in options:
        torch.manual_seed(3000)
        model = build_model(backbone, pretrained=pretrained).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 5) and bool(torch.isfinite(out).all())
    with pytest.raises(ValueError):
        build_model('resnet50', pretrained=False)
