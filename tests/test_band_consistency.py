import copy

import pytest
import torch

from tralo.band_consistency import (apply_strong, band_indices, consistency_loss, gather, log_odds,
                                    natural_center, random_band, strong_parameters, strong_view,
                                    tta_parameters, tta_probabilities, weak_view)


def model_with_batchnorm():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.BatchNorm2d(4), torch.nn.ReLU(),
                               torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 5))


def images(n=6, seed=1):
    return torch.randn(n, 3, 32, 32, generator=torch.Generator().manual_seed(seed))


def test_log_odds_is_class_logit_minus_logsumexp_of_the_rest():
    z = torch.tensor([[1.0, 2.0, 0.5, 3.0, -1.0]])
    expected = 3.0 - torch.log(torch.exp(torch.tensor([1.0, 2.0, 0.5, -1.0])).sum())
    assert torch.allclose(log_odds(z), expected.view(1))
    assert torch.allclose(log_odds(z + 7.0), log_odds(z))


def test_band_indices_take_ranks_around_the_centre_with_stable_ties():
    p3 = torch.tensor([0.1, 0.9, 0.5, 0.5, 0.7, 0.5, 0.2, 0.3])
    # descending, ties by index: 1(.9) 4(.7) 2 3 5 (.5) 7(.3) 6(.2) 0(.1)
    assert band_indices(p3, 3, 1).tolist() == [2, 3]          # ranks 3..4
    assert band_indices(p3, 4, 2).tolist() == [2, 3, 5, 7]    # ranks 3..6
    assert band_indices(p3, 1, 2).tolist() == [1, 4, 2]       # clipped at rank 1
    assert band_indices(p3, 8, 2).tolist() == [6, 0]          # clipped at rank n
    with pytest.raises(ValueError):
        band_indices(p3, 0, 1)


def test_natural_centre_is_the_argmax_count_clipped():
    probs = torch.full((10, 5), 0.1)
    probs[:7, 3] = 0.6
    assert natural_center(probs, 2) == 7
    assert natural_center(probs, 4) == 6        # clipped to n - w
    probs[:, 3] = 0.0
    assert natural_center(probs, 2) == 2        # clipped to w


def test_random_band_excludes_the_cap_band_and_is_deterministic_per_generator():
    p3 = torch.rand(40, generator=torch.Generator().manual_seed(3))
    cap_band = set(band_indices(p3, 12, 3).tolist())
    draws = [random_band(p3, 12, 3, torch.Generator().manual_seed(s)).tolist() for s in (5, 5, 6)]
    assert draws[0] == draws[1] and draws[0] != draws[2]
    for d in draws:
        assert len(d) == 6 and len(set(d)) == 6 and not cap_band & set(d)
    # every non-band item is reachable
    g = torch.Generator().manual_seed(0)
    seen = set().union(*(random_band(p3, 12, 3, g).tolist() for _ in range(200)))
    assert seen == set(range(40)) - cap_band


def test_gather_indexes_the_concatenated_chunks():
    chunks = [torch.arange(4).float().view(4, 1), torch.arange(4, 7).float().view(3, 1)]
    assert gather(chunks, [5, 0, 3, 6]).flatten().tolist() == [5.0, 0.0, 3.0, 6.0]


def test_views_are_deterministic_given_the_generator_shape_preserving_and_finite():
    x = images()
    for view in (weak_view, strong_view):
        a, b = view(x, torch.Generator().manual_seed(4)), view(x, torch.Generator().manual_seed(4))
        c = view(x, torch.Generator().manual_seed(9))
        assert torch.equal(a, b) and not torch.equal(a, c)
        assert a.shape == x.shape and bool(torch.isfinite(a).all())
    assert not torch.allclose(weak_view(x, torch.Generator().manual_seed(4)),
                              strong_view(x, torch.Generator().manual_seed(4)))
    # the weak view is a per-image flip or identity
    w = weak_view(x, torch.Generator().manual_seed(4))
    assert all(torch.equal(a, b) or torch.equal(a, b.flip(-1)) for a, b in zip(w, x))


def test_strong_view_ranges_stay_inside_the_normalised_unit_cube():
    x = images(4)
    s = strong_view(x, torch.Generator().manual_seed(2))
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    pixels = s * std + mean
    assert float(pixels.min()) >= -1e-5 and float(pixels.max()) <= 1 + 1e-5


def test_identity_strong_parameters_return_the_clamped_input():
    x = images(2)
    params = [dict(flip=False, crop=[0, 0, 32, 32], angle=0.0, brightness=1.0, contrast=1.0)] * 2
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    expected = ((x * std + mean).clamp(0, 1) - mean) / std
    assert torch.allclose(apply_strong(x, params), expected, atol=1e-5)


def test_consistency_loss_keeps_batchnorm_stats_restores_mode_and_detaches_the_weak_target():
    model = model_with_batchnorm()
    model.train()
    stats = [b.clone() for b in model.buffers()]
    x = images()
    loss = consistency_loss(model, x, torch.Generator().manual_seed(7))
    assert model.training
    assert all(torch.equal(a, b) for a, b in zip(stats, model.buffers()))
    loss.backward()
    grads = [p.grad.clone() for p in model.parameters()]
    assert any(float(g.abs().sum()) > 0 for g in grads)
    # independent reference: same draws, eval forward, weak log-odds detached
    ref = copy.deepcopy(model)
    ref.zero_grad()
    g = torch.Generator().manual_seed(7)
    weak, strong = weak_view(x, g), strong_view(x, g)
    ref.eval()
    z = log_odds(ref(torch.cat([weak, strong])))
    expected = torch.nn.functional.smooth_l1_loss(z[len(x):], z[:len(x)].detach(), beta=1.0)
    expected.backward()
    assert torch.allclose(loss.detach(), expected.detach(), atol=1e-6)
    for a, b in zip(grads, ref.parameters()):
        assert torch.allclose(a, b.grad, atol=1e-6)


def test_consistency_loss_restores_eval_mode_too():
    model = model_with_batchnorm().eval()
    consistency_loss(model, images(), torch.Generator().manual_seed(7))
    assert not model.training


def test_tta_draws_depend_only_on_the_seed():
    model = model_with_batchnorm()
    chunks = [images(4, 1), images(3, 2)]
    p, sha = tta_probabilities(model, chunks, 123, draws=3)
    other = copy.deepcopy(model)
    with torch.no_grad():
        other[5].bias[3] += 1.0
    q, sha_other = tta_probabilities(other, chunks, 123, draws=3)
    assert sha == sha_other and not torch.allclose(p, q)
    assert tta_probabilities(model, chunks, 124, draws=3)[1] != sha
    assert p.shape == (7, 5) and torch.allclose(p.sum(1), torch.ones(7))
    # the mean of per-draw softmaxes over exactly the declared draws
    params = tta_parameters(7, 32, 32, 123, draws=3)
    x = torch.cat(chunks)
    model.eval()
    with torch.no_grad():
        expected = sum(model(apply_strong(x, d)).softmax(1) for d in params) / 3
    assert torch.allclose(p, expected, atol=1e-6)
    assert strong_parameters(7, 32, 32, torch.Generator().manual_seed(123)) == params[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA device')
def test_cuda_model_gets_the_same_views_loss_and_tta_as_cpu():
    model = model_with_batchnorm()
    gpu = copy.deepcopy(model).cuda()
    x = images()
    a = consistency_loss(model, x, torch.Generator().manual_seed(3))
    b = consistency_loss(gpu, x, torch.Generator().manual_seed(3))
    assert b.device.type == 'cuda' and torch.allclose(a, b.cpu(), atol=1e-5)
    p, sha = tta_probabilities(model, [x], 5, draws=2)
    q, sha_gpu = tta_probabilities(gpu, [x], 5, draws=2)
    assert sha == sha_gpu and torch.allclose(p, q, atol=1e-5)
