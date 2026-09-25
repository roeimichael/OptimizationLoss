import copy
import math

import pytest
import torch

from tralo.constraint_optimizers import CalibratedSGD, ShamOptimizer
from tralo.streamed_constraint import streamed_step

CAPS = [None, None, None, 3, None]


def model_and_chunks(seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.ReLU(), torch.nn.Linear(8, 5))
    with torch.no_grad():
        model[2].bias[3] += 2.0          # every item leans towards the capped class: the cap binds
    chunks = [torch.randn(7, 6), torch.randn(5, 6)]
    return model, chunks


def displacement(before, after):
    return math.sqrt(sum(float((a - b).double().square().sum())
                         for a, b in zip(before.parameters(), after.parameters())))


def step_with(opt_factory, multiplier, seed=0):
    model, chunks = model_and_chunks(seed)
    start = copy.deepcopy(model)
    opt = opt_factory(model)
    out = streamed_step(model, chunks, CAPS, torch.full((5,), float(multiplier)), 0.5, opt)
    assert out['applied']
    return displacement(start, model), opt


def test_audit_d1_adam_ignores_the_multiplier():
    """The defect this module fixes: a fresh Adam's first step is sign(g) * lr."""
    small, _ = step_with(lambda m: torch.optim.Adam(m.parameters(), lr=1e-3), 0.01)
    large, _ = step_with(lambda m: torch.optim.Adam(m.parameters(), lr=1e-3), 10.0)
    assert abs(small - large) / large < 1e-3


def test_calibrated_first_step_matches_the_adam_first_step_dose():
    adam, _ = step_with(lambda m: torch.optim.Adam(m.parameters(), lr=1e-3), 0.01)
    sgd, opt = step_with(lambda m: CalibratedSGD(m.parameters(), adam_lr=1e-3), 0.01)
    assert abs(sgd - adam) / adam < 1e-5
    assert abs(opt.last_displacement - sgd) / sgd < 1e-4  # fp32 parameter storage


def test_calibrated_later_steps_scale_with_the_multiplier():
    """After calibration the lr is fixed, so doubling lambda doubles the next step."""
    model, chunks = model_and_chunks()
    opt = CalibratedSGD(model.parameters(), adam_lr=1e-3)
    streamed_step(model, chunks, CAPS, torch.full((5,), 0.01), 0.5, opt)
    lr = opt.lr
    base = copy.deepcopy(model)
    a = copy.deepcopy(model); oa = CalibratedSGD(a.parameters(), 1e-3); oa.lr = lr
    b = copy.deepcopy(model); ob = CalibratedSGD(b.parameters(), 1e-3); ob.lr = lr
    streamed_step(a, chunks, CAPS, torch.full((5,), 0.01), 0.5, oa)
    streamed_step(b, chunks, CAPS, torch.full((5,), 0.02), 0.5, ob)
    ratio = displacement(base, b) / displacement(base, a)
    assert abs(ratio - 2.0) < 1e-3


def test_sham_matches_the_norm_of_every_tensor_but_not_the_direction():
    model, chunks = model_and_chunks()
    real = copy.deepcopy(model)
    sham_opt = ShamOptimizer(CalibratedSGD(model.parameters(), 1e-3), seed=11)
    real_opt = CalibratedSGD(real.parameters(), 1e-3)
    start = copy.deepcopy(model)
    streamed_step(model, chunks, CAPS, torch.full((5,), 0.01), 0.5, sham_opt)
    streamed_step(real, chunks, CAPS, torch.full((5,), 0.01), 0.5, real_opt)
    assert sham_opt.lr == real_opt.lr
    for s, r, z in zip(model.parameters(), real.parameters(), start.parameters()):
        ds, dr = (s - z).double().flatten(), (r - z).double().flatten()
        if dr.norm() == 0:
            assert ds.norm() == 0
            continue
        assert abs(float(ds.norm() / dr.norm()) - 1.0) < 1e-4  # fp32 parameter storage
        if ds.numel() >= 8:
            assert abs(float(torch.dot(ds, dr) / (ds.norm() * dr.norm()))) < 0.9


def test_sham_is_deterministic_given_its_seed_and_differs_across_seeds():
    def run(seed):
        model, chunks = model_and_chunks()
        opt = ShamOptimizer(CalibratedSGD(model.parameters(), 1e-3), seed=seed)
        streamed_step(model, chunks, CAPS, torch.full((5,), 0.01), 0.5, opt)
        return torch.cat([p.detach().flatten() for p in model.parameters()])
    assert torch.equal(run(3), run(3))
    assert not torch.equal(run(3), run(4))


def test_inactive_constraint_takes_no_step():
    model, chunks = model_and_chunks()
    start = copy.deepcopy(model)
    opt = CalibratedSGD(model.parameters(), 1e-3)
    out = streamed_step(model, chunks, [None, None, None, 10_000, None], torch.full((5,), 0.01), 0.5, opt)
    assert not out['applied'] and opt.lr is None and displacement(start, model) == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU; run natively on the server')
def test_sham_and_sgd_step_a_cuda_model_like_a_cpu_one():
    """The pilot died here: seeded CPU noise met a CUDA norm tensor."""
    def run(device, sham):
        model, chunks = model_and_chunks()
        model = model.to(device)
        inner = CalibratedSGD(model.parameters(), 1e-3)
        opt = ShamOptimizer(inner, seed=3) if sham else inner
        streamed_step(model, chunks, CAPS, torch.full((5,), 0.01, device=device), 0.5, opt)
        return torch.cat([p.detach().cpu().flatten() for p in model.parameters()])
    for sham in (False, True):
        assert torch.allclose(run('cuda', sham), run('cpu', sham), atol=1e-5)
