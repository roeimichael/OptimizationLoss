import copy
import random

import numpy as np
import pytest
import torch

from tralo.knee_end_to_end import infer
from tralo.knee_snapshot_local import (ARMS, RECIPE, SEEDS, average_snapshots,
                                      ensemble_window, snapshot, validate)


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model_and_pool(active=True):
    torch.manual_seed(27)
    model = torch.nn.Linear(2, 5)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
        model.bias[3 if active else 0] = .01
    pool = [torch.ones(50, 2), torch.zeros(50, 2)]
    return model, pool, ["H0"] * 50 + ["H1"] * 50, {"global_cap": 76, "local_caps": {"H0": 48, "H1": 47}}


def test_approved_recipe_and_four_seeds_cannot_silently_change():
    for seed in SEEDS:
        validate(dict(RECIPE, seed=seed))
    for key, value in (("seed", 4700), ("weight_decay", 1e-4), ("cap", 75),
                       ("max_epochs", 7), ("lr", 2e-4), ("batch_size", True)):
        with pytest.raises(ValueError):
            validate(dict(RECIPE, seed=7001, **{key:value}) if key != "seed" else dict(RECIPE, seed=value))
    with pytest.raises(ValueError):
        validate(dict(RECIPE, seed=7001, metadata_input=True))
    assert ensemble_window(8, 13) == list(range(6, 14))
    assert ensemble_window(1, 4) == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        ensemble_window(5, 4)


@pytest.mark.parametrize("active", [True, False])
def test_six_arms_preserve_original_gradients_rng_and_dose(tmp_path, active):
    model, pool, groups, quota = model_and_pool(active)
    probabilities = infer(model, pool)
    model.train()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    state = copy.deepcopy(model.state_dict())
    gradients = [p.grad.clone() for p in model.parameters()]
    rng = torch.get_rng_state().clone()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    events = []
    record = snapshot(model, pool, groups, quota, 7001, 1, tmp_path / "epoch01", probabilities, events.append)
    assert set(record["arms"]) == set(ARMS)
    assert all(torch.equal(model.state_dict()[k], v) for k, v in state.items())
    assert all(torch.equal(p.grad, g) for p, g in zip(model.parameters(), gradients))
    assert model.training and torch.equal(torch.get_rng_state(), rng)
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert record["arms"]["joint_local"]["applied"] == active
    assert record["arms"]["global_native"]["applied"] == active
    for arm in ARMS:
        values = torch.load(tmp_path / "epoch01" / (arm + ".pt"), weights_only=True)
        if not active:
            assert torch.equal(values, probabilities)
        assert record["arms"][arm]["after_counts"]["hard_global"] >= 0
    if active:
        joint, sham = record["arms"]["joint_local"], record["arms"]["joint_sham"]
        assert joint["radius"] == sham["radius"]
        assert all(abs(x-y) < 1e-5 for x, y in zip(joint["tensor_displacement_norms"], sham["tensor_displacement_norms"]))
        assert joint["after_counts"]["hard_global"] <= 76
        assert any(event["event"] == "snapshot_arm_started" for event in events)
    window, averages = average_snapshots(tmp_path, {"1":record}, 1, 1)
    assert window == [1] and torch.equal(averages["pto"], probabilities)


def test_failed_side_step_logs_preserves_artifacts_and_restores_rng(tmp_path, monkeypatch):
    model, pool, groups, quota = model_and_pool()
    probabilities, rng = infer(model, pool), torch.get_rng_state().clone()
    state = copy.deepcopy(model.state_dict())
    events = []

    def failed(*args, **kwargs):
        torch.rand(5)
        random.random()
        np.random.random()
        raise RuntimeError("fictitious finite-search failure")

    monkeypatch.setattr("tralo.knee_snapshot_local.local_targeted_step", failed)
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    with pytest.raises(RuntimeError, match="finite-search failure"):
        snapshot(model, pool, groups, quota, 7001, 1, tmp_path / "epoch01", probabilities, events.append)
    assert torch.equal(torch.get_rng_state(), rng) and random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert all(torch.equal(model.state_dict()[k], v) for k, v in state.items())
    assert (tmp_path / "epoch01" / "global_native.pt").exists()
    assert events[-1]["event"] == "snapshot_arm_failed" and events[-1]["arm"] == "joint_local"


def test_false_pto_reference_repeat_output_and_tampered_snapshots_refused(tmp_path):
    model, pool, groups, quota = model_and_pool()
    probabilities = infer(model, pool)
    wrong = probabilities.roll(1, 1)
    with pytest.raises(RuntimeError, match="PTO reference"):
        snapshot(model, pool, groups, quota, 7001, 1, tmp_path / "wrong", wrong, lambda _:None)
    record = snapshot(model, pool, groups, quota, 7001, 1, tmp_path / "epoch01", probabilities, lambda _:None)
    with pytest.raises(FileExistsError):
        snapshot(model, pool, groups, quota, 7001, 1, tmp_path / "epoch01", probabilities, lambda _:None)
    (tmp_path / "epoch01" / "joint_local.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="snapshot bytes"):
        average_snapshots(tmp_path, {"1":record}, 1, 1)


@pytest.mark.parametrize("active", [False, True])
def test_snapshot_corrections_leave_full_supervised_trajectory_identical(tmp_path, active):
    """Actual Adam state/gradient/augmentation RNG parity across two CPU fits."""
    from tralo.knee_yuval import train_run

    class FictitiousImages:
        labels = [0, 1, 2, 3, 4] * 4
        def weights(self):
            return torch.ones(20, dtype=torch.double)
        def batch(self, indices, transform):
            inputs = torch.tensor([[i / 20., (i % 3) / 3.] for i in indices])
            # Training-only random augmentation analogue, shared by both fits.
            return transform(inputs), torch.tensor([self.labels[i] for i in indices])

    initial, pool, groups, quota = model_and_pool(active)
    stop = [(torch.zeros(5, 2), torch.arange(5))]
    states, records, finals = [], [], []
    config = dict(RECIPE, seed=7001, max_epochs=3, patience=3, batch_size=5)
    for corrected in (False, True):
        model = copy.deepcopy(initial)
        events = []
        def save_epoch(epoch, probabilities):
            if corrected:
                snapshot(model, pool, groups, quota, 7001, epoch,
                         tmp_path / f"epoch{epoch:02d}", probabilities, lambda _:None)
            states.append((corrected, epoch, copy.deepcopy(model.state_dict()),
                           [p.grad.clone() for p in model.parameters()]))
        result = train_run(model, FictitiousImages(), stop, pool, config, torch.ones(5),
                           events.append, save_epoch,
                           transforms=(lambda x:x + torch.rand_like(x) * .001, lambda x:x))
        records.append((result, events))
        finals.append(copy.deepcopy(model.state_dict()))
    assert records[0] == records[1]
    assert all(torch.equal(finals[0][k], finals[1][k]) for k in finals[0])
    reference = {epoch:(state,grads) for corrected,epoch,state,grads in states if not corrected}
    for corrected,epoch,state,grads in states:
        if corrected:
            assert all(torch.equal(v, reference[epoch][0][k]) for k,v in state.items())
            assert all(torch.equal(a,b) for a,b in zip(grads, reference[epoch][1]))


def test_joint_parameter_direction_matches_full_objective_and_finite_differences():
    from tralo.local_targeted_step import local_targeted_step
    model, _, _, _ = model_and_pool()
    model = model.double()
    images = torch.tensor([[1., .2], [2., .1], [.1, 1.], [.3, 2.]], dtype=torch.double)
    groups, caps = ["H0", "H0", "H1", "H1"], {"H0": 1, "H1": 1}
    # All four hard calls are grade 3, so global and both local scopes are active.
    def objective():
        q = model(images).softmax(1)[:, 3]
        return q.sum() / 2 + q[:2].sum() + q[2:].sum()
    objective().backward()
    gradients = [p.grad.detach().clone() for p in model.parameters()]
    norm = sum(float(g.square().sum()) for g in gradients) ** .5
    epsilon = 1e-5
    gap = 0.
    with torch.no_grad():
        for param, gradient in zip(model.parameters(), gradients):
            for index in range(param.numel()):
                value = float(param.flatten()[index])
                param.flatten()[index] = value + epsilon
                plus = float(objective())
                param.flatten()[index] = value - epsilon
                minus = float(objective())
                param.flatten()[index] = value
                gap = max(gap, abs((plus-minus)/(2*epsilon) - float(gradient.flatten()[index])))
    assert gap < 1e-8
    origin = [p.detach().clone() for p in model.parameters()]
    record = local_targeted_step(model, [images[:1], images[1:]], groups, 3, 2, caps, fixed_radius=.001)
    assert abs(record["gradient_norm"] - norm) < 1e-12
    for p, base, gradient in zip(model.parameters(), origin, gradients):
        assert torch.allclose((p.detach()-base)/.001, -gradient/norm, atol=1e-12, rtol=1e-10)
