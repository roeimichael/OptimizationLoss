"""Catch altered sham placement/RNG or reuse of an unrelated native search."""
import copy
from dataclasses import replace

import pytest
import torch

from tralo.targeted_step import targeted_step
from tralo.knee_end_to_end import infer


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class CountedModel(torch.nn.Module):
    def __init__(self, active, dtype):
        super().__init__()
        self.linear = torch.nn.Linear(2, 5, dtype=dtype)
        self.dropout = torch.nn.Dropout(.4)
        self.register_buffer("offset", torch.zeros(5, dtype=dtype))
        self.register_buffer("temporary_offset", torch.zeros(5, dtype=dtype), persistent=False)
        self.gain = 1.
        self.unused = torch.nn.Parameter(torch.ones(2, dtype=dtype))
        self.frozen = torch.nn.Parameter(torch.ones(1, dtype=dtype), requires_grad=False)
        self.calls = 0
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()
            self.linear.bias[3 if active else 0] = .03

    def forward(self, values):
        self.calls += 1
        return self.linear(self.dropout(values) * self.gain) + self.offset + self.temporary_offset + 0 * self.unused.sum()


def inputs(active=True, dtype=torch.float32):
    torch.manual_seed(318)
    model = CountedModel(active, dtype)
    for p in model.parameters():
        p.grad = torch.full_like(p, .2)
    return model, [torch.ones(9, 2, dtype=dtype), torch.zeros(7, 2, dtype=dtype)]


@pytest.mark.parametrize("active,dtype", [(True, torch.float32), (True, torch.float64),
                                         (False, torch.float32), (False, torch.float64)])
def test_same_native_radius_reuse_preserves_exact_sham_parameters_gradients_and_rng(active, dtype):
    origin, chunks = inputs(active, dtype)
    caps = [None, None, None, 6, None]
    native, legacy, reused = [copy.deepcopy(origin) for _ in range(3)]
    token = []
    real = targeted_step(native, chunks, caps, calibration_out=token)
    generator1 = torch.Generator().manual_seed(901)
    generator2 = torch.Generator().manual_seed(901)
    ordinary_rng = torch.get_rng_state().clone()
    before = copy.deepcopy(origin.state_dict())
    old = targeted_step(legacy, chunks, caps, sham_generator=generator1)
    new = targeted_step(reused, chunks, caps, sham_generator=generator2, calibration=token[0])
    assert len(token) == 1 and new["applied"] == old["applied"] == active
    assert torch.equal(generator1.get_state(), generator2.get_state())
    assert torch.equal(torch.get_rng_state(), ordinary_rng)
    assert all(torch.equal(v, reused.state_dict()[k]) for k, v in legacy.state_dict().items())
    for a, b in zip(legacy.parameters(), reused.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            assert torch.equal(a.grad, b.grad)
    assert legacy.training and reused.training
    assert torch.equal(infer(legacy, chunks), infer(reused, chunks))
    for key in ["hard_before", "hard_after", "soft_before", "soft_after", "gradient_norm", "applied", "displacement"]:
        assert new[key] == old[key]
    if active:
        assert new["radius"] == old["radius"] == real["radius"]
        assert new["radius_violating"] == old["radius_violating"]
        assert reused.calls < legacy.calls
        assert new["evaluations"] < old["evaluations"]
    else:
        assert generator2.get_state().equal(torch.Generator().manual_seed(901).get_state())
    assert all(torch.equal(v, origin.state_dict()[k]) for k, v in before.items())


@pytest.mark.parametrize("change", ["model", "buffer", "pool", "caps", "settings", "parameter_mask"])
def test_calibration_from_a_different_copy_input_or_search_is_refused(change):
    origin, chunks = inputs()
    native, sham = copy.deepcopy(origin), copy.deepcopy(origin)
    caps = [None, None, None, 6, None]
    token = []
    targeted_step(native, chunks, caps, calibration_out=token)
    kwargs = {}
    if change == "model":
        with torch.no_grad(): sham.linear.bias[0].add_(.001)
    elif change == "buffer":
        sham.offset[0] += .001
    elif change == "pool":
        chunks = [x.clone() for x in chunks]
        chunks[0][0, 0] += .001
    elif change == "caps": caps[3] = 7
    elif change == "settings": kwargs["iterations"] = 19
    else: sham.linear.weight.requires_grad_(False)
    before = copy.deepcopy(sham.state_dict())
    generator = torch.Generator().manual_seed(901)
    rng = generator.get_state().clone()
    with pytest.raises(ValueError, match="calibration"):
        targeted_step(sham, chunks, caps, sham_generator=generator, calibration=token[0], **kwargs)
    assert all(torch.equal(v, sham.state_dict()[k]) for k, v in before.items())
    assert torch.equal(generator.get_state(), rng)


def test_calibration_is_not_a_json_radius_override_or_a_native_direction_shortcut():
    origin, chunks = inputs()
    caps = [None, None, None, 6, None]
    token = []
    targeted_step(copy.deepcopy(origin), chunks, caps, calibration_out=token)
    for kwargs in [dict(calibration={"radius":.1}, sham_generator=torch.Generator()),
                   dict(calibration=token[0]), dict(calibration_out=[token[0]]),
                   dict(calibration_out=[], sham_generator=torch.Generator())]:
        with pytest.raises(ValueError, match="calibration"):
            targeted_step(copy.deepcopy(origin), chunks, caps, **kwargs)


def test_native_export_keeps_original_native_result_and_probabilities_exact():
    origin, chunks = inputs()
    plain, exporting = copy.deepcopy(origin), copy.deepcopy(origin)
    caps = [None, None, None, 6, None]
    ordinary = targeted_step(plain, chunks, caps)
    token = []
    exported = targeted_step(exporting, chunks, caps, calibration_out=token)
    assert ordinary == exported
    assert all(torch.equal(v, exporting.state_dict()[k]) for k, v in plain.state_dict().items())


@pytest.mark.parametrize("alter", [False, True])
def test_only_original_in_memory_native_token_can_authorize_reuse(alter):
    origin, chunks = inputs()
    caps = [None, None, None, 6, None]
    token = []
    targeted_step(copy.deepcopy(origin), chunks, caps, calibration_out=token)
    forged = replace(token[0], radius=token[0].radius * 2) if alter else copy.copy(token[0])
    with pytest.raises(ValueError, match="calibration"):
        targeted_step(copy.deepcopy(origin), chunks, caps,
                      sham_generator=torch.Generator().manual_seed(901), calibration=forged)


def test_mutated_original_token_cannot_authorize_a_different_radius():
    origin, chunks = inputs()
    caps = [None, None, None, 6, None]
    token = []
    targeted_step(copy.deepcopy(origin), chunks, caps, calibration_out=token)
    object.__setattr__(token[0], 'radius', token[0].radius * 2)
    with pytest.raises(ValueError, match='calibration'):
        targeted_step(copy.deepcopy(origin), chunks, caps,
                      sham_generator=torch.Generator().manual_seed(901), calibration=token[0])


@pytest.mark.parametrize('change', ['nonpersistent_buffer', 'real_gradient'])
def test_extra_model_state_or_recomputed_gradient_mismatch_is_refused(change):
    origin, chunks = inputs()
    caps = [None, None, None, 6, None]
    token = []
    targeted_step(copy.deepcopy(origin), chunks, caps, calibration_out=token)
    sham = copy.deepcopy(origin)
    if change == 'nonpersistent_buffer': sham.temporary_offset[0] += .001
    else: sham.gain = 2.
    generator = torch.Generator().manual_seed(901)
    state = generator.get_state().clone()
    with pytest.raises(ValueError, match='calibration'):
        targeted_step(sham, chunks, caps, sham_generator=generator, calibration=token[0])
    assert torch.equal(generator.get_state(), state)


def test_inactive_calibration_still_authenticates_nonpersistent_buffers():
    origin, chunks = inputs(active=False)
    caps = [None, None, None, 6, None]
    token = []
    targeted_step(copy.deepcopy(origin), chunks, caps, calibration_out=token)
    sham = copy.deepcopy(origin)
    sham.temporary_offset[0] += .001
    with pytest.raises(ValueError, match='calibration'):
        targeted_step(sham, chunks, caps,
                      sham_generator=torch.Generator().manual_seed(901), calibration=token[0])


@pytest.mark.parametrize("active", [False, True])
def test_snapshot_uses_authenticated_native_search_and_keeps_all_six_predictions_exact(tmp_path, monkeypatch, active):
    from tralo.knee_snapshot_local import snapshot, ARMS
    origin, chunks = inputs(active)
    pool = [torch.ones(9, 2), torch.zeros(7, 2)]
    groups = ["H0"] * 9 + ["H1"] * 7
    quota = {"global_cap":6, "local_caps":{"H0":4, "H1":3}}
    probabilities = infer(origin, pool)
    rng = torch.get_rng_state().clone()
    events = []
    optimized = snapshot(origin, pool, groups, quota, 91, 1, tmp_path/'optimized', probabilities, events.append)
    assert optimized['arms']['global_native_sham'].get('radius_calibration_reused') is True
    assert torch.equal(torch.get_rng_state(), rng)
    def no_reuse(*args, **kwargs):
        kwargs.pop('calibration', None)
        return targeted_step(*args, **kwargs)
    monkeypatch.setattr('tralo.knee_snapshot_local.targeted_step', no_reuse)
    reference = snapshot(origin, pool, groups, quota, 91, 1, tmp_path/'reference', probabilities, lambda _:None)
    for arm in ARMS:
        a = torch.load(tmp_path/'optimized'/(arm+'.pt'), weights_only=True)
        b = torch.load(tmp_path/'reference'/(arm+'.pt'), weights_only=True)
        assert torch.equal(a, b)
        for key in ['radius', 'displacement', 'tensor_displacement_norms', 'before_counts', 'after_counts']:
            assert optimized['arms'][arm][key] == reference['arms'][arm][key]
    if active:
        assert optimized['arms']['global_native_sham']['evaluations'] < reference['arms']['global_native_sham']['evaluations']
