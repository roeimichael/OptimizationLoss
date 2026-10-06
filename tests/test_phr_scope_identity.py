"""Fixed PHR scope-identity cases, without RNG or scientific payloads."""
import copy
import math

import pytest
import torch

from analysis import score_fmow_boundary as boundary
from analysis import score_fmow_local_alm as fixed
from tralo.knee_end_to_end import infer
from tralo.local_alm import snapshot_phr_step


class OpposedScopes(torch.nn.Module):
    def __init__(self, slope=25.):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0., dtype=torch.double))
        self.slope = slope

    def forward(self, images):
        one = images[:, 0] + self.theta * self.slope * images[:, 1]
        return torch.stack([torch.zeros_like(one), one], dim=1)


def run(group, *, calibrated=False, slope=25.):
    model = OpposedScopes(slope)
    chunks = [torch.tensor([[math.log(4.), 1.], [math.log(1.5), -1.]],
                           dtype=torch.double)]
    groups = [group, 'B']
    quota = {'global_cap': 1, 'local_caps': {group: 0, 'B': 1}}
    before = infer(model, chunks)
    rng = torch.random.get_rng_state().clone()
    record, dual = snapshot_phr_step(model, chunks, groups, 1, 1,
                                    quota['local_caps'], torch.zeros(3, dtype=torch.double),
                                    boundary_calibrated=calibrated)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert model.training
    return model, record, dual, before, infer(model, chunks), groups, quota


@pytest.mark.parametrize('group', ['global', 'pooled', 'local:A', 'A:B', 'A'])
def test_fixed_phr_retains_each_distinct_normalized_scope(group):
    model, record, _, _, _, _, _ = run(group)
    assert record['scope_derivative_schema'] == 'pooled-local-v1'
    assert record['scope_directional_derivatives'] == pytest.approx(
        {'pooled': 2., 'local:' + group: -4., 'local:B': 6.}, abs=1e-11)
    assert record['gradient_norm'] == pytest.approx(1.2, abs=1e-11)
    assert record['displacement'] == pytest.approx(.1, abs=1e-15)
    assert model.theta.item() == pytest.approx(-.1, abs=1e-15)


def test_boundary_phr_cannot_hide_ascending_violated_pooled_scope():
    model, record, _, before, after, _, _ = run('global', calibrated=True)
    assert record['boundary_policy']['reason'] == 'conflicting_direction'
    assert record['boundary_policy']['conflicting_scopes'] == ['pooled']
    assert record['boundary_policy']['probes'] == []
    assert not record['applied'] and record['radius'] == 0.
    assert record['scope_directional_derivatives']['pooled'] == pytest.approx(2.)
    assert model.theta.item() == 0.
    assert torch.equal(before, after)


def test_fixed_phr_group_relabeling_keeps_exact_math_and_dual_order():
    a = run('A')
    g = run('global')
    assert torch.equal(a[0].theta, g[0].theta)
    assert torch.equal(a[4], g[4])
    for field in ['gradient_norm', 'radius', 'displacement', 'penalty_before',
                  'penalty_after', 'soft_before_global', 'soft_after_global']:
        assert a[1][field] == g[1][field]
    # Lexical order differs: A/B versus B/global. Pooled is always position 0.
    assert a[1]['residuals_before'] == pytest.approx([.4, .8, -.4])
    assert g[1]['residuals_before'] == pytest.approx([.4, -.4, .8])
    assert torch.equal(a[2][[0, 2, 1]], g[2])


def test_zero_gradient_phr_still_declares_schema_and_updates_ordered_dual():
    model, record, dual, before, after, groups, quota = run('global', slope=0.)
    assert record['scope_derivative_schema'] == 'pooled-local-v1'
    assert record['scope_directional_derivatives'] == {}
    assert not record['applied'] and model.theta.item() == 0.
    assert dual.tolist() == pytest.approx([.2, 0., .4])
    assert fixed._audit_phr(record, before, after, groups, quota, [0., 0., 0.]) == dual.tolist()


def test_canonical_recount_keeps_pooled_count_and_cap_separate():
    p = torch.tensor([[.2, .8], [.4, .6]], dtype=torch.double)
    quota = {'global_cap': 1, 'local_caps': {'global': 0, 'B': 1}}
    counts, residuals = fixed._scope_values(p, ['global', 'B'], quota, namespaced=True)
    assert counts == pytest.approx({'pooled': 1.4, 'local:global': .8, 'local:B': .6})
    assert residuals == pytest.approx([.4, -.4, .8])


def test_legacy_recount_refuses_ambiguous_global_group():
    p = torch.tensor([[.2, .8], [.4, .6]], dtype=torch.double)
    quota = {'global_cap': 1, 'local_caps': {'global': 0, 'B': 1}}
    with pytest.raises(RuntimeError, match='ambiguous'):
        fixed._scope_values(p, ['global', 'B'], quota)


def audit(reader, record, before, after, groups, quota):
    if reader is fixed:
        return reader._audit_phr(record, before, after, groups, quota, [0., 0., 0.])
    return reader._audit_side(record, before, after, groups, quota,
                              'phr_local', [0., 0., 0.])


@pytest.mark.parametrize('reader', [fixed, boundary])
def test_native_namespaced_phr_record_passes_correct_reader(reader):
    _, record, dual, before, after, groups, quota = run(
        'global', calibrated=reader is boundary)
    assert audit(reader, record, before, after, groups, quota) == dual.tolist()


@pytest.mark.parametrize('reader', [fixed, boundary])
@pytest.mark.parametrize('schema', ['unrecognized', None, False])
def test_phr_readers_refuse_unknown_schema_even_zero_gradient(reader, schema):
    _, record, _, before, after, groups, quota = run(
        'A', calibrated=reader is boundary, slope=0.)
    record['scope_derivative_schema'] = schema
    with pytest.raises(RuntimeError, match='schema'):
        audit(reader, record, before, after, groups, quota)


@pytest.mark.parametrize('reader', [fixed, boundary])
def test_phr_readers_preserve_unambiguous_legacy_and_refuse_alias(reader):
    _, record, _, before, after, groups, quota = run('A', calibrated=reader is boundary)
    record.pop('scope_derivative_schema')
    record['scope_directional_derivatives'] = {
        ('global' if key == 'pooled' else key.removeprefix('local:')): value
        for key, value in record['scope_directional_derivatives'].items()}
    original = copy.deepcopy(record)
    audit(reader, record, before, after, groups, quota)
    assert record == original
    _, aliased, _, before, after, groups, quota = run('global', calibrated=reader is boundary)
    aliased.pop('scope_derivative_schema')
    aliased['scope_directional_derivatives'] = {'global': -2., 'B': 6.}
    with pytest.raises(RuntimeError, match='ambiguous'):
        audit(reader, aliased, before, after, groups, quota)


def test_phr_reader_rejects_wrong_weighted_scope_derivative():
    _, record, _, before, after, groups, quota = run('global')
    record['scope_directional_derivatives']['pooled'] = -2.
    with pytest.raises(RuntimeError, match='directional gradient identity'):
        audit(fixed, record, before, after, groups, quota)
