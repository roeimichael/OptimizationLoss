"""Scope identity regressions, using fixed CPU examples without RNG draws."""
import copy
import math

import pytest
import torch

from analysis import score_fmow_local as prior
from analysis import score_fmow_local_alm as alm_score
from analysis import score_fmow_local_fixed as fixed
from tralo.local_targeted_step import local_targeted_step


class OpposedScopes(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0., dtype=torch.double))

    def forward(self, images):
        one = images[:, 0] + self.theta * images[:, 1]
        return torch.stack([torch.zeros_like(one), one], dim=1)


def inputs(group):
    return [torch.tensor([[math.log(4.), 25.], [math.log(1.5), -25.]],
                         dtype=torch.double)], [group, 'B'], {group: 0, 'B': 1}


def test_local_global_name_cannot_hide_pooled_ascent():
    # q=(.8,.6), dq=(4,-6): pooled=-2, local(global)=4, direction=-1.
    model = OpposedScopes()
    chunks, groups, caps = inputs('global')
    before = model.theta.detach().clone()
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='scope descent'):
        local_targeted_step(model, chunks, groups, 1, 1, caps,
                            fixed_radius=.0001)
    assert torch.equal(model.theta, before)
    assert model.training
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize('group', ['global', 'pooled', 'local:A', 'A:B', 'A'])
def test_fixed_dose_logs_each_exact_constraint_identity(group):
    model = OpposedScopes()
    chunks, groups, caps = inputs(group)
    out = local_targeted_step(model, chunks, groups, 1, 1, caps,
                              fixed_radius=.0001, require_common_descent=False)
    assert out['scope_derivative_schema'] == 'pooled-local-v1'
    assert out['scope_directional_derivatives'] == pytest.approx(
        {'pooled': 2., 'local:' + group: -4.}, abs=1e-11)
    assert out['gradient_norm'] == pytest.approx(2., abs=1e-11)
    assert model.theta.item() == pytest.approx(-.0001, abs=1e-15)
    assert out['displacement'] == pytest.approx(.0001, abs=1e-15)


def test_group_relabeling_preserves_applied_math_and_observed_counts():
    rows = []
    models = []
    for group in ['A', 'global']:
        model = OpposedScopes()
        chunks, groups, caps = inputs(group)
        out = local_targeted_step(model, chunks, groups, 1, 1, caps,
                                  fixed_radius=.0001, require_common_descent=False)
        rows.append(out)
        models.append(model)
    assert torch.equal(models[0].theta, models[1].theta)
    for key in ['radius', 'displacement', 'gradient_norm', 'soft_after_global',
                'hard_after_global', 'directional_soft_delta_global']:
        assert rows[0][key] == rows[1][key]


def side_record():
    return dict(applied=True, radius=.1, displacement=.1, gradient_norm=.7,
        tensor_displacement_norms=[.1], hard_before_global=3,
        hard_before_local={'A': 2, 'B': 1}, hard_after_global=3,
        hard_after_local={'A': 2, 'B': 1}, soft_before_global=2.4,
        soft_before_local={'A': 1.5, 'B': .9}, soft_after_global=2.4,
        soft_after_local={'A': 1.5, 'B': .9}, active_global=True,
        active_local=['A'], scope_directional_derivatives={'global': -.4, 'A': -.2},
        directional_soft_delta_global=-.01,
        directional_soft_delta_local={'A': -.02, 'B': 0.})


def audit(reader, record):
    quota = {'global_cap': 2, 'local_caps': {'A': 1, 'B': 1}}
    if reader is prior:
        reader._audit_scope_numbers(record, quota)
    else:
        p = torch.tensor([[.1, .8, .1, 0., 0., 0., 0., 0.],
                          [.1, .7, .2, 0., 0., 0., 0., 0.],
                          [.1, .9, 0., 0., 0., 0., 0., 0.]])
        reader._audit_side(record, p, p.clone(), ['A', 'A', 'B'], quota, 'joint')


@pytest.mark.parametrize('reader', [prior, fixed, alm_score])
def test_auditors_accept_namespaced_records_without_rewriting_legacy(reader):
    old = side_record()
    preserved = copy.deepcopy(old)
    audit(reader, old)
    new = copy.deepcopy(old)
    new['scope_derivative_schema'] = 'pooled-local-v1'
    new['scope_directional_derivatives'] = {'pooled': -.4, 'local:A': -.2}
    audit(reader, new)
    assert old == preserved


@pytest.mark.parametrize('reader', [prior, fixed, alm_score])
@pytest.mark.parametrize('schema', ['unrecognized', None])
def test_auditors_refuse_unknown_or_mislabeled_namespaced_logs(reader, schema):
    record = side_record()
    record['scope_derivative_schema'] = schema
    record['scope_directional_derivatives'] = {'pooled': -.4, 'local:A': -.2}
    with pytest.raises(RuntimeError):
        audit(reader, record)


def test_legacy_alias_is_not_accepted_as_two_independent_derivatives():
    record = side_record()
    record['active_local'] = ['global']
    record['scope_directional_derivatives'] = {'global': -.4}
    record['soft_before_local'] = {'global': 1.5, 'B': .9}
    record['soft_after_local'] = {'global': 1.5, 'B': .9}
    record['directional_soft_delta_local'] = {'global': -.02, 'B': 0.}
    with pytest.raises(RuntimeError):
        prior._audit_scope_numbers(record, {'global_cap': 2,
                                          'local_caps': {'global': 1, 'B': 1}})
