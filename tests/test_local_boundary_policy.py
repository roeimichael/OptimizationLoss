"""Hand-checkable contracts for a label-free snapshot acceptance policy."""

import math

import pytest

from tralo.local_boundary_policy import choose_boundary_step


def _choose(probe, **changes):
    inputs = dict(
        pooled_hard=8, pooled_soft=8.0, pooled_cap=5,
        local_soft={"A": 4.0, "B": 4.0},
        local_caps={"A": 3, "B": 4},
        pooled_derivative=-1.0,
        local_derivatives={"A": -1.0, "B": -1.0},
        probe=probe,
    )
    inputs.update(changes)
    return choose_boundary_step(**inputs)


def test_local_quotas_are_upper_bounds_even_when_their_sum_exceeds_global():
    observed = []

    def probe(radius):
        observed.append(radius)
        return dict(pooled_hard=5, pooled_soft=5.0,
                    local_soft={"B": 3.0, "A": 2.0})

    result = _choose(probe, local_caps={"A": 6, "B": 6},
                     local_derivatives={"A": -1.0, "B": -1.0})
    assert result["applied"] is True
    assert result["radius"] == pytest.approx(0.1)
    assert observed == [pytest.approx(0.1)]
    assert result["probes"][0]["accepted"] is True


def test_overshoot_backtracks_to_first_acceptable_probe():
    def probe(radius):
        if radius > 0.05:
            return dict(pooled_hard=0, pooled_soft=1.0,
                        local_soft={"A": 0.5, "B": 0.5})
        return dict(pooled_hard=5, pooled_soft=5.0,
                    local_soft={"A": 2.5, "B": 2.5})

    result = _choose(probe)
    assert result["radius"] == pytest.approx(0.05)
    assert [item["radius"] for item in result["probes"]] == pytest.approx([0.1, 0.05])
    assert "pooled_hard_floor" in result["probes"][0]["rejections"]
    assert result["probes"][1]["accepted"]


def test_violated_scope_with_non_descent_direction_skips_without_probe():
    def probe(_):
        raise AssertionError("conflicting direction must not be probed")

    result = _choose(probe, local_derivatives={"A": 0.0, "B": -1.0})
    assert not result["applied"]
    assert result["reason"] == "conflicting_direction"
    assert result["radius"] == 0.0
    assert result["conflicting_scopes"] == ["local:A"]


def test_hard_count_jump_can_block_all_candidates():
    def probe(radius):
        return dict(pooled_hard=3, pooled_soft=5.0,
                    local_soft={"A": 2.5, "B": 2.5})

    result = _choose(probe, max_halvings=2)
    assert not result["applied"]
    assert result["reason"] == "no_acceptable_probe"
    assert result["radius"] == 0.0
    assert len(result["probes"]) == 3


def test_pooled_and_active_local_violations_must_improve():
    def probe(radius):
        return dict(pooled_hard=6, pooled_soft=7.0,
                    local_soft={"A": 3.5, "B": 3.5})

    result = _choose(probe)
    assert result["applied"]
    assert result["probes"][0]["total_positive_violation"] < result["initial_total_positive_violation"]


def test_active_local_violation_cannot_worsen_even_if_total_improves():
    result = _choose(
        lambda radius: dict(pooled_hard=6, pooled_soft=7.0,
                            local_soft={"A": 4.2, "B": 2.8}),
        max_halvings=0,
    )
    assert not result["applied"]
    assert "worsened_soft_violation:local:A" in result["probes"][0]["rejections"]


@pytest.mark.parametrize("change", [
    {"pooled_hard": True}, {"pooled_soft": math.nan}, {"pooled_cap": -1},
    {"local_soft": {"A": math.inf, "B": 1.0}},
    {"local_soft": {"A": 3.0, "B": 4.0}},
    {"local_caps": {}},
    {"local_caps": {"A": 3}},
    {"pooled_derivative": math.inf},
    {"local_derivatives": {"A": -1.0}},
    {"max_radius": 0.0}, {"max_halvings": -1},
])
def test_invalid_inputs_rejected(change):
    with pytest.raises(ValueError):
        _choose(lambda radius: None, **change)


@pytest.mark.parametrize("bad", [
    {"pooled_hard": 3, "pooled_soft": math.nan,
     "local_soft": {"A": 1.0, "B": 2.0}},
    {"pooled_hard": 3, "pooled_soft": 4.0,
     "local_soft": {"A": 1.0}},
    {"pooled_hard": True, "pooled_soft": 4.0,
     "local_soft": {"A": 1.0, "B": 3.0}},
    {"pooled_hard": 3, "pooled_soft": 4.0,
     "local_soft": {"A": 1.0, "B": 2.0}},
])
def test_invalid_probe_rejected(bad):
    with pytest.raises(ValueError):
        _choose(lambda radius: bad)


def test_group_order_cannot_change_decision_or_logged_evidence():
    def probe(radius):
        return dict(pooled_hard=5, pooled_soft=5.0,
                    local_soft={"B": 2.0, "A": 3.0})

    first = _choose(probe)
    second = _choose(probe, local_soft={"B": 4.0, "A": 4.0},
                     local_caps={"B": 4, "A": 3},
                     local_derivatives={"B": -1.0, "A": -1.0})
    assert first == second


def test_local_group_named_pooled_remains_a_distinct_scope():
    result = _choose(
        lambda radius: dict(pooled_hard=5, pooled_soft=5.0,
                            local_soft={"pooled": 2.0, "B": 3.0}),
        local_soft={"pooled": 4.0, "B": 4.0},
        local_caps={"pooled": 3, "B": 4},
        local_derivatives={"pooled": -1.0, "B": -1.0},
    )
    assert result["applied"]
    assert set(result["initial_positive_violations"]) == {"pooled", "local:pooled", "local:B"}


def test_conflicting_local_group_named_pooled_stays_qualified_in_evidence():
    result = _choose(
        lambda radius: None,
        local_soft={"pooled": 4.0, "B": 4.0},
        local_caps={"pooled": 3, "B": 4},
        local_derivatives={"pooled": 0.0, "B": -1.0},
    )
    assert result["conflicting_scopes"] == ["local:pooled"]


def test_an_initially_inactive_local_scope_cannot_become_materially_violated():
    result = _choose(
        lambda radius: dict(pooled_hard=5, pooled_soft=7.0,
                            local_soft={"A": 2.0, "B": 5.0}),
        max_halvings=0,
    )
    assert not result["applied"]
    assert "worsened_soft_violation:local:B" in result["probes"][0]["rejections"]


def test_linear_residual_ratio_sets_initial_radius_below_ceiling():
    result = _choose(
        lambda radius: dict(pooled_hard=5, pooled_soft=5.0,
                            local_soft={"A": 2.0, "B": 3.0}),
        pooled_derivative=-20.0,
    )
    assert result["applied"]
    assert result["initial_radius"] == pytest.approx(0.03)
