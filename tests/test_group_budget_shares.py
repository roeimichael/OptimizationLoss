"""Externally given per-group budgets: Phi[lambda][i] as a POLICY, not a prevalence.

Every test here has a negative control, because the failure mode this guards
against is silent: a per-group ceiling derived from that group's own count
looks completely normal in every log while encoding the opposite policy.
"""
import pandas as pd
import pytest

from src.training.constraints import (
    compute_global_constraints,
    compute_local_constraints,
    normalize_group_shares,
    split_budget,
)
from src.utils.constants import UNLIMITED

CAPPED = [1, 2]
NUM_CLASSES = 3
SHARES = {0: 0.10, 1: 0.30, 2: 0.60}
# tier -> class -> count. Tiers are near-equal (334/333/333) and the tier
# entitled to 60% of the beds supplies the fewest candidates.
POOL = {0: {0: 14, 1: 120, 2: 200},
        1: {0: 163, 1: 50, 2: 120},
        2: {0: 223, 1: 30, 2: 80}}


def hospital():
    rows = [(c, g) for g, by_c in POOL.items() for c, n in by_c.items() for _ in range(n)]
    return pd.DataFrame(rows, columns=["label", "tier"])


def local(frame, pct, **kw):
    return compute_local_constraints(frame, "label", pct, "tier",
                                     constrained_class=CAPPED,
                                     num_classes=NUM_CLASSES, **kw)


def test_hospital_shape_is_the_policy_not_the_prevalence():
    """100 normal beds and 200 special beds, split 10/30/60 and 20/60/120."""
    frame = hospital()
    psi = compute_global_constraints(frame, "label", 0.50,
                                     constrained_class=CAPPED, num_classes=NUM_CLASSES)
    assert [psi[1], psi[2]] == [100, 200]

    phi = local(frame, 0.50, group_budget_shares=SHARES)
    assert [phi[g][1] for g in (0, 1, 2)] == [10, 30, 60]
    assert [phi[g][2] for g in (0, 1, 2)] == [20, 60, 120]
    # uncapped classes stay uncapped
    assert all(phi[g][0] == UNLIMITED for g in (0, 1, 2))


def test_negative_control_prevalence_gives_a_different_and_wrong_split():
    """The control the policy path has to beat: same totals, reversed order."""
    phi = local(hospital(), 0.50)
    assert [phi[g][1] for g in (0, 1, 2)] == [60, 25, 15]
    assert [phi[g][2] for g in (0, 1, 2)] == [100, 60, 40]
    assert [phi[g][1] for g in (0, 1, 2)] != [10, 30, 60]


def test_equal_percentages_make_local_sum_to_global_exactly():
    """Shifman et al. section 3.2 uses one percentage for both scopes.

    That is the property that makes the global ceiling bind. It must hold
    EXACTLY, not approximately -- rounding each group independently loses items.
    """
    frame = hospital()
    for pct in (0.10, 0.25, 0.33, 0.50, 0.77, 0.90):
        psi = compute_global_constraints(frame, "label", pct,
                                         constrained_class=CAPPED, num_classes=NUM_CLASSES)
        phi = local(frame, pct, group_budget_shares=SHARES)
        for c in CAPPED:
            assert sum(phi[g][c] for g in phi) == psi[c], (pct, c)


def test_unequal_percentages_leave_headroom_and_that_is_the_deviation():
    """L80_G95, the pair every campaign in this corpus ran, cannot bind globally."""
    frame = hospital()
    psi = compute_global_constraints(frame, "label", 0.95,
                                     constrained_class=CAPPED, num_classes=NUM_CLASSES)
    phi = local(frame, 0.80, group_budget_shares=SHARES)
    for c in CAPPED:
        assert sum(phi[g][c] for g in phi) < psi[c]


def test_largest_remainder_beats_independent_rounding():
    """Three equal shares of 100: round() gives 33/33/33 and loses an item."""
    parts = split_budget(100, {0: 1 / 3.0, 1: 1 / 3.0, 2: 1 / 3.0}, "unit")
    assert sum(parts.values()) == 100
    assert sorted(parts.values()) == [33, 33, 34]
    # ties break on the lowest group id, deterministically
    assert parts[0] == 34


def test_split_is_exact_for_awkward_shares():
    for total in (1, 7, 99, 100, 347, 519):
        parts = split_budget(total, SHARES, "unit")
        assert sum(parts.values()) == total


@pytest.mark.parametrize("bad, match", [
    ({0: 0.1, 1: 0.3}, "missing"),
    ({0: 0.1, 1: 0.3, 2: 0.6, 9: 0.0}, "extra"),
    ({0: 0.1, 1: 0.3, 2: 0.5}, "sum to"),
    ({0: -0.1, 1: 0.4, 2: 0.7}, "negative"),
    ([0.1, 0.3, 0.6], "mapping"),
])
def test_negative_controls_every_malformed_policy_raises(bad, match):
    with pytest.raises(ValueError, match=match):
        normalize_group_shares(bad, [0, 1, 2])


def test_negative_control_malformed_policy_raises_through_the_real_entry_point():
    with pytest.raises(ValueError, match="missing"):
        local(hospital(), 0.50, group_budget_shares={0: 0.4, 1: 0.6})


def test_default_path_is_untouched():
    """No shares -> the historical derivation, so every completed run stays comparable."""
    frame = hospital()
    assert local(frame, 0.50) == local(frame, 0.50, group_budget_shares=None)
