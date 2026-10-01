"""Local upper bounds cannot manufacture referrals or ignore missing groups."""

import pytest

from tralo.global_clipper import allocate_local_upper_bound


def test_unfilled_capacity_and_missing_group_are_kept_as_upper_bounds():
    probabilities = [[0.1, 0.9], [0.2, 0.8], [0.1, 0.9],
                     [0.9, 0.1], [0.8, 0.2]]
    groups = ["female", "female", "missing", "male", "male"]
    assigned = allocate_local_upper_bound(
        probabilities, [None, 4], ["a", "b", "c", "d", "e"], groups,
        {"female": 1, "male": 1})
    assert assigned == [1, 0, 1, 0, 0]
    assert sum(label == 1 for label in assigned) == 2  # Four slots were offered.


def test_pooled_cap_uses_score_then_stable_id_after_group_limits():
    probabilities = [[0.2, 0.8], [0.1, 0.9], [0.2, 0.8]]
    assigned = allocate_local_upper_bound(
        probabilities, [None, 1], ["b", "c", "a"],
        ["female", "male", "missing"], {"female": 1, "male": 1})
    assert assigned == [0, 1, 0]


def test_rejects_unseen_local_group():
    with pytest.raises(ValueError, match="invalid local"):
        allocate_local_upper_bound([[0.2, 0.8]], [None, 1], ["a"],
                                   ["female"], {"male": 1})
