import itertools

import pytest

from tralo.global_clipper import allocate, allocate_local_capped_first
from tralo.quotas import audit_quotas


def example():
    probabilities = [
        [0.10, 0.80, 0.10],
        [0.10, 0.70, 0.20],
        [0.10, 0.60, 0.30],
        [0.15, 0.75, 0.10],
        [0.20, 0.65, 0.15],
        [0.50, 0.40, 0.10],
    ]
    return probabilities, ["a", "b", "c", "d", "e", "f"], ["A"] * 3 + ["B"] * 3


def test_joint_local_and_global_caps_match_independent_exhaustive_oracle():
    probs, ids, groups = example()
    caps, local = [None, 3, None], {"A": 1, "B": 3}
    result = allocate_local_capped_first(probs, caps, ids, groups, local)
    assert [ids[i] for i, prediction in enumerate(result) if prediction == 1] == ["a", "d", "e"]
    report = audit_quotas(result, groups, 3, caps,
                          {g: [None, k, None] for g, k in local.items()})
    assert report["feasible"] and report["global_counts"][1] == 3
    assert report["local_counts"]["A"][1] == 1
    feasible = []
    for chosen in itertools.combinations(range(len(ids)), 3):
        if all(sum(groups[i] == g for i in chosen) <= k for g, k in local.items()):
            feasible.append(sum(probs[i][1] for i in chosen))
    assert sum(probs[i][1] for i, prediction in enumerate(result) if prediction == 1) == max(feasible)


def test_without_binding_local_limits_matches_global_capped_first():
    probs, ids, groups = example()
    caps = [None, 3, None]
    assert allocate_local_capped_first(probs, caps, ids, groups, {"A": 3, "B": 3}) == \
        allocate(probs, caps, ids, "capped_first")


def test_permutation_and_tie_break_are_sample_id_based():
    probs, ids, groups = example()
    probs[1] = probs[0].copy()
    reference = allocate_local_capped_first(probs, [None, 2, None], ids, groups, {"A": 1, "B": 2})
    order = list(reversed(range(len(ids))))
    permuted = allocate_local_capped_first([probs[i] for i in order], [None, 2, None],
                                           [ids[i] for i in order], [groups[i] for i in order],
                                           {"A": 1, "B": 2})
    assert dict(zip(ids, reference)) == dict(zip([ids[i] for i in order], permuted))
    assert dict(zip(ids, reference))["a"] == 1


def test_zero_local_cap_and_underfilled_global_cap():
    probs, ids, groups = example()
    result = allocate_local_capped_first(probs, [None, 3, None], ids, groups, {"A": 0, "B": 1})
    assert sum(prediction == 1 for prediction in result) == 1
    assert result[3] == 1


@pytest.mark.parametrize("groups,local", [
    (["A"] * 6, {"B": 1}),
    (["A"] * 5, {"A": 1}),
    (["A"] * 6, {"A": -1}),
    (["A"] * 6, {"A": True}),
])
def test_invalid_local_inputs_rejected(groups, local):
    probs, ids, _ = example()
    with pytest.raises(ValueError):
        allocate_local_capped_first(probs, [None, 3, None], ids, groups, local)
