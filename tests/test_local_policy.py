import pytest

from tralo.local_policy import size_share_caps


def test_largest_remainder_is_exact_and_ties_use_group_code():
    assert size_share_caps(["A"] * 3 + ["B"] * 2, 3) == {"A": 2, "B": 1}
    assert size_share_caps(["C", "B", "A"], 2) == {"A": 1, "B": 1, "C": 0}
    assert size_share_caps(["A", "A", "B", "C"], 2) == {"A": 1, "B": 1, "C": 0}


def test_input_order_does_not_change_quotas():
    groups = ["C", "A", "B", "A", "C", "C"]
    assert size_share_caps(groups, 4) == size_share_caps(list(reversed(groups)), 4)


@pytest.mark.parametrize("groups,total", [([], 0), (["A"], -1), (["A"], 2),
                                              (["A", ""], 1), (["A"], True)])
def test_invalid_policy_inputs_rejected(groups, total):
    with pytest.raises(ValueError):
        size_share_caps(groups, total)
