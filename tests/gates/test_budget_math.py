"""Integer scope arithmetic, independent of historical dataset measurements."""

import pytest
from scripts.headroom import effective_budget
from scripts.verify_caps import duplicate_budget_tags
from src.utils.constants import UNLIMITED

pytestmark = pytest.mark.stage2_budget


def test_effective_budget_respects_both_scopes():
    assert effective_budget({1: 8}, {0: {1: 2}, 1: {1: 3}}, 1) == 5
    assert effective_budget({1: 4}, {0: {1: 2}, 1: {1: 3}}, 1) == 4
    assert effective_budget({1: 8}, {0: {1: 2}, 1: {1: UNLIMITED}}, 1) == 8
    assert effective_budget({1: 0}, {0: {1: 2}}, 1) == 0


def test_two_cap_tags_are_not_two_cap_levels_unless_the_budget_differs():
    assert duplicate_budget_tags({1: {"L30_G30": 3, "L30_G50": 3}}) == [
        (1, 3, ["L30_G30", "L30_G50"])
    ]
    assert duplicate_budget_tags({1: {"L30_G30": 3, "L50_G50": 5}}) == []
