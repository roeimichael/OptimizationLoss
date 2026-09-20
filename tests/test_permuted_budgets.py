"""The budget-permuted twin: the matched control for the transductive claim.

The control is only valid if the permutation moves WHICH group holds a ceiling
and changes nothing else. Every invariant that has to hold for that is asserted
here, because a permutation that quietly altered a class total would make the
twin incomparable to the real arm and the comparison would look fine in a log.
"""
import logging

import pandas as pd
import pytest

from src.training.constraints import (
    compute_local_constraints,
    permute_local_budgets,
)
from src.utils.constants import UNLIMITED

CLASSES = [1, 2]
NUM_CLASSES = 4


def frame():
    """Four groups with deliberately unequal per-group class counts."""
    rows = []
    for group, counts in enumerate([(40, 10), (30, 20), (20, 30), (10, 40)]):
        for c, n in zip(CLASSES, counts):
            rows += [{"grp": group, "label": c}] * n
        rows += [{"grp": group, "label": 0}] * 25
    return pd.DataFrame(rows)


def local(**kw):
    return compute_local_constraints(
        frame(), "label", 0.5, "grp", CLASSES, NUM_CLASSES, **kw)


def test_the_default_is_untouched():
    """No flag means the byte-identical legacy derivation."""
    assert local() == local(permute_group_budgets=None)


@pytest.mark.parametrize("shares", [None, "proportional_to_group_size", "equal"])
def test_class_totals_are_preserved_exactly(shares):
    """The tie to Psi depends on the per-class sum, so it must not move."""
    base = local(group_budget_shares=shares)
    perm = local(group_budget_shares=shares, permute_group_budgets=7)
    for c in CLASSES:
        assert sum(b[c] for b in perm.values()) == sum(b[c] for b in base.values())


@pytest.mark.parametrize("shares", [None, "proportional_to_group_size", "equal"])
def test_the_multiset_of_ceilings_is_preserved(shares):
    base = local(group_budget_shares=shares)
    perm = local(group_budget_shares=shares, permute_group_budgets=7)
    for c in CLASSES:
        assert sorted(b[c] for b in perm.values()) == sorted(b[c] for b in base.values())


def test_the_assignment_actually_moves():
    """An identity permutation would make the control measure nothing."""
    base = local()
    perm = local(permute_group_budgets=7)
    assert perm != base
    for c in CLASSES:
        assert [perm[g][c] for g in sorted(perm)] != [base[g][c] for g in sorted(base)]


def test_it_is_deterministic_given_the_seed():
    assert local(permute_group_budgets=3) == local(permute_group_budgets=3)


def test_a_different_seed_gives_a_different_assignment():
    seen = {tuple(sorted((g, tuple(b)) for g, b in local(permute_group_budgets=s).items()))
            for s in range(8)}
    assert len(seen) > 1


def test_uncapped_classes_and_UNLIMITED_are_untouched():
    base = local()
    perm = local(permute_group_budgets=7)
    for g in base:
        for c in range(NUM_CLASSES):
            if c not in CLASSES:
                assert perm[g][c] == base[g][c] == UNLIMITED


def test_a_single_distinct_ceiling_warns_instead_of_pretending(caplog):
    """Equal shares over equal-sized groups can make every ceiling identical.

    That is a real no-op, and the control measures nothing for that class. It
    must say so rather than report a permutation it did not perform.
    """
    flat = {g: [UNLIMITED, 25, UNLIMITED, UNLIMITED] for g in range(4)}
    with caplog.at_level(logging.WARNING):
        out = permute_local_budgets(flat, [1], seed=7)
    assert out == flat
    assert "measures nothing" in caplog.text


def test_one_capped_group_is_a_no_op():
    single = {0: [UNLIMITED, 12, UNLIMITED, UNLIMITED]}
    assert permute_local_budgets(single, [1], seed=7) == single


def test_the_permutation_does_not_mutate_its_input():
    base = local()
    snapshot = {g: list(b) for g, b in base.items()}
    permute_local_budgets(base, CLASSES, seed=7)
    assert base == snapshot
