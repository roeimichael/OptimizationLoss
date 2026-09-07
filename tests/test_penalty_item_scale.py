"""`penalty_item_scale` -- the units of the constraint gradient. FRAMEWORK 2(z54).

`_penalty` divides the excess by `max(K, 1)`, so `d(pen)/d(soft)` carries units
of 1/budget and a scope's pull PER ITEM is inversely proportional to its own
ceiling. Under `constraint_grad_mode: normalize` the delivered step has fixed
norm and only the RATIOS across scopes steer, so that INVERTS the priority
relative to ALM, whose weight is `lambda + mu * r` in raw items with no
division at all (`src/methodologies/fioretto_alm/train.py:244,253`).

Measured on dom1 over 11,136 logged scope-epochs: TraLO sends 93.5% of its
fixed-norm step to the K = 0 ceilings and 1.7% to the K >= 100 ones; ALM's rule
on the SAME states sends 18.8% and 69.2%.

Every test here has a NEGATIVE CONTROL in the same test: a gate that has never
been shown to fail has never been shown to work.
"""
import pytest
import torch

from src.losses.transductive_loss import MulticlassTransductiveLoss

RHO = 100.0          # rho_target; the ramp's far end, where the spread is widest
NC = 8


def make(item_scale, shape="rational_bounded", rho=RHO):
    return MulticlassTransductiveLoss(
        global_constraints=[1e10] * NC, local_constraints={},
        num_classes=NC, initial_rho=rho,
        penalty_shape=shape, penalty_item_scale=item_scale)


def slope(loss, K, soft, h=1e-4):
    """d(pen)/d(soft) by central difference, in float64 for a clean number."""
    s = torch.tensor(float(soft), dtype=torch.float64)
    kk = float(K)
    up = loss._penalty(s + h, kk)
    dn = loss._penalty(s - h, kk)
    return float((up - dn) / (2 * h))


def pull(loss, K, soft):
    """A_S up to lambda: the per-item weight this scope puts on every logit."""
    return slope(loss, K, soft)


# --------------------------------------------------------------- the property
def test_item_scale_makes_the_slope_dimensionless_across_budgets():
    """The whole point: pull per item stops depending on the scope's ceiling."""
    on = make(True)
    # 10% over budget at four ceilings spanning 3 orders of magnitude
    slopes = [pull(on, K, K * 1.10) for K in (1, 10, 100, 1000)]
    assert min(slopes) > 0
    spread = max(slopes) / min(slopes)
    assert spread < 1.5, (
        "penalty_item_scale should leave the per-item slope comparable across "
        "budgets; got %r, spread %.1fx" % (slopes, spread))

    # NEGATIVE CONTROL: with the flag OFF the same four slopes must span
    # orders of magnitude, or this test is asserting nothing.
    off = make(False)
    off_slopes = [pull(off, K, K * 1.10) for K in (1, 10, 100, 1000)]
    off_spread = max(off_slopes) / min(off_slopes)
    assert off_spread > 100, (
        "the defect this flag fixes must be present when it is off; got %r, "
        "spread %.1fx" % (off_slopes, off_spread))


def test_the_worst_case_ratio_improves_by_orders_of_magnitude():
    """Two real dom1 scope states, at the WORST point for this fix.

    A K = 0 camera group holding 0.55 units of probability mass -- exactly the
    peak of the bounded shape when scale == 1 -- against the global class-7 cap
    at K = 411 sitting 7 items over. The K = 0 scope has nothing at stake: the
    allocator emits zero items there whatever the probabilities are.

    ⛔ AND THE FIX DOES NOT INVERT THIS PAIR, WHICH IS STATED HERE RATHER THAN
    LEFT TO BE DISCOVERED. `scale == 1` at K = 0, so multiplying by it is the
    identity there; the K = 0 pull is UNCHANGED and only the K >= 1 scopes are
    lifted (411x here). The ratio goes 6156x -> 14.9x, a 413x improvement, and
    the K = 0 scope still leads AT THIS POINT.

    It is nonetheless the right change in aggregate, because the 0.55 peak is
    rare in the real distribution -- most K = 0 scopes sit deep in the bounded
    shape's decay region. Measured over all 11,136 dom1 scope-epochs
    (FRAMEWORK 2(z54)):

        budget      TraLO now   item-scaled   ALM
        K = 0           93.5%       15.3%    18.8%
        K = 10..99       4.9%       32.0%    12.0%
        K >= 100         1.7%       52.7%    69.2%

    Fully removing the residual needs a denominator that is not max(K, 1) at
    all -- the scope's ITEM COUNT is the principled one -- which is a second
    arm and is deliberately not folded in here.
    """
    off, on = make(False), make(True)
    k0 = (0, 0.55)            # peak of the bounded shape when scale == 1
    big = (411, 418.0)        # 7 items over, a real dom1 global cap

    before = pull(off, *k0) / pull(off, *big)
    after = pull(on, *k0) / pull(on, *big)
    assert before > 1000, (
        "the defect must be present when the flag is off; ratio %.1fx" % before)
    assert after < before / 100.0, (
        "item scaling must shrink the worst-case ratio by >=100x; %.1fx -> %.1fx"
        % (before, after))
    # the residual, asserted so it cannot silently change
    assert 5.0 < after < 50.0, (
        "the K=0 peak still leads after the fix, by ~15x; got %.1fx" % after)
    # NEGATIVE CONTROL: the K=0 pull itself is untouched (scale == 1 there)
    assert pull(on, *k0) == pytest.approx(pull(off, *k0), rel=1e-12)


def test_linear_shape_becomes_the_raw_excess():
    """`linear` + item scale is exactly ALM's units: slope 1 per item."""
    on = make(True, shape="linear")
    for K in (0, 5, 411):
        assert pull(on, K, K + 3.0) == pytest.approx(1.0, abs=1e-6)
    # NEGATIVE CONTROL: off, the slope is 1/max(K,1) and NOT 1
    off = make(False, shape="linear")
    assert pull(off, 411, 414.0) == pytest.approx(1.0 / 411, rel=1e-3)


# ----------------------------------------------------------- the default holds
def test_default_is_bit_identical_to_the_shipped_penalty():
    """Every stored result must be unaffected. Values from the pre-change form."""
    off = make(False)
    cases = [(0, 0.55), (5, 7.0), (86, 115.25), (411, 418.0), (333, 340.0)]
    for K, soft in cases:
        E = max(0.0, soft - K)
        s = K if K >= 1 else 1.0
        e = E / s
        want = E / (E + s) + RHO * (e * e) / (1 + e * e)
        got = float(off._penalty(torch.tensor(float(soft), dtype=torch.float64), float(K)))
        assert got == pytest.approx(want, rel=1e-6), (K, soft, got, want)

    # NEGATIVE CONTROL: the ON arm must NOT reproduce those values wherever
    # scale != 1, or the flag is inert -- which would be the sixth (rule 3).
    on = make(True)
    differed = 0
    for K, soft in cases:
        a = float(off._penalty(torch.tensor(float(soft), dtype=torch.float64), float(K)))
        b = float(on._penalty(torch.tensor(float(soft), dtype=torch.float64), float(K)))
        if abs(a - b) > 1e-12:
            differed += 1
    assert differed >= 4, (
        "penalty_item_scale changed %d of %d cases; an inert flag is this "
        "project's most frequent failure mode" % (differed, len(cases)))


def test_flag_is_inert_at_K_equals_zero_which_is_why_the_null_is_shared():
    """scale == 1 there, so the arm and its null coincide on K=0 scopes alone."""
    off, on = make(False), make(True)
    t = torch.tensor(0.55, dtype=torch.float64)
    assert float(off._penalty(t, 0.0)) == float(on._penalty(t, 0.0))


# ------------------------------------------------------------- the type guard
def test_a_string_false_is_refused_rather_than_run_as_true():
    """"False" is truthy. Refusing is the only safe reading."""
    with pytest.raises(TypeError, match="penalty_item_scale must be a bool"):
        make("False")
    with pytest.raises(TypeError):
        make(1)
    # NEGATIVE CONTROL: real bools must pass
    make(True)
    make(False)


# ------------------------------------------------------------------ the wiring
def test_the_arm_is_declared_and_differs_from_tralo_in_exactly_one_key():
    import yaml
    with open("configs/protocol.yml", encoding="utf-8") as fh:
        proto = yaml.safe_load(fh)
    arms = proto["arms"]
    assert "tralo_itemscale" in arms, "the arm must exist to be runnable"
    spec = arms["tralo_itemscale"]
    assert spec["methodology"] == "tralo"
    assert spec["null_sibling"] == "tralo_null"

    blocks = proto["blocks"]
    extra = blocks["penalty_item_scale"]
    assert extra == {"penalty_item_scale": True}, extra
    # exactly one key differs from plain `tralo`
    tralo_blocks = arms["tralo"]["blocks"]
    item_blocks = spec["blocks"]
    assert set(item_blocks) - set(tralo_blocks) == {"penalty_item_scale"}
    assert set(tralo_blocks) - set(item_blocks) == set()


def test_the_key_default_is_false_in_the_protocol():
    """A default that lives only in code is invisible to audit_config."""
    import yaml
    with open("configs/protocol.yml", encoding="utf-8") as fh:
        proto = yaml.safe_load(fh)
    assert proto["blocks"]["tralo"]["penalty_item_scale"] is False, (
        "the shipped default must be declared false in the protocol so every "
        "run's config.json records which side of this it was on")


def test_the_trainer_actually_passes_it():
    """AST, not grep: a name in a comment is not an argument."""
    import ast
    src = open("src/methodologies/tralo/train.py", encoding="utf-8").read()
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name != "MulticlassTransductiveLoss":
            continue
        for kw in node.keywords:
            if kw.arg == "penalty_item_scale":
                found = True
    assert found, (
        "tralo/train.py must pass penalty_item_scale to the loss; without it "
        "the config key is read by nobody and the arm is inert flag six")
