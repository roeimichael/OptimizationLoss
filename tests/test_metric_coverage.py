"""Does every number the scorers COMPUTE actually reach a reader?

Written 2026-08-31 after `full_panel.panel()` was found computing `uncF1` and
leaving it out of `GROUPS`. It sat in the frame from line ~413 and was printed
by nothing, so the project's own standing rule -- *read uncF1 beside macroF1* --
could not be satisfied from the tool's output, and every "the constraint damages
the uncapped classes" number had to be recomputed by hand.

**A metric that is computed and not printed is worse than one that is missing:
it looks covered.** That is why this is a gate and not a note.

The class was not unique, which is the whole reason for a permanent test:
`straddle_probe.measured_delta` computed `median` and `max` per class and read
only `q`, while `q` is the anchor of the entire `[q, 2q, 10q]` ladder. Both are
fixed; these gates keep them fixed.

Every gate here carries its own NEGATIVE CONTROL in the same file -- a
deliberately corrupted copy of the source that the checker must REJECT. A gate
nobody has watched fail is not a gate, and this project has shipped four inert
flags that every green check missed.

Nothing here needs a GPU, a dataset, or a run: it is all AST over the repo.
"""
import ast
import io
import os
import sys

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _src(rel):
    return io.open(os.path.join(REPO, rel), encoding="utf-8").read()


# --------------------------------------------------------------------------
# 1. full_panel: every computed metric reaches a printed group
# --------------------------------------------------------------------------

# Keys of the per-run row that are NOT metrics and so are legitimately absent
# from GROUPS. Each one identifies a run or scales a number; none is scored.
# This list is deliberately explicit -- adding a metric and silencing this gate
# by dropping its name in here is a reviewable act, not an accident.
FULL_PANEL_IDENTITY = frozenset({
    "raw_md5",    # the md5 of final_predictions_raw.csv -- the inert-arm check
    "dataset", "model", "cap", "capped", "seed", "arm",   # the cell key
    "items_per_001",  # the dF1 -> items scale; read at the RESOLUTION block
})


def _panel_row_keys(src):
    """String keys of the widest dict literal built inside `panel()`.

    The row dict is the one with every metric in it; `panel` also builds a
    small {label, grp} dict, which is why this takes the widest and not all.
    """
    tree = ast.parse(src)
    best = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "panel":
            for d in ast.walk(node):
                if isinstance(d, ast.Dict):
                    ks = {k.value for k in d.keys
                          if isinstance(k, ast.Constant)
                          and isinstance(k.value, str)}
                    if len(ks) > len(best):
                        best = ks
    return best


def _panel_printed_names(src):
    """Every metric name reachable by a printed group or the resolution block."""
    tree = ast.parse(src)
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names & {"GROUPS", "EQ_RESOLUTION"}:
                for c in ast.walk(node.value):
                    if isinstance(c, ast.Constant) and isinstance(c.value, str):
                        out.add(c.value)
    return out


def _unprinted_metrics(src):
    return _panel_row_keys(src) - _panel_printed_names(src) - FULL_PANEL_IDENTITY


def test_every_metric_full_panel_computes_is_also_printed():
    left = _unprinted_metrics(_src("scripts/full_panel.py"))
    assert not left, (
        "full_panel.panel() computes %s and no printed group names them. "
        "They exist in the frame and reach nobody -- this is the uncF1 bug. "
        "Add them to GROUPS (and to EQ_RESOLUTION if budget-equalized), or, "
        "if they are identity/scale rather than a result, declare them in "
        "FULL_PANEL_IDENTITY with a comment saying why."
        % sorted(left))


def test_NEGATIVE_CONTROL_the_gate_catches_a_metric_dropped_from_GROUPS():
    """Delete uncF1 from both printed lists: the gate must SEE it go dark.

    This is the exact state the repo was in before 2026-08-30.
    """
    src = _src("scripts/full_panel.py")
    assert src.count('"uncF1",') == 2, (
        "expected uncF1 in EQ_RESOLUTION and in GROUPS; the negative control "
        "below assumes both, so a change here must be reflected there")
    corrupted = src.replace('"uncF1",', "", 2)
    assert _unprinted_metrics(corrupted) == {"uncF1"}, (
        "the gate FAILED TO FAIL on a source with uncF1 removed from every "
        "printed group -- it cannot detect the bug it exists to detect")


# --------------------------------------------------------------------------
# 2. straddle_probe: no per-class statistic computed and never read
# --------------------------------------------------------------------------

def _measured_delta_keys(src):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "measured_delta":
            out = set()
            for d in ast.walk(node):
                if isinstance(d, ast.Dict):
                    out |= {k.value for k in d.keys
                            if isinstance(k, ast.Constant)
                            and isinstance(k.value, str)}
            return out
    raise AssertionError("measured_delta not found in straddle_probe")


def _unread_delta_keys(src):
    """A key is READ if it appears as a subscript anywhere outside the builder."""
    keys = _measured_delta_keys(src)
    tree = ast.parse(src)
    read = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                read.add(node.slice.value)
    return keys - read


def test_straddle_probe_reads_every_per_class_statistic_it_computes():
    left = _unread_delta_keys(_src("scripts/straddle_probe.py"))
    assert not left, (
        "straddle_probe.measured_delta computes %s and nothing reads them. "
        "The q95 it does read is the anchor of the [q, 2q, 10q] ladder, so a "
        "reader needs the median and max to tell an anchor from a tail "
        "artefact. Print them or delete them." % sorted(left))


def test_NEGATIVE_CONTROL_the_straddle_gate_catches_an_unread_statistic():
    src = _src("scripts/straddle_probe.py")
    # remove the two print lines that surface median/max, leaving the
    # computation in place -- the state this file was in until 2026-08-31
    corrupted = src.replace('"c%d=%.4f/%.4f" % (c, v["median"], v["max"])', '""')
    assert corrupted != src, "negative control did not modify the source"
    # `max` acquired a SECOND reader on 2026-09-01: `is_inert` uses it to tell
    # an arm identical to its own `_null` from an unreachable cut. So removing
    # the print alone no longer orphans it, and the control must say which
    # site it removed -- otherwise this test silently weakens the day any
    # statistic gains a second consumer.
    assert _unread_delta_keys(corrupted) == {"median"}, (
        "the gate FAILED TO FAIL when median stopped being read")
    both = corrupted.replace('return all(v["max"] == 0.0 for v in disp.values())',
                             "return False")
    assert both != corrupted, "is_inert is no longer the second reader of max"
    assert _unread_delta_keys(both) == {"median", "max"}, (
        "with BOTH readers removed the gate must orphan max as well")


# --------------------------------------------------------------------------
# 3. tralo_lam0 -- the equal-dose control (FRAMEWORK 2(z3))
# --------------------------------------------------------------------------

def _protocol():
    return yaml.safe_load(_src("configs/protocol.yml"))


def test_tralo_lam0_starts_at_zero_lambda_but_keeps_the_ratchet():
    """It must differ from `tralo_null` in exactly one way: the ratchet lives.

    `tralo` takes a real constraint step at epoch 1 (lambda_global 0.01, logged
    grad norm 3.09) while `fioretto` and `hounie` guard on `has_work` and step
    zero times -- a 1-in-29 dose advantage to TraLO in every head-to-head this
    repo has ever run. `tralo_lam0` removes it by starting lambda at 0, so
    epoch 1 carries no gradient and the ratchet lifts lambda at the end of it.

    If `lambda_step` were also 0 this would silently BE `tralo_null` and the
    equal-dose campaign would compare a null against a null.
    """
    hp = _protocol()["blocks"]
    lam0 = hp["tralo_lam0"]
    assert lam0["lambda_global"] == 0.0
    assert lam0["lambda_local"] == 0.0
    assert lam0["lambda_step"] > 0, (
        "tralo_lam0 has lambda_step 0, which makes it byte-identical to "
        "tralo_null: lambda starts at 0 and never rises, so no constraint is "
        "ever applied and the arm measures nothing")
    assert lam0["lambda_step"] == hp["tralo"]["lambda_step"], (
        "the ratchet must be UNCHANGED from `tralo` -- the arm isolates the "
        "STARTING lambda, and changing two things at once makes the contrast "
        "uninterpretable")


def test_tralo_lam0_is_registered_as_trained_with_a_null_sibling():
    arms = _protocol()["arms"]
    entry = arms["tralo_lam0"]
    assert entry["methodology"] == "tralo"
    assert entry["phase"] == "trained"
    assert entry["null_sibling"] == "tralo_null", (
        "a trained arm without a null sibling cannot be attributed: its delta "
        "is confounded with the 29 extra optimizer epochs")
    assert "tralo_lam0" in entry["blocks"], (
        "the arm's own hyperparameter block is not applied, so it would run "
        "as plain `tralo` -- a fifth inert flag")


# --------------------------------------------------------------------------
# 4. the quarantine registry
# --------------------------------------------------------------------------

def test_every_quarantine_entry_carries_a_reason_and_is_unscorable():
    from scripts.quarantine import REGISTRY
    assert REGISTRY, "the registry is empty"
    for name, e in REGISTRY.items():
        assert e.get("reason"), "%s has no reason" % name
        assert e.get("keep_for"), (
            "%s says why it is dead but not what it is still evidence FOR. "
            "Without that the next session deletes a receipt." % name)
        # THREE STATES since 2026-09-04 (FRAMEWORK 2(z40)): `scorable=False`
        # blocks the campaign, `scorable=True` WITH `dead_arms` blocks only
        # contrasts touching those arms, and `scorable=True` with no dead arms
        # is the contradiction this line originally existed to catch -- a
        # registry row that blocks nothing at all.
        assert e.get("scorable") is False or e.get("dead_arms"), (
            "%s is in the quarantine registry, claims to be scorable, and "
            "names no dead arms, so it blocks nothing" % name)


def test_an_unreadable_quarantine_marker_fails_CLOSED(tmp_path):
    """A corrupt marker must quarantine, never wave the campaign through."""
    from scripts.quarantine import MARKER, is_quarantined
    root = tmp_path / "somecampaign"
    root.mkdir()
    (root / MARKER).write_text("{not json", encoding="utf-8")
    verdict = is_quarantined(str(root))
    assert verdict and verdict.get("scorable") is False, (
        "an unreadable QUARANTINE.json read as SCORABLE -- the gate fails "
        "OPEN, which is the one way a quarantine gate must never fail")


# --------------------------------------------------------------------------
# 5. dose_landed prints the CROSS-ARM block
# --------------------------------------------------------------------------

def test_dose_landed_reports_cross_arm_attempts_not_only_within_arm():
    """`applied/attempted` is blind to a dose gap BETWEEN arms.

    It printed 100.0% for all four duals on dom1 while `tralo` and `alm`
    attempted 29 constraint steps per run and `fioretto` and `hounie`
    attempted 28. The within-arm ratio cannot express that, so the cross-arm
    block is the only place the asymmetry is visible.
    """
    tree = ast.parse(_src("scripts/dose_landed.py"))
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "cross_arm_attempts" in defined
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "cross_arm_attempts" in called, (
        "cross_arm_attempts is defined and never called -- the block exists "
        "in the file and prints for nobody, which is the uncF1 bug in a "
        "different tool")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# --------------------------------------------------------------------------
# 6. the budget is equalized: every arm emits EXACTLY K, both directions
# --------------------------------------------------------------------------
# The corpus mistake was that cc-F1 was partly a BUDGET measurement:
# corr(budget, d ccF1) was +0.81 on `hounie`, and the hinge arm emitted 16.3%
# more predictions in 24/24 pairs so part of its +3.23 pp was free fill.
# Verified 2026-08-31 that the live pipeline cannot do this -- 16 of 16 dom1
# arms deploy an identical 296/364 while their RAW counts span 278-391 and
# 430-550. These gates keep it that way, and the one that matters is the
# UNDERSHOOT case: clipping down is the obvious half, filling up is the half
# that makes two arms comparable.

def _alloc(n_cls, argmax_class, n, K, capped):
    """Probabilities whose plain argmax is `argmax_class` for every item."""
    import numpy as np
    from scripts.full_panel import equalize_multi
    rng = np.random.RandomState(0)
    P = rng.uniform(0.0, 0.10, size=(n, n_cls))
    P[:, argmax_class] += 0.80
    P = P / P.sum(axis=1, keepdims=True)
    glob = np.full(n_cls, 10 ** 9, dtype=float)
    for c in capped:
        glob[c] = K
    return equalize_multi(P, None, glob, {}, list(capped)), P


def test_allocator_fills_UP_to_exactly_K_when_the_model_undershoots():
    """The half that makes arms comparable, and the half that is easy to lose.

    `hounie` and `tralo_uniform` both finish UNDER budget on dom1 (raw 288 and
    281 against K=296) and both deploy exactly 296. If undershoot were left
    short, an arm would be penalised for satisfying the cap early and the
    comparison would become a budget measurement.
    """
    import numpy as np
    # argmax is class 0 for every item, so class 1 is raw-EMPTY
    assigned, _ = _alloc(n_cls=3, argmax_class=0, n=100, K=30, capped=(1,))
    got = int((assigned == 1).sum())
    assert got == 30, (
        "the allocator emitted %d for a capped class with K=30 when the model's "
        "own argmax emitted 0. Undershoot must be filled UP to exactly K, or "
        "arms are compared at different budgets." % got)


def test_allocator_clips_DOWN_to_exactly_K_when_the_model_overshoots():
    import numpy as np
    # argmax is the capped class 1 for every one of the 100 items
    assigned, _ = _alloc(n_cls=3, argmax_class=1, n=100, K=30, capped=(1,))
    got = int((assigned == 1).sum())
    assert got == 30, (
        "the allocator emitted %d for a capped class with K=30 when the model "
        "wanted 100. Overshoot must be clipped DOWN to exactly K." % got)


def test_NEGATIVE_CONTROL_a_clip_only_allocator_FAILS_the_undershoot_gate():
    """Prove the undershoot gate can fail, by scoring what clip-only would give.

    A naive `argmax then clip` never adds predictions, so on the undershoot
    fixture it emits 0 against K=30. If the gate above ever passes for an
    implementation that behaves like this, the gate is broken.
    """
    import numpy as np
    _, P = _alloc(n_cls=3, argmax_class=0, n=100, K=30, capped=(1,))
    clip_only = P.argmax(axis=1)                      # never fills, only clips
    got = int((clip_only == 1).sum())
    assert got != 30, "the fixture does not actually exercise undershoot"
    assert got == 0, (
        "expected the clip-only reference to emit 0; got %d" % got)
