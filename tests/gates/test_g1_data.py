"""STAGE 1 -- THE SLICE, before a single image is loaded or a GPU is touched.

Everything here runs on `train_meta.csv` / `test_meta.csv` and pure arithmetic.
That is the point: each gate traces to a failure this project paid for AFTER
spending GPU time, and every one of them was visible in the metadata.

    2(n)    `synth_group = np.arange(len(y)) % 3` made octmnist/tissuemnist
            i.i.d. draws from ONE distribution -- the local scope is empty BY
            CONSTRUCTION, NET -7 and -56 items -- and both ran for months.
    2(n)    a dataset famous for DOMAIN SHIFT is not one with PER-GROUP LABEL
            SHIFT. rxrx1: 1,139 classes, real batch effects, dead, because
            every siRNA appears in every experiment by design.
    2(x1)   `class_balanced` is INERT here. TRAIN is exactly 2500/class; the
            4.5x figure is the TEST set. audit_config and check_parity were
            green -- only hashing the raw predictions found it.
    2(n)    7 of 14 per-group ceilings are K=0, and a zero ceiling binds
            regardless of sum slack. That is what the LOCAL scope rests on.
    2       dermmnist shared `lesion_id` across the split -- 38.7% of test,
            67.3% of melanoma -- because `create_slices.py` pooled the
            corrected splits and re-cut them STRATIFIED ON THE LABEL.
    2(w2c)  the factorial gate COULD NOT FAIL on an atomic group: an atomic
            slice and a wrong `--sep` are indistinguishable, both read ~100%.
    2(w2)   stage 1 is NECESSARY ONLY -- dermmnist passed at z=2.9 and nulled.
"""
import os

import numpy as np
import pandas as pd
import pytest

from .conftest import CAPPED_CLASSES, report

pytestmark = pytest.mark.stage1_data

N_CLASSES = 8


def _write_slice(d, train_rows, test_rows):
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(train_rows).to_csv(os.path.join(d, "train_meta.csv"), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(d, "test_meta.csv"), index=False)
    return d


def _rows(loc, labels):
    return [{"location": loc, "label": int(c)} for c in labels]


def _index_period(groups):
    """The modulus m for which `row_index % m` DETERMINES the group, or None.

    An index-derived column with G distinct values has period exactly G, so G
    is the modulus that must be tried. Vectorised: reshape to (-1, m) and each
    residue class is a COLUMN, which must be constant.
    """
    codes = pd.factorize(np.asarray(groups))[0]
    n, G = len(codes), int(codes.max()) + 1
    for m in sorted({G} | set(range(2, min(G, 12) + 1))):
        if m < 2 or n < 3 * m:
            continue
        pad = (-n) % m
        arr = np.concatenate([codes, np.full(pad, -1, dtype=codes.dtype)])
        arr = arr.reshape(-1, m)
        rows = arr[:-1] if pad else arr
        if len(rows) >= 3 and bool((rows == rows[0]).all()):
            return m
    return None


def _one_distribution(rng, n_train_groups, n_test_groups, per=700):
    """Every group drawn from the SAME class distribution -- rxrx1's shape and
    octmnist's: real held-out groups, uniform per-group label mix."""
    base = rng.dirichlet(np.ones(N_CLASSES))
    tr, te = [], []
    for g in range(n_train_groups):
        tr += _rows(100 + g, rng.choice(N_CLASSES, size=per, p=base))
    for g in range(n_train_groups, n_train_groups + n_test_groups):
        te += _rows(100 + g, rng.choice(N_CLASSES, size=per // 2, p=base))
    return tr, te


# ----------------------------------------------------------------- the gates


def test_a_group_column_that_is_a_function_of_the_row_index_is_dead(slice_dir,
                                                                    tmp_path):
    """FRAMEWORK 2(n). TWO ARMS, and the division of labour is the point.

    The STRUCTURAL arm reads the column and catches `index % 3` outright,
    before any statistics. The STATISTICAL arm is `dataset_screen`'s NET --
    per-group deviation after rescaling by the GLOBAL shift, minus a simulated
    sampling-noise null -- and it catches the same construction SHUFFLED, which
    the structural arm cannot see. Neither alone is the gate.
    """
    from scripts.dataset_screen import screen

    rng = np.random.default_rng(0)
    base = rng.dirichlet(np.ones(N_CLASSES))
    tr_lab = rng.choice(N_CLASSES, size=6000, p=base)
    te_lab = rng.choice(N_CLASSES, size=1500, p=base)
    idx_tr = [{"location": i % 3, "label": int(c)} for i, c in enumerate(tr_lab)]
    idx_te = [{"location": i % 3, "label": int(c)} for i, c in enumerate(te_lab)]
    shuffled = rng.permutation(np.arange(len(te_lab)) % 3)
    shuf_te = [{"location": int(g), "label": int(c)}
               for g, c in zip(shuffled, te_lab)]
    cases = [
        ("iwildcam/oodslice, the shipped slice", slice_dir, None, True),
        ("synth_group = arange(n) % 3 (octmnist, tissuemnist)",
         _write_slice(str(tmp_path / "idx"), idx_tr, idx_te), 3, False),
        ("the same construction, groups SHUFFLED",
         _write_slice(str(tmp_path / "shuf"), idx_tr, shuf_te), None, False),
    ]
    fails = []
    for name, d, period, live in cases:
        te = pd.read_csv(os.path.join(d, "test_meta.csv"))
        got = _index_period(te["location"])
        if got != period:
            fails.append("%s: structural index-period %r, expected %r"
                         % (name, got, period))
        r = screen(d)
        if (r["net_z"] >= 2.0) != live:
            fails.append("%s: NET %+.0f items z=%.1f reads %s, expected %s"
                         % (name, r["net_items"], r["net_z"],
                            "LIVE" if r["net_z"] >= 2.0 else "DEAD",
                            "LIVE" if live else "DEAD"))
    report(fails, "index-derived group failures")


def test_unseen_test_groups_alone_do_not_make_a_slice_live(slice_dir, tmp_path):
    """FRAMEWORK 2(n): "a dataset famous for DOMAIN SHIFT is not automatically
    one with PER-GROUP LABEL SHIFT, and only the second is usable here."

    rxrx1's shape: whole experiments held out, every siRNA in every experiment
    BY DESIGN, so the per-group class mix is uniform. Criterion 1 (unseen
    groups) without criterion 2 (differential label shift) looks exactly like a
    live dataset until the NET is read. The control is built to satisfy 1 and
    fail 2, so a gate reading `unseen > 0` cannot separate it from the slice.
    """
    from scripts.dataset_screen import screen

    tr, te = _one_distribution(np.random.default_rng(7), 6, 4)
    cases = [("iwildcam/oodslice", slice_dir, True),
             ("rxrx1-shaped: unseen groups, one label distribution",
              _write_slice(str(tmp_path / "rxrx"), tr, te), False)]
    fails = []
    for name, d, live in cases:
        r = screen(d)
        if not r["unseen_groups"]:
            fails.append("%s: no unseen groups, so this pair cannot show that "
                         "`unseen > 0` is not the criterion" % name)
        if (r["net_z"] >= 2.0) != live:
            fails.append("%s: %d unseen groups but NET %+.0f at z=%.1f -- "
                         "expected %s" % (name, len(r["unseen_groups"]),
                                          r["net_items"], r["net_z"],
                                          "LIVE" if live else "DEAD"))
    report(fails, "unseen-groups-are-not-the-criterion failures")


def test_a_stage_one_pass_is_never_a_decision(slice_dir):
    """FRAMEWORK 2(w2): stage 1 is NECESSARY ONLY. dermmnist scored +65 items
    at z=2.9 and still nulled in 2(m), where feeding a model the TRUE per-group
    counts moved 6 items.

    And the measured defect in the ladder itself: `summarise` returns nan for z
    when the null spread is zero, `nan < 2.0` is False, so the DEAD branch was
    skipped and an ABSENT MEASUREMENT fell through to a pass. Gated both ways.
    """
    from scripts.dataset_screen import screen, verdict_lines

    r = screen(slice_dir)
    cases = [
        ("the shipped slice", r, ["STAGE 1 PASS", "necessary, not sufficient"], []),
        ("z below 2 (octmnist)", dict(r, net_z=1.0), ["DEAD"], ["PASS"]),
        ("z undefined (no null spread)", dict(r, net_z=float("nan")),
         ["UNDECIDABLE", "not a pass"], ["PASS"]),
        ("no group column at all", dict(r, gcol=None), ["NO GROUP COLUMN"],
         ["PASS"]),
    ]
    fails = []
    for name, res, must, must_not in cases:
        text = "\n".join(verdict_lines(res, "slice"))
        for s in must:
            if s not in text:
                fails.append("%s: verdict omits %r -- got %r" % (name, s, text))
        for s in must_not:
            if s in text:
                fails.append("%s: verdict claims %r -- got %r" % (name, s, text))
    report(fails, "stage-1 verdict-ladder failures")


def test_every_baseline_that_reads_the_training_prior_is_inert_here(slice_dir):
    """FRAMEWORK 2(x1). `cb_lp` carries a DIFFERENT `base_model_id`, genuinely
    retrained, and landed on byte-identical raw predictions in 24/24. The cause
    is arithmetic, not code: TRAIN is exactly 2500/class, so the class-balanced
    weight is exactly 1.0 for every class and weighted CE IS plain CE.

    Run against the SHIPPED criteria in `src/losses/imbalanced_losses.py`, not
    a re-derivation. Negative control: the same criteria on a 10x-imbalanced
    prior, where both must read LIVE.
    """
    import torch
    from src.losses.imbalanced_losses import (class_balanced_criterion,
                                              logit_adjusted_criterion)

    def probe(counts):
        y = torch.as_tensor(np.repeat(np.arange(N_CLASSES), counts))
        w = class_balanced_criterion(y, N_CLASSES, "cpu", beta=0.9999).weight
        prior = logit_adjusted_criterion(y, N_CLASSES, "cpu", tau=1.0).log_prior
        return float((w - 1.0).abs().max()), float(prior.max() - prior.min())

    tr = pd.read_csv(os.path.join(slice_dir, "train_meta.csv"))
    te = pd.read_csv(os.path.join(slice_dir, "test_meta.csv"))
    real = np.bincount(tr["label"].values, minlength=N_CLASSES)
    cb_real, la_real = probe(real)
    cb_skew, la_skew = probe(np.array([2500] * 7 + [250]))
    tc = np.bincount(te["label"].values, minlength=N_CLASSES)

    fails = []
    if real.max() != real.min() or real.min() != 2500:
        fails.append("train bincount %s is not 2500/class -- 2(x1)'s whole "
                     "argument is this arithmetic" % real.tolist())
    if cb_real != 0.0:
        fails.append("class_balanced max|w-1| = %.3e; 2(x1) measured exactly "
                     "0.0, i.e. INERT" % cb_real)
    if la_real != 0.0:
        fails.append("logit_adjust offset spans %.3e across classes; a CONSTANT "
                     "added to every logit is softmax-invariant" % la_real)
    if cb_skew <= 0.1 or la_skew <= 0.1:
        fails.append("NEGATIVE CONTROL DEAD: on a 10x prior the recipes read "
                     "cb=%.3e la=%.3e, so this gate cannot tell inert from live"
                     % (cb_skew, la_skew))
    if abs(tc.max() / tc.min() - 4.5) > 0.05:
        fails.append("test imbalance is %.2fx, not 4.5x -- and 4.5x is the TEST "
                     "set, never the train prior" % (tc.max() / tc.min()))
    report(fails, "training-prior baseline failures")


def test_the_local_scope_binds_because_half_its_ceilings_are_zero(slice_dir,
                                                                  tmp_path):
    """FRAMEWORK 2(n), 2(w1). 7 of 14 per-group ceilings are K=0 -- "predict
    none of this species at this camera" -- and a zero ceiling binds regardless
    of sum slack. On dermmnist `lp_fallback_used` was False with 0 candidates
    on all 52 runs: the local scope never bound the output.

    Two negative controls, for the two ways the local scope goes empty: every
    group holding every capped class (CCT class 5, "local adds nothing"), and
    the two capped classes sharing ONE support, which makes their budgets one
    number divided up. Budgets come from `compute_local_constraints` itself.
    """
    from src.training.constraints import compute_local_constraints

    rng = np.random.default_rng(3)
    every, shared = [], []
    for g in range(4):
        every += _rows(g, rng.integers(0, N_CLASSES, size=400))
        shared += _rows(g, list(CAPPED_CLASSES) * 40 if g < 2 else [0, 1, 3] * 40)
    cases = [("iwildcam/oodslice",
              pd.read_csv(os.path.join(slice_dir, "test_meta.csv")), 7, False),
             ("every group holds every class", pd.DataFrame(every), 0, True),
             ("both capped classes on one support", pd.DataFrame(shared), 4, True)]
    fails = []
    for name, frame, want_zero, want_shared in cases:
        loc = compute_local_constraints(frame, "label", 0.80, "location",
                                        list(CAPPED_CLASSES), N_CLASSES)
        zeros = sum(1 for v in loc.values() for c in CAPPED_CLASSES if v[c] == 0)
        if zeros != want_zero:
            fails.append("%s: %d of %d ceilings are K=0, expected %d"
                         % (name, zeros, 2 * len(loc), want_zero))
        sup = [tuple(sorted(g for g, v in loc.items() if v[c] > 0))
               for c in CAPPED_CLASSES]
        if (sup[0] == sup[1]) != want_shared:
            fails.append("%s: supports for classes %s are %s; one shared support "
                         "makes the two local budgets one number divided up"
                         % (name, list(CAPPED_CLASSES), sup))
    report(fails, "local-ceiling failures")


def test_the_test_cameras_are_held_out_entire(slice_dir, tmp_path, protocol):
    """FRAMEWORK 2(n) criterion 1, the one no earlier dataset satisfied: the
    model holds NO prior for a group it never saw, so the cap is the only
    source. dermmnist, octmnist and tissuemnist all had unseen = 0.

    The guard is `data_loader._check_group_leakage`, and it RAISES only when
    the dataset declares `disjoint_groups: true` -- so the declaration is part
    of the gate. Both directions: one shared camera must raise, and the same
    slice without the declaration must only warn.
    """
    from src.utils.data_loader import _check_group_leakage

    tr = pd.read_csv(os.path.join(slice_dir, "train_meta.csv"))
    te = pd.read_csv(os.path.join(slice_dir, "test_meta.csv"))
    fails = []
    shared = set(tr["location"]) & set(te["location"])
    if shared:
        fails.append("%d camera(s) in BOTH splits: %s"
                     % (len(shared), sorted(shared)[:5]))
    if not protocol["datasets"]["iwildcam"].get("disjoint_groups"):
        fails.append("protocol.yml does not declare `disjoint_groups: true`, so "
                     "the loader would WARN on overlap instead of raising")
    try:
        _check_group_leakage(slice_dir, "location", True)
    except ValueError as e:
        fails.append("the shipped slice fails its own guard: %s" % e)

    poisoned = te.copy()
    poisoned.loc[poisoned.index[:300], "location"] = int(tr["location"].iloc[0])
    leaky = _write_slice(str(tmp_path / "leaky"), tr.to_dict("records"),
                         poisoned.to_dict("records"))
    try:
        _check_group_leakage(leaky, "location", True)
        fails.append("NEGATIVE CONTROL DEAD: one shared camera did not raise")
    except ValueError:
        pass
    try:
        _check_group_leakage(leaky, "location", False)
    except ValueError:
        fails.append("the guard raised WITHOUT `disjoint_groups`, so the "
                     "declaration is not what makes it fatal")
    report(fails, "held-out-group failures")


def test_no_instance_identifier_crosses_the_split(slice_dir, tmp_path):
    """The dermmnist leak, FRAMEWORK 2: HAM10000 photographs a lesion several
    times, `create_slices.py` re-cut the corrected splits stratified on the
    LABEL, and 38.7% of test -- 67.3% of melanoma -- shared a `lesion_id` with
    a training image. Every absolute derm number died with it.

    Generic check: any column in both splits whose values are near-unique
    WITHIN each split is an instance identifier and must not overlap. It also
    refuses to pass vacuously -- no identifier column means the gate measured
    nothing, which is the 2(w2c) defect class rather than a pass.
    """
    def identifiers(a, b):
        return {c: float(b[c].isin(set(a[c])).mean())
                for c in set(a.columns) & set(b.columns)
                if min(a[c].nunique() / len(a), b[c].nunique() / len(b)) >= 0.9}

    tr = pd.read_csv(os.path.join(slice_dir, "train_meta.csv"))
    te = pd.read_csv(os.path.join(slice_dir, "test_meta.csv"))
    fails = []
    found = identifiers(tr, te)
    if not found:
        fails.append("no per-instance identifier column found, so this gate "
                     "measured NOTHING -- it cannot report a pass")
    for col, share in sorted(found.items()):
        if share > 0.0:
            fails.append("%.1f%% of test rows share a %r with a train row"
                         % (100 * share, col))
    leaked = te.copy()
    n = int(round(0.387 * len(leaked)))
    leaked.loc[leaked.index[:n], "filename"] = tr["filename"].values[:n]
    got = identifiers(tr, leaked).get("filename")
    if got is None or abs(got - 0.387) > 0.005:
        fails.append("NEGATIVE CONTROL DEAD: a slice built with dermmnist's "
                     "38.7%% overlap reads %r" % got)
    report(fails, "instance-identifier leakage failures")


def test_the_split_was_cut_by_group_not_stratified_on_the_label(slice_dir,
                                                                tmp_path):
    """FRAMEWORK 2(n)'s closing warning: "`create_slices.py` stratifies on the
    LABEL, which forces test prevalence to match train prevalence... A
    WILDS-style dataset run through a label-stratified splitter would reproduce
    every null. Split BY GROUP, holding groups out."

    The splitter leaves a signature in the labels alone: under stratification
    the test counts reproduce the training prevalence to within sampling noise,
    so `dataset_screen`'s GLOBAL excess collapses. The control is the real
    `StratifiedShuffleSplit` over this slice's own pooled rows, so the contrast
    is the SPLITTER and nothing else. NOT a claim that a global shift is
    exploitable -- 2(j) proved top-K is invariant to a per-class multiplier.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    from scripts.dataset_screen import screen

    tr = pd.read_csv(os.path.join(slice_dir, "train_meta.csv"))
    te = pd.read_csv(os.path.join(slice_dir, "test_meta.csv"))
    pooled = pd.concat([tr, te], ignore_index=True)
    a, b = next(StratifiedShuffleSplit(n_splits=1, test_size=len(te),
                                       random_state=43)
                .split(np.zeros(len(pooled)), pooled["label"].values))
    cases = [("iwildcam/oodslice, cut BY CAMERA", slice_dir, True),
             ("the same rows, StratifiedShuffleSplit on the label",
              _write_slice(str(tmp_path / "strat"), pooled.iloc[a].to_dict("records"),
                           pooled.iloc[b].to_dict("records")), False)]
    fails = []
    for name, d, live in cases:
        r = screen(d)
        if (r["global_z"] >= 2.0) != live:
            fails.append("%s: test prevalence departs from train by %+.0f items "
                         "at z=%.1f -- expected %s"
                         % (name, r["global_items"], r["global_z"],
                            "a real shift" if live else "sampling noise only"))
    report(fails, "splitter-signature failures")


def test_the_factorial_gate_is_not_a_pass_on_an_atomic_group(slice_dir,
                                                             tmp_path):
    """FRAMEWORK 2(w2c). `factorial_control` split `location` on `--sep` and
    read tokens [0] and [-1]; when the separator is ABSENT both are the whole
    string, every unseen group kept the global prior, the additive arm WAS the
    global arm, and `survives` came out ~100% BY ARITHMETIC. It printed
    "iwildcam 100.1%" as a pass -- and that row was the gate's own claimed
    positive control. 8 of 21 candidates rake ZERO groups.

    Three cases: the shipped slice must REFUSE (a camera is atomic), a
    factorial slice must MEASURE, and the SAME slice under a wrong separator
    must refuse -- the pair showing the refusal is not unconditional.
    """
    from scripts.factorial_control import _synthetic, control, report as fc_say

    fact = _synthetic(str(tmp_path / "fact"), sep="|", seed=0)
    cases = [("iwildcam/oodslice, camera = ATOMIC", slice_dir, "|", False),
             ("site|age, raking is exactly right", fact, "|", True),
             ("the same slice, WRONG separator", fact, "@", False)]
    fails = []
    for name, d, sep, measures in cases:
        r = control(d, sep=sep)
        text = "\n".join(fc_say(r, name))
        if measures:
            if r["raked"] != r["unseen"] or not np.isfinite(r["survives"]):
                fails.append("%s: raked %d of %d unseen, survives=%r -- the "
                             "control did not run"
                             % (name, r["raked"], r["unseen"], r["survives"]))
            elif r["survives"] >= 100.0:
                fails.append("%s: survives %.1f%%, so raking absorbed none of "
                             "the novelty it was built to absorb"
                             % (name, r["survives"]))
            if "NOT A CONTROL" in text:
                fails.append("%s: refused a slice it did in fact rake" % name)
            continue
        if r["raked"] != 0 or np.isfinite(r["survives"]):
            fails.append("%s: raked %d groups and printed survives=%r; an atomic "
                         "group has no survival number"
                         % (name, r["raked"], r["survives"]))
        if r["unseen"] == 0 or r["no_sep"] != r["unseen"]:
            # A refusal with nothing to rake is not a measurement of atomicity.
            fails.append("%s: refused with unseen=%d no_sep=%d -- the refusal "
                         "must be BECAUSE the separator is absent from every "
                         "unseen group label" % (name, r["unseen"], r["no_sep"]))
        if "NOT A CONTROL" not in text:
            fails.append("%s: reported a percentage it never measured:\n%s"
                         % (name, text))
    report(fails, "factorial-gate failures")
