"""Maintained behavioral regression fixtures."""

import io

import os

import numpy as np

import pandas as pd

import pytest

from .conftest import CAPPED_CLASSES, rel, report

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
    codes = pd.factorize(np.asarray(groups))[0]
    (n, G) = (len(codes), int(codes.max()) + 1)
    for m in sorted({G} | set(range(2, min(G, 12) + 1))):
        if m < 2 or n < 3 * m:
            continue
        pad = -n % m
        arr = np.concatenate([codes, np.full(pad, -1, dtype=codes.dtype)])
        arr = arr.reshape(-1, m)
        rows = arr[:-1] if pad else arr
        if len(rows) >= 3 and bool((rows == rows[0]).all()):
            return m
    return None


def _one_distribution(rng, n_train_groups, n_test_groups, per=700):
    base = rng.dirichlet(np.ones(N_CLASSES))
    (tr, te) = ([], [])
    for g in range(n_train_groups):
        tr += _rows(100 + g, rng.choice(N_CLASSES, size=per, p=base))
    for g in range(n_train_groups, n_train_groups + n_test_groups):
        te += _rows(100 + g, rng.choice(N_CLASSES, size=per // 2, p=base))
    return (tr, te)


def test_a_group_column_that_is_a_function_of_the_row_index_is_dead(
    slice_dir, tmp_path
):
    from scripts.dataset_screen import screen

    rng = np.random.default_rng(0)
    base = rng.dirichlet(np.ones(N_CLASSES))
    tr_lab = rng.choice(N_CLASSES, size=6000, p=base)
    te_lab = rng.choice(N_CLASSES, size=1500, p=base)
    idx_tr = [{"location": i % 3, "label": int(c)} for (i, c) in enumerate(tr_lab)]
    idx_te = [{"location": i % 3, "label": int(c)} for (i, c) in enumerate(te_lab)]
    shuffled = rng.permutation(np.arange(len(te_lab)) % 3)
    shuf_te = [
        {"location": int(g), "label": int(c)} for (g, c) in zip(shuffled, te_lab)
    ]
    cases = [
        ("fmow2/oodslice, the shipped slice", slice_dir, None, True),
        (
            "synth_group = arange(n) % 3 (octmnist, tissuemnist)",
            _write_slice(str(tmp_path / "idx"), idx_tr, idx_te),
            3,
            False,
        ),
        (
            "the same construction, groups SHUFFLED",
            _write_slice(str(tmp_path / "shuf"), idx_tr, shuf_te),
            None,
            False,
        ),
    ]
    fails = []
    for name, d, period, live in cases:
        te = pd.read_csv(os.path.join(d, "test_meta.csv"))
        got = _index_period(te["location"])
        if got != period:
            fails.append(
                "%s: structural index-period %r, expected %r" % (name, got, period)
            )
        r = screen(d)
        if (r["net_z"] >= 2.0) != live:
            fails.append(
                "%s: NET %+.0f items z=%.1f reads %s, expected %s"
                % (
                    name,
                    r["net_items"],
                    r["net_z"],
                    "LIVE" if r["net_z"] >= 2.0 else "DEAD",
                    "LIVE" if live else "DEAD",
                )
            )
    report(fails, "index-derived group failures")


def test_unseen_test_groups_alone_do_not_make_a_slice_live(slice_dir, tmp_path):
    from scripts.dataset_screen import screen

    (tr, te) = _one_distribution(np.random.default_rng(7), 6, 4)
    cases = [
        ("fmow2/oodslice", slice_dir, True),
        (
            "rxrx1-shaped: unseen groups, one label distribution",
            _write_slice(str(tmp_path / "rxrx"), tr, te),
            False,
        ),
    ]
    fails = []
    for name, d, live in cases:
        r = screen(d)
        if not r["unseen_groups"]:
            fails.append(
                "%s: no unseen groups, so this pair cannot show that `unseen > 0` is not the criterion"
                % name
            )
        if (r["net_z"] >= 2.0) != live:
            fails.append(
                "%s: %d unseen groups but NET %+.0f at z=%.1f -- expected %s"
                % (
                    name,
                    len(r["unseen_groups"]),
                    r["net_items"],
                    r["net_z"],
                    "LIVE" if live else "DEAD",
                )
            )
    report(fails, "unseen-groups-are-not-the-criterion failures")


def test_distribution_diagnostics_do_not_decide_research_viability(slice_dir):
    from scripts.dataset_screen import screen, diagnostic_lines

    r = screen(slice_dir)
    for res in (r, dict(r, net_z=1.0), dict(r, net_z=float("nan")), dict(r, gcol=None)):
        text = "\n".join(diagnostic_lines(res, "slice"))
        assert not any(word in text for word in ("PASS", "DEAD", "MARGINAL"))
        if res["gcol"] is None:
            assert "NO GROUP COLUMN" in text
        else:
            assert "NET excess %+.2f items" % res["net_items"] in text
            assert ("undefined" in text) == (not np.isfinite(res["net_z"]))


def test_the_local_scope_binds_because_half_its_ceilings_are_zero(slice_dir, tmp_path, protocol):
    from src.training.constraints import compute_local_constraints

    rng = np.random.default_rng(3)
    (every, shared) = ([], [])
    for g in range(4):
        every += _rows(g, rng.integers(0, N_CLASSES, size=400))
        shared += _rows(g, list(CAPPED_CLASSES) * 40 if g < 2 else [0, 1, 3] * 40)
    # CAPPED_CLASSES is (2, 7) -- impala and cattle, an iWildCam-era constant.
    # fmow2 replaced iwildcam and its protocol caps THREE classes, [1, 2, 7], so
    # this case was scoring the real slice against a class set nobody runs and
    # an expectation (7 zeros) belonging to neither. The real-slice case now
    # takes its classes from the protocol; the synthetic cases stay on the
    # two-class constant, which is what they are constructed around.
    fmow2_classes = protocol["datasets"]["fmow2"]["constrained_class"]
    cases = [
        (
            "fmow2/oodslice",
            pd.read_csv(os.path.join(slice_dir, "test_meta.csv")),
            fmow2_classes,
            3,
            False,
        ),
        ("every group holds every class", pd.DataFrame(every), list(CAPPED_CLASSES), 0, True),
        ("both capped classes on one support", pd.DataFrame(shared), list(CAPPED_CLASSES), 4, True),
    ]
    fails = []
    for name, frame, classes, want_zero, want_shared in cases:
        loc = compute_local_constraints(
            frame, "label", 0.8, "location", classes, N_CLASSES
        )
        zeros = sum((1 for v in loc.values() for c in classes if v[c] == 0))
        if zeros != want_zero:
            fails.append(
                "%s: %d of %d ceilings are K=0, expected %d"
                % (name, zeros, len(classes) * len(loc), want_zero)
            )
        sup = [
            tuple(sorted((g for (g, v) in loc.items() if v[c] > 0)))
            for c in classes
        ]
        if (len(set(sup)) == 1) != want_shared:
            fails.append(
                "%s: supports for classes %s are %s; one shared support makes the local budgets one number divided up"
                % (name, classes, sup)
            )
    report(fails, "local-ceiling failures")


def test_the_test_cameras_are_held_out_entire(slice_dir, tmp_path, protocol):
    from src.utils.data_loader import _check_group_leakage

    tr = pd.read_csv(os.path.join(slice_dir, "train_meta.csv"))
    te = pd.read_csv(os.path.join(slice_dir, "test_meta.csv"))
    fails = []
    shared = set(tr["location"]) & set(te["location"])
    if shared:
        fails.append(
            "%d camera(s) in BOTH splits: %s" % (len(shared), sorted(shared)[:5])
        )
    if not protocol["datasets"]["fmow2"].get("disjoint_groups"):
        fails.append(
            "protocol.yml does not declare `disjoint_groups: true`, so the loader would WARN on overlap instead of raising"
        )
    try:
        _check_group_leakage(slice_dir, "location", True)
    except ValueError as e:
        fails.append("the shipped slice fails its own guard: %s" % e)
    poisoned = te.copy()
    # NOT int(): iwildcam group ids were integer camera numbers, fmow2's are
    # country codes like 'AUS' and int() raises before the negative control can
    # run. The raw value is a real train group in either dataset, which is the
    # only property this poisoning needs.
    poisoned.loc[poisoned.index[:300], "location"] = tr["location"].iloc[0]
    leaky = _write_slice(
        str(tmp_path / "leaky"), tr.to_dict("records"), poisoned.to_dict("records")
    )
    try:
        _check_group_leakage(leaky, "location", True)
        fails.append("NEGATIVE CONTROL DEAD: one shared camera did not raise")
    except ValueError:
        pass
    try:
        _check_group_leakage(leaky, "location", False)
    except ValueError:
        fails.append(
            "the guard raised WITHOUT `disjoint_groups`, so the declaration is not what makes it fatal"
        )
    report(fails, "held-out-group failures")


def test_no_instance_identifier_crosses_the_split(slice_dir, tmp_path):

    def identifiers(a, b):
        return {
            c: float(b[c].isin(set(a[c])).mean())
            for c in set(a.columns) & set(b.columns)
            if min(a[c].nunique() / len(a), b[c].nunique() / len(b)) >= 0.9
        }

    tr = pd.read_csv(os.path.join(slice_dir, "train_meta.csv"))
    te = pd.read_csv(os.path.join(slice_dir, "test_meta.csv"))
    fails = []
    found = identifiers(tr, te)
    if not found:
        fails.append(
            "no per-instance identifier column found, so this gate measured NOTHING -- it cannot report a pass"
        )
    for col, share in sorted(found.items()):
        if share > 0.0:
            fails.append(
                "%.1f%% of test rows share a %r with a train row" % (100 * share, col)
            )
    leaked = te.copy()
    n = int(round(0.387 * len(leaked)))
    leaked.loc[leaked.index[:n], "filename"] = tr["filename"].values[:n]
    got = identifiers(tr, leaked).get("filename")
    if got is None or abs(got - 0.387) > 0.005:
        fails.append(
            "NEGATIVE CONTROL DEAD: a slice built with dermmnist's 38.7%% overlap reads %r"
            % got
        )
    report(fails, "instance-identifier leakage failures")


def test_the_split_was_cut_by_group_not_stratified_on_the_label(slice_dir, tmp_path):
    from sklearn.model_selection import StratifiedShuffleSplit
    from scripts.dataset_screen import screen

    tr = pd.read_csv(os.path.join(slice_dir, "train_meta.csv"))
    te = pd.read_csv(os.path.join(slice_dir, "test_meta.csv"))
    pooled = pd.concat([tr, te], ignore_index=True)
    (a, b) = next(
        StratifiedShuffleSplit(n_splits=1, test_size=len(te), random_state=43).split(
            np.zeros(len(pooled)), pooled["label"].values
        )
    )
    cases = [
        ("fmow2/oodslice, cut BY CAMERA", slice_dir, True),
        (
            "the same rows, StratifiedShuffleSplit on the label",
            _write_slice(
                str(tmp_path / "strat"),
                pooled.iloc[a].to_dict("records"),
                pooled.iloc[b].to_dict("records"),
            ),
            False,
        ),
    ]
    fails = []
    for name, d, live in cases:
        r = screen(d)
        if (r["global_z"] >= 2.0) != live:
            fails.append(
                "%s: test prevalence departs from train by %+.0f items at z=%.1f -- expected %s"
                % (
                    name,
                    r["global_items"],
                    r["global_z"],
                    "a real shift" if live else "sampling noise only",
                )
            )
    report(fails, "splitter-signature failures")


def test_the_factorial_gate_is_not_a_pass_on_an_atomic_group(slice_dir, tmp_path):
    from scripts.factorial_control import _synthetic, control, report as fc_say

    fact = _synthetic(str(tmp_path / "fact"), sep="|", seed=0)
    cases = [
        ("fmow2/oodslice, camera = ATOMIC", slice_dir, "|", False),
        ("site|age, raking is exactly right", fact, "|", True),
        ("the same slice, WRONG separator", fact, "@", False),
    ]
    fails = []
    for name, d, sep, measures in cases:
        r = control(d, sep=sep)
        text = "\n".join(fc_say(r, name))
        if measures:
            if r["raked"] != r["unseen"] or not np.isfinite(r["survives"]):
                fails.append(
                    "%s: raked %d of %d unseen, survives=%r -- the control did not run"
                    % (name, r["raked"], r["unseen"], r["survives"])
                )
            elif r["survives"] >= 100.0:
                fails.append(
                    "%s: survives %.1f%%, so raking absorbed none of the novelty it was built to absorb"
                    % (name, r["survives"])
                )
            if "NOT A CONTROL" in text:
                fails.append("%s: refused a slice it did in fact rake" % name)
            continue
        if r["raked"] != 0 or np.isfinite(r["survives"]):
            fails.append(
                "%s: raked %d groups and printed survives=%r; an atomic group has no survival number"
                % (name, r["raked"], r["survives"])
            )
        if r["unseen"] == 0 or r["no_sep"] != r["unseen"]:
            fails.append(
                "%s: refused with unseen=%d no_sep=%d -- the refusal must be BECAUSE the separator is absent from every unseen group label"
                % (name, r["unseen"], r["no_sep"])
            )
        if "NOT A CONTROL" not in text:
            fails.append(
                "%s: reported a percentage it never measured:\n%s" % (name, text)
            )
    report(fails, "factorial-gate failures")


def test_every_registered_dataset_can_actually_encode_its_group_column():
    import yaml
    from src.utils.data_loader import _encode_groups, IMAGERY_DATASETS
    from configs.gen_campaign import PROTOCOL_PATH

    fails = []
    got = _encode_groups(pd.Series(["b|2", "a|1", "b|2"]), "location")
    if got.dtype != np.int64 or len(set(got.tolist())) != 2:
        fails.append("a string group column did not encode: %r" % (got,))
    col = pd.Series(["z", "a", "m", "a"])
    fwd = _encode_groups(col, "g")
    rev = _encode_groups(col.iloc[::-1], "g")[::-1]
    if not (fwd == rev).all():
        fails.append("encoding depends on ROW ORDER: %r vs %r" % (fwd, rev))
    ids = _encode_groups(pd.Series([218, 320, 218, 516]), "location")
    if ids.tolist() != [218, 320, 218, 516]:
        fails.append(
            "an integer group column was RENUMBERED to %r -- this would silently change every fmow2 group id"
            % (ids.tolist(),)
        )
    try:
        _encode_groups(pd.Series(["a", None]), "location")
        fails.append("a null group value did not raise")
    except Exception:
        pass
    with io.open(PROTOCOL_PATH, encoding="utf-8") as fh:
        datasets = yaml.safe_load(fh)["datasets"]
    checked = 0
    for name, dc in sorted(datasets.items()):
        if name not in IMAGERY_DATASETS:
            fails.append(
                "%s is declared in protocol.yml but absent from IMAGERY_DATASETS, so the loader would refuse it"
                % name
            )
            continue
        meta = rel(dc["data_dir"], "test_meta.csv")
        if not os.path.exists(meta):
            continue
        tm = pd.read_csv(meta)
        gcol = dc["group_column"]
        if gcol not in tm.columns:
            fails.append(
                "%s: group_column %r is not in test_meta.csv (%s)"
                % (name, gcol, list(tm.columns))
            )
            continue
        try:
            g = _encode_groups(tm[gcol], gcol)
        except Exception as exc:
            fails.append(
                "%s: group column %r does not encode: %s: %s"
                % (name, gcol, type(exc).__name__, exc)
            )
            continue
        checked += 1
        if g.dtype != np.int64:
            fails.append("%s: group ids are %s, not int64" % (name, g.dtype))
        if len(g) != len(tm):
            fails.append("%s: encoded %d ids for %d rows" % (name, len(g), len(tm)))
        if len(set(g.tolist())) != tm[gcol].nunique():
            fails.append(
                "%s: %d distinct group labels collapsed to %d ids"
                % (name, tm[gcol].nunique(), len(set(g.tolist())))
            )
    report(fails, "group-encoding failures (%d slice(s) present)" % checked)
