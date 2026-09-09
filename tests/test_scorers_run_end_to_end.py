"""EVERY SCORER MUST ACTUALLY RUN. The AST gates cannot see a crash.

🛑 THE DEFECT THIS EXISTS FOR (2026-09-04). Wiring the dead-arm filter into the
scorers introduced `quarantine.drop_dead_runs(...)` into `full_panel.py`,
`cell_table.py` and `deployed_h2h.py` with **no module-level import of
`quarantine`** in any of them. THE scorer, THE survey and THE arm-vs-arm tool
were unrunnable on every input, and:

  * every module still PARSED and IMPORTED;
  * `python -m pytest tests -q` was green;
  * `preflight` was green;
  * `tests/gates/test_g6_results.py`'s AST checker read all three as fully
    gated -- it inspects the source, and a `NameError` is not in the source;
  * the crash fires only on the branch a partially-quarantined campaign
    reaches, i.e. the branch that exists to PREVENT a wrong number.

A gate that fires only when the guard is needed is the worst failure mode
available. So this file does the one thing no other test did: it EXECUTES each
scorer, as a subprocess, against a real campaign tree carrying a real PARTIAL
quarantine marker, and demands two things -- it must not crash, and it must not
name a dead arm.

The fixture is deliberately minimal. It does not have to be SCORABLE: a scorer
that reports "no scorable runs" has still walked its enumeration, which is
exactly where all three NameErrors lived. Requiring a fully scorable fixture
would make this test expensive and fragile for no extra coverage.
"""

import io
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NL = chr(10)

# The campaign is named `dom1` on purpose: it is PARTIAL in the real registry
# (`scorable=True` + dead arms `fioretto`, `hounie`), so the gate takes the
# branch that the crash was hiding on. A made-up name would exercise nothing.
CAMPAIGN = "dom1"
DEAD = ("fioretto", "hounie")
LIVE = ("clip", "tralo", "tralo_null", "tralo_reseed")
CAPS = ("L80-80_G95", "L90-90_G95")
SEEDS = (1, 2, 3, 4)

# Signs of a crash, as opposed to a legitimate refusal. A scorer is allowed to
# exit non-zero -- "nothing here is scorable" IS a valid answer and several
# tools signal it that way -- so the exit code cannot be the test.
CRASH = ("Traceback (most recent call last)", "NameError", "AttributeError",
         "ImportError", "ModuleNotFoundError", "UnboundLocalError")


def _run_dir(root, cap, arm, seed):
    return os.path.join(root, "ViTB16", "iwildcam", cap, arm, "seed_%d" % seed)


def _write_run(root, cap, arm, seed):
    d = _run_dir(root, cap, arm, seed)
    os.makedirs(d, exist_ok=True)
    cfg = {
        "status": "completed",
        "model_name": "ViTB16",
        "dataset_name": "iwildcam",
        "methodology": arm,
        "code_version": "0" * 12,
        "cap_tag": cap,
        # `constraint` is [local, global] and a real config always carries it;
        # `sensitivity_screen.local_fraction` RAISES rather than guessing a
        # cap fraction, which is the right refusal. The fixture has to look
        # like a real run, not like the minimum each tool tolerates.
        "constraint": [0.80 if cap.startswith("L80") else 0.90, 0.95],
        "dataset_config": {
            "data_dir": "data/iwildcam/oodslice",
            "num_classes": 8,
            "group_column": "location",
            "constrained_class": [2, 7],
            "disjoint_groups": True,
        },
        "hyperparams": {
            # THE SEED LIVES IN `hyperparams`, and ONLY there:
            # `gen_campaign.py:130` writes `hp["seed"] = seed` and
            # nothing writes a top-level one, while `full_panel.panel`
            # reads `cfg["hyperparams"]["seed"]`. This fixture put it at
            # top level, so every scorer here saw `seed: None` -- and
            # `cell_table` crashed on `sorted()` over Nones the moment
            # the group column was also corrected and it got that far.
            "seed": seed,
            "warmup_epochs": 1,
            "constraint_epochs": 0 if arm in ("clip", "focal_clip") else 29,
            "constraint_fp32": True,
            "constraint_grad_mode": "normalize",
        },
    }
    with io.open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Two prediction files, so the enumeration finds something to open. The
    # numbers are arbitrary; nothing here asserts on a metric.
    # Group_ID, NOT `Group`: `src/training/logging.py` writes `Group_ID` and
    # every production reader looks for that name. This fixture said `Group`,
    # so every group-aware scorer took its no-group FALLBACK branch here --
    # in the one test that exists because three scorers were unrunnable on
    # every real input. Found 2026-09-09 when `paired_noise` began REFUSING a
    # file with no group column instead of silently sorting globally.
    head = ("True_Label,Predicted_Label,Group_ID,"
            + ",".join("Prob_Class_%d" % c for c in range(8)))
    rows = [head]
    for i in range(24):
        y = 2 if i % 3 == 0 else (7 if i % 3 == 1 else 0)
        pred = y if (i + seed) % 4 else 0
        probs = [0.01] * 8
        probs[y] = 0.93
        rows.append("%d,%d,%d,%s" % (y, pred, i % 4,
                                     ",".join("%.4f" % p for p in probs)))
    body = NL.join(rows) + NL
    for name in ("final_predictions.csv", "final_predictions_raw.csv"):
        with io.open(os.path.join(d, name), "w", encoding="utf-8") as f:
            f.write(body)


@pytest.fixture(scope="module")
def partial_campaign(tmp_path_factory):
    """A campaign tree whose NAME carries a real PARTIAL marker."""
    root = str(tmp_path_factory.mktemp("results") / CAMPAIGN)
    for cap in CAPS:
        for arm in LIVE + DEAD:
            for seed in SEEDS:
                _write_run(root, cap, arm, seed)
    return root


def _invoke(argv, cwd=REPO):
    r = subprocess.run([sys.executable] + argv, cwd=cwd,
                       capture_output=True, text=True, timeout=600)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def _argv(mod, root, out_dir):
    """Each scorer's own argument shape. Keep this table honest: a tool
    invoked with the wrong flags errors in argparse and never reaches its
    enumeration, which would make this test pass while checking nothing."""
    return {
        "full_panel": ["-m", "scripts.full_panel", "--campaign", root,
                       "--control", "clip"],
        "cell_table": ["-m", "scripts.cell_table", "--campaign", root,
                       "--out", os.path.join(out_dir, "cells.csv")],
        "deployed_h2h": ["-m", "scripts.deployed_h2h", "--campaign", root,
                         "--control", "clip"],
        "score_scan": ["-m", "scripts.score_scan", root],
        "sensitivity_screen": ["-m", "scripts.sensitivity_screen",
                               "--campaign", root],
        "paired_noise": ["-m", "scripts.paired_noise", "--campaign", root],
    }[mod]


SCORERS = ("full_panel", "cell_table", "deployed_h2h", "score_scan",
           "sensitivity_screen", "paired_noise")


@pytest.mark.parametrize("mod", SCORERS)
def test_the_scorer_RUNS_on_a_partially_quarantined_campaign(
        mod, partial_campaign, tmp_path):
    """It may refuse. It may find nothing. It may NOT crash.

    This is the assertion that would have caught the three missing
    `quarantine` imports the moment they were written, and that every AST
    check in the repo was structurally unable to see.
    """
    rc, out = _invoke(_argv(mod, partial_campaign, str(tmp_path)))
    hit = [c for c in CRASH if c in out]
    assert not hit, (
        "%s CRASHED (%s) on a partially-quarantined campaign. Exit %d. This is "
        "the branch that exists to prevent a wrong number, so it is the worst "
        "one to be broken on." % (mod, ", ".join(hit), rc) + NL + out[-3000:])


@pytest.mark.parametrize("mod", SCORERS)
def test_the_scorer_does_not_NAME_a_dead_arm_in_its_results(
        mod, partial_campaign, tmp_path):
    """Announcing the dead arms is not dropping them.

    Six of seven scorers bound `gate()`'s dead-arm return and never used it,
    so they printed `!! PARTIAL QUARANTINE ... DEAD ARMS: fioretto, hounie`
    and then ranked `fioretto` #1 in the same cell.

    The banner legitimately names them, so the banner and the explanatory
    prose are stripped before looking. What remains is result text, and a dead
    arm has no business in it.
    """
    _rc, out = _invoke(_argv(mod, partial_campaign, str(tmp_path)))
    keep = []
    for line in out.split(NL):
        low = line.lower()
        if any(k in low for k in (
                "quarantin", "dead arm", "dropped", "not comparable",
                "unaffected", "different constraint dose", "keep for",
                "reason", "refus", "cannot read", "excluded rather")):
            continue
        keep.append(line)
    body = NL.join(keep)
    named = [a for a in DEAD if a in body]
    assert not named, (
        "%s printed %s outside the quarantine banner. Those arms ran at 28.00 "
        "attempted constraint steps against 29.00, so any contrast touching "
        "them is not comparable -- and a table that still lists them reads as "
        "complete." % (mod, ", ".join(named)) + NL + body[-3000:])


def test_NEGATIVE_CONTROL_this_file_detects_a_broken_scorer(
        partial_campaign, tmp_path):
    """Break a scorer for real and require the crash check to fire.

    Without this, the two tests above would also pass for a build in which
    every subprocess silently failed to start. The mutation is applied to a
    COPY of the tree so nothing shipped is touched even if the test aborts.
    """
    import shutil
    import tempfile

    work = tempfile.mkdtemp(prefix="scorer_ctl_")
    try:
        for sub in ("scripts", "src", "configs"):
            shutil.copytree(os.path.join(REPO, sub), os.path.join(work, sub))
        shutil.copy2(os.path.join(REPO, "main.py"), work)

        target = os.path.join(work, "scripts", "cell_table.py")
        src = io.open(target, encoding="utf-8").read()
        broken = src.replace("from scripts import quarantine", "", 1)
        assert broken != src, ("cell_table no longer imports quarantine at "
                               "module level, so this control cannot be built")
        io.open(target, "w", encoding="utf-8", newline="").write(broken)

        rc, out = _invoke(_argv("cell_table", partial_campaign, str(tmp_path)),
                          cwd=work)
        assert any(c in out for c in CRASH), (
            "the deliberately broken cell_table did NOT crash, so this file "
            "cannot detect the defect it exists for. Exit %d:" % rc
            + NL + out[-2000:])
    finally:
        shutil.rmtree(work, ignore_errors=True)


# `paired_noise` enforces by REFUSAL, not by filtering: every arm it reads is
# named on the command line, so there is no path list to filter. It is checked
# separately below rather than exempted silently.
FILTER_SHAPED = ("full_panel", "cell_table", "deployed_h2h", "score_scan",
                 "sensitivity_screen")


@pytest.mark.parametrize("mod", FILTER_SHAPED)
def test_the_scorer_REPORTS_dropping_the_dead_runs(mod, partial_campaign,
                                                   tmp_path):
    """The filter must FIRE and SAY SO, per run, with counts.

    ⚠️ This is the assertion that is not vacuous. The companion test above
    ("does not NAME a dead arm") passes for a scorer that refuses before it
    ever prints an arms table -- verified: deleting `full_panel`'s
    `drop_dead_runs` call left it green, because the fixture is not scorable
    and the run ends at a refusal either way. A drop that is REPORTED, with
    both arm names and a count, cannot be faked by exiting early.

    Reporting is required and not merely nice: a silent drop turns a
    disqualified contrast into a missing one, and a missing one reads as
    absence of evidence.
    """
    _rc, out = _invoke(_argv(mod, partial_campaign, str(tmp_path)))
    drop = [L for L in out.split(NL) if "DROPPED" in L]
    assert drop, (
        "%s never reported dropping anything, so the PARTIAL marker was "
        "announced and not enforced -- the exact defect six of seven scorers "
        "shipped with." % mod + NL + out[-2500:])
    joined = NL.join(drop)
    missing = [a for a in DEAD if a not in joined]
    assert not missing, (
        "%s dropped runs but did not name %s in the report. A count without "
        "the arm names cannot be audited." % (mod, ", ".join(missing))
        + NL + joined)


def test_paired_noise_REFUSES_a_dead_arm_by_name(partial_campaign):
    """The refusal shape, checked in both directions.

    `paired_noise` takes its four arms as arguments, so filtering would
    silently substitute a different comparison. It must refuse instead -- and
    it must NOT refuse when every named arm is live, or it is simply broken.
    """
    rc, out = _invoke(["-m", "scripts.paired_noise", "--campaign",
                       partial_campaign, "--treated", "fioretto"])
    assert rc == 1 and "fioretto" in out and "REFUS" in out.upper(), (
        "paired_noise accepted a DEAD arm as its treated arm; every noise "
        "figure it printed would be priced against a disqualified contrast."
        + NL + out[-2000:])

    rc2, out2 = _invoke(["-m", "scripts.paired_noise", "--campaign",
                         partial_campaign, "--treated", "tralo"])
    refused_live = ("is a DEAD arm" in out2)
    assert not refused_live, (
        "paired_noise refused a LIVE arm, so the refusal above proves nothing "
        "-- it refuses everything." + NL + out2[-2000:])


# ---------------------------------------------------------------------------
# A RESET RUN KEEPS ITS PREDICTIONS FILE (2026-09-05).
#
# Resetting a run to `pending` rewrites `config.json` and leaves
# `final_predictions.csv` exactly where it was. The file is intact, parseable
# and describes a model that has since been discarded. `full_panel` and
# `sensitivity_screen` had always refused such a run; `deployed_h2h`,
# `score_scan` and `paired_noise` globbed for the CSV and never read the
# status, so on the live tree four superseded models were eligible for the
# arm-vs-arm table -- and `deployed_h2h` is the scorer that decides whether
# TraLO beats the rival duals.
#
# The stale run is put on a LIVE arm on purpose. On a dead arm the quarantine
# filter would remove it for an unrelated reason and this test would pass
# while checking nothing.
# ---------------------------------------------------------------------------

STALE_ARM, STALE_SEED = "tralo", 3
STATUS_FILTERED = ("deployed_h2h", "score_scan", "paired_noise")
STALE_PHRASE = "not `status: completed`"


@pytest.fixture(scope="module")
def campaign_with_a_stale_run(tmp_path_factory):
    """The partial campaign, plus one live-arm run reset to `pending`."""
    root = str(tmp_path_factory.mktemp("results") / CAMPAIGN)
    for cap in CAPS:
        for arm in LIVE + DEAD:
            for seed in SEEDS:
                _write_run(root, cap, arm, seed)
    d = _run_dir(root, CAPS[0], STALE_ARM, STALE_SEED)
    cfg_path = os.path.join(d, "config.json")
    with io.open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["status"] = "pending"          # reset; predictions deliberately kept
    with io.open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    assert os.path.exists(os.path.join(d, "final_predictions.csv")), (
        "the fixture must keep the predictions file -- that IS the defect")
    return root, d


@pytest.mark.parametrize("mod", STATUS_FILTERED)
def test_the_scorer_DROPS_a_run_that_was_reset_to_pending(
        mod, campaign_with_a_stale_run, tmp_path):
    root, stale = campaign_with_a_stale_run
    code, out = _invoke(_argv(mod, root, str(tmp_path)))
    assert not any(c in out for c in CRASH), out[-2000:]
    assert STALE_PHRASE in out, (
        "%s never reported a status drop; it scored the reset run" % mod)
    # Anchored to the ACTUAL run, not to a static banner. A footer that always
    # prints the phrase would pass the line above while checking nothing --
    # this one can only appear if that specific path was enumerated and cut.
    tail = "/".join(stale.replace(os.sep, "/").split("/")[-2:])
    assert tail in out.replace(os.sep, "/"), (
        "%s printed the phrase but never named %s" % (mod, tail))


@pytest.mark.parametrize("mod", STATUS_FILTERED)
def test_NEGATIVE_CONTROL_no_status_drop_is_reported_when_none_is_due(
        mod, partial_campaign, tmp_path):
    """The all-completed campaign must stay silent. Without this, a scorer
    that printed the phrase unconditionally would pass the test above."""
    code, out = _invoke(_argv(mod, partial_campaign, str(tmp_path)))
    assert STALE_PHRASE not in out, (
        "%s reported dropping a non-completed run on a campaign that has "
        "none -- the message is unconditional and proves nothing" % mod)


# ---------------------------------------------------------------------------
# THE FIXTURE MUST MATCH WHAT THE PIPELINE ACTUALLY WRITES (2026-09-09)
#
# FRAMEWORK 2(z65). The fixture above carries the comment "the fixture has to
# look like a real run, not like the minimum each tool tolerates" -- and for
# weeks it wrote the group column as `Group` while `src/training/logging.py`
# writes `Group_ID`, and put `seed` at top level while `configs/gen_campaign.py`
# writes it only into `hyperparams`. Consequences:
#
#   * every group-aware scorer here took its NO-GROUP FALLBACK branch, so
#     "25 scorers ran without crashing" meant they ran on a file no run has
#     ever produced;
#   * `full_panel.panel` read `seed: None` for every run, which crashed
#     `cell_table` on `sorted()` five frames deep the moment the group column
#     was corrected and it got that far.
#
# An assertion in a comment is not an assertion. These two are executable, and
# they read the AUTHORITIES rather than restating them, so renaming a field in
# the pipeline turns this red instead of silently re-opening the gap.
# ---------------------------------------------------------------------------

def _authority(rel):
    with io.open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return f.read()


def test_the_fixture_names_its_group_column_the_way_the_pipeline_does(tmp_path):
    assert "'Group_ID'] = group_ids" in _authority("src/training/logging.py"), (
        "src/training/logging.py no longer writes `Group_ID`; this test is "
        "reading a stale authority and must be updated, not deleted.")
    _write_run(str(tmp_path), "L80-80_G95", "tralo", 1)
    d = _run_dir(str(tmp_path), "L80-80_G95", "tralo", 1)
    for name in ("final_predictions.csv", "final_predictions_raw.csv"):
        with io.open(os.path.join(d, name), encoding="utf-8") as f:
            head = f.readline().strip().split(",")
        assert "Group_ID" in head, (
            "%s has no `Group_ID`; every group-aware scorer will take its "
            "fallback branch and this file will test nothing. Header: %s"
            % (name, head))
        assert "Group" not in head, (
            "`Group` is not a column any run writes; %s must use `Group_ID`"
            % name)


def test_the_fixture_puts_the_seed_where_gen_campaign_puts_it(tmp_path):
    assert 'hp["seed"] = seed' in _authority("configs/gen_campaign.py"), (
        "gen_campaign no longer writes the seed into `hyperparams`; this test "
        "is reading a stale authority and must be updated, not deleted.")
    assert '(cfg.get("hyperparams") or {}).get("seed")' in _authority(
        "scripts/full_panel.py"), (
        "full_panel.panel no longer reads the seed from `hyperparams`; same.")
    _write_run(str(tmp_path), "L80-80_G95", "tralo", 3)
    d = _run_dir(str(tmp_path), "L80-80_G95", "tralo", 3)
    with io.open(os.path.join(d, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["hyperparams"].get("seed") == 3, (
        "the seed must live in `hyperparams`, where `panel` reads it; got %r"
        % (cfg["hyperparams"].get("seed"),))
    assert "seed" not in cfg, (
        "a real config has NO top-level `seed` -- gen_campaign writes only "
        "`hyperparams.seed`. Carrying both hides which one is load-bearing.")
