"""Gates for `scripts/rig_status.py`.

Each predicate is paired with the case it must NOT fire on. A health check that
fires on everything trains the reader to ignore it, which is worse than not
having it -- this project already lost a night to a starvation warning that
fired on an arm with no penalty.
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scripts.rig_status import (code_version_uniform, interpreter_is_env,  # noqa: E402
                                orphaned_runners, shared_gpus, stale_running)

DISPATCH = "/home/u/anaconda3/envs/optloss/bin/python main.py"
RUNNER = ("/home/u/anaconda3/envs/optloss/bin/python -u -m "
          "src.experiments.runner results/iwc2/x/config.json")


def test_orphaned_runners_catches_the_bug_that_nearly_corrupted_iwc2():
    """Killing the dispatcher left three runner children alive, writing into
    the same run directory a relaunched dispatcher had just claimed. `kill` on
    the parent is not `kill` on the work.
    """
    procs = [
        {"pid": 791680, "ppid": 1, "args": RUNNER},
        {"pid": 793392, "ppid": 791680, "args": RUNNER},
        {"pid": 793455, "ppid": 791680, "args": RUNNER},
    ]
    orphans = orphaned_runners(procs)
    assert [p["pid"] for p in orphans] == [791680], (
        "the parentless runner is the orphan; its two forked dataloader "
        "workers have a live runner parent and must not be double-counted, "
        "or the operator is told to kill three things that are one thing")


def test_a_runner_under_a_live_dispatcher_is_not_an_orphan():
    """NEGATIVE CONTROL. This is the NORMAL state of a healthy campaign -- a
    dispatcher with runner children. Firing here would flag every healthy run.
    """
    procs = [
        {"pid": 806447, "ppid": 1, "args": DISPATCH},
        {"pid": 806533, "ppid": 806447, "args": RUNNER},
        {"pid": 807553, "ppid": 806533, "args": RUNNER},
    ]
    assert orphaned_runners(procs) == []


def test_code_version_split_is_caught_and_uniform_is_not():
    """`code_version` is a git hash. A campaign whose completed runs carry two
    of them was built by two different commits and its arms are not comparable
    -- which is the entire reason the training path is frozen mid-campaign.
    """
    split = [{"run_code_version": "3bb7e8b4"}, {"run_code_version": "6206c687"}]
    ok, seen = code_version_uniform(split)
    assert not ok and len(seen) == 2

    uniform = [{"run_code_version": "3bb7e8b4"}] * 19
    ok, seen = code_version_uniform(uniform)
    assert ok and seen == {"3bb7e8b4": 19}, (
        "19 runs from one commit is the healthy state and must stay silent")


def test_code_version_falls_back_to_the_generator_stamp():
    """A run that has not started yet carries only the generator's
    `code_version`; the runner writes `run_code_version` when it actually runs.
    Reading only one of the two would call a fresh campaign broken.
    """
    ok, seen = code_version_uniform([{"code_version": "3bb7e8b4"},
                                     {"run_code_version": "3bb7e8b4"}])
    assert ok and seen == {"3bb7e8b4": 2}


def test_shared_gpu_is_flagged_and_a_solo_gpu_is_not():
    """The house rule is never to share a GPU with another user. During this
    session GPU 0 picked up a second user while three others were already on
    it, which is exactly the state a manual nvidia-smi read glosses over.
    """
    apps = [
        {"gpu": "GPU-0", "pid": "1", "user": "zehavid"},
        {"gpu": "GPU-0", "pid": "2", "user": "michaer8"},
        {"gpu": "GPU-3", "pid": "3", "user": "michaer8"},
    ]
    assert shared_gpus(apps, "michaer8") == ["GPU-0"], (
        "GPU-3 is mine alone and must not be flagged")

    solo = [{"gpu": "GPU-3", "pid": "3", "user": "michaer8"}]
    assert shared_gpus(solo, "michaer8") == []


def test_someone_elses_busy_gpu_is_not_my_problem():
    """NEGATIVE CONTROL. Two other users sharing a GPU I am not on is not a
    finding about my rig; flagging it would make the check noise.
    """
    apps = [{"gpu": "GPU-1", "pid": "1", "user": "zehavid"},
            {"gpu": "GPU-1", "pid": "2", "user": "nirgal"}]
    assert shared_gpus(apps, "michaer8") == []


@pytest.mark.parametrize("path,expected", [
    ("/home/u/anaconda3/envs/optloss/bin/python", True),
    ("/home/u/anaconda3/bin/python", False),
    ("/usr/bin/python3", False),
])
def test_base_conda_is_distinguished_from_the_env(path, expected):
    """Base conda carries a CPU-only torch here. A campaign launched under it
    runs to completion and writes plausible results -- on CPU. The only signal
    at launch was one `Device: CPU` line in a log nobody was reading.
    """
    assert interpreter_is_env(path) is expected


def test_rig_status_runs_end_to_end_and_reports():
    """It must survive a machine with no GPU, no campaigns and no server --
    i.e. the laptop it will be developed on -- rather than raising."""
    r = subprocess.run([sys.executable, "-m", "scripts.rig_status"],
                       cwd=REPO, capture_output=True, text=True, timeout=300)
    out = r.stdout + r.stderr
    assert "RIG STATUS" in out, out[-800:]
    assert "training path" in out, out[-800:]
    assert "git topology" in out, out[-800:]
    assert r.returncode in (0, 1), (
        "exit code is a gate signal: 0 clean, 1 means a FAIL row exists")


def test_a_running_status_with_no_dispatcher_is_a_lie():
    """`running` is written by the dispatcher that started the run and is only
    reset to `pending` when a dispatcher next starts on THAT root. So a
    campaign showing `running` with nothing dispatching it holds a run that
    DIED -- and it reads as alive, which is worse than reading as pending
    because nothing prompts anyone to look at it.

    Found on this tool's first execution: mc29, mnv3bar, vit_ceskip and
    vit_diag each held one, while the only live dispatcher was on iwc2.
    """
    owned = {"/home/u/optloss-audit/results/iwc2"}
    assert stale_running("mc29", 1, owned)
    assert stale_running("vit_diag", 1, owned)


def test_the_campaign_a_dispatcher_owns_is_not_stale():
    """NEGATIVE CONTROL. iwc2 has a live dispatcher, so its `running=1` is the
    healthy state -- one run actually training. Flagging it would fire on
    every campaign that is working, which is the noise mode this file exists
    to prevent. A trailing slash on EXPERIMENT_DIR must not change the answer.
    """
    assert not stale_running("iwc2", 1, {"/home/u/optloss-audit/results/iwc2"})
    assert not stale_running("iwc2", 1, {"/home/u/optloss-audit/results/iwc2/"})


def test_a_finished_campaign_is_never_stale():
    """NEGATIVE CONTROL. running=0 is the resting state of every completed or
    queued campaign -- iwc3 sits at pending=180 with no dispatcher and is
    perfectly healthy. Only a nonzero `running` can be a lie.
    """
    assert not stale_running("iwc3", 0, set())
    assert not stale_running("dosefix", 0, {"/home/u/optloss-audit/results/iwc2"})
