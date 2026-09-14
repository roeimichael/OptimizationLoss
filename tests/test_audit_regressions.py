"""Maintained behavioral regression fixtures."""

import subprocess

import sys

from pathlib import Path

import pytest

from src.utils import filesystem_manager as fs


@pytest.mark.parametrize(
    "config",
    [
        {"status": "completed"},
        {"status": "completed", "results": {"accuracy": 0.0}},
        {"status": "pending", "results": {"accuracy": 0.0}},
    ],
)
def test_crash_recovery_preserves_completed_or_zero_accuracy_runs(config):
    from scripts.reset_crashed import eligible

    (allowed, reason) = eligible(config, 0)
    assert not allowed, reason


@pytest.mark.parametrize("failure", ["serialize", "replace"])
def test_failed_config_write_preserves_previous_record(tmp_path, monkeypatch, failure):
    original = {"status": "completed", "results": {"accuracy": 0.0}}
    fs.save_config_to_path(original, tmp_path)
    before = (tmp_path / "config.json").read_bytes()
    replacement = {"status": "pending"}
    if failure == "serialize":
        replacement["invalid"] = object()
        error = TypeError
    else:

        def unavailable(*args):
            raise OSError("simulated storage failure")

        monkeypatch.setattr("os.replace", unavailable)
        error = OSError
    with pytest.raises(error):
        fs.save_config_to_path(replacement, tmp_path)
    assert (tmp_path / "config.json").read_bytes() == before
    assert list(tmp_path.iterdir()) == [tmp_path / "config.json"]


def test_config_replacement_roundtrips_and_leaves_no_temporary_file(tmp_path):
    fs.save_config_to_path({"status": "pending"}, tmp_path)
    expected = {"status": "completed", "note": "מדידה", "results": {"accuracy": 0.0}}
    path = fs.save_config_to_path(expected, tmp_path)
    assert Path(path) == tmp_path / "config.json"
    assert fs.load_config_from_path(tmp_path) == expected
    assert list(tmp_path.iterdir()) == [tmp_path / "config.json"]


def test_determinism_setup_failure_prevents_training(monkeypatch):
    from src.pipeline import setup

    def unavailable(*args, **kwargs):
        raise RuntimeError("deterministic setup failed")

    monkeypatch.setattr(setup.torch, "use_deterministic_algorithms", unavailable)
    with pytest.raises(RuntimeError, match="deterministic setup failed"):
        setup.seed_all(1)


@pytest.mark.parametrize("child_code, expected", [(0, 0), (1, 1), (130, 130)])
def test_dispatcher_exit_status_reaches_the_calling_shell(
    tmp_path, child_code, expected
):
    script = "\nimport runpy, subprocess\nfrom unittest.mock import patch\nimport src.utils.filesystem_manager as fs\nconfig = {'exp_name': 'test-run', 'methodology': 'heuristic'}\npending = [('fake-run', config)]\nbuckets = {'pending': pending, 'completed': [], 'blocked': []}\ndef child(*args, **kwargs):\n    if CODE == 130:\n        raise KeyboardInterrupt\n    buckets['pending'] = []\n    buckets['completed' if CODE == 0 else 'blocked'] = pending\n    return subprocess.CompletedProcess(args[0], CODE)\nwith patch('torch.cuda.is_available', return_value=False),      patch.object(fs, 'get_experiments_by_status', side_effect=lambda *a: buckets),      patch.object(fs, 'print_status_summary'),      patch('subprocess.run', side_effect=child):\n    runpy.run_path('main.py', run_name='__main__')\n".replace(
        "CODE", str(child_code)
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == expected, result.stdout + result.stderr


@pytest.mark.parametrize("missing", ["file", "column", "value"])
def test_declared_disjoint_groups_require_verifiable_metadata(tmp_path, missing):
    import pandas as pd
    from src.utils.data_loader import _check_group_leakage

    pd.DataFrame({"location": [2]}).to_csv(tmp_path / "test_meta.csv", index=False)
    if missing == "column":
        pd.DataFrame({"label": [0]}).to_csv(tmp_path / "train_meta.csv", index=False)
    elif missing == "value":
        pd.DataFrame({"location": [None]}).to_csv(
            tmp_path / "train_meta.csv", index=False
        )
    with pytest.raises(ValueError, match="(?i)(metadata|missing|null|disjoint)"):
        _check_group_leakage(str(tmp_path), "location", True)


def test_disjoint_groups_pass_and_real_overlap_fails(tmp_path):
    import pandas as pd
    from src.utils.data_loader import _check_group_leakage

    pd.DataFrame({"location": [1]}).to_csv(tmp_path / "train_meta.csv", index=False)
    pd.DataFrame({"location": [2]}).to_csv(tmp_path / "test_meta.csv", index=False)
    _check_group_leakage(str(tmp_path), "location", True)
    pd.DataFrame({"location": [1]}).to_csv(tmp_path / "test_meta.csv", index=False)
    with pytest.raises(ValueError, match="BOTH splits"):
        _check_group_leakage(str(tmp_path), "location", True)
    _check_group_leakage(str(tmp_path), "location", False)
