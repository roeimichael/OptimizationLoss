"""Reproductions from the 2026-09-13 audit; no dataset training or network."""

import subprocess
import sys
from pathlib import Path

import pytest

from src.utils import filesystem_manager as fs


@pytest.mark.parametrize("config", [
    {"status": "completed"},
    {"status": "completed", "results": {"accuracy": 0.0}},
    {"status": "pending", "results": {"accuracy": 0.0}},
])
def test_crash_recovery_preserves_completed_or_zero_accuracy_runs(config):
    from scripts.reset_crashed import eligible
    allowed, reason = eligible(config, 0)
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
def test_dispatcher_exit_status_reaches_the_calling_shell(tmp_path, child_code, expected):
    # Execute the actual __main__ block, using a fake child and no GPU/data.
    # A test calling main() directly would miss a forgotten sys.exit(main()).
    script = """
import runpy, subprocess
from unittest.mock import patch
import src.utils.filesystem_manager as fs
config = {'exp_name': 'test-run', 'methodology': 'heuristic'}
pending = [('fake-run', config)]
buckets = {'pending': pending, 'completed': [], 'blocked': []}
def child(*args, **kwargs):
    if CODE == 130:
        raise KeyboardInterrupt
    buckets['pending'] = []
    buckets['completed' if CODE == 0 else 'blocked'] = pending
    return subprocess.CompletedProcess(args[0], CODE)
with patch('torch.cuda.is_available', return_value=False), \
     patch.object(fs, 'get_experiments_by_status', side_effect=lambda *a: buckets), \
     patch.object(fs, 'print_status_summary'), \
     patch('subprocess.run', side_effect=child):
    runpy.run_path('main.py', run_name='__main__')
""".replace("CODE", str(child_code))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == expected, result.stdout + result.stderr


def test_dead_code_scan_includes_top_level_dispatcher(tmp_path):
    from scripts.dead_code import DEFAULT_PATHS, scan, dead
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "helpers.py").write_text(
        "def used_by_dispatcher(): pass\ndef unused(): pass\n", encoding="utf-8")
    (tmp_path / "main.py").write_text(
        "from src.helpers import used_by_dispatcher\nused_by_dispatcher()\n",
        encoding="utf-8")
    definitions, uses = scan(DEFAULT_PATHS, root=str(tmp_path))
    names = {name for _, name, _ in dead(definitions, uses)}
    assert "used_by_dispatcher" not in names
    assert "unused" in names


@pytest.mark.parametrize("missing", ["file", "column", "value"])
def test_declared_disjoint_groups_require_verifiable_metadata(tmp_path, missing):
    import pandas as pd
    from src.utils.data_loader import _check_group_leakage
    pd.DataFrame({"location": [2]}).to_csv(tmp_path / "test_meta.csv", index=False)
    if missing == "column":
        pd.DataFrame({"label": [0]}).to_csv(tmp_path / "train_meta.csv", index=False)
    elif missing == "value":
        pd.DataFrame({"location": [None]}).to_csv(tmp_path / "train_meta.csv", index=False)
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


def test_numbered_rng_streams_are_floor_instruments_not_competitors():
    from scripts.deployed_h2h import rankable_arms, rank_cell, rng_floor
    cell = {arm: {s: value for s in range(1, 5)} for arm, value in {
        "clip": 100, "tralo_snap": 110, "alm": 108,
        "tralo_snap_null": 109, "tralo_snap_reseed": 115,
        "tralo_snap_reseed2": 130, "tralo_snap_reseed12": 135,
        "tralo_lam0": 105,
    }.items()}
    assert set(rankable_arms(cell, "clip")) == {"tralo_snap", "alm", "tralo_lam0"}
    order, _ = rank_cell(cell, "clip", lambda record: record)
    assert order[0][0] == "tralo_snap"
    # Exclusion from the ranking must not discard the observations themselves.
    _, observations, streams = rng_floor(cell, lambda record: record)
    assert (observations, streams) == (24, 4)


def test_arm_restriction_preserves_all_rng_streams(monkeypatch):
    from scripts import deployed_h2h as h2h
    cell = h2h._cell({"clip": [100] * 4, "tralo": [110] * 4,
                      "alm": [108] * 4, "tralo_null": [102] * 4,
                      "tralo_reseed": [105] * 4,
                      "tralo_reseed2": [101] * 4})
    key = ("fixture", "MobileNetV2", "iwildcam", "L70_G95", "2")
    expected_floor = h2h.rng_floor(cell, lambda r: r["TP"])
    original_report = h2h.report

    def check_report(cells, control):
        assert "alm" not in cells[key]
        assert h2h.rng_floor(cells[key], lambda r: r["TP"]) == expected_floor
        return original_report(cells, control, w=lambda message: None)

    monkeypatch.setattr(sys, "argv", ["deployed_h2h", "--campaign", "fixture",
                                     "--arms", "tralo"])
    monkeypatch.setattr(h2h.quarantine, "gate", lambda *args: (False, ()))
    monkeypatch.setattr(h2h, "collect", lambda *args: {key: cell})
    monkeypatch.setattr(h2h, "cap_invariant_arms", lambda *args: {})
    monkeypatch.setattr(h2h, "report", check_report)
    assert h2h.main() == 0


def test_seed_coverage_banner_uses_the_actual_ranking_population():
    from scripts import deployed_h2h as h2h
    cell = h2h._cell({"clip": [100] * 12, "tralo": [110] * 12,
                      "alm": [108] * 12, "tralo_null": [102] * 4,
                      "tralo_reseed": [105] * 4,
                      "tralo_reseed2": [101] * 4})
    key = ("fixture", "MobileNetV2", "iwildcam", "L70_G95", "2")
    output = []
    rows = h2h.report({key: cell}, "clip", w=output.append)
    assert rows[0]["seeds"] == 12
    assert "SEED COVERAGE IS RAGGED" not in "".join(output)
    # A thin COMPETITOR really does limit the ordering, unlike a floor stream.
    cell["alm"] = {s: r for s, r in cell["alm"].items() if s <= 4}
    output.clear()
    rows = h2h.report({key: cell}, "clip", w=output.append)
    assert rows[0]["seeds"] == 4
    assert "ordering below uses 4" in "".join(output)
