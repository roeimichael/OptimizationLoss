"""A partial block cannot access private development labels."""

import json

import pytest

from analysis import score_tabular_persistent as scorer
from tralo.tabular_image_data import ISIC_CACHED_DRAFT_POLICY, ISIC_DRAFT_POLICY


def test_partial_fixed_block_stops_before_private_development_labels(
        tmp_path, monkeypatch):
    (tmp_path / "seed6801").mkdir()
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text("{}", encoding="utf-8")
    def forbidden(*_args, **_kwargs):
        raise AssertionError("private development label path was opened")
    monkeypatch.setattr(scorer, "_private_labels", forbidden)
    with pytest.raises(RuntimeError, match="complete fixed four-seed block"):
        scorer.score(tmp_path, prepared, tmp_path / "score.json")


def test_partial_draft_seed_block_stops_before_private_labels(tmp_path, monkeypatch):
    full = tmp_path / "full"
    full.mkdir()
    (full / "seed6811").mkdir()
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text(
        json.dumps({"decode_policy": ISIC_DRAFT_POLICY}), encoding="utf-8")
    monkeypatch.setattr(scorer, "_private_labels", lambda *_a, **_kw:
                        pytest.fail("private labels opened for partial block"))
    with pytest.raises(RuntimeError, match="complete fixed four-seed block"):
        scorer.score(full, prepared, tmp_path / "score.json")


def test_partial_cached_seed_block_stops_before_private_labels(tmp_path, monkeypatch):
    full = tmp_path / "full"
    full.mkdir()
    (full / "seed6831").mkdir()
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "manifest.json").write_text(
        json.dumps({"decode_policy": ISIC_CACHED_DRAFT_POLICY}), encoding="utf-8")
    monkeypatch.setattr(scorer, "_private_labels", lambda *_a, **_kw:
                        pytest.fail("private labels opened for partial block"))
    with pytest.raises(RuntimeError, match="complete fixed four-seed block"):
        scorer.score(full, prepared, tmp_path / "score.json")


def test_weighted_and_group_metrics_include_zero_support_class():
    report = scorer._metrics([0, 0, 1, 1], [0, 1, 1, 0],
                             ["female", "female", "male", "male"])
    assert report["cc_f1"] == 0.5
    assert report["weighted_f1"] == 0.5
    assert report["groups"]["female"]["confusion"] == [[1, 1], [0, 0]]
    assert report["groups"]["male"]["confusion"] == [[0, 0], [1, 1]]


def test_completed_cell_cost_includes_pilot_gate_wall_and_checks_ownership(tmp_path):
    cell = tmp_path / "cell"
    full = cell / "full"
    pilot = cell / "pilot" / "seed6800"
    full.mkdir(parents=True)
    pilot.mkdir(parents=True)
    (pilot / "summary.json").write_text("{}", encoding="utf-8")
    from tralo.knee_experiment import digest
    def write(name, row):
        (cell / name).write_text(json.dumps(row), encoding="utf-8")
    write("pilot_6800.launch.json", {
        "seed": 6800, "started_utc": "2026-10-01T20:00:00+00:00",
        "release_commit": "a" * 40, "gpu_uuid": "GPU-test",
        "source_sha256": {"module.py": "b" * 64},
        "prepared_manifest_sha256": "d" * 64})
    write("pilot_6800.complete.json", {"exit_code": 0, "release_commit": "a" * 40})
    write("pilot_gate.json", {"status": "label_blind_integrity_pass",
                              "seed": 6800,
                              "summary_sha256": digest(pilot / "summary.json")})
    audited = []
    for seed in range(6801, 6805):
        destination = str(full / f"seed{seed}")
        write(f"full_{seed}.launch.json", {
            "seed": seed, "release_commit": "a" * 40,
            "source_sha256": {"module.py": "b" * 64},
            "config_sha256": "c" * 64,
            "prepared_manifest_sha256": "d" * 64,
            "output_dir": destination, "gpu_uuid": "GPU-test"})
        write(f"full_{seed}.complete.json", {
            "seed": seed, "exit_code": 0, "release_commit": "a" * 40,
            "gpu_uuid": "GPU-test", "elapsed_seconds": 60,
            "ended_utc": f"2026-10-01T20:{seed - 6799:02d}:00+00:00"})
        audited.append({"config": {"seed": seed}, "elapsed_seconds": 59,
                        "identity": {"source_sha256": {"module.py": "b" * 64},
                                     "config_sha256": "c" * 64,
                                     "prepared_manifest_sha256": "d" * 64}})
    report = scorer._completed_cell_cost(full, audited, 6800)
    assert report["full_queue_gpu_hours"] == pytest.approx(4 / 60)
    assert report["cell_lease_gpu_hours_including_pilot_gate"] == pytest.approx(5 / 60)
    original = json.loads((cell / "full_6804.complete.json").read_text())
    write("full_6804.complete.json", {**original, "gpu_uuid": "GPU-foreign"})
    with pytest.raises(RuntimeError, match="ownership"):
        scorer._completed_cell_cost(full, audited, 6800)
