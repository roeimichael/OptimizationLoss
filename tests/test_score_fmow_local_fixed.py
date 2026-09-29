"""Independent small examples for fixed-dose integrity and scoring rules."""

import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis import score_fmow_local_fixed as fixed


def _matrix(rows):
    return torch.tensor([row + [0.] * 5 for row in rows])


def _joint_record(pto, side, groups, *, radius=0.1):
    before_g, before_l = fixed.prior._count(pto.argmax(1).tolist(), groups)
    after_g, after_l = fixed.prior._count(side.argmax(1).tolist(), groups)
    return {"applied": True, "radius": radius, "displacement": radius,
            "gradient_norm": .7,
            "tensor_displacement_norms": [radius],
            "hard_before_global": before_g, "hard_before_local": before_l,
            "hard_after_global": after_g, "hard_after_local": after_l,
            "soft_before_global": float(pto[:, 1].sum()),
            "soft_after_global": float(side[:, 1].sum()),
            "soft_before_local": {g: float(pto[[i for i, group in enumerate(groups)
                                               if group == g], 1].sum()) for g in set(groups)},
            "soft_after_local": {g: float(side[[i for i, group in enumerate(groups)
                                             if group == g], 1].sum()) for g in set(groups)},
            "active_global": True, "active_local": ["A"],
            "scope_directional_derivatives": {"global": -0.4, "A": +0.2},
            "directional_soft_delta_global": -0.01,
            "directional_soft_delta_local": {"A": 0.01, "B": -0.02}}


def test_fixed_dose_accepts_conflicting_derivative_and_raw_infeasibility():
    pto = _matrix([[.1, .8, .1], [.1, .7, .2], [.1, .9, .0]])
    side = _matrix([[.11, .79, .1], [.11, .69, .2], [.11, .89, .0]])
    groups = ["A", "A", "B"]
    quota = {"global_cap": 2, "local_caps": {"A": 1, "B": 1}}
    record = _joint_record(pto, side, groups)
    fixed._audit_side(record, pto, side, groups, quota, "joint")
    assert record["hard_after_global"] == 3  # Reported rather than rejected.
    assert record["scope_directional_derivatives"]["A"] > 0
    record["radius"] = .05
    record["displacement"] = .05
    record["tensor_displacement_norms"] = [.05]
    with pytest.raises(RuntimeError, match="fixed dose"):
        fixed._audit_side(record, pto, side, groups, quota, "joint")


def test_fixed_dose_rejects_nonfinite_or_missing_scope_derivatives():
    pto = _matrix([[.1, .8, .1], [.1, .7, .2], [.1, .9, .0]])
    groups = ["A", "A", "B"]
    quota = {"global_cap": 2, "local_caps": {"A": 1, "B": 1}}
    record = _joint_record(pto, pto.clone(), groups)
    record["scope_directional_derivatives"] = {"global": float("nan"), "A": .2}
    with pytest.raises(RuntimeError, match="nonfinite joint scope"):
        fixed._audit_side(record, pto, pto, groups, quota, "joint")
    record["scope_directional_derivatives"] = {"global": -.4}
    with pytest.raises(RuntimeError, match="missing/nonfinite joint scope"):
        fixed._audit_side(record, pto, pto, groups, quota, "joint")


def test_inactive_side_must_be_unchanged():
    pto = _matrix([[.8, .1, .1]])
    quota = {"global_cap": 1, "local_caps": {"A": 1}}
    fixed._audit_side({"applied": False, "displacement": 0.0}, pto, pto.clone(),
                      ["A"], quota, "sham")
    with pytest.raises(RuntimeError, match="inactive dose control"):
        fixed._audit_side({"applied": False, "displacement": 0.0}, pto,
                          _matrix([[.7, .2, .1]]), ["A"], quota, "sham")


def test_labels_opened_only_for_full_scoring_and_manifest_must_be_label_free(tmp_path,
                                                                               monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    arr = np.zeros(3442, dtype=np.int64)
    arr[[10, 20, 30]] = [1, 2, 3]
    np.save(data / "test_labels.npy", arr)
    ids, groups = ["test10", "test20", "test30"], ["IRQ", "NLD", "DZA"]
    manifest = {"files": fixed.prior.FILES, "counts": {"dev": 3}, "quotas": {},
                "dev_countries": ["IRQ", "NLD", "DZA", "PHL", "TUR"],
                "reserved_countries": sorted(fixed.prior.RESERVED),
                "rows": [dict(sample_id=i, location=g, split="val") for i, g in zip(ids, groups)]}
    run = tmp_path / "seed6200"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps(manifest))
    receipt = ({}, {"quotas": {}}, {"counts": {"dev": 3}, "quotas": {}}, [], ids, groups)
    monkeypatch.setattr(fixed.prior, "FILES", {**fixed.prior.FILES,
                                              "test_labels.npy": fixed.prior.sha256(
                                                  data / "test_labels.npy")})
    manifest["files"] = fixed.prior.FILES
    (run / "manifest.json").write_text(json.dumps(manifest))
    assert fixed._manifest_and_labels(run, receipt, data) == [1, 2, 3]
    manifest["rows"][0]["label"] = 1
    (run / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="not label-free"):
        fixed._manifest_and_labels(run, receipt, data)


def test_complete_denominator_and_negative_deltas_are_retained(tmp_path, monkeypatch):
    rows = {}
    for seed in fixed.SEEDS:
        (tmp_path / f"seed{seed}").mkdir()
        caps = {}
        for divisor in fixed.prior.DIVISORS:
            arms = {}
            for arm in fixed.ARMS:
                value = .3 + (seed - fixed.SEEDS[0]) * .001
                if arm == "ens_joint_fixed":
                    value -= .01 if seed % 2 else .005
                arms[arm] = {"allocated": {metric: value for metric in fixed.prior.METRICS},
                             "prediction_sha256": f"{seed}-{divisor}-{arm}"}
            caps[str(divisor)] = {"arms": arms}
        rows[seed] = {"seed": seed, "manifest_sha256": "same", "caps": caps}
    monkeypatch.setattr(fixed, "load_seed", lambda path, data_root: rows[int(path.name[4:])])
    monkeypatch.setattr(fixed.prior, "source", lambda: {"source": "hash"})
    output = tmp_path / "result.json"
    report = fixed.main(tmp_path, tmp_path, output)
    assert len(report["seeds"]) == 12
    assert all(row["mean"] < 0 for row in report["contrasts"]["cc_f1"].values())
    assert len(report["primary_family"]) == 4
    assert output.is_file()
    (tmp_path / "seed6211").rename(tmp_path / "removed")
    with pytest.raises(RuntimeError, match="incomplete"):
        fixed.main(tmp_path, tmp_path)


def test_runner_shaped_pilot_gate_is_label_blind(tmp_path, monkeypatch):
    # Reuse the existing runner-shaped artifact generator, not its audit logic.
    path = Path(__file__).with_name("test_score_fmow_local.py")
    spec = importlib.util.spec_from_file_location("score_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pilot_root, reference_root, pilot, reference, identities, quotas = module.make_runner_shaped_pilot(
        tmp_path, monkeypatch)
    configs = tmp_path / "input_configs"
    configs.mkdir()
    monkeypatch.setattr(fixed, "CONFIGS", configs)
    for directory in (pilot, reference):
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text())
        config.update(seed=6199, study=fixed.STUDY, step_radius=fixed.RADIUS)
        config_path.write_text(json.dumps(config))
        job = "6199_step" if config["snapshot_steps"] else "6199_ref"
        input_config = configs / f"fmow_local_{job}.json"
        input_config.write_text(json.dumps(config, indent=2) + "\n")
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["seed"] = 6199
        summary_path.write_text(json.dumps(summary))
        events_path = directory / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        events[0]["config_sha256"] = fixed.prior.sha256(input_config)
        next(row for row in events if row["event"] == "completed")["seconds"] = 3600.0
        events_path.write_text("\n".join(json.dumps(row) for row in events) + "\n")
    pilot.rename(pilot_root / "seed6199")
    reference.rename(reference_root / "seed6199_ref")
    result = fixed.gate(pilot_root, reference_root)
    assert result["development_labels_accessed"] is False
    assert result["pto_equal"] is True
    assert result["projected_total_gpu_hours"] == 14.0
    step_input = configs / "fmow_local_6199_step.json"
    original = step_input.read_bytes()
    step_input.write_bytes(original.rstrip(b"\n"))
    with pytest.raises(RuntimeError, match="input/run config provenance"):
        fixed.gate(pilot_root, reference_root)
    step_input.write_bytes(original)

    # A separate command may open only the verified development label indices.
    data = tmp_path / "data"
    data.mkdir()
    labels = np.zeros(3442, dtype=np.int64)
    for i in range(0, 1673, 10):
        labels[i] = 1
    np.save(data / "test_labels.npy", labels)
    monkeypatch.setattr(fixed.prior, "FILES", {**fixed.prior.FILES,
                                              "test_labels.npy": fixed.prior.sha256(
                                                  data / "test_labels.npy")})
    run = pilot_root / "seed6199"
    manifest = {"files": fixed.prior.FILES,
                "counts": {"train": 15841, "stop": 1829, "dev": 1673},
                "quotas": quotas, "stop_countries": [],
                "dev_countries": ["IRQ", "NLD", "DZA", "PHL", "TUR"],
                "reserved_countries": sorted(fixed.prior.RESERVED),
                "rows": [dict(row, split="val") for row in identities]}
    (run / "manifest.json").write_text(json.dumps(manifest))
    top_path = run / "events.jsonl"
    top = [json.loads(line) for line in top_path.read_text().splitlines()]
    top[0]["manifest_sha256"] = fixed.prior.sha256(run / "manifest.json")
    top[0]["data_files"] = fixed.prior.FILES
    top_path.write_text("\n".join(json.dumps(row) for row in top) + "\n")
    scored = fixed.pilot_score(pilot_root, data)
    assert scored["development_labels_accessed_offline"] is True
    assert scored["seed"]["seed"] == 6199
    assert set(scored["seed"]["caps"]["10"]["arms"]) == set(fixed.ARMS)
