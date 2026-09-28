"""Small independent examples for the fixed fmow2 local-study scorer."""

import json

import pytest
import torch

from analysis import score_fmow_local as score


def test_joint_allocator_is_order_independent_and_respects_both_caps():
    p = [row + [0.] * 5 for row in [[.1, .8, .1], [.1, .7, .2], [.1, .6, .3], [.1, .9, .0]]]
    ids = ["z", "b", "a", "x"]
    groups = ["A", "A", "B", "B"]
    expected = [1, 2, 2, 1]
    assert score.allocate_joint(p, ids, groups, 2, {"A": 1, "B": 1}) == expected
    order = [3, 1, 0, 2]
    permuted = score.allocate_joint([p[i] for i in order], [ids[i] for i in order],
                                    [groups[i] for i in order], 2, {"A": 1, "B": 1})
    assert [permuted[order.index(i)] for i in range(4)] == expected


def test_joint_allocator_tie_break_and_underfill():
    p = [[.2, .7, .1] + [0.] * 5 for _ in range(3)]
    assert score.allocate_joint(p, ["b", "a", "c"], ["X", "X", "Y"], 2,
                                {"X": 1, "Y": 0}) == [0, 1, 0]


def test_independent_scoring_reports_raw_and_deployed_counts():
    matrix = torch.tensor([row + [0.] * 5 for row in (
        [.1, .8, .1], [.1, .7, .2], [.1, .6, .3], [.1, .9, .0])])
    report = score._arm_score(matrix, [1, 1, 0, 1], ["z", "b", "a", "x"],
                              ["A", "A", "B", "B"],
                              {"global_cap": 2, "local_caps": {"A": 1, "B": 1}})
    assert report["raw_counts"] == (4, {"A": 2, "B": 2})
    assert report["allocated_counts"] == (2, {"A": 1, "B": 1})
    assert report["local_tp"] == {"A": 1, "B": 1}
    assert report["allocated"]["cc_f1"] == pytest.approx(4 / 5)


def test_inactive_dose_control_must_equal_pto():
    pto = torch.tensor([[.8, .2] + [0.] * 6])
    record = {"applied": False, "displacement": 0.0}
    quota = {"global_cap": 1, "local_caps": {"X": 1}}
    score._audit_step(record, pto, pto.clone(), ["X"], quota, "sham")
    with pytest.raises(RuntimeError, match="inactive control"):
        score._audit_step(record, pto, torch.tensor([[.7, .3] + [0.] * 6]),
                          ["X"], quota, "sham")


def make_runner_shaped_pilot(tmp_path, monkeypatch):
    pilot, reference = tmp_path / "pilot", tmp_path / "reference"
    a, b = pilot / "seed6099", reference / "seed6099_ref"
    groups = ["IRQ", "NLD", "DZA", "PHL", "TUR"] * 334 + ["IRQ", "NLD", "DZA"]
    identities = [{"sample_id": f"test{i}", "location": group}
                  for i, group in enumerate(groups)]
    n = len(groups)
    quotas = score._quotas(groups)
    for d in (a, b):
        (d / "retrain1").mkdir(parents=True)
        # Gate may hash these opaque bytes, but must not parse label-bearing manifests.
        (d / "manifest.json").write_bytes(b"labels are deliberately not JSON")
        (d / "pool_identity.json").write_text(json.dumps(identities))
        (d / "config.json").write_text(json.dumps(dict(score.RECIPE, seed=6099,
                                                        snapshot_steps=d == a)))
        event = {"sequence": 0, "event": "started", "source_sha256": {"x.py": "hash"},
                 "config_sha256": score.sha256(d / "config.json"), "data_files": {"f": "data"},
                 "counts": {"train": 15841, "stop": 1829, "dev": n}, "quotas": quotas,
                 "manifest_sha256": score.sha256(d / "manifest.json"),
                 "pool_identity_sha256": score.sha256(d / "pool_identity.json")}
        init = {"sequence": 1, "event": "model_initialized", "initial_sha256": "initial",
                "architecture": "mobilenet_v3_large", "classes": 8}
        done = {"sequence": 2, "event": "completed", "epochs_run": 6, "task_updates": 2976}
        (d / "events.jsonl").write_text("\n".join(json.dumps(x) for x in (event, init, done)) + "\n")
        steps = {}
        events = []
        for epoch in range(1, 7):
            steps[str(epoch)] = {}
            if d == a:
                for divisor in score.DIVISORS:
                    record = {"joint": {"applied": False, "displacement": 0.0,
                                        "hard_before_global": 0,
                                        "hard_before_local": {g: 0 for g in set(groups)},
                                        "soft_before_global": .2 * n,
                                        "soft_before_local": {g: .2 * groups.count(g) for g in set(groups)},
                                        "active_global": False, "active_local": []},
                              "global_dose": {"applied": False, "displacement": 0.0},
                              "sham": {"applied": False, "displacement": 0.0}}
                    steps[str(epoch)][str(divisor)] = record
                    events.append(("snapshot_cap", {"epoch": epoch, "divisor": divisor,
                                                    "quota": quotas[str(divisor)], "steps": record}))
                    capdir = d / "retrain1" / f"cap{divisor}"
                    capdir.mkdir(exist_ok=True)
                    for arm in ("joint", "global_dose", "sham"):
                        side_path = capdir / f"epoch{epoch:02d}_{arm}.pt"
                        torch.save(torch.tensor([[.8, .2] + [0.] * 6] * n), side_path)
                        record[arm]["probability_sha256"] = score.sha256(side_path)
                    events[-1][1]["steps"] = record
            events.append(("epoch", {"epoch": epoch, "hard_counts": [n, 0] + [0] * 6,
                                     "training_loss": 1.0, "stop_loss": 1.0,
                                     "base_lr": 1e-4, "last_lr": 1e-4, "mean_gate": .5,
                                     "live_false_positives": 0.0, "soft_count_capped": .2 * n,
                                     "improved": epoch == 1}))
        events.append(("training_completed", {"task_updates": 2976}))
        train_events = [dict(sequence=i, event=kind, **extras) for i, (kind, extras) in enumerate(events)]
        (d / "retrain1" / "events.jsonl").write_text("\n".join(json.dumps(x) for x in train_events) + "\n")
        pto = torch.tensor([[.8, .2] + [0.] * 6] * n)
        for epoch in range(1, 7):
            torch.save(pto, d / "retrain1" / f"epoch{epoch:02d}.pt")
        torch.save(pto, d / "retrain1" / "final_probabilities.pt")
        snapshot_hashes = {str(epoch): score.sha256(d / "retrain1" / f"epoch{epoch:02d}.pt")
                           for epoch in range(1, 7)}
        (d / "summary.json").write_text(json.dumps({"seed": 6099, "initial_sha256": "initial",
                                                   "quotas": quotas,
                                                   "retrain": {"best_epoch": 1, "epochs_run": 6,
                                                                   "first_order_sha256": "order",
                                                                   "first_batch_sha256": "batch",
                                                                   "best_stop_loss": 1.0,
                                                       "task_updates": 2976},
                                                   "pto_snapshot_sha256": snapshot_hashes,
                                                   "final_probability_sha256": score.sha256(
                                                       d / "retrain1" / "final_probabilities.pt"),
                                                   "steps": steps if d == a else {}}))
    monkeypatch.setattr(score, "source", lambda: {"x.py": "hash"})
    monkeypatch.setattr(score, "FILES", {"f": "data"})
    return pilot, reference, a, b, identities, quotas


def test_pilot_gate_never_reads_manifest_labels(tmp_path, monkeypatch):
    pilot, reference, a, _, identities, _ = make_runner_shaped_pilot(tmp_path, monkeypatch)
    n = len(identities)
    score.gate(pilot, reference)
    summary_path, event_path = a / "summary.json", a / "retrain1" / "events.jsonl"
    summary_bytes, event_bytes = summary_path.read_bytes(), event_path.read_bytes()
    altered = json.loads(summary_bytes)
    altered["steps"]["1"]["10"]["joint"]["hard_before_local"]["IRQ"] = 1
    summary_path.write_text(json.dumps(altered))
    event_rows = [json.loads(line) for line in event_bytes.splitlines()]
    next(row for row in event_rows if row["event"] == "snapshot_cap" and row["epoch"] == 1
         and row["divisor"] == 10)["steps"]["joint"]["hard_before_local"]["IRQ"] = 1
    event_path.write_text("\n".join(json.dumps(row) for row in event_rows) + "\n")
    with pytest.raises(RuntimeError, match="hard-before record disagrees with PTO"):
        score.gate(pilot, reference)
    summary_path.write_bytes(summary_bytes)
    event_path.write_bytes(event_bytes)
    side_path = a / "retrain1" / "cap10" / "epoch01_sham.pt"
    original = side_path.read_bytes()
    torch.save(torch.tensor([[.7, .3] + [0.] * 6] * n), side_path)
    with pytest.raises(RuntimeError, match="creation-time side snapshot hash mismatch"):
        score.gate(pilot, reference)
    side_path.write_bytes(original)
    torch.save(torch.tensor([[.3, .7] + [0.] * 6] * n), a / "retrain1" / "epoch02.pt")
    summary = json.loads((a / "summary.json").read_text())
    summary["pto_snapshot_sha256"]["2"] = score.sha256(a / "retrain1" / "epoch02.pt")
    (a / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="PTO snapshot mismatch"):
        score.gate(pilot, reference)


def test_complete_seed_schema_scored_without_reserved_countries(tmp_path, monkeypatch):
    _, _, pilot, _, identities, quotas = make_runner_shaped_pilot(tmp_path, monkeypatch)
    directory = tmp_path / "seed6100"
    pilot.rename(directory)
    config_path, summary_path = directory / "config.json", directory / "summary.json"
    config = json.loads(config_path.read_text())
    config["seed"] = 6100
    config_path.write_text(json.dumps(config))
    summary = json.loads(summary_path.read_text())
    summary["seed"] = 6100
    summary_path.write_text(json.dumps(summary))
    manifest = {"files": score.FILES, "counts": {"train": 15841, "stop": 1829, "dev": 1673},
                "quotas": quotas, "stop_countries": [],
                "dev_countries": ["IRQ", "NLD", "DZA", "PHL", "TUR"],
                "reserved_countries": ["EGY", "CAN", "IND", "MEX", "JPN"],
                "rows": [dict(row, split="val", label=int(i % 10 == 0))
                         for i, row in enumerate(identities)]}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    top_path = directory / "events.jsonl"
    top = [json.loads(line) for line in top_path.read_text().splitlines()]
    top[0]["config_sha256"] = score.sha256(config_path)
    top[0]["manifest_sha256"] = score.sha256(directory / "manifest.json")
    top_path.write_text("\n".join(json.dumps(row) for row in top) + "\n")
    row = score.load_seed(directory)
    assert row["seed"] == 6100
    assert row["window"] == list(range(1, 7))
    assert set(row["caps"]) == {"10", "20"}
    for cap in row["caps"].values():
        assert cap["arms"]["ens_joint"]["allocated_counts"][0] <= cap["quota"]["global_cap"]
        assert cap["arms"]["ens_pto"]["raw_counts"][0] == 0


def test_zero_seed_variance_has_no_inferential_interval():
    row = score._paired([.02, .02, .02, .02])
    assert row["mean"] == .02
    assert row["interval_available"] is False
    assert row["lo"] is None and row["hi"] is None
    assert row["p"] == 1.0


def test_complete_block_aggregation_and_json(tmp_path, monkeypatch):
    rows = {}
    for seed in score.SEEDS:
        (tmp_path / f"seed{seed}").mkdir()
        caps = {}
        for divisor in score.DIVISORS:
            arms = {}
            for arm in score.ARMS:
                value = .35 + (seed - 6100) * .0001
                if arm == "ens_joint":
                    value += .003 if seed % 2 else .001
                arms[arm] = {"allocated": {metric: value for metric in score.METRICS},
                             "prediction_sha256": f"{arm}-{divisor}-{seed}"}
            caps[str(divisor)] = {"arms": arms}
        rows[seed] = {"seed": seed, "manifest_sha256": "fixed", "caps": caps}
    monkeypatch.setattr(score, "load_seed", lambda path: rows[int(path.name.removeprefix("seed"))])
    output = tmp_path / "analysis.json"
    report = score.main(tmp_path, output)
    assert output.is_file()
    assert len(report["seeds"]) == 48
    assert len(report["primary_family"]) == 4
    assert all(report["contrasts"]["cc_f1"][name]["interval_available"]
               for name in report["primary_family"])
    assert json.loads(output.read_text())["status"] == "complete_48_seed_exploratory_development"
