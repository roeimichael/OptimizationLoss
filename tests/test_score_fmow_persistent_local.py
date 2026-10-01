"""Independent scorer checks, including mutations that must close the gate."""

import copy
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from analysis import score_fmow_persistent_local as score
from tralo.fmow_persistent_local import correction, write_snapshot


def probabilities():
    value = torch.full((3, 8), .02, dtype=torch.float32)
    value[0, 1], value[1, 1], value[2, 1] = .86, .72, .08
    for i in range(3):
        value[i, 0] = 1 - float(value[i, 1]) - 6 * .02
    return value


def quota():
    return {"global_cap": 1, "local_caps": {"A": 1, "B": 1}}


def correction_record(method="phr"):
    scopes = score._scopes(probabilities(), ["A", "A", "B"], quota())
    dual_after = [max(0., .5 * scopes[g]["signed_residual"])
                  for g in ("global", "A", "B")]
    controller = {"applied": False, "radius": 0., "displacement": 0.,
                  "boundary_policy": {"applied": False, "radius": 0.,
                                      "reason": "no_positive_violation", "probes": []},
                  "dual_before": [0., 0., 0.], "dual_after": dual_after}
    for label in ("before", "after"):
        controller[f"hard_{label}_global"] = scopes["global"]["hard"]
        controller[f"hard_{label}_local"] = {g: scopes[g]["hard"] for g in ("A", "B")}
        controller[f"soft_{label}_global"] = scopes["global"]["soft"]
        controller[f"soft_{label}_local"] = {g: scopes[g]["soft"] for g in ("A", "B")}
        residuals = [scopes[g]["signed_residual"] for g in ("global", "A", "B")]
        controller[f"residuals_{label}"] = residuals
        controller[f"penalty_{label}"] = sum(max(0., .5 * value) ** 2 for value in residuals)
    return {"attempted_constraint_updates": 1, "applied_constraint_updates": 0,
            "skipped_constraint_updates": 1, "before": scopes, "after": scopes,
            "rng_neutral": True, "optimizer_before_sha256": "same",
            "optimizer_after_sha256": "same", "model_before_sha256": "same",
            "model_after_sha256": "same", "attempted_radius": .1,
            "applied_radius": 0., "actual_displacement": 0.,
            "controller": controller, "dual_after": dual_after if method == "phr" else None}


def test_scope_recount_and_partition_rejects_mutated_soft_count():
    values = probabilities()
    row = score._scopes(values, ["A", "A", "B"], quota())
    score._audit_scopes(row, quota(), "test", actual=row)
    altered = copy.deepcopy(row)
    altered["A"]["soft"] += .1
    with pytest.raises(RuntimeError, match="residual|partition"):
        score._audit_scopes(altered, quota(), "test", actual=row)


def test_phr_dual_recomputed_even_when_parameter_step_skips():
    row = correction_record()
    updated = score._audit_correction(row, quota(), "phr", [0., 0., 0.])
    assert updated == row["dual_after"]
    changed = copy.deepcopy(row)
    changed["dual_after"][0] += .1
    with pytest.raises(RuntimeError, match="dual update"):
        score._audit_correction(changed, quota(), "phr", [0., 0., 0.])
    with pytest.raises(RuntimeError, match="continuity"):
        score._audit_correction(row, quota(), "phr", [1., 0., 0.])


def test_null_slot_rejects_fake_dose_and_state_change():
    row = correction_record("null")
    row.update(attempted_constraint_updates=0, skipped_constraint_updates=0,
               attempted_radius=0.)
    row["controller"] = {"applied": False, "radius": 0., "displacement": 0.,
                         "skip_reason": "scheduled_zero_step"}
    score._audit_correction(row, quota(), "null", None)
    bad = copy.deepcopy(row)
    bad["model_after_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="zero-step"):
        score._audit_correction(bad, quota(), "null", None)
    bad = copy.deepcopy(row)
    bad["applied_constraint_updates"] = 1
    with pytest.raises(RuntimeError, match="dose"):
        score._audit_correction(bad, quota(), "null", None)


def test_snapshot_independently_checks_bytes_and_model_optimizer_state(tmp_path):
    model = torch.nn.Linear(2, 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    record = write_snapshot(tmp_path / "ce_null", 5, model, optimizer,
                            probabilities(), None)
    saved = score._snapshot(tmp_path, "ce_null", 5, record, 3, None)
    assert torch.equal(saved, probabilities())
    broken = copy.deepcopy(record)
    broken["model_state_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="checkpoint state"):
        score._snapshot(tmp_path, "ce_null", 5, broken, 3, None)
    path = tmp_path / "ce_null" / record["probability_file"]
    path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="digest"):
        score._snapshot(tmp_path, "ce_null", 5, record, 3, None)


def test_checkpoint_replay_rejects_mutated_probabilities(tmp_path):
    model = torch.nn.Linear(2, 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.tensor([[.2, .8], [.7, .1]])
    expected = model(inputs).softmax(1).detach()
    record = write_snapshot(tmp_path, 5, model, optimizer, expected, None)
    path = tmp_path / record["checkpoint_file"]
    factory = lambda **kwargs: torch.nn.Linear(2, 8)
    result = score._replay_checkpoint(path, [inputs], expected, "tiny", "cpu",
                                      model_factory=factory)
    assert result["max_tolerance_ratio"] == 0
    corrupted = expected.clone()
    corrupted[0, 1] += .01
    with pytest.raises(RuntimeError, match="checkpoint-to-probability replay"):
        score._replay_checkpoint(path, [inputs], corrupted, "tiny", "cpu",
                                 model_factory=factory)


@pytest.mark.parametrize("method", ["tralo", "phr"])
def test_real_pilot_correction_replay_detects_gradient_and_prestate_mutation(
        tmp_path, monkeypatch, method):
    monkeypatch.setattr(score.base, "CLASSES", 3)
    monkeypatch.setitem(score.RECIPE, "development_batch_size", 5)
    torch.manual_seed(3)
    model = torch.nn.Linear(4, 3)
    with torch.no_grad():
        model.weight.zero_()
        model.bias[:] = torch.tensor([0., 2., 0.])
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    chunks = [torch.randn(5, 4), torch.randn(5, 4)]
    groups = ["A"] * 5 + ["B"] * 5
    ids = [f"test{i}" for i in range(10)]
    limits = {"global_cap": 4, "local_caps": {"A": 2, "B": 3}}
    arm = f"cap10_{method}"
    dual = torch.zeros(3) if method == "phr" else None
    directory = tmp_path / arm
    record, next_dual = correction(
        model, optimizer, chunks, groups, limits, method, dual,
        {"pilot": True, "reference": False, "radius": .1, "rho": .5},
        pre_snapshot={"directory": directory, "epoch": 5,
                      "cap_divisor": 10, "sample_ids": ids})
    record.update(epoch=5, cap_divisor=10)
    post = write_snapshot(directory, 5, model, optimizer, score.infer(model, chunks),
                          next_dual)
    replay = score._replay_pilot_correction(
        tmp_path, arm, 5, record, post, chunks, ids, groups, limits, "cpu",
        model_factory=lambda **_: torch.nn.Linear(4, 3))
    assert replay["post_model_sha256"] == record["model_after_sha256"]
    changed = copy.deepcopy(record)
    changed["controller"]["gradient_norm"] += .1
    with pytest.raises(RuntimeError, match="independent constraint gradient"):
        score._replay_pilot_correction(
            tmp_path, arm, 5, changed, post, chunks, ids, groups, limits,
            "cpu", model_factory=lambda **_: torch.nn.Linear(4, 3))
    changed = copy.deepcopy(record)
    changed["controller"]["boundary_policy"]["radius"] += .02
    with pytest.raises(RuntimeError, match="boundary accepted radius differs"):
        score._replay_pilot_correction(
            tmp_path, arm, 5, changed, post, chunks, ids, groups, limits,
            "cpu", model_factory=lambda **_: torch.nn.Linear(4, 3))
    changed = copy.deepcopy(record)
    changed["pre_correction_snapshot"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="artifact digest"):
        score._replay_pilot_correction(
            tmp_path, arm, 5, changed, post, chunks, ids, groups, limits,
            "cpu", model_factory=lambda **_: torch.nn.Linear(4, 3))


def test_config_rejects_nonfixed_pretrained_weight_and_focal_recipe():
    config = {**score.RECIPE, "seed": 6700, "pilot": True,
              "reference": False, "pretrained_sha256": score.WEIGHT_SHA}
    score._check_config(config, 6700, False)
    with pytest.raises(RuntimeError, match="checkpoint"):
        score._check_config({**config, "pretrained_sha256": "0" * 64}, 6700, False)
    with pytest.raises(RuntimeError, match="config"):
        score._check_config({**config, "focal_gamma": 1.0}, 6700, False)


def test_real_transform_and_first_batches_bind_training_to_preflight(tmp_path, monkeypatch):
    import numpy as np
    images = {"train": np.zeros((5, 224, 224, 3), dtype=np.uint8),
              "test": np.zeros((1, 224, 224, 3), dtype=np.uint8)}
    for i in range(5):
        images["train"][i, :, :, i % 3] = 40 + i * 20
    split = {"train": [0, 1, 2, 3], "dev": [0]}
    preflight = score._training_input_fingerprints(
        images, np.array([0, 1, 0, 1, 0]), split, [6700])
    assert len(preflight["first_batches"]["6700"]) == 7
    run_root = tmp_path / "run"
    (run_root / "seed6700").mkdir(parents=True)
    costs = tmp_path / ".fmow-persistent-local-cost"
    attempt = costs / hashlib.sha256(str(run_root.resolve()).encode()).hexdigest()
    attempt.mkdir(parents=True)
    record = {**preflight, "passed": True, "run_root": str(run_root.resolve()),
              "source_sha256": score.source(), "preprocessing": score.PREPROCESSING}
    (attempt / "preflight.json").write_text(json.dumps(record))
    identity = {"release_commit": "a" * 40, "host": "dsisco02",
                "gpu_uuid": "GPU-fake", "run_root": str(run_root.resolve()),
                "mode": "pilot-step"}
    (attempt / "attempt.json").write_text(json.dumps(identity))
    epochs = [{"epoch": epoch, **preflight["first_batches"]["6700"][str(epoch)]}
              for epoch in score.EPOCHS]
    summary = {"arms": {"ce_null": {"epochs": epochs},
                        "focal_clip": {"epochs": copy.deepcopy(epochs)}}}
    config = {"seed": 6700, "pilot": True, "reference": False}
    launch = identity
    score._audit_training_fingerprints(run_root / "seed6700", config, summary, launch)
    with pytest.raises(RuntimeError, match="host, GPU or root"):
        score._audit_training_fingerprints(run_root / "seed6700", config, summary,
                                            {**launch, "gpu_uuid": "GPU-other"})
    changed = copy.deepcopy(summary)
    changed["arms"]["focal_clip"]["epochs"][3]["first_batch_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="first batch/order"):
        score._audit_training_fingerprints(run_root / "seed6700", config, changed, launch)
    train, evaluate = score.transforms_for()
    monkeypatch.setattr(score, "transforms_for", lambda: (evaluate, evaluate))
    with pytest.raises(RuntimeError, match="augmentation differs"):
        score._training_input_fingerprints(images, np.array([0, 1, 0, 1, 0]), split,
                                           [6700])


def test_exclusive_launch_receipt_binds_host_uuid_source_and_config(tmp_path):
    run = tmp_path / "seed6700"
    run.mkdir()
    config = {**score.RECIPE, "seed": 6700, "pilot": True,
              "reference": False, "pretrained_sha256": score.WEIGHT_SHA}
    manifest = {"config_sha256": "config-hash", "source_sha256": {"src": "source-hash"}}
    launch = {"job": "6700_step", "seed": 6700, "reference": False,
              "host": "dsisco02.example.edu", "gpu_index": 3,
              "gpu_uuid": "GPU-physical-uuid", "precision": "fp32_tf32_off",
              "release_commit": "a" * 40, "source_sha256": manifest["source_sha256"],
              "config_sha256": manifest["config_sha256"],
              "run_root": str(tmp_path.resolve()), "output_dir": str(run.resolve())}
    complete = {key: launch[key] for key in ("job", "seed", "host", "gpu_uuid",
                                             "release_commit", "output_dir")}
    complete["exit_code"] = 0
    lpath, cpath = tmp_path / "6700_step.launch.json", tmp_path / "6700_step.complete.json"
    lpath.write_text(json.dumps(launch))
    cpath.write_text(json.dumps(complete))
    events = [{"event": "started", "precision": "fp32_tf32_off"}]
    assert score._audit_launch(run, config, manifest, events,
                               replay_host="dsisco02")["gpu_uuid"] == launch["gpu_uuid"]
    for change in ({"config_sha256": "bad"}, {"gpu_uuid": "GPU-other"},
                   {"host": "dsisco01"}):
        bad = {**launch, **change}
        lpath.write_text(json.dumps(bad))
        with pytest.raises(RuntimeError, match="receipt|host"):
            score._audit_launch(run, config, manifest, events, replay_host="dsisco02")
    lpath.write_text(json.dumps(launch))
    cpath.write_text(json.dumps({**complete, "exit_code": 1}))
    with pytest.raises(RuntimeError, match="receipt"):
        score._audit_launch(run, config, manifest, events)


def test_replay_refuses_unpinned_or_occupied_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="UUID-pinned"):
        score._exclusive_replay_uuid("cuda:0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abcdef")
    class Result:
        stdout = "987654\n"
    monkeypatch.setattr(score.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(RuntimeError, match="foreign compute"):
        score._exclusive_replay_uuid("cuda:0")
    Result.stdout = f"{score.os.getpid()}\n"
    assert score._exclusive_replay_uuid("cuda:0") == "GPU-abcdef"


def test_replay_refuses_cooperating_gpu_lease(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abcdef")
    class Result:
        stdout = ""
    monkeypatch.setattr(score.subprocess, "run", lambda *args, **kwargs: Result())
    def busy(*args):
        raise BlockingIOError("claimed")
    monkeypatch.setitem(sys.modules, "fcntl", types.SimpleNamespace(
        flock=busy, LOCK_EX=1, LOCK_NB=2))
    with pytest.raises(RuntimeError, match="lease already held"):
        with score._exclusive_replay_lease("cuda:0", tmp_path):
            pytest.fail("must not enter occupied replay card")


def test_cost_registry_recounts_failed_smoke_and_rejects_missing_preflight_or_smoke(
        tmp_path, monkeypatch):
    monkeypatch.setattr(score, "source", lambda: {"runner": "fixed"})
    monkeypatch.setattr(score, "FILES", {"data": "fixed"})
    monkeypatch.setattr(score, "_source_at_release", lambda _: score.source())
    monkeypatch.setattr(score, "_scorer_identity", lambda: {"release_commit": "a" * 40})
    registry = tmp_path / "cost"
    attempt = registry / "one"
    attempt.mkdir(parents=True)
    run_root = str(tmp_path / "pilot")
    def write(name, value):
        (attempt / name).write_text(json.dumps(value))
    write("attempt.json", {"study": score.RECIPE["study"], "run_root": run_root,
                           "mode": "pilot-step", "host": "dsisco02",
                           "gpu_uuid": "GPU-abc", "release_commit": "a" * 40})
    with pytest.raises(RuntimeError, match="unresolved persistent cost"):
        score._recount_cost_registry(registry)
    write("preflight.start.json", {"run_root": run_root})
    write("preflight.json", {"run_root": run_root, "source_sha256": score.source(),
                             "data_files": score.FILES, "passed": True,
                             "train_count": score.TRAIN_COUNT,
                             "stop_count": score.STOP_COUNT,
                             "development_count": score.POOL_COUNT,
                             "reserved_images": score.TEST_COUNT - score.POOL_COUNT,
                             "pretrained_weight": {"sha256": score.WEIGHT_SHA},
                             "preprocessing": score.PREPROCESSING, "quotas": {},
                             "train_transform_probe_sha256": "a" * 64,
                             "evaluation_transform_probe_sha256": "b" * 64,
                             "first_batches": {"6700": {str(epoch): {
                                 "sample_order_sha256": "c" * 64,
                                 "first_batch_sha256": "d" * 64}
                                 for epoch in score.EPOCHS}}})
    with pytest.raises(RuntimeError, match="unresolved persistent CUDA smoke"):
        score._recount_cost_registry(registry)
    write("smoke.start.json", {"run_root": run_root, "gpu_uuid": "GPU-abc"})
    write("smoke.json", {"run_root": run_root, "gpu_uuid": "GPU-abc",
                         "source_sha256": score.source(), "passed": False,
                         "elapsed_seconds": 17.5})
    counted = score._recount_cost_registry(registry)
    assert counted["gpu_seconds"] == 17.5
    with pytest.raises(RuntimeError, match="missing successful queue"):
        score._recount_cost_registry(registry, [run_root])
    write("smoke.json", {"run_root": run_root, "gpu_uuid": "GPU-abc",
                         "source_sha256": score.source(), "passed": True,
                         "elapsed_seconds": 17.5, "precision": "fp32_tf32_off",
                         "gpu_name": "Fake GPU", "ce_loss": 1., "focal_loss": .2,
                         "ce_backbone_grad_norm": 1.,
                         "focal_backbone_grad_norm": 1.})
    assert score._recount_cost_registry(registry, [run_root])["gpu_seconds"] == 17.5
    write("smoke.json", {"run_root": run_root, "gpu_uuid": "GPU-abc",
                         "source_sha256": score.source(), "passed": True,
                         "elapsed_seconds": 17.5, "precision": "fp32_tf32_off",
                         "gpu_name": "Fake GPU", "ce_loss": 1., "focal_loss": .2,
                         "ce_backbone_grad_norm": 0.,
                         "focal_backbone_grad_norm": 1.})
    with pytest.raises(RuntimeError, match="grad_norm"):
        score._recount_cost_registry(registry, [run_root])
    write("smoke.json", {"run_root": run_root, "gpu_uuid": "GPU-abc",
                         "source_sha256": score.source(), "passed": True,
                         "elapsed_seconds": 17.5, "precision": "fp32_tf32_off",
                         "gpu_name": "Fake GPU", "ce_loss": 1., "focal_loss": .2,
                         "ce_backbone_grad_norm": 1.,
                         "focal_backbone_grad_norm": 1.})
    write("fresh_gate.start.json", {"run_root": run_root, "gpu_uuid": "GPU-abc"})
    with pytest.raises(RuntimeError, match="unresolved full-gate"):
        score._recount_cost_registry(registry, [run_root])
    (attempt / "fresh_gate.start.json").unlink()
    reference_run_root = str(tmp_path / "reference")
    reference_attempt = registry / "reference"
    reference_attempt.mkdir()
    for name in ("attempt.json", "preflight.start.json", "preflight.json",
                 "smoke.start.json", "smoke.json"):
        copied = json.loads((attempt / name).read_text())
        copied["run_root"] = reference_run_root
        if name == "attempt.json":
            copied["mode"] = "pilot-ref"
        (reference_attempt / name).write_text(json.dumps(copied))
    pilot_gate_start = {"source_sha256": score.source(),
                         "pilot_root": run_root + "/seed6700",
                         "reference_root": reference_run_root + "/seed6700_ref",
                         "gpu_uuid": "GPU-abc"}
    (registry / "pilot_gate_1.start.json").write_text(json.dumps(pilot_gate_start))
    with pytest.raises(RuntimeError, match="unresolved pilot-gate"):
        score._recount_cost_registry(registry, [run_root])
    (registry / "pilot_gate_1.json").write_text(json.dumps({**pilot_gate_start,
                                                              "passed": False,
                                                              "elapsed_seconds": 23.}))
    assert score._recount_cost_registry(registry, [run_root])["gpu_seconds"] == 58.0
    # The final accounting must read the *completed* fresh replay receipt.
    write("fresh_gate.start.json", {"run_root": run_root, "gpu_uuid": "GPU-abc"})
    write("fresh_gate.json", {"run_root": run_root, "gpu_uuid": "GPU-abc",
                              "elapsed_seconds": 13.0, "exit_code": 0})
    gate_record = {"cost_registry": {"root": str(registry), "gpu_seconds": 58.},
                   "pilot_root": run_root + "/seed6700",
                   "reference_root": reference_run_root + "/seed6700_ref",
                   "pilot_seconds": 100., "reference_seconds": 100.,
                   "prior_gpu_hours": 58. / 3600}
    seed_rows = [{"summary": {"elapsed_seconds": 100.},
                  "replay": {"seconds": 1.}}]
    hours, completed = score._actual_study_hours(gate_record, seed_rows, run_root)
    assert completed["gpu_seconds"] == 71.0
    assert hours == pytest.approx((71 + 100 + 100 + 101) / 3600)
    (attempt / "fresh_gate.json").unlink()
    with pytest.raises(RuntimeError, match="unresolved full-gate"):
        score._actual_study_hours(gate_record, seed_rows, run_root)


def test_historical_failed_preflight_is_preserved_and_does_not_poison_new_release(
        tmp_path, monkeypatch):
    old_commit, new_commit = "1" * 40, "2" * 40
    old_source, new_source = {"runner.py": "old"}, {"runner.py": "new"}
    monkeypatch.setattr(score, "_source_at_release", lambda commit:
                        {old_commit: old_source, new_commit: new_source}[commit])
    monkeypatch.setattr(score, "source", lambda: new_source)
    monkeypatch.setattr(score, "_scorer_identity", lambda:
                        {"release_commit": new_commit})
    registry = tmp_path / "cost"
    old = registry / "old"
    old.mkdir(parents=True)
    root = str(tmp_path / "old_run")
    (old / "attempt.json").write_text(json.dumps({
        "study": score.RECIPE["study"], "run_root": root, "mode": "pilot-step",
        "host": "dsisco02", "gpu_uuid": "GPU-old", "release_commit": old_commit}))
    (old / "preflight.start.json").write_text(json.dumps({"run_root": root}))
    receipt = {"run_root": root, "source_sha256": old_source,
               "passed": False, "error": "FileNotFoundError('data')"}
    (old / "preflight.json").write_text(json.dumps(receipt))
    smoked = registry / "old_smoke"
    smoked.mkdir()
    smoke_root = str(tmp_path / "old_smoke_run")
    (smoked / "attempt.json").write_text(json.dumps({
        "study": score.RECIPE["study"], "run_root": smoke_root, "mode": "pilot-ref",
        "host": "dsisco02", "gpu_uuid": "GPU-old", "release_commit": old_commit}))
    (smoked / "preflight.start.json").write_text(json.dumps({"run_root": smoke_root}))
    (smoked / "preflight.json").write_text(json.dumps({
        "run_root": smoke_root, "source_sha256": old_source, "passed": True,
        "data_files": {"old.npy": "a" * 64}, "train_count": score.TRAIN_COUNT,
        "stop_count": score.STOP_COUNT, "development_count": score.POOL_COUNT,
        "reserved_images": score.TEST_COUNT - score.POOL_COUNT,
        "pretrained_weight": {"sha256": score.WEIGHT_SHA},
        "preprocessing": score.PREPROCESSING, "quotas": {},
        "train_transform_probe_sha256": "a" * 64,
        "evaluation_transform_probe_sha256": "b" * 64,
        "first_batches": {"6700": {str(epoch): {
            "sample_order_sha256": "c" * 64, "first_batch_sha256": "d" * 64}
            for epoch in score.EPOCHS}}}))
    (smoked / "smoke.start.json").write_text(json.dumps({"run_root": smoke_root}))
    (smoked / "smoke.json").write_text(json.dumps({
        "run_root": smoke_root, "gpu_uuid": "GPU-old", "source_sha256": old_source,
        "passed": False, "elapsed_seconds": 7.3, "error": "CUDA failure"}))
    count = score._recount_cost_registry(registry)
    assert count["gpu_seconds"] == 7.3
    assert {entry["release_commit"] for entry in count["entries"]} == {old_commit}
    with pytest.raises(RuntimeError, match="required current preflight"):
        score._recount_cost_registry(registry, [root])
    (old / "preflight.json").write_text(json.dumps({**receipt, "source_sha256": new_source}))
    with pytest.raises(RuntimeError, match="identity differs"):
        score._recount_cost_registry(registry)
    (old / "preflight.json").write_text(json.dumps({**receipt, "error": None}))
    with pytest.raises(RuntimeError, match="preserved error"):
        score._recount_cost_registry(registry)


def test_fresh_full_gate_rejects_stale_artifact_and_forged_attempt_before_replay(
        tmp_path, monkeypatch):
    pilot, reference, full = (tmp_path / name / seed for name, seed in
                              (("pilot", "seed6700"), ("reference", "seed6700_ref"),
                               ("full", "seed6701")))
    for root in (pilot, reference, full):
        root.mkdir(parents=True)
    for root in (pilot, reference):
        (root / "manifest.json").write_text('{"fixed": true}')
        (root / "summary.json").write_text('{"elapsed_seconds": 100}')
    old = {"pilot_root": str(pilot.resolve()), "reference_root": str(reference.resolve()),
           "cost_registry": {"root": str((tmp_path / "cost").resolve()),
                             "entries": [{"run_root": str(pilot.parent.resolve()),
                                          "smoke_sha256": "old"}]},
           "pilot_manifest_sha256": score._hash(pilot / "manifest.json"),
           "reference_manifest_sha256": score._hash(reference / "manifest.json"),
           "pilot_summary_sha256": score._hash(pilot / "summary.json"),
           "reference_summary_sha256": score._hash(reference / "summary.json"),
           "pilot_seconds": 100, "reference_seconds": 100,
           "pilot_replay": {"seconds": 1}, "reference_replay": {"seconds": 1}}
    monkeypatch.setattr(score, "_data_hashes", lambda _: {})
    monkeypatch.setattr(score, "_verified_gate_receipt", lambda *args: old)
    monkeypatch.setattr(score, "gate", lambda *args, **kwargs:
                        pytest.fail("must reject before CUDA replay"))
    monkeypatch.setattr(score, "_recount_cost_registry", lambda *args, **kwargs:
                        {"entries": [{"run_root": str(pilot.parent.resolve()),
                                      "smoke_sha256": "changed"}]})
    with pytest.raises(RuntimeError, match="forged cost"):
        score.recount_full_gate("old", pilot, reference, "data", tmp_path / "cost",
                                full.parent, replay_device="cuda:0")
    (pilot / "manifest.json").write_text('{"fixed": false}')
    with pytest.raises(RuntimeError, match="stale pilot gate artifact"):
        score.recount_full_gate("old", pilot, reference, "data", tmp_path / "cost",
                                full.parent, replay_device="cuda:0")


def test_full_scoring_requires_fresh_root_bound_gate_and_unchanged_pilot(
        tmp_path, monkeypatch):
    full, pilot, reference = (tmp_path / name for name in ("full", "pilot", "reference"))
    for path in (full, pilot, reference):
        path.mkdir()
    for path in (pilot, reference):
        (path / "manifest.json").write_text("{}")
        (path / "summary.json").write_text("{}")
    old_gate = tmp_path / "old_gate.json"
    old_gate.write_text("{}")
    monkeypatch.setattr(score, "source", lambda: {"fixed": "source"})
    monkeypatch.setattr(score, "_scorer_identity", lambda: {"release_commit": "a" * 40})
    cost = {"run_root": str(full.resolve()), "mode": "full", "smoke_sha256": "abc",
            "fresh_gate_sha256": None, "fresh_gate_seconds": 0.}
    pilot_gate_cost = {"start_path": str(tmp_path / "cost" / "pilot_gate.start.json"),
                       "start_sha256": "start", "completion_sha256": "completed",
                       "elapsed_seconds": 2., "passed": True}
    receipt = {"status": "full_dispatch_fresh_integrity_pass",
               "full_run_root": str(full.resolve()),
               "source_sha256": score.source(), "scorer_identity": score._scorer_identity(),
               "data_files": {},
               "development_labels_accessed": False, "reference_equal": True,
               "method_named_nulls_equal": True, "ceiling_gpu_hours": 24,
               "projected_gpu_hours": 2., "old_gate_path": str(old_gate),
               "old_gate_sha256": score._hash(old_gate),
               "cost_registry": {"root": str(tmp_path / "cost"), "entries": [cost],
                                 "pilot_gate_receipts": [pilot_gate_cost]},
               "pilot_root": str(pilot), "reference_root": str(reference)}
    for label, root in (("pilot", pilot), ("reference", reference)):
        for artifact in ("manifest", "summary"):
            receipt[f"{label}_{artifact}_sha256"] = score._hash(root / f"{artifact}.json")
    gate_path = full / "fresh_full_gate.json"
    gate_path.write_text(json.dumps(receipt))
    monkeypatch.setattr(score, "_recount_cost_registry", lambda *args, **kwargs:
                        {"entries": [{**cost, "fresh_gate_sha256": "completion",
                                      "fresh_gate_seconds": 1.}],
                         "pilot_gate_receipts": [pilot_gate_cost]})
    assert score._verified_full_gate_receipt(gate_path, {}, full)["status"] == (
        "full_dispatch_fresh_integrity_pass")
    gate_path.write_text(json.dumps({**receipt, "scorer_identity": {
        "release_commit": "b" * 40}}))
    with pytest.raises(RuntimeError, match="missing or invalid fresh full gate"):
        score._verified_full_gate_receipt(gate_path, {}, full)
    gate_path.write_text(json.dumps(receipt))
    (pilot / "manifest.json").write_text('{"forged": true}')
    with pytest.raises(RuntimeError, match="pilot evidence changed"):
        score._verified_full_gate_receipt(gate_path, {}, full)
    (pilot / "manifest.json").write_text("{}")
    old_gate.write_text('{"forged": true}')
    with pytest.raises(RuntimeError, match="original pilot gate changed"):
        score._verified_full_gate_receipt(gate_path, {}, full)


def test_gate_never_opens_development_labels(monkeypatch):
    config = {**score.RECIPE, "seed": 6700, "pilot": True,
              "reference": False, "pretrained_sha256": score.WEIGHT_SHA}
    fake = {"pool": [{"sample_id": "test0", "location": "A"}],
            "manifest": {"split_sha256": "same"},
            "config": config, "summary": {"elapsed_seconds": 100},
            "events": [{"event": "started", "device": "Fake GPU"}],
            "replay": {"seconds": 10, "host": "same", "gpu_name": "Fake GPU"},
            "launch": {"host": "same", "gpu_uuid": "GPU-fake",
                       "release_commit": score.PILOT_RUNNER_RELEASE}}
    reference = copy.deepcopy(fake)
    reference["config"]["reference"] = True
    reference["launch"]["gpu_uuid"] = "GPU-another-free-card-same-host"
    monkeypatch.setattr(score, "_data_hashes", lambda _: {})
    monkeypatch.setattr(score, "_hash", lambda _: "synthetic-hash")
    monkeypatch.setattr(score, "audit_seed", lambda path, *args, **kwargs:
                        reference if kwargs.get("reference") else fake)
    monkeypatch.setattr(score, "_assert_equal_trajectory", lambda *args: None)
    monkeypatch.setattr(score, "_scorer_identity", lambda: {"release_commit": "a" * 40})
    monkeypatch.setattr(score, "_source_at_release", lambda _: score.source())
    monkeypatch.setattr(score, "_development_labels", lambda *args:
                        (_ for _ in ()).throw(AssertionError("labels touched by gate")))
    result = score.gate("pilot", "reference", "data", replay_device="cuda:0")
    assert result["development_labels_accessed"] is False
    assert result["projected_gpu_hours"] == pytest.approx(1540 / 3600)
    reference["launch"]["release_commit"] = "b" * 40
    with pytest.raises(RuntimeError, match="identical release bytes"):
        score.gate("pilot", "reference", "data", replay_device="cuda:0")
    fake["launch"]["release_commit"] = "b" * 40
    with pytest.raises(RuntimeError, match="pilot runner release source"):
        score.gate("pilot", "reference", "data", replay_device="cuda:0")


def test_full_block_requires_every_seed_before_opening_labels(tmp_path, monkeypatch):
    for seed in score.SEEDS[:-1]:
        (tmp_path / f"seed{seed}").mkdir()
    monkeypatch.setattr(score, "_development_labels", lambda *args:
                        (_ for _ in ()).throw(AssertionError("labels touched")))
    with pytest.raises(RuntimeError, match="incomplete"):
        score.main(tmp_path, "unused", "gate", replay_device="cuda:0")


def test_duplicate_pto_predictions_rejected_before_development_labels(tmp_path, monkeypatch):
    for seed in score.SEEDS:
        (tmp_path / f"seed{seed}").mkdir()
    monkeypatch.setattr(score, "_data_hashes", lambda _: {})
    monkeypatch.setattr(score, "_verified_full_gate_receipt", lambda *args:
                         {"split_sha256": "same", "pilot_launch": {
                             "release_commit": score.PILOT_RUNNER_RELEASE}})
    identical = torch.full((2, 3), 1 / 3)
    monkeypatch.setattr(score, "audit_seed", lambda *args, **kwargs: {
        "launch": {"release_commit": "a" * 40},
        "manifest": {"split_sha256": "same"},
        "snapshots": {"ce_null": {e: identical for e in score.SNAPSHOTS}}})
    monkeypatch.setattr(score, "_development_labels", lambda *args:
                         pytest.fail("labels opened before duplicate PTO check"))
    monkeypatch.setattr(score, "_scorer_identity", lambda: {"release_commit": "a" * 40})
    monkeypatch.setattr(score, "_source_at_release", lambda _: score.source())
    with pytest.raises(RuntimeError, match="duplicate CE/null"):
        score.main(tmp_path, "data", "gate", replay_device="cuda:0")


def test_actual_cost_includes_completed_full_gate_lease_and_replay(tmp_path, monkeypatch):
    dispatch_seconds = 200.
    completed_seconds = 237.5  # Fresh gate elapsed time includes lease handoff.
    gate = {"cost_registry": {"root": str(tmp_path / "cost"),
                              "gpu_seconds": dispatch_seconds},
            "prior_gpu_hours": dispatch_seconds / 3600,
            "pilot_root": str(tmp_path / "pilot" / "seed6700"),
            "reference_root": str(tmp_path / "ref" / "seed6700_ref"),
            "pilot_seconds": 100., "reference_seconds": 110.}
    audited = [{"summary": {"elapsed_seconds": 120.},
                "replay": {"seconds": 2.}}]
    monkeypatch.setattr(score, "_recount_cost_registry", lambda *args, **kwargs:
                        {"gpu_seconds": completed_seconds})
    actual, _ = score._actual_study_hours(gate, audited, tmp_path / "full")
    assert actual == pytest.approx((completed_seconds + 100 + 110 + 120 + 2) / 3600)


def test_eight_unique_holm_contrasts_and_fresh_output(tmp_path, monkeypatch):
    for seed in score.SEEDS:
        (tmp_path / f"seed{seed}").mkdir()
    monkeypatch.setattr(score, "_data_hashes", lambda _: {})
    monkeypatch.setattr(score, "_verified_full_gate_receipt",
                        lambda *args: {"split_sha256": "same", "prior_gpu_hours": 202 / 3600,
                                       "cost_registry": {"root": str(tmp_path / "cost"),
                                                         "gpu_seconds": 202},
                                       "pilot_root": str(tmp_path / "pilot" / "seed6700"),
                                       "reference_root": str(tmp_path / "ref" / "seed6700_ref"),
                                       "pilot_seconds": 100, "reference_seconds": 100,
                                       "pilot_replay": {"seconds": 1},
                                       "reference_replay": {"seconds": 1},
                                        "pilot_launch": {"release_commit":
                                                         score.PILOT_RUNNER_RELEASE},
                                       "ceiling_gpu_hours": 24})
    monkeypatch.setattr(score, "_recount_cost_registry", lambda *args, **kwargs:
                        {"gpu_seconds": 202, "entries": [], "pilot_gate_receipts": []})
    monkeypatch.setattr(score, "_development_labels", lambda *args: [1])
    monkeypatch.setattr(score, "source", lambda: {})
    monkeypatch.setattr(score, "_scorer_identity", lambda: {"release_commit": "a" * 40})
    monkeypatch.setattr(score, "_source_at_release", lambda _: score.source())
    def fake_audit(path, *args, **kwargs):
        index = int(Path(path).name.removeprefix("seed")) - score.SEEDS[0]
        pto = torch.full((1, 2), float(index))
        return {"seed": int(Path(path).name.removeprefix("seed")),
                  "manifest": {"split_sha256": "same"}, "pool": [{"sample_id": "test0"}],
                  "replay": {"seconds": 1}, "summary": {"elapsed_seconds": 100},
                  "snapshots": {"ce_null": {epoch: pto for epoch in score.SNAPSHOTS}},
                 "launch": {"release_commit": "a" * 40}}
    monkeypatch.setattr(score, "audit_seed", fake_audit)
    def fake_score(seed, labels):
        idx = seed["seed"] - score.SEEDS[0]
        arms = {"ce_null": .4 + idx * .001, "focal_clip": .41 + idx * .001,
                "cap10_tralo": .42 + idx * .0015, "cap10_phr": .39 + idx * .0005,
                "cap20_tralo": .38 + idx * .0015, "cap20_phr": .37 + idx * .0005}
        return {"seed": seed["seed"], "caps": {
            key: {"arms": {name: {"allocated": {metric: value for metric in score.base.METRICS},
                                   "prediction_sha256": f"{idx}-{name}"}
                           for name, value in arms.items()}}
            for key in ("10", "20")}}
    monkeypatch.setattr(score, "_score_audited", fake_score)
    target = tmp_path / "scores.json"
    report = score.main(tmp_path, "data", "gate", target, replay_device="cuda:0")
    assert len(report["primary_family"]) == 8
    assert len(set(report["primary_family"])) == 8
    assert report["actual_study_gpu_hours"] == pytest.approx((202 + 200 + 12 * 101) / 3600)
    assert all(len(row["per_seed"]) == 12 and 0 <= row["holm_p"] <= 1
               for row in report["primary_contrasts"].values())
    assert json.loads(target.read_text())["primary_family"] == report["primary_family"]
    with pytest.raises(FileExistsError):
        score.main(tmp_path, "data", "gate", target, replay_device="cuda:0")


def test_scorer_consumes_runner_seven_epoch_fixture(tmp_path, monkeypatch):
    """Exercise the real runner's artifact schema with a tiny CPU fixture.

    The runner fixture uses synthetic correction kernels; the independent
    mathematical correction audit has separate mutation tests above and is
    suppressed only for those synthetic treated rows here.
    """
    path = Path(__file__).with_name("test_fmow_persistent_local.py")
    spec = importlib.util.spec_from_file_location("persistent_runner_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The runner's CPU-only fixture bypasses CUDA; this patch is limited to
    # that fixture while its production path continues to require CUDA.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    module.test_seven_epoch_cpu_fixture_keeps_focal_and_persistent_branches_separate(
        monkeypatch, tmp_path)
    root, reference = tmp_path / "run", tmp_path / "reference"
    manifest = json.loads((root / "manifest.json").read_text())
    monkeypatch.setattr(score, "_split_and_pool", lambda _: (manifest["split"],
        [{"sample_id": row["sample_id"], "location": row["location"]}
         for row in manifest["pool_rows"]]))
    monkeypatch.setattr(score, "TRAIN_COUNT", 4)
    monkeypatch.setattr(score, "POOL_COUNT", 3)
    monkeypatch.setattr(score, "WEIGHT_SHA", "0" * 64)
    monkeypatch.setattr(score.base, "CLASSES", 2)
    monkeypatch.setattr(score.base, "_quotas", lambda groups: manifest["quotas"])
    original = score._audit_correction

    def synthetic_correction(row, quota, method, previous):
        if method == "null":
            return original(row, quota, method, previous)
        return row["dual_after"] if method == "phr" else None

    monkeypatch.setattr(score, "_audit_correction", synthetic_correction)
    full = score.audit_seed(root, tmp_path, data_hashes=manifest["data_files"],
                            expected_seed=6700)
    ref = score.audit_seed(reference, tmp_path, data_hashes=manifest["data_files"],
                           expected_seed=6700, reference=True)
    assert set(full["snapshots"]) == set(score.ARMS + score.PILOT_NULLS)
    assert set(ref["snapshots"]) == {"ce_null"}
    score._assert_equal_trajectory(full, ref)
    mutated = copy.deepcopy(full["summary"])
    events = copy.deepcopy(full["events"])
    mutated["arms"]["cap10_tralo"]["epochs"][2]["pre_epoch_model_sha256"] = "0" * 64
    event = next(row for row in events if row.get("event") == "epoch" and
                 row.get("arm") == "cap10_tralo" and row.get("epoch") == 3)
    event["pre_epoch_model_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="persistent model/optimizer hash continuity"):
        score._audit_arm(root, "cap10_tralo", mutated, events,
                         [row["location"] for row in manifest["pool_rows"]],
                         manifest["quotas"],
                         [row["sample_id"] for row in manifest["pool_rows"]])
    missing = copy.deepcopy(full["summary"])
    missing_events = copy.deepcopy(full["events"])
    missing["arms"]["cap10_tralo"]["corrections"][0].pop("pre_correction_snapshot")
    event = next(row for row in missing_events if row.get("event") == "correction" and
                 row.get("arm") == "cap10_tralo" and row.get("epoch") == 2)
    event.pop("pre_correction_snapshot")
    with pytest.raises(RuntimeError, match="missing pre-correction replay evidence"):
        score._audit_arm(root, "cap10_tralo", missing, missing_events,
                         [row["location"] for row in manifest["pool_rows"]],
                         manifest["quotas"],
                         [row["sample_id"] for row in manifest["pool_rows"]])
