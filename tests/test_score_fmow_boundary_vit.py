"""Protocol, provenance, denominator and label-boundary tests for ViT scorer."""

import copy
import json

import pytest
import torch

from analysis import score_fmow_boundary_vit as score


def _config(seed=6500, steps=True):
    return {**score.RECIPE, "seed": seed, "snapshot_steps": steps,
            "study": score.STUDY, "step_radius": .1, "alm_rho": .5}


def test_input_config_bytes_bound_to_vit_protocol(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    monkeypatch.setattr(score, "CONFIGS", configs)
    config = _config()
    path = configs / "fmow_local_6500_step.json"
    path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
    started = {"config_sha256": score.base.sha256(path)}
    score._config_and_provenance(tmp_path / "seed6500", config, started)
    path.write_text(json.dumps(config, sort_keys=True) + "\n\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="input/run config provenance"):
        score._config_and_provenance(tmp_path / "seed6500", config, started)


@pytest.mark.parametrize("field,value", [
    ("backbone", "mobilenet_v3_large"), ("study", "local_boundary_v1"),
    ("seed", 6400), ("batch_size", 32), ("development_batch_size", 16),
    ("step_radius", .2), ("snapshot_steps", 1),
])
def test_mismatched_mobile_or_violation_of_fixed_vit_recipe_rejected(tmp_path, field, value):
    config = _config()
    config[field] = value
    with pytest.raises(RuntimeError, match="fixed ViT boundary protocol"):
        score._config_and_provenance(tmp_path / "seed6500", config, {})


def test_step_off_only_allowed_for_pilot(tmp_path):
    config = _config(seed=6501, steps=False)
    with pytest.raises(RuntimeError, match="fixed ViT boundary protocol"):
        score._config_and_provenance(tmp_path / "seed6501_ref", config, {})


def test_memory_smoke_bound_to_release_host_gpu_and_bytes(tmp_path, monkeypatch):
    config = _config()
    d = tmp_path / "seed6500"
    receipt = tmp_path / "vit_memory_smoke.json"
    smoke_launch_path = tmp_path / "vit_memory_smoke.launch.json"
    complete_path = tmp_path / "vit_memory_smoke.complete.json"
    generator = tmp_path / "smoke.py"
    generator.write_text("# fixed generator\n")
    monkeypatch.setattr(score, "SMOKE_GENERATOR", generator)
    launch = {"release_commit": "a" * 40, "host": "dsisco02", "gpu_index": 3,
              "gpu_uuid": "GPU-123", "memory_smoke_receipt_path": str(receipt),
              "memory_smoke_execution": "queue_executed",
              "memory_smoke_launch_path": str(smoke_launch_path),
              "memory_smoke_complete_path": str(complete_path),
              "started_utc": "1970-01-01T00:00:04+00:00"}
    smoke_launch = {"release_commit": launch["release_commit"],
                    "host": launch["host"], "gpu_index": 3,
                    "gpu_uuid": launch["gpu_uuid"], "receipt_path": str(receipt),
                    "generator_sha256": score.base.sha256(generator),
                    "started_utc": "1970-01-01T00:00:00+00:00"}
    smoke_launch_path.write_text(json.dumps(smoke_launch) + "\n", encoding="utf-8")
    launch["memory_smoke_launch_sha256"] = score.base.sha256(smoke_launch_path)
    complete = {"exit_code": 0, "release_commit": launch["release_commit"],
                "host": launch["host"], "gpu_uuid": launch["gpu_uuid"],
                "receipt_path": str(receipt), "ended_utc": "1970-01-01T00:00:03+00:00"}
    complete_path.write_text(json.dumps(complete) + "\n", encoding="utf-8")
    launch["memory_smoke_complete_sha256"] = score.base.sha256(complete_path)
    smoke = {"release_commit": launch["release_commit"], "host": launch["host"],
             "gpu_uuid": launch["gpu_uuid"], "backbone": "vit_b_16",
             "batch_size": 16, "development_batch_size": 8,
             "weight_sha256": score.WEIGHT_SHA256, "precision": "fp32",
             "label_free": True, "memory_smoke_passed": True,
             "peak_allocated_bytes": 100, "total_memory_bytes": 1000,
             "source_sha256": score.base.source(), "device_name": "GPU",
             "data_files": score.base.FILES, "gpu_index": 3,
             "started_utc": 1., "ended_utc": 2.,
             "development_pool_count": 1673,
             "smoke_generator_sha256": score.base.sha256(generator),
             "phases": {
                 "train_backward": {"completed": True, "seconds": .1,
                                    "peak_allocated_bytes": 100, "loss": 1.,
                                    "gradient_norm": 2., "optimizer_step": True,
                                    "input_shape": [16, 3, 224, 224]},
                 "development_inference": {"completed": True, "seconds": .2,
                                           "peak_allocated_bytes": 90,
                                           "probabilities_shape": [1673, 8],
                                           "finite_rows": 1673, "row_sum_max_error": 0.},
                 "side_copy_constraint_gradient": {"completed": True, "seconds": .3,
                                                   "peak_allocated_bytes": 95,
                                                   "pto_unchanged": True,
                                                   "caps": {str(d): {
                                                       "joint_gradient_norm": 1.,
                                                       "phr_gradient_norm": 1.,
                                                       "joint_applied": False,
                                                       "phr_applied": False,
                                                       "all_four_arms": True,
                                                       "scope_derivatives_finite": True,
                                                       "pto_unchanged": True}
                                                            for d in score.base.DIVISORS}}}}
    receipt.write_text(json.dumps(smoke) + "\n", encoding="utf-8")
    launch["memory_smoke_receipt_sha256"] = score.base.sha256(receipt)
    score._memory_smoke(d, config, launch, {"device": "GPU"})
    launch["started_utc"] = "1970-01-01T00:00:01+00:00"
    with pytest.raises(RuntimeError, match="did not complete before queue launch"):
        score._memory_smoke(d, config, launch, {"device": "GPU"})
    launch["started_utc"] = "1970-01-01T00:00:04+00:00"
    changed = copy.deepcopy(smoke)
    changed["gpu_uuid"] = "GPU-foreign"
    receipt.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash/path"):
        score._memory_smoke(d, config, launch, {"device": "GPU"})
    launch["memory_smoke_receipt_sha256"] = score.base.sha256(receipt)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        score._memory_smoke(d, config, launch, {"device": "GPU"})
    launch["memory_smoke_execution"] = "supplied_external_receipt"
    with pytest.raises(RuntimeError, match="queue-owned"):
        score._memory_smoke(d, config, launch, {"device": "GPU"})


def test_queue_smoke_completion_must_be_successful_and_precede_job(tmp_path, monkeypatch):
    generator = tmp_path / "smoke.py"
    generator.write_text("# fixed generator\n")
    monkeypatch.setattr(score, "SMOKE_GENERATOR", generator)
    receipt = tmp_path / "vit_memory_smoke.json"
    smoke_launch_path = tmp_path / "vit_memory_smoke.launch.json"
    complete_path = tmp_path / "vit_memory_smoke.complete.json"
    launch = {"release_commit": "a" * 40, "host": "dsisco02", "gpu_index": 3,
              "gpu_uuid": "GPU-123", "memory_smoke_receipt_path": str(receipt),
              "memory_smoke_execution": "queue_executed",
              "memory_smoke_launch_path": str(smoke_launch_path),
              "memory_smoke_complete_path": str(complete_path),
              "started_utc": "1970-01-01T00:00:04+00:00"}
    smoke_launch = {"release_commit": launch["release_commit"],
                    "host": launch["host"], "gpu_index": 3,
                    "gpu_uuid": launch["gpu_uuid"], "receipt_path": str(receipt),
                    "generator_sha256": score.base.sha256(generator),
                    "started_utc": "1970-01-01T00:00:00+00:00"}
    smoke_launch_path.write_text(json.dumps(smoke_launch) + "\n", encoding="utf-8")
    launch["memory_smoke_launch_sha256"] = score.base.sha256(smoke_launch_path)
    complete = {"exit_code": 7, "release_commit": launch["release_commit"],
                "host": launch["host"], "gpu_uuid": launch["gpu_uuid"],
                "receipt_path": str(receipt), "ended_utc": "1970-01-01T00:00:03+00:00"}
    complete_path.write_text(json.dumps(complete) + "\n", encoding="utf-8")
    launch["memory_smoke_complete_sha256"] = score.base.sha256(complete_path)
    receipt.write_text("{}\n", encoding="utf-8")
    launch["memory_smoke_receipt_sha256"] = score.base.sha256(receipt)
    with pytest.raises(RuntimeError, match="queue completion identity"):
        score._memory_smoke(tmp_path / "seed6500", _config(), launch, {"device": "GPU"})


def _real_fixture(root, monkeypatch):
    generator = root / "real.py"
    generator.write_text("# immutable numerical preflight\n", encoding="utf-8")
    monkeypatch.setattr(score, "REAL_PREFLIGHT_GENERATOR", generator)
    names = {kind: root / f"vit_real_preflight.{suffix}" for kind, suffix in
             (("launch", "launch.json"), ("receipt", "json"),
              ("complete", "complete.json"))}
    artifacts = root / "vit_real_preflight.artifacts"
    artifacts.mkdir()
    hashes = {}
    for arm in score.boundary.ARMS:
        path = artifacts / f"epoch01_{arm}.pt"
        path.write_bytes(arm.encode())
        hashes[path.name] = score.base.sha256(path)
    release, host, uuid = "a" * 40, "dsisco02", "GPU-123"
    queued = {"release_commit": release, "host": host, "gpu_uuid": uuid,
              "gpu_index": 3, "receipt_path": str(names["receipt"]),
              "generator_sha256": score.base.sha256(generator),
              "started_utc": "1970-01-01T00:00:00+00:00"}
    complete = {"release_commit": release, "host": host, "gpu_uuid": uuid,
                "receipt_path": str(names["receipt"]), "exit_code": 0,
                "ended_utc": "1970-01-01T00:00:03+00:00"}
    scopes = ("pooled", *sorted(score.base.COUNTRIES))
    finite = {f"{scope}@{epsilon}": {"analytic": 1., "numeric": 1.,
                                      "error": 0., "tolerance": .053}
              for scope in scopes for epsilon in ("0.01", "0.02")}
    receipt = {"release_commit": release, "host": host, "gpu_uuid": uuid,
               "gpu_index": 3, "source_sha256": score.base.source(),
               "data_file_sha256": score.base.FILES, "precision": "fp32",
               "backbone": "vit_b_16", "development_labels_accessed": False,
               "preflight_passed": True, "images_count": 15,
               "weight_sha256": score.WEIGHT_SHA256, "device_name": "GPU",
               "development_batch_size": 8, "chunk_sizes": [8, 7],
               "preflight_generator_sha256": score.base.sha256(generator),
               "country_counts": {country: 3 for country in sorted(score.base.COUNTRIES)},
               "pto_unchanged": True,
               "pretrained_weight": {"file": "/cache/vit_b_16-c867db91.pth",
                                     "sha256": score.WEIGHT_SHA256},
               "started_utc": 1., "ended_utc": 2.,
               "max_probability_difference": 1e-7,
               "gradient_relative_errors": dict.fromkeys(scopes, .001),
               "finite_differences": finite,
               "arms_audited": list(score.boundary.ARMS),
               "artifact_sha256": hashes}
    for kind, value in (("launch", queued), ("receipt", receipt),
                        ("complete", complete)):
        names[kind].write_text(json.dumps(value) + "\n", encoding="utf-8")
    launch = {"release_commit": release, "host": host, "gpu_uuid": uuid,
              "gpu_index": 3, "started_utc": "1970-01-01T00:00:04+00:00",
              "real_preflight_execution": "queue_executed"}
    for kind, path in names.items():
        launch[f"real_preflight_{kind}_path"] = str(path)
        launch[f"real_preflight_{kind}_sha256"] = score.base.sha256(path)
    return launch, receipt, names, artifacts


def test_real_image_preflight_bound_and_measured(tmp_path, monkeypatch):
    launch, receipt, names, _ = _real_fixture(tmp_path, monkeypatch)
    assert score._real_preflight(tmp_path / "seed6500", launch) == 3.
    launch["real_preflight_execution"] = "supplied_external_receipt"
    with pytest.raises(RuntimeError, match="not queue executed"):
        score._real_preflight(tmp_path / "seed6500", launch)
    launch["real_preflight_execution"] = "queue_executed"
    names["complete"].write_text(json.dumps({**json.loads(names["complete"].read_text()),
                                               "exit_code": 1}))
    launch["real_preflight_complete_sha256"] = score.base.sha256(names["complete"])
    with pytest.raises(RuntimeError, match="queue completion"):
        score._real_preflight(tmp_path / "seed6500", launch)


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(source_sha256="wrong"), "identity/label"),
    (lambda r: r.update(weight_sha256="0" * 64), "identity/label"),
    (lambda r: r.update(development_batch_size=3), "identity/label"),
    (lambda r: r.update(chunk_sizes=[3, 3, 3, 3, 3]), "identity/label"),
    (lambda r: r.update(development_labels_accessed=True), "identity/label"),
    (lambda r: r.update(max_probability_difference=1e-4), "probability parity"),
    (lambda r: r["gradient_relative_errors"].update(DZA=.1), "gradient parity"),
    (lambda r: r["finite_differences"].pop("DZA@0.01"), "scopes incomplete"),
    (lambda r: r["finite_differences"]["DZA@0.01"].update(numeric=3.),
     "failed recount"),
    (lambda r: r["artifact_sha256"].update({"epoch01_joint.pt": "0" * 64}),
     "artifact hash"),
])
def test_real_preflight_mutations_rejected_before_labels(
        tmp_path, monkeypatch, mutation, match):
    launch, receipt, names, _ = _real_fixture(tmp_path, monkeypatch)
    mutation(receipt)
    names["receipt"].write_text(json.dumps(receipt) + "\n")
    launch["real_preflight_receipt_sha256"] = score.base.sha256(names["receipt"])
    monkeypatch.setattr(score.alm, "_manifest_and_labels",
                        lambda *_args: pytest.fail("labels opened before preflight"))
    with pytest.raises(RuntimeError, match=match):
        score._real_preflight(tmp_path / "seed6500", launch)


@pytest.mark.parametrize("mutation,match", [
    (lambda s: s["phases"].pop("train_backward"), "phases absent"),
    (lambda s: s["phases"]["train_backward"].update(seconds=0), "not measured"),
    (lambda s: s["phases"]["train_backward"].update(optimizer_step=False), "train-backward"),
    (lambda s: s["phases"]["development_inference"].update(finite_rows=2), "development-inference"),
    (lambda s: s["phases"]["development_inference"].update(row_sum_max_error=.1),
     "development-inference"),
    (lambda s: s["phases"]["side_copy_constraint_gradient"].update(pto_unchanged=False), "side-copy"),
    (lambda s: s["phases"]["side_copy_constraint_gradient"]["caps"]["10"].update(
        joint_gradient_norm=0), "cap10"),
    (lambda s: s["phases"]["side_copy_constraint_gradient"]["caps"]["20"].update(
        all_four_arms=False), "cap20"),
    (lambda s: s.update(smoke_generator_sha256="0" * 64), "identity mismatch"),
    (lambda s: s.update(source_sha256={}), "identity mismatch"),
    (lambda s: s.update(device_name="other GPU"), "identity mismatch"),
    (lambda s: s.update(label_free=False), "identity mismatch"),
    (lambda s: s.update(data_files={}), "identity mismatch"),
    (lambda s: s.update(peak_allocated_bytes=101), "peak recount"),
])
def test_memory_smoke_rejects_missing_or_fabricated_phase_evidence(
        tmp_path, monkeypatch, mutation, match):
    # Start from the valid receipt produced by the preceding unit fixture's
    # equivalent schema, then recompute its hash to catch content fabrication.
    generator = tmp_path / "smoke.py"
    generator.write_text("# fixed generator\n")
    monkeypatch.setattr(score, "SMOKE_GENERATOR", generator)
    receipt = tmp_path / "vit_memory_smoke.json"
    smoke_launch_path = tmp_path / "vit_memory_smoke.launch.json"
    complete_path = tmp_path / "vit_memory_smoke.complete.json"
    launch = {"release_commit": "a" * 40, "host": "dsisco02", "gpu_index": 3,
              "gpu_uuid": "GPU-123", "memory_smoke_receipt_path": str(receipt),
              "memory_smoke_execution": "queue_executed",
              "memory_smoke_launch_path": str(smoke_launch_path),
              "memory_smoke_complete_path": str(complete_path),
              "started_utc": "1970-01-01T00:00:04+00:00"}
    smoke_launch = {"release_commit": launch["release_commit"],
                    "host": launch["host"], "gpu_index": 3,
                    "gpu_uuid": launch["gpu_uuid"], "receipt_path": str(receipt),
                    "generator_sha256": score.base.sha256(generator),
                    "started_utc": "1970-01-01T00:00:00+00:00"}
    smoke_launch_path.write_text(json.dumps(smoke_launch) + "\n", encoding="utf-8")
    launch["memory_smoke_launch_sha256"] = score.base.sha256(smoke_launch_path)
    complete = {"exit_code": 0, "release_commit": launch["release_commit"],
                "host": launch["host"], "gpu_uuid": launch["gpu_uuid"],
                "receipt_path": str(receipt), "ended_utc": "1970-01-01T00:00:03+00:00"}
    complete_path.write_text(json.dumps(complete) + "\n", encoding="utf-8")
    launch["memory_smoke_complete_sha256"] = score.base.sha256(complete_path)
    phases = {
        "train_backward": {"completed": True, "seconds": .1,
                           "peak_allocated_bytes": 100, "loss": 1.,
                           "gradient_norm": 2., "optimizer_step": True,
                           "input_shape": [16, 3, 224, 224]},
        "development_inference": {"completed": True, "seconds": .2,
                                  "peak_allocated_bytes": 90,
                                  "probabilities_shape": [1673, 8],
                                  "finite_rows": 1673, "row_sum_max_error": 0.},
        "side_copy_constraint_gradient": {"completed": True, "seconds": .3,
                                          "peak_allocated_bytes": 95,
                                          "pto_unchanged": True,
                                          "caps": {str(d): {
                                              "joint_gradient_norm": 1.,
                                              "phr_gradient_norm": 1.,
                                              "joint_applied": False,
                                              "phr_applied": False,
                                              "all_four_arms": True,
                                              "scope_derivatives_finite": True,
                                              "pto_unchanged": True}
                                                   for d in score.base.DIVISORS}}}
    smoke = {"release_commit": launch["release_commit"], "host": launch["host"],
             "gpu_uuid": launch["gpu_uuid"], "backbone": "vit_b_16",
             "batch_size": 16, "development_batch_size": 8,
             "weight_sha256": score.WEIGHT_SHA256, "precision": "fp32",
             "label_free": True, "memory_smoke_passed": True,
             "peak_allocated_bytes": 100, "total_memory_bytes": 1000,
             "source_sha256": score.base.source(), "device_name": "GPU",
             "data_files": score.base.FILES, "gpu_index": 3,
             "started_utc": 1., "ended_utc": 2.,
             "development_pool_count": 1673,
             "smoke_generator_sha256": score.base.sha256(generator),
             "phases": phases}
    mutation(smoke)
    receipt.write_text(json.dumps(smoke) + "\n", encoding="utf-8")
    launch["memory_smoke_receipt_sha256"] = score.base.sha256(receipt)
    with pytest.raises(RuntimeError, match=match):
        score._memory_smoke(tmp_path / "seed6500", _config(), launch, {"device": "GPU"})


def test_pretrained_checkpoint_and_transform_provenance(tmp_path):
    config = _config()
    summary = {"seed": 6500, "quotas": "fixed", "initial_sha256": "init"}
    started = {"quotas": "fixed"}
    init = {"initial_sha256": "init", "architecture": "vit_b_16",
            "classes": score.base.CLASSES, "transform": score.TRANSFORM,
            "pretrained_weight": {"file": "/cache/vit_b_16-c867db91.pth",
                                  "sha256": score.WEIGHT_SHA256}}
    score._model_identity(tmp_path, config, summary, started, init)
    changed = copy.deepcopy(init)
    changed["pretrained_weight"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="ViT model"):
        score._model_identity(tmp_path, config, summary, started, changed)
    changed = copy.deepcopy(init)
    changed["transform"] = "center crop"
    with pytest.raises(RuntimeError, match="ViT model"):
        score._model_identity(tmp_path, config, summary, started, changed)


def test_fixed_twelve_seed_denominator_checked_before_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(score.alm, "_manifest_and_labels",
                        lambda *_args: pytest.fail("labels opened before denominator"))
    with pytest.raises(RuntimeError, match="ViT block incomplete"):
        score.main(tmp_path, tmp_path)


def test_pilot_gate_does_not_open_labels(tmp_path, monkeypatch):
    pilot, ref = tmp_path / "pilot", tmp_path / "ref"
    (pilot / "seed6500").mkdir(parents=True)
    (ref / "seed6500_ref").mkdir(parents=True)
    monkeypatch.setattr(score.alm, "_manifest_and_labels",
                        lambda *_args: pytest.fail("pilot gate opened labels"))
    config = _config()
    reference = _config(steps=False)
    quota = {str(d): {"global_cap": 1, "local_caps": {"A": 1}}
             for d in score.base.DIVISORS}
    result = {"epochs_run": 1, "best_epoch": 1}
    steps = {"1": {str(d): {"joint": {"applied": False}}
                    for d in score.base.DIVISORS}}
    started = {"quotas": quota, "device": "GPU", "precision": "fp32"}
    launch = {"host": "dsisco02", "gpu_uuid": "GPU-123", "data_root": str(tmp_path),
              "release_commit": "a" * 40, "_smoke_seconds": 3.,
              "_preflight_seconds": 4.}
    epoch = {"event": "epoch", "epoch": 1, "hard_counts": {"1": 1}, "stop_loss": 1.}

    def receipt(directory):
        is_ref = directory.name.endswith("_ref")
        cfg = reference if is_ref else config
        summary = {"retrain": result, "initial_sha256": "same",
                   "pto_sha256": "same", "steps": {} if is_ref else steps}
        events = [epoch, {"event": "completed", "seconds": 90. if is_ref else 100.}]
        return (cfg, summary, started, events, ["test0"], ["A"], launch)

    monkeypatch.setattr(score, "_receipt", receipt)
    monkeypatch.setattr(score.base, "sha256", lambda *_args: "same")
    monkeypatch.setattr(score.boundary, "_data_bytes", lambda *_args: None)
    monkeypatch.setattr(score.base, "_probabilities", lambda *_args: torch.ones(1, 1))
    monkeypatch.setattr(score.boundary, "_steps", lambda *_args: ({}, {}, {}))
    receipt = tmp_path / "pilot_gate.json"
    gate = score.gate(pilot, ref, receipt)
    assert gate["development_labels_accessed"] is False
    assert json.loads(receipt.read_text()) == gate
    with pytest.raises(FileExistsError):
        score.gate(pilot, ref, receipt)
    assert gate["projected_total_gpu_hours"] == pytest.approx(
        (1.25 * (13 * 100 + 90) + 3 + 3 + 3 + 4 + 4 + 4) / 3600)


def test_full_contrasts_keep_viT_seeds_and_six_test_family(tmp_path, monkeypatch):
    (tmp_path / "vit_pilot_gate_recheck.json").write_text("{}\n")
    for seed in score.SEEDS:
        d = tmp_path / f"seed{seed}"
        d.mkdir()
        (d / "manifest.json").write_text("same fixed manifest")
    monkeypatch.setattr(score.boundary, "_data_bytes", lambda *_args: None)
    monkeypatch.setattr(score, "_receipt", lambda *_args: (None,) * 6 +
                        ({"release_commit": "a" * 40},))
    monkeypatch.setattr(score.boundary, "_steps", lambda *_args: None)
    monkeypatch.setattr(score, "_full_pilot_gate", lambda *_args: {})
    monkeypatch.setattr(score, "_full_cost_gate", lambda *_args: {})

    def row(directory, *_args, **_kwargs):
        seed = int(directory.name.removeprefix("seed"))
        offset = (seed - score.SEEDS[0]) * .001
        arms = {name: {"allocated": {metric: .4 + offset + shift
                                    for metric in score.base.METRICS},
                       "prediction_sha256": f"{seed}-{name}"}
                for name, shift in (("ens_pto", 0.), ("ens_joint", .02),
                                    ("ens_sham", 0.), ("ens_phr_local", -.01))}
        return {"manifest_sha256": "same", "release_commit": "a" * 40,
                "caps": {"10": {"arms": arms}, "20": {"arms": arms}}}

    monkeypatch.setattr(score, "load_seed", row)
    report = score.main(tmp_path, tmp_path)
    assert len(report["seeds"]) == 12
    assert len(report["primary_family"]) == 6
    assert set(report["contrasts"]["cc_f1"]) == set(report["primary_family"])
    assert set(report["contrasts"]["cc_f1"]["cap_divisor_10_joint_minus_ens_pto"]["per_seed"]) == set(score.SEEDS)
    assert report["contrasts"]["cc_f1"]["cap_divisor_10_joint_minus_ens_pto"]["mean"] == pytest.approx(.02)


def test_full_pilot_gate_recomputes_instead_of_trusting_success_flag(tmp_path, monkeypatch):
    path = tmp_path / "vit_pilot_gate_recheck.json"
    saved = {"status": "vit_pilot_integrity_pass", "development_labels_accessed": False,
             "release_commit": "a" * 40, "host": "dsisco02",
             "pilot_step_root": "/runs/step", "pilot_ref_root": "/runs/ref"}
    path.write_text(json.dumps(saved) + "\n")
    launch = {"release_commit": "a" * 40, "host": "dsisco02",
              "pilot_gate_receipt_path": str(path),
              "pilot_gate_receipt_sha256": score.base.sha256(path)}
    audited = [((None,) * 6 + (launch,), None)]
    actual_audit = copy.deepcopy(saved)
    monkeypatch.setattr(score, "gate", lambda *_args: copy.deepcopy(actual_audit))
    assert score._full_pilot_gate(tmp_path, audited) == saved
    saved["pto_equal"] = False
    path.write_text(json.dumps(saved) + "\n")
    launch["pilot_gate_receipt_sha256"] = score.base.sha256(path)
    with pytest.raises(RuntimeError, match="fresh label-blind audit"):
        score._full_pilot_gate(tmp_path, audited)


def test_full_cost_gate_recounts_jobs_and_smoke_before_labels(tmp_path):
    step, ref = tmp_path / "pilot_step", tmp_path / "pilot_ref"
    step.mkdir()
    ref.mkdir()
    for root, job, seconds in ((step, "6500_step", 100), (ref, "6500_ref", 90)):
        (root / f"seed{job}.launch.json").write_text(json.dumps({
            "release_commit": "a" * 40, "host": "dsisco02",
            "started_utc": "1970-01-01T00:00:00+00:00"}))
        (root / f"seed{job}.complete.json").write_text(json.dumps({
            "release_commit": "a" * 40, "host": "dsisco02", "exit_code": 0,
            "ended_utc": f"1970-01-01T00:{seconds // 60:02d}:{seconds % 60:02d}+00:00"}))
    gate_path = tmp_path / "vit_pilot_gate_recheck.json"
    gate_path.write_text("{}\n")
    cost_path = tmp_path / "vit_cost_gate.json"
    pilot_gate = {"pilot_step_root": str(step), "pilot_ref_root": str(ref),
                  "pilot_smoke_seconds": [3., 4.],
                  "pilot_preflight_seconds": [2., 3.]}
    projected = (1.25 * (13 * 100 + 90) + 3 + 4 + 5 + 2 + 3 + 4) / 3600
    cost = {"release_commit": "a" * 40, "host": "dsisco02",
            "ceiling_gpu_hours": 24.0, "gate_passed": True,
            "pilot_step_root": str(step), "pilot_ref_root": str(ref),
            "full_root": str(tmp_path.resolve()),
            "pilot_gate_receipt_path": str(gate_path),
            "pilot_gate_receipt_sha256": score.base.sha256(gate_path),
            "pilot_step_seconds": 100., "pilot_ref_seconds": 90.,
            "smoke_seconds": {"pilot_step": 3., "pilot_ref": 4., "full": 5.},
            "preflight_seconds": {"pilot_step": 2., "pilot_ref": 3., "full": 4.},
            "projected_gpu_hours": projected}
    cost_path.write_text(json.dumps(cost) + "\n")
    launch = {"release_commit": "a" * 40, "host": "dsisco02",
              "_smoke_seconds": 5., "_preflight_seconds": 4.,
              "cost_gate_receipt_path": str(cost_path),
              "cost_gate_receipt_sha256": score.base.sha256(cost_path)}
    audited = [((None,) * 6 + (launch,), None)]
    assert score._full_cost_gate(tmp_path, audited, pilot_gate) == cost
    cost["projected_gpu_hours"] = 0.001
    cost_path.write_text(json.dumps(cost) + "\n")
    launch["cost_gate_receipt_sha256"] = score.base.sha256(cost_path)
    with pytest.raises(RuntimeError, match="projection exceeds or differs"):
        score._full_cost_gate(tmp_path, audited, pilot_gate)
    cost["projected_gpu_hours"] = projected
    cost["preflight_seconds"]["full"] = 0.01
    cost_path.write_text(json.dumps(cost) + "\n")
    launch["cost_gate_receipt_sha256"] = score.base.sha256(cost_path)
    with pytest.raises(RuntimeError, match="preflight runtime recount"):
        score._full_cost_gate(tmp_path, audited, pilot_gate)


def test_failed_label_free_seed_audit_prevents_any_label_open(tmp_path, monkeypatch):
    for seed in score.SEEDS:
        (tmp_path / f"seed{seed}").mkdir()
    monkeypatch.setattr(score.boundary, "_data_bytes", lambda *_args: None)
    monkeypatch.setattr(score.alm, "_manifest_and_labels",
                        lambda *_args: pytest.fail("labels opened before audit"))
    monkeypatch.setattr(score, "_receipt", lambda *_args: (_ for _ in ()).throw(
        RuntimeError("wrong checkpoint provenance")))
    with pytest.raises(RuntimeError, match="wrong checkpoint provenance"):
        score.main(tmp_path, tmp_path)
