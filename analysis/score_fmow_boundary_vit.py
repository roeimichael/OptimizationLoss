"""Offline integrity gate and scorer for the fixed ViT boundary study.

Usage: python analysis/score_fmow_boundary_vit.py --gate PILOT_ROOT REF_ROOT [NEW_RECEIPT_JSON]
       python analysis/score_fmow_boundary_vit.py --pilot-score PILOT_ROOT DATA_ROOT
       python analysis/score_fmow_boundary_vit.py FULL_ROOT DATA_ROOT [OUTPUT_JSON]

The gate never opens development labels. The full scorer opens them only after
all twelve seeds and their label-free provenance and side steps pass audit.
"""

import json
import math
from datetime import datetime
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import score_fmow_boundary as boundary
from analysis import score_fmow_local as base
from analysis import score_fmow_local_alm as alm

PILOT = 6600
PILOT_RELEASE = "0f12bde246ccfabc64d549046550e9bd44cfda9a"
RELEASE_ROOT = Path(__file__).resolve().parents[1]
PILOT_RELEASE_ROOT = Path("/home/dsi/michaer8/tralo-rebuild/releases") / PILOT_RELEASE
SEEDS = tuple(range(6601, 6613))
STUDY = "local_boundary_vit_v2"
WEIGHT_SHA256 = "c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9"
TRANSFORM = "full-frame RGB 224x224 ImageNet mean/std"
CONFIGS = Path(__file__).resolve().parents[1] / "experiments/configs/fmow_local_boundary_vit_v2_20261001"
SMOKE_GENERATOR = Path(__file__).resolve().parents[1] / "tools/fmow_local_boundary_vit_smoke.py"
REAL_PREFLIGHT_GENERATOR = (Path(__file__).resolve().parents[1] /
                            "tools/fmow_local_boundary_vit_real_preflight.py")
SMOKE_FIXTURE = "fixed_sine_head_class1_bias1_v1"
RECIPE = {**base.RECIPE, "backbone": "vit_b_16", "batch_size": 16,
          "development_batch_size": 8}
EXPECTED_KEYS = set(RECIPE) | {"seed", "snapshot_steps", "study", "step_radius", "alm_rho"}
PRIMARY = boundary.PRIMARY
PRIOR_FAILED_RESERVE_HOURS = .5


def _config_and_provenance(directory, config, started):
    """Check the named ViT protocol and hash the runner's original input file."""
    d = Path(directory)
    if (set(config) != EXPECTED_KEYS or
            any(config.get(key) != value or type(config[key]) is not type(value)
                for key, value in RECIPE.items()) or
            type(config.get("seed")) is not int or config["seed"] not in SEEDS + (PILOT,) or
            type(config.get("snapshot_steps")) is not bool or
            not config["snapshot_steps"] and config["seed"] != PILOT or
            config.get("study") != STUDY or
            type(config.get("step_radius")) is not float or
            config["step_radius"] != boundary.RADIUS or
            type(config.get("alm_rho")) is not float or
            config["alm_rho"] != boundary.RHO):
        raise RuntimeError(f"{d.name}: config differs from fixed ViT boundary protocol")
    job = f"{config['seed']}_{'step' if config['snapshot_steps'] else 'ref'}"
    input_config = CONFIGS / f"fmow_local_{job}.json"
    if (base._json(input_config) != config or
            started.get("config_sha256") != base.sha256(input_config)):
        raise RuntimeError(f"{d.name}: immutable input/run config provenance mismatch")


def _positive_finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _attention_replay(value):
    """Independently reject missing or invalid ordinary-head parity evidence."""
    return (isinstance(value, dict) and
            set(value) == {"passed", "images_count", "max_absolute_difference",
                           "max_tolerance_ratio"} and
            value["passed"] is True and type(value["images_count"]) is int and
            value["images_count"] == 8 and
            type(value["max_absolute_difference"]) in (int, float) and
            type(value["max_tolerance_ratio"]) in (int, float) and
            math.isfinite(value["max_absolute_difference"]) and
            math.isfinite(value["max_tolerance_ratio"]) and
            0 <= value["max_absolute_difference"] <= 1.1e-6 and
            0 <= value["max_tolerance_ratio"] <= 1)


def _active_smoke_scopes(scopes, quota):
    """Independently recount diagnostic activity against the run's fixed quotas."""
    if (not isinstance(scopes, dict) or set(scopes) != {"pooled", "countries"} or
            not isinstance(quota, dict) or
            set(quota) != {"global_cap", "local_total", "local_caps"} or
            type(quota["local_total"]) is not int or
            not isinstance(quota["local_caps"], dict) or
            set(quota["local_caps"]) != set(base.COUNTRIES) or
            any(type(value) is not int for value in quota["local_caps"].values()) or
            sum(quota["local_caps"].values()) != quota["local_total"] or
            not isinstance(scopes["countries"], dict) or
            set(scopes["countries"]) != set(base.COUNTRIES)):
        return False
    records = [(scopes["pooled"], quota["global_cap"])] + [
        (scopes["countries"][country], quota["local_caps"][country])
        for country in base.COUNTRIES]
    for record, expected_cap in records:
        if (not isinstance(record, dict) or set(record) != {"hard", "soft", "cap"} or
                type(expected_cap) is not int or type(record["cap"]) is not int or
                record["cap"] != expected_cap or type(record["hard"]) is not int or
                type(record["soft"]) not in (int, float) or
                not math.isfinite(record["soft"]) or
                record["hard"] <= expected_cap or record["soft"] <= expected_cap):
            return False
    countries = scopes["countries"].values()
    return (scopes["pooled"]["hard"] == sum(row["hard"] for row in countries) and
            math.isclose(scopes["pooled"]["soft"],
                         math.fsum(row["soft"] for row in countries),
                         rel_tol=1e-5, abs_tol=1e-5))


def _iso_timestamp(value):
    if not isinstance(value, str):
        raise ValueError("timestamp is not a string")
    instant = datetime.fromisoformat(value)
    if instant.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return instant.timestamp()


def _memory_smoke(directory, config, launch, started):
    """Bind a label-free, same-release and same-card GPU preflight to the run."""
    d = Path(directory)
    receipt = d.parent / "vit_memory_smoke.json"
    smoke_launch_path = d.parent / "vit_memory_smoke.launch.json"
    complete_path = d.parent / "vit_memory_smoke.complete.json"
    if (launch.get("memory_smoke_execution") != "queue_executed" or
            launch.get("memory_smoke_receipt_path") != str(receipt) or
            launch.get("memory_smoke_receipt_sha256") != base.sha256(receipt)):
        raise RuntimeError(f"{d.name}: queue-owned ViT memory-smoke receipt hash/path mismatch")
    if (launch.get("memory_smoke_complete_path") != str(complete_path) or
            launch.get("memory_smoke_complete_sha256") != base.sha256(complete_path)):
        raise RuntimeError(f"{d.name}: ViT memory-smoke completion hash/path mismatch")
    if (launch.get("memory_smoke_launch_path") != str(smoke_launch_path) or
            launch.get("memory_smoke_launch_sha256") != base.sha256(smoke_launch_path)):
        raise RuntimeError(f"{d.name}: ViT memory-smoke queue launch hash/path mismatch")
    smoke_launch = base._json(smoke_launch_path)
    expected_smoke_launch = {"release_commit": launch["release_commit"],
                             "host": launch["host"], "gpu_index": launch["gpu_index"],
                             "gpu_uuid": launch["gpu_uuid"], "receipt_path": str(receipt),
                             "generator_sha256": base.sha256(SMOKE_GENERATOR)}
    if any(smoke_launch.get(key) != value or
           type(smoke_launch[key]) is not type(value)
           for key, value in expected_smoke_launch.items()):
        raise RuntimeError(f"{d.name}: ViT memory-smoke queue launch identity mismatch")
    complete = base._json(complete_path)
    expected_complete = {"exit_code": 0, "release_commit": launch["release_commit"],
                         "host": launch["host"], "gpu_uuid": launch["gpu_uuid"],
                         "receipt_path": str(receipt)}
    if any(complete.get(key) != value or type(complete[key]) is not type(value)
           for key, value in expected_complete.items()):
        raise RuntimeError(f"{d.name}: ViT memory-smoke queue completion identity mismatch")
    smoke = base._json(receipt)
    expected = {"release_commit": launch["release_commit"], "host": launch["host"],
                "gpu_uuid": launch["gpu_uuid"], "backbone": "vit_b_16",
                "batch_size": RECIPE["batch_size"],
                "development_batch_size": RECIPE["development_batch_size"],
                "weight_sha256": WEIGHT_SHA256, "precision": "fp32",
                "label_free": True, "memory_smoke_passed": True,
                "mha_fastpath_enabled": False,
                "source_sha256": base.source(),
                "data_files": base.FILES,
                "device_name": started["device"], "development_pool_count": 1673,
                "smoke_generator_sha256": base.sha256(SMOKE_GENERATOR),
                "gpu_index": launch["gpu_index"]}
    if any(smoke.get(key) != value or type(smoke[key]) is not type(value)
           for key, value in expected.items()):
        raise RuntimeError(f"{d.name}: ViT memory-smoke identity mismatch")
    if not _attention_replay(smoke.get("ordinary_head_replay")):
        raise RuntimeError(f"{d.name}: ViT ordinary-head memory-smoke replay failed")
    peak, total = smoke.get("peak_allocated_bytes"), smoke.get("total_memory_bytes")
    if (type(peak) is not int or type(total) is not int or not 0 < peak < .9 * total):
        raise RuntimeError(f"{d.name}: ViT memory-smoke capacity invalid")
    if (not _positive_finite(smoke.get("started_utc")) or
            not _positive_finite(smoke.get("ended_utc")) or
            smoke["ended_utc"] <= smoke["started_utc"]):
        raise RuntimeError(f"{d.name}: ViT memory-smoke timing invalid")
    try:
        smoke_launched_at = _iso_timestamp(smoke_launch["started_utc"])
        completed_at = _iso_timestamp(complete["ended_utc"])
        launched_at = _iso_timestamp(launch["started_utc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{d.name}: ViT queue smoke/launch timestamps invalid") from exc
    if not (smoke_launched_at <= smoke["started_utc"] < smoke["ended_utc"] <=
            completed_at <= launched_at):
        raise RuntimeError(f"{d.name}: ViT smoke did not complete before queue launch")
    phases = smoke.get("phases")
    if not isinstance(phases, dict) or set(phases) != {
            "train_backward", "development_inference", "side_copy_constraint_gradient"}:
        raise RuntimeError(f"{d.name}: ViT memory-smoke phases absent/extra")
    for name, phase in phases.items():
        if (not isinstance(phase, dict) or phase.get("completed") is not True or
                not _positive_finite(phase.get("seconds")) or
                type(phase.get("peak_allocated_bytes")) is not int or
                not 0 < phase["peak_allocated_bytes"] <= peak):
            raise RuntimeError(f"{d.name}: ViT memory-smoke {name} not measured")
    if peak != max(phase["peak_allocated_bytes"] for phase in phases.values()):
        raise RuntimeError(f"{d.name}: ViT memory-smoke peak recount mismatch")
    train = phases["train_backward"]
    if (not _positive_finite(train.get("loss")) or
            not _positive_finite(train.get("gradient_norm")) or
            train.get("optimizer_step") is not True or
            train.get("input_shape") != [16, 3, 224, 224]):
        raise RuntimeError(f"{d.name}: ViT train-backward smoke evidence invalid")
    development = phases["development_inference"]
    if (development.get("probabilities_shape") != [1673, base.CLASSES] or
            type(development.get("finite_rows")) is not int or
            development["finite_rows"] != 1673 or
            type(development.get("row_sum_max_error")) not in (int, float) or
            not math.isfinite(development["row_sum_max_error"]) or
            not 0 <= development["row_sum_max_error"] < 1e-4):
        raise RuntimeError(f"{d.name}: ViT development-inference smoke evidence invalid")
    side = phases["side_copy_constraint_gradient"]
    precheck = side.get("pooled_gradient_precheck")
    if (side.get("pto_unchanged") is not True or
            side.get("fixture") != SMOKE_FIXTURE or
            not isinstance(precheck, dict) or
            not _positive_finite(precheck.get("total_norm")) or
            not _positive_finite(precheck.get("backbone_norm")) or
            precheck.get("pto_unchanged") is not True or
            not isinstance(side.get("caps"), dict) or
            set(side["caps"]) != {str(d) for d in base.DIVISORS}):
        raise RuntimeError(f"{d.name}: ViT side-copy smoke evidence invalid")
    for divisor in base.DIVISORS:
        cap = side["caps"][str(divisor)]
        if (not isinstance(cap, dict) or not _positive_finite(cap.get("joint_gradient_norm")) or
                not _positive_finite(cap.get("phr_gradient_norm")) or
                cap.get("joint_applied") is not True or
                cap.get("phr_applied") is not True or
                cap.get("all_four_arms") is not True or
                cap.get("scope_derivatives_finite") is not True or
                cap.get("pto_unchanged") is not True or
                not isinstance(started.get("quotas"), dict) or
                not _active_smoke_scopes(cap.get("scopes"),
                                         started["quotas"].get(str(divisor)))):
            raise RuntimeError(f"{d.name}: ViT side-copy cap{divisor} smoke evidence invalid")
    return completed_at - smoke_launched_at


def _real_preflight(directory, launch):
    """Recount queue-owned real-image numerical checks without opening labels."""
    d = Path(directory)
    root = d.parent
    names = {"launch": root / "vit_real_preflight.launch.json",
             "receipt": root / "vit_real_preflight.json",
             "complete": root / "vit_real_preflight.complete.json"}
    if launch.get("real_preflight_execution") != "queue_executed":
        raise RuntimeError(f"{d.name}: real-image preflight was not queue executed")
    for kind, path in names.items():
        if (launch.get(f"real_preflight_{kind}_path") != str(path) or
                launch.get(f"real_preflight_{kind}_sha256") != base.sha256(path)):
            raise RuntimeError(f"{d.name}: real-image preflight {kind} hash/path mismatch")
    prelaunch, receipt, complete = (base._json(names[key]) for key in
                                    ("launch", "receipt", "complete"))
    identity = {"release_commit": launch["release_commit"],
                "host": launch["host"], "gpu_uuid": launch["gpu_uuid"],
                "gpu_index": launch["gpu_index"]}
    launch_identity = {**identity, "receipt_path": str(names["receipt"]),
                       "generator_sha256": base.sha256(REAL_PREFLIGHT_GENERATOR)}
    if any(prelaunch.get(key) != value or type(prelaunch[key]) is not type(value)
           for key, value in launch_identity.items()):
        raise RuntimeError(f"{d.name}: real-image preflight queue launch identity mismatch")
    completion_identity = {"release_commit": identity["release_commit"],
                           "host": identity["host"], "gpu_uuid": identity["gpu_uuid"],
                           "receipt_path": str(names["receipt"]), "exit_code": 0}
    if any(complete.get(key) != value or type(complete[key]) is not type(value)
           for key, value in completion_identity.items()):
        raise RuntimeError(f"{d.name}: real-image preflight queue completion mismatch")
    expected = {**identity, "source_sha256": base.source(),
                "data_file_sha256": base.FILES, "precision": "fp32",
                "backbone": "vit_b_16", "development_labels_accessed": False,
                "preflight_passed": True, "images_count": 15,
                "mha_fastpath_enabled": False,
                "weight_sha256": WEIGHT_SHA256,
                "development_batch_size": RECIPE["development_batch_size"],
                "chunk_sizes": [8, 7],
                "preflight_generator_sha256": base.sha256(REAL_PREFLIGHT_GENERATOR),
                "country_counts": {country: 3 for country in sorted(base.COUNTRIES)},
                "pto_unchanged": True}
    if any(receipt.get(key) != value or type(receipt[key]) is not type(value)
           for key, value in expected.items()):
        raise RuntimeError(f"{d.name}: real-image preflight identity/label boundary mismatch")
    if not _attention_replay(receipt.get("ordinary_head_replay")):
        raise RuntimeError(f"{d.name}: real-image ordinary-head replay failed")
    weight = receipt.get("pretrained_weight")
    if (not isinstance(weight, dict) or weight.get("sha256") != WEIGHT_SHA256 or
            Path(weight.get("file", "")).name != "vit_b_16-c867db91.pth"):
        raise RuntimeError(f"{d.name}: real-image preflight checkpoint differs")
    if not isinstance(receipt.get("device_name"), str) or not receipt["device_name"]:
        raise RuntimeError(f"{d.name}: real-image preflight device identity missing")
    try:
        launched_at = _iso_timestamp(prelaunch["started_utc"])
        completed_at = _iso_timestamp(complete["ended_utc"])
        job_at = _iso_timestamp(launch["started_utc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{d.name}: real-image preflight queue timestamps invalid") from exc
    if (not _positive_finite(receipt.get("started_utc")) or
            not _positive_finite(receipt.get("ended_utc")) or
            not launched_at <= receipt["started_utc"] < receipt["ended_utc"] <=
                completed_at <= job_at):
        raise RuntimeError(f"{d.name}: real-image preflight did not finish before job")
    probability_error = receipt.get("max_probability_difference")
    if (type(probability_error) not in (int, float) or
            not math.isfinite(probability_error) or not 0 <= probability_error <= 1e-6):
        raise RuntimeError(f"{d.name}: full/chunk real-image probability parity failed")
    scopes = ("pooled", *sorted(base.COUNTRIES))
    gradient_errors = receipt.get("gradient_relative_errors")
    if (not isinstance(gradient_errors, dict) or set(gradient_errors) != set(scopes) or
            any(type(value) not in (int, float) or not math.isfinite(value) or
                not 0 <= value <= .01 for value in gradient_errors.values())):
        raise RuntimeError(f"{d.name}: full/chunk real-image gradient parity failed")
    finite = receipt.get("finite_differences")
    expected_fd = {f"{scope}@{epsilon}" for scope in scopes for epsilon in ("0.01", "0.02")}
    if not isinstance(finite, dict) or set(finite) != expected_fd:
        raise RuntimeError(f"{d.name}: real-image finite-difference scopes incomplete")
    for name, record in finite.items():
        if (not isinstance(record, dict) or
                any(type(record.get(key)) not in (int, float) or
                    not math.isfinite(record[key]) for key in
                    ("analytic", "numeric", "error", "tolerance"))):
            raise RuntimeError(f"{d.name}: real-image finite-difference {name} invalid")
        analytic, numeric = record["analytic"], record["numeric"]
        error = abs(analytic - numeric)
        tolerance = .05 * max(abs(analytic), abs(numeric)) + .003
        if (not math.isclose(record["error"], error, rel_tol=1e-5, abs_tol=1e-6) or
                not math.isclose(record["tolerance"], tolerance, rel_tol=1e-5, abs_tol=1e-6) or
                error > tolerance):
            raise RuntimeError(f"{d.name}: real-image finite-difference {name} failed recount")
    if receipt.get("arms_audited") != list(boundary.ARMS):
        raise RuntimeError(f"{d.name}: real-image preflight missed side-copy arms")
    artifacts = receipt.get("artifact_sha256")
    expected_artifacts = {f"epoch01_{arm}.pt" for arm in boundary.ARMS}
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise RuntimeError(f"{d.name}: real-image preflight artifacts incomplete")
    artifact_root = root / "vit_real_preflight.artifacts"
    if any(artifacts[name] != base.sha256(artifact_root / name)
           for name in expected_artifacts):
        raise RuntimeError(f"{d.name}: real-image preflight artifact hash mismatch")
    return completed_at - launched_at


def _model_identity(directory, config, summary, started, init):
    d = Path(directory)
    weight = init.get("pretrained_weight")
    if (summary["seed"] != config["seed"] or
            summary["quotas"] != started["quotas"] or
            init["initial_sha256"] != summary["initial_sha256"] or
            init["architecture"] != "vit_b_16" or init["classes"] != base.CLASSES or
            init.get("transform") != TRANSFORM or not isinstance(weight, dict) or
            weight.get("sha256") != WEIGHT_SHA256 or
            Path(weight.get("file", "")).name != "vit_b_16-c867db91.pth"):
        raise RuntimeError(f"{d.name}: ViT model or quota identity mismatch")


def _receipt(directory):
    """Audit input, PTO training and unlabeled quotas before any label access."""
    d = Path(directory)
    top = base._events(d / "events.jsonl")
    train = base._events(d / "retrain1/events.jsonl", terminal="training_completed")
    started, init, done = (base._one(top, name) for name in
                           ("started", "model_initialized", "completed"))
    config, summary = base._json(d / "config.json"), base._json(d / "summary.json")
    _config_and_provenance(d, config, started)
    if started.get("mha_fastpath_enabled") is not False:
        raise RuntimeError(f"{d.name}: ViT MHA fastpath was not disabled before PTO")
    launch = alm._launch_receipt(d, config, started)
    launch = {**launch, "_smoke_seconds": _memory_smoke(d, config, launch, started),
              "_preflight_seconds": _real_preflight(d, launch)}
    if (started["source_sha256"] != base.source() or
            started["data_files"] != base.FILES or
            started["counts"] != {"train": 15841, "stop": 1829, "dev": 1673} or
            started["manifest_sha256"] != base.sha256(d / "manifest.json")):
        raise RuntimeError(f"{d.name}: source/data/manifest provenance mismatch")
    ids, groups = base._pool_identity(d, started)
    _model_identity(d, config, summary, started, init)
    result = summary["retrain"]
    epochs, best = result["epochs_run"], result["best_epoch"]
    if (type(epochs) is not int or type(best) is not int or
            not 1 <= best <= epochs <= RECIPE["max_epochs"] or
            done["epochs_run"] != epochs or
            done["task_updates"] != result["task_updates"] or
            result["task_updates"] != math.ceil(15841 / config["batch_size"]) * epochs or
            base._one(train, "training_completed")["task_updates"] != result["task_updates"]):
        raise RuntimeError(f"{d.name}: epoch/update dose mismatch")
    epoch_rows = [row for row in train if row["event"] == "epoch"]
    if [row["epoch"] for row in epoch_rows] != list(range(1, epochs + 1)):
        raise RuntimeError(f"{d.name}: missing training epoch")
    best_loss, best_epoch, waited = math.inf, 0, 0
    for row in epoch_rows:
        if any(not math.isfinite(row[key]) for key in
               ("training_loss", "stop_loss", "base_lr", "last_lr", "mean_gate",
                "live_false_positives", "soft_count_capped")):
            raise RuntimeError(f"{d.name}: nonfinite training log")
        improved = row["stop_loss"] < best_loss
        if row["improved"] is not improved:
            raise RuntimeError(f"{d.name}: early-stop flag differs")
        if improved:
            best_loss, best_epoch, waited = row["stop_loss"], row["epoch"], 0
        else:
            waited += 1
        if waited >= config["patience"] and row["epoch"] < epochs:
            raise RuntimeError(f"{d.name}: training continued after patience")
    if (best != best_epoch or result["best_stop_loss"] != best_loss or
            epochs < config["max_epochs"] and waited < config["patience"]):
        raise RuntimeError(f"{d.name}: stopping endpoint mismatch")
    if set(summary["pto_snapshot_sha256"]) != {str(e) for e in range(1, epochs + 1)}:
        raise RuntimeError(f"{d.name}: PTO snapshot hashes incomplete")
    for epoch in range(1, epochs + 1):
        path = d / "retrain1" / f"epoch{epoch:02d}.pt"
        if base.sha256(path) != summary["pto_snapshot_sha256"][str(epoch)]:
            raise RuntimeError(f"{d.name}: PTO snapshot hash mismatch")
    if base.sha256(d / "retrain1/final_probabilities.pt") != summary["final_probability_sha256"]:
        raise RuntimeError(f"{d.name}: final PTO hash mismatch")
    return config, summary, started, train, ids, groups, launch


def _pilot_hashes(directory, summary, artifacts):
    """Bind all label-free events, input copies and probability artifacts."""
    d = Path(directory)
    paths = ["events.jsonl", "retrain1/events.jsonl", "config.json", "summary.json",
             "manifest.json", "pool_identity.json"]
    paths.extend(artifacts)
    for epoch in range(1, summary["retrain"]["epochs_run"] + 1):
        paths.append(f"retrain1/epoch{epoch:02d}.pt")
    paths.append("retrain1/final_probabilities.pt")
    return {name: base.sha256(d / name) for name in sorted(set(paths))}


def _queue_hashes(directory, job):
    root = Path(directory)
    return {name: base.sha256(root / name) for name in (
        f"seed{job}.launch.json", f"seed{job}.complete.json",
        "vit_memory_smoke.launch.json", "vit_memory_smoke.json",
        "vit_memory_smoke.complete.json",
        "vit_real_preflight.launch.json", "vit_real_preflight.json",
        "vit_real_preflight.complete.json")}


def gate(pilot_root, reference_root, output=None):
    """Validate the matched pilot with labels still sealed."""
    pilot_root, reference_root = Path(pilot_root).resolve(), Path(reference_root).resolve()
    if sorted(p.name for p in pilot_root.glob("seed*") if p.is_dir()) != ["seed6600"]:
        raise RuntimeError("ViT pilot root must contain exactly seed6600")
    if sorted(p.name for p in reference_root.glob("seed*") if p.is_dir()) != ["seed6600_ref"]:
        raise RuntimeError("ViT reference root must contain exactly seed6600_ref")
    a, b = _receipt(pilot_root / "seed6600"), _receipt(reference_root / "seed6600_ref")
    ac, summary, started, events, ids, groups, launch = a
    bc, ref, ref_started, ref_events, ref_ids, ref_groups, ref_launch = b
    if (ac["seed"] != PILOT or bc["seed"] != PILOT or
            ac["snapshot_steps"] is not True or bc["snapshot_steps"] is not False or
            (ids, groups) != (ref_ids, ref_groups) or
            base.sha256(pilot_root / "seed6600/manifest.json") !=
            base.sha256(reference_root / "seed6600_ref/manifest.json") or
            started["quotas"] != ref_started["quotas"] or
            summary["retrain"] != ref["retrain"] or
            summary["initial_sha256"] != ref["initial_sha256"] or
            summary.get("pto_sha256") != ref.get("pto_sha256") or
            started.get("device") != ref_started.get("device") or
            started.get("precision") != "fp32" or
            launch["host"] != ref_launch["host"] or
            launch.get("data_root") != ref_launch.get("data_root") or
            launch["release_commit"] != ref_launch["release_commit"] or
            ref["steps"] or any(row["event"] == "snapshot_cap" for row in ref_events)):
        raise RuntimeError("ViT pilot/reference setup or PTO metadata mismatch")
    data_root = Path(launch.get("data_root", ""))
    boundary._data_bytes(data_root)
    if ([(row["epoch"], row["hard_counts"], row["stop_loss"]) for row in events
         if row["event"] == "epoch"] !=
            [(row["epoch"], row["hard_counts"], row["stop_loss"]) for row in ref_events
             if row["event"] == "epoch"]):
        raise RuntimeError("ViT pilot/reference epoch trajectory differs")
    epochs = summary["retrain"]["epochs_run"]
    for name in [f"epoch{e:02d}.pt" for e in range(1, epochs + 1)] + ["final_probabilities.pt"]:
        left = base._probabilities(pilot_root / "seed6600/retrain1" / name, len(ids))
        right = base._probabilities(reference_root / "seed6600_ref/retrain1" / name, len(ids))
        if not torch.equal(left, right):
            raise RuntimeError(f"ViT PTO trajectory differs: {name}")
    _, _, step_artifacts = boundary._steps(pilot_root / "seed6600", a)
    step_done = base._one(base._events(pilot_root / "seed6600/events.jsonl"), "completed")
    ref_done = base._one(base._events(reference_root / "seed6600_ref/events.jsonl"), "completed")
    seconds = (step_done.get("seconds"), ref_done.get("seconds"))
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in seconds):
        raise RuntimeError("ViT pilot durations missing or nonfinite")
    smoke_seconds = (launch["_smoke_seconds"], ref_launch["_smoke_seconds"])
    if any(not _positive_finite(value) for value in smoke_seconds):
        raise RuntimeError("ViT queue smoke durations missing or nonfinite")
    preflight_seconds = (launch["_preflight_seconds"], ref_launch["_preflight_seconds"])
    if any(not _positive_finite(value) for value in preflight_seconds):
        raise RuntimeError("ViT real-image preflight durations missing or nonfinite")
    full_smoke_projection = max(smoke_seconds)
    full_preflight_projection = max(preflight_seconds)
    projected = (1.25 * (13 * seconds[0] + seconds[1]) + sum(smoke_seconds) +
                 full_smoke_projection + sum(preflight_seconds) +
                 full_preflight_projection) / 3600
    if projected + PRIOR_FAILED_RESERVE_HOURS > 24:
        raise RuntimeError(f"projected ViT study plus failed-attempt reserve "
                           f"{projected + PRIOR_FAILED_RESERVE_HOURS:.3f} exceeds 24 GPU-hours")
    result = {"status": "vit_pilot_integrity_pass", "epochs": epochs, "pto_equal": True,
            "development_labels_accessed": False, "step_seconds": seconds[0],
            "reference_seconds": seconds[1],
            "pilot_smoke_seconds": list(smoke_seconds),
            "projected_full_smoke_seconds": full_smoke_projection,
            "pilot_preflight_seconds": list(preflight_seconds),
            "projected_full_preflight_seconds": full_preflight_projection,
            "projected_total_gpu_hours": projected,
            "pilot_step_root": str(pilot_root.resolve()),
            "pilot_ref_root": str(reference_root.resolve()),
            "release_commit": launch["release_commit"], "host": launch["host"],
            "gpu_uuids": {"step": launch["gpu_uuid"], "ref": ref_launch["gpu_uuid"]},
            "source_sha256": base.source(), "data_file_sha256": base.FILES,
            "input_config_sha256": {
                "step": base.sha256(CONFIGS / "fmow_local_6600_step.json"),
                "ref": base.sha256(CONFIGS / "fmow_local_6600_ref.json")},
            "pilot_artifact_sha256": {
                "step": _pilot_hashes(pilot_root / "seed6600", summary, step_artifacts),
                "ref": _pilot_hashes(reference_root / "seed6600_ref", ref, ())},
            "pilot_queue_sha256": {
                "step": _queue_hashes(pilot_root, "6600_step"),
                "ref": _queue_hashes(reference_root, "6600_ref")},
            "joint_applied_by_cap": {str(divisor): sum(
                bool(summary["steps"][str(epoch)][str(divisor)]["joint"]["applied"])
                for epoch in range(1, epochs + 1)) for divisor in base.DIVISORS}}
    if output is not None:
        with Path(output).open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return result


def load_seed(directory, data_root, *, allow_pilot=False, _audited=None):
    """Score a ViT seed after checking its saved artifacts and fixed identities."""
    d, data_root = Path(directory), Path(data_root)
    receipt = _receipt(d) if _audited is None else _audited[0]
    config, summary, _, _, ids, groups, launch = receipt
    expected = (PILOT,) if allow_pilot else SEEDS
    if (d.name != f"seed{config['seed']}" or config["seed"] not in expected or
            config["snapshot_steps"] is not True):
        raise RuntimeError("seed/config differs from fixed ViT block")
    if Path(launch.get("data_root", "")).resolve() != data_root.resolve():
        raise RuntimeError("ViT scoring data root differs from launch")
    pto, sides, artifacts = boundary._steps(d, receipt) if _audited is None else _audited[1]
    labels = alm._manifest_and_labels(d, receipt[:6], data_root)
    epochs, best = summary["retrain"]["epochs_run"], summary["retrain"]["best_epoch"]
    window = list(range(max(1, best - 2), epochs + 1))
    out = {"seed": config["seed"], "epochs_run": epochs, "best_epoch": best,
           "window": window, "manifest_sha256": base.sha256(d / "manifest.json"),
           "config_sha256": base.sha256(d / "config.json"), "artifacts": artifacts,
           "release_commit": launch["release_commit"],
           "class_supports": {str(c): labels.count(c) for c in range(base.CLASSES)},
           "caps": {}}
    for divisor in base.DIVISORS:
        quota = summary["quotas"][str(divisor)]
        ensembles = {"ens_pto": [pto[e] for e in window]}
        for arm in boundary.ARMS:
            ensembles[boundary.SCORED_ARMS[arm]] = [sides[e, divisor][arm] for e in window]
        scored = {name: base._arm_score(torch.stack(values).mean(0), labels, ids, groups, quota)
                  for name, values in ensembles.items()}
        for arm in scored.values():
            selected = arm["predictions"]
            arm["selected_ids"] = [sample_id for sample_id, value in zip(ids, selected)
                                   if value == base.CAPPED]
            if len(arm["selected_ids"]) != quota["global_cap"]:
                raise RuntimeError("ViT allocated cap did not fill declared pooled slots")
            arm["class1_confusion"] = {
                "tp": sum(y == base.CAPPED and p == base.CAPPED for y, p in zip(labels, selected)),
                "fp": sum(y != base.CAPPED and p == base.CAPPED for y, p in zip(labels, selected)),
                "fn": sum(y == base.CAPPED and p != base.CAPPED for y, p in zip(labels, selected))}
            arm["class1_precision"] = (arm["class1_confusion"]["tp"] /
                                      max(1, arm["class1_confusion"]["tp"] +
                                          arm["class1_confusion"]["fp"]))
            arm["class1_recall"] = (arm["class1_confusion"]["tp"] /
                                   max(1, arm["class1_confusion"]["tp"] +
                                       arm["class1_confusion"]["fn"]))
        joint = scored["ens_joint"]["predictions"]
        out["caps"][str(divisor)] = {
            "quota": quota,
            "arms": {name: {key: value for key, value in row.items() if key != "predictions"}
                     for name, row in scored.items()},
            "slot_turnover": {f"joint_vs_{name}": boundary._movement(
                joint, row["predictions"], labels, ids)
                              for name, row in scored.items() if name != "ens_joint"},
            "all_epoch_step_diagnostics": {
                str(epoch): {arm: {key: value for key, value in
                                   summary["steps"][str(epoch)][str(divisor)][arm].items()
                                   if key != "probability_sha256"}
                             for arm in boundary.ARMS} for epoch in range(1, epochs + 1)},
            "applied_epochs": {arm: sum(bool(summary["steps"][str(epoch)][str(divisor)][arm]["applied"])
                                        for epoch in range(1, epochs + 1)) for arm in boundary.ARMS}}
    return out


def pilot_score(pilot_root, data_root):
    """Exploratory pilot metrics; they cannot select ViT settings."""
    root = Path(pilot_root).resolve()
    if sorted(p.name for p in root.glob("seed*") if p.is_dir()) != ["seed6600"]:
        raise RuntimeError("ViT pilot scoring root must contain exactly seed6600")
    boundary._data_bytes(data_root)
    return {"status": "vit_pilot_metrics_exploratory_not_for_setting_selection",
            "seed": load_seed(root / "seed6600", data_root, allow_pilot=True),
            "development_labels_accessed_offline": True}


def _full_pilot_gate(root, audited):
    """Recompute the score-free pilot gate before any full-block label read."""
    path = Path(root) / "vit_pilot_gate_recheck.json"
    digest = base.sha256(path)
    launches = [item[0][6] for item in audited]
    if any(launch.get("pilot_gate_receipt_path") != str(path) or
           launch.get("pilot_gate_receipt_sha256") != digest for launch in launches):
        raise RuntimeError("ViT full block lacks same queue-owned pilot gate receipt")
    saved = base._json(path)
    if (saved.get("status") != "vit_pilot_integrity_pass" or
            saved.get("development_labels_accessed") is not False or
            saved.get("release_commit") != PILOT_RELEASE or
            saved.get("host") != launches[0]["host"] or
            saved.get("source_sha256") != base.source() or
            saved.get("data_file_sha256") != base.FILES or
            saved.get("input_config_sha256") != {
                "step": base.sha256(CONFIGS / "fmow_local_6600_step.json"),
                "ref": base.sha256(CONFIGS / "fmow_local_6600_ref.json")} or
            not isinstance(saved.get("pilot_step_root"), str) or
            not isinstance(saved.get("pilot_ref_root"), str)):
        raise RuntimeError("ViT full pilot gate identity/status mismatch")
    recomputed = gate(saved["pilot_step_root"], saved["pilot_ref_root"])
    if recomputed != saved:
        raise RuntimeError("ViT full pilot gate differs from fresh label-blind audit")
    return saved


def _job_queue_seconds(root, job, release, host):
    d = Path(root)
    launch = base._json(d / f"seed{job}.launch.json")
    done = base._json(d / f"seed{job}.complete.json")
    if (launch.get("release_commit") != release or done.get("release_commit") != release or
            launch.get("host") != host or done.get("host") != host or
            type(done.get("exit_code")) is not int or done["exit_code"] != 0):
        raise RuntimeError("ViT pilot queue job provenance/completion differs")
    try:
        seconds = _iso_timestamp(done["ended_utc"]) - _iso_timestamp(launch["started_utc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("ViT pilot queue job timestamps invalid") from exc
    if not _positive_finite(seconds):
        raise RuntimeError("ViT pilot queue job runtime invalid")
    return seconds


def _cross_release_identity(root, full_release):
    """Recheck the queue's byte comparison without opening development labels."""
    path = Path(root) / "vit_cross_release_identity.json"
    saved = base._json(path)
    config_root = Path("experiments/configs/fmow_local_boundary_vit_v2_20261001")
    modules = {item.relative_to(RELEASE_ROOT) for item in (RELEASE_ROOT / "tralo").glob("*.py")}
    pilot_modules = {item.relative_to(PILOT_RELEASE_ROOT)
                     for item in (PILOT_RELEASE_ROOT / "tralo").glob("*.py")}
    expected = modules | {config_root / f"fmow_local_{job}.json"
                          for job in ("6600_step", "6600_ref",
                                      *(f"{seed}_step" for seed in SEEDS))} | {
        Path("tools/fmow_local_boundary_vit_smoke.py"),
        Path("tools/fmow_local_boundary_vit_real_preflight.py")}
    if (RELEASE_ROOT.name != full_release or modules != pilot_modules or
            saved.get("pilot_release_commit") != PILOT_RELEASE or
            saved.get("full_release_commit") != full_release or
            saved.get("source_config_and_preflights_equal") is not True or
            not isinstance(saved.get("files"), dict) or
            set(saved["files"]) != {item.as_posix() for item in expected}):
        raise RuntimeError("ViT pilot/full source, config or generator identity differs")
    for item in expected:
        digest = saved["files"][item.as_posix()]
        if (not isinstance(digest, str) or len(digest) != 64 or
                base.sha256(RELEASE_ROOT / item) != digest or
                base.sha256(PILOT_RELEASE_ROOT / item) != digest):
            raise RuntimeError(f"ViT pilot/full bytes differ: {item}")
    return path


def _full_cost_gate(root, audited, pilot_gate):
    """Recount the pre-dispatch cost gate including actual full-queue smoke."""
    d = Path(root)
    path = d / "vit_cost_gate.json"
    digest = base.sha256(path)
    launches = [item[0][6] for item in audited]
    if any(launch.get("cost_gate_receipt_path") != str(path) or
           launch.get("cost_gate_receipt_sha256") != digest for launch in launches):
        raise RuntimeError("ViT full block lacks same queue-owned cost gate receipt")
    saved = base._json(path)
    pilot_gate_path = d / "vit_pilot_gate_recheck.json"
    identity_path = _cross_release_identity(d, launches[0]["release_commit"])
    expected = {"release_commit": launches[0]["release_commit"],
                "pilot_release_commit": PILOT_RELEASE,
                "host": launches[0]["host"], "ceiling_gpu_hours": 24.0,
                "prior_failed_reserve_gpu_hours": PRIOR_FAILED_RESERVE_HOURS,
                "gate_passed": True,
                "pilot_step_root": pilot_gate["pilot_step_root"],
                "pilot_ref_root": pilot_gate["pilot_ref_root"],
                "full_root": str(d.resolve()),
                "pilot_gate_receipt_path": str(pilot_gate_path),
                "pilot_gate_receipt_sha256": base.sha256(pilot_gate_path),
                "cross_release_identity_path": str(identity_path),
                "cross_release_identity_sha256": base.sha256(identity_path)}
    if any(saved.get(key) != value or type(saved[key]) is not type(value)
           for key, value in expected.items()):
        raise RuntimeError("ViT full cost gate identity/status mismatch")
    step_seconds = _job_queue_seconds(pilot_gate["pilot_step_root"], "6600_step",
                                      PILOT_RELEASE, expected["host"])
    ref_seconds = _job_queue_seconds(pilot_gate["pilot_ref_root"], "6600_ref",
                                     PILOT_RELEASE, expected["host"])
    smoke_seconds = {"pilot_step": pilot_gate["pilot_smoke_seconds"][0],
                     "pilot_ref": pilot_gate["pilot_smoke_seconds"][1],
                     "full": launches[0]["_smoke_seconds"]}
    preflight_seconds = {"pilot_step": pilot_gate["pilot_preflight_seconds"][0],
                         "pilot_ref": pilot_gate["pilot_preflight_seconds"][1],
                         "full": launches[0]["_preflight_seconds"]}
    for key, value in (("pilot_step_seconds", step_seconds),
                       ("pilot_ref_seconds", ref_seconds)):
        if not _positive_finite(saved.get(key)) or not math.isclose(
                saved[key], value, rel_tol=1e-6, abs_tol=.01):
            raise RuntimeError("ViT full cost gate job runtime recount mismatch")
    if set(saved.get("smoke_seconds", {})) != set(smoke_seconds):
        raise RuntimeError("ViT full cost gate smoke scopes differ")
    for key, value in smoke_seconds.items():
        if not _positive_finite(saved["smoke_seconds"].get(key)) or not math.isclose(
                saved["smoke_seconds"][key], value, rel_tol=1e-6, abs_tol=.01):
            raise RuntimeError("ViT full cost gate smoke runtime recount mismatch")
    if set(saved.get("preflight_seconds", {})) != set(preflight_seconds):
        raise RuntimeError("ViT full cost gate real-image preflight scopes differ")
    for key, value in preflight_seconds.items():
        if not _positive_finite(saved["preflight_seconds"].get(key)) or not math.isclose(
                saved["preflight_seconds"][key], value, rel_tol=1e-6, abs_tol=.01):
            raise RuntimeError("ViT full cost gate real-image preflight runtime recount mismatch")
    projected = (1.25 * (13 * step_seconds + ref_seconds) +
                 sum(smoke_seconds.values()) + sum(preflight_seconds.values())) / 3600
    if (projected + PRIOR_FAILED_RESERVE_HOURS > 24 or
            not _positive_finite(saved.get("projected_gpu_hours")) or
            not math.isclose(saved["projected_gpu_hours"], projected,
                             rel_tol=1e-6, abs_tol=1e-4)):
        raise RuntimeError("ViT full cost gate projection exceeds or differs")
    return saved


def main(run_root, data_root, output=None):
    """Require twelve complete paired ViT seeds before opening labels."""
    root = Path(run_root).resolve()
    found = {p.name for p in root.glob("seed*") if p.is_dir()}
    expected = {f"seed{seed}" for seed in SEEDS}
    if found != expected:
        raise RuntimeError(f"ViT block incomplete/extra: missing {sorted(expected-found)}, "
                           f"extra {sorted(found-expected)}")
    boundary._data_bytes(data_root)
    audited = []
    for seed in SEEDS:
        directory = root / f"seed{seed}"
        receipt = _receipt(directory)
        audited.append((receipt, boundary._steps(directory, receipt)))
    if len({record[0][6]["release_commit"] for record in audited}) != 1:
        raise RuntimeError("ViT full block used different source releases")
    if len({base.sha256(root / f"seed{seed}/manifest.json") for seed in SEEDS}) != 1:
        raise RuntimeError("ViT development manifest differs across seeds")
    pilot_gate = _full_pilot_gate(root, audited)
    cost_gate = _full_cost_gate(root, audited, pilot_gate)
    rows = [load_seed(root / f"seed{seed}", data_root, _audited=item)
            for seed, item in zip(SEEDS, audited)]
    if len({row["manifest_sha256"] for row in rows}) != 1:
        raise RuntimeError("ViT development manifest differs across seeds")
    hashes = [row["caps"]["10"]["arms"]["ens_pto"]["prediction_sha256"] for row in rows]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("duplicate ViT PTO predictions across seeds")
    contrasts = {}
    for metric in base.METRICS:
        comparisons = []
        for divisor, control in PRIMARY:
            differences = [row["caps"][str(divisor)]["arms"]["ens_joint"]["allocated"][metric] -
                           row["caps"][str(divisor)]["arms"][control]["allocated"][metric]
                           for row in rows]
            comparisons.append((f"cap_divisor_{divisor}_joint_minus_{control}",
                                base._paired(differences), differences))
        adjusted = base._holm([stat["p"] for _, stat, _ in comparisons]) if metric == "cc_f1" else [None] * 6
        contrasts[metric] = {name: {**stat, "holm_p": p,
                                    "per_seed": dict(zip(SEEDS, diffs))}
                             for (name, stat, diffs), p in zip(comparisons, adjusted)}
    signals = {}
    for divisor in base.DIVISORS:
        primary = [f"cap_divisor_{divisor}_joint_minus_{control}"
                   for control in ("ens_pto", "ens_sham")]
        positive = all(contrasts["cc_f1"][name]["mean"] > 0 and
                       contrasts["cc_f1"][name]["holm_p"] < .05 for name in primary)
        harmed = any(contrasts[metric][name]["interval_available"] and
                     contrasts[metric][name]["hi"] < 0
                     for metric in ("accuracy", "macro_f1", "weighted_f1")
                     for name in primary)
        signals[str(divisor)] = {"positive_vs_pto_and_sham": positive,
                                 "secondary_dominated": harmed,
                                 "registered_exploratory_lead": positive and not harmed}
    report = {"status": "complete_12_seed_vit_exploratory_development",
              "provenance": {"source_sha256": base.source(), "data_file_sha256": base.FILES,
                             "development_manifest_sha256": rows[0]["manifest_sha256"]},
              "seeds": rows, "contrasts": contrasts,
              "pilot_gate_sha256": base.sha256(root / "vit_pilot_gate_recheck.json"),
              "cost_gate": cost_gate,
              "primary_family": [f"cap_divisor_{d}_joint_minus_{control}"
                                 for d, control in PRIMARY],
              "exploratory_signal_by_cap": signals,
              "limitations": ["Development countries were previously viewed; this is exploratory.",
                              "PTO/Clipper is a zero-step post-hoc analogue, not historical Clipper training.",
                              "PHR is a calibrated snapshot direction, not full ALM training.",
                              "Rejected probe probabilities and side weights are not saved; the offline gate checks their logged decision arithmetic and the accepted output, while parameter dose is verified by runner logs.",
                              "ViT and MobileNetV3 have separate seed blocks and must not be pooled."]}
    if output is not None:
        path = Path(output)
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    if len(sys.argv) in (4, 5) and sys.argv[1] == "--gate":
        print(json.dumps(gate(sys.argv[2], sys.argv[3],
                              sys.argv[4] if len(sys.argv) == 5 else None), indent=2))
    elif len(sys.argv) == 4 and sys.argv[1] == "--pilot-score":
        print(json.dumps(pilot_score(sys.argv[2], sys.argv[3]), indent=2))
    elif len(sys.argv) in (3, 4):
        result = main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else None)
        print(json.dumps({"status": result["status"], "primary": result["contrasts"]["cc_f1"]},
                         indent=2))
    else:
        raise SystemExit(__doc__)
