"""Offline integrity gate and scorer for the fmow2 local PHR direction study.

Usage: python analysis/score_fmow_local_alm.py --gate PILOT_ROOT REFERENCE_ROOT
       python analysis/score_fmow_local_alm.py --pilot-score PILOT_ROOT DATA_ROOT
       python analysis/score_fmow_local_alm.py RUN_ROOT DATA_ROOT [OUTPUT_JSON]

The gate hashes, but never opens, the label file or label-bearing manifest.
The full scorer reads only development indices from the verified label array.
"""

import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import score_fmow_local as prior

PILOT = 6300
SEEDS = tuple(range(6301, 6313))
STUDY = "local_alm_direction_v1"
RADIUS = 0.1
RHO = 0.5
CONFIGS = Path(__file__).resolve().parents[1] / "experiments/configs/fmow_local_alm_20260930"
ARMS = ("ens_pto", "ens_joint_fixed", "ens_global_dose", "ens_sham", "ens_phr_local")
PRIMARY = tuple((d, control) for d in prior.DIVISORS
                for control in ("ens_joint_fixed", "ens_pto"))


def _launch_receipt(directory, config, started):
    """Bind queue ownership/completion to the immutable runner provenance."""
    d = Path(directory)
    job = f"{config['seed']}_{'step' if config['snapshot_steps'] else 'ref'}"
    stem = d.parent / f"seed{job}"
    launch = prior._json(stem.with_name(stem.name + ".launch.json"))
    complete = prior._json(stem.with_name(stem.name + ".complete.json"))
    common = {"job": job, "seed": config["seed"],
              "run_root": str(d.parent.resolve()), "output_dir": str(d.resolve())}
    if (not isinstance(launch, dict) or not isinstance(complete, dict) or
            any(launch.get(key) != value or complete.get(key) != value
                for key, value in common.items()) or
            launch.get("host") != complete.get("host") or
            launch.get("gpu_uuid") != complete.get("gpu_uuid") or
            launch.get("release_commit") != complete.get("release_commit") or
            not isinstance(launch.get("host"), str) or not launch["host"] or
            not isinstance(launch.get("gpu_uuid"), str) or
            not launch["gpu_uuid"].startswith("GPU-") or
            not isinstance(launch.get("release_commit"), str) or
            re.fullmatch(r"[0-9a-f]{40}", launch["release_commit"]) is None or
            type(launch.get("gpu_index")) is not int or launch["gpu_index"] < 0 or
            launch.get("precision") != "fp32" or started.get("precision") != "fp32" or
            launch.get("source_sha256") != started["source_sha256"] or
            launch.get("config_sha256") != started["config_sha256"] or
            type(complete.get("exit_code")) is not int or complete["exit_code"] != 0 or
            not isinstance(launch.get("started_utc"), str) or not launch["started_utc"] or
            not isinstance(complete.get("ended_utc"), str) or not complete["ended_utc"]):
        raise RuntimeError(f"{d.name}: launch/completion receipt provenance mismatch")
    return launch


def _receipt(directory):
    """Audit common provenance without parsing any label-bearing artifact."""
    d = Path(directory)
    top = prior._events(d / "events.jsonl")
    train = prior._events(d / "retrain1" / "events.jsonl", terminal="training_completed")
    started, init, done = (prior._one(top, event) for event in
                           ("started", "model_initialized", "completed"))
    config, summary = prior._json(d / "config.json"), prior._json(d / "summary.json")
    if (set(config) != set(prior.RECIPE) | {"seed", "snapshot_steps", "study", "step_radius", "alm_rho"}
            or any(config.get(key) != value or type(config[key]) is not type(value)
                   for key, value in prior.RECIPE.items())
            or type(config.get("seed")) is not int or config["seed"] not in SEEDS + (PILOT,)
            or type(config.get("snapshot_steps")) is not bool
            or (not config["snapshot_steps"] and config["seed"] != PILOT)
            or config.get("study") != STUDY
            or type(config.get("step_radius")) not in (int, float)
            or config["step_radius"] != RADIUS
            or type(config.get("alm_rho")) is not float or config["alm_rho"] != RHO):
        raise RuntimeError(f"{d.name}: config differs from local ALM direction protocol")
    job = f"{config['seed']}_{'step' if config['snapshot_steps'] else 'ref'}"
    input_config = CONFIGS / f"fmow_local_{job}.json"
    if (prior._json(input_config) != config or
            started["config_sha256"] != prior.sha256(input_config)):
        raise RuntimeError(f"{d.name}: input/run config provenance mismatch")
    _launch_receipt(d, config, started)
    if (started["source_sha256"] != prior.source() or started["data_files"] != prior.FILES
            or started["counts"] != {"train": 15841, "stop": 1829, "dev": 1673}
            or started["manifest_sha256"] != prior.sha256(d / "manifest.json")):
        raise RuntimeError(f"{d.name}: source/data/config/manifest provenance mismatch")
    ids, groups = prior._pool_identity(d, started)
    if (summary["seed"] != config["seed"] or summary["quotas"] != started["quotas"]
            or init["initial_sha256"] != summary["initial_sha256"]
            or init["architecture"] != "mobilenet_v3_large" or init["classes"] != prior.CLASSES):
        raise RuntimeError(f"{d.name}: model or quota identity mismatch")
    result = summary["retrain"]
    epochs, best = result["epochs_run"], result["best_epoch"]
    if (not 1 <= best <= epochs <= prior.RECIPE["max_epochs"]
            or done["epochs_run"] != epochs
            or done["task_updates"] != result["task_updates"]
            or result["task_updates"] != math.ceil(15841 / config["batch_size"]) * epochs
            or prior._one(train, "training_completed")["task_updates"] != result["task_updates"]):
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
            raise RuntimeError(f"{d.name}: stopping flag disagrees with losses")
        if improved:
            best_loss, best_epoch, waited = row["stop_loss"], row["epoch"], 0
        else:
            waited += 1
        if waited >= config["patience"] and row["epoch"] < epochs:
            raise RuntimeError(f"{d.name}: training continued after patience")
    if (best != best_epoch or result["best_stop_loss"] != best_loss
            or epochs < config["max_epochs"] and waited < config["patience"]):
        raise RuntimeError(f"{d.name}: stopping endpoint differs from logs")
    if set(summary["pto_snapshot_sha256"]) != {str(e) for e in range(1, epochs + 1)}:
        raise RuntimeError(f"{d.name}: PTO snapshot receipts incomplete")
    for epoch in range(1, epochs + 1):
        path = d / "retrain1" / f"epoch{epoch:02d}.pt"
        if prior.sha256(path) != summary["pto_snapshot_sha256"][str(epoch)]:
            raise RuntimeError(f"{d.name}: PTO snapshot hash mismatch")
    if (prior.sha256(d / "retrain1" / "final_probabilities.pt") !=
            summary["final_probability_sha256"]):
        raise RuntimeError(f"{d.name}: final PTO hash mismatch")
    return config, summary, started, train, ids, groups


def _audit_side(record, pto, side, groups, quota, arm):
    prior._audit_displacement(record)
    before, local_before = prior._count(pto.argmax(1).tolist(), groups)
    if arm != "joint" and not record["applied"]:
        if ({key: value for key, value in record.items() if key != "probability_sha256"} !=
                {"applied": False, "displacement": 0.0} or not torch.equal(pto, side)):
            raise RuntimeError("inactive dose control differs from PTO")
        return
    if (record["hard_before_global"] != before or record["hard_before_local"] != local_before
            or abs(record["soft_before_global"] - float(pto[:, prior.CAPPED].sum())) > 1e-3
            or any(abs(record["soft_before_local"][group] - float(
                pto[[i for i, g in enumerate(groups) if g == group], prior.CAPPED].sum())) > 1e-3
                for group in quota["local_caps"])):
        raise RuntimeError("side-step pre-count differs from PTO")
    after, local_after = prior._count(side.argmax(1).tolist(), groups)
    if record["applied"]:
        gradient_norm = record.get("gradient_norm")
        if (type(gradient_norm) not in (int, float) or not math.isfinite(gradient_norm)
                or gradient_norm <= 0):
            raise RuntimeError("missing/nonfinite side-step gradient norm")
        if (record["hard_after_global"] != after or record["hard_after_local"] != local_after
                or abs(record["radius"] - RADIUS) > 1e-12
                or abs(record["displacement"] - RADIUS) > 1e-5
                or abs(record["soft_after_global"] - float(side[:, prior.CAPPED].sum())) > 1e-3
                or any(abs(record["soft_after_local"][group] - float(
                    side[[i for i, g in enumerate(groups) if g == group], prior.CAPPED].sum())) > 1e-3
                    for group in quota["local_caps"])):
            raise RuntimeError("side-step post-count or fixed dose differs from snapshot")
    elif not torch.equal(pto, side):
        raise RuntimeError("inactive side snapshot differs from PTO")
    if arm == "joint":
        active_global = before > quota["global_cap"]
        active_local = sorted(g for g, count in local_before.items()
                              if count > quota["local_caps"][g])
        if (record["active_global"] is not active_global
                or record["active_local"] != active_local
                or record["applied"] is not bool(active_global or active_local)):
            raise RuntimeError("joint active scopes disagree with hard-before counts")
        # Conflicting derivatives are scientific evidence, not an integrity failure.
        if record["applied"]:
            derivatives = record["scope_directional_derivatives"]
            expected = prior._scope_derivative_keys(record)
            if set(derivatives) != expected or any(not math.isfinite(x) for x in derivatives.values()):
                raise RuntimeError("missing/nonfinite joint scope derivatives")
        for field in ("soft_before_global", "soft_after_global", "directional_soft_delta_global"):
            if field in record and not math.isfinite(record[field]):
                raise RuntimeError(f"nonfinite joint {field}")
        for field in ("soft_before_local", "soft_after_local", "directional_soft_delta_local"):
            if field in record and (set(record[field]) != set(quota["local_caps"]) or
                                    any(not math.isfinite(value) for value in record[field].values())):
                raise RuntimeError(f"nonfinite/incomplete joint {field}")


def _phr_scope_names(record, local_caps):
    """Keep positional PHR scopes distinct while reading old, unambiguous logs."""
    if "scope_derivative_schema" in record:
        if record["scope_derivative_schema"] != "pooled-local-v1":
            raise RuntimeError("unknown PHR scope derivative schema")
        return ["pooled", *["local:" + group for group in sorted(local_caps)]]
    if "global" in local_caps:
        raise RuntimeError("ambiguous legacy PHR pooled/local derivative identity")
    return ["global", *sorted(local_caps)]


def _scope_values(probabilities, groups, quota, *, namespaced=False):
    # The legacy return shape is retained for existing unambiguous fixtures.
    names = _phr_scope_names(
        {"scope_derivative_schema": "pooled-local-v1"} if namespaced else {},
        quota["local_caps"])
    q = probabilities[:, prior.CAPPED]
    values = [float(q.sum()), *[float(q[[g == group for g in groups]].sum())
                               for group in sorted(quota["local_caps"])]]
    caps = [quota["global_cap"], *[quota["local_caps"][group]
                                 for group in sorted(quota["local_caps"])]]
    counts = dict(zip(names, values))
    residuals = [(value - cap) / max(cap, 1) for value, cap in zip(values, caps)]
    return counts, residuals


def _near(actual, expected, message, tol=1e-5):
    if (type(actual) not in (int, float) or not math.isfinite(actual)
            or abs(actual - expected) > tol):
        raise RuntimeError(message)


def _audit_phr(record, pto, side, groups, quota, dual):
    """Recount every PHR scope and its projected dual without development labels."""
    prior._audit_displacement(record)
    names = _phr_scope_names(record, quota["local_caps"])
    if type(record.get("rho")) not in (int, float) or record["rho"] != RHO:
        raise RuntimeError("PHR rho differs from fixed protocol")
    if (len(dual) != len(names) or any(not math.isfinite(x) or x < 0 for x in dual)
            or len(record["dual_before"]) != len(names) or any(
                type(value) not in (int, float) or not math.isfinite(value) or
                abs(value - expected) > 1e-6
                for value, expected in zip(record["dual_before"], dual))):
        raise RuntimeError("PHR dual continuity differs from previous snapshot")
    before, residual_before = _scope_values(pto, groups, quota, namespaced=True)
    after, residual_after = _scope_values(side, groups, quota, namespaced=True)
    for field, matrix, counts in (("before", pto, before), ("after", side, after)):
        hard, local = prior._count(matrix.argmax(1).tolist(), groups)
        if (record[f"hard_{field}_global"] != hard or
                record[f"hard_{field}_local"] != local):
            raise RuntimeError("PHR hard-count recount differs from snapshot")
        _near(record[f"soft_{field}_global"], counts["pooled"], "PHR pooled soft recount differs", 1e-3)
        values = record[f"soft_{field}_local"]
        if set(values) != set(quota["local_caps"]):
            raise RuntimeError("PHR local soft scopes incomplete")
        for name in quota["local_caps"]:
            _near(values[name], counts["local:" + name], "PHR local soft recount differs", 1e-3)
    for field, expected in (("residuals_before", residual_before),
                            ("residuals_after", residual_after)):
        values = record[field]
        if len(values) != len(names):
            raise RuntimeError("PHR residual scopes incomplete")
        for value, target in zip(values, expected):
            _near(value, target, "PHR signed residual recount differs", 1e-6)
    def penalty(residuals):
        return sum((max(0.0, lam + RHO * g) ** 2 - lam ** 2) / (2 * RHO)
                   for lam, g in zip(dual, residuals))
    before_penalty, after_penalty = penalty(residual_before), penalty(residual_after)
    _near(record["penalty_before"], before_penalty, "PHR initial penalty differs",
          1e-5 * max(1.0, abs(before_penalty)))
    _near(record["penalty_after"], after_penalty, "PHR final penalty differs",
          1e-5 * max(1.0, abs(after_penalty)))
    projected = [max(0.0, lam + RHO * g) for lam, g in zip(dual, residual_after)]
    if len(record["dual_after"]) != len(names):
        raise RuntimeError("PHR projected dual scopes incomplete")
    for value, target in zip(record["dual_after"], projected):
        _near(value, target, "PHR projected dual differs", 1e-5 * max(1.0, abs(target)))
    gradient_norm = record["gradient_norm"]
    if (type(gradient_norm) not in (int, float) or not math.isfinite(gradient_norm)
            or gradient_norm < 0 or record["applied"] is not (gradient_norm > 0)):
        raise RuntimeError("PHR gradient/application mismatch")
    if record["applied"]:
        if record.get("activation_reason") != "finite_nonzero_phr_gradient":
            raise RuntimeError("PHR applied activation reason missing")
        if record["radius"] != RADIUS or abs(record["displacement"] - RADIUS) > 1e-5:
            raise RuntimeError("PHR fixed dose differs from protocol")
        derivatives = record["scope_directional_derivatives"]
        if set(derivatives) != set(names) or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in derivatives.values()):
            raise RuntimeError("PHR scope directional derivatives incomplete/nonfinite")
        weighted = sum(max(0.0, lam + RHO * g) * derivatives[name]
                       for lam, g, name in zip(dual, residual_before, names))
        # Each logged derivative is for the normalized residual; their PHR sum
        # must be the negative norm of the parameter gradient on the unit step.
        _near(weighted, -gradient_norm, "PHR directional gradient identity differs",
              1e-3 * max(1.0, gradient_norm))
    elif (not torch.equal(pto, side) or record.get("activation_reason") != "zero_phr_gradient"
          or record.get("scope_directional_derivatives") != {}):
        raise RuntimeError("inactive PHR snapshot or activation record differs from PTO")
    return record["dual_after"]


def _steps(directory, receipt):
    d = Path(directory)
    _, summary, started, events, ids, groups = receipt
    epochs = summary["retrain"]["epochs_run"]
    if set(summary["steps"]) != {str(e) for e in range(1, epochs + 1)}:
        raise RuntimeError(f"{d.name}: side-step epochs incomplete")
    cap_events = [row for row in events if row["event"] == "snapshot_cap"]
    if (len(cap_events) != len(prior.DIVISORS) * epochs or
            {(row["epoch"], row["divisor"]) for row in cap_events} !=
            {(e, divisor) for e in range(1, epochs + 1) for divisor in prior.DIVISORS}):
        raise RuntimeError(f"{d.name}: cap events incomplete or duplicate")
    pto, sides, artifacts = {}, {}, {}
    dual = {divisor: [0.0] * (1 + len(started["quotas"][str(divisor)]["local_caps"]))
            for divisor in prior.DIVISORS}
    for epoch in range(1, epochs + 1):
        path = d / "retrain1" / f"epoch{epoch:02d}.pt"
        pto[epoch] = prior._probabilities(path, len(ids))
        artifacts[str(path.relative_to(d))] = prior.sha256(path)
        for divisor in prior.DIVISORS:
            quota = started["quotas"][str(divisor)]
            records = summary["steps"][str(epoch)][str(divisor)]
            event = next(row for row in cap_events if
                         (row["epoch"], row["divisor"]) == (epoch, divisor))
            if event["quota"] != quota or event["steps"] != records or set(records) != {
                    "joint", "global_dose", "sham", "phr_local"}:
                raise RuntimeError("step event/summary/quota disagreement")
            sides[epoch, divisor] = {}
            for arm in ("joint", "global_dose", "sham", "phr_local"):
                path = d / "retrain1" / f"cap{divisor}" / f"epoch{epoch:02d}_{arm}.pt"
                side = prior._probabilities(path, len(ids))
                if prior.sha256(path) != records[arm]["probability_sha256"]:
                    raise RuntimeError("creation-time side snapshot hash mismatch")
                if arm == "phr_local":
                    dual[divisor] = _audit_phr(records[arm], pto[epoch], side, groups,
                                              quota, dual[divisor])
                else:
                    _audit_side(records[arm], pto[epoch], side, groups, quota, arm)
                sides[epoch, divisor][arm] = side
                artifacts[str(path.relative_to(d))] = prior.sha256(path)
            joint = records["joint"]
            if any(records[arm]["applied"] != joint["applied"] for arm in ("global_dose", "sham")):
                raise RuntimeError("same-dose arm application mismatch")
            if joint["applied"]:
                norms = joint["tensor_displacement_norms"]
                sham = records["sham"]["tensor_displacement_norms"]
                if len(norms) != len(sham) or any(abs(a - b) > 1e-5 for a, b in zip(norms, sham)):
                    raise RuntimeError("sham per-tensor displacement differs from joint")
    final_path = d / "retrain1" / "final_probabilities.pt"
    final = prior._probabilities(final_path, len(ids))
    if not torch.equal(final, pto[summary["retrain"]["best_epoch"]]):
        raise RuntimeError("restored PTO differs from best snapshot")
    artifacts[str(final_path.relative_to(d))] = prior.sha256(final_path)
    return pto, sides, artifacts


def gate(pilot_root, reference_root):
    """Audit the matched pilot using labels neither in memory nor on disk."""
    root = Path(pilot_root)
    jobs = sorted(path.name for path in root.glob("seed*") if path.is_dir())
    if jobs != ["seed6300"]:
        raise RuntimeError("pilot root must contain exactly seed6300")
    pilot = root / "seed6300"
    reference = Path(reference_root) / "seed6300_ref"
    if not reference.is_dir() or sorted(p.name for p in Path(reference_root).glob("seed*")
                                         if p.is_dir()) != ["seed6300_ref"]:
        raise RuntimeError("reference root must contain exactly seed6300_ref")
    a, b = _receipt(pilot), _receipt(reference)
    ac, summary, started, events, ids, groups = a
    bc, ref, ref_started, ref_events, ref_ids, ref_groups = b
    launch, ref_launch = (_launch_receipt(pilot, ac, started),
                          _launch_receipt(reference, bc, ref_started))
    if (ac["seed"] != PILOT or bc["seed"] != PILOT or ac["snapshot_steps"] is not True
            or bc["snapshot_steps"] is not False or (ids, groups) != (ref_ids, ref_groups)
            or prior.sha256(pilot / "manifest.json") != prior.sha256(reference / "manifest.json")
            or started["quotas"] != ref_started["quotas"] or summary["retrain"] != ref["retrain"]
            or summary["initial_sha256"] != ref["initial_sha256"]
            or summary.get("pto_sha256") != ref.get("pto_sha256")
            or started.get("device") != ref_started.get("device")
            or started.get("precision") != ref_started.get("precision")
            or not isinstance(started.get("device"), str) or not started["device"]
            or started.get("precision") != "fp32"
            or launch["host"] != ref_launch["host"]
            or launch["release_commit"] != ref_launch["release_commit"]
            or ref["steps"] or any(row["event"] == "snapshot_cap" for row in ref_events)):
        raise RuntimeError("pilot/reference setup or PTO metadata mismatch")
    if ([(row["epoch"], row["hard_counts"], row["stop_loss"]) for row in events
         if row["event"] == "epoch"] !=
            [(row["epoch"], row["hard_counts"], row["stop_loss"]) for row in ref_events
             if row["event"] == "epoch"]):
        raise RuntimeError("pilot/reference epoch trajectory differs")
    epochs = summary["retrain"]["epochs_run"]
    for name in [f"epoch{e:02d}.pt" for e in range(1, epochs + 1)] + ["final_probabilities.pt"]:
        left = prior._probabilities(pilot / "retrain1" / name, len(ids))
        right = prior._probabilities(reference / "retrain1" / name, len(ids))
        if not torch.equal(left, right):
            raise RuntimeError(f"PTO trajectory differs: {name}")
    _steps(pilot, a)
    active_by_cap = {str(divisor): sum(
        bool(summary["steps"][str(epoch)][str(divisor)]["phr_local"]["applied"])
        for epoch in range(1, epochs + 1)) for divisor in prior.DIVISORS}
    if any(count == 0 for count in active_by_cap.values()):
        raise RuntimeError(f"pilot has no meaningful active PHR step at each cap: {active_by_cap}")
    pilot_done = prior._one(prior._events(pilot / "events.jsonl"), "completed")
    reference_done = prior._one(prior._events(reference / "events.jsonl"), "completed")
    step_seconds, ref_seconds = pilot_done.get("seconds"), reference_done.get("seconds")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
           for value in (step_seconds, ref_seconds)):
        raise RuntimeError("pilot durations missing or nonfinite")
    projected_gpu_hours = (13 * step_seconds + ref_seconds) / 3600
    if projected_gpu_hours > 8:
        raise RuntimeError(f"projected study cost {projected_gpu_hours:.3f} GPU-hours exceeds 8")
    return {"status": "pilot_integrity_pass", "epochs": epochs,
            "pto_equal": True, "development_labels_accessed": False,
            "phr_active_epochs_by_cap": active_by_cap,
            "pilot_step_seconds": step_seconds, "pilot_reference_seconds": ref_seconds,
            "projected_total_gpu_hours": projected_gpu_hours}


def _manifest_and_labels(directory, receipt, data_root):
    d = Path(directory)
    _, summary, started, _, ids, groups = receipt
    manifest = prior._json(d / "manifest.json")
    if (manifest["files"] != prior.FILES or manifest["counts"] != started["counts"]
            or manifest["quotas"] != started["quotas"]
            or manifest["quotas"] != summary["quotas"]
            or manifest["dev_countries"] != ["IRQ", "NLD", "DZA", "PHL", "TUR"]
            or set(manifest["reserved_countries"]) != prior.RESERVED
            or set(manifest["dev_countries"]) & set(manifest["reserved_countries"])):
        raise RuntimeError("manifest data roles/quotas differ from protocol")
    rows = manifest["rows"]
    if (len(rows) != len(ids) or any(set(row) != {"sample_id", "location", "split"}
                                    or row["split"] != "val" for row in rows)
            or [row["sample_id"] for row in rows] != ids
            or [row["location"] for row in rows] != groups):
        raise RuntimeError("manifest is not label-free or pool identity differs")
    labels_path = Path(data_root) / "test_labels.npy"
    if prior.sha256(labels_path) != prior.FILES["test_labels.npy"]:
        raise RuntimeError("development label file hash differs from fixed data")
    all_labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    indices = [int(sample_id.removeprefix("test")) for sample_id in ids]
    if tuple(all_labels.shape) != (3442,) or len(set(indices)) != len(indices) or any(
            index < 0 or index >= len(all_labels) for index in indices):
        raise RuntimeError("development label indices differ from data identity")
    labels = [int(all_labels[index]) for index in indices]
    del all_labels
    if any(label < 0 or label >= prior.CLASSES for label in labels):
        raise RuntimeError("development labels outside class range")
    return labels


def load_seed(directory, data_root, *, allow_pilot=False):
    d = Path(directory)
    receipt = _receipt(d)
    config, summary, _, _, ids, groups = receipt
    seed = int(d.name.removeprefix("seed")) if d.name.startswith("seed") else -1
    if (seed not in ((PILOT,) if allow_pilot else SEEDS)
            or config["seed"] != seed or not config["snapshot_steps"]):
        raise RuntimeError("seed/config differs from local ALM block")
    pto, sides, artifacts = _steps(d, receipt)
    labels = _manifest_and_labels(d, receipt, data_root)
    epochs, best = summary["retrain"]["epochs_run"], summary["retrain"]["best_epoch"]
    window = list(range(max(1, best - 2), epochs + 1))
    out = {"seed": seed, "epochs_run": epochs, "best_epoch": best, "window": window,
           "manifest_sha256": prior.sha256(d / "manifest.json"),
           "config_sha256": prior.sha256(d / "config.json"), "artifacts": artifacts, "caps": {}}
    for divisor in prior.DIVISORS:
        quota = summary["quotas"][str(divisor)]
        ensembles = {"ens_pto": [pto[e] for e in window]}
        for arm, name in (("joint", "ens_joint_fixed"), ("global_dose", "ens_global_dose"),
                          ("sham", "ens_sham"), ("phr_local", "ens_phr_local")):
            ensembles[name] = [sides[e, divisor][arm] for e in window]
        scored = {name: prior._arm_score(torch.stack(values).mean(0), labels, ids, groups, quota)
                  for name, values in ensembles.items()}
        for row in scored.values():
            assigned = row["predictions"]
            row["selected_ids"] = [sample_id for sample_id, prediction in zip(ids, assigned)
                                   if prediction == prior.CAPPED]
            row["class1_confusion"] = {
                "tp": sum(y == prior.CAPPED and p == prior.CAPPED
                          for y, p in zip(labels, assigned)),
                "fp": sum(y != prior.CAPPED and p == prior.CAPPED
                          for y, p in zip(labels, assigned)),
                "fn": sum(y == prior.CAPPED and p != prior.CAPPED
                          for y, p in zip(labels, assigned))}
        joint = scored["ens_joint_fixed"]["predictions"]
        phr = scored["ens_phr_local"]["predictions"]
        baseline = scored["ens_pto"]["predictions"]
        dose = scored["ens_global_dose"]["predictions"]

        def movement(other):
            entries = [i for i in range(len(joint)) if joint[i] == prior.CAPPED and
                       other[i] != prior.CAPPED]
            exits = [i for i in range(len(joint)) if joint[i] != prior.CAPPED and
                     other[i] == prior.CAPPED]
            return {"entries": len(entries), "exits": len(exits),
                    "correct_entries": sum(labels[i] == prior.CAPPED for i in entries),
                    "correct_exits": sum(labels[i] == prior.CAPPED for i in exits)}

        step_diagnostics = {}
        phr_diagnostics = {}
        for epoch in range(1, epochs + 1):
            record = summary["steps"][str(epoch)][str(divisor)]["joint"]
            after_g = record.get("hard_after_global", record["hard_before_global"])
            after_l = record.get("hard_after_local", record["hard_before_local"])
            derivatives = record.get("scope_directional_derivatives", {})
            step_diagnostics[str(epoch)] = {
                "applied": record["applied"],
                "radius": record.get("radius", 0.0),
                "hard_before_global": record["hard_before_global"],
                "hard_before_local": record["hard_before_local"],
                "hard_after_global": after_g,
                "hard_after_local": after_l,
                "raw_joint_feasible_after": after_g <= quota["global_cap"] and all(
                    after_l[g] <= cap for g, cap in quota["local_caps"].items()),
                "scope_directional_derivatives": derivatives,
                "worsening_active_scopes": sorted(g for g, value in derivatives.items() if value >= 0),
            }
            phr_record = summary["steps"][str(epoch)][str(divisor)]["phr_local"]
            phr_after_g = phr_record["hard_after_global"]
            phr_after_l = phr_record["hard_after_local"]
            scope_names = ["global", *sorted(quota["local_caps"])]
            active = [name for name, lam, residual in zip(
                scope_names, phr_record["dual_before"], phr_record["residuals_before"])
                if lam + RHO * residual > 0]
            phr_diagnostics[str(epoch)] = {
                **{key: value for key, value in phr_record.items()
                   if key != "probability_sha256"},
                "active_phr_scopes": active,
                "worsening_active_scopes": [name for name in active if phr_record["applied"] and
                                            phr_record["scope_directional_derivatives"][name] >= 0],
                "raw_phr_feasible_after": phr_after_g <= quota["global_cap"] and all(
                    phr_after_l[g] <= cap for g, cap in quota["local_caps"].items())}
        out["caps"][str(divisor)] = {
            "quota": quota,
            "arms": {name: {key: value for key, value in row.items() if key != "predictions"}
                     for name, row in scored.items()},
            "joint_vs_pto_slots": movement(baseline),
            "joint_vs_global_dose_slots": movement(dose),
            "phr_vs_pto_slots": {"entries": sum(a == prior.CAPPED and b != prior.CAPPED
                                                 for a, b in zip(phr, baseline)),
                                 "exits": sum(a != prior.CAPPED and b == prior.CAPPED
                                              for a, b in zip(phr, baseline)),
                                 "correct_entries": sum(a == prior.CAPPED and b != prior.CAPPED and
                                                        y == prior.CAPPED for a, b, y in
                                                        zip(phr, baseline, labels)),
                                 "correct_exits": sum(a != prior.CAPPED and b == prior.CAPPED and
                                                      y == prior.CAPPED for a, b, y in
                                                      zip(phr, baseline, labels))},
            "all_epoch_step_diagnostics": step_diagnostics,
            "joint_active_epochs": sum(bool(summary["steps"][str(e)][str(divisor)]["joint"]["applied"])
                                       for e in window),
            "phr_active_epochs": sum(bool(summary["steps"][str(e)][str(divisor)]["phr_local"]["applied"])
                                     for e in window),
            "phr_all_epoch_diagnostics": phr_diagnostics,
            "raw_phr_infeasible_epochs": sum(not row["raw_phr_feasible_after"]
                                             for row in phr_diagnostics.values()),
            "raw_joint_infeasible_epochs": sum(not row["raw_joint_feasible_after"]
                                               for row in step_diagnostics.values())}
    return out


def pilot_score(pilot_root, data_root):
    """Separate, explicit offline pilot metric check; never part of label-blind gate."""
    root = Path(pilot_root)
    found = sorted(path.name for path in root.glob("seed*") if path.is_dir())
    if found != ["seed6300"]:
        raise RuntimeError("pilot scoring root must contain exactly seed6300")
    row = load_seed(root / "seed6300", data_root, allow_pilot=True)
    return {"status": "pilot_metrics_exploratory_not_for_setting_selection", "seed": row,
            "development_labels_accessed_offline": True}


def main(run_root, data_root, output=None):
    root = Path(run_root)
    found = {path.name for path in root.glob("seed*") if path.is_dir()}
    expected = {f"seed{seed}" for seed in SEEDS}
    if found != expected:
        raise RuntimeError(f"local ALM 12-seed block incomplete or extra: missing {sorted(expected - found)}, "
                           f"extra {sorted(found - expected)}")
    rows = [load_seed(root / f"seed{seed}", data_root) for seed in SEEDS]
    if len({row["manifest_sha256"] for row in rows}) != 1:
        raise RuntimeError("development manifest differs across seeds")
    pto_hashes = [row["caps"]["10"]["arms"]["ens_pto"]["prediction_sha256"] for row in rows]
    if len(set(pto_hashes)) != len(pto_hashes):
        raise RuntimeError("duplicate PTO predictions across seeds")
    contrasts = {}
    attribution = {}
    for metric in prior.METRICS:
        comparisons = []
        for divisor, control in PRIMARY:
            differences = [row["caps"][str(divisor)]["arms"]["ens_phr_local"]["allocated"][metric] -
                           row["caps"][str(divisor)]["arms"][control]["allocated"][metric]
                           for row in rows]
            comparisons.append((f"cap_divisor_{divisor}_phr_minus_{control}",
                                prior._paired(differences), differences))
        corrections = prior._holm([item[1]["p"] for item in comparisons]) if metric == "cc_f1" else [None] * 4
        contrasts[metric] = {name: {**stat, "holm_p": corrected,
                                   "per_seed": dict(zip(SEEDS, differences))}
                             for (name, stat, differences), corrected in zip(comparisons, corrections)}
        attribution[metric] = {}
        for divisor in prior.DIVISORS:
            differences = [row["caps"][str(divisor)]["arms"]["ens_phr_local"]["allocated"][metric] -
                           row["caps"][str(divisor)]["arms"]["ens_sham"]["allocated"][metric]
                           for row in rows]
            attribution[metric][f"cap_divisor_{divisor}_phr_minus_sham"] = {
                **prior._paired(differences), "per_seed": dict(zip(SEEDS, differences))}
    report = {"status": "complete_12_seed_exploratory_development",
              "provenance": {"source_sha256": prior.source(), "data_file_sha256": prior.FILES,
                             "development_manifest_sha256": rows[0]["manifest_sha256"]},
              "seeds": rows, "contrasts": contrasts, "sham_attribution": attribution,
              "primary_family": [f"cap_divisor_{d}_phr_minus_{control}" for d, control in PRIMARY],
              "limitations": ["Previously viewed development countries; not independent confirmation.",
                              "Snapshot PHR direction is not full ALM training.",
                              "Arm activation and realized dose may differ despite a common maximum radius.",
                              "Raw cap violations and worsening scope derivatives are retained diagnostics."]}
    report["exploratory_signal_by_cap"] = {}
    for divisor in prior.DIVISORS:
        names = [f"cap_divisor_{divisor}_phr_minus_{control}" for control in
                 ("ens_joint_fixed", "ens_pto")]
        primary_positive = all(contrasts["cc_f1"][name]["mean"] > 0 and
                               contrasts["cc_f1"][name]["holm_p"] < .05 for name in names)
        secondary_dominated = any(
            contrasts[metric][name]["interval_available"] and
            contrasts[metric][name]["hi"] < 0
            for metric in ("accuracy", "macro_f1", "weighted_f1") for name in names)
        report["exploratory_signal_by_cap"][str(divisor)] = {
            "both_holm_primary_positive": primary_positive,
            "secondary_dominated": secondary_dominated,
            "meets_registered_exploratory_signal": primary_positive and not secondary_dominated}
    if output is not None:
        path = Path(output)
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--gate":
        print(json.dumps(gate(sys.argv[2], sys.argv[3]), indent=2))
    elif len(sys.argv) == 4 and sys.argv[1] == "--pilot-score":
        print(json.dumps(pilot_score(sys.argv[2], sys.argv[3]), indent=2))
    elif len(sys.argv) in (3, 4):
        result = main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else None)
        print(json.dumps({"status": result["status"], "primary": result["contrasts"]["cc_f1"]}, indent=2))
    else:
        raise SystemExit(__doc__)
