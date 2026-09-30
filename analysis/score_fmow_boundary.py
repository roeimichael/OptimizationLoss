"""Independent, label-blind gate and offline scorer for local_boundary_v1.

Usage: python analysis/score_fmow_boundary.py --gate PILOT_ROOT REF_ROOT
       python analysis/score_fmow_boundary.py --pilot-score PILOT_ROOT DATA_ROOT
       python analysis/score_fmow_boundary.py FULL_ROOT DATA_ROOT [OUTPUT_JSON]

Only the explicit scoring paths open development labels. The gate reads the
label-free pool identity, saved probabilities, and hash/queue receipts.
"""

import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import score_fmow_local as base
from analysis import score_fmow_local_alm as alm

PILOT = 6400
SEEDS = tuple(range(6401, 6413))
STUDY = "local_boundary_v1"
CONFIGS = Path(__file__).resolve().parents[1] / "experiments/configs/fmow_local_boundary_20260930"
RADIUS = 0.1
RHO = 0.5
ARMS = ("joint", "global_dose", "sham", "phr_local")
SCORED_ARMS = {"pto": "ens_pto", "joint": "ens_joint", "global_dose": "ens_global_dose",
               "sham": "ens_sham", "phr_local": "ens_phr_local"}
CONTROLS = ("ens_pto", "ens_sham", "ens_phr_local")
PRIMARY = tuple((divisor, control) for divisor in base.DIVISORS for control in CONTROLS)


def _near(actual, expected, what, tol=1e-5):
    if (type(actual) not in (float, int) or not math.isfinite(actual) or
            abs(actual - expected) > tol):
        raise RuntimeError(f"{what}: expected {expected}, found {actual}")


def _receipt(directory):
    """Recount immutable input, complete PTO training, and label-free quotas."""
    d = Path(directory)
    top = base._events(d / "events.jsonl")
    train = base._events(d / "retrain1/events.jsonl", terminal="training_completed")
    started, init, done = (base._one(top, name) for name in
                           ("started", "model_initialized", "completed"))
    config, summary = base._json(d / "config.json"), base._json(d / "summary.json")
    expected_keys = set(base.RECIPE) | {"seed", "snapshot_steps", "study", "step_radius", "alm_rho"}
    if (set(config) != expected_keys or
            any(config.get(key) != value or type(config[key]) is not type(value)
                for key, value in base.RECIPE.items()) or
            type(config.get("seed")) is not int or config["seed"] not in SEEDS + (PILOT,) or
            type(config.get("snapshot_steps")) is not bool or
            not config["snapshot_steps"] and config["seed"] != PILOT or
            config.get("study") != STUDY or
            type(config.get("step_radius")) is not float or config["step_radius"] != RADIUS or
            type(config.get("alm_rho")) is not float or config["alm_rho"] != RHO):
        raise RuntimeError(f"{d.name}: config differs from boundary protocol")
    job = f"{config['seed']}_{'step' if config['snapshot_steps'] else 'ref'}"
    input_config = CONFIGS / f"fmow_local_{job}.json"
    if base._json(input_config) != config or started["config_sha256"] != base.sha256(input_config):
        raise RuntimeError(f"{d.name}: input/run config provenance mismatch")
    launch = alm._launch_receipt(d, config, started)
    if (started["source_sha256"] != base.source() or started["data_files"] != base.FILES or
            started["counts"] != {"train": 15841, "stop": 1829, "dev": 1673} or
            started["manifest_sha256"] != base.sha256(d / "manifest.json")):
        raise RuntimeError(f"{d.name}: source/data/manifest provenance mismatch")
    ids, groups = base._pool_identity(d, started)
    if (summary["seed"] != config["seed"] or summary["quotas"] != started["quotas"] or
            init["initial_sha256"] != summary["initial_sha256"] or
            init["architecture"] != "mobilenet_v3_large" or init["classes"] != base.CLASSES):
        raise RuntimeError(f"{d.name}: model or quota identity mismatch")
    result = summary["retrain"]
    epochs, best = result["epochs_run"], result["best_epoch"]
    if (type(epochs) is not int or type(best) is not int or
            not 1 <= best <= epochs <= base.RECIPE["max_epochs"] or
            done["epochs_run"] != epochs or done["task_updates"] != result["task_updates"] or
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


def _counts(probabilities, groups, quota):
    hard, local_hard = base._count(probabilities.argmax(1).tolist(), groups)
    q = probabilities[:, base.CAPPED]
    local_soft = {group: float(q[[g == group for g in groups]].sum())
                  for group in sorted(quota["local_caps"])}
    pooled_soft = float(q.sum())
    if not math.isclose(pooled_soft, math.fsum(local_soft.values()), rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError("pooled/country soft partition mismatch")
    return hard, local_hard, pooled_soft, local_soft


def _record_counts(record, prefix, actual):
    hard, local_hard, soft, local_soft = actual
    if record[f"hard_{prefix}_global"] != hard or record[f"hard_{prefix}_local"] != local_hard:
        raise RuntimeError(f"{prefix} hard count differs from probabilities")
    _near(record[f"soft_{prefix}_global"], soft, f"{prefix} pooled soft", 1e-3)
    if set(record[f"soft_{prefix}_local"]) != set(local_soft):
        raise RuntimeError(f"{prefix} local soft scopes incomplete")
    for group, value in local_soft.items():
        _near(record[f"soft_{prefix}_local"][group], value,
              f"{prefix} local soft {group}", 1e-3)


def _policy(record, before, after, quota, *, require_local_hard):
    """Independently reconstruct the bounded decision from saved probe evidence."""
    policy = record["boundary_policy"]
    if (not isinstance(policy, dict) or type(policy.get("applied")) is not bool or
            type(policy.get("radius")) not in (float, int) or
            not isinstance(policy.get("probes"), list)):
        raise RuntimeError("boundary policy record incomplete")
    if policy["applied"] is not record["applied"] or policy["radius"] != record["radius"]:
        raise RuntimeError("boundary decision/step mismatch")
    if record["applied"]:
        if policy.get("reason") != "accepted" or not policy["probes"]:
            raise RuntimeError("accepted boundary step lacks accepted probe")
    elif record["radius"] != 0 or record["displacement"] != 0:
        raise RuntimeError("rejected boundary direction moved parameters")
    caps = {"pooled": quota["global_cap"], **{
        f"local:{g}": cap for g, cap in quota["local_caps"].items()}}
    soft0 = {"pooled": before[2], **{f"local:{g}": value for g, value in before[3].items()}}
    violation0 = {name: max(0.0, (value - caps[name]) / max(caps[name], 1))
                  for name, value in soft0.items()}
    total0 = math.fsum(violation0.values())
    if policy["reason"] in ("no_hard_active_scope", "zero_phr_gradient"):
        if policy["probes"] or record["applied"]:
            raise RuntimeError("inactive direction has boundary probes")
        return
    required = {"initial_positive_violations", "initial_total_positive_violation",
                "pooled_hard_floor", "pooled_soft_floor", "initial_radius"}
    if not required <= set(policy):
        raise RuntimeError("initial boundary evidence incomplete")
    if set(policy["initial_positive_violations"]) != set(caps):
        raise RuntimeError("initial boundary violation scopes differ")
    for name, value in violation0.items():
        _near(policy["initial_positive_violations"][name], value, "initial violation")
    _near(policy["initial_total_positive_violation"], total0, "initial violation total")
    _near(policy["pooled_hard_floor"], max(0, min(before[0], caps["pooled"]) - 1),
          "pooled hard floor")
    _near(policy["pooled_soft_floor"], max(0.0, min(before[2], caps["pooled"]) - 1),
          "pooled soft floor")
    derivatives = record["scope_directional_derivatives"]
    if set(derivatives) != set(caps) or any(type(value) not in (int, float) or
                                           not math.isfinite(value) for value in derivatives.values()):
        raise RuntimeError("boundary derivatives incomplete/nonfinite")
    conflicts = sorted(name for name, value in violation0.items()
                       if value > 0 and derivatives[name] >= 0)
    if total0 == 0:
        expected_reason, initial = "no_positive_violation", 0.0
    elif conflicts:
        expected_reason, initial = "conflicting_direction", 0.0
        if policy.get("conflicting_scopes") != conflicts:
            raise RuntimeError("boundary conflicting scopes mismatch")
    else:
        initial = min(RADIUS, *(violation0[name] / -derivatives[name]
                                for name in violation0 if violation0[name] > 0))
        expected_reason = "search" if initial > 0 and math.isfinite(initial) else "unrepresentable_initial_radius"
    # The runner accumulated FP32 soft counts on GPU. Recounting saved
    # probabilities can move the derived radius by a few nanounits.
    radius_tolerance = max(1e-7, 1e-5 * abs(initial))
    _near(policy["initial_radius"], initial if expected_reason == "search" else 0.0,
          "boundary initial radius",
          radius_tolerance if expected_reason == "search" else 1e-12)
    if expected_reason != "search":
        if policy["reason"] != expected_reason or policy["probes"]:
            raise RuntimeError("boundary skip reason/probe mismatch")
        return
    if not 1 <= len(policy["probes"]) <= 13:
        raise RuntimeError("boundary probe count differs from fixed schedule")
    accepted = False
    for i, probe in enumerate(policy["probes"]):
        expected_radius = initial / (2 ** i)
        if probe.get("halving") != i:
            raise RuntimeError("boundary halving index mismatch")
        _near(probe.get("radius"), expected_radius, "boundary probe radius",
              max(1e-12, radius_tolerance / (2 ** i)))
        if type(probe.get("pooled_hard")) is not int or probe["pooled_hard"] < 0:
            raise RuntimeError("invalid boundary probe hard count")
        if set(probe.get("local_soft", {})) != set(quota["local_caps"]):
            raise RuntimeError("boundary probe local scopes incomplete")
        observed_soft = {"pooled": probe["pooled_soft"], **{
            f"local:{g}": probe["local_soft"][g] for g in quota["local_caps"]}}
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
               for v in observed_soft.values()) or not math.isclose(
                   observed_soft["pooled"], math.fsum(probe["local_soft"].values()),
                   rel_tol=1e-5, abs_tol=1e-5):
            raise RuntimeError("boundary probe nonfinite/partition mismatch")
        observed_v = {name: max(0.0, (value - caps[name]) / max(caps[name], 1))
                      for name, value in observed_soft.items()}
        if set(probe.get("positive_violations", {})) != set(caps):
            raise RuntimeError("boundary probe violation scopes mismatch")
        for name, value in observed_v.items():
            _near(probe["positive_violations"][name], value, "boundary probe violation")
        total = math.fsum(observed_v.values())
        _near(probe["total_positive_violation"], total, "boundary probe total violation")
        reasons = []
        if probe["pooled_hard"] < max(0, min(before[0], caps["pooled"]) - 1):
            reasons.append("pooled_hard_floor")
        if observed_soft["pooled"] < max(0.0, min(before[2], caps["pooled"]) - 1):
            reasons.append("pooled_soft_floor")
        for name in sorted(caps):
            if observed_v[name] > violation0[name] + 1e-6:
                reasons.append(f"worsened_soft_violation:{name}")
        if not total <= total0 - 1e-6:
            reasons.append("insufficient_total_violation_reduction")
        if probe.get("rejections") != reasons or probe.get("accepted") is not (not reasons):
            raise RuntimeError("boundary probe decision differs from counts")
        if require_local_hard and (set(probe.get("local_hard", {})) != set(quota["local_caps"]) or
                                   any(type(v) is not int or v < 0 for v in probe["local_hard"].values()) or
                                   sum(probe["local_hard"].values()) != probe["pooled_hard"]):
            raise RuntimeError("boundary probe local hard scopes incomplete")
        if not reasons:
            if i != len(policy["probes"]) - 1:
                raise RuntimeError("boundary search continued after acceptance")
            accepted = True
    if accepted != record["applied"]:
        raise RuntimeError("boundary acceptance differs from step")
    if policy["reason"] != ("accepted" if accepted else "no_acceptable_probe"):
        raise RuntimeError("boundary terminal reason mismatch")
    if accepted:
        last = policy["probes"][-1]
        _near(record["radius"], last["radius"], "accepted boundary radius", 1e-12)
        if after[0] != last["pooled_hard"]:
            raise RuntimeError("accepted probe hard count differs from side output")
        if require_local_hard and after[1] != last["local_hard"]:
            raise RuntimeError("accepted probe local hard differs from side output")
        _near(after[2], last["pooled_soft"], "accepted probe pooled soft", 1e-5)
        for group, value in after[3].items():
            _near(value, last["local_soft"][group], "accepted probe local soft", 1e-5)


def _audit_side(record, pto, side, groups, quota, arm, dual=None):
    base._audit_displacement(record)
    before, after = _counts(pto, groups, quota), _counts(side, groups, quota)
    _record_counts(record, "before", before)
    _record_counts(record, "after", after)
    if not record["applied"] and not torch.equal(pto, side):
        raise RuntimeError("skipped boundary side differs from PTO")
    if record["applied"] and not 0 < record["radius"] <= RADIUS:
        raise RuntimeError("boundary radius exceeds ceiling")
    if arm in ("joint", "phr_local"):
        if arm == "joint":
            active_global = before[0] > quota["global_cap"]
            active_local = sorted(g for g, count in before[1].items()
                                  if count > quota["local_caps"][g])
            if (record["active_global"] is not active_global or
                    record["active_local"] != active_local):
                raise RuntimeError("joint hard-active scopes mismatch")
            if not active_global and not active_local:
                if record["boundary_policy"]["reason"] != "no_hard_active_scope":
                    raise RuntimeError("joint inactive reason mismatch")
            else:
                norm = record.get("gradient_norm")
                if type(norm) not in (float, int) or not math.isfinite(norm) or norm <= 0:
                    raise RuntimeError("joint gradient norm missing/nonfinite")
                derivatives = record["scope_directional_derivatives"]
                weighted = (derivatives["pooled"] if active_global else 0.0) + sum(
                    derivatives[f"local:{g}"] for g in active_local)
                _near(weighted, -norm, "joint directional gradient identity",
                      1e-3 * max(1.0, norm))
            _policy(record, before, after, quota, require_local_hard=True)
        else:
            if (record["rho"] != RHO or dual is None or
                    len(record["dual_before"]) != len(dual)):
                raise RuntimeError("PHR rho/dual scope mismatch")
            for actual, target in zip(record["dual_before"], dual):
                _near(actual, target, "PHR dual continuity", 1e-6)
            caps = [quota["global_cap"], *[quota["local_caps"][g]
                                                for g in sorted(quota["local_caps"])]]
            soft = [before[2], *[before[3][g] for g in sorted(quota["local_caps"])]]
            after_soft = [after[2], *[after[3][g] for g in sorted(quota["local_caps"])]]
            residual = [(value - cap) / max(cap, 1) for value, cap in zip(soft, caps)]
            residual_after = [(value - cap) / max(cap, 1)
                              for value, cap in zip(after_soft, caps)]
            for field, target in (("residuals_before", residual),
                                  ("residuals_after", residual_after)):
                if len(record[field]) != len(target):
                    raise RuntimeError("PHR residual scope count mismatch")
                for value, expected in zip(record[field], target):
                    _near(value, expected, "PHR residual recount", 1e-6)
            def penalty(values):
                return sum((max(0.0, lam + RHO * g) ** 2 - lam ** 2) / (2 * RHO)
                           for lam, g in zip(dual, values))
            _near(record["penalty_before"], penalty(residual), "PHR before penalty", 1e-4)
            _near(record["penalty_after"], penalty(residual_after), "PHR after penalty", 1e-4)
            projected = [max(0.0, lam + RHO * g) for lam, g in zip(dual, residual_after)]
            if len(record["dual_after"]) != len(projected):
                raise RuntimeError("PHR projected dual scope mismatch")
            for value, expected in zip(record["dual_after"], projected):
                _near(value, expected, "PHR projected dual", 1e-5)
            norm = record["gradient_norm"]
            if type(norm) not in (float, int) or not math.isfinite(norm) or norm < 0:
                raise RuntimeError("PHR gradient norm invalid")
            if norm > 0:
                derivatives = record["scope_directional_derivatives"]
                names = ["global", *sorted(quota["local_caps"])]
                if set(derivatives) != set(names):
                    raise RuntimeError("PHR directional derivatives incomplete")
                weighted = sum(max(0.0, lam + RHO * g) * derivatives[name]
                               for lam, g, name in zip(dual, residual, names))
                _near(weighted, -norm, "PHR directional gradient identity",
                      1e-3 * max(1.0, norm))
                policy_record = {**record, "scope_directional_derivatives": {
                    "pooled": derivatives["global"],
                    **{f"local:{g}": derivatives[g] for g in quota["local_caps"]}}}
                _policy(policy_record, before, after, quota, require_local_hard=False)
            elif (record["applied"] or record["boundary_policy"].get("applied") is not False or
                  record["boundary_policy"]["reason"] != "zero_phr_gradient" or
                  record["scope_directional_derivatives"] != {} or
                  record["boundary_policy"].get("probes") != [] or
                  record["boundary_policy"].get("radius") != 0):
                raise RuntimeError("inactive PHR gradient/decision mismatch")
            expected_activation = ("boundary_" + record["boundary_policy"]["reason"]
                                   if norm > 0 else "zero_phr_gradient")
            if record["activation_reason"] != expected_activation:
                raise RuntimeError("PHR activation reason mismatch")
            return projected
    elif arm in ("global_dose", "sham"):
        if not record["applied"]:
            if (record["radius"] != 0 or record["skip_reason"] != "joint_boundary_skip" or
                    any(value != 0 for value in record["tensor_displacement_norms"])):
                raise RuntimeError("inactive matched control differs from joint skip")
    return dual


def _steps(directory, receipt):
    d = Path(directory)
    _, summary, started, events, ids, groups, _ = receipt
    epochs = summary["retrain"]["epochs_run"]
    if set(summary["steps"]) != {str(e) for e in range(1, epochs + 1)}:
        raise RuntimeError("boundary side-step epochs incomplete")
    cap_events = [row for row in events if row["event"] == "snapshot_cap"]
    if (len(cap_events) != len(base.DIVISORS) * epochs or
            {(row["epoch"], row["divisor"]) for row in cap_events} !=
            {(epoch, divisor) for epoch in range(1, epochs + 1)
             for divisor in base.DIVISORS}):
        raise RuntimeError("boundary cap events incomplete or duplicate")
    pto, sides, artifacts = {}, {}, {}
    duals = {divisor: [0.0] * (1 + len(started["quotas"][str(divisor)]["local_caps"]))
             for divisor in base.DIVISORS}
    for epoch in range(1, epochs + 1):
        path = d / "retrain1" / f"epoch{epoch:02d}.pt"
        pto[epoch] = base._probabilities(path, len(ids))
        artifacts[str(path.relative_to(d))] = base.sha256(path)
        for divisor in base.DIVISORS:
            quota = started["quotas"][str(divisor)]
            records = summary["steps"][str(epoch)][str(divisor)]
            event = next(row for row in cap_events if
                         row["epoch"] == epoch and row["divisor"] == divisor)
            if event["quota"] != quota or event["steps"] != records or set(records) != set(ARMS):
                raise RuntimeError("boundary step event/summary/quota mismatch")
            sides[epoch, divisor] = {}
            for arm in ARMS:
                path = d / "retrain1" / f"cap{divisor}" / f"epoch{epoch:02d}_{arm}.pt"
                matrix = base._probabilities(path, len(ids))
                digest = base.sha256(path)
                if digest != records[arm]["probability_sha256"]:
                    raise RuntimeError("boundary side artifact hash mismatch")
                duals[divisor] = _audit_side(records[arm], pto[epoch], matrix, groups,
                                            quota, arm, duals[divisor])
                sides[epoch, divisor][arm] = matrix
                artifacts[str(path.relative_to(d))] = digest
            joint = records["joint"]
            if any(records[arm]["applied"] is not joint["applied"]
                   for arm in ("global_dose", "sham")):
                raise RuntimeError("matched control application differs from joint")
            if joint["applied"]:
                for arm in ("global_dose", "sham"):
                    _near(records[arm]["radius"], joint["radius"],
                          "matched control radius")
                    _near(records[arm]["displacement"], joint["displacement"],
                          "matched control dose")
                norm1, norm2 = joint["tensor_displacement_norms"], records["sham"]["tensor_displacement_norms"]
                if len(norm1) != len(norm2):
                    raise RuntimeError("sham tensor dose count mismatch")
                for a, b in zip(norm1, norm2):
                    _near(a, b, "sham per-tensor dose")
    final_path = d / "retrain1/final_probabilities.pt"
    final = base._probabilities(final_path, len(ids))
    if not torch.equal(final, pto[summary["retrain"]["best_epoch"]]):
        raise RuntimeError("restored PTO differs from best snapshot")
    artifacts[str(final_path.relative_to(d))] = base.sha256(final_path)
    return pto, sides, artifacts


def gate(pilot_root, reference_root):
    """Audit matched pilot without reading any outcome labels."""
    pilot_root, reference_root = Path(pilot_root), Path(reference_root)
    if sorted(p.name for p in pilot_root.glob("seed*") if p.is_dir()) != ["seed6400"]:
        raise RuntimeError("pilot root must contain exactly seed6400")
    if sorted(p.name for p in reference_root.glob("seed*") if p.is_dir()) != ["seed6400_ref"]:
        raise RuntimeError("reference root must contain exactly seed6400_ref")
    a, b = _receipt(pilot_root / "seed6400"), _receipt(reference_root / "seed6400_ref")
    ac, summary, started, events, ids, groups, launch = a
    bc, ref, ref_started, ref_events, ref_ids, ref_groups, ref_launch = b
    if (ac["seed"] != PILOT or bc["seed"] != PILOT or
            ac["snapshot_steps"] is not True or bc["snapshot_steps"] is not False or
            (ids, groups) != (ref_ids, ref_groups) or
            base.sha256(pilot_root / "seed6400/manifest.json") !=
            base.sha256(reference_root / "seed6400_ref/manifest.json") or
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
        raise RuntimeError("pilot/reference setup or PTO metadata mismatch")
    data_root = Path(launch.get("data_root", ""))
    if not data_root.is_dir():
        raise RuntimeError("pilot data root absent")
    for name, expected in base.FILES.items():
        if base.sha256(data_root / name) != expected:
            raise RuntimeError(f"pilot data byte hash differs: {name}")
    if ([(row["epoch"], row["hard_counts"], row["stop_loss"]) for row in events
         if row["event"] == "epoch"] !=
            [(row["epoch"], row["hard_counts"], row["stop_loss"]) for row in ref_events
             if row["event"] == "epoch"]):
        raise RuntimeError("pilot/reference epoch trajectory differs")
    epochs = summary["retrain"]["epochs_run"]
    for name in [f"epoch{e:02d}.pt" for e in range(1, epochs + 1)] + ["final_probabilities.pt"]:
        left = base._probabilities(pilot_root / "seed6400/retrain1" / name, len(ids))
        right = base._probabilities(reference_root / "seed6400_ref/retrain1" / name, len(ids))
        if not torch.equal(left, right):
            raise RuntimeError(f"PTO trajectory differs: {name}")
    _steps(pilot_root / "seed6400", a)
    step_done = base._one(base._events(pilot_root / "seed6400/events.jsonl"), "completed")
    ref_done = base._one(base._events(reference_root / "seed6400_ref/events.jsonl"), "completed")
    seconds = (step_done.get("seconds"), ref_done.get("seconds"))
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in seconds):
        raise RuntimeError("pilot durations missing or nonfinite")
    projected = (13 * seconds[0] + seconds[1]) / 3600
    if projected > 8:
        raise RuntimeError(f"projected boundary study cost {projected:.3f} exceeds 8 GPU-hours")
    return {"status": "pilot_integrity_pass", "epochs": epochs, "pto_equal": True,
            "development_labels_accessed": False, "step_seconds": seconds[0],
            "reference_seconds": seconds[1], "projected_total_gpu_hours": projected,
            "joint_applied_by_cap": {str(divisor): sum(
                bool(summary["steps"][str(epoch)][str(divisor)]["joint"]["applied"])
                for epoch in range(1, epochs + 1)) for divisor in base.DIVISORS}}


def _movement(selected, control, labels, ids):
    """Report actual selected people, including correct entries and exits."""
    entered = [i for i, (a, b) in enumerate(zip(selected, control))
               if a == base.CAPPED and b != base.CAPPED]
    exited = [i for i, (a, b) in enumerate(zip(selected, control))
              if a != base.CAPPED and b == base.CAPPED]
    return {"entries": len(entered), "exits": len(exited),
            "correct_entries": sum(labels[i] == base.CAPPED for i in entered),
            "correct_exits": sum(labels[i] == base.CAPPED for i in exited),
            "entry_ids": [ids[i] for i in entered], "exit_ids": [ids[i] for i in exited]}


def _data_bytes(data_root):
    """Hash, without decoding, every fixed data file including labels."""
    root = Path(data_root)
    if not root.is_dir():
        raise RuntimeError("fixed data directory absent")
    for name, expected in base.FILES.items():
        if base.sha256(root / name) != expected:
            raise RuntimeError(f"fixed data byte hash differs: {name}")


def load_seed(directory, data_root, *, allow_pilot=False, _audited=None):
    """Score a fully audited seed using only the fixed development identities."""
    d, data_root = Path(directory), Path(data_root)
    receipt = _receipt(d) if _audited is None else _audited[0]
    config, summary, _, _, ids, groups, launch = receipt
    expected = (PILOT,) if allow_pilot else SEEDS
    if (d.name != f"seed{config['seed']}" or config["seed"] not in expected or
            config["snapshot_steps"] is not True):
        raise RuntimeError("seed/config differs from fixed boundary block")
    if Path(launch.get("data_root", "")).resolve() != data_root.resolve():
        raise RuntimeError("scoring data root differs from launch")
    pto, sides, artifacts = _steps(d, receipt) if _audited is None else _audited[1]
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
        for arm in ARMS:
            ensembles[SCORED_ARMS[arm]] = [sides[e, divisor][arm] for e in window]
        scored = {name: base._arm_score(torch.stack(values).mean(0), labels, ids, groups, quota)
                  for name, values in ensembles.items()}
        for arm in scored.values():
            selected = arm["predictions"]
            arm["selected_ids"] = [sample_id for sample_id, value in zip(ids, selected)
                                   if value == base.CAPPED]
            if len(arm["selected_ids"]) != quota["global_cap"]:
                raise RuntimeError("allocated cap did not fill declared pooled slots")
            arm["class1_confusion"] = {
                "tp": sum(y == base.CAPPED and p == base.CAPPED
                          for y, p in zip(labels, selected)),
                "fp": sum(y != base.CAPPED and p == base.CAPPED
                          for y, p in zip(labels, selected)),
                "fn": sum(y == base.CAPPED and p != base.CAPPED
                          for y, p in zip(labels, selected))}
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
            "slot_turnover": {f"joint_vs_{name}": _movement(joint, row["predictions"],
                                                            labels, ids)
                              for name, row in scored.items() if name != "ens_joint"},
            "all_epoch_step_diagnostics": {
                str(epoch): {arm: {key: value for key, value in
                                   summary["steps"][str(epoch)][str(divisor)][arm].items()
                                   if key != "probability_sha256"}
                             for arm in ARMS} for epoch in range(1, epochs + 1)},
            "applied_epochs": {arm: sum(bool(summary["steps"][str(epoch)][str(divisor)][arm]["applied"])
                                        for epoch in range(1, epochs + 1)) for arm in ARMS}}
    return out


def pilot_score(pilot_root, data_root):
    """Separate exploratory score, callable only after a passed pilot gate."""
    root = Path(pilot_root)
    if sorted(p.name for p in root.glob("seed*") if p.is_dir()) != ["seed6400"]:
        raise RuntimeError("pilot scoring root must contain exactly seed6400")
    _data_bytes(data_root)
    return {"status": "pilot_metrics_exploratory_not_for_setting_selection",
            "seed": load_seed(root / "seed6400", data_root, allow_pilot=True),
            "development_labels_accessed_offline": True}


def main(run_root, data_root, output=None):
    """Require the whole fixed denominator, then score all arms and six tests."""
    root = Path(run_root)
    found = {p.name for p in root.glob("seed*") if p.is_dir()}
    expected = {f"seed{seed}" for seed in SEEDS}
    if found != expected:
        raise RuntimeError(f"boundary block incomplete/extra: missing {sorted(expected-found)}, "
                           f"extra {sorted(found-expected)}")
    _data_bytes(data_root)
    audited = []
    for seed in SEEDS:
        directory = root / f"seed{seed}"
        receipt = _receipt(directory)
        audited.append((receipt, _steps(directory, receipt)))
    if len({record[0][6]["release_commit"] for record in audited}) != 1:
        raise RuntimeError("full block used different source releases")
    if len({base.sha256(root / f"seed{seed}/manifest.json") for seed in SEEDS}) != 1:
        raise RuntimeError("development manifest differs across seeds")
    rows = [load_seed(root / f"seed{seed}", data_root, _audited=item)
            for seed, item in zip(SEEDS, audited)]
    if len({row["manifest_sha256"] for row in rows}) != 1:
        raise RuntimeError("development manifest differs across seeds")
    if len({row["release_commit"] for row in rows}) != 1:
        raise RuntimeError("full block used different source releases")
    hashes = [row["caps"]["10"]["arms"]["ens_pto"]["prediction_sha256"] for row in rows]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("duplicate PTO predictions across seeds")
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
    report = {"status": "complete_12_seed_exploratory_development",
              "provenance": {"source_sha256": base.source(), "data_file_sha256": base.FILES,
                             "development_manifest_sha256": rows[0]["manifest_sha256"]},
              "seeds": rows, "contrasts": contrasts,
              "primary_family": [f"cap_divisor_{d}_joint_minus_{control}"
                                 for d, control in PRIMARY],
              "exploratory_signal_by_cap": signals,
              "limitations": ["Development countries were previously viewed; this is exploratory.",
                              "PTO/Clipper is a zero-step post-hoc analogue, not historical Clipper training.",
                              "PHR is a calibrated snapshot direction, not full ALM training.",
                              "Rejected probe probabilities and side weights are not saved; the offline gate checks their logged decision arithmetic and the accepted output, while parameter dose is verified by runner logs.",
                              "The fixed countries do not establish independent geographic generalization."]}
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
        print(json.dumps({"status": result["status"], "primary": result["contrasts"]["cc_f1"]},
                         indent=2))
    else:
        raise SystemExit(__doc__)
