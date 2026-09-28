"""Independent offline audit/scorer for the fixed fmow2 joint-local study.

Usage: python analysis/score_fmow_local.py --gate PILOT_ROOT REFERENCE_ROOT
       python analysis/score_fmow_local.py RUN_ROOT [OUTPUT_JSON]

The pilot gate hashes but never parses the label-bearing manifest. The full
scorer reads development labels only after provenance and step checks. Neither
path opens reserved-country data. Incomplete seeds fail rather than disappearing
from the prespecified 48-seed denominator.
"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from scipy import stats
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tralo.fmow_yuval import CAPPED, CLASSES, FILES  # noqa: E402
from tralo.global_clipper import allocate_local_capped_first  # noqa: E402
from tralo.knee_experiment import source  # noqa: E402

PILOT = 6099
SEEDS = tuple(range(6100, 6148))
DIVISORS = (10, 20)
COUNTRIES = frozenset(("IRQ", "NLD", "DZA", "PHL", "TUR"))
RESERVED = frozenset(("EGY", "CAN", "IND", "MEX", "JPN"))
ARMS = ("ens_pto", "ens_joint", "ens_global_dose", "ens_sham")
METRICS = ("cc_f1", "accuracy", "macro_f1", "weighted_f1")
PRIMARY = tuple((divisor, arm) for divisor in DIVISORS
                for arm in ("ens_global_dose", "ens_pto"))
RECIPE = {"backbone": "mobilenet_v3_large", "capped_class": 1, "max_epochs": 75,
          "patience": 5, "batch_size": 32, "lr": 1e-4, "weight_decay": 1e-4,
          "decay_epoch": 5, "decay_factor": 0.8, "development_batch_size": 16}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _events(path, terminal="completed"):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if not rows or [row["sequence"] for row in rows] != list(range(len(rows))):
        raise RuntimeError(f"{path}: missing/out-of-order events")
    if rows[-1]["event"] != terminal or any(row["event"] == "failed" for row in rows):
        raise RuntimeError(f"{path}: no successful terminal event")
    return rows


def _one(rows, event):
    found = [row for row in rows if row["event"] == event]
    if len(found) != 1:
        raise RuntimeError(f"expected one {event} event, got {len(found)}")
    return found[0]


def _budget(groups, divisor):
    """Independent integer Hamilton calculation using only country IDs."""
    n = len(groups)
    global_cap = n // divisor
    total = math.ceil(5 * global_cap / 4)
    sizes = Counter(groups)
    quotients = {g: divmod(total * size, n) for g, size in sizes.items()}
    local = {g: q for g, (q, _) in quotients.items()}
    for g in sorted(sizes, key=lambda g: (-quotients[g][1], g))[:total - sum(local.values())]:
        local[g] += 1
    return {"global_cap": global_cap, "local_total": total, "local_caps": dict(sorted(local.items()))}


def _quotas(groups):
    if len(groups) != 1673 or set(groups) != COUNTRIES:
        raise RuntimeError("development country identity/count differs from fixed protocol")
    return {str(d): _budget(groups, d) for d in DIVISORS}


def _pool_identity(directory, started):
    """Load only label-free sample IDs and countries; bind them to run receipts."""
    d = Path(directory)
    path = d / "pool_identity.json"
    if sha256(path) != started["pool_identity_sha256"]:
        raise RuntimeError(f"{d.name}: pool identity hash mismatch")
    rows = _json(path)
    if not isinstance(rows, list) or len(rows) != 1673 or any(
            not isinstance(row, dict) or set(row) != {"sample_id", "location"} or
            type(row["sample_id"]) is not str or re.fullmatch(r"test[0-9]+", row["sample_id"]) is None or
            type(row["location"]) is not str for row in rows):
        raise RuntimeError(f"{d.name}: label-free pool identity differs from protocol")
    ids = [row["sample_id"] for row in rows]
    groups = [row["location"] for row in rows]
    if len(set(ids)) != len(ids) or set(groups) != COUNTRIES:
        raise RuntimeError(f"{d.name}: duplicate IDs or wrong development countries")
    if started["quotas"] != _quotas(groups):
        raise RuntimeError(f"{d.name}: quotas disagree with label-free country sizes")
    return ids, groups


def _probabilities(path, n):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (n, CLASSES):
        raise RuntimeError(f"{path}: unexpected probability shape")
    if not bool(torch.isfinite(value).all()) or bool((value < 0).any()) or bool((value > 1).any()):
        raise RuntimeError(f"{path}: invalid probabilities")
    if not bool(torch.allclose(value.sum(1), torch.ones(n), atol=1e-6, rtol=0)):
        raise RuntimeError(f"{path}: probability rows do not sum to one")
    return value


def _pred_hash(predictions):
    return hashlib.sha256(json.dumps(predictions, separators=(",", ":")).encode()).hexdigest()


def _count(predictions, groups):
    local = {g: 0 for g in set(groups)}
    for value, group in zip(predictions, groups):
        local[group] += int(value == CAPPED)
    return sum(local.values()), dict(sorted(local.items()))


def allocate_joint(probabilities, ids, groups, global_cap, local_caps):
    """Independent top-probability selection under a uniform matroid intersection.

    For the single capped class, the best feasible set consists of each group's
    top local-cap items, then the top global-cap items from that union. All other
    items choose their best uncapped class. IDs break equal-probability ties.
    """
    if not (len(probabilities) == len(ids) == len(groups)) or len(set(ids)) != len(ids):
        raise ValueError("probabilities, unique IDs and groups must align")
    if set(local_caps) != set(groups) or type(global_cap) is not int or global_cap < 0:
        raise ValueError("invalid declared ceilings")
    if any(type(cap) is not int or cap < 0 for cap in local_caps.values()):
        raise ValueError("invalid local ceiling")
    rows = [list(row) for row in probabilities]
    if any(len(row) != CLASSES or any(not math.isfinite(p) for p in row) for row in rows):
        raise ValueError("invalid probability matrix")
    order = lambda i: (-rows[i][CAPPED], ids[i])
    eligible = []
    for group in sorted(local_caps):
        members = [i for i, g in enumerate(groups) if g == group]
        eligible.extend(sorted(members, key=order)[:local_caps[group]])
    chosen = set(sorted(eligible, key=order)[:global_cap])
    uncapped = tuple(c for c in range(CLASSES) if c != CAPPED)
    return [CAPPED if i in chosen else max(uncapped, key=lambda c: (rows[i][c], -c))
            for i in range(len(rows))]


def _audit_scope_numbers(record, quota):
    """The declared first-order direction must descend every active scope."""
    required = {"soft_before_global", "soft_before_local"}
    if record["applied"]:
        required |= {"soft_after_global", "soft_after_local", "directional_soft_delta_global",
                     "directional_soft_delta_local", "scope_directional_derivatives"}
    if not required <= set(record):
        raise RuntimeError("missing side-step scope measurements")
    for field in ("soft_before_global", "soft_after_global", "directional_soft_delta_global"):
        if field in record and not math.isfinite(record[field]):
            raise RuntimeError(f"nonfinite side-step {field}")
    for field in ("soft_before_local", "soft_after_local", "directional_soft_delta_local"):
        if field in record and (set(record[field]) != set(quota["local_caps"]) or
                                any(not math.isfinite(value) for value in record[field].values())):
            raise RuntimeError(f"nonfinite or incomplete side-step {field}")
    if record["applied"]:
        expected = set(record["active_local"]) | ({"global"} if record["active_global"] else set())
        derivatives = record["scope_directional_derivatives"]
        if set(derivatives) != expected or any(not math.isfinite(x) or x >= 0 for x in derivatives.values()):
            raise RuntimeError("joint direction does not descend every active scope")


def _audit_displacement(record):
    if not record["applied"]:
        if record["displacement"] != 0:
            raise RuntimeError("inactive step has nonzero displacement")
        return
    norms = record["tensor_displacement_norms"]
    if (not isinstance(norms, list) or not norms or
            any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                for value in norms) or
            not math.isfinite(record["displacement"]) or record["displacement"] <= 0 or
            not math.isfinite(record["radius"]) or record["radius"] <= 0 or
            abs(math.sqrt(sum(value * value for value in norms)) - record["displacement"]) > 1e-5 or
            abs(record["radius"] - record["displacement"]) > 1e-5):
        raise RuntimeError("side-step tensor displacement does not match total/radius")


def _score(labels, predictions):
    from sklearn.metrics import accuracy_score, f1_score
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    per_class = f1_score(labels, predictions, labels=list(range(CLASSES)), average=None,
                         zero_division=0)
    support = np.bincount(labels, minlength=CLASSES)
    return {"cc_f1": float(per_class[CAPPED]),
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(per_class.mean()),
            "weighted_f1": float(np.dot(per_class, support) / len(labels))}


def _audit_common(directory, *, labels_allowed):
    """Check identities before any scorer opens the manifest's label-bearing rows."""
    d = Path(directory)
    top = _events(d / "events.jsonl")
    train = _events(d / "retrain1" / "events.jsonl", terminal="training_completed")
    started, init, done = (_one(top, x) for x in ("started", "model_initialized", "completed"))
    config, summary = _json(d / "config.json"), _json(d / "summary.json")
    if (set(config) != set(RECIPE) | {"seed", "snapshot_steps"} or
            any(config[k] != value or type(config[k]) is not type(value)
                for k, value in RECIPE.items()) or type(config["seed"]) is not int or
            config["seed"] not in SEEDS + (PILOT,) or type(config["snapshot_steps"]) is not bool):
        raise RuntimeError(f"{d.name}: config differs from fixed protocol")
    if started["source_sha256"] != source() or started["data_files"] != FILES:
        raise RuntimeError(f"{d.name}: source/data hashes differ from frozen release")
    if started["config_sha256"] != sha256(d / "config.json"):
        raise RuntimeError(f"{d.name}: config hash mismatch")
    if started["counts"] != {"train": 15841, "stop": 1829, "dev": 1673}:
        raise RuntimeError(f"{d.name}: data role counts differ from fixed protocol")
    if sha256(d / "manifest.json") != started["manifest_sha256"]:
        raise RuntimeError(f"{d.name}: creation-time manifest hash mismatch")
    _pool_identity(d, started)
    if (config["seed"] != summary["seed"] or init["initial_sha256"] != summary["initial_sha256"]
            or init["architecture"] != "mobilenet_v3_large" or init["classes"] != CLASSES):
        raise RuntimeError(f"{d.name}: configuration/model identity mismatch")
    retrain = summary["retrain"]
    epochs = retrain["epochs_run"]
    if (not 1 <= retrain["best_epoch"] <= epochs <= 75 or
            done["epochs_run"] != epochs or done["task_updates"] != retrain["task_updates"]):
        raise RuntimeError(f"{d.name}: epoch/update mismatch")
    if [r["epoch"] for r in train if r["event"] == "epoch"] != list(range(1, epochs + 1)):
        raise RuntimeError(f"{d.name}: missing training epoch event")
    if set(summary["pto_snapshot_sha256"]) != {str(e) for e in range(1, epochs + 1)}:
        raise RuntimeError(f"{d.name}: PTO snapshot hash receipts missing")
    for epoch in range(1, epochs + 1):
        path = d / "retrain1" / f"epoch{epoch:02d}.pt"
        if sha256(path) != summary["pto_snapshot_sha256"][str(epoch)]:
            raise RuntimeError(f"{d.name}: creation-time PTO snapshot hash mismatch: {path.name}")
    if sha256(d / "retrain1" / "final_probabilities.pt") != summary["final_probability_sha256"]:
        raise RuntimeError(f"{d.name}: creation-time final PTO hash mismatch")
    if _one(train, "training_completed")["task_updates"] != retrain["task_updates"]:
        raise RuntimeError(f"{d.name}: training completion differs from summary")
    if retrain["task_updates"] != math.ceil(started["counts"]["train"] / config["batch_size"]) * epochs:
        raise RuntimeError(f"{d.name}: task update dose differs from fixed batch schedule")
    best_loss, first_best, waited = math.inf, 0, 0
    for row in train:
        if row["event"] == "epoch":
            if any(not math.isfinite(row[key]) for key in (
                    "training_loss", "stop_loss", "base_lr", "last_lr", "mean_gate",
                    "live_false_positives", "soft_count_capped")):
                raise RuntimeError(f"{d.name}: nonfinite training/stop event")
            improved = row["stop_loss"] < best_loss
            if row["improved"] is not improved:
                raise RuntimeError(f"{d.name}: early-stop improvement flag differs from logged loss")
            if improved:
                best_loss, first_best, waited = row["stop_loss"], row["epoch"], 0
            else:
                waited += 1
            if waited >= config["patience"] and row["epoch"] < epochs:
                raise RuntimeError(f"{d.name}: training continued after fixed patience")
    if first_best != retrain["best_epoch"] or best_loss != retrain["best_stop_loss"] or (
            epochs < config["max_epochs"] and waited < config["patience"]):
        raise RuntimeError(f"{d.name}: restored best epoch or early-stop endpoint differs from logged losses")
    if not labels_allowed:
        return config, summary, started, train
    manifest = _json(d / "manifest.json")
    if manifest["files"] != FILES or manifest["counts"] != started["counts"]:
        raise RuntimeError(f"{d.name}: data manifest identity mismatch")
    if (manifest["counts"] != {"train": 15841, "stop": 1829, "dev": 1673}
            or manifest["dev_countries"] != ["IRQ", "NLD", "DZA", "PHL", "TUR"]
            or set(manifest["reserved_countries"]) != RESERVED
            or set(manifest["dev_countries"]) & set(manifest["reserved_countries"])):
        raise RuntimeError(f"{d.name}: data roles differ from protocol")
    rows = manifest["rows"]
    ids, groups = [r["sample_id"] for r in rows], [r["location"] for r in rows]
    if (ids, groups) != _pool_identity(d, started):
        raise RuntimeError(f"{d.name}: label-free pool identity differs from label-bearing manifest")
    if len(rows) != 1673 or len(set(ids)) != len(ids) or any(r["split"] != "val" for r in rows):
        raise RuntimeError(f"{d.name}: development rows invalid")
    quotas = _quotas(groups)
    if quotas != manifest["quotas"] or quotas != summary["quotas"] or quotas != started["quotas"]:
        raise RuntimeError(f"{d.name}: declared quotas differ from label-free size policy")
    return config, summary, started, train, manifest


def gate(pilot_root, reference_root):
    """Integrity-only 6099 gate. Does not parse a manifest or compute a metric."""
    jobs = sorted(p for p in Path(pilot_root).glob("seed*") if p.is_dir())
    if [p.name for p in jobs] != ["seed6099"]:
        raise RuntimeError("pilot root must contain exactly seed6099")
    pilot, reference = jobs[0], Path(reference_root) / "seed6099_ref"
    if not reference.is_dir():
        raise RuntimeError("missing seed6099_ref")
    a = _audit_common(pilot, labels_allowed=False)
    b = _audit_common(reference, labels_allowed=False)
    ac, s, started, events = a
    bc, ref, ref_started, ref_events = b
    ids, groups = _pool_identity(pilot, started)
    if _pool_identity(reference, ref_started) != (ids, groups):
        raise RuntimeError("pilot/reference sample/country alignment differs")
    if (ac["seed"] != PILOT or bc["seed"] != PILOT or ac["snapshot_steps"] is not True
            or bc["snapshot_steps"] is not False):
        raise RuntimeError("pilot/reference configs do not identify the fixed gate")
    if ref["steps"] or any(row["event"] == "snapshot_cap" for row in ref_events):
        raise RuntimeError("steps-off reference contains side-step evidence")
    if (sha256(pilot / "manifest.json") != sha256(reference / "manifest.json")
            or started["counts"] != ref_started["counts"] or started["quotas"] != ref_started["quotas"]
            or s["retrain"] != ref["retrain"] or
            {k: v for k, v in s.items() if k not in ("steps", "pto_snapshot_sha256",
                                                    "final_probability_sha256")} !=
            {k: v for k, v in ref.items() if k not in ("steps", "pto_snapshot_sha256",
                                                      "final_probability_sha256")}):
        raise RuntimeError("pilot/reference data or PTO trajectory metadata mismatch")
    if [(e["event"], e.get("epoch"), e.get("hard_counts")) for e in events if e["event"] == "epoch"] != [
            (e["event"], e.get("epoch"), e.get("hard_counts")) for e in ref_events if e["event"] == "epoch"]:
        raise RuntimeError("pilot/reference training epoch events differ")
    epochs = s["retrain"]["epochs_run"]
    for name in [f"epoch{epoch:02d}.pt" for epoch in range(1, epochs + 1)] + ["final_probabilities.pt"]:
        left = _probabilities(pilot / "retrain1" / name, started["counts"]["dev"])
        right = _probabilities(reference / "retrain1" / name, started["counts"]["dev"])
        if not torch.equal(left, right):
            raise RuntimeError(f"PTO snapshot mismatch: {name}")
    if set(s["steps"]) != {str(epoch) for epoch in range(1, epochs + 1)}:
        raise RuntimeError("pilot missing step records")
    cap_events = [row for row in events if row["event"] == "snapshot_cap"]
    if len(cap_events) != 2 * epochs or {(r["epoch"], r["divisor"]) for r in cap_events} != {
            (epoch, divisor) for epoch in range(1, epochs + 1) for divisor in DIVISORS}:
        raise RuntimeError("pilot missing or duplicate cap events")
    n = started["counts"]["dev"]
    for epoch in range(1, epochs + 1):
        pto = _probabilities(pilot / "retrain1" / f"epoch{epoch:02d}.pt", n)
        hard_before, local_before = _count(pto.argmax(1).tolist(), groups)
        for divisor in DIVISORS:
            quota = started["quotas"][str(divisor)]
            records = s["steps"][str(epoch)][str(divisor)]
            event = next(r for r in cap_events if (r["epoch"], r["divisor"]) == (epoch, divisor))
            if event["steps"] != records or event["quota"] != quota or set(records) != {"joint", "global_dose", "sham"}:
                raise RuntimeError("pilot cap event disagrees with summary")
            joint = records["joint"]
            _audit_scope_numbers(joint, quota)
            if (joint["hard_before_global"] != hard_before or
                    abs(joint["soft_before_global"] - float(pto[:, CAPPED].sum())) > 1e-3 or
                    joint["hard_before_local"] != local_before or
                    any(abs(joint["soft_before_local"][g] - float(pto[[i for i, group in enumerate(groups)
                                                                         if group == g], CAPPED].sum())) > 1e-3
                        for g in quota["local_caps"])):
                raise RuntimeError("pilot hard-before record disagrees with PTO")
            active_g = hard_before > quota["global_cap"]
            active_l = sorted(g for g, count in joint["hard_before_local"].items()
                              if count > quota["local_caps"][g])
            if (joint["active_global"] != active_g or joint["active_local"] != active_l or
                    joint["applied"] != bool(active_g or active_l)):
                raise RuntimeError("pilot active scopes disagree with hard-before counts")
            for arm in ("joint", "global_dose", "sham"):
                side = _probabilities(pilot / "retrain1" / f"cap{divisor}" /
                                      f"epoch{epoch:02d}_{arm}.pt", n)
                row = records[arm]
                _audit_displacement(row)
                side_path = pilot / "retrain1" / f"cap{divisor}" / f"epoch{epoch:02d}_{arm}.pt"
                if sha256(side_path) != row["probability_sha256"]:
                    raise RuntimeError("pilot creation-time side snapshot hash mismatch")
                if row["applied"] != joint["applied"]:
                    raise RuntimeError("pilot same-dose arm application mismatch")
                if not row["applied"]:
                    if not torch.equal(pto, side):
                        raise RuntimeError("pilot inactive side snapshot differs from PTO")
                    continue
                after, local_after = _count(side.argmax(1).tolist(), groups)
                if (row["hard_after_global"] != after or row["hard_after_local"] != local_after or
                        abs(row["soft_after_global"] - float(side[:, CAPPED].sum())) > 1e-3 or
                        any(abs(row["soft_after_local"][g] - float(side[[i for i, group in enumerate(groups)
                                                                           if group == g], CAPPED].sum())) > 1e-3
                            for g in quota["local_caps"]) or
                        abs(row["radius"] - joint["radius"]) > 1e-12 or
                        abs(row["displacement"] - joint["displacement"]) > 1e-5):
                    raise RuntimeError("pilot side-step count or dose mismatch")
                if arm == "sham" and (len(row["tensor_displacement_norms"]) !=
                                      len(joint["tensor_displacement_norms"]) or any(
                        abs(a - b) > 1e-5 for a, b in zip(row["tensor_displacement_norms"],
                                                         joint["tensor_displacement_norms"]))):
                    raise RuntimeError("pilot sham per-tensor dose differs from joint")
                if arm == "joint" and (after > quota["global_cap"] or any(
                        row["hard_after_local"][g] > limit for g, limit in quota["local_caps"].items())):
                    raise RuntimeError("pilot joint side step is not feasible")
    print(f"PILOT GATE PASSED: {epochs} PTO probability tensors exactly equal; no pilot labels accessed")


def _audit_step(record, pto, side, groups, quota, arm):
    _audit_displacement(record)
    before_g, before_l = _count(pto.argmax(1).tolist(), groups)
    # The runner emits only applied/displacement for inactive dose controls.
    if arm != "joint" and not record["applied"]:
        if ({k: v for k, v in record.items() if k != "probability_sha256"} !=
                {"applied": False, "displacement": 0.0} or not torch.equal(pto, side)):
            raise RuntimeError(f"{arm}: inactive control differs from PTO")
        return
    if record["hard_before_global"] != before_g or record["hard_before_local"] != before_l:
        raise RuntimeError(f"{arm}: hard-before counts disagree with snapshot")
    if (abs(record["soft_before_global"] - float(pto[:, CAPPED].sum())) > 1e-3 or
            any(abs(record["soft_before_local"][g] - float(pto[[i for i, group in enumerate(groups)
                                                                 if group == g], CAPPED].sum())) > 1e-3
                for g in quota["local_caps"])):
        raise RuntimeError(f"{arm}: soft-before counts disagree with snapshot")
    after_g, after_l = _count(side.argmax(1).tolist(), groups)
    if record["applied"]:
        if (record["hard_after_global"] != after_g or record["hard_after_local"] != after_l
                or not math.isfinite(record["radius"]) or record["radius"] <= 0):
            raise RuntimeError(f"{arm}: hard-after counts/radius disagree with snapshot")
        if (abs(record["soft_after_global"] - float(side[:, CAPPED].sum())) > 1e-3 or
                any(abs(record["soft_after_local"][g] - float(side[[i for i, group in enumerate(groups)
                                                                     if group == g], CAPPED].sum())) > 1e-3
                    for g in quota["local_caps"])):
            raise RuntimeError(f"{arm}: soft-after counts disagree with snapshot")
    elif not torch.equal(pto, side):
        raise RuntimeError(f"{arm}: unstepped side snapshot differs from PTO")
    if arm == "joint":
        _audit_scope_numbers(record, quota)
        active_g = before_g > quota["global_cap"]
        active_l = sorted(g for g, count in before_l.items() if count > quota["local_caps"][g])
        if record["active_global"] != active_g or record["active_local"] != active_l:
            raise RuntimeError("joint active scopes disagree with hard counts")
        if record["applied"] != bool(active_g or active_l):
            raise RuntimeError("joint application disagrees with active scopes")
        if record["applied"] and (after_g > quota["global_cap"] or any(
                after_l[g] > cap for g, cap in quota["local_caps"].items())):
            raise RuntimeError("joint side step violates fixed hard ceilings")


def _arm_score(probabilities, labels, ids, groups, quota):
    matrix = probabilities.tolist()
    raw = probabilities.argmax(1).tolist()
    assigned = allocate_joint(matrix, ids, groups, quota["global_cap"], quota["local_caps"])
    caps = [None] * CLASSES
    caps[CAPPED] = quota["global_cap"]
    if assigned != allocate_local_capped_first(matrix, caps, ids, groups, quota["local_caps"]):
        raise RuntimeError("independent joint allocation disagrees with production allocator")
    global_count, local_count = _count(assigned, groups)
    if global_count > quota["global_cap"] or any(local_count[g] > cap for g, cap in quota["local_caps"].items()):
        raise RuntimeError("allocated predictions violate joint ceilings")
    return {"raw": _score(labels, raw), "allocated": _score(labels, assigned),
            "raw_counts": _count(raw, groups), "allocated_counts": (global_count, local_count),
            "local_tp": {g: sum(p == CAPPED and y == CAPPED for p, y, group in zip(assigned, labels, groups)
                            if group == g) for g in sorted(quota["local_caps"])},
            "prediction_sha256": _pred_hash(assigned), "predictions": assigned}


def load_seed(directory):
    d = Path(directory)
    config, summary, _, events, manifest = _audit_common(d, labels_allowed=True)
    seed = int(d.name.removeprefix("seed")) if d.name.startswith("seed") else -1
    if seed not in SEEDS or seed != summary["seed"] or config["seed"] != seed or not config["snapshot_steps"]:
        raise RuntimeError(f"{d.name}: seed/config mismatch")
    rows = manifest["rows"]
    ids, groups, labels = [r["sample_id"] for r in rows], [r["location"] for r in rows], [r["label"] for r in rows]
    if any(type(y) is not int or not 0 <= y < CLASSES for y in labels):
        raise RuntimeError(f"{d.name}: invalid development labels")
    epochs = summary["retrain"]["epochs_run"]
    best = summary["retrain"]["best_epoch"]
    if set(summary["steps"]) != {str(e) for e in range(1, epochs + 1)}:
        raise RuntimeError(f"{d.name}: missing or duplicate step epoch")
    cap_events = [row for row in events if row["event"] == "snapshot_cap"]
    if {(e["epoch"], e["divisor"]) for e in cap_events} != {
            (epoch, divisor) for epoch in range(1, epochs + 1) for divisor in DIVISORS} or len(cap_events) != 2 * epochs:
        raise RuntimeError(f"{d.name}: missing/duplicate cap events")
    pto = {}
    artifacts = {}
    for epoch in range(1, epochs + 1):
        path = d / "retrain1" / f"epoch{epoch:02d}.pt"
        pto[epoch] = _probabilities(path, len(rows))
        artifacts[str(path.relative_to(d))] = sha256(path)
        for divisor in DIVISORS:
            quota = manifest["quotas"][str(divisor)]
            record = summary["steps"][str(epoch)][str(divisor)]
            event = next(x for x in cap_events if (x["epoch"], x["divisor"]) == (epoch, divisor))
            if record != event["steps"] or quota != event["quota"] or set(record) != {"joint", "global_dose", "sham"}:
                raise RuntimeError(f"{d.name}: side-step event/summary mismatch")
            sides = {}
            for arm in ("joint", "global_dose", "sham"):
                path = d / "retrain1" / f"cap{divisor}" / f"epoch{epoch:02d}_{arm}.pt"
                sides[arm] = _probabilities(path, len(rows))
                artifacts[str(path.relative_to(d))] = sha256(path)
                if artifacts[str(path.relative_to(d))] != record[arm]["probability_sha256"]:
                    raise RuntimeError(f"{d.name}: creation-time side snapshot hash mismatch")
                _audit_step(record[arm], pto[epoch], sides[arm], groups, quota, arm)
            joint = record["joint"]
            for arm in ("global_dose", "sham"):
                if (record[arm]["applied"] != joint["applied"] or joint["applied"] and (
                        abs(record[arm]["radius"] - joint["radius"]) > 1e-12 or
                        abs(record[arm]["displacement"] - joint["displacement"]) > 1e-5)):
                    raise RuntimeError(f"{d.name}: {arm} dose differs from joint")
            if joint["applied"] and (len(record["sham"]["tensor_displacement_norms"]) !=
                                     len(joint["tensor_displacement_norms"]) or any(
                    abs(a - b) > 1e-5 for a, b in zip(record["sham"]["tensor_displacement_norms"],
                                                     joint["tensor_displacement_norms"]))):
                raise RuntimeError(f"{d.name}: sham per-tensor dose differs from joint")
    final_path = d / "retrain1" / "final_probabilities.pt"
    if not torch.equal(_probabilities(final_path, len(rows)), pto[best]):
        raise RuntimeError(f"{d.name}: final PTO probabilities differ from best epoch")
    artifacts[str(final_path.relative_to(d))] = sha256(final_path)
    period = list(range(max(1, best - 2), epochs + 1))
    out = {"seed": seed, "epochs_run": epochs, "best_epoch": best, "window": period,
           "manifest_sha256": sha256(d / "manifest.json"), "config_sha256": sha256(d / "config.json"),
           "artifacts": artifacts, "caps": {}}
    for divisor in DIVISORS:
        quota = manifest["quotas"][str(divisor)]
        snapshots = {"ens_pto": [pto[e] for e in period]}
        for arm in ("joint", "global_dose", "sham"):
            snapshots[f"ens_{arm}"] = [
                _probabilities(d / "retrain1" / f"cap{divisor}" / f"epoch{e:02d}_{arm}.pt", len(rows))
                for e in period]
        scored = {arm: _arm_score(torch.stack(values).mean(0), labels, ids, groups, quota)
                  for arm, values in snapshots.items()}
        pto_set = {i for i, value in enumerate(scored["ens_pto"]["predictions"]) if value == CAPPED}
        dose_set = {i for i, value in enumerate(scored["ens_global_dose"]["predictions"]) if value == CAPPED}
        joint_set = {i for i, value in enumerate(scored["ens_joint"]["predictions"]) if value == CAPPED}
        out["caps"][str(divisor)] = {
            "quota": quota, "arms": {arm: {k: v for k, v in row.items() if k != "predictions"}
                                   for arm, row in scored.items()},
            "joint_applied": sum(summary["steps"][str(e)][str(divisor)]["joint"]["applied"] for e in period),
            "turnover_vs_pto": len(joint_set ^ pto_set),
            "turnover_vs_global_dose": len(joint_set ^ dose_set),
            "per_seed_deltas": {"joint_minus_global_dose": scored["ens_joint"]["allocated"]["cc_f1"] -
                                scored["ens_global_dose"]["allocated"]["cc_f1"],
                                "joint_minus_pto": scored["ens_joint"]["allocated"]["cc_f1"] -
                                scored["ens_pto"]["allocated"]["cc_f1"]}}
    return out


def _paired(values):
    values = np.asarray(values, dtype=float)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    if sd == 0:
        return {"n": len(values), "mean": mean, "sd": sd, "lo": None, "hi": None,
                "interval_available": False, "p": 1.0}
    half = float(stats.t.ppf(.975, len(values) - 1) * sd / math.sqrt(len(values)))
    p = float(2 * stats.t.sf(abs(mean) / (sd / math.sqrt(len(values))), len(values) - 1))
    return {"n": len(values), "mean": mean, "sd": sd, "lo": mean - half, "hi": mean + half,
            "interval_available": True, "p": p}


def _holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adjusted, running = [0.] * len(ps), 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(ps) - rank) * ps[i]))
        adjusted[i] = running
    return adjusted


def main(root, output=None):
    root = Path(root)
    directories = sorted(p for p in root.glob("seed*") if p.is_dir())
    names = [p.name for p in directories]
    expected = {f"seed{s}" for s in SEEDS}
    if len(names) != len(set(names)) or set(names) != expected:
        raise RuntimeError(f"fixed 48-seed block incomplete or extra: missing {sorted(expected - set(names))}, "
                           f"extra {sorted(set(names) - expected)}")
    rows = [load_seed(root / f"seed{s}") for s in SEEDS]
    if len({row["manifest_sha256"] for row in rows}) != 1:
        raise RuntimeError("development manifest differs across seeds")
    seen = {}
    for row in rows:
        pto = row["caps"]["10"]["arms"]["ens_pto"]["prediction_sha256"]
        if pto in seen:
            raise RuntimeError(f"duplicate PTO predictions: seeds {seen[pto]} and {row['seed']}")
        seen[pto] = row["seed"]
    contrasts = {}
    for metric in METRICS:
        family = []
        for divisor, control in PRIMARY:
            differences = [r["caps"][str(divisor)]["arms"]["ens_joint"]["allocated"][metric] -
                           r["caps"][str(divisor)]["arms"][control]["allocated"][metric]
                           for r in rows]
            family.append((f"cap_divisor_{divisor}_joint_minus_{control}", _paired(differences), differences))
        adjusted = _holm([r[1]["p"] for r in family]) if metric == "cc_f1" else [None] * len(family)
        contrasts[metric] = {name: {**stat, "holm_p": correction, "per_seed": dict(zip(SEEDS, differences))}
                             for (name, stat, differences), correction in zip(family, adjusted)}
    report = {"status": "complete_48_seed_exploratory_development",
              "provenance": {"source_sha256": source(), "data_file_sha256": FILES,
                             "development_manifest_sha256": rows[0]["manifest_sha256"]},
              "seeds": rows,
              "contrasts": contrasts, "primary_family": [f"cap_divisor_{d}_joint_minus_{a}" for d, a in PRIMARY],
              "limitations": ["Repeatedly viewed development countries, not independent confirmation."]}
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if output is not None:
        path = Path(output)
        if path.exists():
            raise FileExistsError(path)
        path.write_text(encoded + "\n", encoding="utf-8")
    for divisor, control in PRIMARY:
        name = f"cap_divisor_{divisor}_joint_minus_{control}"
        stat = contrasts["cc_f1"][name]
        interval = (f"[{100 * stat['lo']:+.3f}, {100 * stat['hi']:+.3f}]"
                    if stat["interval_available"] else "unavailable (zero seed variance)")
        print(f"{name}: {100 * stat['mean']:+.3f} pp {interval}, Holm p={stat['holm_p']:.4g}")
    print("All 48 seeds audited; raw/allocated and secondary metrics are in the JSON report.")
    return report


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--gate":
        gate(sys.argv[2], sys.argv[3])
    elif len(sys.argv) in (2, 3):
        main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
    else:
        raise SystemExit(__doc__)
