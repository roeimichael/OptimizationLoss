"""Independent label-blind gate and complete-block tabular quota scorer.

Gate:  python -m analysis.score_tabular_persistent --gate SEED_ROOT PREPARED_ROOT OUTPUT
Score: python -m analysis.score_tabular_persistent --score FULL_ROOT PREPARED_ROOT OUTPUT
Only --score opens scorer/development_labels.jsonl, after every seed is gated.
"""

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
import sys

import torch

from tralo.global_clipper import allocate_local_upper_bound
from tralo.fmow_persistent_local import stable_state_hash
from tralo.knee_experiment import digest, save, source
from tralo.metrics import classification_metrics
from tralo.tabular_backbones import (configure_fp32, image_transforms,
                                      make_binary_model)
from tralo.tabular_image_data import (ISIC_CACHED_DRAFT_POLICY,
                                      ISIC_DRAFT_POLICY, PreparedImageRows,
                                      load_runner_cohort)
from tralo.tabular_persistent_train import (ARMS, BACKBONES,
                                             correction_step_size_for_arm,
                                             EPOCHS, MAX_DISPLACEMENT,
                                             _pool_batches, _predict,
                                             validate_config)
from tralo.tabular_quota_policy import caps_for_unlabeled_pool


FULL_SEEDS = tuple(range(6801, 6805))


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _events(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if [row["sequence"] for row in rows] != list(range(len(rows))):
        raise RuntimeError("event log has a gap or reorder")
    return rows


def _artifact(root, relative, sha):
    path = root / relative
    if (Path(relative).is_absolute() or ".." in Path(relative).parts or
            path.is_symlink() or not path.is_file() or digest(path) != sha):
        raise RuntimeError("run artifact missing, linked or changed: " + relative)
    return path


def _finite(value, name, *, minimum=None):
    if (type(value) not in (int, float) or not math.isfinite(value) or
            (minimum is not None and value < minimum)):
        raise RuntimeError("invalid " + name)


def _private_labels(prepared_root, expected_ids, manifest_sha):
    """The only function that opens development targets; call after all gates."""
    root = Path(prepared_root)
    private_manifest_path = root / "scorer" / "manifest.json"
    private_path = root / "scorer" / "development_labels.jsonl"
    if private_manifest_path.is_symlink() or private_path.is_symlink():
        raise RuntimeError("linked private scorer artifact")
    private_manifest = _json(private_manifest_path)
    if (private_manifest["runner_manifest_sha256"] != manifest_sha or
            digest(private_path) != private_manifest["development_labels_sha256"]):
        raise RuntimeError("private development label provenance changed")
    with private_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if ([row["sample_id"] for row in rows] != expected_ids or
            any(set(row) != {"sample_id", "label"} or row["label"] not in (0, 1)
                for row in rows)):
        raise RuntimeError("private development labels do not align")
    return [row["label"] for row in rows], {
        "scorer_manifest_sha256": digest(private_manifest_path),
        "development_labels_sha256": digest(private_path)}


def audit_seed(run_root, prepared_root, *, replay=True):
    """Return authenticated label-free probabilities for every arm."""
    root = Path(run_root)
    config_path = root / "config.json"
    summary_path = root / "summary.json"
    run_manifest_path = root / "manifest.json"
    config, summary, run_manifest = (_json(path) for path in
                                     (config_path, summary_path, run_manifest_path))
    validate_config(config)
    if summary.get("config") != config or set(summary.get("arms", {})) != set(ARMS):
        raise RuntimeError("incomplete or altered arm block")
    identity = summary["identity"]
    if (identity != run_manifest["identity"] or
            identity["config_sha256"] != digest(config_path) or
            identity["prepared_manifest_sha256"] != digest(
                Path(prepared_root) / "manifest.json") or
            identity["source_sha256"] != source() or
            identity["development_labels_loaded"] is not False or
            identity["precision"] != "fp32_tf32_off"):
        raise RuntimeError("source/config/data or label boundary changed")
    manifest, rows = load_runner_cohort(prepared_root, config["dataset"])
    if manifest["files_sha256"] != identity["runner_files_sha256"]:
        raise RuntimeError("prepared runner rows changed")
    if identity["preprocessing"].get("decode_policy") != manifest.get("decode_policy"):
        raise RuntimeError("image decoder policy changed")
    expected_ids = [row["sample_id"] for row in rows["development_pool"]]
    groups = [row["group"] for row in rows["development_pool"]]
    expected_pool_hash = hashlib.sha256(json.dumps(
        expected_ids, separators=(",", ":")).encode()).hexdigest()
    if run_manifest["pool_ids_sha256"] != expected_pool_hash:
        raise RuntimeError("development pool ID/order hash changed")
    quotas = caps_for_unlabeled_pool(config["dataset"], groups)
    if (run_manifest["quotas"] != quotas or
            run_manifest["split_sizes"] != {key: len(value) for key, value in rows.items()}):
        raise RuntimeError("label-free quota or split recount changed")
    with (root / "events.jsonl").open("rb") as stream:
        if not stream.read():
            raise RuntimeError("empty event log")
    events = _events(root / "events.jsonl")
    completed = [event for event in events if event["event"] == "completed"]
    if len(completed) != 1 or completed[0]["summary_sha256"] != digest(summary_path):
        raise RuntimeError("missing or changed completion event")
    started = [event for event in events if event["event"] == "started"]
    if len(started) != 1 or started[0]["manifest_sha256"] != digest(run_manifest_path):
        raise RuntimeError("missing or changed launch event")
    _finite(summary["elapsed_seconds"], "elapsed_seconds", minimum=0)
    probabilities = {}
    if replay:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required for independent model replay")
        configure_fp32()
        torch.use_deterministic_algorithms(True)
        _training_tf, eval_tf = image_transforms(config["backbone"])
        data = PreparedImageRows(manifest["image_dir"], rows["development_pool"],
                                 eval_tf, manifest.get("decode_policy"))
        batch_size = BACKBONES[config["backbone"]][0]
        batches = _pool_batches(data, batch_size, torch.device("cuda"))
    for name in ARMS:
        arm = summary["arms"][name]
        epochs = arm["epochs"]
        previous_dual = ({scope: 0.0 for scope in
                          ("global", *quotas[name.split("_")[0]]["local_caps"])}
                         if name.endswith("_phr") else None)
        if len(epochs) != EPOCHS or [row["epoch"] for row in epochs] != list(
                range(1, EPOCHS + 1)):
            raise RuntimeError("incomplete training epochs: " + name)
        for row in epochs:
            for key in ("training_loss", "stop_loss", "max_task_gradient_norm",
                        "elapsed_seconds"):
                _finite(row[key], name + "." + key, minimum=0)
            if row["task_updates"] <= 0 or len(row["corrections"]) != (
                    0 if row["epoch"] == 1 else 1 if name.endswith(
                        ("_tralo", "_phr")) else 2):
                raise RuntimeError("task/correction opportunity mismatch")
            for correction in row["corrections"]:
                dose = correction["correction"]
                _finite(dose["actual_displacement_norm"], "correction dose", minimum=0)
                _finite(correction["gradient"]["parameter_gradient_norm"],
                        "correction gradient", minimum=0)
                if dose["actual_displacement_norm"] > MAX_DISPLACEMENT * (1 + 1e-5):
                    raise RuntimeError("constraint correction exceeded frozen dose")
                expected_proposed = (correction_step_size_for_arm(config, name) *
                    correction["gradient"]["parameter_gradient_norm"])
                if (abs(dose["proposed_displacement_norm"] - expected_proposed) >
                        1e-7 * max(1.0, expected_proposed) or
                        (dose["applied"] and abs(dose["actual_displacement_norm"] -
                         expected_proposed) > 1e-5 * max(1.0, expected_proposed))):
                    raise RuntimeError("correction magnitude lost frozen loss scale")
                if name in ("pto", "sham") and (dose["applied"] or
                    correction["before"] != correction["after"]):
                    raise RuntimeError("zero-correction control changed pool")
                if name.endswith("_phr"):
                    expected_dual = {scope: max(0.0, previous_dual[scope] +
                        0.5 * observed["signed_residual"])
                        for scope, observed in correction["before"].items()}
                    if (set(correction["dual"]) != set(expected_dual) or any(
                            abs(correction["dual"][scope] - value) > 1e-6
                            for scope, value in expected_dual.items())):
                        raise RuntimeError("PHR dual continuity failed")
                    previous_dual = correction["dual"]
        selected = arm["selected_epoch"]
        if selected not in range(1, EPOCHS + 1) or arm["selected_stop_loss"] != min(
                row["stop_loss"] for row in epochs):
            raise RuntimeError("checkpoint selected using non-stop information")
        checkpoint = _artifact(root, arm["checkpoint"], arm["checkpoint_sha256"])
        predictions = _artifact(root, arm["probabilities"],
                                arm["probability_sha256"])
        stored = torch.load(predictions, weights_only=True, map_location="cpu")
        p = stored["probabilities"]
        if (stored["sample_ids"] != expected_ids or not isinstance(p, torch.Tensor) or
                tuple(p.shape) != (len(expected_ids), 2) or
                not bool(torch.isfinite(p).all()) or
                not bool(torch.allclose(p.sum(1), torch.ones(len(p)), atol=1e-6))):
            raise RuntimeError("selected probabilities are misaligned or invalid")
        probabilities[name] = p
        selected_events = [event for event in events if
                           event["event"] == "selected_checkpoint" and
                           event.get("arm") == name]
        if (not selected_events or selected_events[-1]["epoch"] != selected or
                selected_events[-1]["checkpoint_sha256"] != arm["checkpoint_sha256"] or
                selected_events[-1]["probability_sha256"] !=
                arm["probability_sha256"]):
            raise RuntimeError("selected checkpoint event/artifact mismatch")
        if selected >= 2:
            selected_row = epochs[selected - 1]
            from tralo.tabular_persistent_train import _observations
            for correction in selected_row["corrections"]:
                level = correction["level"]
                independent = _observations(p, groups, quotas[level])
                for scope, observed in independent.items():
                    recorded = correction["after"][scope]
                    if (recorded["hard"] != observed["hard"] or
                            abs(recorded["soft"] - observed["soft"]) > 1e-4):
                        raise RuntimeError("selected snapshot quota log differs")
        if replay:
            torch.manual_seed(config["seed"])
            model, verified_weight = make_binary_model(config["backbone"])
            if verified_weight != identity["weight"]:
                raise RuntimeError("pretrained bytes changed during replay")
            if stable_state_hash(model.state_dict()) != identity["initial_model_sha256"]:
                raise RuntimeError("seeded initial model differs from recorded initialization")
            model.load_state_dict(torch.load(checkpoint, weights_only=True,
                                             map_location="cpu"), strict=True)
            model.cuda().eval()
            actual, replay_ids = _predict(model, batches)
            if replay_ids != expected_ids or not torch.allclose(
                    actual, p, atol=1e-6, rtol=0):
                raise RuntimeError("independent selected checkpoint replay differs")
            del model
            torch.cuda.empty_cache()
    if (not torch.equal(probabilities["pto"], probabilities["sham"]) or
            summary["arms"]["pto"]["selected_epoch"] !=
            summary["arms"]["sham"]["selected_epoch"]):
        raise RuntimeError("sham/PTO predictions or stop selection differ")
    baseline = summary["arms"]["pto"]["epochs"]
    for name in ARMS:
        for left, right in zip(baseline, summary["arms"][name]["epochs"]):
            if (left["sample_order_sha256"] != right["sample_order_sha256"] or
                    left["first_batch_sha256"] != right["first_batch_sha256"] or
                    left["task_updates"] != right["task_updates"]):
                raise RuntimeError("matched task inputs or dose differ: " + name)
    return {"config": config, "identity": identity, "quotas": quotas,
            "groups": groups, "sample_ids": expected_ids,
            "probabilities": probabilities, "elapsed_seconds": summary["elapsed_seconds"],
            "summary_sha256": digest(summary_path)}


def gate(run_root, prepared_root, output):
    audited = audit_seed(run_root, prepared_root)
    summary = _json(Path(run_root) / "summary.json")
    applied = {name: sum(
        int(correction["correction"]["applied"])
        for epoch in summary["arms"][name]["epochs"]
        for correction in epoch["corrections"])
        for name in ARMS if name.endswith(("_tralo", "_phr"))}
    if any(count == 0 for count in applied.values()):
        raise RuntimeError("pilot has an inactive treated arm; preserve as negative evidence")
    record = {"status": "label_blind_integrity_pass", "seed": audited["config"]["seed"],
              "scorer_sha256": digest(__file__),
              "dataset": audited["config"]["dataset"],
              "backbone": audited["config"]["backbone"],
              "elapsed_seconds": audited["elapsed_seconds"],
              "applied_correction_counts": applied,
              "projected_four_seed_gpu_hours": audited["elapsed_seconds"] * 4 / 3600,
              "summary_sha256": audited["summary_sha256"],
              "development_labels_accessed": False}
    if record["projected_four_seed_gpu_hours"] > 96:
        raise RuntimeError("pilot projects above the weekend aggregate ceiling")
    with Path(output).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return record


def _group_metrics(labels, predictions, groups):
    return {group: classification_metrics(
        [labels[i] for i, item in enumerate(groups) if item == group],
        [predictions[i] for i, item in enumerate(groups) if item == group], 2, [1])
            for group in sorted(set(groups))}


def _metrics(labels, predictions, groups):
    metrics = classification_metrics(labels, predictions, 2, [1])
    metrics["weighted_f1"] = sum(
        row["f1"] * row["support"] for row in metrics["per_class"]) / len(labels)
    metrics["groups"] = _group_metrics(labels, predictions, groups)
    return metrics


def _completed_cell_cost(full_root, audited, pilot_seed):
    """Authenticate all guarded queue receipts before opening private labels."""
    cell = Path(full_root).parent
    pilot_launch = _json(cell / f"pilot_{pilot_seed}.launch.json")
    pilot_complete = _json(cell / f"pilot_{pilot_seed}.complete.json")
    gate = _json(cell / "pilot_gate.json")
    if (pilot_launch["seed"] != pilot_seed or pilot_complete["exit_code"] != 0 or
            pilot_launch["release_commit"] != pilot_complete["release_commit"] or
            gate["status"] != "label_blind_integrity_pass" or
            gate["seed"] != pilot_seed or
            pilot_launch["prepared_manifest_sha256"] !=
            audited[0]["identity"]["prepared_manifest_sha256"] or
            pilot_launch["source_sha256"] != audited[0]["identity"]["source_sha256"] or
            gate["summary_sha256"] != digest(
                cell / f"pilot/seed{pilot_seed}/summary.json")):
        raise RuntimeError("pilot launch, completion or gate receipt changed")
    beginning = datetime.fromisoformat(pilot_launch["started_utc"])
    elapsed_full = 0
    last_end = None
    for row in audited:
        seed = row["config"]["seed"]
        launch = _json(cell / f"full_{seed}.launch.json")
        complete = _json(cell / f"full_{seed}.complete.json")
        if (launch["seed"] != seed or complete["seed"] != seed or
                complete["exit_code"] != 0 or
                launch["release_commit"] != pilot_launch["release_commit"] or
                complete["release_commit"] != pilot_launch["release_commit"] or
                launch["source_sha256"] != row["identity"]["source_sha256"] or
                launch["config_sha256"] != row["identity"]["config_sha256"] or
                launch["prepared_manifest_sha256"] !=
                row["identity"]["prepared_manifest_sha256"] or
                launch["output_dir"] != str(Path(full_root) / f"seed{seed}") or
                launch["gpu_uuid"] != complete["gpu_uuid"] or
                launch["gpu_uuid"] != pilot_launch["gpu_uuid"]):
            raise RuntimeError("full queue cost/ownership provenance changed")
        _finite(complete["elapsed_seconds"], "queue elapsed", minimum=0)
        if complete["elapsed_seconds"] + 2 < row["elapsed_seconds"]:
            raise RuntimeError("runner duration exceeds guarded queue receipt")
        elapsed_full += complete["elapsed_seconds"]
        last_end = datetime.fromisoformat(complete["ended_utc"])
    cell_wall = (last_end - beginning).total_seconds()
    if cell_wall < elapsed_full or cell_wall > 86400 + 60:
        raise RuntimeError("completed cell wall exceeds fixed 24-hour ceiling")
    return {"full_queue_gpu_hours": elapsed_full / 3600,
            "cell_lease_gpu_hours_including_pilot_gate": cell_wall / 3600,
            "pilot_gate_sha256": digest(cell / "pilot_gate.json"),
            "gpu_uuid": pilot_launch["gpu_uuid"],
            "release_commit": pilot_launch["release_commit"]}


def score(full_root, prepared_root, output):
    root = Path(full_root)
    prepared_manifest = _json(Path(prepared_root) / "manifest.json")
    policy = prepared_manifest.get("decode_policy")
    if policy == ISIC_DRAFT_POLICY:
        allowed = (tuple(range(6811, 6815)), tuple(range(6821, 6825)))
    elif policy == ISIC_CACHED_DRAFT_POLICY:
        allowed = (tuple(range(6831, 6835)), tuple(range(6841, 6845)),
                   tuple(range(6851, 6855)), tuple(range(6891, 6895)))
    else:
        allowed = (FULL_SEEDS, tuple(range(6881, 6885)))
    found = {path.name for path in root.iterdir() if path.is_dir()}
    blocks = [block for block in allowed if found ==
              {f"seed{seed}" for seed in block}]
    if len(blocks) != 1:
        raise RuntimeError("complete fixed four-seed block required before labels")
    seeds = blocks[0]
    expected = [root / f"seed{seed}" for seed in seeds]
    audited = [audit_seed(path, prepared_root) for path in expected]
    first = audited[0]
    if (any(row["config"]["seed"] != seed or row["config"]["pilot"] or
            row["config"]["dataset"] != first["config"]["dataset"] or
            row["config"]["backbone"] != first["config"]["backbone"] or
            row["identity"]["prepared_manifest_sha256"] !=
            first["identity"]["prepared_manifest_sha256"] or
            row["quotas"] != first["quotas"] or
            row["sample_ids"] != first["sample_ids"]
            for seed, row in zip(seeds, audited)) or
            (policy in (ISIC_DRAFT_POLICY, ISIC_CACHED_DRAFT_POLICY) and
             (first["config"]["dataset"] != "isic2020" or
              first["config"]["backbone"] !=
              ("mobilenet_v3_large" if seeds[0] in (6811, 6831, 6891) else
               "convnext_tiny" if seeds[0] == 6841 else "vit_b_16")))):
        raise RuntimeError("fixed seed/data/backbone parity failed before labels")
    if sum(row["elapsed_seconds"] for row in audited) / 3600 > 96:
        raise RuntimeError("complete block exceeds aggregate weekend ceiling")
    cost = _completed_cell_cost(full_root, audited, seeds[0] - 1)
    labels, private_identity = _private_labels(
        prepared_root, first["sample_ids"],
        first["identity"]["prepared_manifest_sha256"])
    per_seed = []
    for row in audited:
        levels = {}
        for level, quota in row["quotas"].items():
            arms = {}
            for arm, p in row["probabilities"].items():
                values = p.tolist()
                raw = [int(item[1] > item[0]) for item in values]
                allocated = allocate_local_upper_bound(
                    values, [None, quota["global_cap"]], row["sample_ids"],
                    row["groups"], quota["local_caps"])
                if (sum(value == 1 for value in allocated) > quota["global_cap"] or
                        any(sum(value == 1 and group == scope
                                for value, group in zip(allocated, row["groups"])) > cap
                            for scope, cap in quota["local_caps"].items())):
                    raise RuntimeError("allocation violated a declared capacity")
                arms[arm] = {"raw": _metrics(labels, raw, row["groups"]),
                             "allocated": _metrics(labels, allocated, row["groups"])}
            levels[level] = arms
        per_seed.append({"seed": row["config"]["seed"], "levels": levels,
                         "summary_sha256": row["summary_sha256"],
                         "elapsed_seconds": row["elapsed_seconds"]})
    contrasts = {}
    for level in first["quotas"]:
        contrasts[level] = {}
        for control in ("pto", "sham", f"{level}_phr"):
            deltas = [seed["levels"][level][f"{level}_tralo"]["allocated"]["cc_f1"] -
                      seed["levels"][level][control]["allocated"]["cc_f1"]
                      for seed in per_seed]
            mean = sum(deltas) / len(deltas)
            variance = sum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1)
            half = 3.182446 * math.sqrt(variance / len(deltas))
            contrasts[level]["tralo_minus_" + control] = {
                "per_seed": deltas, "mean": mean,
                "paired_95pct_t_interval": [mean - half, mean + half]}
    result = {"status": "complete_block_scored", "dataset": first["config"]["dataset"],
              "scorer_sha256": digest(__file__),
              "backbone": first["config"]["backbone"],
              "pretrained_weight": first["identity"]["weight"],
              "runner_source_sha256": first["identity"]["source_sha256"],
              "queue_cost_and_ownership": cost,
              "private_labels": private_identity, "seeds": per_seed,
              "paired_contrasts": contrasts,
              "training_only_gpu_hours": sum(row["elapsed_seconds"] for row in audited) / 3600,
              "interpretation": "development comparison; requires independent held-out replication"}
    with Path(output).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    if sys.argv[1] == "--gate":
        gate(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "--score":
        score(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit(__doc__)
