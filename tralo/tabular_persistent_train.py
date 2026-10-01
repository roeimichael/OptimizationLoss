"""Prospective image-only training with pooled and metadata-group quotas.

Usage: python -m tralo.tabular_persistent_train PREPARED_ROOT CONFIG_JSON NEW_OUTPUT
The runner loads training/stop targets and an unlabeled development pool only.
It never opens the scorer directory. Every arm uses the same image backbone,
sample order, augmentation stream, task optimizer and upper-bound allocator.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from .events import EventLog
from .fmow_persistent_local import stable_state_hash
from .knee_experiment import digest, save, source
from .tabular_backbones import (configure_fp32, image_transforms,
                                make_binary_model)
from .tabular_constraint_gradient import (apply_fixed_correction,
                                         streaming_parameter_gradient)
from .tabular_image_data import PreparedImageRows, load_runner_cohort
from .tabular_quota_policy import caps_for_unlabeled_pool


STUDY = "tabular_persistent_v1"
EPOCHS = 6
BACKBONES = {"mobilenet_v3_large": (32, 1e-4),
             "vit_b_16": (8, 2e-5), "convnext_tiny": (16, 5e-5)}
ARMS = ("pto", "sham", "level1_tralo", "level1_phr",
        "level2_tralo", "level2_phr")
CORRECTION_STEP_SIZE = 0.01
MAX_DISPLACEMENT = 0.1
RHO = 0.5
MULTIPLIER = 1.0


def validate_config(config):
    if (not isinstance(config, dict) or set(config) !=
            {"study", "dataset", "backbone", "seed", "pilot"} or
            config["study"] != STUDY or config["dataset"] not in
            ("isic2020", "celeba") or config["backbone"] not in BACKBONES or
            type(config["seed"]) is not int or config["seed"] < 0 or
            type(config["pilot"]) is not bool):
        raise ValueError("unfrozen tabular training configuration")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _ids_hash(ids):
    return _sha(json.dumps(ids, separators=(",", ":")).encode())


def _orders(rows, seed, dataset):
    generator = torch.Generator().manual_seed(seed + 105729)
    n = len(rows)
    if dataset == "isic2020":
        positive = sum(row["label"] for row in rows)
        if not 0 < positive < n:
            raise RuntimeError("ISIC training class support is invalid")
        weights = torch.tensor([1 / (2 * (positive if row["label"] else n - positive))
                                for row in rows], dtype=torch.double)
        return [torch.multinomial(weights, n, replacement=True,
                                  generator=generator).tolist() for _ in range(EPOCHS)]
    return [torch.randperm(n, generator=generator).tolist() for _ in range(EPOCHS)]


def _batches(data, indices, batch_size, device):
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        values = [data[index] for index in selected]
        images = torch.stack([value[0] for value in values]).to(device)
        labels = torch.tensor([value[1] for value in values],
                              dtype=torch.long, device=device)
        yield images, labels, [value[2] for value in values], [value[3] for value in values]


def _pool_batches(data, batch_size, device):
    def factory():
        for images, labels, _groups, ids in _batches(
                data, range(len(data)), batch_size, device):
            if bool((labels != -1).any()):
                raise RuntimeError("development target entered constraint pool")
            yield images, ids
    return factory


def _predict(model, batches):
    model.eval()
    result, identities = [], []
    with torch.no_grad():
        for images, ids in batches():
            values = model(images).softmax(1).cpu()
            if not bool(torch.isfinite(values).all()):
                raise RuntimeError("nonfinite development probability")
            result.append(values)
            identities.extend(ids)
    return torch.cat(result), identities


def _stop_loss(model, stop_data, batch_size, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, labels, _groups, _ids in _batches(
                stop_data, range(len(stop_data)), batch_size, device):
            total += float(F.cross_entropy(model(images), labels,
                                            reduction="sum"))
    return total / len(stop_data)


def _train_epoch(model, optimizer, data, order, transform_seed, batch_size, device):
    model.train()
    loss_sum, max_norm, first_hash, updates = 0.0, 0.0, None, 0
    for batch_index, start in enumerate(range(0, len(order), batch_size)):
        torch.manual_seed(transform_seed + 2 * batch_index)
        indices = order[start:start + batch_size]
        images, labels, _groups, _ids = next(_batches(data, indices, batch_size, device))
        if first_hash is None:
            first_hash = _sha(images.detach().cpu().contiguous().numpy().tobytes())
        torch.manual_seed(transform_seed + 2 * batch_index + 1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite task loss")
        loss.backward()
        norms = [param.grad.detach().double().square().sum()
                 for param in model.parameters() if param.grad is not None]
        norm = float(torch.sqrt(torch.stack(norms).sum())) if norms else 0.0
        if not math.isfinite(norm) or norm == 0:
            raise RuntimeError("invalid task gradient")
        max_norm = max(max_norm, norm)
        optimizer.step()
        updates += 1
        loss_sum += float(loss.detach()) * len(indices)
    return {"training_loss": loss_sum / len(order), "task_updates": updates,
            "max_task_gradient_norm": max_norm, "first_batch_sha256": first_hash,
            "sample_order_sha256": _ids_hash([data.rows[index]["sample_id"]
                                               for index in order])}


def _observations(probabilities, groups, quota):
    q = probabilities[:, 1]
    calls = q > 0.5
    scopes = {"global": (list(range(len(groups))), quota["global_cap"])}
    scopes.update({group: ([i for i, actual in enumerate(groups) if actual == group],
                           cap) for group, cap in quota["local_caps"].items()})
    result = {}
    for name, (indices, cap) in scopes.items():
        soft = float(q[indices].sum())
        hard = int(calls[indices].sum())
        result[name] = {"cap": cap, "soft": soft, "hard": hard,
                        "signed_residual": (soft - cap) / max(cap, 1),
                        "hard_excess": max(0, hard - cap)}
    return result


def _run_arm(name, initial, rows, image_dir, orders, train_tf, eval_tf,
             quotas, config, output, log, device, decode_policy=None):
    model = copy.deepcopy(initial).to(device)
    batch_size, lr = BACKBONES[config["backbone"]]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    train = PreparedImageRows(image_dir, rows["train"], train_tf, decode_policy)
    stop = PreparedImageRows(image_dir, rows["stop"], eval_tf, decode_policy)
    pool = PreparedImageRows(image_dir, rows["development_pool"], eval_tf,
                             decode_policy)
    pool_groups = [row["group"] for row in rows["development_pool"]]
    expected_ids = [row["sample_id"] for row in rows["development_pool"]]
    pool_batches = _pool_batches(pool, batch_size, device)
    treated = name.endswith("_tralo") or name.endswith("_phr")
    level = name.split("_")[0] if treated else None
    dual = ({scope: 0.0 for scope in ("global", *quotas[level]["local_caps"])}
            if name.endswith("_phr") else None)
    best = (float("inf"), None, None)
    epochs = []
    directory = output / name
    directory.mkdir()
    for epoch, order in enumerate(orders, start=1):
        started = time.monotonic()
        row = _train_epoch(model, optimizer, train, order,
                           config["seed"] * 1000003 + epoch * 10007,
                           batch_size, device)
        corrections = []
        if epoch >= 2:
            if not treated:
                before_null, null_ids = _predict(model, pool_batches)
                if null_ids != expected_ids:
                    raise RuntimeError("development sample order changed")
            for quota_name, quota in quotas.items():
                if treated and quota_name != level:
                    continue
                if treated:
                    model.eval()
                    before, identities = _predict(model, pool_batches)
                    if identities != expected_ids:
                        raise RuntimeError("development sample order changed")
                else:
                    before = before_null
                observed_before = _observations(before, pool_groups, quota)
                optimizer_before = stable_state_hash(optimizer.state_dict())
                if treated:
                    method = "tralo" if name.endswith("_tralo") else "phr"
                    multipliers = ({scope: MULTIPLIER for scope in
                                    ("global", *quota["local_caps"])}
                                   if method == "tralo" else None)
                    gradient = streaming_parameter_gradient(
                        model, pool_batches, pool_groups, quota, method,
                        multipliers=multipliers, dual=dual, rho=RHO)
                    correction = apply_fixed_correction(
                        model, CORRECTION_STEP_SIZE, MAX_DISPLACEMENT)
                    if method == "phr":
                        dual = gradient["next_dual"]
                else:
                    gradient = {"parameter_gradient_norm": 0.0, "active": False,
                                "fixed_weight_probability_max_abs_gap": 0.0,
                                "sample_count": len(pool_groups), "next_dual": None}
                    correction = {"applied": False, "reason": "scheduled_zero_step",
                                  "raw_gradient_norm": 0.0,
                                  "proposed_displacement_norm": 0.0,
                                  "actual_displacement_norm": 0.0}
                if stable_state_hash(optimizer.state_dict()) != optimizer_before:
                    raise RuntimeError("correction changed task optimizer")
                if treated:
                    after, after_ids = _predict(model, pool_batches)
                    if after_ids != expected_ids:
                        raise RuntimeError("development sample order changed after correction")
                else:
                    after = before
                record = {"level": quota_name, "before": observed_before,
                          "after": _observations(after, pool_groups, quota),
                          "gradient": gradient, "correction": correction,
                          "dual": dual if treated and name.endswith("_phr") else None}
                corrections.append(record)
                log.emit("correction", arm=name, epoch=epoch, **record)
        row["stop_loss"] = _stop_loss(model, stop, batch_size, device)
        if not math.isfinite(row["stop_loss"]):
            raise RuntimeError("nonfinite stop loss")
        row["epoch"] = epoch
        row["corrections"] = corrections
        row["elapsed_seconds"] = time.monotonic() - started
        row["model_sha256"] = stable_state_hash(model.state_dict())
        row["optimizer_sha256"] = stable_state_hash(optimizer.state_dict())
        epochs.append(row)
        log.emit("epoch", arm=name, **row)
        if row["stop_loss"] < best[0]:
            checkpoint = directory / f"epoch{epoch:02d}.pt"
            with checkpoint.open("xb") as stream:
                torch.save({key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()}, stream)
            probabilities, identities = _predict(model, pool_batches)
            if identities != expected_ids:
                raise RuntimeError("selected snapshot sample order changed")
            probability_path = directory / f"epoch{epoch:02d}_probabilities.pt"
            with probability_path.open("xb") as stream:
                torch.save({"sample_ids": identities,
                            "probabilities": probabilities}, stream)
            best = (row["stop_loss"], epoch, checkpoint)
            log.emit("selected_checkpoint", arm=name, epoch=epoch,
                     checkpoint_sha256=digest(checkpoint),
                     probability_sha256=digest(probability_path), stop_loss=best[0])
    model.cpu()
    return {"epochs": epochs, "selected_epoch": best[1],
            "selected_stop_loss": best[0],
            "checkpoint": str(best[2].relative_to(output)),
            "checkpoint_sha256": digest(best[2]),
            "probabilities": str((best[2].parent /
                                  f"epoch{best[1]:02d}_probabilities.pt").relative_to(output)),
            "probability_sha256": digest(best[2].parent /
                                           f"epoch{best[1]:02d}_probabilities.pt"),
            "final_model_sha256": stable_state_hash(model.state_dict())}


def run(prepared_root, config_path, output_root):
    config_path, output = Path(config_path), Path(output_root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for a claimed image training run")
    configure_fp32()
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    manifest, rows = load_runner_cohort(prepared_root, config["dataset"])
    expected = Path(prepared_root) / "manifest.json"
    image_dir = manifest["image_dir"]
    quotas = caps_for_unlabeled_pool(
        config["dataset"], [row["group"] for row in rows["development_pool"]])
    output.mkdir(parents=True, exist_ok=False)
    with EventLog(output / "events.jsonl") as log:
        torch.manual_seed(config["seed"])
        initial, weight = make_binary_model(config["backbone"])
        initial_sha = stable_state_hash(initial.state_dict())
        train_tf, eval_tf = image_transforms(config["backbone"])
        identity = {"source_sha256": source(), "config_sha256": digest(config_path),
                    "prepared_manifest_sha256": digest(expected),
                    "runner_files_sha256": manifest["files_sha256"],
                    "weight": weight, "initial_model_sha256": initial_sha,
                    "precision": "fp32_tf32_off", "development_labels_loaded": False,
                    "preprocessing": {"train": repr(train_tf), "eval": repr(eval_tf),
                                      "decode_policy": manifest.get("decode_policy")},
                    "device": torch.cuda.get_device_name()}
        (output / "config.json").write_bytes(config_path.read_bytes())
        save(output / "manifest.json", {"identity": identity, "quotas": quotas,
                                        "pool_ids_sha256": _ids_hash([
                                            row["sample_id"] for row in rows["development_pool"]]),
                                        "split_sizes": {key: len(value) for key, value in rows.items()}})
        log.emit("started", **identity, quotas=quotas,
                 manifest_sha256=digest(output / "manifest.json"))
        orders = _orders(rows["train"], config["seed"], config["dataset"])
        device = torch.device("cuda")
        results = {}
        for name in ARMS:
            results[name] = _run_arm(name, initial, rows, image_dir, orders,
                                     train_tf, eval_tf, quotas, config, output,
                                     log, device, manifest.get("decode_policy"))
        baseline = results["pto"]["epochs"]
        null = results["sham"]["epochs"]
        if (results["pto"]["final_model_sha256"] !=
                results["sham"]["final_model_sha256"] or any(
                    left["model_sha256"] != right["model_sha256"] or
                    left["sample_order_sha256"] != right["sample_order_sha256"] or
                    left["first_batch_sha256"] != right["first_batch_sha256"]
                    for left, right in zip(baseline, null))):
            raise RuntimeError("schedule-matched zero step changed PTO trajectory")
        for name, record in results.items():
            if any(left["sample_order_sha256"] != right["sample_order_sha256"] or
                   left["first_batch_sha256"] != right["first_batch_sha256"] or
                   left["task_updates"] != right["task_updates"]
                   for left, right in zip(baseline, record["epochs"])):
                raise RuntimeError("task dose or training inputs differ: " + name)
        save(output / "summary.json", {"config": config, "identity": identity,
                                       "arms": results, "elapsed_seconds":
                                       time.monotonic() - started})
        log.emit("completed", summary_sha256=digest(output / "summary.json"),
                 elapsed_seconds=time.monotonic() - started)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    run(sys.argv[1], sys.argv[2], sys.argv[3])
