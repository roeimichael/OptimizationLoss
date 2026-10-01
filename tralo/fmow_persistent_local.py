"""Proposed persistent fmow2 pooled-plus-country training comparison.

Preparation only. No seed may run until its protocol, release, data and GPU
gates have been frozen independently. Usage after release:
python -m tralo.fmow_persistent_local DATA_ROOT CONFIG_JSON NEW_OUTPUT_ROOT
"""

import copy
import hashlib
import json
import math
import random
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .fmow_yuval import (ArrayImages, CAPPED, CLASSES, FILES, load,
                         make_model, pool_chunks, transforms_for)
from .fmow_local import budgets
from .global_comparison import _state_hash, audited_arm_log
from .knee_end_to_end import infer
from .knee_experiment import cuda_setup, digest, save, source
from .local_alm import snapshot_phr_step
from .local_targeted_step import local_targeted_step


STUDY = "fmow_persistent_local_v1"
PILOT = 6700
FULL_SEEDS = range(6701, 6713)
EPOCHS = 7
SNAPSHOTS = (5, 6, 7)
CORRECTION_EPOCHS = range(2, 8)
CONFIG = dict(study=STUDY, backbone="mobilenet_v3_large", epochs=EPOCHS,
              batch_size=32, development_batch_size=16, lr=1e-4,
              weight_decay=0.0, decay_epoch=5, decay_factor=0.8,
              radius=0.1, rho=0.5, focal_alpha=0.25, focal_gamma=2.0)
PREPROCESSING = dict(size=[224, 224], color="RGB",
                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                     train_augmentation=dict(horizontal_flip=0.5, rotation_degrees=3,
                                             affine_translate=[0.1, 0.1],
                                             affine_scale=[0.9, 1.1],
                                             color_jitter=0.2))


def validate_config(config):
    expected = set(CONFIG) | {"seed", "pilot", "reference", "pretrained_sha256"}
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError("persistent protocol requires every explicit config key")
    for key, value in CONFIG.items():
        if type(config[key]) is not type(value) or config[key] != value:
            raise ValueError("persistent protocol mismatch: " + key)
    if (type(config["pilot"]) is not bool or type(config["reference"]) is not bool or
            type(config["seed"]) is not int):
        raise ValueError("pilot, reference and seed must have exact types")
    if config["pilot"] != (config["seed"] == PILOT) or config["seed"] not in (PILOT, *FULL_SEEDS):
        raise ValueError("seed is outside pilot/full family")
    if config["reference"] and not config["pilot"]:
        raise ValueError("only the pilot seed has a separate reference")
    sha = config["pretrained_sha256"]
    if type(sha) is not str or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("pretrained checkpoint SHA-256 must be explicit")


def pretrained_weight_provenance(expected_sha):
    """Require cached V2 bytes; never download different weights during a run."""
    from torchvision import models

    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
    path = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    if not path.is_file() or digest(path) != expected_sha:
        raise RuntimeError("MobileNetV3-Large V2 pretrained checkpoint is absent or changed")
    return {"file": str(path), "sha256": expected_sha, "weight_enum": str(weights)}


def stable_state_hash(value):
    """Hash nested optimizer/model state independent of pickle container bytes."""
    h = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            data = item.detach().cpu().contiguous()
            h.update(b"T")
            h.update(str(data.dtype).encode())
            h.update(json.dumps(list(data.shape)).encode())
            h.update(data.numpy().tobytes())
        elif isinstance(item, dict):
            h.update(b"D")
            for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(b"L" if isinstance(item, list) else b"U")
            h.update(str(len(item)).encode())
            for entry in item:
                visit(entry)
        elif item is None or isinstance(item, (bool, int, float, str)):
            h.update(json.dumps([type(item).__name__, item], allow_nan=False).encode())
        else:
            raise TypeError("unhashable state entry: " + type(item).__name__)

    visit(value)
    return h.hexdigest()


def clone_warm_branch(model, optimizer):
    """Clone weights and Adam moments, remapping state to the clone's params."""
    clone = copy.deepcopy(model)
    defaults = dict(optimizer.defaults)
    branch_optimizer = torch.optim.Adam(clone.parameters(), **defaults)
    branch_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    own = set(clone.parameters())
    if set(branch_optimizer.state) != own or any(
            param not in own for group in branch_optimizer.param_groups
            for param in group["params"]):
        raise RuntimeError("cloned optimizer is not bound to cloned model")
    if stable_state_hash(branch_optimizer.state_dict()) != stable_state_hash(optimizer.state_dict()):
        raise RuntimeError("warmup optimizer moments changed during clone")
    if _state_hash(clone) != _state_hash(model):
        raise RuntimeError("warmup model changed during clone")
    return clone, branch_optimizer


def focal_loss(logits, labels, *, alpha=0.25, gamma=2.0):
    """Multiclass focal: mean -alpha (1-p_true)^gamma log(p_true)."""
    if logits.ndim != 2 or labels.ndim != 1 or len(logits) != len(labels):
        raise ValueError("focal logits and labels must align")
    logp = logits.log_softmax(1).gather(1, labels[:, None]).squeeze(1)
    p = logp.exp()
    return (-alpha * (1 - p).pow(gamma) * logp).mean()


def sample_orders(data, seed, epochs):
    """Record one balanced with-replacement index order per epoch."""
    generator = torch.Generator().manual_seed(seed + 105729)
    weights = data.weights()
    return [torch.multinomial(weights, len(weights), replacement=True,
                              generator=generator).tolist() for _ in range(epochs)]


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _batch_seed(seed, epoch, batch_index, forward):
    return seed * 1000003 + epoch * 10007 + batch_index * 2 + int(forward)


def train_epoch(model, optimizer, data, train_transform, order, seed, epoch,
                epoch_index, loss_kind, *, batch_size, base_lr):
    """Train one branch with common sample/augmentation/dropout random streams.

    The epoch index is zero-based. Each batch's transform and forward receive
    their own fixed RNG state, so different branch losses cannot shift later
    augmentation draws. The supplied order is identical across all arms.
    """
    if loss_kind not in ("ce", "focal") or not order or batch_size <= 0:
        raise ValueError("invalid training epoch inputs")
    if any(type(index) is not int or not 0 <= index < len(data.labels) for index in order):
        raise ValueError("sample order contains an invalid training index")
    before_model_sha = _state_hash(model)
    before_optimizer_sha = stable_state_hash(optimizer.state_dict())
    device = next(model.parameters()).device
    cuda_devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                    if device.type == "cuda" else [])
    for group in optimizer.param_groups:
        group["lr"] = base_lr
    model.train()
    sample_ids = [int(data.indices[index]) for index in order]
    order_sha = _sha_bytes(json.dumps(sample_ids, separators=(",", ":")).encode())
    first_batch_sha = None
    total_loss, applied, max_grad = 0.0, 0, 0.0
    with torch.random.fork_rng(devices=cuda_devices):
        for batch_index, start in enumerate(range(0, len(order), batch_size)):
            torch.manual_seed(_batch_seed(seed, epoch, batch_index, False))
            images, labels = data.batch(order[start:start + batch_size], train_transform)
            if first_batch_sha is None:
                first_batch_sha = _sha_bytes(images.numpy().tobytes())
            images, labels = images.to(device), labels.to(device)
            torch.manual_seed(_batch_seed(seed, epoch, batch_index, True))
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = (F.cross_entropy(logits, labels) if loss_kind == "ce" else
                    focal_loss(logits, labels))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite supervised loss")
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads:
                raise RuntimeError("missing task gradient")
            # Reduce on the device before transferring a single scalar: one
            # host synchronization per batch, not one per model tensor.
            grad_sq = torch.stack([g.detach().double().square().sum()
                                   for g in grads]).sum()
            grad_norm = float(grad_sq.sqrt())
            if not math.isfinite(grad_norm):
                raise RuntimeError("nonfinite task gradient")
            max_grad = max(max_grad, grad_norm)
            optimizer.step()
            applied += 1
            total_loss += float(loss.detach()) * len(labels)
    if any(not bool(torch.isfinite(param).all()) for param in model.parameters()):
        raise RuntimeError("nonfinite supervised weights")
    return dict(epoch=epoch, loss_kind=loss_kind, training_loss=total_loss / len(order),
                base_lr=base_lr, last_lr=base_lr,
                pre_epoch_model_sha256=before_model_sha,
                pre_epoch_optimizer_sha256=before_optimizer_sha,
                sample_order_sha256=order_sha,
                first_batch_sha256=first_batch_sha,
                attempted_task_updates=applied, applied_task_updates=applied,
                skipped_task_updates=0, max_task_gradient_norm=max_grad,
                model_sha256=_state_hash(model),
                optimizer_state_sha256=stable_state_hash(optimizer.state_dict()))


def stop_loss(model, batches, loss_kind):
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total = count = 0
    try:
        with torch.no_grad():
            for images, labels in batches:
                logits = model(images.to(device))
                loss = (F.cross_entropy(logits, labels.to(device)) if loss_kind == "ce"
                        else focal_loss(logits, labels.to(device)))
                total += float(loss) * len(labels)
                count += len(labels)
    finally:
        model.train(was_training)
    return total / count


def observations(probabilities, groups, quota):
    """Label-free soft residuals and raw hard calls, pooled then country."""
    q = probabilities[:, CAPPED]
    calls = probabilities.argmax(1) == CAPPED
    names = ["global"] + sorted(quota["local_caps"])
    caps = [quota["global_cap"]] + [quota["local_caps"][name] for name in names[1:]]
    positions = [list(range(len(groups)))] + [
        [i for i, group in enumerate(groups) if group == name] for name in names[1:]]
    scopes = {}
    for name, cap, indices in zip(names, caps, positions):
        soft = float(q[indices].sum())
        hard = int(calls[indices].sum())
        residual = (soft - cap) / max(cap, 1)
        scopes[name] = dict(cap=cap, soft=soft, hard=hard,
                            signed_residual=residual,
                            positive_residual=max(residual, 0.0),
                            hard_excess=max(hard - cap, 0))
    return scopes


def load_label_free_training_data(data_root):
    """The only dataset entry: stop/train labels, development images and IDs."""
    images, train_labels, pool_rows, roles = load(data_root, include_pool_labels=False)
    if any("label" in row for row in pool_rows):
        raise RuntimeError("development labels entered the runner")
    return images, train_labels, pool_rows, roles


def _buffer_hash(model):
    return stable_state_hash({name: value for name, value in model.named_buffers()})


def write_pre_correction_snapshot(directory, epoch, arm, cap_divisor, model,
                                  optimizer, dual, probabilities, pool,
                                  sample_ids):
    """Persist pilot-only state before a controller acts, for independent replay."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"epoch{epoch:02d}_pre_correction"
    checkpoint_path = directory / f"{prefix}.pt"
    probability_path = directory / f"{prefix}_probabilities.pt"
    dual_path = directory / f"{prefix}_dual.json"
    if any(path.exists() for path in (checkpoint_path, probability_path, dual_path)):
        raise FileExistsError("pilot pre-correction snapshot already exists")
    if len(sample_ids) != len(probabilities) or not pool or len(pool[0]) > len(probabilities):
        raise ValueError("pre-correction pool IDs and probabilities differ")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("pre-correction pool IDs repeat")
    numpy_rng = np.random.get_state()
    state = dict(epoch=epoch, arm=arm, cap_divisor=cap_divisor,
                 model=model.state_dict(), optimizer=optimizer.state_dict(),
                 dual=dual.detach().cpu().clone() if dual is not None else None,
                 model_training=model.training,
                 rng=dict(torch_cpu=torch.get_rng_state().clone(),
                          torch_cuda=[value.clone() for value in torch.cuda.get_rng_state_all()]
                          if torch.cuda.is_available() else [],
                          python=random.getstate(),
                          numpy=dict(bit_generator=numpy_rng[0],
                                     state=numpy_rng[1].tolist(),
                                     position=int(numpy_rng[2]),
                                     has_gauss=int(numpy_rng[3]),
                                     cached_gaussian=float(numpy_rng[4]))))
    first_input_sha = _sha_bytes(pool[0].detach().cpu().contiguous().numpy().tobytes())
    predictions = dict(sample_ids=list(sample_ids),
                       probabilities=probabilities.detach().cpu().clone(),
                       first_batch_probabilities=probabilities[:len(pool[0])].detach().cpu().clone(),
                       first_batch_input_sha256=first_input_sha)
    with checkpoint_path.open("xb") as stream:
        torch.save(state, stream)
    with probability_path.open("xb") as stream:
        torch.save(predictions, stream)
    save(dual_path, dual.tolist() if dual is not None else None)
    return dict(checkpoint_file=checkpoint_path.name,
                checkpoint_sha256=digest(checkpoint_path),
                probability_file=probability_path.name,
                probability_sha256=digest(probability_path),
                dual_file=dual_path.name, dual_sha256=digest(dual_path),
                model_state_sha256=_state_hash(model),
                optimizer_state_sha256=stable_state_hash(optimizer.state_dict()),
                first_batch_input_sha256=first_input_sha,
                sample_ids_sha256=_sha_bytes(json.dumps(
                    sample_ids, separators=(",", ":")).encode()))


def correction(model, optimizer, pool, groups, quota, arm, dual, config,
               *, pre_snapshot=None):
    """One persistent correction on this branch, with optimizer/RNG neutrality."""
    before_probs = infer(model, pool)
    before = observations(before_probs, groups, quota)
    model_before = _state_hash(model)
    optimizer_before = stable_state_hash(optimizer.state_dict())
    buffers_before = _buffer_hash(model)
    rng_cpu = torch.get_rng_state().clone()
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    pilot_pre = None
    if pre_snapshot is not None:
        if arm not in ("tralo", "phr") or not config["pilot"] or config["reference"]:
            raise ValueError("pre-correction evidence is pilot treated-arm only")
        pilot_pre = write_pre_correction_snapshot(
            pre_snapshot["directory"], pre_snapshot["epoch"], arm,
            pre_snapshot["cap_divisor"], model, optimizer, dual, before_probs,
            pool, pre_snapshot["sample_ids"])
        if (pilot_pre["model_state_sha256"] != model_before or
                pilot_pre["optimizer_state_sha256"] != optimizer_before):
            raise RuntimeError("pre-correction snapshot changed branch state")
    if arm == "tralo":
        record = local_targeted_step(
            model, pool, groups, CAPPED, quota["global_cap"], quota["local_caps"],
            boundary_calibrated=True, max_radius=config["radius"])
        next_dual = None
    elif arm == "phr":
        record, next_dual = snapshot_phr_step(
            model, pool, groups, CAPPED, quota["global_cap"], quota["local_caps"],
            dual, rho=config["rho"], radius=config["radius"],
            boundary_calibrated=True)
    elif arm == "null":
        record, next_dual = dict(applied=False, radius=0.0, displacement=0.0,
                                 skip_reason="scheduled_zero_step"), None
    else:
        raise ValueError("unknown correction arm")
    after_probs = infer(model, pool)
    after = observations(after_probs, groups, quota)
    if stable_state_hash(optimizer.state_dict()) != optimizer_before:
        raise RuntimeError("correction changed task optimizer")
    if _buffer_hash(model) != buffers_before:
        raise RuntimeError("correction changed BatchNorm/model buffers")
    if not torch.equal(torch.get_rng_state(), rng_cpu) or any(
            not torch.equal(current, previous) for current, previous in
            zip(torch.cuda.get_rng_state_all(), rng_cuda)):
        raise RuntimeError("correction changed training RNG state")
    if arm == "null" and (_state_hash(model) != model_before or
                            not torch.equal(before_probs, after_probs)):
        raise RuntimeError("zero-step branch changed weights or predictions")
    if bool(record["applied"]) != (float(record["displacement"]) > 0):
        raise RuntimeError("correction applied flag disagrees with displacement")
    if float(record["displacement"]) > config["radius"] + 1e-5:
        raise RuntimeError("correction exceeds fixed radius")
    if arm == "phr" and next_dual is None:
        raise RuntimeError("PHR dual failed to advance")
    result = dict(method=arm, attempted_constraint_updates=int(arm != "null"),
                applied_constraint_updates=int(bool(record["applied"])),
                skipped_constraint_updates=int(arm != "null" and not record["applied"]),
                before=before, after=after, model_before_sha256=model_before,
                model_after_sha256=_state_hash(model),
                optimizer_before_sha256=optimizer_before,
                optimizer_after_sha256=stable_state_hash(optimizer.state_dict()),
                buffer_sha256=buffers_before, rng_neutral=True,
                gradient_norm=record.get("gradient_norm"),
                attempted_radius=config["radius"] if arm != "null" else 0.0,
                applied_radius=float(record["radius"]),
                actual_displacement=float(record["displacement"]),
                controller=record, dual_after=next_dual.tolist() if next_dual is not None else None)
    if pilot_pre is not None:
        pilot_pre["post_model_state_sha256"] = result["model_after_sha256"]
        pilot_pre["post_optimizer_state_sha256"] = result["optimizer_after_sha256"]
        result["pre_correction_snapshot"] = pilot_pre
    return result, next_dual


def write_snapshot(directory, epoch, model, optimizer, probabilities, dual):
    """Save post-correction state, predictions and dual with byte-level hashes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"epoch{epoch:02d}_post.pt"
    probability_path = directory / f"epoch{epoch:02d}_probabilities.pt"
    dual_path = directory / f"epoch{epoch:02d}_dual.json"
    if any(path.exists() for path in (checkpoint_path, probability_path, dual_path)):
        raise FileExistsError("persistent snapshot already exists")
    torch.save(dict(epoch=epoch, model=model.state_dict(),
                    optimizer=optimizer.state_dict(), dual=dual), checkpoint_path)
    torch.save(probabilities.detach().cpu(), probability_path)
    save(dual_path, dual.tolist() if dual is not None else None)
    return dict(checkpoint_file=checkpoint_path.name,
                checkpoint_sha256=digest(checkpoint_path),
                probability_file=probability_path.name,
                probability_sha256=digest(probability_path),
                dual_file=dual_path.name, dual_sha256=digest(dual_path),
                model_state_sha256=_state_hash(model),
                optimizer_state_sha256=stable_state_hash(optimizer.state_dict()))


def run(data_root, config_path, output):
    config = json.loads(Path(config_path).read_text())
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with audited_arm_log(output / "events.jsonl") as log:
        cuda_setup()
        weight = pretrained_weight_provenance(config["pretrained_sha256"])
        images, train_labels, pool_rows, roles = load_label_free_training_data(data_root)
        groups = [row["location"] for row in pool_rows]
        quota = budgets(groups)
        split = dict(train=roles["train"], stop=roles["stop"], dev=roles["dev"],
                     stop_countries=roles["stop_countries"],
                     dev_countries=roles["dev_countries"],
                     reserved_countries=roles["reserved_countries"])
        split_sha = _sha_bytes(json.dumps(split, sort_keys=True, separators=(",", ":")).encode())
        identity = dict(source_sha256=source(), config_sha256=digest(config_path),
                        data_files=FILES, split_sha256=split_sha,
                        pretrained_weight=weight,
                        preprocessing=PREPROCESSING,
                        preprocessing_sha256=_sha_bytes(json.dumps(
                            PREPROCESSING, sort_keys=True, separators=(",", ":")).encode()),
                        task_loss=dict(ce="torch.nn.functional.cross_entropy",
                                       focal="mean(-0.25*(1-p_true)^2*log(p_true))"),
                        task_optimizer="Adam, weight_decay=0, epoch decay x0.8 after epoch5",
                        development_labels_loaded=False,
                        reserved_images_used=False)
        shutil.copyfile(config_path, output / "config.json")
        if digest(output / "config.json") != identity["config_sha256"]:
            raise RuntimeError("copied config differs from immutable input bytes")
        save(output / "manifest.json", dict(**identity, split=split,
                                            pool_rows=pool_rows,
                                            quotas=quota,
                                            country_sizes={g: groups.count(g) for g in sorted(set(groups))},
                                            train_label_counts=[int((train_labels[roles["train"]] == c).sum())
                                                                for c in range(CLASSES)]))
        log.emit("started", **identity, quotas=quota,
                 manifest_sha256=digest(output / "manifest.json"),
                 device=torch.cuda.get_device_name(), precision="fp32_tf32_off")
        train_tf, eval_tf = transforms_for()
        train = ArrayImages(images["train"], roles["train"], train_labels)
        held = ArrayImages(images["train"], roles["stop"], train_labels)
        stop = [held.batch(list(range(start, min(start + config["batch_size"], len(held.labels)))),
                           eval_tf)
                for start in range(0, len(held.labels), config["batch_size"])]
        pool = pool_chunks(images["test"], roles["dev"], eval_tf,
                           config["development_batch_size"])
        torch.manual_seed(config["seed"])
        initial = make_model(backbone="mobilenet_v3_large").cuda()
        initial_sha = _state_hash(initial)
        orders = sample_orders(train, config["seed"], EPOCHS)
        ce_model = copy.deepcopy(initial)
        ce_optimizer = torch.optim.Adam(ce_model.parameters(), lr=config["lr"],
                                        weight_decay=0.0)
        focal_model = copy.deepcopy(initial)
        focal_optimizer = torch.optim.Adam(focal_model.parameters(), lr=config["lr"],
                                           weight_decay=0.0)
        if _state_hash(ce_model) != initial_sha or _state_hash(focal_model) != initial_sha:
            raise RuntimeError("arms differ at initialization")
        warm = train_epoch(ce_model, ce_optimizer, train, train_tf, orders[0],
                           config["seed"], 1, 0, "ce", batch_size=config["batch_size"],
                           base_lr=config["lr"])
        warm["stop_loss"] = stop_loss(ce_model, stop, "ce")
        log.emit("ce_warmup_completed", **warm)
        arms = {"ce_null": (ce_model, ce_optimizer, "ce", None)}
        if not config["reference"]:
            for divisor in (10, 20):
                for method in ("tralo", "phr"):
                    model, optimizer = clone_warm_branch(ce_model, ce_optimizer)
                    dual = (torch.zeros(1 + len(quota[str(divisor)]["local_caps"]))
                            if method == "phr" else None)
                    arms[f"cap{divisor}_{method}"] = model, optimizer, method, dual
            if config["pilot"]:
                for name in ("tralo_null", "alm_null"):
                    model, optimizer = clone_warm_branch(ce_model, ce_optimizer)
                    arms[name] = model, optimizer, "null", None
            arms["focal_clip"] = focal_model, focal_optimizer, "focal", None
        summaries = {}
        for name, (model, optimizer, method, dual) in arms.items():
            directory = output / name
            directory.mkdir()
            loss_kind = "focal" if method == "focal" else "ce"
            epochs = [] if method == "focal" else [copy.deepcopy(warm)]
            correction_records = []
            first = 0 if method == "focal" else 1
            for epoch_index in range(first, EPOCHS):
                epoch = epoch_index + 1
                lr = config["lr"] * config["decay_factor"] ** (
                    epoch_index // config["decay_epoch"])
                row = train_epoch(model, optimizer, train, train_tf, orders[epoch_index],
                                  config["seed"], epoch, epoch_index, loss_kind,
                                  batch_size=config["batch_size"], base_lr=lr)
                row["stop_loss"] = stop_loss(model, stop, loss_kind)
                if epoch in CORRECTION_EPOCHS:
                    if method in ("tralo", "phr"):
                        divisor = name.split("_")[0][3:]
                        pre_snapshot = (dict(directory=directory, epoch=epoch,
                                             cap_divisor=int(divisor),
                                             sample_ids=[row["sample_id"] for row in pool_rows])
                                        if config["pilot"] else None)
                        rec, dual = correction(model, optimizer, pool, groups,
                                               quota[divisor], method, dual, config,
                                               pre_snapshot=pre_snapshot)
                        rec.update(epoch=epoch, cap_divisor=int(divisor))
                        correction_records.append(rec)
                        log.emit("correction", arm=name, **rec)
                    else:
                        # CE, focal and explicit nulls all occupy the same
                        # post-task no-op slot; only task loss differs for focal.
                        for divisor in (10, 20):
                            rec, _ = correction(model, optimizer, pool, groups,
                                                quota[str(divisor)], "null", None, config)
                            rec.update(epoch=epoch, cap_divisor=divisor)
                            correction_records.append(rec)
                            log.emit("correction", arm=name, **rec)
                    row["post_correction_stop_loss"] = stop_loss(model, stop, loss_kind)
                if epoch in SNAPSHOTS:
                    probabilities = infer(model, pool)
                    artifact = write_snapshot(directory, epoch, model, optimizer,
                                              probabilities, dual)
                    row["post_correction_snapshot"] = artifact
                    log.emit("snapshot", arm=name, epoch=epoch, **artifact)
                row["post_epoch_model_sha256"] = _state_hash(model)
                row["post_epoch_optimizer_sha256"] = stable_state_hash(optimizer.state_dict())
                epochs.append(row)
                log.emit("epoch", arm=name, **row)
            summaries[name] = dict(loss_kind=loss_kind, epochs=epochs,
                                   corrections=correction_records,
                                   final_model_sha256=_state_hash(model),
                                   final_optimizer_sha256=stable_state_hash(optimizer.state_dict()))
        baseline_epochs = summaries["ce_null"]["epochs"]
        for name, arm_summary in summaries.items():
            for epoch_index, (baseline, actual) in enumerate(
                    zip(baseline_epochs, arm_summary["epochs"]), start=1):
                if (baseline["sample_order_sha256"] != actual["sample_order_sha256"] or
                        baseline["first_batch_sha256"] != actual["first_batch_sha256"] or
                        baseline["applied_task_updates"] != actual["applied_task_updates"]):
                    raise RuntimeError(f"{name} lost matched training input at epoch {epoch_index}")
        if config["pilot"] and not config["reference"]:
            ce = summaries["ce_null"]
            for name in ("tralo_null", "alm_null"):
                other = summaries[name]
                if (ce["final_model_sha256"] != other["final_model_sha256"] or
                        ce["final_optimizer_sha256"] != other["final_optimizer_sha256"] or
                        any(ce["epochs"][i]["sample_order_sha256"] !=
                            other["epochs"][i]["sample_order_sha256"] or
                            ce["epochs"][i]["first_batch_sha256"] !=
                            other["epochs"][i]["first_batch_sha256"]
                            for i in range(EPOCHS))):
                    raise RuntimeError(name + " failed exact PTO replay")
                for epoch in SNAPSHOTS:
                    a = torch.load(output / "ce_null" / f"epoch{epoch:02d}_probabilities.pt",
                                   map_location="cpu", weights_only=True)
                    b = torch.load(output / name / f"epoch{epoch:02d}_probabilities.pt",
                                   map_location="cpu", weights_only=True)
                    if not torch.equal(a, b):
                        raise RuntimeError(name + " prediction differs from CE PTO")
        save(output / "summary.json", dict(seed=config["seed"], pilot=config["pilot"],
                                           reference=config["reference"],
                                           identity=identity, initial_model_sha256=initial_sha,
                                           ce_warmup=warm, quotas=quota, arms=summaries,
                                           elapsed_seconds=time.monotonic() - started))
        log.emit("completed", arms=list(arms),
                 summary_sha256=digest(output / "summary.json"),
                 elapsed_seconds=time.monotonic() - started)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    run(sys.argv[1], sys.argv[2], sys.argv[3])
