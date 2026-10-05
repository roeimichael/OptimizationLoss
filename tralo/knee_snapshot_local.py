"""Isolated snapshot corrections for the approved knee local-attribution pilot.

This module is preparation infrastructure, not a campaign launcher. It reuses
the existing derivative/controller implementations without changing old CLIs.
"""

import copy
import math
from pathlib import Path
import random

import torch

from .global_comparison import _state_hash
from .knee_end_to_end import infer
from .knee_snapshot_data import sha256
from .local_targeted_step import local_targeted_step
from .targeted_step import targeted_step

SEEDS = (7001, 7002, 7003, 7004)
ARMS = ("pto", "global_native", "global_native_sham", "joint_local",
        "global_at_joint_radius", "joint_sham")
RECIPE = dict(backbone="mobilenet_v3_large", cap=76, max_epochs=75, patience=5,
              batch_size=32, lr=1e-4, weight_decay=0.0, decay_epoch=5,
              decay_factor=0.8, development_batch_size=16)


def validate(config):
    if not isinstance(config, dict) or set(config) != set(RECIPE) | {"seed"}:
        raise ValueError("configuration differs from approved snapshot pilot")
    if type(config["seed"]) is not int or config["seed"] not in SEEDS:
        raise ValueError("seed outside the four prospective reservations")
    for key, value in RECIPE.items():
        if type(config[key]) is not type(value) or config[key] != value:
            raise ValueError("approved fixed recipe mismatch: " + key)


def ensemble_window(best_epoch, last_epoch):
    if (type(best_epoch) is not int or type(last_epoch) is not int
            or not 1 <= best_epoch <= last_epoch):
        raise ValueError("invalid training-only best/last epoch")
    return list(range(max(1, best_epoch - 2), last_epoch + 1))


def _counts(values, groups, capped):
    calls = values.argmax(1).tolist()
    return dict(hard_global=sum(c == capped for c in calls),
                soft_global=float(values[:, capped].sum()),
                hard_local={g: sum(c == capped for c, name in zip(calls, groups) if name == g)
                            for g in sorted(set(groups))},
                soft_local={g: float(values[[i for i, name in enumerate(groups) if name == g], capped].sum())
                            for g in sorted(set(groups))})


def _norms(model, origin):
    return [float((param.detach() - before).double().norm())
            for param, before in zip(model.parameters(), origin) if param.requires_grad]


def _check_dose(real, control, *, tensor_match):
    if real.get("applied", False) != control.get("applied", False):
        raise RuntimeError("matched control activation differs")
    for key in ("radius", "displacement"):
        if not math.isclose(real.get(key, 0.0), control.get(key, 0.0), rel_tol=1e-4, abs_tol=1e-5):
            raise RuntimeError("matched control " + key + " differs")
    if tensor_match:
        a, b = real["tensor_displacement_norms"], control["tensor_displacement_norms"]
        if len(a) != len(b) or any(not math.isclose(x, y, rel_tol=1e-4, abs_tol=1e-5)
                                   for x, y in zip(a, b)):
            raise RuntimeError("matched sham per-tensor displacements differ")


def snapshot(model, pool, groups, quota, seed, epoch, directory, pto_probabilities, emit):
    """Save six copies, restoring training RNG even on a failed side correction.

    The original model and its gradients are inspected, never differentiated,
    placed, forwarded or given to an optimizer by this function. Synthetic tests
    may supply small quotas; the production quota contract belongs to the pack.
    """
    if (type(seed) is not int or type(epoch) is not int or epoch < 1
            or not isinstance(pool, (list, tuple)) or not pool
            or len(groups) != sum(len(chunk) for chunk in pool)):
        raise ValueError("invalid snapshot identity or cohort")
    if (pto_probabilities.shape != (len(groups), 5) or not torch.isfinite(pto_probabilities).all()
            or not torch.allclose(pto_probabilities.sum(1), pto_probabilities.new_ones(len(groups)), atol=1e-6, rtol=0)):
        raise ValueError("PTO probabilities do not match the public pool")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    python_rng = random.getstate()
    import numpy as np
    numpy_rng = np.random.get_state()
    original_hash, mode = _state_hash(model), model.training
    gradients = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    origin = [p.detach().clone() for p in model.parameters()]
    records, files = {}, {}
    joint_delta = pooled_delta = None
    caps = [None, None, None, quota["global_cap"], None]
    before_counts = _counts(pto_probabilities, groups, 3)

    def preserve(arm, values, record):
        values = values.detach().cpu()
        if (not torch.isfinite(values).all() or values.shape != pto_probabilities.shape
                or not torch.allclose(values.sum(1), values.new_ones(len(groups)), atol=1e-6, rtol=0)):
            raise RuntimeError("invalid side probabilities: " + arm)
        path = directory / (arm + ".pt")
        with path.open("xb") as stream:
            torch.save(values, stream)
        files[arm] = dict(file=path.name, sha256=sha256(path.read_bytes()))
        after_counts = _counts(values, groups, 3)
        record.update(before_counts=before_counts, after_counts=after_counts,
                      global_residual_before=before_counts["hard_global"] - quota["global_cap"],
                      global_residual_after=after_counts["hard_global"] - quota["global_cap"],
                      local_residual_before={g: before_counts["hard_local"][g] - k
                                             for g, k in quota["local_caps"].items()},
                      local_residual_after={g: after_counts["hard_local"][g] - k
                                            for g, k in quota["local_caps"].items()},
                      multiplier="not_applicable", augmentation_coefficient="not_applicable",
                      planned_checks=0 if arm == "pto" else 1,
                      attempted_updates=int(record.get("gradient_norm", 0.0) > 0),
                      applied_updates=int(record["applied"]),
                      skipped_updates=int(arm != "pto" and not record["applied"]))
        records[arm] = record
        emit(dict(event="snapshot_arm_completed", arm=arm, epoch=epoch, seed=seed,
                  artifact=files[arm], **record))

    try:
        preserve("pto", pto_probabilities, dict(applied=False, radius=0.0, displacement=0.0,
                                                tensor_displacement_norms=[0.0 for p in model.parameters()
                                                                           if p.requires_grad]))
        for arm in ARMS[1:]:
            emit(dict(event="snapshot_arm_started", arm=arm, epoch=epoch, seed=seed,
                      original_state_sha256=original_hash))
            side = copy.deepcopy(model)
            try:
                if not torch.equal(infer(side, pool).cpu(), pto_probabilities.cpu()):
                    raise RuntimeError("PTO reference does not replay on the isolated epoch copy")
                if arm in ("global_native", "global_native_sham"):
                    generator = (torch.Generator().manual_seed(seed + 7 + 1000 * epoch)
                                 if arm.endswith("sham") else None)
                    record = targeted_step(side, pool, caps, sham_generator=generator)
                elif arm == "joint_local":
                    record = local_targeted_step(side, pool, groups, 3, quota["global_cap"],
                                                 quota["local_caps"], r0=.001, max_doublings=30,
                                                 scan_points=24, max_radius=.1,
                                                 require_common_descent=True)
                elif not records["joint_local"]["applied"]:
                    record = dict(applied=False, radius=0.0, displacement=0.0,
                                  skip_reason="joint_inactive_matched_control")
                else:
                    kwargs = dict(fixed_radius=records["joint_local"]["radius"])
                    if arm == "global_at_joint_radius":
                        kwargs["global_only_direction"] = True
                    else:
                        kwargs["sham_generator"] = torch.Generator().manual_seed(seed + 17 + 1000 * epoch)
                    record = local_targeted_step(side, pool, groups, 3, quota["global_cap"],
                                                 quota["local_caps"], **kwargs)
                norms = _norms(side, origin)
                record.update(radius=record.get("radius", 0.0), tensor_displacement_norms=norms,
                              displacement=math.sqrt(sum(x*x for x in norms)),
                              side_state_sha256=_state_hash(side))
                if arm in ("joint_local", "global_at_joint_radius") and record["applied"]:
                    delta = [(p.detach() - o).double().cpu() for p, o in zip(side.parameters(), origin)
                             if p.requires_grad]
                    if arm == "joint_local":
                        joint_delta = delta
                    else:
                        pooled_delta = delta
                values = infer(side, pool)
                if not record["applied"] and not torch.equal(values.cpu(), pto_probabilities.cpu()):
                    raise RuntimeError("inactive snapshot differs from PTO")
                preserve(arm, values, record)
            except BaseException as exc:
                emit(dict(event="snapshot_arm_failed", arm=arm, epoch=epoch, seed=seed,
                          exception=type(exc).__name__, reason=str(exc), attempted_checks=1))
                raise
            finally:
                del side
        _check_dose(records["global_native"], records["global_native_sham"], tensor_match=True)
        _check_dose(records["joint_local"], records["global_at_joint_radius"], tensor_match=False)
        _check_dose(records["joint_local"], records["joint_sham"], tensor_match=True)
        alignment = None
        if joint_delta is not None and pooled_delta is not None:
            dot = sum(float((a*b).sum()) for a, b in zip(joint_delta, pooled_delta))
            denom = records["joint_local"]["displacement"] * records["global_at_joint_radius"]["displacement"]
            alignment = dot / denom if denom else None
        emit(dict(event="snapshot_completed", epoch=epoch, seed=seed,
                  joint_pooled_displacement_cosine=alignment, original_state_sha256=original_hash,
                  files=files))
        return dict(arms=records, files=files, joint_pooled_displacement_cosine=alignment)
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        if (_state_hash(model) != original_hash or model.training != mode
                or any((p.grad is None) != (g is None) or (g is not None and not torch.equal(p.grad, g))
                       for p, g in zip(model.parameters(), gradients))):
            raise RuntimeError("side correction changed original PTO state/gradients/mode")


def average_snapshots(directory, records, best_epoch, last_epoch):
    """Authenticate all six files in the same training-only selected window."""
    window = ensemble_window(best_epoch, last_epoch)
    averages = {}
    for arm in ARMS:
        values = []
        for epoch in window:
            artifact = records[str(epoch)]["files"][arm]
            if artifact["file"] != arm + ".pt":
                raise ValueError("unexpected snapshot artifact path")
            path = Path(directory) / f"epoch{epoch:02d}" / artifact["file"]
            data = path.read_bytes()
            if sha256(data) != artifact["sha256"]:
                raise ValueError("snapshot bytes changed")
            import io
            values.append(torch.load(io.BytesIO(data), map_location="cpu", weights_only=True))
        averages[arm] = torch.stack(values).mean(0)
    return window, averages
