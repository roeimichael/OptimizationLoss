"""Exclusive, label-free CUDA memory smoke for the fixed fmow2 ViT-B/16 study.

Usage: python -m tools.fmow_local_boundary_vit_smoke RELEASE_SHA DATA_ROOT GPU_INDEX NEW_RECEIPT_JSON
Run from the immutable release with no CUDA context already present. The
receipt must live below /home/dsi/michaer8/tralo-rebuild/runs and is never
overwritten. Failures leave a negative receipt and any partial side artifacts.
"""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


RUNS = Path("/home/dsi/michaer8/tralo-rebuild/runs")
WEIGHT_SHA = "c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9"
PHASES = ("train_backward", "development_inference", "side_copy_constraint_gradient")


def _command(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def _physical_uuid(index):
    value = _command("nvidia-smi", "-i", str(index), "--query-gpu=uuid",
                     "--format=csv,noheader")
    if not value.startswith("GPU-") or "\n" in value:
        raise RuntimeError("physical GPU UUID unavailable")
    return value


def _assert_gpu_free(index, uuid):
    if _physical_uuid(index) != uuid:
        raise RuntimeError("physical GPU index changed")
    pids = _command("nvidia-smi", "-i", uuid, "--query-compute-apps=pid",
                    "--format=csv,noheader")
    if pids:
        raise RuntimeError("GPU has compute PIDs: " + pids)


def _verify_release(release, sha):
    if _command("git", "-c", "gc.auto=0", "-C", str(release), "rev-parse", "HEAD") != sha:
        raise RuntimeError("smoke source is not the named release")
    if _command("git", "-c", "gc.auto=0", "-C", str(release), "status",
                "--porcelain", "--untracked-files=all"):
        raise RuntimeError("smoke release checkout is dirty")
    tracked = _command("git", "-c", "gc.auto=0", "-C", str(release),
                       "ls-files", "--error-unmatch",
                       "tools/fmow_local_boundary_vit_smoke.py")
    if tracked != "tools/fmow_local_boundary_vit_smoke.py":
        raise RuntimeError("smoke generator is not tracked in release")


def validate_success_receipt(receipt):
    """Check that the measured workload, not merely a declarative flag, passed."""
    if (receipt.get("memory_smoke_passed") is not True or
            receipt.get("label_free") is not True or
            receipt.get("precision") != "fp32" or
            receipt.get("backbone") != "vit_b_16" or
            receipt.get("batch_size") != 16 or
            receipt.get("development_batch_size") != 8 or
            receipt.get("weight_sha256") != WEIGHT_SHA or
            receipt.get("development_pool_count") != 1673):
        raise ValueError("smoke identity or status differs from the fixed ViT study")
    total = receipt.get("total_memory_bytes")
    peak = receipt.get("peak_allocated_bytes")
    if type(total) is not int or type(peak) is not int or not 0 < peak < total:
        raise ValueError("invalid device memory measurements")
    phases = receipt.get("phases")
    if not isinstance(phases, dict) or set(phases) != set(PHASES):
        raise ValueError("missing measured smoke phase")
    for name in PHASES:
        phase = phases[name]
        if (not isinstance(phase, dict) or phase.get("completed") is not True or
                not isinstance(phase.get("seconds"), (int, float)) or
                not math.isfinite(phase["seconds"]) or phase["seconds"] <= 0 or
                type(phase.get("peak_allocated_bytes")) is not int or
                not 0 < phase["peak_allocated_bytes"] <= peak):
            raise ValueError("incomplete or invalid measured phase: " + name)
    train = phases["train_backward"]
    if (train.get("input_shape") != [16, 3, 224, 224] or
            train.get("optimizer_step") is not True or
            not _positive_finite(train.get("loss")) or
            not _positive_finite(train.get("gradient_norm"))):
        raise ValueError("training backward/optimizer smoke failed")
    dev = phases["development_inference"]
    if (dev.get("probabilities_shape") != [1673, 8] or
            dev.get("finite_rows") != 1673 or
            not isinstance(dev.get("row_sum_max_error"), (int, float)) or
            not 0 <= dev["row_sum_max_error"] < 1e-4):
        raise ValueError("development inference smoke failed")
    side = phases["side_copy_constraint_gradient"]
    if side.get("pto_unchanged") is not True or set(side.get("caps", {})) != {"10", "20"}:
        raise ValueError("side-copy smoke missed PTO neutrality or a cap")
    for cap in side["caps"].values():
        if (not _positive_finite(cap.get("joint_gradient_norm")) or
                not _positive_finite(cap.get("phr_gradient_norm")) or
                type(cap.get("joint_applied")) is not bool or
                type(cap.get("phr_applied")) is not bool or
                cap.get("all_four_arms") is not True or
                cap.get("scope_derivatives_finite") is not True or
                cap.get("pto_unchanged") is not True):
            raise ValueError("side-copy gradient smoke failed")
    return True


def _positive_finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _measure(torch, func):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    detail = func()
    torch.cuda.synchronize()
    return dict(completed=True, seconds=time.monotonic() - started,
                peak_allocated_bytes=int(torch.cuda.max_memory_allocated()), **detail)


def _workload(receipt, data_root, artifact_root):
    # CUDA_VISIBLE_DEVICES is set in main before importing these modules.
    import torch
    from tralo.fmow_local import (VIT_RECIPE, budgets, snapshot_side_steps,
                                  vit_weight_provenance)
    from tralo.fmow_yuval import CLASSES, FILES, load, make_model, pool_chunks, transforms_for
    from tralo.knee_end_to_end import infer
    from tralo.knee_experiment import cuda_setup, digest, source
    from tralo.global_comparison import _state_hash

    torch.set_num_threads(8)
    torch.manual_seed(6500)
    cuda_setup()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exclusive CUDA device unavailable")
    receipt["device_name"] = torch.cuda.get_device_name(0)
    receipt["total_memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    receipt["source_sha256"] = source()
    receipt["smoke_generator_sha256"] = digest(Path(__file__))
    receipt["data_files"] = FILES
    if vit_weight_provenance()["sha256"] != WEIGHT_SHA:
        raise RuntimeError("ViT checkpoint hash mismatch")
    images, _train_labels, rows, roles = load(data_root, include_pool_labels=False)
    groups = [row["location"] for row in rows]
    quotas = budgets(groups)
    receipt["development_pool_count"] = len(groups)
    if len(roles["dev"]) != 1673 or any("label" in row for row in rows):
        raise RuntimeError("development role or label boundary changed")
    _train_tf, eval_tf = transforms_for()
    pool = pool_chunks(images["test"], roles["dev"], eval_tf,
                       VIT_RECIPE["development_batch_size"])
    if (sum(len(chunk) for chunk in pool) != 1673 or
            max(len(chunk) for chunk in pool) != 8 or
            any(tuple(chunk.shape[1:]) != (3, 224, 224) for chunk in pool)):
        raise RuntimeError("development transform or chunk shape changed")

    model = make_model(backbone="vit_b_16").cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=VIT_RECIPE["lr"],
                                 weight_decay=VIT_RECIPE["weight_decay"])
    phases = receipt["phases"]

    def train_backward():
        model.train()
        x = torch.zeros((16, 3, 224, 224), device="cuda", dtype=torch.float32)
        logits = model(x)
        if tuple(logits.shape) != (16, CLASSES):
            raise RuntimeError("ViT eight-class head mismatch")
        loss = logits.square().mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite synthetic ViT training loss")
        loss.backward()
        grad_norm = math.sqrt(sum(float(p.grad.double().square().sum())
                                  for p in model.parameters() if p.grad is not None))
        if not _positive_finite(grad_norm):
            raise RuntimeError("invalid ViT training gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return dict(input_shape=list(x.shape), loss=float(loss),
                    gradient_norm=grad_norm, optimizer_step=True)

    phases["train_backward"] = _measure(torch, train_backward)
    model.eval()

    def dev_inference():
        probabilities = infer(model, pool)
        if tuple(probabilities.shape) != (1673, 8) or not bool(torch.isfinite(probabilities).all()):
            raise RuntimeError("invalid ViT development inference")
        row_error = float((probabilities.sum(dim=1) - 1).abs().max())
        if row_error >= 1e-4:
            raise RuntimeError("ViT development probabilities do not sum to one")
        return dict(probabilities_shape=list(probabilities.shape), finite_rows=1673,
                    row_sum_max_error=row_error, probabilities=probabilities)

    dev_phase = _measure(torch, dev_inference)
    probabilities = dev_phase.pop("probabilities")
    phases["development_inference"] = dev_phase
    model_sha = _state_hash(model)
    artifact_root.mkdir(exist_ok=False)

    def side_copy():
        cap_records = {}
        for divisor in (10, 20):
            folder = artifact_root / f"cap{divisor}"
            folder.mkdir()
            records = snapshot_side_steps(
                model, pool, groups, quotas[str(divisor)], 6500, 1, folder,
                fixed_radius=0.1, phr_state={"dual": torch.zeros(6)},
                phr_rho=0.5, boundary_calibrated=True,
                pto_probabilities=probabilities)
            joint, phr = records["joint"], records["phr_local"]
            derivatives = (joint.get("scope_directional_derivatives", {}),
                           phr.get("scope_directional_derivatives", {}))
            all_finite = all(math.isfinite(float(value)) for scope in derivatives
                             for value in scope.values())
            unchanged = _state_hash(model) == model_sha
            cap_records[str(divisor)] = dict(
                joint_gradient_norm=joint.get("gradient_norm"),
                phr_gradient_norm=phr.get("gradient_norm"),
                joint_applied=joint["applied"], phr_applied=phr["applied"],
                all_four_arms=set(records) == {"joint", "global_dose", "sham", "phr_local"},
                scope_derivatives_finite=all_finite,
                pto_unchanged=unchanged)
            if not (unchanged and all_finite and
                    _positive_finite(joint.get("gradient_norm")) and
                    _positive_finite(phr.get("gradient_norm"))):
                raise RuntimeError(f"ViT side-copy gradient/neutrality failed at cap{divisor}")
        return dict(pto_unchanged=_state_hash(model) == model_sha, caps=cap_records)

    phases["side_copy_constraint_gradient"] = _measure(torch, side_copy)
    receipt["peak_allocated_bytes"] = max(
        phase["peak_allocated_bytes"] for phase in phases.values())
    if receipt["peak_allocated_bytes"] >= receipt["total_memory_bytes"] * 0.9:
        raise RuntimeError("ViT smoke leaves less than 10% device-memory headroom")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 4:
        raise SystemExit(__doc__)
    import fcntl
    sha, data_root, gpu_index, receipt_path = argv
    if len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha):
        raise ValueError("invalid release SHA")
    if not gpu_index.isdecimal():
        raise ValueError("invalid physical GPU index")
    release = Path(__file__).resolve().parents[1]
    _verify_release(release, sha)
    output = Path(receipt_path).resolve()
    runs = RUNS.resolve(strict=True)
    if not output.is_relative_to(runs) or output == runs or not output.parent.is_dir():
        raise ValueError("smoke receipt must be a new file under the runs directory")
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite a smoke receipt")
    artifact_root = output.with_suffix(".artifacts")
    if artifact_root.exists() or artifact_root.is_symlink():
        raise FileExistsError("refusing to overwrite smoke artifacts")

    uuid = _physical_uuid(gpu_index)
    locks = runs / ".fmow-local-boundary-gpu-locks"
    locks.mkdir(exist_ok=True)
    lock_path = locks / f"{uuid}.lock"
    inherited_fd = os.environ.get("FMOW_VIT_INHERITED_LOCK_FD")
    if inherited_fd is None:
        lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        own_lock_fd = True
    else:
        if not inherited_fd.isdecimal() or not os.path.samefile(
                f"/proc/self/fd/{inherited_fd}", lock_path):
            raise RuntimeError("queue lock FD does not match selected physical GPU")
        lock_fd = int(inherited_fd)
        own_lock_fd = False
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _assert_gpu_free(gpu_index, uuid)
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        receipt = dict(release_commit=sha, host=_command("hostname", "-f"),
                       gpu_uuid=uuid, gpu_index=int(gpu_index), backbone="vit_b_16",
                       batch_size=16, development_batch_size=8,
                       weight_sha256=WEIGHT_SHA, precision="fp32", label_free=True,
                       memory_smoke_passed=False, phases={},
                       started_utc=time.time())
        try:
            _workload(receipt, data_root, artifact_root)
            receipt["memory_smoke_passed"] = True
            validate_success_receipt(receipt)
        except Exception as exc:
            receipt["failure_type"] = type(exc).__name__
            receipt["failure_message"] = str(exc)
            raise
        finally:
            receipt["ended_utc"] = time.time()
            with output.open("x", encoding="utf-8") as stream:
                json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
    finally:
        if own_lock_fd:
            os.close(lock_fd)


if __name__ == "__main__":
    main()
