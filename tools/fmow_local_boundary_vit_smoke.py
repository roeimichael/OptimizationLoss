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
COUNTRIES = ("DZA", "IRQ", "NLD", "PHL", "TUR")
FIXTURE_NAME = "fixed_sine_head_class1_bias1_v1"


def _install_diagnostic_head(head):
    """Make only this smoke model's head exercise active, differentiable scopes."""
    import torch
    with torch.no_grad():
        rows = torch.arange(1, head.out_features + 1, device=head.weight.device)
        columns = torch.arange(1, head.in_features + 1, device=head.weight.device)
        values = 1e-4 * torch.sin(0.017 * rows[:, None] * columns[None, :])
        head.weight.copy_(values.to(head.weight.dtype))
        head.bias.zero_()
        head.bias[1] = 1.0


def _scope_diagnostics(probabilities, groups, quota):
    """Record class-1 counts without development labels or model selection."""
    if len(probabilities) != len(groups) or set(groups) != set(quota["local_caps"]):
        raise ValueError("smoke scope cohort and quota differ")
    selected = probabilities.argmax(1) == 1
    class_one = probabilities[:, 1]
    result = {"pooled": dict(hard=int(selected.sum()), soft=float(class_one.sum()),
                             cap=quota["global_cap"]), "countries": {}}
    for name in sorted(quota["local_caps"]):
        indices = [i for i, group in enumerate(groups) if group == name]
        result["countries"][name] = dict(
            hard=int(selected[indices].sum()), soft=float(class_one[indices].sum()),
            cap=quota["local_caps"][name])
    return result


def _assert_active_scopes(scopes):
    entries = {"pooled": scopes["pooled"]}
    entries.update({f"country:{name}": value
                    for name, value in scopes["countries"].items()})
    inactive = [name for name, value in entries.items()
                if value["hard"] <= value["cap"] or value["soft"] <= value["cap"]]
    if inactive:
        raise RuntimeError("inactive diagnostic scope: " + ", ".join(inactive))
    return True


def _valid_active_scopes(scopes):
    if not isinstance(scopes, dict) or set(scopes) != {"pooled", "countries"}:
        return False
    if not isinstance(scopes["countries"], dict) or set(scopes["countries"]) != set(COUNTRIES):
        return False
    records = [scopes["pooled"], *scopes["countries"].values()]
    if any(not isinstance(record, dict) or set(record) != {"hard", "soft", "cap"} or
           type(record["hard"]) is not int or type(record["cap"]) is not int or
           not isinstance(record["soft"], (int, float)) or
           not math.isfinite(record["soft"]) or record["hard"] <= record["cap"] or
           record["soft"] <= record["cap"] for record in records):
        return False
    pooled = scopes["pooled"]
    return (pooled["hard"] == sum(record["hard"] for record in scopes["countries"].values())
            and math.isclose(pooled["soft"],
                             math.fsum(record["soft"] for record in scopes["countries"].values()),
                             rel_tol=1e-5, abs_tol=1e-5))


def _pooled_gradient_diagnostic(model, head, pool):
    """Measure pooled class-1 gradient, including non-head parameters."""
    model.zero_grad(set_to_none=True)
    head_ids = {id(parameter) for parameter in head.parameters()}
    device = next(model.parameters()).device
    try:
        for images in pool:
            model(images.to(device)).softmax(1)[:, 1].sum().backward()
        total = 0.0
        backbone = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                squared = float(parameter.grad.double().square().sum())
                total += squared
                if id(parameter) not in head_ids:
                    backbone += squared
        result = dict(total_norm=math.sqrt(total), backbone_norm=math.sqrt(backbone))
        if not all(_positive_finite(value) for value in result.values()):
            raise RuntimeError("diagnostic pooled or backbone gradient is zero or nonfinite")
        return result
    finally:
        model.zero_grad(set_to_none=True)


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
            receipt.get("mha_fastpath_enabled") is not False or
            receipt.get("precision") != "fp32" or
            receipt.get("backbone") != "vit_b_16" or
            receipt.get("batch_size") != 16 or
            receipt.get("development_batch_size") != 8 or
            receipt.get("weight_sha256") != WEIGHT_SHA or
            receipt.get("development_pool_count") != 1673):
        raise ValueError("smoke identity or status differs from the fixed ViT study")
    if not _valid_attention_replay(receipt.get("ordinary_head_replay")):
        raise ValueError("ordinary ViT attention replay failed")
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
    if (side.get("pto_unchanged") is not True or side.get("fixture") != FIXTURE_NAME or
            set(side.get("caps", {})) != {"10", "20"}):
        raise ValueError("side-copy smoke missed PTO neutrality or a cap")
    for cap in side["caps"].values():
        if (not _positive_finite(cap.get("joint_gradient_norm")) or
                not _positive_finite(cap.get("phr_gradient_norm")) or
                cap.get("joint_applied") is not True or
                cap.get("phr_applied") is not True or
                cap.get("all_four_arms") is not True or
                cap.get("scope_derivatives_finite") is not True or
                cap.get("pto_unchanged") is not True or
                not _valid_active_scopes(cap.get("scopes"))):
            raise ValueError("side-copy gradient smoke failed")
    precheck = side.get("pooled_gradient_precheck")
    if (not isinstance(precheck, dict) or
            not _positive_finite(precheck.get("total_norm")) or
            not _positive_finite(precheck.get("backbone_norm")) or
            precheck.get("pto_unchanged") is not True):
        raise ValueError("diagnostic backbone gradient smoke failed")
    return True


def _positive_finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _valid_attention_replay(value):
    return (isinstance(value, dict) and
            set(value) == {"passed", "images_count", "max_absolute_difference",
                           "max_tolerance_ratio"} and
            value["passed"] is True and value["images_count"] == 8 and
            type(value["max_absolute_difference"]) in (int, float) and
            type(value["max_tolerance_ratio"]) in (int, float) and
            math.isfinite(value["max_absolute_difference"]) and
            math.isfinite(value["max_tolerance_ratio"]) and
            0 <= value["max_absolute_difference"] <= 1.1e-6 and
            0 <= value["max_tolerance_ratio"] <= 1)


def _measure(torch, func):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    detail = func()
    torch.cuda.synchronize()
    result = dict(detail)
    result.update(completed=True, seconds=time.monotonic() - started,
                  peak_allocated_bytes=int(torch.cuda.max_memory_allocated()))
    return result


def _workload(receipt, data_root, artifact_root):
    # CUDA_VISIBLE_DEVICES is set in main before importing these modules.
    import torch
    from tralo.fmow_local import (VIT_RECIPE, budgets, disable_vit_mha_fastpath,
                                  snapshot_side_steps, vit_attention_replay,
                                  vit_weight_provenance)
    from tralo.fmow_yuval import CLASSES, FILES, load, make_model, pool_chunks, transforms_for
    from tralo.knee_end_to_end import infer
    from tralo.knee_experiment import cuda_setup, digest, source
    from tralo.global_comparison import _state_hash

    torch.set_num_threads(8)
    torch.manual_seed(6600)
    cuda_setup()
    receipt["mha_fastpath_enabled"] = disable_vit_mha_fastpath()
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
    dev_phase.pop("probabilities")
    phases["development_inference"] = dev_phase
    receipt["ordinary_head_replay"] = vit_attention_replay(model, pool[0])
    if not receipt["ordinary_head_replay"]["passed"]:
        raise RuntimeError("ordinary ViT head no-grad/grad attention replay differs")
    # Diagnostic fixture is applied only after the real 1,673-image inference
    # memory phase. It is never a study seed's training head or scored model.
    _install_diagnostic_head(model.heads.head)
    model_sha = _state_hash(model)
    artifact_root.mkdir(exist_ok=False)
    side_evidence = dict(completed=False, fixture=FIXTURE_NAME, pto_unchanged=True,
                         diagnostic_model_sha256=model_sha, caps={})
    phases["side_copy_constraint_gradient"] = side_evidence

    def side_copy():
        diagnostic_probabilities = infer(model, pool)
        if not bool(torch.isfinite(diagnostic_probabilities).all()):
            raise RuntimeError("nonfinite diagnostic ViT probabilities")
        side_evidence["pooled_gradient_precheck"] = dict(stage="started")
        try:
            precheck = _pooled_gradient_diagnostic(model, model.heads.head, pool)
            precheck["pto_unchanged"] = _state_hash(model) == model_sha
            side_evidence["pooled_gradient_precheck"] = precheck
            if not precheck["pto_unchanged"]:
                raise RuntimeError("pooled gradient precheck changed PTO weights")
        except Exception as exc:
            side_evidence["pooled_gradient_precheck"]["error"] = str(exc)
            raise
        for divisor in (10, 20):
            folder = artifact_root / f"cap{divisor}"
            folder.mkdir()
            record = {"scopes": _scope_diagnostics(
                diagnostic_probabilities, groups, quotas[str(divisor)]),
                "pto_unchanged": _state_hash(model) == model_sha}
            side_evidence["caps"][str(divisor)] = record
            _assert_active_scopes(record["scopes"])
            try:
                records = snapshot_side_steps(
                    model, pool, groups, quotas[str(divisor)], 6600, 1, folder,
                    fixed_radius=0.1, phr_state={"dual": torch.zeros(6)},
                    phr_rho=0.5, boundary_calibrated=True,
                    pto_probabilities=diagnostic_probabilities)
            except Exception as exc:
                record["pto_unchanged"] = _state_hash(model) == model_sha
                record["side_copy_error"] = str(exc)
                side_evidence["pto_unchanged"] = record["pto_unchanged"]
                raise
            joint, phr = records["joint"], records["phr_local"]
            derivatives = (joint.get("scope_directional_derivatives", {}),
                           phr.get("scope_directional_derivatives", {}))
            all_finite = all(math.isfinite(float(value)) for scope in derivatives
                             for value in scope.values())
            unchanged = _state_hash(model) == model_sha
            record.update(
                joint_gradient_norm=joint.get("gradient_norm"),
                phr_gradient_norm=phr.get("gradient_norm"),
                joint_applied=joint["applied"], phr_applied=phr["applied"],
                all_four_arms=set(records) == {"joint", "global_dose", "sham", "phr_local"},
                scope_derivatives_finite=all_finite,
                pto_unchanged=unchanged)
            side_evidence["pto_unchanged"] = unchanged
            if not (unchanged and all_finite and
                    _positive_finite(joint.get("gradient_norm")) and
                    _positive_finite(phr.get("gradient_norm")) and
                    joint["applied"] and phr["applied"]):
                raise RuntimeError(f"ViT side-copy gradient/neutrality failed at cap{divisor}")
        side_evidence["pto_unchanged"] = _state_hash(model) == model_sha
        if not side_evidence["pto_unchanged"]:
            raise RuntimeError("ViT smoke changed PTO diagnostic weights")
        return side_evidence

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
