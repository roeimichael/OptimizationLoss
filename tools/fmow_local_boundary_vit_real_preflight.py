"""Label-free real-image ViT numerical preflight on an exclusive physical GPU.

Usage: python -m tools.fmow_local_boundary_vit_real_preflight RELEASE_SHA DATA_ROOT GPU_INDEX NEW_RECEIPT_JSON
Run from the immutable release. The queue owns the physical-card lock and
retains both successful and failed receipts under its new, exclusive run root.
"""

import json
import math
import os
from pathlib import Path
import sys
import time

from tools import fmow_local_boundary_vit_smoke as guard

COUNTRIES = ("DZA", "IRQ", "NLD", "PHL", "TUR")
SCOPES = ("pooled", *COUNTRIES)
EPSILONS = (0.01, 0.02)
ARMS = ("joint", "global_dose", "sham", "phr_local")
WEIGHT_SHA = guard.WEIGHT_SHA


def validate_success_receipt(receipt):
    """Reject missing or nonfinite measured numerical checks."""
    if (receipt.get("preflight_passed") is not True or
            receipt.get("development_labels_accessed") is not False or
            receipt.get("precision") != "fp32" or
            receipt.get("backbone") != "vit_b_16" or
            receipt.get("weight_sha256") != WEIGHT_SHA or
            receipt.get("images_count") != 15 or
            receipt.get("development_batch_size") != 8 or
            receipt.get("chunk_sizes") != [8, 7] or
            receipt.get("country_counts") != dict.fromkeys(COUNTRIES, 3) or
            receipt.get("pto_unchanged") is not True or
            receipt.get("arms_audited") != list(ARMS)):
        raise ValueError("real-image preflight identity, cohort or audit differs")
    if not _finite_below(receipt.get("max_probability_difference"), 1e-6):
        raise ValueError("full/chunk probability parity failed")
    errors = receipt.get("gradient_relative_errors")
    if not isinstance(errors, dict) or set(errors) != set(SCOPES):
        raise ValueError("missing full/chunk gradient scope")
    if any(not _finite_below(value, 0.01) for value in errors.values()):
        raise ValueError("full/chunk gradient parity failed")
    differences = receipt.get("finite_differences")
    if not isinstance(differences, dict) or set(differences) != {
            f"{scope}@{epsilon}" for scope in SCOPES for epsilon in EPSILONS}:
        raise ValueError("missing finite-difference scope or step size")
    for row in differences.values():
        if (not isinstance(row, dict) or
                set(row) != {"analytic", "numeric", "error", "tolerance"} or
                any(not _finite(value) for value in row.values())):
            raise ValueError("nonfinite finite-difference record")
        analytic, numeric = row.get("analytic"), row.get("numeric")
        error = abs(analytic - numeric)
        tolerance = 0.05 * max(abs(analytic), abs(numeric)) + 0.003
        if (not math.isclose(row.get("error"), error, abs_tol=1e-8) or
                not math.isclose(row.get("tolerance"), tolerance, abs_tol=1e-8) or
                error > tolerance):
            raise ValueError("finite-difference derivative failed")
    artifacts = receipt.get("artifact_sha256")
    if not isinstance(artifacts, dict) or set(artifacts) != {
            f"epoch01_{arm}.pt" for arm in ARMS} or any(
                not isinstance(value, str) or len(value) != 64 for value in artifacts.values()):
        raise ValueError("four-arm artifact inventory differs")
    return True


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _finite_below(value, limit):
    return _finite(value) and 0 <= value <= limit


def _gradients(torch, model, batches, groups):
    params = [p for p in model.parameters() if p.requires_grad]
    out = {name: [torch.zeros_like(p) for p in params] for name in SCOPES}
    offset = 0
    for images in batches:
        logits = model(images.cuda())
        q = logits.softmax(1)[:, 1]
        batch_groups = groups[offset:offset + len(images)]
        for index, name in enumerate(SCOPES):
            if name == "pooled":
                value = q.sum() / 3
            else:
                mask = torch.tensor([group == name for group in batch_groups], device=q.device)
                value = q[mask].sum()
            row = torch.autograd.grad(value, params, retain_graph=index < len(SCOPES) - 1,
                                      allow_unused=True)
            for target, gradient in zip(out[name], row):
                if gradient is not None:
                    target.add_(gradient.detach())
        offset += len(images)
    if offset != len(groups):
        raise RuntimeError("gradient cohort length changed")
    return out, params


def _scalar(torch, probabilities, groups, name):
    if name == "pooled":
        return float(probabilities[:, 1].sum()) / 3
    mask = torch.tensor([group == name for group in groups])
    return float(probabilities[mask, 1].sum())


def _workload(receipt, data_root, artifact_root):
    import torch
    from analysis.score_fmow_boundary import _audit_side
    from tralo.fmow_local import VIT_RECIPE, snapshot_side_steps, vit_weight_provenance
    from tralo.fmow_yuval import FILES, load, make_model, pool_chunks, transforms_for
    from tralo.global_comparison import _state_hash
    from tralo.knee_end_to_end import infer
    from tralo.knee_experiment import cuda_setup, digest, source
    from tralo.targeted_step import _place

    torch.set_num_threads(8)
    torch.manual_seed(6500)
    cuda_setup()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exclusive CUDA device unavailable")
    receipt["device_name"] = torch.cuda.get_device_name(0)
    receipt["source_sha256"] = source()
    receipt["preflight_generator_sha256"] = digest(Path(__file__))
    # FILES maps each exact name to its preregistered SHA-256. load() independently
    # recomputes every file digest, including label files without reading dev labels.
    receipt["data_file_sha256"] = FILES
    receipt["pretrained_weight"] = vit_weight_provenance()
    images, _train_labels, rows, roles = load(data_root, include_pool_labels=False)
    if any("label" in row for row in rows) or len(rows) != 1673:
        raise RuntimeError("development label or pool boundary changed")
    chosen, groups = [], []
    for country in COUNTRIES:
        positions = [i for i, row in enumerate(rows) if row["location"] == country][:3]
        if len(positions) != 3:
            raise RuntimeError("real-image country cohort changed")
        chosen.extend(roles["dev"][i] for i in positions)
        groups.extend([country] * 3)
    _, eval_transform = transforms_for()
    pool = pool_chunks(images["test"], chosen, eval_transform,
                       VIT_RECIPE["development_batch_size"])
    receipt["chunk_sizes"] = [len(chunk) for chunk in pool]
    if receipt["chunk_sizes"] != [8, 7] or any(
            tuple(chunk.shape[1:]) != (3, 224, 224) for chunk in pool):
        raise RuntimeError("real-image chunk transform changed")
    model = make_model(backbone="vit_b_16").cuda().eval()
    # A fixed diagnostic head activates all six constraints. It is never saved
    # or used in training, model selection, or development-label evaluation.
    with torch.no_grad():
        head = model.heads.head
        head.weight.normal_(0, 0.0001)
        head.bias.zero_()
        head.bias[1] = 1.0
    pto_hash = _state_hash(model)
    full = infer(model, [torch.cat(pool)])
    chunked = infer(model, pool)
    receipt["max_probability_difference"] = float((full - chunked).abs().max())
    if not torch.allclose(full, chunked, atol=1e-6, rtol=1e-6):
        raise RuntimeError("full/chunked real-image probabilities differ")
    caps = dict.fromkeys(COUNTRIES, 1)
    quota = dict(global_cap=3, local_caps=caps)
    prediction = full.argmax(1) == 1
    if int(prediction.sum()) <= 3 or any(int(prediction[i * 3:(i + 1) * 3].sum()) <= 1
                                        for i in range(5)):
        raise RuntimeError("diagnostic head failed to activate all six constraints")

    full_grads, params = _gradients(torch, model, [torch.cat(pool)], groups)
    chunk_grads, _ = _gradients(torch, model, pool, groups)
    parity = {}
    for name in SCOPES:
        numerator = math.sqrt(sum(float((a - b).double().square().sum())
                                  for a, b in zip(full_grads[name], chunk_grads[name])))
        denominator = max(1e-9, math.sqrt(sum(float(a.double().square().sum())
                                            for a in full_grads[name])))
        parity[name] = numerator / denominator
        if not _finite_below(parity[name], 0.01):
            raise RuntimeError(f"full/chunk gradient mismatch for {name}: {parity[name]}")
    receipt["gradient_relative_errors"] = parity
    joint = [sum(full_grads[name][i] for name in SCOPES) for i in range(len(params))]
    norm = math.sqrt(sum(float(g.double().square().sum()) for g in joint))
    if not _finite(norm) or norm <= 0:
        raise RuntimeError("nonfinite or zero constraint gradient")
    direction = [-gradient / norm for gradient in joint]
    origin = [p.detach().clone() for p in params]
    differences = {}
    try:
        for epsilon in EPSILONS:
            _place(params, origin, direction, epsilon)
            plus = infer(model, pool)
            _place(params, origin, direction, -epsilon)
            minus = infer(model, pool)
            for name in SCOPES:
                analytic = sum(float((gradient.double() * vector.double()).sum())
                               for gradient, vector in zip(full_grads[name], direction))
                numeric = (_scalar(torch, plus, groups, name) -
                           _scalar(torch, minus, groups, name)) / (2 * epsilon)
                error = abs(analytic - numeric)
                tolerance = 0.05 * max(abs(analytic), abs(numeric)) + 0.003
                differences[f"{name}@{epsilon}"] = dict(analytic=analytic, numeric=numeric,
                                                         error=error, tolerance=tolerance)
                if not _finite(numeric) or error > tolerance:
                    raise RuntimeError(f"finite difference mismatch {name}@{epsilon}")
    finally:
        _place(params, origin, direction, 0.0)
    receipt["finite_differences"] = differences
    if not torch.equal(infer(model, pool), chunked):
        raise RuntimeError("finite-difference restore changed PTO predictions")
    artifact_root.mkdir(exist_ok=False)
    records = snapshot_side_steps(model, pool, groups, quota, 6500, 1, artifact_root,
                                  fixed_radius=0.1, phr_state={"dual": torch.zeros(6)},
                                  phr_rho=0.5, boundary_calibrated=True,
                                  pto_probabilities=chunked)
    dual = [0.0] * 6
    for arm in ARMS:
        side = torch.load(artifact_root / f"epoch01_{arm}.pt", weights_only=True)
        dual = _audit_side(records[arm], chunked, side, groups, quota, arm, dual)
    receipt["pto_unchanged"] = _state_hash(model) == pto_hash
    receipt["arms_audited"] = list(ARMS)
    receipt["artifact_sha256"] = {p.name: digest(p) for p in artifact_root.glob("*.pt")}
    receipt["joint_record"] = records["joint"]
    receipt["phr_record"] = records["phr_local"]
    if not receipt["pto_unchanged"]:
        raise RuntimeError("real-image side steps changed PTO state")


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
    guard._verify_release(release, sha)
    tracked = guard._command("git", "-c", "gc.auto=0", "-C", str(release),
                             "ls-files", "--error-unmatch",
                             "tools/fmow_local_boundary_vit_real_preflight.py")
    if tracked != "tools/fmow_local_boundary_vit_real_preflight.py":
        raise RuntimeError("real-image preflight generator is not tracked in release")
    output = Path(receipt_path).resolve()
    runs = guard.RUNS.resolve(strict=True)
    if not output.is_relative_to(runs) or output == runs or not output.parent.is_dir():
        raise ValueError("real-image receipt must be under owned runs")
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite real-image receipt")
    artifacts = output.with_suffix(".artifacts")
    if artifacts.exists() or artifacts.is_symlink():
        raise FileExistsError("refusing to overwrite real-image artifacts")
    uuid = guard._physical_uuid(gpu_index)
    locks = runs / ".fmow-local-boundary-gpu-locks"
    locks.mkdir(exist_ok=True)
    lock_path = locks / f"{uuid}.lock"
    inherited_fd = os.environ.get("FMOW_VIT_INHERITED_LOCK_FD")
    if inherited_fd is None:
        lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        own_lock = True
    else:
        if not inherited_fd.isdecimal() or not os.path.samefile(
                f"/proc/self/fd/{inherited_fd}", lock_path):
            raise RuntimeError("queue lock FD differs from selected GPU")
        lock_fd, own_lock = int(inherited_fd), False
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard._assert_gpu_free(gpu_index, uuid)
        os.environ["CUDA_VISIBLE_DEVICES"] = uuid
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        receipt = dict(release_commit=sha, host=guard._command("hostname", "-f"),
                       gpu_uuid=uuid, gpu_index=int(gpu_index), precision="fp32",
                       backbone="vit_b_16", weight_sha256=WEIGHT_SHA,
                       development_batch_size=8,
                       development_labels_accessed=False, preflight_passed=False,
                       images_count=15, country_counts=dict.fromkeys(COUNTRIES, 3),
                       started_utc=time.time())
        try:
            _workload(receipt, data_root, artifacts)
            receipt["preflight_passed"] = True
            validate_success_receipt(receipt)
        except Exception as exc:
            receipt.update(failure_type=type(exc).__name__, failure_message=str(exc))
            raise
        finally:
            receipt["ended_utc"] = time.time()
            with output.open("x", encoding="utf-8") as stream:
                json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
    finally:
        if own_lock:
            os.close(lock_fd)


if __name__ == "__main__":
    main()
