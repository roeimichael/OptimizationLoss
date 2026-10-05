"""Fictitious RGB input-to-ensemble rehearsal, plus a CPU MobileNet task gradient.

Usage: python tools/knee_snapshot_cpu_readiness.py EXCLUSIVE_OUTPUT_DIRECTORY
This is deliberately NOT a campaign CLI: no real data, pretrained checkpoint,
scientific seed claim or CUDA device is used. Full campaign gates remain open.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(output):
    import builtins
    import copy
    import math
    try:
        import resource
    except ImportError:
        resource = None
    import socket
    from PIL import Image
    import torch

    from tralo.global_clipper import allocate, allocate_local_capped_first
    from tralo.global_comparison import _state_hash
    from tralo.knee_e2e_v3 import build_model
    from tralo.knee_end_to_end import development_images, infer
    from tralo.knee_snapshot_data import load_public, prepare, sha256
    from tralo.knee_snapshot_local import ARMS, RECIPE, average_snapshots, snapshot
    from tralo.knee_yuval import Images, train_run, transforms_for

    torch.set_num_threads(1)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source, public = output / "fictitious_source", output / "public"
    private = output / "private" / "development_labels.json"
    for split, count, offset in (("train", 40, 1000000), ("val", 100, 2000000)):
        for i in range(count):
            path = source / split / str(i % 5) / f"{offset+i:07d}L.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (7, 9), (i, offset//1000000, i//2)).save(path)
    (source / "test").mkdir()
    (source / "test" / "SEALED_FICTITIOUS_SENTINEL").write_text("must never be read")
    preparation = prepare(source, public, private, {"train": 40, "development": 100})
    original_open = builtins.open
    original_path_open = Path.open
    access = []

    def allowed(path):
        if isinstance(path, (str, bytes, os.PathLike)):
            candidate = Path(path).resolve()
            if candidate == private or source in candidate.parents:
                raise RuntimeError("private/original source accessed during public rehearsal")
            access.append(str(candidate))

    def guarded_open(path, *args, **kwargs):
        allowed(path)
        return original_open(path, *args, **kwargs)

    def guarded_path_open(path, *args, **kwargs):
        allowed(path)
        return original_path_open(path, *args, **kwargs)

    builtins.open, Path.open = guarded_open, guarded_path_open
    try:
        manifest, rows = load_public(public, preparation["public_manifest_sha256"], allow_synthetic=True)
        train_tf, eval_tf = transforms_for()
        training, stopping = Images(public, rows["train"]), Images(public, rows["stop"])
        stop = [stopping.batch(list(range(len(stopping.labels))), eval_tf)]
        pool = development_images(public, rows["development"], eval_tf, 16)
        groups = [r["group"] for r in rows["development"]]
        ids = [r["sample_id"] for r in rows["development"]]
        quota = {"global_cap": 76, "local_caps": manifest["local_caps"]}
        torch.manual_seed(31)  # fictitious initialization, not a scientific seed
        initial = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 5))
        with torch.no_grad():
            initial[-1].weight.zero_()
            initial[-1].bias.zero_()
            initial[-1].bias[0] = 1.
        fits, snapshots = [], {}
        # Explicit simulation horizon; it never enters the scientific recipe.
        config = dict(RECIPE, seed=31, max_epochs=3, patience=3)
        for corrected in (False, True):
            model = copy.deepcopy(initial)
            events, epoch_hashes = [], {}
            def save_epoch(epoch, probabilities):
                epoch_hashes[str(epoch)] = _state_hash(model)
                if corrected:
                    snapshots[str(epoch)] = snapshot(model, pool, groups, quota, 31, epoch,
                                                    output / "snapshots" / f"epoch{epoch:02d}",
                                                    probabilities, lambda event: None)
            result = train_run(model, training, stop, pool, config, torch.ones(5),
                               events.append, save_epoch)
            fits.append(dict(result=result, events=events, epoch_hashes=epoch_hashes,
                             restored_model_sha256=_state_hash(model)))
        if fits[0] != fits[1]:
            raise RuntimeError("CPU prepared-image PTO trajectory parity failed")
        window, averages = average_snapshots(output / "snapshots", snapshots,
                                             fits[1]["result"]["best_epoch"], fits[1]["result"]["epochs_run"])
        allocations = {}
        for arm in ARMS:
            values = averages[arm].tolist()
            caps = [None, None, None, 76, None]
            predictions = dict(global_only=allocate(values, caps, ids, "capped_first"),
                               global_local=allocate_local_capped_first(values, caps, ids, groups, quota["local_caps"]))
            for policy, calls in predictions.items():
                if sum(c == 3 for c in calls) > 76:
                    raise RuntimeError("pooled allocation exceeds cap")
                if policy == "global_local" and any(sum(c == 3 for c, g in zip(calls, groups) if g == group) > cap
                                                    for group, cap in quota["local_caps"].items()):
                    raise RuntimeError("local allocation exceeds cap")
            allocations[arm] = predictions
        # One full unfrozen architecture task backward, on fictitious RGB pixels.
        # No checkpoint download: this does not certify the pretrained recipe.
        torch.manual_seed(32)
        mobile = build_model("mobilenet_v3_large", pretrained=False)
        images, labels = training.batch([0, 1], eval_tf)
        loss = torch.nn.functional.cross_entropy(mobile(images), labels)
        loss.backward()
        gradients = [p.grad for p in mobile.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError("MobileNet fictitious-image gradient failed")
        gradient_norm = math.sqrt(sum(float(g.double().square().sum()) for g in gradients))
        if not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise RuntimeError("MobileNet task gradient is zero/nonfinite")
        if torch.cuda.is_initialized() or any(p.device.type != "cpu" for p in mobile.parameters()):
            raise RuntimeError("CPU rehearsal initialized CUDA")
        files = {str(p.relative_to(output)): sha256(p.read_bytes()) for p in sorted(output.rglob("*"))
                 if p.is_file() and public in p.parents}
        receipt = dict(status="passed", host=socket.gethostname(), cpu_only=True, cuda_initialized=False,
                       fictitious_labels_images=True, scientific_seed_claims=0, gpu_hours=0,
                       public_manifest_sha256=preparation["public_manifest_sha256"], files=files,
                       source_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])):sha256(p.read_bytes())
                                      for p in (Path(__file__), Path(__file__).resolve().parents[1]/"tralo/knee_snapshot_data.py",
                                                Path(__file__).resolve().parents[1]/"tralo/knee_snapshot_local.py")},
                       pto_trajectory_parity=True, fit=fits[1], window=window,
                       snapshots=snapshots, allocations=allocations,
                       private_and_original_reads_denied=True, checked_open_calls=len(access),
                       mobile_task_gradient_norm=gradient_norm,
                       mobile_nonzero_gradient_tensors=sum(int(torch.count_nonzero(g)>0) for g in gradients),
                       seconds=time.monotonic()-started,
                       peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None,
                       limitations="Fictitious CPU rehearsal; no real-source, pretrained, GPU cost/dose, seed freshness, OS isolation or budget certification.")
        (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False))
        print(json.dumps({k:receipt[k] for k in ("status", "host", "seconds", "peak_rss_kib", "mobile_task_gradient_norm")}))
    finally:
        builtins.open, Path.open = original_open, original_path_open


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run(sys.argv[1])
