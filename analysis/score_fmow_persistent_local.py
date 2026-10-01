"""Independent integrity gate and offline scorer for persistent fmow2 training.

The gate never loads development labels. The complete-block scorer audits every
seed before opening labels, and refuses an incomplete or modified block.

Usage:
  python analysis/score_fmow_persistent_local.py --replay-device cuda:0 --gate PILOT REF DATA OUTPUT
  python analysis/score_fmow_persistent_local.py --replay-device cuda:0 --pilot-score PILOT DATA GATE_RECEIPT OUTPUT
  python analysis/score_fmow_persistent_local.py --replay-device cuda:0 FULL_ROOT DATA GATE_RECEIPT OUTPUT
"""

import csv
from contextlib import contextmanager
import hashlib
import json
import math
import os
import platform
import random
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import score_fmow_local as base  # noqa: E402
from tralo.fmow_yuval import ArrayImages, FILES, roles, transforms_for  # noqa: E402
from tralo.fmow_yuval import make_model  # noqa: E402
from tralo.knee_end_to_end import infer  # noqa: E402
from tralo.local_alm import snapshot_phr_step  # noqa: E402
from tralo.local_targeted_step import local_targeted_step  # noqa: E402
from tralo.knee_experiment import source  # noqa: E402

PILOT = 6700
PILOT_RUNNER_RELEASE = "1bacdb448210a2083b181d86aa46bb8f0b29c6db"
SEEDS = tuple(range(6701, 6713))
EPOCHS = tuple(range(1, 8))
SNAPSHOTS = (5, 6, 7)
CORRECTIONS = tuple(range(2, 8))
ARMS = ("ce_null", "focal_clip", "cap10_tralo", "cap10_phr",
        "cap20_tralo", "cap20_phr")
PILOT_NULLS = ("tralo_null", "alm_null")
WEIGHT_SHA = "5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997"
TRAIN_COUNT, STOP_COUNT, POOL_COUNT, TEST_COUNT = 15841, 1829, 1673, 3442
RECIPE = {"study": "fmow_persistent_local_v1", "backbone": "mobilenet_v3_large",
          "epochs": 7, "batch_size": 32, "development_batch_size": 16,
          "lr": 1e-4, "weight_decay": 0.0, "decay_epoch": 5,
          "decay_factor": 0.8, "radius": 0.1, "rho": 0.5,
          "focal_alpha": 0.25, "focal_gamma": 2.0}
FAMILY = tuple((d, treated, control) for d in (10, 20) for treated, control in (
    ("tralo", "ce_null"), ("phr", "ce_null"),
    ("tralo", "phr"), ("tralo", "focal_clip")))
REPLAY_ATOL, REPLAY_RTOL = 1e-7, 1e-6
PREPROCESSING = {"size": [224, 224], "color": "RGB",
                 "mean": [.485, .456, .406], "std": [.229, .224, .225],
                 "train_augmentation": {"horizontal_flip": .5, "rotation_degrees": 3,
                                        "affine_translate": [.1, .1],
                                        "affine_scale": [.9, 1.1],
                                        "color_jitter": .2}}


def _hash(path):
    return base.sha256(path)


def _json(path):
    return base._json(path)


def _scorer_identity():
    """Pin the independent scorer and queue to a clean immutable release."""
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(
            ["git", "-c", "gc.auto=0", "-C", str(root), *args],
            text=True, stderr=subprocess.DEVNULL).strip()
    commit = git("rev-parse", "HEAD")
    if (not re.fullmatch(r"[0-9a-f]{40}", commit) or
            git("status", "--porcelain", "--untracked-files=all")):
        raise RuntimeError("independent scorer release is dirty")
    queue = root / "tools" / "fmow_persistent_local_queue.sh"
    for name in ("analysis/score_fmow_persistent_local.py",
                 "tools/fmow_persistent_local_queue.sh"):
        git("ls-files", "--error-unmatch", name)
    return {"release_commit": commit, "scorer_sha256": _hash(__file__),
            "queue_sha256": _hash(queue)}


def _source_at_release(commit):
    """Verify archived attempt source bytes against their immutable commit."""
    if not re.fullmatch(r"[0-9a-f]{40}", str(commit)):
        raise RuntimeError("cost attempt release commit missing")
    root = Path(__file__).resolve().parents[1]
    paths = subprocess.check_output(
        ["git", "-c", "gc.auto=0", "-C", str(root), "ls-tree", "-r",
         "--name-only", commit, "--", "tralo"], text=True).splitlines()
    names = [p for p in paths if re.fullmatch(r"tralo/[^/]+\.py", p)]
    if not names:
        raise RuntimeError("cost attempt release has no tracked source")
    result = {}
    for name in names:
        data = subprocess.check_output(
            ["git", "-c", "gc.auto=0", "-C", str(root), "show", f"{commit}:{name}"])
        result[Path(name).name] = hashlib.sha256(data).hexdigest()
    return result


def _events(path):
    return base._events(path)


def _one(rows, event, arm=None, epoch=None, divisor=None):
    found = [r for r in rows if r["event"] == event and
             (arm is None or r.get("arm") == arm) and
             (epoch is None or r.get("epoch") == epoch) and
             (divisor is None or r.get("cap_divisor") == divisor)]
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one {event}/{arm}/{epoch}, found {len(found)}")
    return found[0]


def _finite(value, name, *, minimum=None):
    if type(value) not in (int, float) or not math.isfinite(value) or (
            minimum is not None and value < minimum):
        raise RuntimeError(f"{name} is missing or invalid")
    return float(value)


def _near(value, expected, name, atol=1e-5):
    if abs(_finite(value, name) - expected) > atol:
        raise RuntimeError(f"{name} differs from independent recount")


def _state_hash(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("checkpoint model state is not tensor-only")
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _nested_hash(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous()
            digest.update(b"T")
            digest.update(str(array.dtype).encode())
            digest.update(json.dumps(list(array.shape)).encode())
            digest.update(array.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"D")
            for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"L" if isinstance(item, list) else b"U")
            digest.update(str(len(item)).encode())
            for entry in item:
                visit(entry)
        elif item is None or isinstance(item, (bool, int, float, str)):
            digest.update(json.dumps([type(item).__name__, item], allow_nan=False).encode())
        else:
            raise RuntimeError("unsupported optimizer state type")

    visit(value)
    return digest.hexdigest()


def _check_config(config, seed, reference):
    required = set(RECIPE) | {"seed", "pilot", "reference", "pretrained_sha256"}
    if not isinstance(config, dict) or set(config) != required or any(
            type(config.get(key)) is not type(value) or config[key] != value
            for key, value in RECIPE.items()):
        raise RuntimeError("config differs from fixed persistent protocol")
    if (type(config["seed"]) is not int or config["seed"] != seed or
            type(config["pilot"]) is not bool or config["pilot"] != (seed == PILOT) or
            type(config["reference"]) is not bool or config["reference"] != reference or
            (reference and seed != PILOT) or seed not in (PILOT, *SEEDS)):
        raise RuntimeError("pilot/full/reference identity differs")
    if config["pretrained_sha256"] != WEIGHT_SHA:
        raise RuntimeError("pretrained checkpoint digest differs from fixed recipe")


def _data_hashes(data_root):
    root = Path(data_root)
    actual = {name: _hash(root / name) for name in FILES}
    if actual != FILES:
        raise RuntimeError("fmow2 data files differ from pinned bytes")
    return actual


def _split_and_pool(data_root):
    """Rebuild roles from locations only; test labels are never parsed here."""
    root = Path(data_root)
    def locations(path):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            column = header.index("location")
            return [{"location": row[column]} for row in reader]
    training = locations(root / "train_meta.csv")
    test = locations(root / "test_meta.csv")
    if len(training) != TRAIN_COUNT + STOP_COUNT or len(test) != TEST_COUNT:
        raise RuntimeError("fmow2 metadata shape differs")
    split = roles(training, test)
    expected = {"train": split["train"], "stop": split["stop"],
                "dev": split["dev"], "stop_countries": split["stop_countries"],
                "dev_countries": split["dev_countries"],
                "reserved_countries": split["reserved_countries"]}
    pool = [{"sample_id": f"test{i}", "location": test[i]["location"]}
            for i in split["dev"]]
    if (len(split["train"]) != TRAIN_COUNT or len(split["stop"]) != STOP_COUNT or
            len(pool) != POOL_COUNT or set(expected["dev_countries"]) != base.COUNTRIES or
            set(expected["reserved_countries"]) != base.RESERVED):
        raise RuntimeError("fmow2 country/data role identity differs")
    return expected, pool


def _training_input_fingerprints(images, train_labels, split, seeds):
    """Rebuild actual seeded first train batches and fixed transform probes."""
    from PIL import Image
    from torchvision import transforms as T

    train = ArrayImages(images["train"], split["train"], train_labels)
    train_tf, eval_tf = transforms_for()
    normalize = T.Normalize(mean=PREPROCESSING["mean"], std=PREPROCESSING["std"])
    expected_train = T.Compose([
        T.Resize((224, 224)), T.RandomHorizontalFlip(p=.5), T.RandomRotation(3),
        T.RandomAffine(degrees=0, translate=(.1, .1), scale=(.9, 1.1)),
        T.ColorJitter(brightness=.2, contrast=.2, saturation=.2),
        T.ToTensor(), normalize])
    expected_eval = T.Compose([T.Resize((224, 224)), T.ToTensor(), normalize])
    train_image = Image.fromarray(np.array(images["train"][split["train"][0]], copy=True))
    eval_image = Image.fromarray(np.array(images["test"][split["dev"][0]], copy=True))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7726700)
        actual_train = train_tf(train_image)
        torch.manual_seed(7726700)
        reference_train = expected_train(train_image)
    if not torch.equal(actual_train, reference_train):
        raise RuntimeError("actual training augmentation differs from fixed transform")
    actual_eval, reference_eval = eval_tf(eval_image), expected_eval(eval_image)
    if not torch.equal(actual_eval, reference_eval):
        raise RuntimeError("actual evaluation transform differs from fixed transform")
    outcome = {}
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed + 105729)
        weights = train.weights()
        outcome[str(seed)] = {}
        with torch.random.fork_rng(devices=[]):
            for epoch in EPOCHS:
                order = torch.multinomial(weights, len(weights), replacement=True,
                                          generator=generator).tolist()
                ids = [int(train.indices[index]) for index in order]
                order_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
                torch.manual_seed(seed * 1000003 + epoch * 10007)
                batch, _ = train.batch(order[:RECIPE["batch_size"]], train_tf)
                outcome[str(seed)][str(epoch)] = {
                    "sample_order_sha256": order_hash,
                    "first_batch_sha256": hashlib.sha256(batch.numpy().tobytes()).hexdigest()}
    return {"train_transform_probe_sha256": hashlib.sha256(
                actual_train.contiguous().numpy().tobytes()).hexdigest(),
            "evaluation_transform_probe_sha256": hashlib.sha256(
                actual_eval.contiguous().numpy().tobytes()).hexdigest(),
            "first_batches": outcome}


def _audit_identity(directory, data_root, expected_hashes):
    d = Path(directory)
    config, manifest, summary = (_json(d / name) for name in
                                 ("config.json", "manifest.json", "summary.json"))
    events = _events(d / "events.jsonl")
    started, completed = _one(events, "started"), _one(events, "completed")
    seed = summary.get("seed")
    reference = config.get("reference")
    _check_config(config, seed, reference)
    if (summary.get("pilot") is not config["pilot"] or
            summary.get("reference") is not config["reference"] or
            started.get("config_sha256") != _hash(d / "config.json") or
            started.get("manifest_sha256") != _hash(d / "manifest.json") or
            completed.get("summary_sha256") != _hash(d / "summary.json")):
        raise RuntimeError("run config/manifest/summary byte provenance differs")
    identity = summary.get("identity")
    required = {"source_sha256", "config_sha256", "data_files", "split_sha256",
                "pretrained_weight", "development_labels_loaded", "reserved_images_used",
                "preprocessing", "preprocessing_sha256", "task_loss", "task_optimizer"}
    if (not isinstance(identity, dict) or set(identity) != required or
            any(manifest.get(key) != value for key, value in identity.items()) or
            any(started.get(key) != value for key, value in identity.items()) or
            identity["source_sha256"] != source() or
            identity["config_sha256"] != _hash(d / "config.json") or
            identity["data_files"] != expected_hashes or
            identity["development_labels_loaded"] is not False or
            identity["reserved_images_used"] is not False or
            identity["pretrained_weight"].get("sha256") != WEIGHT_SHA or
            identity["preprocessing"] != PREPROCESSING or
            identity["preprocessing_sha256"] != hashlib.sha256(json.dumps(
                PREPROCESSING, sort_keys=True, separators=(",", ":")).encode()).hexdigest() or
            identity["task_loss"] != {"ce": "torch.nn.functional.cross_entropy",
                                       "focal": "mean(-0.25*(1-p_true)^2*log(p_true))"} or
            identity["task_optimizer"] != "Adam, weight_decay=0, epoch decay x0.8 after epoch5" or
            started.get("precision") != "fp32_tf32_off"):
        raise RuntimeError("source/data/privacy/precision provenance differs")
    split, pool = _split_and_pool(data_root)
    split_sha = hashlib.sha256(json.dumps(split, sort_keys=True,
                             separators=(",", ":")).encode()).hexdigest()
    observed_pool = manifest.get("pool_rows")
    if (not isinstance(observed_pool, list) or
            any(not isinstance(row, dict) or
                set(row) != {"split", "sample_id", "location"} or
                row["split"] != "val" for row in observed_pool)):
        raise RuntimeError("runner development pool fields differ from label-free contract")
    projected_pool = [{"sample_id": row["sample_id"],
                       "location": row["location"]} for row in observed_pool]
    if (manifest.get("split") != split or projected_pool != pool or
            identity["split_sha256"] != split_sha or
            any(set(row) != {"sample_id", "location"} for row in pool)):
        raise RuntimeError("label-free split or pool identity differs")
    groups = [row["location"] for row in pool]
    quotas = base._quotas(groups)
    if (manifest.get("quotas") != quotas or summary.get("quotas") != quotas or
            started.get("quotas") != quotas or
            manifest.get("country_sizes") != {g: groups.count(g) for g in sorted(set(groups))}):
        raise RuntimeError("pooled/country quotas differ from unlabeled recount")
    return config, manifest, summary, events, pool


def _audit_launch(directory, config, manifest, events, *, replay_host=None):
    """Bind a run to the exclusive physical-card launch and completion receipts."""
    d = Path(directory).resolve()
    job = f"{config['seed']}_{'ref' if config['reference'] else 'step'}"
    launch = _json(d.parent / f"{job}.launch.json")
    complete = _json(d.parent / f"{job}.complete.json")
    common = {"job": job, "seed": config["seed"],
              "output_dir": str(d), "release_commit": launch.get("release_commit")}
    if (any(launch.get(key) != value or complete.get(key) != value
            for key, value in common.items() if key != "seed") or
            type(launch.get("seed")) is not int or launch["seed"] != config["seed"] or
            type(complete.get("seed")) is not int or complete["seed"] != config["seed"] or
            complete.get("exit_code") != 0 or
            launch.get("reference") is not config["reference"] or
            launch.get("precision") != "fp32_tf32_off" or
            launch.get("gpu_uuid", "").startswith("GPU-") is False or
            launch.get("gpu_uuid") != complete.get("gpu_uuid") or
            launch.get("host") != complete.get("host") or
            launch.get("config_sha256") != manifest["config_sha256"] or
            launch.get("source_sha256") != manifest["source_sha256"] or
            launch.get("run_root") != str(d.parent) or
            not isinstance(launch.get("release_commit"), str) or
            not re.fullmatch(r"[0-9a-f]{40}", launch["release_commit"])):
        raise RuntimeError("exclusive launch/completion receipt differs from run")
    started = _one(events, "started")
    if (started.get("precision") != launch["precision"] or
            (replay_host is not None and
             replay_host.split(".")[0] != launch["host"].split(".")[0])):
        raise RuntimeError("launch host/precision differs from replay/training")
    return launch


def _audit_training_fingerprints(directory, config, summary, launch):
    """Tie actual epochs to the queue's real-image transform preflight."""
    run_root = Path(directory).resolve().parent
    costs = run_root.parent / ".fmow-persistent-local-cost"
    attempt = costs / hashlib.sha256(str(run_root).encode()).hexdigest()
    preflight = _json(attempt / "preflight.json")
    if (preflight.get("passed") is not True or
            preflight.get("run_root") != str(run_root) or
            preflight.get("source_sha256") != source() or
            preflight.get("preprocessing") != PREPROCESSING):
        raise RuntimeError("run lacks real-image/source transform preflight")
    for key in ("train_transform_probe_sha256", "evaluation_transform_probe_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(preflight.get(key, ""))):
            raise RuntimeError("run lacks real-image transform probe")
    expected = preflight.get("first_batches", {}).get(str(config["seed"]))
    if not isinstance(expected, dict) or set(expected) != {str(e) for e in EPOCHS}:
        raise RuntimeError("run lacks seven independent real first-batch hashes")
    for arm in summary["arms"].values():
        for epoch in arm["epochs"]:
            row = expected[str(epoch["epoch"])]
            if (epoch.get("first_batch_sha256") != row.get("first_batch_sha256") or
                    epoch.get("sample_order_sha256") != row.get("sample_order_sha256")):
                raise RuntimeError("trained first batch/order differs from real-data preflight")
    attempt_record = _json(attempt / "attempt.json")
    if (launch["release_commit"] != attempt_record.get("release_commit") or
            launch.get("host") != attempt_record.get("host") or
            launch.get("gpu_uuid") != attempt_record.get("gpu_uuid") or
            launch.get("run_root") != attempt_record.get("run_root") or
            attempt_record.get("run_root") != str(run_root) or
            attempt_record.get("mode") != ("pilot-ref" if config.get("reference") else
                                           "pilot-step" if config.get("pilot") else "full")):
        raise RuntimeError("training/preflight release, host, GPU or root differs")
    return {"preflight_sha256": _hash(attempt / "preflight.json"),
            "attempt_sha256": _hash(attempt / "attempt.json")}


def _scopes(probabilities, groups, quota):
    q = probabilities[:, base.CAPPED]
    called = probabilities.argmax(1) == base.CAPPED
    out = {}
    for name, cap, indices in [("global", quota["global_cap"], list(range(len(groups))))] + [
            (g, quota["local_caps"][g], [i for i, group in enumerate(groups) if group == g])
            for g in sorted(quota["local_caps"])]:
        soft = float(q[indices].sum())
        hard = int(called[indices].sum())
        residual = (soft - cap) / max(cap, 1)
        out[name] = {"cap": cap, "soft": soft, "hard": hard,
                     "signed_residual": residual,
                     "positive_residual": max(0., residual),
                     "hard_excess": max(0, hard - cap)}
    return out


def _audit_scopes(record, quota, name, *, actual=None):
    caps = {"global": quota["global_cap"], **quota["local_caps"]}
    if not isinstance(record, dict) or set(record) != set(caps):
        raise RuntimeError(f"{name} scopes incomplete")
    for scope, cap in caps.items():
        row = record[scope]
        soft = _finite(row.get("soft"), f"{name}.{scope}.soft", minimum=0)
        hard = row.get("hard")
        if (row.get("cap") != cap or type(hard) is not int or hard < 0 or
                hard > POOL_COUNT):
            raise RuntimeError(f"{name}.{scope} cap/hard count invalid")
        residual = (soft - cap) / max(cap, 1)
        _near(row.get("signed_residual"), residual, f"{name}.{scope}.residual")
        _near(row.get("positive_residual"), max(0, residual), f"{name}.{scope}.positive")
        if row.get("hard_excess") != max(0, hard - cap):
            raise RuntimeError(f"{name}.{scope}.hard_excess differs")
        if actual is not None:
            observed = actual[scope]
            if hard != observed["hard"]:
                raise RuntimeError(f"{name}.{scope} hard count differs from saved probabilities")
            _near(soft, observed["soft"], f"{name}.{scope} soft count", atol=1e-3)
    if abs(record["global"]["soft"] - math.fsum(record[g]["soft"] for g in quota["local_caps"])) > 1e-3:
        raise RuntimeError(f"{name} country soft counts do not partition pool")
    if record["global"]["hard"] != sum(record[g]["hard"] for g in quota["local_caps"]):
        raise RuntimeError(f"{name} country hard counts do not partition pool")


def _audit_correction(row, quota, method, previous_dual):
    before, after = row["before"], row["after"]
    _audit_scopes(before, quota, "before")
    _audit_scopes(after, quota, "after")
    applied = row.get("applied_constraint_updates")
    skipped = row.get("skipped_constraint_updates")
    attempted = 0 if method == "null" else 1
    if (row.get("attempted_constraint_updates") != attempted or
            type(applied) is not int or applied not in (0, 1) or
            skipped != attempted - applied or row.get("rng_neutral") is not True or
            row.get("optimizer_before_sha256") != row.get("optimizer_after_sha256")):
        raise RuntimeError("correction dose or optimizer/RNG neutrality invalid")
    radius = _finite(row.get("applied_radius"), "applied radius", minimum=0)
    displacement = _finite(row.get("actual_displacement"), "displacement", minimum=0)
    if (radius > .1 + 1e-8 or abs(radius - displacement) > 1e-5 or
            bool(applied) != (radius > 0)):
        raise RuntimeError("correction dose exceeds fixed radius or differs from displacement")
    controller = row["controller"]
    for when, values in (("before", before), ("after", after)):
        if (controller.get(f"hard_{when}_global") != values["global"]["hard"] or
                controller.get(f"hard_{when}_local") != {
                    g: values[g]["hard"] for g in quota["local_caps"]}):
            if method != "null":
                raise RuntimeError(f"{method} controller hard {when} differs")
        if method != "null":
            _near(controller.get(f"soft_{when}_global"), values["global"]["soft"],
                  f"{method} controller pooled soft {when}", atol=1e-3)
            for group in quota["local_caps"]:
                _near(controller[f"soft_{when}_local"][group], values[group]["soft"],
                      f"{method} controller {group} soft {when}", atol=1e-3)
    if controller.get("applied") is not bool(applied):
        raise RuntimeError("controller applied flag differs")
    _near(controller.get("radius"), radius, "controller radius")
    _near(controller.get("displacement"), displacement, "controller displacement")
    if method == "null":
        if (applied or radius or displacement or row.get("attempted_radius") != 0 or
                row.get("dual_after") is not None or before != after or
                row["model_before_sha256"] != row["model_after_sha256"] or
                controller.get("skip_reason") != "scheduled_zero_step"):
            raise RuntimeError("zero-step slot changed state or count")
        return None
    policy = controller.get("boundary_policy")
    if not isinstance(policy, dict) or policy.get("applied") is not bool(applied):
        raise RuntimeError("boundary policy missing or disagrees with correction")
    _near(policy.get("radius"), radius, "boundary policy radius")
    if applied:
        probes = policy.get("probes")
        if not isinstance(probes, list) or not probes or policy.get("reason") != "accepted":
            raise RuntimeError("applied correction lacks accepted boundary probe")
        last = probes[-1]
        if last.get("pooled_hard") != after["global"]["hard"]:
            raise RuntimeError("accepted probe pooled hard count differs")
        _near(last.get("pooled_soft"), after["global"]["soft"], "accepted pooled soft", atol=1e-3)
        for group in quota["local_caps"]:
            _near(last["local_soft"][group], after[group]["soft"],
                  f"accepted {group} soft", atol=1e-3)
    elif row["model_before_sha256"] != row["model_after_sha256"]:
        raise RuntimeError("skipped correction changed model")
    if method == "phr":
        scopes = ["global"] + sorted(quota["local_caps"])
        dual_before = controller.get("dual_before")
        dual_after = row.get("dual_after")
        if (not isinstance(dual_before, list) or not isinstance(dual_after, list) or
                len(dual_before) != len(scopes) or len(dual_after) != len(scopes)):
            raise RuntimeError("PHR dual scopes incomplete")
        if previous_dual is None:
            previous_dual = [0.] * len(scopes)
        for label, values in (("before", before), ("after", after)):
            residuals = controller.get(f"residuals_{label}")
            if not isinstance(residuals, list) or len(residuals) != len(scopes):
                raise RuntimeError(f"PHR {label} residual vector incomplete")
            for i, scope in enumerate(scopes):
                _near(residuals[i], values[scope]["signed_residual"],
                      f"PHR {label} {scope} residual", atol=1e-4)
            penalty = sum((max(0., dual_before[i] + .5 * residuals[i]) ** 2 -
                           dual_before[i] ** 2) for i in range(len(scopes)))
            _near(controller.get(f"penalty_{label}"), penalty,
                  f"PHR {label} penalty", atol=1e-4)
        for i, scope in enumerate(scopes):
            _near(dual_before[i], previous_dual[i], f"PHR {scope} dual continuity")
            _near(dual_after[i], max(0., dual_before[i] + .5 * after[scope]["signed_residual"]),
                  f"PHR {scope} dual update")
        if controller.get("dual_after") != dual_after:
            raise RuntimeError("PHR controller dual differs")
        return dual_after
    if row.get("dual_after") is not None:
        raise RuntimeError("TraLO should have no PHR dual")
    active_global = before["global"]["hard"] > quota["global_cap"]
    active_local = sorted(g for g in quota["local_caps"] if
                          before[g]["hard"] > quota["local_caps"][g])
    if (controller.get("active_global") is not active_global or
            controller.get("active_local") != active_local):
        raise RuntimeError("TraLO hard-active scopes differ from independent recount")
    return None


def _snapshot(directory, arm, epoch, record, n, expected_dual):
    """Check creation-time hashes and replay checkpoint state against receipts."""
    d = Path(directory) / arm
    names = ("checkpoint", "probability", "dual")
    paths = {}
    for name in names:
        filename = record.get(f"{name}_file")
        if (type(filename) is not str or Path(filename).name != filename or
                not filename.startswith(f"epoch{epoch:02d}_")):
            raise RuntimeError(f"{arm}/{epoch} {name} artifact path invalid")
        path = d / filename
        if _hash(path) != record.get(f"{name}_sha256"):
            raise RuntimeError(f"{arm}/{epoch} {name} artifact digest differs")
        paths[name] = path
    probabilities = base._probabilities(paths["probability"], n)
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    if (checkpoint.get("epoch") != epoch or
            _state_hash(checkpoint["model"]) != record.get("model_state_sha256") or
            _nested_hash(checkpoint["optimizer"]) != record.get("optimizer_state_sha256")):
        raise RuntimeError(f"{arm}/{epoch} checkpoint state hashes differ")
    dual = _json(paths["dual"])
    if checkpoint.get("dual") is None:
        if dual is not None or expected_dual is not None:
            raise RuntimeError(f"{arm}/{epoch} unexpected dual state")
    else:
        actual_dual = checkpoint["dual"].tolist()
        if dual != actual_dual or dual != expected_dual:
            raise RuntimeError(f"{arm}/{epoch} PHR dual artifact differs")
    return probabilities


def _pre_correction_snapshot(directory, arm, epoch, correction, sample_ids,
                             expected_dual):
    """Bind pilot-only pre-state to the logged correction before GPU replay."""
    record = correction.get("pre_correction_snapshot")
    if not isinstance(record, dict):
        raise RuntimeError(f"{arm}/{epoch} missing pre-correction replay evidence")
    d = Path(directory) / arm
    paths = {}
    for kind in ("checkpoint", "probability", "dual"):
        name = record.get(f"{kind}_file")
        if (type(name) is not str or Path(name).name != name or
                not name.startswith(f"epoch{epoch:02d}_pre_correction")):
            raise RuntimeError(f"{arm}/{epoch} pre-correction path invalid")
        path = d / name
        if _hash(path) != record.get(f"{kind}_sha256"):
            raise RuntimeError(f"{arm}/{epoch} pre-correction artifact digest differs")
        paths[kind] = path
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    artifact = torch.load(paths["probability"], map_location="cpu", weights_only=True)
    if (checkpoint.get("epoch") != epoch or checkpoint.get("arm") != arm.rsplit("_", 1)[-1] or
            checkpoint.get("cap_divisor") != correction.get("cap_divisor") or
            checkpoint.get("model_training") is not True or
            _state_hash(checkpoint["model"]) != record.get("model_state_sha256") or
            _nested_hash(checkpoint["optimizer"]) != record.get("optimizer_state_sha256") or
            record["model_state_sha256"] != correction.get("model_before_sha256") or
            record["optimizer_state_sha256"] != correction.get("optimizer_before_sha256") or
            record.get("post_model_state_sha256") != correction.get("model_after_sha256") or
            record.get("post_optimizer_state_sha256") != correction.get("optimizer_after_sha256")):
        raise RuntimeError(f"{arm}/{epoch} pre-correction state differs")
    dual = _json(paths["dual"])
    loaded_dual = checkpoint.get("dual")
    if (dual != expected_dual or
            (loaded_dual is None) != (dual is None) or
            (loaded_dual is not None and loaded_dual.tolist() != dual)):
        raise RuntimeError(f"{arm}/{epoch} pre-correction PHR dual differs")
    ids_hash = hashlib.sha256(json.dumps(sample_ids, separators=(",", ":")).encode()).hexdigest()
    probs = artifact.get("probabilities")
    first = artifact.get("first_batch_probabilities")
    if (artifact.get("sample_ids") != sample_ids or
            record.get("sample_ids_sha256") != ids_hash or
            not isinstance(probs, torch.Tensor) or
            tuple(probs.shape) != (len(sample_ids), base.CLASSES) or
            not bool(torch.isfinite(probs).all()) or
            not isinstance(first, torch.Tensor) or
            not torch.equal(first, probs[:len(first)]) or
            len(first) != min(RECIPE["development_batch_size"], len(probs)) or
            artifact.get("first_batch_input_sha256") != record.get("first_batch_input_sha256")):
        raise RuntimeError(f"{arm}/{epoch} pre-correction probabilities/IDs differ")
    return checkpoint, probs, artifact["first_batch_input_sha256"]


def _compare_replay(expected, actual, artifact):
    if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape or not bool(
            torch.isfinite(actual).all()):
        raise RuntimeError(f"{artifact}: replay probabilities missing/nonfinite/shape changed")
    delta = (actual.double() - expected.double()).abs()
    tolerance = REPLAY_ATOL + REPLAY_RTOL * expected.double().abs()
    ratio = delta / tolerance
    largest = float(delta.max())
    worst = float(ratio.max())
    if worst > 1:
        raise RuntimeError(f"{artifact}: checkpoint-to-probability replay differs "
                           f"(max_abs={largest:.9g}, max_tolerance_ratio={worst:.5g})")
    return {"max_absolute_difference": largest, "max_tolerance_ratio": worst}


def _replay_checkpoint(path, batches, expected, artifact, device, *, model_factory=make_model):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = model_factory(pretrained=False, backbone="mobilenet_v3_large")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    output = []
    with torch.inference_mode():
        for batch in batches:
            output.append(model(batch.to(device)).softmax(1).cpu())
    actual = torch.cat(output)
    result = _compare_replay(expected, actual, artifact)
    del checkpoint, model, output, actual
    return result


def _compare_controller(expected, actual, path="controller"):
    """Require the independently rerun controller to agree, with fixed FP tolerance."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise RuntimeError(f"{path} replay fields differ")
        for key in expected:
            _compare_controller(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise RuntimeError(f"{path} replay list differs")
        for i, (left, right) in enumerate(zip(expected, actual)):
            _compare_controller(left, right, f"{path}[{i}]")
    elif type(expected) in (float, int) and type(actual) in (float, int):
        if not math.isclose(expected, actual, rel_tol=2e-4, abs_tol=2e-5):
            raise RuntimeError(f"{path} replay value differs")
    elif type(expected) is not type(actual) or expected != actual:
        raise RuntimeError(f"{path} replay value differs")


def _independent_direction(model, batches, groups, quota, method, dual, before):
    """Differentiate normalized real-image soft counts independently of runner."""
    names = ["global"] + sorted(quota["local_caps"])
    caps = {"global": quota["global_cap"], **quota["local_caps"]}
    observed = _scopes(before, groups, quota)
    active = ([name for name in names if observed[name]["hard"] > caps[name]]
              if method == "tralo" else names)
    if method == "tralo" and (not active or
            all(observed[name]["soft"] <= caps[name] for name in names)):
        return 0., {}
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("correction replay has no trainable parameters")
    device = params[0].device
    scope_grad = {name: [torch.zeros_like(p) for p in params] for name in names}
    start = 0
    model.eval()
    for batch in batches:
        logits = model(batch.to(device))
        q = logits.softmax(1)[:, base.CAPPED]
        end = start + len(batch)
        for j, name in enumerate(names):
            value = q if name == "global" else q[torch.tensor(
                [group == name for group in groups[start:end]], device=device)]
            grads = torch.autograd.grad(value.sum() / max(caps[name], 1), params,
                                        retain_graph=j + 1 < len(names), allow_unused=True)
            for total, grad in zip(scope_grad[name], grads):
                if grad is not None:
                    total.add_(grad.detach())
        start = end
    if start != len(groups):
        raise RuntimeError("correction replay pool length differs")
    if method == "phr":
        coefficients = [max(0., float(dual[i]) + RECIPE["rho"] *
                            observed[name]["signed_residual"])
                        for i, name in enumerate(names)]
        gradient = [sum((scope_grad[name][i] * coefficients[j]
                         for j, name in enumerate(names)), torch.zeros_like(param))
                    for i, param in enumerate(params)]
    else:
        gradient = [sum((scope_grad[name][i] for name in active),
                        torch.zeros_like(param)) for i, param in enumerate(params)]
    norm = math.sqrt(sum(float(g.double().square().sum()) for g in gradient))
    if not math.isfinite(norm):
        raise RuntimeError("correction replay gradient nonfinite")
    direction = [-g / norm for g in gradient] if norm else [torch.zeros_like(g) for g in gradient]
    derivatives = {(name if method == "phr" else
                    "pooled" if name == "global" else f"local:{name}"):
                   sum(float((g.double() * d.double()).sum())
                       for g, d in zip(scope_grad[name], direction))
                   for name in names} if norm else {}
    return norm, derivatives


def _independent_boundary_policy(correction, quota, method):
    """Recalculate the fixed acceptance arithmetic from logged observed probes."""
    before = correction["before"]
    caps = {"pooled": quota["global_cap"], **{
        f"local:{name}": cap for name, cap in quota["local_caps"].items()}}
    soft = {"pooled": before["global"]["soft"], **{
        f"local:{name}": before[name]["soft"] for name in quota["local_caps"]}}
    hard = before["global"]["hard"]
    violation = {name: max(0., (soft[name] - cap) / max(cap, 1))
                 for name, cap in caps.items()}
    policy = correction["controller"]["boundary_policy"]
    probes = policy.get("probes")
    if not isinstance(probes, list):
        raise RuntimeError("boundary probe log missing")
    reason = policy.get("reason")
    if method == "tralo" and reason == "no_hard_active_scope":
        if (hard > quota["global_cap"] or any(before[name]["hard"] > cap
                for name, cap in quota["local_caps"].items()) or probes):
            raise RuntimeError("boundary no-hard-active decision differs")
        return
    if reason == "zero_phr_gradient":
        if method != "phr" or probes or correction["controller"].get("gradient_norm") != 0:
            raise RuntimeError("boundary zero-PHR-gradient decision differs")
        return
    if not sum(violation.values()):
        if reason != "no_positive_violation" or probes:
            raise RuntimeError("boundary slack decision differs")
        return
    derivatives = correction["controller"].get("scope_directional_derivatives", {})
    if method == "phr":
        derivatives = {"pooled": derivatives.get("global"), **{
            f"local:{name}": derivatives.get(name) for name in quota["local_caps"]}}
    if set(derivatives) != set(caps) or any(not isinstance(x, (int, float)) or
            not math.isfinite(x) for x in derivatives.values()):
        raise RuntimeError("boundary scope derivatives missing")
    conflicts = sorted(name for name in caps if violation[name] > 0 and
                       derivatives[name] >= 0)
    if conflicts:
        if reason != "conflicting_direction" or probes or policy.get("conflicting_scopes") != conflicts:
            raise RuntimeError("boundary conflicting-scope decision differs")
        return
    initial = min(RECIPE["radius"], min(
        violation[name] / -derivatives[name] for name in caps if violation[name] > 0))
    if not math.isclose(initial, policy.get("initial_radius", math.nan),
                        rel_tol=2e-4, abs_tol=2e-5):
        raise RuntimeError("boundary initial radius differs")
    hard_floor = max(0, min(hard, quota["global_cap"]) - 1)
    soft_floor = max(0., min(soft["pooled"], quota["global_cap"]) - 1.)
    accepted = False
    for index, probe in enumerate(probes):
        radius = initial / (2 ** index)
        if (probe.get("halving") != index or not math.isclose(
                probe.get("radius", math.nan), radius, rel_tol=2e-4, abs_tol=2e-5)):
            raise RuntimeError("boundary probe radius/halving differs")
        candidate = {"pooled": probe["pooled_soft"], **{
            f"local:{name}": probe["local_soft"][name]
            for name in quota["local_caps"]}}
        candidate_violation = {name: max(0., (candidate[name] - cap) / max(cap, 1))
                               for name, cap in caps.items()}
        rejection = []
        if probe["pooled_hard"] < hard_floor:
            rejection.append("pooled_hard_floor")
        if candidate["pooled"] < soft_floor:
            rejection.append("pooled_soft_floor")
        for name in sorted(caps):
            if candidate_violation[name] > violation[name] + 1e-6:
                rejection.append(f"worsened_soft_violation:{name}")
        if not sum(candidate_violation.values()) <= sum(violation.values()) - 1e-6:
            rejection.append("insufficient_total_violation_reduction")
        if (probe.get("rejections") != rejection or
                probe.get("accepted") is not (not rejection)):
            raise RuntimeError("boundary probe acceptance/reasons differ")
        if not rejection:
            accepted = True
            if index != len(probes) - 1:
                raise RuntimeError("boundary probes continued after acceptance")
    if (accepted != bool(policy.get("applied")) or
            reason != ("accepted" if accepted else "no_acceptable_probe") or
            not math.isclose(policy.get("radius", math.nan),
                             probes[-1]["radius"] if accepted else 0.,
                             rel_tol=2e-4, abs_tol=2e-5)):
        raise RuntimeError("boundary accepted radius differs")


def _replay_pilot_correction(directory, arm, epoch, correction, post_snapshot_record,
                             batches, sample_ids, groups, quota, device,
                             *, model_factory=make_model):
    """Replay one real-image pilot intervention from immutable pre-state bytes."""
    checkpoint, before, first_input_hash = _pre_correction_snapshot(
        directory, arm, epoch, correction, sample_ids,
        correction["controller"].get("dual_before") if arm.endswith("_phr") else None)
    actual_first_hash = hashlib.sha256(batches[0].contiguous().numpy().tobytes()).hexdigest()
    if first_input_hash != actual_first_hash:
        raise RuntimeError(f"{arm}/{epoch} real-image first-batch preprocessing differs")
    model = model_factory(pretrained=False, backbone="mobilenet_v3_large")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).train(checkpoint["model_training"])
    actual_before = infer(model, batches)
    replay = _compare_replay(before, actual_before, f"{arm}/{epoch} pre-correction")
    _audit_scopes(correction["before"], quota, f"{arm}/{epoch} independent before",
                  actual=_scopes(actual_before, groups, quota))
    method = arm.rsplit("_", 1)[-1]
    dual = checkpoint["dual"]
    independent_norm, independent_derivatives = _independent_direction(
        model, batches, groups, quota, method, dual, actual_before)
    logged = correction["controller"]
    if logged.get("gradient_norm") is None:
        if independent_norm != 0:
            raise RuntimeError(f"{arm}/{epoch} missing nonzero gradient norm")
    elif not math.isclose(independent_norm, logged["gradient_norm"],
                          rel_tol=2e-4, abs_tol=2e-5):
        raise RuntimeError(f"{arm}/{epoch} independent constraint gradient differs")
    for name, derivative in independent_derivatives.items():
        if not math.isclose(derivative,
                            logged.get("scope_directional_derivatives", {}).get(name, math.nan),
                            rel_tol=2e-4, abs_tol=2e-5):
            raise RuntimeError(f"{arm}/{epoch} independent directional derivative differs")
    _independent_boundary_policy(correction, quota, method)
    optimizer = torch.optim.Adam(model.parameters(), lr=RECIPE["lr"], weight_decay=0.)
    optimizer.load_state_dict(checkpoint["optimizer"])
    optimizer_sha = _nested_hash(optimizer.state_dict())
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    saved_rng = checkpoint.get("rng")
    if (not isinstance(saved_rng, dict) or
            not isinstance(saved_rng.get("torch_cpu"), torch.Tensor) or
            not isinstance(saved_rng.get("torch_cuda"), list) or
            not isinstance(saved_rng.get("python"), tuple) or
            not isinstance(saved_rng.get("numpy"), dict)):
        raise RuntimeError(f"{arm}/{epoch} pre-correction RNG evidence missing")
    cuda_devices = [torch.device(device).index or 0] if str(device).startswith("cuda:") else []
    if cuda_devices and len(saved_rng["torch_cuda"]) != len(cuda_devices):
        raise RuntimeError(f"{arm}/{epoch} pre-correction CUDA RNG state differs")
    numpy_state = saved_rng["numpy"]
    if set(numpy_state) != {"bit_generator", "state", "position", "has_gauss", "cached_gaussian"}:
        raise RuntimeError(f"{arm}/{epoch} pre-correction NumPy RNG state differs")
    previous_python, previous_numpy = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=cuda_devices):
        try:
            torch.set_rng_state(saved_rng["torch_cpu"])
            if cuda_devices:
                torch.cuda.set_rng_state_all(saved_rng["torch_cuda"])
            random.setstate(saved_rng["python"])
            np.random.set_state((numpy_state["bit_generator"],
                                    np.asarray(numpy_state["state"], dtype=np.uint32),
                                    numpy_state["position"], numpy_state["has_gauss"],
                                    numpy_state["cached_gaussian"]))
            model.train(checkpoint["model_training"])
            if method == "tralo":
                replayed = local_targeted_step(
                    model, batches, groups, base.CAPPED, quota["global_cap"],
                    quota["local_caps"], boundary_calibrated=True,
                    max_radius=RECIPE["radius"])
                next_dual = None
            else:
                replayed, next_dual = snapshot_phr_step(
                    model, batches, groups, base.CAPPED, quota["global_cap"],
                    quota["local_caps"], dual, rho=RECIPE["rho"],
                    radius=RECIPE["radius"], boundary_calibrated=True)
            if (not torch.equal(torch.get_rng_state(), saved_rng["torch_cpu"]) or
                    (cuda_devices and any(not torch.equal(x, y) for x, y in zip(
                        torch.cuda.get_rng_state_all(), saved_rng["torch_cuda"]))) or
                    random.getstate() != saved_rng["python"] or
                    np.random.get_state()[1].tolist() != numpy_state["state"]):
                raise RuntimeError(f"{arm}/{epoch} correction replay changed RNG")
        finally:
            random.setstate(previous_python)
            np.random.set_state(previous_numpy)
    if (_nested_hash(optimizer.state_dict()) != optimizer_sha or
            any(not torch.equal(value, buffers[name]) for name, value in
                model.named_buffers())):
        raise RuntimeError(f"{arm}/{epoch} correction replay changed optimizer/buffers")
    _compare_controller(logged, replayed)
    after = infer(model, batches)
    _audit_scopes(correction["after"], quota, f"{arm}/{epoch} independent after",
                  actual=_scopes(after, groups, quota))
    if (_state_hash(model.state_dict()) != correction["model_after_sha256"] or
            (next_dual.tolist() if next_dual is not None else None) != correction["dual_after"]):
        raise RuntimeError(f"{arm}/{epoch} replayed post model or PHR dual differs")
    if epoch in SNAPSHOTS:
        artifact = _snapshot(directory, arm, epoch, post_snapshot_record, len(groups),
                             correction["dual_after"])
        _compare_replay(artifact, after, f"{arm}/{epoch} post-correction")
    del model
    return {"pre_probability": replay, "gradient_norm": independent_norm,
            "directional_derivatives": independent_derivatives,
            "post_model_sha256": correction["model_after_sha256"],
            "applied_radius": correction["applied_radius"]}


def _exclusive_replay_uuid(device):
    """Require UUID pinning and no foreign compute process before replay."""
    if device != "cuda:0":
        raise RuntimeError("scientific replay requires an explicitly selected CUDA device")
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", uuid):
        raise RuntimeError("scientific replay requires a physical UUID-pinned GPU")
    _replay_pid_check(uuid)
    return uuid


def _replay_pid_check(uuid):
    try:
        found = subprocess.run(["nvidia-smi", "-i", uuid, "--query-compute-apps=pid",
                                "--format=csv,noheader"], check=True, capture_output=True,
                               text=True, timeout=10).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("cannot verify exclusive replay GPU process ownership") from error
    pids = [line.strip() for line in found.splitlines() if line.strip()]
    if any(not pid.isdigit() or int(pid) != os.getpid() for pid in pids):
        raise RuntimeError(f"replay GPU {uuid} has a foreign compute process")


@contextmanager
def _exclusive_replay_lease(device, lock_directory=None):
    """Hold the same cooperating physical-card lock as the training queue."""
    import fcntl

    uuid = _exclusive_replay_uuid(device)
    directory = Path(lock_directory or
                     "/home/dsi/michaer8/tralo-rebuild/runs/.fmow-persistent-local-gpu-locks")
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / f"{uuid}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"replay GPU {uuid} lease already held") from error
        _replay_pid_check(uuid)
        yield uuid
    finally:
        os.close(fd)


def _replay_seed(directory, data_root, summary, snapshots, split, device):
    """Reload every post-correction checkpoint and infer label-free real images."""
    with _exclusive_replay_lease(device) as uuid:
        return _replay_seed_locked(directory, data_root, summary, snapshots,
                                   split, device, uuid)


def _replay_seed_locked(directory, data_root, summary, snapshots, split, device, uuid):
    if not torch.cuda.is_available():
        raise RuntimeError("scientific replay CUDA device unavailable")
    from PIL import Image
    from torchvision import transforms as T

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                           T.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    images = np.load(Path(data_root) / "test_images.npy", mmap_mode="r", allow_pickle=False)
    indices = split["dev"]
    if images.shape != (TEST_COUNT, 224, 224, 3) or len(indices) != POOL_COUNT:
        raise RuntimeError("real-image replay data shape differs")
    batches = []
    for start in range(0, len(indices), RECIPE["development_batch_size"]):
        batch = torch.stack([transform(Image.fromarray(np.array(images[i], copy=True)))
                             for i in indices[start:start + RECIPE["development_batch_size"]]])
        batches.append(batch)
    del images
    result = {}
    t0 = time.perf_counter()
    torch.cuda.synchronize(device)
    for arm, epochs in snapshots.items():
        result[arm] = {}
        for epoch, expected in epochs.items():
            _replay_pid_check(uuid)
            artifact = f"{arm}/epoch{epoch:02d}"
            record = summary["arms"][arm]["epochs"][epoch - 1]["post_correction_snapshot"]
            result[arm][str(epoch)] = _replay_checkpoint(
                Path(directory) / arm / record["checkpoint_file"],
                batches, expected, artifact, device)
    correction_replay = {}
    if summary["pilot"] and not summary["reference"]:
        independent_split, pool = _split_and_pool(data_root)
        if independent_split != split:
            raise RuntimeError("pilot correction real-image cohort differs")
        sample_ids = [row["sample_id"] for row in pool]
        groups = [row["location"] for row in pool]
        for divisor in (10, 20):
            for method in ("tralo", "phr"):
                arm = f"cap{divisor}_{method}"
                correction_replay[arm] = {}
                records = {row["epoch"]: row for row in summary["arms"][arm]["corrections"]}
                for epoch in CORRECTIONS:
                    _replay_pid_check(uuid)
                    post_record = (summary["arms"][arm]["epochs"][epoch - 1].get(
                        "post_correction_snapshot") if epoch in SNAPSHOTS else None)
                    correction_replay[arm][str(epoch)] = _replay_pilot_correction(
                        directory, arm, epoch, records[epoch], post_record, batches,
                        sample_ids, groups, summary["quotas"][str(divisor)], device)
    torch.cuda.synchronize(device)
    return {"host": platform.node(), "device": device, "gpu_uuid": uuid,
            "gpu_name": torch.cuda.get_device_name(device),
            "precision": "fp32_tf32_off", "atol": REPLAY_ATOL,
            "rtol": REPLAY_RTOL, "seconds": time.perf_counter() - t0,
            "artifacts": result, "pilot_corrections": correction_replay}


def _audit_arm(directory, arm, summary, events, groups, quotas, sample_ids=None):
    name = Path(directory).name
    row = summary["arms"][arm]
    epochs = row.get("epochs")
    if not isinstance(epochs, list) or [e.get("epoch") for e in epochs] != list(EPOCHS):
        raise RuntimeError(f"{name}/{arm} epoch list incomplete")
    expected_loss = "focal" if arm == "focal_clip" else "ce"
    if row.get("loss_kind") != expected_loss:
        raise RuntimeError(f"{arm} supervised loss identity differs")
    predictions = {}
    n = len(groups)
    method = arm.rsplit("_", 1)[-1]
    divisor = int(arm[3:5]) if arm.startswith("cap") else None
    correction_rows = row.get("corrections")
    required_corrections = 6 if method in ("tralo", "phr") else 12
    if not isinstance(correction_rows, list) or len(correction_rows) != required_corrections:
        raise RuntimeError(f"{arm} correction log incomplete")
    indexed = {(r.get("epoch"), r.get("cap_divisor")): r for r in correction_rows}
    wanted = {(epoch, cap) for epoch in CORRECTIONS for cap in
              ((divisor,) if divisor is not None else (10, 20))}
    if len(indexed) != len(correction_rows) or set(indexed) != wanted:
        raise RuntimeError(f"{arm} correction epochs/caps incomplete or duplicated")
    dual = None
    for epoch_row in epochs:
        epoch = epoch_row["epoch"]
        if epoch == 1:
            expected_model = summary["initial_model_sha256"]
            expected_optimizer = summary["ce_warmup"]["pre_epoch_optimizer_sha256"]
        else:
            previous = epochs[epoch - 2]
            expected_model = previous.get("post_epoch_model_sha256", previous.get("model_sha256"))
            expected_optimizer = previous.get("post_epoch_optimizer_sha256",
                                              previous.get("optimizer_state_sha256"))
        if (epoch_row.get("pre_epoch_model_sha256") != expected_model or
                epoch_row.get("pre_epoch_optimizer_sha256") != expected_optimizer):
            raise RuntimeError(f"{arm}/{epoch} persistent model/optimizer hash continuity differs")
        if epoch == 1 and arm != "focal_clip":
            if epoch_row != summary["ce_warmup"]:
                raise RuntimeError(f"{arm} did not inherit byte-equal CE warm-up")
        else:
            event = _one(events, "epoch", arm, epoch)
            if any(event.get(key) != value for key, value in epoch_row.items()):
                raise RuntimeError(f"{arm}/{epoch} epoch event differs from summary")
        required_hashes = ["sample_order_sha256", "first_batch_sha256", "model_sha256",
                           "optimizer_state_sha256"]
        if epoch > 1 or arm == "focal_clip":
            required_hashes += ["post_epoch_model_sha256", "post_epoch_optimizer_sha256"]
        if epoch_row.get("loss_kind") != expected_loss or any(
                not re.fullmatch("[0-9a-f]{64}", epoch_row.get(key, ""))
                for key in required_hashes):
            raise RuntimeError(f"{arm}/{epoch} training hash missing")
        updates = math.ceil(TRAIN_COUNT / 32)
        if (epoch_row.get("attempted_task_updates") != updates or
                epoch_row.get("applied_task_updates") != updates or
                epoch_row.get("skipped_task_updates") != 0 or
                epoch_row.get("base_lr") != 1e-4 * .8 ** ((epoch - 1) // 5)):
            raise RuntimeError(f"{arm}/{epoch} task dose differs")
        for field in ("training_loss", "stop_loss", "max_task_gradient_norm"):
            _finite(epoch_row.get(field), f"{arm}/{epoch} {field}", minimum=0)
        if epoch in CORRECTIONS:
            epoch_dual = dual
            for cap in ((divisor,) if divisor is not None else (10, 20)):
                correction_row = indexed[epoch, cap]
                event = _one(events, "correction", arm, epoch, cap)
                if any(event.get(key) != value for key, value in correction_row.items()):
                    raise RuntimeError(f"{arm}/{epoch}/{cap} correction event differs")
                if correction_row["optimizer_before_sha256"] != epoch_row["optimizer_state_sha256"]:
                    raise RuntimeError(f"{arm}/{epoch}/{cap} task optimizer changed before correction")
                if correction_row["model_before_sha256"] != epoch_row["model_sha256"]:
                    raise RuntimeError(f"{arm}/{epoch}/{cap} task model differs before correction")
                if method in ("tralo", "phr") and summary.get("pilot"):
                    if sample_ids is None:
                        raise RuntimeError("pilot pre-correction sample IDs missing")
                    _pre_correction_snapshot(directory, arm, epoch, correction_row,
                                             sample_ids,
                                             (correction_row["controller"].get("dual_before")
                                              if method == "phr" else None))
                elif correction_row.get("pre_correction_snapshot") is not None:
                    raise RuntimeError("unexpected pre-correction artifact outside treated pilot")
                if method == "phr":
                    dual = _audit_correction(correction_row, quotas[str(cap)], method, epoch_dual)
                else:
                    _audit_correction(correction_row, quotas[str(cap)],
                                      method if method == "tralo" else "null", None)
            last = indexed[epoch, divisor] if divisor is not None else indexed[epoch, 20]
            if (epoch_row["post_epoch_model_sha256"] != last["model_after_sha256"] or
                    epoch_row["post_epoch_optimizer_sha256"] != last["optimizer_after_sha256"]):
                raise RuntimeError(f"{arm}/{epoch} post-correction state differs")
        if epoch in SNAPSHOTS:
            record = epoch_row.get("post_correction_snapshot")
            if not isinstance(record, dict) or any(
                    _one(events, "snapshot", arm, epoch).get(key) != value
                    for key, value in record.items()):
                raise RuntimeError(f"{arm}/{epoch} snapshot event differs")
            expected_dual = dual if method == "phr" else None
            probabilities = _snapshot(directory, arm, epoch, record, n, expected_dual)
            predictions[epoch] = probabilities
            if epoch_row["post_epoch_model_sha256"] != record["model_state_sha256"] or (
                    epoch_row["post_epoch_optimizer_sha256"] != record["optimizer_state_sha256"]):
                raise RuntimeError(f"{arm}/{epoch} post-correction state differs")
            if method in ("tralo", "phr"):
                _audit_scopes(indexed[epoch, divisor]["after"], quotas[str(divisor)],
                              f"{arm}/{epoch} after",
                              actual=_scopes(probabilities, groups, quotas[str(divisor)]))
            else:
                for cap in (10, 20):
                    _audit_scopes(indexed[epoch, cap]["after"], quotas[str(cap)],
                                  f"{arm}/{epoch}/{cap} after",
                                  actual=_scopes(probabilities, groups, quotas[str(cap)]))
        if epoch == EPOCHS[-1]:
            if (row.get("final_model_sha256") != epoch_row["post_epoch_model_sha256"] or
                    row.get("final_optimizer_sha256") != epoch_row["post_epoch_optimizer_sha256"]):
                raise RuntimeError(f"{arm} final training state differs")
    return predictions


def audit_seed(directory, data_root, *, data_hashes=None, expected_seed=None,
               reference=False, replay_device=None):
    """Audit one run without opening development labels; returns saved predictions."""
    hashes = _data_hashes(data_root) if data_hashes is None else data_hashes
    config, manifest, summary, events, pool = _audit_identity(directory, data_root, hashes)
    seed = config["seed"]
    if expected_seed is not None and seed != expected_seed:
        raise RuntimeError("seed directory/config mismatch")
    if config["reference"] is not reference:
        raise RuntimeError("reference identity differs")
    arms = ("ce_null",) if reference else ARMS + (PILOT_NULLS if config["pilot"] else ())
    if (set(summary.get("arms", {})) != set(arms) or
            set(_one(events, "completed").get("arms", [])) != set(arms)):
        raise RuntimeError("arm set differs from fixed protocol")
    warm = summary.get("ce_warmup")
    event = _one(events, "ce_warmup_completed")
    if (not isinstance(warm, dict) or any(event.get(key) != value for key, value in warm.items()) or
            warm.get("epoch") != 1 or warm.get("loss_kind") != "ce" or
            warm.get("applied_task_updates") != math.ceil(TRAIN_COUNT / 32)):
        raise RuntimeError("CE warm-up event or dose differs")
    groups = [row["location"] for row in pool]
    quotas = manifest["quotas"]
    sample_ids = [row["sample_id"] for row in pool]
    snapshots = {arm: _audit_arm(directory, arm, summary, events, groups, quotas,
                                 sample_ids)
                 for arm in arms}
    for epoch in CORRECTIONS:
        comparable = [summary["arms"][arm]["epochs"][epoch - 1]
                      for arm in arms if arm != "focal_clip"]
        if len({(r["sample_order_sha256"], r["first_batch_sha256"]) for r in comparable}) != 1:
            raise RuntimeError(f"epoch {epoch} arm sample/augmentation streams differ")
        focal = summary["arms"].get("focal_clip")
        if focal is not None:
            r = focal["epochs"][epoch - 1]
            if (r["sample_order_sha256"], r["first_batch_sha256"]) != (
                    comparable[0]["sample_order_sha256"], comparable[0]["first_batch_sha256"]):
                raise RuntimeError(f"epoch {epoch} focal sample/augmentation stream differs")
    if not reference:
        focal1 = summary["arms"]["focal_clip"]["epochs"][0]
        if (focal1["loss_kind"] != "focal" or
                focal1["sample_order_sha256"] != warm["sample_order_sha256"] or
                focal1["first_batch_sha256"] != warm["first_batch_sha256"] or
                focal1["model_sha256"] == warm["model_sha256"]):
            raise RuntimeError("focal epoch 1 did not branch independently")
    launch = (_audit_launch(directory, config, manifest, events)
              if replay_device is not None else None)
    fingerprints = (_audit_training_fingerprints(directory, config, summary, launch)
                    if replay_device is not None else None)
    replay = (_replay_seed(directory, data_root, summary, snapshots,
                            manifest["split"], replay_device)
              if replay_device is not None else None)
    if replay is not None and replay["host"].split(".")[0] != launch["host"].split(".")[0]:
        raise RuntimeError("launch host differs from independent replay host")
    return {"seed": seed, "config": config, "manifest": manifest,
             "summary": summary, "events": events, "pool": pool,
             "snapshots": snapshots, "replay": replay, "launch": launch,
             "training_preflight": fingerprints,
             "root": str(directory)}


def _assert_equal_trajectory(left, right, arm_left="ce_null", arm_right="ce_null"):
    a, b = left["summary"]["arms"][arm_left], right["summary"]["arms"][arm_right]
    if left["summary"]["initial_model_sha256"] != right["summary"]["initial_model_sha256"]:
        raise RuntimeError("null/reference initial model differs")
    if left["summary"]["ce_warmup"] != right["summary"]["ce_warmup"]:
        raise RuntimeError("null/reference CE warm-up differs")
    for epoch in CORRECTIONS:
        x, y = a["epochs"][epoch - 1], b["epochs"][epoch - 1]
        for key in ("sample_order_sha256", "first_batch_sha256", "training_loss",
                    "stop_loss", "model_sha256", "optimizer_state_sha256",
                    "post_epoch_model_sha256", "post_epoch_optimizer_sha256"):
            if x[key] != y[key]:
                raise RuntimeError(f"null/reference epoch {epoch} {key} differs")
    for epoch in SNAPSHOTS:
        if not torch.equal(left["snapshots"][arm_left][epoch], right["snapshots"][arm_right][epoch]):
            raise RuntimeError(f"null/reference epoch {epoch} probabilities differ")


def _write_new(path, report):
    target = Path(path)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _recount_cost_registry(cost_root, required_run_roots=(), *, inflight_gate_root=None,
                           inflight_pilot_gate=None):
    """Recount every queue attempt, including failed smokes and unresolved work."""
    root = Path(cost_root).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("persistent study cost registry missing or linked")
    entries, seen_roots, gpu_seconds = [], set(), 0.
    required = {str(Path(p).resolve()) for p in required_run_roots}
    release_sources = {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest_path = directory / "attempt.json"
        preflight_start = directory / "preflight.start.json"
        preflight_path = directory / "preflight.json"
        if any(not path.is_file() or path.is_symlink()
               for path in (manifest_path, preflight_start, preflight_path)):
            raise RuntimeError(f"unresolved persistent cost attempt: {directory}")
        manifest, preflight = _json(manifest_path), _json(preflight_path)
        run_root = manifest.get("run_root")
        release = manifest.get("release_commit")
        if release not in release_sources:
            release_sources[release] = _source_at_release(release)
        historical_source = release_sources[release]
        if (not isinstance(run_root, str) or not Path(run_root).is_absolute() or
                 run_root in seen_roots or manifest.get("study") != RECIPE["study"] or
                manifest.get("mode") not in ("pilot-step", "pilot-ref", "full") or
                not isinstance(manifest.get("gpu_uuid"), str) or
                not manifest["gpu_uuid"].startswith("GPU-") or
                preflight.get("run_root") != run_root or
                 preflight.get("source_sha256") != historical_source or
                 preflight.get("passed") not in (True, False)):
            raise RuntimeError(f"persistent cost/data preflight identity differs: {directory}")
        if run_root in required and (release not in
                                    {PILOT_RUNNER_RELEASE,
                                     _scorer_identity()["release_commit"]} or
                                      historical_source != source() or
                                      preflight.get("data_files") != FILES):
            raise RuntimeError(f"required current preflight release/data differs: {directory}")
        if _json(preflight_start).get("run_root") != run_root:
            raise RuntimeError(f"preflight start root differs: {directory}")
        if not preflight["passed"] and not isinstance(preflight.get("error"), str):
            raise RuntimeError(f"failed preflight lacks preserved error: {directory}")
        if preflight["passed"]:
            expected_seeds = ({str(seed) for seed in SEEDS} if manifest["mode"] == "full"
                              else {str(PILOT)})
            first_batches = preflight.get("first_batches")
            if (preflight.get("train_count") != TRAIN_COUNT or
                    preflight.get("stop_count") != STOP_COUNT or
                    preflight.get("development_count") != POOL_COUNT or
                    preflight.get("reserved_images") != TEST_COUNT - POOL_COUNT or
                     preflight.get("pretrained_weight", {}).get("sha256") != WEIGHT_SHA or
                     not isinstance(preflight.get("data_files"), dict) or
                    preflight.get("preprocessing") != PREPROCESSING or
                    not isinstance(preflight.get("quotas"), dict) or
                    not re.fullmatch(r"[0-9a-f]{64}", str(preflight.get(
                        "train_transform_probe_sha256", ""))) or
                    not re.fullmatch(r"[0-9a-f]{64}", str(preflight.get(
                        "evaluation_transform_probe_sha256", ""))) or
                    not isinstance(first_batches, dict) or
                    set(first_batches) != expected_seeds or
                    any(not isinstance(first_batches[seed], dict) or
                            set(first_batches[seed]) != {str(epoch) for epoch in EPOCHS} or
                            any(not all(re.fullmatch(r"[0-9a-f]{64}", str(
                                first_batches[seed][str(epoch)].get(key, ""))) for key in
                                ("sample_order_sha256", "first_batch_sha256"))
                                for epoch in EPOCHS)
                            for seed in expected_seeds)):
                raise RuntimeError(f"real-data preflight counts/weights differ: {directory}")
        seen_roots.add(run_root)
        smoke_start = directory / "smoke.start.json"
        smoke_path = directory / "smoke.json"
        smoke = None
        if preflight["passed"]:
            if (not smoke_start.is_file() or not smoke_path.is_file() or
                    smoke_start.is_symlink() or smoke_path.is_symlink()):
                raise RuntimeError(f"unresolved persistent CUDA smoke: {directory}")
            smoke = _json(smoke_path)
            if (smoke.get("run_root") != run_root or
                    smoke.get("gpu_uuid") != manifest["gpu_uuid"] or
                     smoke.get("source_sha256") != historical_source or
                    smoke.get("passed") not in (True, False)):
                raise RuntimeError(f"persistent CUDA smoke identity differs: {directory}")
            if _json(smoke_start).get("run_root") != run_root:
                raise RuntimeError(f"CUDA smoke start root differs: {directory}")
            if smoke["passed"]:
                if (smoke.get("precision") != "fp32_tf32_off" or
                        not isinstance(smoke.get("gpu_name"), str) or
                        not smoke["gpu_name"]):
                    raise RuntimeError(f"CUDA smoke GPU/precision differs: {directory}")
                for key in ("ce_loss", "focal_loss", "ce_backbone_grad_norm",
                            "focal_backbone_grad_norm"):
                    _finite(smoke.get(key), f"CUDA smoke {key}",
                            minimum=1e-20 if key.endswith("grad_norm") else 0)
            gpu_seconds += _finite(smoke.get("elapsed_seconds"),
                                   "CUDA smoke seconds", minimum=0)
        elif smoke_start.exists() or smoke_path.exists():
            raise RuntimeError(f"failed preflight must precede CUDA use: {directory}")
        gate_start, gate_done = directory / "fresh_gate.start.json", directory / "fresh_gate.json"
        gate_seconds = 0.
        if gate_start.exists():
            if not gate_start.is_file() or gate_start.is_symlink():
                raise RuntimeError(f"invalid full-gate start receipt: {directory}")
            if not gate_done.is_file():
                if run_root != inflight_gate_root:
                    raise RuntimeError(f"unresolved full-gate GPU attempt: {directory}")
            else:
                if gate_done.is_symlink():
                    raise RuntimeError(f"linked full-gate completion receipt: {directory}")
                gate_receipt = _json(gate_done)
                if gate_receipt.get("run_root") != run_root or gate_receipt.get("gpu_uuid") != manifest["gpu_uuid"]:
                    raise RuntimeError(f"full-gate cost identity differs: {directory}")
                gate_seconds = _finite(gate_receipt.get("elapsed_seconds"),
                                       "full-gate GPU seconds", minimum=0)
                gpu_seconds += gate_seconds
        elif gate_done.exists():
            raise RuntimeError(f"full-gate completion has no start receipt: {directory}")
        entries.append({"run_root": run_root, "mode": manifest["mode"],
                        "release_commit": release,
                        "host": manifest.get("host"), "gpu_uuid": manifest["gpu_uuid"],
                        "preflight_passed": preflight["passed"],
                        "smoke_passed": smoke["passed"] if smoke is not None else None,
                        "smoke_seconds": smoke["elapsed_seconds"] if smoke is not None else 0.,
                        "fresh_gate_seconds": gate_seconds,
                        "attempt_sha256": _hash(manifest_path),
                        "preflight_start_sha256": _hash(preflight_start),
                        "preflight_sha256": _hash(preflight_path),
                        "smoke_start_sha256": _hash(smoke_start) if smoke is not None else None,
                        "smoke_sha256": _hash(smoke_path) if smoke is not None else None,
                        "fresh_gate_start_sha256": _hash(gate_start) if gate_start.exists() else None,
                        "fresh_gate_sha256": _hash(gate_done) if gate_done.exists() else None})
    if not entries:
        raise RuntimeError("no registered persistent study GPU attempts")
    for path in required:
        matches = [row for row in entries if row["run_root"] == path]
        if len(matches) != 1 or not matches[0]["preflight_passed"] or not matches[0]["smoke_passed"]:
            raise RuntimeError(f"missing successful queue preflight/smoke for {path}")
    pilot_gate_receipts = []
    for marker in sorted(root.glob("pilot_gate_*.start.json")):
        if not marker.is_file() or marker.is_symlink():
            raise RuntimeError("invalid registered pilot-gate start receipt")
        done = root / marker.name.replace(".start.json", ".json")
        started = _json(marker)
        matching = [entry for entry in entries if entry["run_root"] ==
                    str(Path(started.get("pilot_root", "/missing")).parent)]
        references = [entry for entry in entries if entry["run_root"] ==
                      str(Path(started.get("reference_root", "/missing")).parent)]
        if (len(matching) != 1 or len(references) != 1 or
                matching[0]["mode"] != "pilot-step" or
                references[0]["mode"] != "pilot-ref" or
                matching[0]["release_commit"] != references[0]["release_commit"] or
                matching[0]["host"] != references[0]["host"] or
                started.get("source_sha256") != release_sources[matching[0]["release_commit"]] or
                started.get("pilot_root") is None or
                started.get("reference_root") is None or
                not str(started.get("gpu_uuid", "")).startswith("GPU-")):
            raise RuntimeError("pilot-gate attempt identity differs")
        row = {"start_path": str(marker), "start_sha256": _hash(marker),
               "completion_sha256": None, "elapsed_seconds": 0.}
        if done.exists():
            if not done.is_file() or done.is_symlink():
                raise RuntimeError("invalid pilot-gate completion receipt")
            completed = _json(done)
            if any(completed.get(key) != started[key] for key in
                   ("source_sha256", "pilot_root", "reference_root", "gpu_uuid")):
                raise RuntimeError("pilot-gate completion identity differs")
            elapsed = _finite(completed.get("elapsed_seconds"),
                              "pilot-gate GPU attempt seconds", minimum=0)
            gpu_seconds += elapsed
            row.update(completion_sha256=_hash(done), elapsed_seconds=elapsed,
                       passed=completed.get("passed"))
        elif str(marker) != inflight_pilot_gate:
            raise RuntimeError("unresolved pilot-gate GPU attempt")
        pilot_gate_receipts.append(row)
    return {"root": str(root), "entries": entries,
            "pilot_gate_receipts": pilot_gate_receipts,
            "gpu_seconds": gpu_seconds}


def _gate_core(pilot_root, reference_root, data_root, output=None, *, replay_device,
               prior_hours=0., ceiling_hours=24., cost_root=None,
               required_run_roots=(), inflight_gate_root=None,
               inflight_pilot_gate=None):
    """Pilot check; never reads test_labels.npy or a development label column."""
    if not isinstance(replay_device, str) or not replay_device.startswith("cuda:"):
        raise RuntimeError("explicit exclusive CUDA replay device required")
    cost = (_recount_cost_registry(cost_root, [Path(pilot_root).parent,
                                              Path(reference_root).parent,
                                              *required_run_roots],
                                         inflight_gate_root=inflight_gate_root,
                                         inflight_pilot_gate=inflight_pilot_gate)
            if cost_root is not None else None)
    prior_hours = _finite(prior_hours, "prior GPU hours", minimum=0) + (
        cost["gpu_seconds"] / 3600 if cost is not None else 0.)
    hashes = _data_hashes(data_root)
    pilot = audit_seed(pilot_root, data_root, data_hashes=hashes,
                       expected_seed=PILOT, reference=False, replay_device=replay_device)
    reference = audit_seed(reference_root, data_root, data_hashes=hashes,
                           expected_seed=PILOT, reference=True, replay_device=replay_device)
    if (pilot["pool"] != reference["pool"] or
            pilot["manifest"]["split_sha256"] != reference["manifest"]["split_sha256"] or
            {k: v for k, v in pilot["config"].items() if k != "reference"} !=
            {k: v for k, v in reference["config"].items() if k != "reference"}):
        raise RuntimeError("pilot/reference cohort or scientific config differs")
    if (pilot["launch"]["release_commit"] != reference["launch"]["release_commit"] or
             pilot["launch"]["host"] != reference["launch"]["host"] or
             pilot["replay"]["host"] != reference["replay"]["host"]):
        raise RuntimeError("pilot/reference must use identical release bytes and host")
    scorer_identity = _scorer_identity()
    if (pilot["launch"]["release_commit"] != PILOT_RUNNER_RELEASE or
            _source_at_release(PILOT_RUNNER_RELEASE) != source()):
        raise RuntimeError("pilot runner release source differs from scorer source")
    _assert_equal_trajectory(pilot, reference)
    for arm in PILOT_NULLS:
        _assert_equal_trajectory(pilot, pilot, "ce_null", arm)
    pilot_seconds = _finite(pilot["summary"].get("elapsed_seconds"), "pilot seconds", minimum=0)
    reference_seconds = _finite(reference["summary"].get("elapsed_seconds"),
                                "reference seconds", minimum=0)
    replay_pilot = _finite(pilot["replay"]["seconds"], "pilot replay seconds", minimum=0)
    replay_ref = _finite(reference["replay"]["seconds"], "reference replay seconds", minimum=0)
    # The pilot includes two extra method-named nulls; projecting each full
    # seed at its entire pilot cost is conservative. Replay GPU time is charged.
    projected = prior_hours + (pilot_seconds + reference_seconds + replay_pilot +
                               replay_ref + 12 * (pilot_seconds + replay_pilot)) / 3600
    if projected > ceiling_hours:
        raise RuntimeError("persistent study projected GPU-hours exceed ceiling")
    report = {"status": "pilot_integrity_pass", "seed": PILOT,
              "development_labels_accessed": False, "reference_equal": True,
              "method_named_nulls_equal": True,
               "source_sha256": source(), "scorer_identity": scorer_identity,
               "data_files": hashes,
              "split_sha256": pilot["manifest"]["split_sha256"],
              "pilot_manifest_sha256": _hash(Path(pilot_root) / "manifest.json"),
              "reference_manifest_sha256": _hash(Path(reference_root) / "manifest.json"),
              "pilot_summary_sha256": _hash(Path(pilot_root) / "summary.json"),
              "reference_summary_sha256": _hash(Path(reference_root) / "summary.json"),
              "pilot_seconds": pilot_seconds, "reference_seconds": reference_seconds,
               "pilot_replay": pilot["replay"], "reference_replay": reference["replay"],
               "pilot_launch": pilot["launch"], "reference_launch": reference["launch"],
               "pilot_root": str(Path(pilot_root).resolve()),
               "reference_root": str(Path(reference_root).resolve()),
               "cost_registry": cost,
               "prior_gpu_hours": prior_hours, "projected_gpu_hours": projected,
              "ceiling_gpu_hours": ceiling_hours}
    if output is not None:
        _write_new(output, report)
    return report


def gate(pilot_root, reference_root, data_root, output=None, *, replay_device,
         prior_hours=0., ceiling_hours=24., cost_root=None,
         required_run_roots=(), inflight_gate_root=None,
         register_gate_attempt=True):
    """Label-blind gate; register every production GPU replay attempt."""
    if cost_root is None or not register_gate_attempt:
        return _gate_core(pilot_root, reference_root, data_root, output,
                          replay_device=replay_device, prior_hours=prior_hours,
                          ceiling_hours=ceiling_hours, cost_root=cost_root,
                          required_run_roots=required_run_roots,
                          inflight_gate_root=inflight_gate_root)
    cost_path = Path(cost_root).resolve(strict=True)
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", uuid):
        raise RuntimeError("pilot gate requires a physical UUID-pinned GPU")
    marker = cost_path / f"pilot_gate_{time.time_ns()}_{os.getpid()}.start.json"
    started = {"pilot_root": str(Path(pilot_root).resolve()),
               "reference_root": str(Path(reference_root).resolve()),
               "source_sha256": source(), "gpu_uuid": uuid}
    _write_new(marker, started)
    begun = time.perf_counter()
    passed = False
    try:
        report = _gate_core(pilot_root, reference_root, data_root, output,
                            replay_device=replay_device, prior_hours=prior_hours,
                            ceiling_hours=ceiling_hours, cost_root=cost_root,
                            required_run_roots=required_run_roots,
                            inflight_gate_root=inflight_gate_root,
                            inflight_pilot_gate=str(marker))
        passed = True
        return report
    finally:
        completion = {**started, "passed": passed,
                      "elapsed_seconds": time.perf_counter() - begun}
        _write_new(cost_path / marker.name.replace(".start.json", ".json"), completion)


def _verified_gate_receipt(path, expected_hashes):
    receipt = _json(path)
    for key in ("prior_gpu_hours", "pilot_seconds", "reference_seconds"):
        _finite(receipt.get(key), key, minimum=0)
    for key in ("pilot_replay", "reference_replay"):
        replay = receipt.get(key)
        if not isinstance(replay, dict):
            raise RuntimeError("gate replay cost record missing")
        _finite(replay.get("seconds"), f"{key} seconds", minimum=0)
    if (receipt.get("status") != "pilot_integrity_pass" or
            receipt.get("seed") != PILOT or
            receipt.get("development_labels_accessed") is not False or
            receipt.get("reference_equal") is not True or
            receipt.get("method_named_nulls_equal") is not True or
             receipt.get("source_sha256") != source() or
             receipt.get("scorer_identity") != _scorer_identity() or
            receipt.get("data_files") != expected_hashes or
            not isinstance(receipt.get("pilot_root"), str) or
            not isinstance(receipt.get("reference_root"), str) or
            not isinstance(receipt.get("cost_registry"), dict) or
            not isinstance(receipt["cost_registry"].get("entries"), list) or
            not isinstance(receipt["cost_registry"].get("pilot_gate_receipts"), list) or
            not isinstance(receipt.get("split_sha256"), str) or
            receipt.get("ceiling_gpu_hours") != 24 or
            _finite(receipt.get("projected_gpu_hours"), "gate projection", minimum=0) >
            _finite(receipt.get("ceiling_gpu_hours"), "gate ceiling", minimum=0)):
        raise RuntimeError("missing or invalid label-blind pilot gate receipt")
    return receipt


def recount_full_gate(old_gate_path, pilot_root, reference_root, data_root,
                      cost_root, full_run_root, output=None, *, replay_device):
    """Fresh label-blind pilot replay and all-attempt cost recount before full claims."""
    hashes = _data_hashes(data_root)
    old = _verified_gate_receipt(old_gate_path, hashes)
    pilot_root = str(Path(pilot_root).resolve())
    reference_root = str(Path(reference_root).resolve())
    full_run_root = str(Path(full_run_root).resolve())
    cost_root = str(Path(cost_root).resolve())
    if (old["pilot_root"] != pilot_root or old["reference_root"] != reference_root or
            old["cost_registry"].get("root") != cost_root):
        raise RuntimeError("stale pilot gate root or cost registry path")
    for root, prefix in ((pilot_root, "pilot"), (reference_root, "reference")):
        for name in ("manifest", "summary"):
            if old.get(f"{prefix}_{name}_sha256") != _hash(Path(root) / f"{name}.json"):
                raise RuntimeError("stale pilot gate artifact bytes")
        if old.get(f"{prefix}_seconds") != _json(Path(root) / "summary.json").get("elapsed_seconds"):
            raise RuntimeError("stale pilot gate training time")
    cost = _recount_cost_registry(cost_root, [Path(pilot_root).parent,
                                              Path(reference_root).parent,
                                              full_run_root],
                                  inflight_gate_root=full_run_root)
    current = {row["run_root"]: row for row in cost["entries"]}
    for prior in old["cost_registry"]["entries"]:
        if not isinstance(prior, dict) or current.get(prior.get("run_root")) != prior:
            raise RuntimeError("stale or forged cost attempt receipt")
    full_attempt = _json(Path(cost_root) / hashlib.sha256(
        full_run_root.encode()).hexdigest() / "attempt.json")
    if (old["pilot_launch"]["release_commit"] != PILOT_RUNNER_RELEASE or
            old["reference_launch"]["release_commit"] != PILOT_RUNNER_RELEASE or
            full_attempt.get("run_root") != full_run_root or
            full_attempt.get("release_commit") != _scorer_identity()["release_commit"] or
            _source_at_release(PILOT_RUNNER_RELEASE) != source()):
        raise RuntimeError("pilot/full runner source or scorer release differs")
    current_gates = {row["start_path"]: row for row in cost["pilot_gate_receipts"]}
    if not old["cost_registry"]["pilot_gate_receipts"]:
        raise RuntimeError("original pilot replay was not registered in GPU cost ledger")
    for prior in old["cost_registry"]["pilot_gate_receipts"]:
        latest = current_gates.get(prior.get("start_path"))
        if latest is None or latest.get("start_sha256") != prior.get("start_sha256"):
            raise RuntimeError("stale or forged pilot replay cost receipt")
        if prior.get("completion_sha256") is None:
            if (latest.get("completion_sha256") is None or
                    latest.get("passed") is not True):
                raise RuntimeError("original pilot replay never completed successfully")
        elif latest != prior:
            raise RuntimeError("prior pilot replay cost receipt changed")
    historical_manual_hours = (old["prior_gpu_hours"] -
                               old["cost_registry"]["gpu_seconds"] / 3600)
    if historical_manual_hours < 0:
        raise RuntimeError("negative manually preserved prior cost")
    fresh = gate(pilot_root, reference_root, data_root, replay_device=replay_device,
                  prior_hours=historical_manual_hours,
                  cost_root=cost_root, required_run_roots=[full_run_root],
                  inflight_gate_root=full_run_root, register_gate_attempt=False)
    for key in ("source_sha256", "scorer_identity", "data_files", "split_sha256",
                "pilot_manifest_sha256", "reference_manifest_sha256",
                "pilot_summary_sha256", "reference_summary_sha256",
                "pilot_seconds", "reference_seconds", "pilot_launch", "reference_launch"):
        if fresh[key] != old[key]:
            raise RuntimeError(f"fresh pilot gate changed {key}")
    fresh["status"] = "full_dispatch_fresh_integrity_pass"
    fresh["old_gate_sha256"] = _hash(old_gate_path)
    fresh["old_gate_path"] = str(Path(old_gate_path).resolve())
    fresh["full_run_root"] = full_run_root
    if output is not None:
        _write_new(output, fresh)
    return fresh


def _verified_full_gate_receipt(path, expected_hashes, full_run_root):
    """Require the fresh, root-bound dispatch gate before opening full labels."""
    root = Path(full_run_root).resolve()
    if Path(path).resolve() != root / "fresh_full_gate.json":
        raise RuntimeError("full score requires queue-owned fresh gate receipt")
    receipt = _json(path)
    if (receipt.get("status") != "full_dispatch_fresh_integrity_pass" or
            receipt.get("full_run_root") != str(root) or
             receipt.get("source_sha256") != source() or
             receipt.get("scorer_identity") != _scorer_identity() or
            receipt.get("data_files") != expected_hashes or
            receipt.get("development_labels_accessed") is not False or
            receipt.get("reference_equal") is not True or
            receipt.get("method_named_nulls_equal") is not True or
            receipt.get("ceiling_gpu_hours") != 24 or
            _finite(receipt.get("projected_gpu_hours"), "fresh full projection", minimum=0) > 24):
        raise RuntimeError("missing or invalid fresh full gate")
    old_path = receipt.get("old_gate_path")
    if (not isinstance(old_path, str) or
            _hash(old_path) != receipt.get("old_gate_sha256")):
        raise RuntimeError("original pilot gate changed after fresh recount")
    cost_record = receipt.get("cost_registry")
    if not isinstance(cost_record, dict):
        raise RuntimeError("fresh full gate lacks attempted GPU cost recount")
    current = _recount_cost_registry(cost_record["root"],
                                     [Path(receipt["pilot_root"]).parent,
                                      Path(receipt["reference_root"]).parent,
                                      root])
    now = {row["run_root"]: row for row in current["entries"]}
    for earlier in cost_record["entries"]:
        latest = now.get(earlier.get("run_root"))
        if latest is None:
            raise RuntimeError("fresh full gate attempt disappeared")
        for key, value in earlier.items():
            if key in ("fresh_gate_sha256", "fresh_gate_seconds") and value in (None, 0.) and latest[key] not in (None, 0.):
                continue  # This completion is written immediately after the gate.
            if latest.get(key) != value:
                raise RuntimeError("fresh full gate cost evidence changed")
    completed_gates = {row["start_path"]: row for row in current["pilot_gate_receipts"]}
    for earlier in cost_record["pilot_gate_receipts"]:
        latest = completed_gates.get(earlier.get("start_path"))
        if latest is None or latest.get("start_sha256") != earlier.get("start_sha256"):
            raise RuntimeError("pilot replay cost evidence changed")
        if earlier.get("completion_sha256") is None:
            if latest.get("completion_sha256") is None or latest.get("passed") is not True:
                raise RuntimeError("pilot replay completion missing")
        elif latest != earlier:
            raise RuntimeError("pilot replay cost completion changed")
    for prefix in ("pilot", "reference"):
        pilot_root = Path(receipt[f"{prefix}_root"])
        for artifact in ("manifest", "summary"):
            if _hash(pilot_root / f"{artifact}.json") != receipt[f"{prefix}_{artifact}_sha256"]:
                raise RuntimeError("pilot evidence changed since full dispatch gate")
    return receipt


def _development_labels(data_root, pool):
    """Only call after all label-blind seed integrity checks have succeeded."""
    path = Path(data_root) / "test_labels.npy"
    if _hash(path) != FILES["test_labels.npy"]:
        raise RuntimeError("development label file hash differs")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    indices = [int(row["sample_id"].removeprefix("test")) for row in pool]
    if tuple(array.shape) != (TEST_COUNT,) or len(set(indices)) != len(indices):
        raise RuntimeError("development label alignment differs")
    labels = [int(array[i]) for i in indices]
    if any(not 0 <= y < base.CLASSES for y in labels):
        raise RuntimeError("development label outside declared classes")
    return labels


def _metric_row(probabilities, labels, ids, groups, quota):
    score = base._arm_score(probabilities, labels, ids, groups, quota)
    assigned = score.pop("predictions")
    tp = sum(p == base.CAPPED and y == base.CAPPED for p, y in zip(assigned, labels))
    positive = sum(y == base.CAPPED for y in labels)
    predicted = sum(p == base.CAPPED for p in assigned)
    score["constrained_precision"] = tp / predicted if predicted else 0.
    score["constrained_recall"] = tp / positive if positive else 0.
    score["constrained_support"] = positive
    score["uncapped_class_f1"] = _per_class_f1(labels, assigned)
    return score, assigned


def _per_class_f1(labels, predictions):
    from sklearn.metrics import f1_score
    values = f1_score(labels, predictions, labels=list(range(base.CLASSES)),
                      average=None, zero_division=0)
    return {str(c): float(values[c]) for c in range(base.CLASSES) if c != base.CAPPED}


def _slot_movement(treated, control, labels):
    entries = [i for i, (x, y) in enumerate(zip(treated, control))
               if x == base.CAPPED and y != base.CAPPED]
    exits = [i for i, (x, y) in enumerate(zip(treated, control))
             if x != base.CAPPED and y == base.CAPPED]
    return {"entries": len(entries), "exits": len(exits),
            "correct_entries": sum(labels[i] == base.CAPPED for i in entries),
            "correct_exits": sum(labels[i] == base.CAPPED for i in exits)}


def _score_audited(seed, labels):
    ids = [r["sample_id"] for r in seed["pool"]]
    groups = [r["location"] for r in seed["pool"]]
    out = {"seed": seed["seed"], "caps": {}, "checkpoint_replay": seed["replay"],
           "manifest_sha256": _hash(Path(seed["root"]) / "manifest.json")}
    for divisor in (10, 20):
        quota = seed["manifest"]["quotas"][str(divisor)]
        names = ("ce_null", "focal_clip", f"cap{divisor}_tralo", f"cap{divisor}_phr")
        arms, assignments = {}, {}
        for name in names:
            ens = torch.stack([seed["snapshots"][name][e] for e in SNAPSHOTS]).mean(0)
            arms[name], assignments[name] = _metric_row(ens, labels, ids, groups, quota)
        out["caps"][str(divisor)] = {"quota": quota, "arms": arms,
            "tralo_vs_ce_slots": _slot_movement(assignments[f"cap{divisor}_tralo"],
                                                 assignments["ce_null"], labels),
            "tralo_vs_phr_slots": _slot_movement(assignments[f"cap{divisor}_tralo"],
                                                  assignments[f"cap{divisor}_phr"], labels),
            "training": {name: {"corrections": seed["summary"]["arms"][name]["corrections"],
                                "epochs": seed["summary"]["arms"][name]["epochs"]}
                         for name in names}}
    return out


def _actual_study_hours(gate_record, audited, full_run_root):
    """Recount completed GPU receipts after dispatch, before opening labels."""
    completed = _recount_cost_registry(
        gate_record["cost_registry"]["root"],
        [Path(gate_record["pilot_root"]).parent,
         Path(gate_record["reference_root"]).parent, full_run_root])
    recorded_at_dispatch = _finite(gate_record["cost_registry"].get("gpu_seconds"),
                                   "dispatch registry GPU seconds", minimum=0)
    external_prior = (_finite(gate_record["prior_gpu_hours"],
                              "dispatch prior GPU hours", minimum=0) -
                      recorded_at_dispatch / 3600)
    if external_prior < -1e-8:
        raise RuntimeError("dispatch GPU cost registry exceeds recorded prior hours")
    # The ledger includes every smoke, pilot-gate replay and the *completed*
    # fresh full-gate replay. Training and this final scorer replay are charged
    # separately because their elapsed times live in the run artifacts.
    training_and_scoring_seconds = (
        _finite(gate_record["pilot_seconds"], "pilot training seconds", minimum=0) +
        _finite(gate_record["reference_seconds"], "reference training seconds", minimum=0) +
        sum(_finite(seed["summary"]["elapsed_seconds"], "seed runtime", minimum=0) +
            _finite(seed["replay"]["seconds"], "final scorer replay seconds", minimum=0)
            for seed in audited))
    return (max(0., external_prior) +
            (completed["gpu_seconds"] + training_and_scoring_seconds) / 3600), completed


def pilot_score(pilot_root, data_root, gate_receipt, output=None, *, replay_device):
    hashes = _data_hashes(data_root)
    gate_record = _verified_gate_receipt(gate_receipt, hashes)
    audited = audit_seed(pilot_root, data_root, data_hashes=hashes,
                         expected_seed=PILOT, replay_device=replay_device)
    if (gate_record["pilot_manifest_sha256"] != _hash(Path(pilot_root) / "manifest.json") or
            gate_record["pilot_summary_sha256"] != _hash(Path(pilot_root) / "summary.json") or
            gate_record["split_sha256"] != audited["manifest"]["split_sha256"]):
        raise RuntimeError("pilot changed after label-blind gate")
    labels = _development_labels(data_root, audited["pool"])
    report = {"status": "exploratory_pilot_not_for_setting_selection",
              "scorer_identity": _scorer_identity(),
               "development_labels_accessed_offline": True,
              "seed": _score_audited(audited, labels)}
    if output is not None:
        _write_new(output, report)
    return report


def main(run_root, data_root, gate_receipt, output=None, *, replay_device):
    root = Path(run_root)
    found = {p.name for p in root.glob("seed*") if p.is_dir()}
    expected = {f"seed{s}" for s in SEEDS}
    if found != expected:
        raise RuntimeError(f"persistent block incomplete/extra: missing {sorted(expected-found)}, "
                           f"extra {sorted(found-expected)}")
    hashes = _data_hashes(data_root)
    gate_record = _verified_full_gate_receipt(gate_receipt, hashes, run_root)
    audited = [audit_seed(root / f"seed{s}", data_root, data_hashes=hashes,
                          expected_seed=s, replay_device=replay_device) for s in SEEDS]
    if (gate_record["pilot_launch"]["release_commit"] != PILOT_RUNNER_RELEASE or
            _source_at_release(PILOT_RUNNER_RELEASE) != source() or
            any(row["launch"]["release_commit"] !=
                _scorer_identity()["release_commit"] for row in audited)):
        raise RuntimeError("full seed runner source or release differs")
    if (len({row["manifest"]["split_sha256"] for row in audited}) != 1 or
             audited[0]["manifest"]["split_sha256"] != gate_record["split_sha256"]):
        raise RuntimeError("seeds use different development cohorts")
    # This integrity check uses only audited, label-free CE probabilities.
    pto_digests = []
    for seed in audited:
        ensemble = torch.stack([seed["snapshots"]["ce_null"][epoch]
                                for epoch in SNAPSHOTS]).mean(0)
        pto_digests.append(hashlib.sha256(
            ensemble.contiguous().cpu().numpy().tobytes()).hexdigest())
    if len(set(pto_digests)) != len(pto_digests):
        raise RuntimeError("duplicate CE/null prediction sets across seeds")
    actual_hours, completed_cost = _actual_study_hours(gate_record, audited, root)
    # Opening labels is intentionally after every seed's byte/log/cost audit.
    labels = _development_labels(data_root, audited[0]["pool"])
    rows = [_score_audited(row, labels) for row in audited]
    contrasts = {}
    secondary = {}
    primary = []
    for divisor, treated, control in FAMILY:
        tname = f"cap{divisor}_{treated}"
        cname = f"cap{divisor}_{control}" if control == "phr" else control
        name = f"cap{divisor}_{treated}_minus_{control}"
        primary.append(name)
        deltas = [row["caps"][str(divisor)]["arms"][tname]["allocated"]["cc_f1"] -
                  row["caps"][str(divisor)]["arms"][cname]["allocated"]["cc_f1"]
                  for row in rows]
        contrasts[name] = {**base._paired(deltas),
                           "per_seed": dict(zip(SEEDS, deltas))}
        secondary[name] = {}
        for metric in ("accuracy", "macro_f1", "weighted_f1"):
            values = [row["caps"][str(divisor)]["arms"][tname]["allocated"][metric] -
                      row["caps"][str(divisor)]["arms"][cname]["allocated"][metric]
                      for row in rows]
            secondary[name][metric] = {**base._paired(values),
                                       "per_seed": dict(zip(SEEDS, values))}
    for name, value in zip(primary, base._holm([contrasts[n]["p"] for n in primary])):
        contrasts[name]["holm_p"] = value
    arm_means = {}
    for divisor in (10, 20):
        arm_means[str(divisor)] = {}
        for name in ("ce_null", "focal_clip", f"cap{divisor}_tralo", f"cap{divisor}_phr"):
            arm_means[str(divisor)][name] = {}
            for metric in base.METRICS:
                values = np.asarray([row["caps"][str(divisor)]["arms"][name]["allocated"][metric]
                                     for row in rows], dtype=float)
                arm_means[str(divisor)][name][metric] = {
                    "mean": float(values.mean()), "seed_sd": float(values.std(ddof=1))}
    report = {"status": "complete_12_seed_exploratory_development",
              "provenance": {"source_sha256": source(),
                             "scorer_identity": _scorer_identity(),
                             "data_file_sha256": hashes,
                             "split_sha256": audited[0]["manifest"]["split_sha256"]},
              "seeds": rows, "arm_means": arm_means, "primary_family": primary,
               "primary_contrasts": contrasts, "secondary_contrasts": secondary,
                "scorer_replay_gpu_hours": sum(seed["replay"]["seconds"] for seed in audited) / 3600,
                "completed_cost_registry": completed_cost,
               "actual_study_gpu_hours": actual_hours,
               "cost_ceiling_exceeded": actual_hours > gate_record["ceiling_gpu_hours"],
              "limitations": ["Repeatedly viewed development countries; no geographic confirmation.",
                              "PHR is an inexact persistent primal/dual comparator, not an exact ALM solve."]}
    if output is not None:
        _write_new(output, report)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--fresh-full-gate", action="store_true")
    parser.add_argument("--pilot-score", action="store_true")
    parser.add_argument("--replay-device", required=True,
                         help="Exclusive CUDA device, e.g. cuda:0 after UUID pinning")
    parser.add_argument("--prior-gpu-hours", type=float, default=0.,
                        help="All failed/aborted new-study card-hours before this pilot")
    parser.add_argument("--cost-root", help="Queue-owned all-attempt preflight/smoke registry")
    parser.add_argument("paths", nargs="+")
    arguments = parser.parse_args()
    if (arguments.gate and not arguments.pilot_score and not arguments.fresh_full_gate and
            len(arguments.paths) == 4 and arguments.cost_root):
        print(json.dumps(gate(*arguments.paths, replay_device=arguments.replay_device,
                              prior_hours=arguments.prior_gpu_hours,
                              cost_root=arguments.cost_root), indent=2))
    elif (arguments.fresh_full_gate and not arguments.gate and not arguments.pilot_score and
          len(arguments.paths) == 7):
        print(json.dumps(recount_full_gate(*arguments.paths,
                                           replay_device=arguments.replay_device), indent=2))
    elif arguments.pilot_score and not arguments.gate and not arguments.fresh_full_gate and len(arguments.paths) == 4:
        print(json.dumps(pilot_score(*arguments.paths,
                                     replay_device=arguments.replay_device), indent=2))
    elif not arguments.gate and not arguments.pilot_score and not arguments.fresh_full_gate and len(arguments.paths) == 4:
        result = main(*arguments.paths, replay_device=arguments.replay_device)
        print(json.dumps({"status": result["status"],
                          "primary_contrasts": result["primary_contrasts"]}, indent=2))
    else:
        raise SystemExit(__doc__)
