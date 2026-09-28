"""Fixed fmow2 pooled-plus-country side-step study.

Usage: python -m tralo.fmow_local DATA_ROOT CONFIG_JSON OUTPUT_DIRECTORY
Protocol: experiments/fmow_joint_local_protocol_20260928.md.
Training and stopping are the existing fmow2 Yuval pipeline. Only side copies
receive constraint steps; the PTO training trajectory is never modified.
"""

import copy
import json
import math
from pathlib import Path
import sys
import time

from .fmow_yuval import (ArrayImages, CAPPED, CLASSES, FILES, _state_hash,
                         load, make_model, pool_chunks, transforms_for)
from .global_comparison import audited_arm_log
from .knee_end_to_end import infer
from .knee_experiment import cuda_setup, digest, save, source
from .knee_yuval import SHAM_OFFSET, train_run
from .local_policy import size_share_caps
from .local_targeted_step import local_targeted_step

PILOT = 6099
SEEDS = range(6100, 6148)
DIVISORS = (10, 20)
RECIPE = dict(backbone="mobilenet_v3_large", capped_class=CAPPED,
              max_epochs=75, patience=5, batch_size=32, lr=1e-4,
              weight_decay=1e-4, decay_epoch=5, decay_factor=0.8,
              development_batch_size=16)


def validate(config):
    if not isinstance(config, dict) or set(config) != set(RECIPE) | {"seed", "snapshot_steps"}:
        raise ValueError("config keys differ from the fixed joint-local protocol")
    if type(config["seed"]) is not int or config["seed"] not in SEEDS and config["seed"] != PILOT:
        raise ValueError("seed is outside the fixed study and pilot blocks")
    if type(config["snapshot_steps"]) is not bool or not config["snapshot_steps"] and config["seed"] != PILOT:
        raise ValueError("only the pilot may disable side steps for trajectory parity")
    for key, value in RECIPE.items():
        if config[key] != value or type(config[key]) is not type(value):
            raise ValueError("config differs from the fixed fmow2 recipe: " + key)


def budgets(groups):
    """Unlabeled country sizes determine both fixed-policy cap levels."""
    n = len(groups)
    if n != 1673 or set(groups) != {"IRQ", "NLD", "DZA", "PHL", "TUR"}:
        raise ValueError("unexpected fmow2 development country pool")
    out = {}
    for divisor in DIVISORS:
        global_cap = n // divisor
        local_total = math.ceil(5 * global_cap / 4)
        out[str(divisor)] = dict(global_cap=global_cap,
                                 local_total=local_total,
                                 local_caps=size_share_caps(groups, local_total))
    return out


def snapshot_side_steps(model, pool, groups, quota, seed, epoch, directory):
    """Save joint, same-radius pooled-direction and sham snapshots for one cap."""
    import torch
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    state = _state_hash(model)
    steps = {}
    try:
        for arm in ("joint", "global_dose", "sham"):
            side = copy.deepcopy(model)
            try:
                kwargs = {}
                if arm == "global_dose":
                    kwargs.update(fixed_radius=steps["joint"]["radius"], global_only_direction=True)
                if arm == "sham":
                    kwargs.update(fixed_radius=steps["joint"]["radius"],
                                  sham_generator=torch.Generator().manual_seed(
                                      seed + SHAM_OFFSET + 1000 * epoch))
                if arm != "joint" and not steps["joint"]["applied"]:
                    steps[arm] = dict(applied=False, displacement=0.0)
                else:
                    steps[arm] = local_targeted_step(
                        side, pool, groups, CAPPED, quota["global_cap"],
                        quota["local_caps"], **kwargs)
                artifact = Path(directory) / f"epoch{epoch:02d}_{arm}.pt"
                torch.save(infer(side, pool).cpu(), artifact)
                steps[arm]["probability_sha256"] = digest(artifact)
            finally:
                del side
        if _state_hash(model) != state:
            raise RuntimeError("side steps changed the PTO model")
        if steps["joint"]["applied"] and any(
                abs(steps[arm]["displacement"] - steps["joint"]["displacement"]) > 1e-5
                for arm in ("global_dose", "sham")):
            raise RuntimeError("side-step doses differ")
        if steps["joint"]["applied"]:
            sham_norms = steps["sham"]["tensor_displacement_norms"]
            joint_norms = steps["joint"]["tensor_displacement_norms"]
            if len(sham_norms) != len(joint_norms) or any(
                    abs(sham - joint) > 1e-5 for sham, joint in zip(sham_norms, joint_norms)):
                raise RuntimeError("sham per-tensor doses differ")
        return steps
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def run(data_root, config_path, output):
    import torch
    config = json.loads(Path(config_path).read_text())
    validate(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with audited_arm_log(output / "events.jsonl") as log:
        cuda_setup()
        images, train_labels, pool_rows, roles = load(data_root)
        groups = [row["location"] for row in pool_rows]
        quota = budgets(groups)
        counts = dict(train=len(roles["train"]), stop=len(roles["stop"]), dev=len(roles["dev"]))
        save(output / "manifest.json", dict(files=FILES, counts=counts, quotas=quota,
                                              stop_countries=roles["stop_countries"],
                                              dev_countries=roles["dev_countries"],
                                              reserved_countries=roles["reserved_countries"],
                                              rows=pool_rows))
        save(output / "pool_identity.json", [
            {"sample_id": row["sample_id"], "location": row["location"]}
            for row in pool_rows])
        save(output / "config.json", config)
        log.emit("started", source_sha256=source(), config_sha256=digest(config_path),
                 data_files=FILES, counts=counts, quotas=quota,
                 manifest_sha256=digest(output / "manifest.json"),
                 pool_identity_sha256=digest(output / "pool_identity.json"),
                 device=str(torch.cuda.get_device_name()), precision="fp32")
        train_tf, eval_tf = transforms_for()
        data = ArrayImages(images["train"], roles["train"], train_labels)
        held = ArrayImages(images["train"], roles["stop"], train_labels)
        bs = config["batch_size"]
        stop = [held.batch(list(range(i, min(i + bs, len(held.labels)))), eval_tf)
                for i in range(0, len(held.labels), bs)]
        pool = pool_chunks(images["test"], roles["dev"], eval_tf, config["development_batch_size"])
        torch.manual_seed(config["seed"])
        base = make_model()
        initial_sha = _state_hash(base)
        log.emit("model_initialized", initial_sha256=initial_sha, architecture=config["backbone"],
                 classes=CLASSES, sampler="class-balanced with replacement", early_stop=True,
                 train_label_counts=[data.labels.count(c) for c in range(CLASSES)])
        model = copy.deepcopy(base).cuda()
        directory = output / "retrain1"
        directory.mkdir()
        started = time.monotonic()
        side_steps = {}
        with audited_arm_log(directory / "events.jsonl") as rlog:
            snapshots = {}
            snapshot_hashes = {}

            def snapshot(epoch, probabilities):
                artifact = directory / f"epoch{epoch:02d}.pt"
                torch.save(probabilities.cpu(), artifact)
                snapshot_hashes[str(epoch)] = digest(artifact)
                snapshots[epoch] = probabilities
                if config["snapshot_steps"]:
                    side_steps[str(epoch)] = {}
                    for divisor in DIVISORS:
                        cap_directory = directory / f"cap{divisor}"
                        cap_directory.mkdir(exist_ok=True)
                        step_record = snapshot_side_steps(
                            model, pool, groups, quota[str(divisor)], config["seed"], epoch,
                            cap_directory)
                        side_steps[str(epoch)][str(divisor)] = step_record
                        rlog.emit("snapshot_cap", epoch=epoch, divisor=divisor,
                                  quota=quota[str(divisor)], steps=step_record)

            result = train_run(model, data, stop, pool, config, torch.ones(CLASSES),
                               lambda row: rlog.emit(row["event"], **{
                                   key: value for key, value in row.items() if key != "event"}),
                               snapshot, capped=CAPPED, transforms=(train_tf, eval_tf))
            probabilities = infer(model, pool)
            if not torch.equal(probabilities, snapshots[result["best_epoch"]]):
                raise RuntimeError("restored best PTO weights differ from their snapshot")
            torch.save(probabilities.cpu(), directory / "final_probabilities.pt")
            rlog.emit("training_completed", **result)
        save(output / "summary.json", dict(seed=config["seed"], initial_sha256=initial_sha,
                                           quotas=quota, retrain=result, steps=side_steps,
                                           pto_snapshot_sha256=snapshot_hashes,
                                           final_probability_sha256=digest(directory / "final_probabilities.pt"),
                                           pto_sha256=_state_hash(model)))
        log.emit("completed", epochs_run=result["epochs_run"],
                 task_updates=result["task_updates"], seconds=time.monotonic() - started)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    run(sys.argv[1], sys.argv[2], sys.argv[3])
