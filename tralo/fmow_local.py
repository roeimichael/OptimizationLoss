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
from .local_alm import snapshot_phr_step

PILOT = 6099
SEEDS = range(6100, 6148)
FIXED_PILOT = 6199
FIXED_SEEDS = range(6200, 6212)
ALM_PILOT = 6300
ALM_SEEDS = range(6301, 6313)
BOUNDARY_PILOT = 6400
BOUNDARY_SEEDS = range(6401, 6413)
DIVISORS = (10, 20)
RECIPE = dict(backbone="mobilenet_v3_large", capped_class=CAPPED,
              max_epochs=75, patience=5, batch_size=32, lr=1e-4,
              weight_decay=1e-4, decay_epoch=5, decay_factor=0.8,
              development_batch_size=16)


def validate(config):
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    alm = config.get("study") == "local_alm_direction_v1"
    boundary = config.get("study") == "local_boundary_v1"
    fixed = config.get("study") == "local_fixed_dose_v1" or alm or boundary
    expected = set(RECIPE) | {"seed", "snapshot_steps"}
    if fixed:
        expected |= {"study", "step_radius"}
        if type(config.get("step_radius")) is not float or config["step_radius"] != 0.1:
            raise ValueError("fixed-dose study requires radius 0.1")
    if alm or boundary:
        expected.add("alm_rho")
        if type(config.get("alm_rho")) is not float or config["alm_rho"] != 0.5:
            raise ValueError(("boundary study" if boundary else
                              "local ALM direction study") + " requires rho 0.5")
    if set(config) != expected:
        raise ValueError("config keys differ from the named local protocol")
    pilot = BOUNDARY_PILOT if boundary else ALM_PILOT if alm else FIXED_PILOT if fixed else PILOT
    seeds = BOUNDARY_SEEDS if boundary else ALM_SEEDS if alm else FIXED_SEEDS if fixed else SEEDS
    if type(config["seed"]) is not int or config["seed"] not in seeds and config["seed"] != pilot:
        raise ValueError("seed is outside the named study and pilot blocks")
    if type(config["snapshot_steps"]) is not bool or not config["snapshot_steps"] and config["seed"] != pilot:
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


def snapshot_side_steps(model, pool, groups, quota, seed, epoch, directory,
                        fixed_radius=None, phr_state=None, phr_rho=None,
                        boundary_calibrated=False, pto_probabilities=None):
    """Save joint, same-radius pooled-direction and sham snapshots for one cap."""
    import torch
    if type(boundary_calibrated) is not bool:
        raise ValueError("boundary_calibrated must be a bool")
    if boundary_calibrated and (fixed_radius != 0.1 or phr_state is None or
                                phr_rho != 0.5):
        raise ValueError("boundary study requires radius ceiling 0.1 and PHR rho 0.5")
    if boundary_calibrated and not isinstance(pto_probabilities, torch.Tensor):
        raise ValueError("boundary study requires the saved PTO snapshot probabilities")
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    state = _state_hash(model)
    steps = {}
    try:
        for arm in ("joint", "global_dose", "sham"):
            side = copy.deepcopy(model)
            try:
                kwargs = {}
                if arm == "joint" and boundary_calibrated:
                    kwargs.update(boundary_calibrated=True, max_radius=fixed_radius)
                elif arm == "joint" and fixed_radius is not None:
                    kwargs.update(fixed_radius=fixed_radius, require_common_descent=False)
                if arm == "global_dose":
                    kwargs.update(fixed_radius=steps["joint"]["radius"], global_only_direction=True)
                if arm == "sham":
                    kwargs.update(fixed_radius=steps["joint"]["radius"],
                                  sham_generator=torch.Generator().manual_seed(
                                      seed + SHAM_OFFSET + 1000 * epoch))
                if arm != "joint" and not steps["joint"]["applied"]:
                    if boundary_calibrated:
                        joint = steps["joint"]
                        steps[arm] = dict(
                            applied=False, radius=0.0, displacement=0.0,
                            skip_reason="joint_boundary_skip",
                            hard_before_global=joint["hard_before_global"],
                            hard_after_global=joint["hard_before_global"],
                            hard_before_local=joint["hard_before_local"],
                            hard_after_local=joint["hard_before_local"],
                            soft_before_global=joint["soft_before_global"],
                            soft_after_global=joint["soft_before_global"],
                            soft_before_local=joint["soft_before_local"],
                            soft_after_local=joint["soft_before_local"],
                            tensor_displacement_norms=[
                                0.0 for p in side.parameters() if p.requires_grad])
                    else:
                        steps[arm] = dict(applied=False, displacement=0.0)
                else:
                    steps[arm] = local_targeted_step(
                        side, pool, groups, CAPPED, quota["global_cap"],
                        quota["local_caps"], **kwargs)
                artifact = Path(directory) / f"epoch{epoch:02d}_{arm}.pt"
                probabilities = infer(side, pool).cpu()
                if boundary_calibrated and not steps[arm]["applied"] and not torch.equal(
                        probabilities, pto_probabilities.cpu()):
                    raise RuntimeError("skipped boundary arm differs from PTO probabilities")
                torch.save(probabilities, artifact)
                steps[arm]["probability_sha256"] = digest(artifact)
            finally:
                del side
        if phr_state is not None:
            side = copy.deepcopy(model)
            try:
                record, next_dual = snapshot_phr_step(
                    side, pool, groups, CAPPED, quota["global_cap"],
                    quota["local_caps"], phr_state["dual"], rho=phr_rho,
                    radius=fixed_radius,
                    boundary_calibrated=boundary_calibrated)
                artifact = Path(directory) / f"epoch{epoch:02d}_phr_local.pt"
                probabilities = infer(side, pool).cpu()
                if boundary_calibrated and not record["applied"] and not torch.equal(
                        probabilities, pto_probabilities.cpu()):
                    raise RuntimeError("skipped PHR arm differs from PTO probabilities")
                torch.save(probabilities, artifact)
                record["probability_sha256"] = digest(artifact)
                steps["phr_local"] = record
                phr_state["dual"] = next_dual
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
        images, train_labels, pool_rows, roles = load(
            data_root, include_pool_labels=config.get("study") not in
            ("local_fixed_dose_v1", "local_alm_direction_v1", "local_boundary_v1"))
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
            phr_states = ({str(divisor): {"dual": torch.zeros(1 + len(quota[str(divisor)]["local_caps"]))}
                          for divisor in DIVISORS}
                          if config.get("study") in ("local_alm_direction_v1",
                                                      "local_boundary_v1") else None)

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
                            cap_directory, fixed_radius=config.get("step_radius"),
                            phr_state=phr_states[str(divisor)] if phr_states is not None else None,
                            phr_rho=config.get("alm_rho"),
                            boundary_calibrated=config.get("study") == "local_boundary_v1",
                            pto_probabilities=probabilities)
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
