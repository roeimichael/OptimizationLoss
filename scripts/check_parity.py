"""Check paired campaign budgets, data, training recipe and execution regime."""

import argparse
import collections
import json
from pathlib import Path
import sys

from configs.gen_campaign import load_protocol, cap_pair
from src.pipeline.config import validate_hyperparams

SHARED_KEYS = [
    "lr",
    "lr_constraint",
    "dropout",
    "batch_size",
    "pretrained",
    "class_weighted_ce",
    "constraint_chunk_size",
    "inference_chunk_size",
    "constraint_grad_clip",
    "constraint_grad_mode",
    "constraint_fp32",
]
TRAINED_METHODOLOGIES = {"tralo", "fioretto_ldf", "hounie_rcl", "fioretto_alm"}


def load(root):
    runs = []
    for p in sorted(Path(root).rglob("config.json")):
        c = json.loads(p.read_text(encoding="utf-8"))
        c["_done"] = (p.parent / "final_predictions.csv").exists()
        runs.append(c)
    return runs


def _identity_keys():
    return load_protocol()["warmup_identity_keys"]


def _check_lr_trap(runs, fails):
    for c in runs:
        hp = c["hyperparams"]
        if "lr_constraint" in hp and hp["lr_constraint"] != hp["lr"]:
            fails.append("LR TRAP: lr_constraint must equal lr on " + c["arm"])


def _check_dose(runs, fails):
    for c in runs:
        hp = c["hyperparams"]
        if c["methodology"] in TRAINED_METHODOLOGIES:
            if (
                hp.get("constraint_grad_mode") != "normalize"
                or hp.get("constraint_fp32") is not True
            ):
                fails.append(
                    "UNMATCHED CONSTRAINT DOSE: FP32/normalize required on " + c["arm"]
                )


def check(runs):
    P = load_protocol()
    fails = []
    arms = {c["arm"] for c in runs}
    if not set(P["mandatory_arms"]) <= arms:
        fails.append("both clippers are required")
    cells = collections.defaultdict(set)
    seen = set()
    data = collections.defaultdict(set)
    warmups = collections.defaultdict(set)
    versions, regimes = set(), set()
    for c in runs:
        arm, hp = c["arm"], c["hyperparams"]
        if arm not in P["arms"]:
            fails.append("unknown arm " + arm)
            continue
        spec = P["arms"][arm]
        if c["methodology"] != spec["methodology"]:
            fails.append("methodology mismatch on " + arm)
        validate_hyperparams(c["methodology"], hp)
        expected = (30, 0) if spec["phase"] == "posthoc" else (1, 29)
        if (hp["warmup_epochs"], hp["constraint_epochs"]) != expected:
            fails.append("UNEQUAL COMPUTE: wrong warm-up/constraint epochs on " + arm)
        if c["constraint"] != cap_pair(c["constraint_tag"]):
            fails.append("constraint does not match cap tag on " + arm)
        dc = c["dataset_config"]
        key = (
            c["dataset_mode"],
            c["model_name"],
            c["constraint_tag"],
            json.dumps(dc["constrained_class"]),
            hp["seed"],
        )
        if (arm, key) in seen:
            fails.append("duplicate cell-seed on " + arm)
        seen.add((arm, key))
        cells[arm].add(key)
        data[c["dataset_mode"]].add(json.dumps(dc, sort_keys=True))
        projection = [
            c["model_name"],
            c["dataset_mode"],
            dc["data_dir"],
            dc["num_classes"],
            {k: hp.get(k) for k in _identity_keys()},
        ]
        warmups[c["base_model_id"]].add(json.dumps(projection, sort_keys=True))
        version = c.get("run_code_version") or c.get("code_version")
        if not version or version == "unknown":
            fails.append("CODE VERSION IS unknown")
        versions.add(version)
        runtime = (c.get("results") or {}).get("runtime") or {}
        if runtime:
            regimes.add(
                tuple(
                    str(runtime.get(k))
                    for k in ("gpu_name", "amp_dtype", "grad_scaler")
                )
            )
    for k in SHARED_KEYS:
        vals = {json.dumps(c["hyperparams"][k]) for c in runs if k in c["hyperparams"]}
        if len(vals) > 1:
            fails.append(k + " differs across arms")
    _check_lr_trap(runs, fails)
    _check_dose(runs, fails)
    for c in runs:
        if c["arm"] == "tralo_null":
            for key in ("lambda_global", "lambda_local", "lambda_step"):
                if c["hyperparams"].get(key) != 0:
                    fails.append("tralo_null requires %s=0" % key)
    if (
        any(c["methodology"] in TRAINED_METHODOLOGIES for c in runs)
        and "tralo_null" not in arms
    ):
        fails.append("matched tralo_null control required")
    if cells and any(v != next(iter(cells.values())) for v in cells.values()):
        fails.append("unpaired cells or seeds")
    # A cap axis belongs to each dataset/backbone, not to the union of roots.
    capsets = collections.defaultdict(set)
    for c in runs:
        capsets[(c["dataset_mode"], c["model_name"])].add(json.dumps(c["constraint"]))
    if any(len(v) < 2 for v in capsets.values()):
        fails.append("at least two distinct cap levels required per dataset/backbone")
    if any(len(v) > 1 for v in data.values()):
        fails.append("dataset_config differs across runs")
    if any(len(v) > 1 for v in warmups.values()):
        fails.append("base_model_id collision: DIFFERENT warm-up identities")
    if len(versions) > 1:
        fails.append("MIXED CODE VERSIONS")
    if len(regimes) > 1:
        fails.append("MIXED NUMERIC REGIMES")
    if any(v and "-dirty" in v for v in versions):
        print("WARN: dirty source stamp is not an executable-byte fingerprint")
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    args = ap.parse_args()
    try:
        runs = load(args.root)
        if not runs:
            raise ValueError("no configs under " + args.root)
        fails = check(runs)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print("PARITY FAILED: %s" % exc)
        return 1
    for f in fails:
        print("FAIL: " + f)
    print("PARITY %s: %d configs" % ("FAILED" if fails else "OK", len(runs)))
    print(
        "Equal training epochs do not imply equal FLOPs; report constraint compute separately."
    )
    return int(bool(fails))


if __name__ == "__main__":
    sys.exit(main())
