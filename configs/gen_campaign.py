"""THE campaign generator. It is the only one, and it can only emit configs that
satisfy `docs/FRAMEWORK.md` section 1.

This replaces 55 one-off generators totalling 5,481 lines. Every one of them
re-declared the same protocol by hand, and every protocol violation this project
has retracted a result over -- warm-up 50, unequal compute, a mismatched
lr_constraint, a single cap level, a campaign with no clipper in it -- entered
through one of those copies. The protocol now lives in exactly one place and is
asserted, not remembered.

Usage
-----
    python -m src.config_generators.gen_campaign \\
        --root results/<name> --datasets dermmnist tissuemnist \\
        --models MobileNetV3 --caps L30_G30 L50_G50 --arms clip focal_clip tralo

Arms
----
    clip         CE warm-up 30 + post-hoc          the strongest quality bar
    focal_clip   focal warm-up 30 + post-hoc       the strongest calibration bar
    tralo        CE warm-up 1 + 29 constraint      the method
    fioretto     Fioretto-LDF dual, same split     required to claim vs the duals
    hounie       Hounie-RCL dual, same split       required to claim vs the duals

`clip` and `focal_clip` are ALWAYS added, whether or not you ask for them: an
arm-vs-arm delta is not a result until both bars are in the same campaign.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys


def compute_base_model_id(model_name, hp, dataset_mode, data_dir, dataset_config):
    """Hash identifying a warm-up-trained model, so arms that share a warm-up
    share its cache.

    Any hyperparameter that changes WHAT WARM-UP OPTIMIZES must be in this key,
    or the second arm silently loads the first one's cached model. That has bitten
    this project repeatedly -- most recently `rank_pair_weight`, where two doses
    hashed identically and the sweep became one arm measured twice.
    """
    key = {"model_name": model_name, "lr": hp["lr"], "dropout": hp["dropout"],
           "batch_size": hp["batch_size"], "warmup_epochs": hp["warmup_epochs"],
           "pretrained": hp.get("pretrained", False),
           "class_weighted_ce": hp.get("class_weighted_ce", False),
           "dataset_mode": dataset_mode, "data_dir": data_dir,
           "num_classes": dataset_config.get("num_classes"),
           "image_size": dataset_config.get("image_size")}
    if "seed" in hp:
        key["seed"] = hp["seed"]
    # the warm-up objective itself, when an arm swaps it (focal_clip)
    for k in ("warmup_loss", "focal_alpha", "focal_gamma"):
        if k in hp:
            key[k] = hp[k]
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return "%s_%s_%s" % (model_name, dataset_mode, h)


def code_version():
    """Short git SHA + dirty flag, stamped into every config so a re-run can
    detect code drift."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"],
                                stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"

# ---------------------------------------------------------------- the protocol
# Every value here is load-bearing and was paid for with a retracted result.
WARMUP_TOTAL = 30          # optimizer epochs, IDENTICAL on both sides = equal compute
TRAINED_WARMUP = 1         # warm-up 50 saturates CE; warm-up 5 is a dead zone
SEEDS = [1, 2, 3, 4]

SHARED = {
    "lr": 1e-4,
    "lr_constraint": 1e-4,      # MUST equal lr -- unequal LR fabricated a -16.7pp finding
    "dropout": 0.3,
    "batch_size": 64,
    "pretrained": True,
    "class_weighted_ce": False,
    "constraint_chunk_size": 256,
    "stable_count_threshold": 31,
    # NOTE: enable_ce_skip and alpha_kl are GONE, not set to False/0 -- the
    # CE-skip and KL machinery was deleted from the pipeline entirely, so a
    # config can no longer imply a knob that does not exist.
}
TRALO = {"lambda_step": 0.05, "initial_rho": 0.5, "rho_target": 100.0}

DATASETS = {
    "dermmnist": {"data_dir": "data/dermmnist/slice_1", "num_classes": 7,
                  "image_size": 224, "target_column": "label",
                  "group_column": "loc_group", "constrained_class": 4},
    "tissuemnist": {"data_dir": "data/tissuemnist/slice_1", "num_classes": 8,
                    "image_size": 224, "target_column": "label",
                    "group_column": "synth_group", "constrained_class": 4},
    "octmnist": {"data_dir": "data/octmnist/slice_1", "num_classes": 4,
                 "image_size": 224, "target_column": "label",
                 "group_column": "synth_group", "constrained_class": 0},
}
MODELS = ["MobileNetV3", "MobileNetV2", "RegNetY400MF", "ShuffleNetV2"]

ARMS = {
    "clip":       ("heuristic",    {}),
    "focal_clip": ("heuristic",    {"warmup_loss": "focal", "base_loss": "focal",
                                    "focal_alpha": 1.0, "focal_gamma": 2.0}),
    "tralo":      ("tralo",        {}),
    "fioretto":   ("fioretto_ldf", {}),
    "hounie":     ("hounie_rcl",   {}),
}
POSTHOC_ARMS = {"clip", "focal_clip"}


def cap_pair(tag):
    """'L30_G30' -> [0.30, 0.30]. Caps are a FRACTION of the true positive count."""
    local, glob = tag.split("_")
    return [int(local[1:]) / 100, int(glob[1:]) / 100]


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--root", required=True)
    a.add_argument("--datasets", nargs="+", required=True, choices=sorted(DATASETS))
    a.add_argument("--models", nargs="+", default=["MobileNetV3"], choices=MODELS)
    a.add_argument("--caps", nargs="+", default=["L30_G30", "L50_G50"])
    a.add_argument("--arms", nargs="+", default=["tralo"], choices=sorted(ARMS))
    args = a.parse_args()

    # -- protocol assertions: refuse to generate an invalid campaign ------------
    if len(set(args.caps)) < 2:
        sys.exit("REFUSED: at least two cap levels are required. A claim from cells "
                 "sharing one cap level has been retracted three times.")
    arms = sorted(set(args.arms) | POSTHOC_ARMS)
    if arms != sorted(set(args.arms)):
        print("NOTE: added the mandatory clippers ->", " ".join(sorted(POSTHOC_ARMS)))

    todo = [(ds, mdl, tag, arm, seed)
            for seed in SEEDS for ds in args.datasets for mdl in args.models
            for tag in args.caps for arm in arms]          # seed-major dispatch order

    version, written, skipped = code_version(), 0, 0
    for ds, mdl, tag, arm, seed in todo:
        dc = DATASETS[ds]
        methodology, extra = ARMS[arm]
        posthoc = arm in POSTHOC_ARMS
        hp = {**SHARED, **({} if posthoc else TRALO), **extra, "seed": seed,
              "warmup_epochs": WARMUP_TOTAL if posthoc else TRAINED_WARMUP,
              "constraint_epochs": 0 if posthoc else WARMUP_TOTAL - TRAINED_WARMUP}
        assert hp["warmup_epochs"] + hp["constraint_epochs"] == WARMUP_TOTAL, "equal compute"
        assert hp["lr"] == hp["lr_constraint"], "lr_constraint must equal lr"

        path = "%s/%s/%s/%s/%s/seed_%d" % (args.root, mdl, ds, tag, arm, seed)
        cfg = {"methodology": methodology, "model_name": mdl,
               "constraint": cap_pair(tag), "constraint_tag": tag,
               "dataset_mode": ds, "dataset_config": dc, "hyperparams": hp,
               "base_model_id": compute_base_model_id(mdl, hp, ds, dc["data_dir"], dc),
               "arm": arm, "sweep_tag": os.path.basename(args.root),
               "exp_name": "%s_%s_%s_%s_seed%d" % (mdl, ds, arm, tag, seed),
               "experiment_path": path, "status": "pending"}
        dest = os.path.join(path, "config.json")
        if os.path.exists(dest):
            try:
                if json.load(open(dest)).get("status") == "completed":
                    skipped += 1
                    continue          # never reset a finished run back to pending
            except (ValueError, OSError):
                pass
        cfg["code_version"] = version
        os.makedirs(path, exist_ok=True)
        json.dump(cfg, open(dest, "w"), indent=2)
        written += 1

    cells = len(args.datasets) * len(args.models) * len(args.caps)
    print("%d written, %d already completed (skipped) -> %s"
          % (written, skipped, args.root))
    print("  %d cells (dataset x model x cap) x %d arms x %d seeds"
          % (cells, len(arms), len(SEEDS)))
    print("  arms:", " ".join(arms))
    print("  trained arms: warm-up %d + constraint %d | post-hoc arms: warm-up %d + 0"
          % (TRAINED_WARMUP, WARMUP_TOTAL - TRAINED_WARMUP, WARMUP_TOTAL))
    print("  code_version:", version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
