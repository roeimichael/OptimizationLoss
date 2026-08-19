"""THE campaign generator. It reads `configs/protocol.yml` and contains no
experimental constants of its own.

If a number decides what an experiment does, it is in the YAML. This file only
assembles: it picks each arm's blocks, splits the epoch budget, hashes the
warm-up identity, and refuses to emit a campaign that violates the protocol.

    python -m configs.gen_campaign --root results/<name> \\
        --datasets dermmnist tissuemnist --models MobileNetV3 \\
        --caps L30_G30 L50_G50 --arms all

Caps
----
`L30_G50` sets the LOCAL (per-group) cap to 30% and the GLOBAL cap to 50% of the
constrained class's true test-set count. The two are independent, so an
asymmetric sweep is just a list of tags. Both are turned into integer budgets by
`src/training/constraints.py` against the actual test labels -- the percentage
only standardizes how hard the cap binds across datasets.

Constrained classes
-------------------
`constrained_class` may be a single index or a list, per dataset in the YAML, and
`--constrained-class` overrides it for one run. Indices are validated against the
dataset's `num_classes`, because a cap on a class that does not exist is silently
skipped by the loss.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL_PATH = os.path.join(HERE, "protocol.yml")


def load_protocol(path=PROTOCOL_PATH):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_block(P, name):
    """A block name is either a top-level section or an entry under `blocks`."""
    if name in P.get("blocks", {}):
        return P["blocks"][name]
    if name in P:
        return P[name]
    raise KeyError("protocol.yml: unknown block %r" % name)


def build_hyperparams(P, arm_spec, seed):
    """Assemble exactly the keys this arm's methodology reads, plus the contract
    keys that scripts/check_parity.py verifies on every arm."""
    total = P["protocol"]["total_epochs"]
    trained_warmup = P["protocol"]["trained_warmup"]
    posthoc = arm_spec["phase"] == "posthoc"

    hp = dict(P["core"])
    for name in arm_spec.get("blocks") or []:
        hp.update(resolve_block(P, name))
    hp["seed"] = seed
    hp["warmup_epochs"] = total if posthoc else trained_warmup
    hp["constraint_epochs"] = 0 if posthoc else total - trained_warmup
    return hp


def compute_base_model_id(P, model_name, hp, dataset_mode, dc):
    """Identity of the WARM-UP-trained model, so arms that share a warm-up share
    its cache and arms that do not, do not.

    The key is built from `warmup_identity_keys` in the YAML plus the dataset
    identity. Anything that changes what the warm-up optimizes must be listed
    there, or a second arm silently loads the first one's model -- which has
    happened four times in this project.
    """
    key = {"model_name": model_name,
           "dataset_mode": dataset_mode,
           "data_dir": dc["data_dir"],
           "num_classes": dc["num_classes"]}
    for k in P["warmup_identity_keys"]:
        if k in hp:
            key[k] = hp[k]
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return "%s_%s_%s" % (model_name, dataset_mode, h)


def code_version():
    """Short git SHA + dirty flag, so a re-run can detect code drift."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"],
                                stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _cls_tag(dc):
    """Capped class(es) as a filename-safe tag: 4 -> "4", [4, 5] -> "4-5"."""
    c = dc["constrained_class"]
    return "-".join(str(x) for x in (c if isinstance(c, list) else [c]))


def cap_pair(tag):
    """'L30_G50' -> [0.30, 0.50]: (local, global), independent by construction."""
    try:
        local, glob = tag.split("_")
        if local[0] != "L" or glob[0] != "G":
            raise ValueError
        return [int(local[1:]) / 100.0, int(glob[1:]) / 100.0]
    except (ValueError, IndexError):
        sys.exit("bad cap tag %r -- expected L<pct>_G<pct>, e.g. L30_G50" % tag)


def resolve_datasets(P, args):
    """Dataset config per dataset with `--constrained-class` already applied.

    Resolved BEFORE validation, not inside the emit loop: validating the YAML
    default while emitting the override let `--constrained-class 9` through on a
    7-class dataset, where the loss would have skipped the cap silently.
    """
    out = {}
    for ds in args.datasets:
        dc = dict(P["datasets"][ds])
        if args.constrained_class is not None:
            dc["constrained_class"] = (args.constrained_class[0]
                                       if len(args.constrained_class) == 1
                                       else list(args.constrained_class))
        out[ds] = dc
    return out


def validate(P, args, resolved):
    if len(set(args.caps)) < 2:
        sys.exit("REFUSED: at least two cap levels are required. A claim from cells "
                 "sharing one cap level has been retracted three times.")
    # Two distinct TAGS can be the same EXPERIMENT. The binding budget is
    # min(global_K, sum of local_K), so L30_G30 and L30_G50 are identical
    # wherever the global cap is already slack -- and duplicate runs manufacture
    # significance: 8 pairs where 4 duplicate the other 4 gave p=0.0078 when the
    # honest n=4 has an exact floor of 0.125.
    locals_ = {}
    for tag in args.caps:
        lp, gp = cap_pair(tag)
        locals_.setdefault(lp, []).append((tag, gp))
    for lp, tags in locals_.items():
        if len({gp for _t, gp in tags}) > 1 and len(tags) > 1:
            print("NOTE: caps %s share local %d%%. They are the SAME experiment "
                  "wherever the global cap is slack (it can only bind BELOW the "
                  "sum of the local caps). Run `python -m scripts.verify_caps "
                  "--caps %s` on the real slices before trusting them as two "
                  "levels."
                  % ([t for t, _g in tags], int(lp * 100),
                     " ".join(t for t, _g in tags)))
    lr = P["core"]["lr"]
    lr_c = P["constraint_phase"]["lr_constraint"]
    if lr != lr_c:
        sys.exit("REFUSED: lr (%s) != lr_constraint (%s). Unequal learning rates "
                 "fabricated a -16.7pp finding that was -1.7pp once equalized." % (lr, lr_c))
    for ds, dc in resolved.items():
        classes = dc["constrained_class"]
        classes = classes if isinstance(classes, list) else [classes]
        if not classes:
            sys.exit("REFUSED: %s has no constrained class." % ds)
        if len(set(classes)) != len(classes):
            sys.exit("REFUSED: %s constrained_class %s repeats a class; the second "
                     "cap would overwrite the first." % (ds, classes))
        for c in classes:
            if not 0 <= int(c) < int(dc["num_classes"]):
                sys.exit("REFUSED: %s constrained_class %s is out of range for "
                         "num_classes=%s. A cap on a nonexistent class is silently "
                         "skipped by the loss." % (ds, c, dc["num_classes"]))


def main():
    P = load_protocol()
    a = argparse.ArgumentParser()
    a.add_argument("--root", required=True)
    a.add_argument("--datasets", nargs="+", required=True, choices=sorted(P["datasets"]))
    a.add_argument("--models", nargs="+", default=[P["models"][0]], choices=P["models"])
    a.add_argument("--caps", nargs="+", default=["L30_G30", "L50_G50"],
                   help="L<local>_G<global>, independent; e.g. L30_G50")
    a.add_argument("--arms", nargs="+", default=["tralo"],
                   choices=sorted(P["arms"]) + ["all"],
                   help="'all' runs the full panel: every baseline the paper claims")
    a.add_argument("--constrained-class", nargs="+", type=int, default=None,
                   help="override the YAML's capped class(es) for every dataset; "
                        "one index or several for the coupled multi-class setting")
    a.add_argument("--protocol", default=PROTOCOL_PATH, help="alternate protocol.yml")
    args = a.parse_args()
    if args.protocol != PROTOCOL_PATH:
        P = load_protocol(args.protocol)

    requested = set(P["arms"]) if "all" in args.arms else set(args.arms)
    mandatory = set(P["mandatory_arms"])
    arms = sorted(requested | mandatory)
    added = sorted(mandatory - requested)
    if added:
        print("NOTE: added the mandatory clippers ->", " ".join(added))

    resolved = resolve_datasets(P, args)
    validate(P, args, resolved)

    todo = [(ds, mdl, tag, arm, seed)
            for seed in P["protocol"]["seeds"] for ds in args.datasets
            for mdl in args.models for tag in args.caps for arm in arms]

    version, written, skipped = code_version(), 0, 0
    total = P["protocol"]["total_epochs"]
    for ds, mdl, tag, arm, seed in todo:
        dc = resolved[ds]
        spec = P["arms"][arm]
        hp = build_hyperparams(P, spec, seed)
        assert hp["warmup_epochs"] + hp["constraint_epochs"] == total, "equal compute"

        path = "%s/%s/%s/%s/%s/seed_%d" % (args.root, mdl, ds, tag, arm, seed)
        cfg = {"methodology": spec["methodology"], "model_name": mdl,
               "constraint": cap_pair(tag), "constraint_tag": tag,
               "dataset_mode": ds, "dataset_config": dc, "hyperparams": hp,
               "base_model_id": compute_base_model_id(P, mdl, hp, ds, dc),
               "arm": arm,
               "exp_name": "%s_%s_%s_%s_c%s_seed%d"
                           % (mdl, ds, arm, tag, _cls_tag(dc), seed),
               "status": "pending", "code_version": version}
        dest = os.path.join(path, "config.json")
        if os.path.exists(dest):
            try:
                prev = json.load(open(dest))
            except (ValueError, OSError):
                prev = None
            if prev is not None:
                prev_cls = prev.get("dataset_config", {}).get("constrained_class")
                if prev_cls is not None and prev_cls != dc["constrained_class"]:
                    sys.exit(
                        "REFUSED: %s already holds a run with constrained_class "
                        "%s, but this campaign asks for %s. Writing here would "
                        "leave two different capped classes in one cell -- a "
                        "completed run is never reset, so the old one would "
                        "survive and be pooled with the new. Use a different "
                        "--root." % (path, prev_cls, dc["constrained_class"]))
                if prev.get("status") == "completed":
                    skipped += 1
                    continue          # never reset a finished run back to pending
        os.makedirs(path, exist_ok=True)
        json.dump(cfg, open(dest, "w"), indent=2)
        written += 1

    cells = len(args.datasets) * len(args.models) * len(args.caps)
    print("%d written, %d already completed (skipped) -> %s"
          % (written, skipped, args.root))
    print("  %d cells (dataset x model x cap) x %d arms x %d seeds"
          % (cells, len(arms), len(P["protocol"]["seeds"])))
    print("  arms:", " ".join(arms))
    print("  trained arms: warm-up %d + constraint %d | post-hoc arms: warm-up %d + 0"
          % (P["protocol"]["trained_warmup"],
             total - P["protocol"]["trained_warmup"], total))
    # POWER, before the GPU time is spent rather than after.
    # The scorer's atomic unit is the CELL, and the exact two-sided Wilcoxon
    # floor at n non-zero pairs is 2^(1-n). With 11 metrics in the BH family, a
    # metric that is the only mover gets q = p * 11, so q < 0.05 needs
    # p < 0.00455 and therefore n >= 9 cells. Below that, NO isolated metric can
    # ever print *** WIN or *** LOSS however large the effect.
    #
    # This guard used to live only in full_panel, i.e. after the campaign ran.
    # The recorded failure is exactly that shape: "at n=2 Wilcoxon floors at
    # p=0.5 so in-flight campaigns ALWAYS read as ties -- arms were abandoned on
    # that."
    floor = 2.0 ** (1 - cells)
    n_family = 11
    print("  POWER: %d cells -> exact Wilcoxon floor p=%.5f; a lone mover needs"
          % (cells, floor))
    print("         q = p x %d < 0.05, i.e. p < %.5f" % (n_family, 0.05 / n_family))
    if floor > 0.05 / n_family:
        need = 9
        print("  *** UNDERPOWERED: with %d cells NO single metric can reach a "
              "*** verdict," % cells)
        print("      whatever the effect size. %d cells is the minimum for one. "
              "Add a" % need)
        print("      backbone, a dataset or a cap level -- or accept that this "
              "campaign can")
        print("      only report DIRECTION and per-cell consistency, never "
              "significance.")
    # Those three are NOT interchangeable, and saying so plainly matters more
    # than the floor above. Every cell inside one dataset shares that dataset's
    # fixed test set and the K derived from it, so a new backbone or a new cap
    # level buys RESOLUTION on the cells we ran; only a new dataset buys an
    # independent test set. full_panel prints a dataset-clustered readout beside
    # the per-cell one for exactly this reason.
    n_ds = len(args.datasets)
    ds_floor = 2.0 ** (1 - n_ds)
    print("  GENERALIZATION: %d dataset(s) -> exact sign-flip floor p=%.3f "
          "on the" % (n_ds, ds_floor))
    print("         clustered unit. Cells within a dataset are NOT independent "
          "draws:")
    print("         a backbone or a cap level adds resolution, a DATASET adds "
          "independence.")
    if ds_floor > 0.05:
        print("         *** at %d dataset(s) no clustered result can reach "
              "p<0.05 --" % n_ds)
        print("         *** with all three it is still 0.25. Generality here is "
              "a DIRECTION")
        print("         *** claim across datasets, never a significant one.")
    print("  protocol: %s" % os.path.relpath(args.protocol))
    print("  code_version:", version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
