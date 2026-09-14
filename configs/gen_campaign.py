"""Generate paired seven-arm campaigns from the maintained protocol."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
import yaml
from src.pipeline.config import validate_hyperparams
from src.utils.gitver import git_version
from src.pipeline.campaign import stage_campaign

PROTOCOL_PATH = str(Path(__file__).with_name("protocol.yml"))
PUBLIC_ARMS = ("tralo", "tralo_null", "clip", "focal_clip", "fioretto", "hounie", "alm")


def load_protocol(path=PROTOCOL_PATH):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_block(P, name):
    return P["blocks"][name] if name in P.get("blocks", {}) else P[name]


def build_hyperparams(P, arm_spec, seed, pretrained=None):
    hp = dict(P["core"])
    for name in arm_spec.get("blocks") or []:
        hp.update(resolve_block(P, name))
    if pretrained is not None:
        hp["pretrained"] = bool(pretrained)
    total = P["protocol"]["total_epochs"]
    hp["seed"] = seed
    hp["warmup_epochs"] = (
        total if arm_spec["phase"] == "posthoc" else P["protocol"]["trained_warmup"]
    )
    hp["constraint_epochs"] = total - hp["warmup_epochs"]
    validate_hyperparams(arm_spec["methodology"], hp)
    return hp


def compute_base_model_id(P, model_name, hp, dataset_mode, dc):
    key = {
        "model_name": model_name,
        "dataset_mode": dataset_mode,
        "data_dir": dc["data_dir"],
        "num_classes": dc["num_classes"],
    }
    key.update({k: hp[k] for k in P["warmup_identity_keys"] if k in hp})
    digest = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return "%s_%s_%s" % (model_name, dataset_mode, digest)


def code_version():
    return git_version()


def cap_pair(tag):
    try:
        local, global_ = tag.split("_")
        if not local.startswith("L") or not global_.startswith("G"):
            raise ValueError
        values = [int(x) / 100.0 for x in local[1:].split("-")]
        glob = int(global_[1:]) / 100.0
        if any(x < 0 for x in values) or glob < 0:
            raise ValueError
        return [values if len(values) > 1 else values[0], glob]
    except (ValueError, IndexError):
        raise SystemExit("bad cap tag %r -- expected L<pct>_G<pct>" % tag)


def resolve_datasets(P, args):
    result = {ds: dict(P["datasets"][ds]) for ds in args.datasets}
    if args.constrained_class is not None:
        for dc in result.values():
            dc["constrained_class"] = (
                args.constrained_class[0]
                if len(args.constrained_class) == 1
                else args.constrained_class
            )
    return result


def validate(P, args, resolved, arms):
    if len({json.dumps(cap_pair(tag)) for tag in args.caps}) < 2:
        raise SystemExit(
            "REFUSED: at least two cap levels with distinct fractions are required"
        )
    if P["core"]["lr"] != P["constraint_phase"]["lr_constraint"]:
        raise SystemExit("REFUSED: lr_constraint must equal lr")
    if not 0 <= P["protocol"]["trained_warmup"] <= P["protocol"]["total_epochs"]:
        raise SystemExit("REFUSED: warm-up must fit the total epoch budget")
    seeds = P["protocol"]["seeds"]
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(s) is not int or s < 0 for s in seeds)
    ):
        raise SystemExit("REFUSED: seeds must be distinct nonnegative integers")
    for ds, dc in resolved.items():
        classes = dc["constrained_class"]
        classes = classes if isinstance(classes, list) else [classes]
        if not classes or any(c < 0 or c >= dc["num_classes"] for c in classes):
            raise SystemExit("REFUSED: constrained_class out of range for %s" % ds)
        if len(set(classes)) != len(classes):
            raise SystemExit("REFUSED: constrained_class repeats a class for %s" % ds)
        for tag in args.caps:
            lp, gp = cap_pair(tag)
            if isinstance(lp, list) and len(lp) != len(classes):
                raise SystemExit(
                    "REFUSED: per-class cap count must match constrained_class"
                )
            meta = Path(dc["data_dir"]) / "test_meta.csv"
            if meta.exists():
                import pandas as pd
                from src.training.constraints import (
                    compute_global_constraints,
                    compute_local_constraints,
                )

                frame = pd.read_csv(meta)
                required = {"label", dc["group_column"]}
                if not required.issubset(frame.columns):
                    raise SystemExit(
                        "REFUSED: test metadata missing columns %s"
                        % sorted(required - set(frame.columns))
                    )
                labels = frame["label"]
                if (
                    labels.isna().any()
                    or ((labels % 1) != 0).any()
                    or ((labels < 0) | (labels >= dc["num_classes"])).any()
                ):
                    raise SystemExit("REFUSED: labels outside declared class schema")
                compute_global_constraints(
                    frame,
                    "label",
                    gp,
                    constrained_class=classes,
                    num_classes=dc["num_classes"],
                )
                compute_local_constraints(
                    frame,
                    "label",
                    lp,
                    dc["group_column"],
                    constrained_class=classes,
                    num_classes=dc["num_classes"],
                )
    for arm in arms:
        build_hyperparams(P, P["arms"][arm], seeds[0])


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--protocol", default=PROTOCOL_PATH)
    known, _ = pre.parse_known_args()
    P = load_protocol(known.protocol)
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--datasets", nargs="+", required=True, choices=sorted(P["datasets"])
    )
    parser.add_argument(
        "--models", nargs="+", default=[P["models"][0]], choices=P["models"]
    )
    parser.add_argument("--caps", nargs="+", default=["L30_G30", "L50_G50"])
    parser.add_argument(
        # Any arm DECLARED in the protocol may be named explicitly; "all" still
        # means the seven-arm public comparison, so adding an arm to
        # protocol.yml cannot silently change what an existing `--arms all`
        # campaign generates.
        "--arms", nargs="+", default=["tralo"],
        choices=[*sorted(set(P["arms"]) | set(PUBLIC_ARMS)), "all"]
    )
    parser.add_argument("--pretrained", choices=["true", "false"], default=None)
    parser.add_argument("--constrained-class", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    if args.seeds is not None:
        P["protocol"]["seeds"] = args.seeds
    requested = set(PUBLIC_ARMS) if "all" in args.arms else set(args.arms)
    arms = sorted(requested | set(P["mandatory_arms"]))
    if any(P["arms"][arm]["phase"] == "trained" for arm in arms):
        arms = sorted(set(arms) | {"tralo_null"})
    resolved = resolve_datasets(P, args)
    validate(P, args, resolved, arms)
    configs, version = [], code_version()
    for seed in P["protocol"]["seeds"]:
        for ds, dc in resolved.items():
            for model in args.models:
                for tag in args.caps:
                    for arm in arms:
                        spec = P["arms"][arm]
                        hp = build_hyperparams(
                            P,
                            spec,
                            seed,
                            pretrained=None
                            if args.pretrained is None
                            else args.pretrained == "true",
                        )
                        cls = dc["constrained_class"]
                        cls_tag = "-".join(
                            map(str, cls if isinstance(cls, list) else [cls])
                        )
                        path = (
                            Path(args.root)
                            / model
                            / ds
                            / tag
                            / arm
                            / ("seed_%d" % seed)
                            / "config.json"
                        )
                        cfg = {
                            "methodology": spec["methodology"],
                            "model_name": model,
                            "constraint": cap_pair(tag),
                            "constraint_tag": tag,
                            "dataset_mode": ds,
                            "dataset_config": dc,
                            "hyperparams": hp,
                            "base_model_id": compute_base_model_id(
                                P, model, hp, ds, dc
                            ),
                            "arm": arm,
                            "exp_name": "%s_%s_%s_%s_c%s_seed%d"
                            % (model, ds, arm, tag, cls_tag, seed),
                            "status": "pending",
                            "code_version": version,
                        }
                        configs.append((path, cfg))
    stage_campaign(args.root, {p.relative_to(Path(args.root)).as_posix(): cfg
                               for p, cfg in configs}, P)
    print("%d written -> %s; arms: %s" % (len(configs), args.root, " ".join(arms)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
