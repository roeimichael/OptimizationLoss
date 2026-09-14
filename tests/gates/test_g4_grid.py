"""Maintained behavioral regression fixtures."""

import contextlib

import io

import json

import os

import subprocess

import sys

import pytest

import yaml

from .conftest import ROOT, report

from configs.gen_campaign import (
    build_hyperparams,
    cap_pair,
    compute_base_model_id,
    load_protocol,
)

pytestmark = pytest.mark.stage4_grid

MODEL = "MobileNetV2"

CAPS = ["L70_G95", "L80_G95"]

TRIO = ["tralo", "tralo_null"]

MIXED = ["clip", "focal_clip", "tralo", "tralo_null"]


def _cfg(P, arm, cap, seed, model=MODEL, ds="iwildcam"):
    hp = build_hyperparams(P, P["arms"][arm], seed)
    if "constraint_fp32" in hp:
        hp["constraint_fp32"] = True
        hp["constraint_grad_mode"] = "normalize"
    dc = dict(P["datasets"][ds])
    return {
        "methodology": P["arms"][arm]["methodology"],
        "model_name": model,
        "constraint": cap_pair(cap),
        "constraint_tag": cap,
        "arm": arm,
        "dataset_mode": ds,
        "dataset_config": dc,
        "hyperparams": hp,
        "base_model_id": compute_base_model_id(P, model, hp, ds, dc),
        "status": "completed",
        "code_version": "1111aaaa2222",
    }


def _campaign(root, P, arms, caps=CAPS, seeds=(1, 2), mutate=None):
    for arm in arms:
        for cap in caps:
            for seed in seeds:
                cfg = _cfg(P, arm, cap, seed)
                if mutate:
                    mutate(cfg)
                d = os.path.join(
                    str(root), cfg["model_name"], "iwildcam", cap, arm, "seed_%d" % seed
                )
                os.makedirs(d, exist_ok=True)
                with io.open(
                    os.path.join(d, "config.json"), "w", encoding="utf-8"
                ) as fh:
                    json.dump(cfg, fh)
    return str(root)


def _on_disk(root):
    out = {}
    for path, _d, files in os.walk(str(root)):
        if "config.json" in files:
            with io.open(os.path.join(path, "config.json"), encoding="utf-8") as fh:
                out[path] = json.load(fh)
    return out


def _parity(root):
    from scripts import check_parity

    (buf, argv, cwd) = (io.StringIO(), sys.argv, os.getcwd())
    try:
        os.chdir(ROOT)
        sys.argv = ["check_parity", str(root)]
        with contextlib.redirect_stdout(buf):
            rc = check_parity.main()
    finally:
        sys.argv = argv
        os.chdir(cwd)
    return (rc, buf.getvalue())


def _gen(root, arms, caps=CAPS, extra=(), protocol=None):
    cmd = [
        sys.executable,
        "-m",
        "configs.gen_campaign",
        "--root",
        str(root),
        "--datasets",
        "iwildcam",
        "--models",
        MODEL,
        "--caps",
    ]
    cmd += list(caps) + ["--arms"] + list(arms) + list(extra)
    cmd += ["--protocol", str(protocol)] if protocol else []
    p = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return (p.returncode, (p.stdout or "") + (p.stderr or ""))


@pytest.fixture(scope="session")
def protocol_yml():
    return load_protocol()


@pytest.fixture(scope="session")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("g4_valid") / "camp"
    (rc, out) = _gen(root, TRIO, extra=[])
    assert rc == 0, "the generator refused a VALID campaign:\n%s" % out
    return (str(root), _on_disk(root), out)


def test_warmup_1_29_trained_30_0_posthoc_and_equal_compute(
    protocol_yml, generated, tmp_path
):
    (P, fails) = (protocol_yml, [])
    got = (P["protocol"]["total_epochs"], P["protocol"]["trained_warmup"])
    if got != (30, 1):
        fails.append("protocol total/warm-up is %s, not (30, 1)" % (got,))
    for arm, spec in sorted(P["arms"].items()):
        hp = build_hyperparams(P, spec, 1)
        split = (hp["warmup_epochs"], hp["constraint_epochs"])
        want = (30, 0) if spec["phase"] == "posthoc" else (1, 29)
        if split != want:
            fails.append("%s splits %s, protocol says %s" % (arm, split, want))
    for cfg in generated[1].values():
        h = cfg["hyperparams"]
        if h["warmup_epochs"] + h["constraint_epochs"] != 30:
            fails.append(
                "%s: %d+%d epochs, not 30"
                % (cfg["arm"], h["warmup_epochs"], h["constraint_epochs"])
            )
    if _parity(generated[0])[0] != 0:
        fails.append("check_parity REFUSED the valid generated campaign")

    def warmup_50(cfg):
        if cfg["arm"] == "tralo":
            cfg["hyperparams"]["warmup_epochs"] = 50

    (rc, out) = _parity(_campaign(tmp_path / "wu50", P, MIXED, mutate=warmup_50))
    if rc == 0 or "UNEQUAL COMPUTE" not in out:
        fails.append("warm-up 50 on one arm was NOT rejected (rc=%d)" % rc)
    if _parity(_campaign(tmp_path / "ok", P, MIXED))[0] != 0:
        fails.append("a 1+29 / 30+0 campaign was rejected")
    report(fails, "equal-compute defects")


def test_both_clippers_and_at_least_two_cap_levels(protocol_yml, generated, tmp_path):
    (P, fails) = (protocol_yml, [])
    if sorted(P["mandatory_arms"]) != ["clip", "focal_clip"]:
        fails.append("mandatory_arms is %s" % P["mandatory_arms"])
    arms = {c["arm"] for c in generated[1].values()}
    for bar in ("clip", "focal_clip"):
        if bar not in arms:
            fails.append("generator did not auto-add %s (got %s)" % (bar, sorted(arms)))
    if len({c["constraint_tag"] for c in generated[1].values()}) < 2:
        fails.append("the generated campaign carries one cap level")
    (rc, out) = _gen(tmp_path / "onecap", TRIO, caps=[CAPS[0]], extra=[])
    if rc == 0 or "at least two cap levels" not in out:
        fails.append("a single-cap campaign GENERATED (rc=%d)" % rc)
    (rc, out) = _parity(_campaign(tmp_path / "p1", P, MIXED, caps=[CAPS[0]]))
    if rc == 0 or "two distinct cap levels" not in out:
        fails.append("check_parity accepted a single-cap tree (rc=%d)" % rc)
    if _parity(_campaign(tmp_path / "p2", P, MIXED))[0] != 0:
        fails.append("check_parity rejected a two-cap tree")
    report(fails, "clipper/cap-level defects")


def test_one_code_version_stamp_across_the_campaign(protocol_yml, generated, tmp_path):
    (P, fails) = (protocol_yml, [])
    stamps = {c["code_version"] for c in generated[1].values()}
    if len(stamps) != 1:
        fails.append(
            "the generator emitted %d stamps: %s" % (len(stamps), sorted(stamps))
        )

    def two(cfg):
        cfg["code_version"] = "deadbeef" if cfg["arm"] == "tralo" else "1111aaaa2222"

    for name, mut, want in [
        ("two_stamps", two, "MIXED CODE VERSIONS"),
        (
            "all_unknown",
            lambda c: c.update(code_version="unknown"),
            "CODE VERSION IS unknown",
        ),
    ]:
        (rc, out) = _parity(_campaign(tmp_path / name, P, MIXED, mutate=mut))
        if rc == 0 or want not in out:
            fails.append("%s accepted (rc=%d, wanted %r)" % (name, rc, want))
    if _parity(_campaign(tmp_path / "onestamp", P, MIXED))[0] != 0:
        fails.append("a one-stamp campaign was rejected")
    report(fails, "code-version defects")


def test_lr_parity_and_the_lr_trap(protocol_yml, tmp_path):
    (P, fails) = (protocol_yml, [])
    if P["core"]["lr"] != P["constraint_phase"]["lr_constraint"]:
        fails.append(
            "protocol lr %s != lr_constraint %s"
            % (P["core"]["lr"], P["constraint_phase"]["lr_constraint"])
        )
    path = tmp_path / "trapped.yml"
    with io.open(str(path), "w", encoding="utf-8") as fh:
        fh.write(
            yaml.safe_dump(
                dict(
                    P,
                    constraint_phase=dict(
                        P["constraint_phase"], lr_constraint=5e-06, constraint_fp32=True
                    ),
                )
            )
        )
    (rc, out) = _gen(tmp_path / "lrt", TRIO, protocol=path)
    if rc == 0 or "lr_constraint" not in out:
        fails.append("the generator emitted an LR-trapped campaign (rc=%d)" % rc)

    def trap(cfg):
        if "lr_constraint" in cfg["hyperparams"]:
            cfg["hyperparams"]["lr_constraint"] = 5e-06

    def disagree(cfg):
        if cfg["arm"] == "tralo":
            cfg["hyperparams"]["lr"] = 5e-06

    for name, mut, want in [
        ("lr_agreed", trap, "LR TRAP"),
        ("lr_split", disagree, "lr differs across arms"),
    ]:
        (rc, out) = _parity(_campaign(tmp_path / name, P, MIXED, mutate=mut))
        if rc == 0 or want not in out:
            fails.append("%s accepted (rc=%d, wanted %r)" % (name, rc, want))
    if _parity(_campaign(tmp_path / "lr_ok", P, MIXED))[0] != 0:
        fails.append("an equal-lr campaign was rejected")
    report(fails, "learning-rate defects")


def test_every_trained_arm_ATTEMPTS_every_constraint_epoch(tmp_path):
    import scripts.smoke_arms as sa
    from src.experiments.runner import TRAIN_FNS
    from lean_fixtures import protocol_with_nulls

    P_ = protocol_with_nulls()
    fails = []
    EPOCHS = 2
    TRAINED = ["tralo", "alm", "fioretto", "hounie"]
    NULLS = ["tralo_null", "fioretto_null", "hounie_null", "alm_null"]
    for arm in TRAINED + NULLS:
        if arm not in P_["arms"]:
            fails.append("%s is gone from protocol.yml" % arm)
            continue
        (inputs, _g, _l) = sa.make_inputs(P_, arm, str(tmp_path))
        out = TRAIN_FNS[inputs.config["methodology"]](inputs)
        got = (out.summary or {}).get("constraint_steps_attempted")
        want = 0 if arm in NULLS else EPOCHS
        if got != want:
            fails.append(
                "%s attempted %r constraint steps, expected %d" % (arm, got, want)
            )
        applied = (out.summary or {}).get("constraint_steps_applied")
        if arm in TRAINED and applied != got:
            fails.append(
                "%s applied %r of %r attempted -- a non-finite constraint gradient dropped a step"
                % (arm, applied, got)
            )
    report(fails, "constraint-dose defects")
