"""STAGE 4 -- THE EXPERIMENT GRID. Is this campaign apples-to-apples?

Answered BEFORE launch, from configs alone: every gate runs on a config tree,
on CPU, in seconds. Each builds a BROKEN campaign in tmp_path and shows the
check reject it beside a valid one it accepts (conftest rule 2). Sources:
configs/gen_campaign.py, configs/protocol.yml, scripts/check_parity.py,
scripts/quarantine.py, scripts/full_panel.py, docs/FRAMEWORK.md 2(u), 2(z3),
2(z16)/2(z17), 2(z24), docs/archive/REJECTED_full_2026-08-18.md.
"""
import contextlib
import io
import json
import re
import os
import subprocess
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import ROOT, report          # noqa: E402  (gate scaffolding)

from configs.gen_campaign import (build_hyperparams, cap_pair,  # noqa: E402
                                  compute_base_model_id, count_control_arms,
                                  load_protocol, _null_of)

pytestmark = pytest.mark.stage4_grid

# The two caps `taskwin2` runs, measured INSIDE MobileNetV3's task window
# (configs/task_windows.yml, 2(z24)). Two levels, because a claim from one has
# been retracted three times.
CAPS = ["L70-90_G95", "L80-100_G95"]
DEAD_CAPS = ["L20_G50", "L30_G50"]     # 24 of 24 cells pose no question, 2(z17)
SLICE = os.path.join(ROOT, "data", "iwildcam", "oodslice", "test_meta.csv")
TRIO = ["tralo", "tralo_null", "tralo_reseed"]
MIXED = ["clip", "tralo", "tralo_null"]


def _cfg(P, arm, cap, seed, model="MobileNetV3", ds="iwildcam"):
    """One run's config.json, built through the GENERATOR's own functions, so a
    broken variant differs from a real campaign in exactly one field."""
    hp = build_hyperparams(P, P["arms"][arm], seed)
    if "constraint_fp32" in hp:              # what --constraint-fp32 writes
        hp["constraint_fp32"] = True
        hp["constraint_grad_mode"] = "normalize"
    dc = dict(P["datasets"][ds])
    return {"methodology": P["arms"][arm]["methodology"], "model_name": model,
            "constraint": cap_pair(cap), "constraint_tag": cap, "arm": arm,
            "dataset_mode": ds, "dataset_config": dc, "hyperparams": hp,
            "base_model_id": compute_base_model_id(P, model, hp, ds, dc),
            "status": "completed", "code_version": "1111aaaa2222"}


def _campaign(root, P, arms, caps=CAPS, seeds=(1, 2), mutate=None):
    """The arm x cap x seed product under `root`. `mutate(cfg)` edits a config
    in place -- that is how every negative control below is built."""
    for arm in arms:
        for cap in caps:
            for seed in seeds:
                cfg = _cfg(P, arm, cap, seed)
                if mutate:
                    mutate(cfg)
                d = os.path.join(str(root), cfg["model_name"], "iwildcam", cap,
                                 arm, "seed_%d" % seed)
                os.makedirs(d, exist_ok=True)
                with io.open(os.path.join(d, "config.json"), "w",
                             encoding="utf-8") as fh:
                    json.dump(cfg, fh)
    return str(root)


def _on_disk(root):
    out = {}
    for path, _d, files in os.walk(str(root)):
        if "config.json" in files:
            with io.open(os.path.join(path, "config.json"),
                         encoding="utf-8") as fh:
                out[path] = json.load(fh)
    return out


def _parity(root):
    """scripts/check_parity.py, in process. Returns (exit code, report)."""
    from scripts import check_parity
    buf, argv, cwd = io.StringIO(), sys.argv, os.getcwd()
    try:
        os.chdir(ROOT)
        sys.argv = ["check_parity", str(root)]
        with contextlib.redirect_stdout(buf):
            rc = check_parity.main()
    finally:
        sys.argv = argv
        os.chdir(cwd)
    return rc, buf.getvalue()


def _gen(root, arms, caps=CAPS, extra=(), protocol=None):
    """The real generator, as a subprocess. Never into results/."""
    cmd = [sys.executable, "-m", "configs.gen_campaign", "--root", str(root),
           "--datasets", "iwildcam", "--models", "MobileNetV3", "--caps"]
    cmd += list(caps) + ["--arms"] + list(arms) + list(extra)
    cmd += ["--protocol", str(protocol)] if protocol else []
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


@pytest.fixture(scope="session")
def protocol_yml():
    return load_protocol()


@pytest.fixture(scope="session")
def generated(tmp_path_factory):
    """ONE valid campaign, generated once. `--constraint-fp32` is not optional
    (2(u)) and neither is the reseed floor, so this is the minimum legal
    trained campaign: 3 named arms + 2 auto-added clippers x 2 caps x 4 seeds."""
    root = tmp_path_factory.mktemp("g4_valid") / "camp"
    rc, out = _gen(root, TRIO, extra=["--constraint-fp32"])
    assert rc == 0, "the generator refused a VALID campaign:\n%s" % out
    return str(root), _on_disk(root), out


def test_warmup_1_29_trained_30_0_posthoc_and_equal_compute(protocol_yml,
                                                            generated, tmp_path):
    """CLAUDE.md rule 1 / protocol.yml `protocol:`. Warm-up 50 saturates CE and
    every method becomes identical; warm-up 5 is a dead zone. So the two sides
    are 1+29 and 30+0, 30 optimizer epochs each, and nothing interpolates.
    NEGATIVE CONTROL: one arm at warm-up 50 must read UNEQUAL COMPUTE."""
    P, fails = protocol_yml, []
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
            fails.append("%s: %d+%d epochs, not 30"
                         % (cfg["arm"], h["warmup_epochs"],
                            h["constraint_epochs"]))
    if _parity(generated[0])[0] != 0:
        fails.append("check_parity REFUSED the valid generated campaign")

    def warmup_50(cfg):
        if cfg["arm"] == "tralo":
            cfg["hyperparams"]["warmup_epochs"] = 50
    rc, out = _parity(_campaign(tmp_path / "wu50", P, MIXED, mutate=warmup_50))
    if rc == 0 or "UNEQUAL COMPUTE" not in out:
        fails.append("warm-up 50 on one arm was NOT rejected (rc=%d)" % rc)
    if _parity(_campaign(tmp_path / "ok", P, MIXED))[0] != 0:
        fails.append("a 1+29 / 30+0 campaign was rejected")
    report(fails, "equal-compute defects")


def test_both_clippers_and_at_least_two_cap_levels(protocol_yml, generated,
                                                   tmp_path):
    """CLAUDE.md rule 2 / protocol.yml `mandatory_arms`. `clip` is the stronger
    quality bar and an arm-vs-arm delta is not a result until BOTH bars sit in
    the same campaign. NEGATIVE CONTROLS: the generator must refuse a single-cap
    campaign, and check_parity must refuse a single-cap tree on disk."""
    P, fails = protocol_yml, []
    if sorted(P["mandatory_arms"]) != ["clip", "focal_clip"]:
        fails.append("mandatory_arms is %s" % P["mandatory_arms"])
    arms = {c["arm"] for c in generated[1].values()}
    for bar in ("clip", "focal_clip"):
        if bar not in arms:
            fails.append("generator did not auto-add %s (got %s)"
                         % (bar, sorted(arms)))
    if len({c["constraint_tag"] for c in generated[1].values()}) < 2:
        fails.append("the generated campaign carries one cap level")
    rc, out = _gen(tmp_path / "onecap", TRIO, caps=[CAPS[0]],
                   extra=["--constraint-fp32"])
    if rc == 0 or "at least two cap levels" not in out:
        fails.append("a single-cap campaign GENERATED (rc=%d)" % rc)
    rc, out = _parity(_campaign(tmp_path / "p1", P, MIXED, caps=[CAPS[0]]))
    if rc == 0 or "single cap level" not in out:
        fails.append("check_parity accepted a single-cap tree (rc=%d)" % rc)
    if _parity(_campaign(tmp_path / "p2", P, MIXED))[0] != 0:
        fails.append("check_parity rejected a two-cap tree")
    report(fails, "clipper/cap-level defects")


def test_every_trained_arm_has_a_null_twin_and_the_reseed_floor(
        protocol_yml, generated, tmp_path):
    """CLAUDE.md 'Three rules' / protocol.yml `count_control`. Without the null
    no count trajectory is attributable -- CE alone swings the capped count
    242/227/324/233 -- and `tralo_reseed` is the RNG floor: the constraint moves
    that count RMS 75-95 items and a reseed moves it 83-95. `_null_of` resolves
    through `null_sibling`; matching on the name silently skipped `tralo_margin`,
    the arm that most needed one. NEGATIVE CONTROL: no floor, no campaign."""
    P, fails = protocol_yml, []
    if count_control_arms(P) != {"tralo_reseed"}:
        fails.append("count_control arms are %s" % sorted(count_control_arms(P)))
    for arm, spec in sorted(P["arms"].items()):
        if spec.get("phase") == "trained" and not arm.endswith("_null") \
                and _null_of(P, arm) not in P["arms"]:
            fails.append("%s resolves to null %r, which does not exist -- the "
                         "gate skips it silently" % (arm, _null_of(P, arm)))
    arms = {c["arm"] for c in generated[1].values()}
    for need in ("tralo_null", "tralo_reseed"):
        if need not in arms:
            fails.append("%s missing from the valid campaign" % need)
    rc, out = _gen(tmp_path / "nofloor", ["tralo", "tralo_null"],
                   extra=["--constraint-fp32"])
    if rc == 0 or "no reseed control" not in out:
        fails.append("trained campaign GENERATED with no reseed (rc=%d)" % rc)
    report(fails, "control-arm defects")


def test_constraint_fp32_is_the_dose_and_the_default_is_off(protocol_yml,
                                                            generated, tmp_path):
    """FRAMEWORK 2(u). true = 15284/15284 constraint steps over 532 runs and 6
    campaigns; false = 86.9% over 189, and that group IS the quarantine list.
    `taskwin1` was staged without it, landed 20/29, was killed at 3/48. The
    protocol default OFF is pinned deliberately -- it is the trap. NEGATIVE
    CONTROL: trained arms without the flag are refused, the refusal is scoped
    to campaigns that take a step, and the documented override still works."""
    P, fails = protocol_yml, []
    if P["constraint_phase"].get("constraint_fp32") is not False:
        fails.append("protocol default constraint_fp32 is %r; this gate assumes "
                     "the default-off trap"
                     % P["constraint_phase"].get("constraint_fp32"))
    for cfg in generated[1].values():        # LIVENESS: the flag is not inert
        hp = cfg["hyperparams"]
        if hp["constraint_epochs"] and hp.get("constraint_fp32") is not True:
            fails.append("%s carries constraint_fp32=%r after the flag"
                         % (cfg["arm"], hp.get("constraint_fp32")))
    rc, out = _gen(tmp_path / "fp16", TRIO)
    if rc == 0 or "constraint_fp32: false" not in out:
        fails.append("trained campaign GENERATED without fp32 (rc=%d)" % rc)
    if _gen(tmp_path / "posthoc", ["clip", "focal_clip"])[0] != 0:
        fails.append("a post-hoc-only campaign was refused for fp32")
    rc, out = _gen(tmp_path / "override", TRIO,
                   extra=["--allow-fp16-constraint"])
    if rc != 0 or "allow-fp16-constraint" not in out:
        fails.append("--allow-fp16-constraint did not warn-and-emit (rc=%d)" % rc)
    report(fails, "constraint-dose defects")


def test_one_code_version_stamp_across_the_campaign(protocol_yml, generated,
                                                    tmp_path):
    """CLAUDE.md infrastructure. `code_version` is `git rev-parse HEAD`, so ANY
    commit -- a docs-only one included -- desynchronises a staged campaign: the
    configs keep the old stamp while the runner would write the new one. TWO
    NEGATIVE CONTROLS: two stamps, and the subtler `unknown` everywhere, which
    gitver writes when git is unavailable and which satisfies every equality
    check -- one value is not one commit."""
    P, fails = protocol_yml, []
    stamps = {c["code_version"] for c in generated[1].values()}
    if len(stamps) != 1:
        fails.append("the generator emitted %d stamps: %s"
                     % (len(stamps), sorted(stamps)))

    def two(cfg):
        cfg["code_version"] = ("deadbeef" if cfg["arm"] == "tralo"
                               else "1111aaaa2222")
    for name, mut, want in [("two_stamps", two, "MIXED CODE VERSIONS"),
                            ("all_unknown",
                             lambda c: c.update(code_version="unknown"),
                             "CODE VERSION IS `unknown`")]:
        rc, out = _parity(_campaign(tmp_path / name, P, MIXED, mutate=mut))
        if rc == 0 or want not in out:
            fails.append("%s accepted (rc=%d, wanted %r)" % (name, rc, want))
    if _parity(_campaign(tmp_path / "onestamp", P, MIXED))[0] != 0:
        fails.append("a one-stamp campaign was rejected")
    report(fails, "code-version defects")


def test_lr_parity_and_the_lr_trap(protocol_yml, tmp_path):
    """2,972 completed pairs were made illegal by `lr_constraint` 5e-6 against
    `lr` 1e-4: a trained arm builds its constraint-phase optimizer on
    `lr_constraint`, so 29 of its 30 epochs run detuned. Cross-arm agreement
    does NOT catch it -- every arm agreed and the campaign printed PARITY OK --
    so the value is compared to `lr` itself. THREE NEGATIVE CONTROLS: a trapped
    protocol, an agreed-but-wrong tree, a cross-arm disagreement."""
    P, fails = protocol_yml, []
    if P["core"]["lr"] != P["constraint_phase"]["lr_constraint"]:
        fails.append("protocol lr %s != lr_constraint %s"
                     % (P["core"]["lr"], P["constraint_phase"]["lr_constraint"]))
    path = tmp_path / "trapped.yml"
    with io.open(str(path), "w", encoding="utf-8") as fh:
        fh.write(yaml.safe_dump(dict(P, constraint_phase=dict(
            P["constraint_phase"], lr_constraint=5e-6, constraint_fp32=True))))
    rc, out = _gen(tmp_path / "lrt", TRIO, protocol=path)
    if rc == 0 or "lr_constraint" not in out:
        fails.append("the generator emitted an LR-trapped campaign (rc=%d)" % rc)

    def trap(cfg):
        if "lr_constraint" in cfg["hyperparams"]:
            cfg["hyperparams"]["lr_constraint"] = 5e-6

    def disagree(cfg):
        if cfg["arm"] == "tralo":
            cfg["hyperparams"]["lr"] = 5e-6
    for name, mut, want in [("lr_agreed", trap, "THE LR TRAP"),
                            ("lr_split", disagree, "lr differs across the arms")]:
        rc, out = _parity(_campaign(tmp_path / name, P, MIXED, mutate=mut))
        if rc == 0 or want not in out:
            fails.append("%s accepted (rc=%d, wanted %r)" % (name, rc, want))
    if _parity(_campaign(tmp_path / "lr_ok", P, MIXED))[0] != 0:
        fails.append("an equal-lr campaign was rejected")
    report(fails, "learning-rate defects")


def test_tralo_lam0_rides_along_wherever_tralo_meets_a_subgradient_dual(
        protocol_yml, tmp_path):
    """FRAMEWORK 2(z3), measured on `dom1`, 24 runs per arm, at the GRADIENT
    level. The subgradient duals guard their step on `any(lambda > 0)` and
    update the dual at the END of the epoch, so epoch 1 logs grad norm 0.0 and
    they take 28 steps; `tralo` starts at lambda 0.06 and takes 29 -- a
    1-in-29 = 3.4% dose advantage in every head-to-head run here, invisible to
    `dose_landed`, which divides WITHIN an arm and printed 100.0% for all four.
    It is faithful to the published algorithms, so the fix is not hacking a
    baseline but `tralo_lam0`: `tralo` at lambda_init 0 with the ratchet intact
    (NOT `tralo_null`, which zeroes `lambda_step` too). NEGATIVE CONTROL: tralo
    beside fioretto with no lam0 arm."""
    P, fails, B = protocol_yml, [], protocol_yml["blocks"]
    if not (B["tralo"]["lambda_global"] > 0 and B["tralo"]["lambda_local"] > 0):
        fails.append("tralo no longer starts at lambda > 0; 2(z3) is stale")
    for dual in ("fioretto", "alm"):
        if B[dual].get("fioretto_lambda_init") != 0.0:
            fails.append("%s lambda_init is %r, not 0.0"
                         % (dual, B[dual].get("fioretto_lambda_init")))
    lam0 = B.get("tralo_lam0", {})
    if lam0.get("lambda_global") != 0.0 or lam0.get("lambda_local") != 0.0:
        fails.append("tralo_lam0 does not start at lambda 0: %r" % lam0)
    if not lam0.get("lambda_step"):
        fails.append("tralo_lam0 zeroes lambda_step too -- that is tralo_null")
    # The guard by SEMANTICS, not by name: 2(z3) calls it `has_work`, which is
    # what the fioretto trainers call it -- `hounie_rcl` calls the same
    # any(lambda > 0) test `has_active` and hard-codes lam 0.0 with no config
    # knob, so it is the one dual whose 28 steps are invisible in protocol.yml.
    for fam, names in [("fioretto_ldf", ("has_work",)),
                       ("hounie_rcl", ("has_work", "has_active"))]:
        src = io.open(os.path.join(ROOT, "src", "methodologies", fam,
                                   "train.py"), encoding="utf-8").read()
        if not any(n in src for n in names):
            fails.append("%s no longer guards on any(lambda > 0) %s" % (fam, names))
        if fam == "hounie_rcl" and "lam_g = {c: 0.0 for c in K_global}" not in src:
            fails.append("hounie_rcl no longer inits lambda to 0.0 in code")

    def head_start(root):
        arms = {c["arm"] for c in _on_disk(root).values()}
        return (bool(arms & {"fioretto", "hounie"}) and "tralo" in arms
                and "tralo_lam0" not in arms)
    base = ["clip", "tralo", "tralo_null", "fioretto"]
    if not head_start(_campaign(tmp_path / "headstart", P, base)):
        fails.append("tralo vs fioretto with no tralo_lam0 was NOT flagged")
    if head_start(_campaign(tmp_path / "clean", P, base + ["tralo_lam0"])):
        fails.append("a campaign carrying tralo_lam0 was flagged anyway")
    report(fails, "head-start defects")


def test_dead_arms_and_quarantined_campaigns_are_not_merely_unfinished(
        protocol_yml, tmp_path):
    """Two ways a campaign lies about being alive. Three arms once shipped with
    an undefined name in `train()`, burned 29 constraint epochs, died, were
    reset to `pending`, and the campaign came back looking merely unfinished
    with `audit_config` and `check_parity` both GREEN. The tell is an
    `error_log*.json` beside the config -- the GLOB, not the literal name, or
    tidying the log aside restores the blindness. Thirteen campaigns are
    quarantined, and `is_quarantined` walks every ANCESTOR because
    `--campaign results/iwc3/ViTB16` used to step past the marker at
    `results/iwc3`. POSITIVE CONTROL: pending with no crash log is in flight."""
    from scripts import quarantine
    P, fails = protocol_yml, []

    def dead(root):
        return {c["arm"] for d, c in _on_disk(root).items()
                if c.get("status") != "completed"
                and any(f.startswith("error_log") and f.endswith(".json")
                        for f in os.listdir(d))}

    def crash(cfg):
        if cfg["arm"] == "tralo":
            cfg["status"] = "pending"
    for name, log in [("dead", "error_log.json"),
                      ("tidied", "error_log.2026-08-21.json")]:
        root = _campaign(tmp_path / name, P, MIXED, mutate=crash)
        for d, cfg in _on_disk(root).items():
            if cfg["arm"] == "tralo":
                with io.open(os.path.join(d, log), "w", encoding="utf-8") as fh:
                    json.dump({"exception_type": "NameError"}, fh)
        if dead(root) != {"tralo"}:
            fails.append("crashed arm read as pending (%s): %s"
                         % (log, sorted(dead(root))))
    if dead(_campaign(tmp_path / "inflight", P, MIXED, mutate=crash)):
        fails.append("an in-flight campaign was called dead")
    if "error_log*.json" not in io.open(
            os.path.join(ROOT, "scripts", "full_panel.py"),
            encoding="utf-8").read():
        fails.append("full_panel no longer globs error_log*.json")
    # The count is NOT hardcoded here. A literal in a third place makes every
    # legitimate quarantine fail this gate, and the fix becomes "bump the
    # number" -- a chore, not a check. Read the word CLAUDE.md actually prints
    # and require the registry to match it, so adding a campaign forces the doc
    # update and nothing else.
    _WORDS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
              "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
              "eighteen": 18, "nineteen": 19, "twenty": 20}
    _m = re.search(r"([A-Za-z]+)\s+campaigns are marked",
                   io.open(os.path.join(ROOT, "CLAUDE.md"),
                           encoding="utf-8").read())
    if not _m:
        fails.append("CLAUDE.md no longer states how many campaigns are marked")
    elif _m.group(1).lower() not in _WORDS:
        # fail CLOSED: an unparseable word must not read as agreement
        fails.append("CLAUDE.md says %r campaigns are marked, which is not a "
                     "number word this gate knows" % _m.group(1))
    elif len(quarantine.REGISTRY) != _WORDS[_m.group(1).lower()]:
        fails.append("%d quarantined campaigns, CLAUDE.md says %s"
                     % (len(quarantine.REGISTRY), _m.group(1).upper()))
    if any(e.get("scorable") for e in quarantine.REGISTRY.values()):
        fails.append("a quarantine entry claims to be scorable")
    q = _campaign(tmp_path / "quar", P, MIXED)
    if quarantine.is_quarantined(q) is not None:
        fails.append("a clean campaign read as quarantined")
    with io.open(os.path.join(q, quarantine.MARKER), "w",
                 encoding="utf-8") as fh:
        json.dump({"reason": "synthetic", "scorable": False}, fh)
    if quarantine.is_quarantined(q) is None:
        fails.append("a marked campaign read as clean")
    if quarantine.is_quarantined(os.path.join(q, "MobileNetV3")) is None:
        fails.append("scoring one backbone walked past the root marker")
    report(fails, "liveness defects")


@pytest.mark.skipif(not os.path.exists(SLICE),
                    reason="iwildcam/oodslice absent -- the task-window gate "
                           "CANNOT run and is NOT being checked here")
def test_the_generator_refuses_caps_outside_the_measured_task_window(tmp_path):
    """FRAMEWORK 2(z16)/2(z17). A cap poses a question only where it evicts
    >=10 predictions, leaves errors inside K and cuts at p@K < 0.99. On all four
    backbones, 24 of 24 (backbone x class x cap) cells at L20/L30/L50 fail at
    least one -- the best single explanation on record for why so many arms
    tied. NEGATIVE CONTROL: an L20/L30 campaign is refused. LIVENESS both ways:
    `--allow-nontask` generates and says what it let through, and `taskwin2`'s
    measured caps still pass."""
    fails, fp32 = [], ["--constraint-fp32"]
    rc, out = _gen(tmp_path / "dead", TRIO, caps=DEAD_CAPS, extra=fp32)
    if rc == 0 or "OUTSIDE the measured task window" not in out:
        fails.append("an L20/L30 campaign GENERATED (rc=%d)" % rc)
    rc, out = _gen(tmp_path / "forced", TRIO, caps=DEAD_CAPS,
                   extra=fp32 + ["--allow-nontask"])
    if rc != 0 or "pose NO question" not in out:
        fails.append("--allow-nontask did not emit-and-say-so (rc=%d)" % rc)
    rc, out = _gen(tmp_path / "live", TRIO, extra=fp32)
    if rc != 0:
        fails.append("the measured in-window caps %s were refused:\n%s"
                     % (CAPS, out))
    report(fails, "task-window defects")
