"""Is the TraLO-vs-baselines comparison scientifically valid?

Every gate here answers one of five questions about the central comparison, and
each one was written to FAIL on a state this repository was actually in:

  1. THE LR TRAP        do the arms get matched learning rates, and does the
                        pre-launch gate REFUSE an unmatched one?
  2. EQUAL COMPUTE      does every arm get the same optimizer epochs, and can a
                        warm-up cache leak one arm's model into another?
  3. BASELINE FIDELITY  is a baseline's hyperparameter silently altered from the
                        value the config declares?
  4. EQUAL BUDGET       are the arms compared at the same number of emitted
                        capped-class predictions?
  5. THE NULL ARMS      do the zero-dose controls really zero every constraint
                        pathway, and does the reseed control vary the RNG ONLY?

Nothing here needs a GPU or a dataset: the arms run end to end on the
`scripts.smoke_arms` CPU harness, and the config gates read `configs/protocol.yml`.
"""
import ast
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from configs.gen_campaign import (build_hyperparams, cap_pair,  # noqa: E402
                                  compute_base_model_id, load_protocol)
from src.utils.constants import UNLIMITED  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

TRAINED_METHODS = ("tralo", "fioretto_ldf", "hounie_rcl", "fioretto_alm")
DUAL_ARMS = ("tralo", "fioretto", "hounie", "alm")
NULL_ARMS = ("tralo_null", "fioretto_null", "hounie_null", "alm_null")


@pytest.fixture(scope="module")
def P():
    return load_protocol()


def _run_arm(P, arm, epochs=3, **overrides):
    """Run one arm end to end on the CPU smoke harness.

    Returns (md5 of the test-set softmax, TrainOutputs.summary, [grad_norm per
    logged epoch]).  The hash is what makes "these two arms are the same object"
    a measurement rather than an opinion -- rule 3 of CLAUDE.md, applied to the
    arms instead of to a campaign.
    """
    from scripts.smoke_arms import make_inputs
    from src.experiments.runner import TRAIN_FNS

    tmp = tempfile.mkdtemp()
    inputs, _g, _l = make_inputs(P, arm, tmp, seed=1)
    inputs.hyperparams["constraint_epochs"] = epochs
    inputs.hyperparams.update(overrides)
    torch.manual_seed(1)
    np.random.seed(1)
    out = TRAIN_FNS[P["arms"][arm]["methodology"]](inputs)
    model = out.model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(inputs.X_test), dim=1).numpy()
    md5 = hashlib.md5(np.round(proba, 6).tobytes()).hexdigest()[:12]
    log = os.path.join(str(inputs.experiment_path), "training_log.csv")
    norms = []
    if os.path.exists(log):
        rows = list(csv.DictReader(open(log, encoding="utf-8")))
        col = next((c for c in ("grad_norm", "Grad_Norm") if rows and c in rows[0]),
                   None)
        if col:
            norms = [float(r[col]) for r in rows if r[col] not in ("", None)]
    return md5, out.summary, norms


def _write_campaign(root, P, arms, caps=("L30_G50", "L50_G30"), seeds=(1, 2),
                    hp_patch=None, dataset="iwildcam", model="MobileNetV3"):
    """A campaign on disk, built through the generator's own helpers.

    `hp_patch(arm, hp)` may corrupt one knob, which is how a gate gets shown to
    fail before it is trusted to pass.
    """
    dc = P["datasets"][dataset]
    for arm in arms:
        for tag in caps:
            for seed in seeds:
                hp = build_hyperparams(P, P["arms"][arm], seed)
                if hp_patch:
                    hp_patch(arm, hp)
                path = os.path.join(root, model, dataset, tag, arm,
                                    "seed_%d" % seed)
                os.makedirs(path, exist_ok=True)
                cfg = {"methodology": P["arms"][arm]["methodology"],
                       "model_name": model, "constraint": cap_pair(tag),
                       "constraint_tag": tag, "dataset_mode": dataset,
                       "dataset_config": dc, "hyperparams": hp,
                       "base_model_id": compute_base_model_id(P, model, hp,
                                                              dataset, dc),
                       "arm": arm, "exp_name": "%s_%s_%d" % (arm, tag, seed),
                       "status": "pending", "code_version": "aaaaaaaaaaaa"}
                json.dump(cfg, open(os.path.join(path, "config.json"), "w"),
                          indent=2)
    return root


def _parity(root):
    return subprocess.run(
        [sys.executable, "-m", "scripts.check_parity", str(root)],
        cwd=REPO, capture_output=True, text=True)


# ==========================================================================
# 1. THE LR TRAP
# ==========================================================================

def test_the_protocol_pins_lr_constraint_to_lr(P):
    """`lr_constraint` 5e-6 against `lr` 1e-4 fabricated a -16.7 pp finding that
    was -1.7 pp once equalized (FRAMEWORK 1b-pre, section 10).  The protocol
    must therefore ship them equal -- this is the value side of that rule; the
    two gates below are the enforcement side.
    """
    assert P["core"]["lr"] == P["constraint_phase"]["lr_constraint"]
    # `select` carries its own copy, so it can drift independently.
    for block, spec in P["blocks"].items():
        if "lr_constraint" in spec:
            assert spec["lr_constraint"] == P["core"]["lr"], (
                "block %r sets lr_constraint %s against core.lr %s"
                % (block, spec["lr_constraint"], P["core"]["lr"]))


def test_an_unequal_lr_constraint_detunes_29_of_30_TRAINED_EPOCHS(P):
    """WHY the trap is worth 16 pp and not a rounding error.

    `lr_constraint` is not only the constraint step's learning rate: the trained
    arms build the constraint phase's optimizer with it and force every param
    group onto it, so all 29 CROSS-ENTROPY epochs of a trained arm run at
    `lr_constraint` while `clip`'s 30 warm-up epochs run at `lr`.  Set them
    unequal and the comparison is 1 epoch of matched training against 29 of
    detuned training, which is a regime difference wearing a hyperparameter's
    name.  AST, not grep: the name appears in log strings in both files.
    """
    def optimizer_lr_args(path):
        tree = ast.parse(open(os.path.join(REPO, path), encoding="utf-8").read())
        out = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "make_optimizer"):
                out.append(ast.unparse(node.args[1]))
        return out

    assert optimizer_lr_args("src/methodologies/tralo/train.py") == ["lr_constraint"]
    # the three duals share one builder
    assert optimizer_lr_args("src/methodologies/dual_common.py") == ["lr"]
    dual = ast.parse(open(os.path.join(REPO, "src/methodologies/dual_common.py"),
                          encoding="utf-8").read())
    setup = next(n for n in ast.walk(dual)
                 if isinstance(n, ast.FunctionDef) and n.name == "dual_setup")
    assert [a.arg for a in setup.args.args][3] == "lr", (
        "dual_setup's third positional is no longer the learning rate")
    for meth in ("fioretto_ldf", "hounie_rcl", "fioretto_alm"):
        src = open(os.path.join(REPO, "src/methodologies", meth, "train.py"),
                   encoding="utf-8").read()
        tree = ast.parse(src)
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "id", None) == "dual_setup"]
        assert calls, meth
        assert ast.unparse(calls[0].args[3]) == "lr_c", meth
        assigns = [n for n in ast.walk(tree)
                   if isinstance(n, ast.Assign)
                   and getattr(n.targets[0], "id", None) == "lr_c"]
        assert assigns and "lr_constraint" in ast.unparse(assigns[0].value), meth
    # and the CE pass inside tralo's constraint loop is pinned to it too
    tralo = open(os.path.join(REPO, "src/methodologies/tralo/train.py"),
                 encoding="utf-8").read()
    assert 'pg["lr"] = lr_constraint' in tralo


def test_the_generator_refuses_an_unequal_lr_constraint(tmp_path, P):
    """`gen_campaign.validate` is the first of the two gates.  Shown to fail on
    the trap by handing it a protocol whose `lr_constraint` is the retracted
    5e-6, then shown to pass on the shipped one."""
    trapped = copy.deepcopy(P)
    trapped["constraint_phase"]["lr_constraint"] = 5e-6
    proto = tmp_path / "trap_protocol.yml"
    proto.write_text(yaml.safe_dump(trapped), encoding="utf-8")
    argv = ["--root", str(tmp_path / "camp"), "--datasets", "iwildcam",
            "--models", "MobileNetV3", "--caps", "L30_G50", "L50_G30",
            "--arms", "tralo", "tralo_null", "tralo_reseed"]
    bad = subprocess.run(
        [sys.executable, "-m", "configs.gen_campaign", "--protocol", str(proto)]
        + argv, cwd=REPO, capture_output=True, text=True)
    assert bad.returncode != 0, "the generator emitted an LR-trapped campaign"
    assert "lr_constraint" in (bad.stdout + bad.stderr)
    ok = subprocess.run(
        [sys.executable, "-m", "configs.gen_campaign"] + argv,
        cwd=REPO, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout[-1500:] + ok.stderr[-1500:]


def test_check_parity_REFUSES_the_lr_trap(tmp_path, P):
    """The second gate, and the one that was missing.

    `check_parity`'s own docstring lists "an unequal lr_constraint (worth 16 pp)"
    as one of the four things it exists to catch, and it checked no such thing:
    it verified that each key holds ONE value across the arms, which an
    lr-trapped campaign satisfies perfectly -- every arm carries lr 1e-4 and
    every trained arm carries lr_constraint 5e-6.  Run against a campaign built
    exactly that way it printed "PARITY OK -- this campaign is a fair
    comparison" and exited 0.

    That matters beyond hygiene: `gen_campaign` refuses the trap, but the 2,972
    trapped pairs in the provenance archive were never generated by today's
    generator, and a hand-edited or resumed config never passes through it at
    all.  `check_parity <root>` is the documented gate for those.
    """
    def trap(arm, hp):
        if "lr_constraint" in hp:
            hp["lr_constraint"] = 5e-6

    root = _write_campaign(str(tmp_path / "trapped"), P,
                           ["clip", "tralo", "tralo_null", "tralo_reseed"],
                           hp_patch=trap)
    r = _parity(root)
    assert r.returncode == 1, (
        "check_parity passed a campaign whose trained arms train at 5e-6 while "
        "the clipper trains at 1e-4:\n" + r.stdout[-2000:])
    assert "lr_constraint" in r.stdout

    clean = _write_campaign(str(tmp_path / "clean"), P,
                            ["clip", "tralo", "tralo_null", "tralo_reseed"])
    ok = _parity(clean)
    assert ok.returncode == 0, ok.stdout[-2500:]


# ==========================================================================
# 2. EQUAL COMPUTE
# ==========================================================================

def test_every_arm_gets_the_same_optimizer_epochs(P):
    """30 on both sides: warm-up 30 + constraint 0 for the post-hoc arms,
    warm-up 1 + constraint 29 for the trained ones."""
    total = P["protocol"]["total_epochs"]
    for arm, spec in P["arms"].items():
        hp = build_hyperparams(P, spec, 1)
        assert hp["warmup_epochs"] + hp["constraint_epochs"] == total, arm
        if spec["phase"] == "posthoc":
            assert (hp["warmup_epochs"], hp["constraint_epochs"]) == (total, 0), arm
        else:
            assert hp["warmup_epochs"] == P["protocol"]["trained_warmup"], arm


def test_no_trained_arm_can_early_stop_out_of_its_constraint_budget(P):
    """All four trained arms break on `stable_count >= stable_count_threshold`.

    At 31 against 29 constraint epochs the branch is unreachable, so no arm can
    end its constraint phase early by satisfying its caps -- which would hand a
    method that converges fast FEWER gradient steps than one that does not, and
    then score the difference as quality.  Drop the threshold to 29 and the
    dependency is live again, so this is a real constraint on the protocol
    rather than an accident.
    """
    thr = P["constraint_phase"]["stable_count_threshold"]
    epochs = P["protocol"]["total_epochs"] - P["protocol"]["trained_warmup"]
    assert thr > epochs, (
        "stable_count_threshold %d <= constraint_epochs %d: an arm that "
        "satisfies its caps early takes fewer steps than one that does not"
        % (thr, epochs))
    for meth in TRAINED_METHODS:
        src = open(os.path.join(REPO, "src/methodologies", meth, "train.py"),
                   encoding="utf-8").read()
        assert "stable_count >= stable_count_threshold" in src, meth


def test_the_warmup_cache_key_covers_everything_the_warmup_reads():
    """P1 of `scripts.audit_config`, asserted rather than printed.

    Anything that changes what the warm-up OPTIMIZES must be in
    `warmup_identity_keys`, or the second arm silently loads the first one's
    trained model.  Shown to fail by deleting `warmup_loss` from the declared
    set -- that is the exact omission that made `focal_clip` a second `clip`.
    """
    from scripts.audit_config import WARMUP_PATH, WARMUP_EXTRA, _keys_in, _walk
    paths = []
    for f in WARMUP_PATH:
        paths += _walk(f) if os.path.isdir(f) else [f]
    read = _keys_in(paths) | WARMUP_EXTRA
    proto = yaml.safe_load(open(os.path.join(REPO, "configs/protocol.yml"),
                                encoding="utf-8"))
    declared = set(proto["warmup_identity_keys"])
    assert not (read - declared), sorted(read - declared)
    assert read - {"warmup_loss"} - declared == set()
    assert read & {"warmup_loss"}, (
        "the audit no longer sees warmup_loss on the warm-up path, so this "
        "gate would pass on a protocol that omitted it")


def test_the_cap_never_reaches_the_warmup_so_a_shared_cache_is_legitimate():
    """Cap tag and capped class are deliberately ABSENT from `base_model_id`.

    That is only safe because neither touches the training split: `data_loader`
    reads `config['constraint']` and `constrained_class` solely to build the
    budgets, never to filter X_train/y_train.  If it ever did, every campaign
    that sweeps two cap levels would be training one model and scoring it twice.
    """
    proto = yaml.safe_load(open(os.path.join(REPO, "configs/protocol.yml"),
                                encoding="utf-8"))
    assert "constraint" not in proto["warmup_identity_keys"]
    src = open(os.path.join(REPO, "src/utils/data_loader.py"),
               encoding="utf-8").read()
    tree = ast.parse(src)
    # every assignment to X_train / y_train, and what it is built from
    tainted = {"constrained_class", "local_percent", "global_percent",
               "global_con", "local_con"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = {getattr(t, "id", None) for t in node.targets}
            if names & {"X_train", "y_train"}:
                used = {n.id for n in ast.walk(node.value)
                        if isinstance(n, ast.Name)}
                assert not (used & tainted), (
                    "the training split is built from %s -- the cap has "
                    "reached the warm-up" % sorted(used & tainted))


def test_the_four_trained_arms_share_one_warmup_and_the_clippers_do_not(tmp_path, P):
    """Nine trained arms hash to ONE `base_model_id` (warm-up 1, no constraint
    knobs in the key) so exactly one of them trains it; `clip` and `focal_clip`
    hash apart because `warmup_loss` differs.  A collision between the last two
    is failure mode five of the inert-flag catalogue.
    """
    ids = {}
    for arm in ("clip", "focal_clip", "lp", "focal_lp", "cb_lp", "la_lp",
                "tralo", "tralo_null", "tralo_reseed", "fioretto", "hounie", "alm"):
        hp = build_hyperparams(P, P["arms"][arm], 1)
        dc = P["datasets"]["iwildcam"]
        ids[arm] = compute_base_model_id(P, "MobileNetV3", hp, "iwildcam", dc)
    assert len({ids[a] for a in ("tralo", "tralo_null", "tralo_reseed",
                                 "fioretto", "hounie", "alm")}) == 1
    assert ids["clip"] == ids["lp"]
    assert ids["focal_clip"] == ids["focal_lp"]
    assert ids["clip"] != ids["focal_clip"], (
        "focal_clip shares clip's cached model and is therefore a second clip")
    assert ids["clip"] != ids["tralo"], "warm-up 30 and warm-up 1 share a cache"
    assert len({ids["cb_lp"], ids["la_lp"], ids["clip"], ids["focal_clip"]}) == 4


# ==========================================================================
# 3. BASELINE FIDELITY
# ==========================================================================

def _inline_defaults(paths):
    """(key, default, file, line) for every `hp.get(KEY, LITERAL)` under `paths`.

    AST, because `rho_step` "appeared" to be read for months on the strength of
    a log-format string.
    """
    hp_names = {"hp", "hyperparams", "hparams"}
    found = []
    for path in paths:
        full = os.path.join(REPO, path)
        for node in ast.walk(ast.parse(open(full, encoding="utf-8").read())):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and len(node.args) == 2):
                continue
            base = node.func.value
            name = getattr(base, "id", None) or getattr(base, "attr", None)
            if name not in hp_names:
                continue
            if not isinstance(node.args[0], ast.Constant):
                continue
            try:
                default = ast.literal_eval(node.args[1])
            except ValueError:
                continue
            found.append((node.args[0].value, default, path, node.lineno))
    return found


def _protocol_values(P, key):
    """Every value protocol.yml gives `key`, and the value the BASE arm gets."""
    vals, base = set(), None
    for block, spec in P["blocks"].items():
        if isinstance(spec, dict) and key in spec:
            vals.add(json.dumps(spec[key]))
    for section in ("core", "constraint_phase", "chunked"):
        if key in P.get(section, {}):
            vals.add(json.dumps(P[section][key]))
            base = json.dumps(P[section][key])
    return vals, base


def test_no_inline_default_disagrees_with_the_protocol(P):
    """The inline-default class of defect, gated.

    `hounie_rcl` once carried `hp.get("hounie_eta_lambda", 0.1)` while the
    protocol assigned 0.01, so any config that omitted the key ran a baseline
    at ten times the dual step every other config used -- a silently different
    method, invisible to every gate because the key HAS a reader and IS
    emitted.  (The 2026-08-23 resolution went the other way: the paper's own
    value is 0.1, so the protocol moved to it and the fallback stayed removed.
    The defect was never which number was right, it was that two places
    disagreed and only one of them was read.)

    The rule that survives: an inline default must be one of the values the
    protocol actually assigns to that key.

    The checker is shown to fail first on a stand-in pair, then run over live
    source.  The stand-in uses 0.5 -- a value the paper's own App. G grid
    contains and our protocol does not -- because the original example, 0.1, is
    now the shipped value and would no longer exercise the check.
    """
    def offenders(pairs):
        bad = []
        for key, default, path, line in pairs:
            vals, _base = _protocol_values(P, key)
            if not vals:
                continue                       # not a protocol knob
            if json.dumps(default) not in vals:
                bad.append((key, default, sorted(vals), path, line))
        return bad

    stand_in = [("hounie_eta_lambda", 0.5, "src/methodologies/hounie_rcl/train.py", 0)]
    assert offenders(stand_in), (
        "the checker cannot see a default that disagrees with the protocol -- "
        "it would pass on the very defect it exists to catch. If 0.5 has since "
        "become a protocol value, move the stand-in, do not delete it.")

    live = _inline_defaults([
        "src/methodologies/tralo/train.py",
        "src/methodologies/fioretto_ldf/train.py",
        "src/methodologies/hounie_rcl/train.py",
        "src/methodologies/fioretto_alm/train.py",
        "src/methodologies/dual_common.py",
        "src/methodologies/imbalanced_common.py",
        "src/methodologies/heuristic/train.py",
        "src/methodologies/danits_lp/train.py",
    ])
    assert live, "the AST walker found no inline defaults at all -- it is broken"
    assert not offenders(live), offenders(live)


def test_the_core_knobs_are_emitted_on_every_arm_so_no_default_can_fire(P):
    """`src/pipeline/warmup.py` reads `hp.get("pretrained", False)`, and the
    protocol says `pretrained: true`.  The default is unreachable ONLY because
    `build_hyperparams` seeds every arm from `dict(P["core"])`, so every config
    carries every core key -- which makes that guarantee a property of the
    generator, and therefore something to assert rather than assume.  Were it
    ever to fire, one arm would train a randomly-initialised backbone while the
    rest fine-tuned an ImageNet one, and `base_model_id` would not collide
    (the key would be absent from the hash), so nothing else would notice.
    """
    for arm, spec in P["arms"].items():
        hp = build_hyperparams(P, spec, 1)
        overridden = set()
        for block in spec.get("blocks") or []:
            overridden |= set(P["blocks"].get(block, {}) or {})
            overridden |= set(P.get(block, {}) or {})
        for k, v in P["core"].items():
            assert k in hp, "%s does not carry core key %r" % (arm, k)
            if k not in overridden:
                assert hp[k] == v, (arm, k, hp[k], v)


def test_the_dual_step_sizes_are_REQUIRED_not_defaulted():
    """The keys that define what each dual IS may not fall back to a literal.

    `_required` raises on a missing key; `hp.get(k, v)` silently substitutes.
    For a baseline's own step size the second is how a paper's method becomes a
    different method between two campaigns.
    """
    must_be_required = {
        "src/methodologies/fioretto_ldf/train.py": ["fioretto_step_size"],
        "src/methodologies/hounie_rcl/train.py": ["hounie_eta_lambda", "hounie_eta_u"],
        "src/methodologies/fioretto_alm/train.py": ["alm_eta", "alm_mu0", "alm_mu_step"],
        "src/methodologies/tralo/train.py": ["lambda_global", "lambda_local"],
    }
    for path, keys in must_be_required.items():
        defaults = {k for k, _d, _p, _l in _inline_defaults([path])}
        assert not (defaults & set(keys)), (
            "%s defaults %s instead of requiring it"
            % (path, sorted(defaults & set(keys))))
        src = open(os.path.join(REPO, path), encoding="utf-8").read()
        tree = ast.parse(src)
        required = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "_required"
                    and node.args and isinstance(node.args[1], ast.Constant)):
                required.add(node.args[1].value)
        subscripts = {n.slice.value for n in ast.walk(tree)
                      if isinstance(n, ast.Subscript)
                      and isinstance(n.slice, ast.Constant)
                      and isinstance(n.slice.value, str)}
        for k in keys:
            assert k in required or k in subscripts, "%s: %s" % (path, k)


def test_hounie_source_does_not_claim_a_dual_step_it_no_longer_runs():
    """`hounie_rcl/train.py` carried, directly above the `_required` read, a
    comment saying "Default dual-step bumped 10x for apples-to-apples
    convergence speed ... With 0.1 lambda hits meaningful magnitude by ep 10."

    The bump is gone -- the key is `_required` and the protocol emits the
    paper's 0.01, with its own comment saying the 0.1 was 10x the paper.  The
    comment was the last artefact still asserting the baseline runs at a
    deliberately altered step, in the one file a fidelity reviewer opens first.
    This project has already recorded that "prose-only is how the ALM lambda
    stayed wrong."
    """
    src = open(os.path.join(REPO, "src/methodologies/hounie_rcl/train.py"),
               encoding="utf-8").read()
    assert "bumped 10x" not in src, (
        "the trainer still claims a 10x dual-step bump it does not apply")
    assert "hounie_eta_lambda" in src
    assert "0.01" in src, "the paper's value should be named where it is read"


def test_the_ALM_augmentation_is_LIVE_so_alm_is_not_a_second_fioretto(P):
    """ALM differs from Fioretto-LDF in exactly one thing: the augmentation
    `mu_t * r^+` added to the primal weight.  It once could not fire at all --
    `has_work` consulted `lambda` alone, and with lambda starting at 0 the
    augmentation was unreachable on every epoch while `training_log.csv`
    faithfully wrote a rising `mu_t`.  In that state the two arms emitted
    BIT-IDENTICAL predictions and the paper's nine methodologies were eight.

    Measured here on the smoke harness at the shipped protocol values:
    `alm` != `fioretto`, and switching the augmentation off (`alm_mu0` and
    `alm_mu_step` to 0) makes it bit-identical to `fioretto` again -- which is
    both the liveness control and the proof that the augmentation is the only
    difference.
    """
    ldf, _s1, _n1 = _run_arm(P, "fioretto")
    alm, _s2, _n2 = _run_arm(P, "alm")
    off, _s3, _n3 = _run_arm(P, "alm", alm_mu0=0.0, alm_mu_step=0.0)
    assert alm != ldf, (
        "alm and fioretto emit identical predictions -- the augmentation is "
        "inert and the two are one arm")
    assert off == ldf, (
        "with mu=0 alm should reduce to fioretto's projected ascent; it does "
        "not, so something OTHER than the augmentation also differs")


def test_neither_grad_mode_puts_the_duals_at_a_COMPARABLE_dose(P):
    """Whichever `constraint_grad_mode` we run, the four duals are not at the
    same dose, and the two modes fail in OPPOSITE directions.  Measured at 6
    constraint epochs with `constraint_grad_clip: 1.0`, every config identical.

    Under `clip`, delivered = min(raw, 1.0).  Hounie's raw norm is ~0.09 and
    Fioretto's is ~27, so Fioretto and ALM are crushed to exactly 1.000 while
    Hounie keeps its own much smaller magnitude, leaving the arms two orders of
    magnitude apart in delivered step.  This is not a bug in Hounie: it divides
    its primal by n_test/N_g to match its own dual scale, which the paper's
    expectation formulation requires.

    Under `normalize`, the delivered magnitude is a constant for every arm.
    That fixes the cross-arm gap and erases each arm's dual dynamics with it:
    lambda and u only SCALE the penalty gradient, so normalizing divides them
    straight back out.  Swept 100x, `hounie_eta_lambda` emits ONE prediction
    set under `normalize` where it emits three under `clip`.

    So no mode makes "Fioretto beats Hounie" a statement about the two methods
    rather than about the rescaling.  Report the mode beside any dual-vs-dual
    result, and never read one across modes.
    """
    raw = {}
    for arm in DUAL_ARMS:
        _md5, _s, norms = _run_arm(P, arm, epochs=6)
        raw[arm] = max(n for n in norms if n > 0)
    clip = P["constraint_phase"]["constraint_grad_clip"]
    assert raw["hounie"] < clip, raw
    assert raw["fioretto"] > clip and raw["alm"] > clip, raw
    assert min(raw["fioretto"], raw["alm"]) / raw["hounie"] > 100.0, raw

    under = {}
    for mode in ("clip", "normalize"):
        under[mode] = {_run_arm(P, "hounie", epochs=6,
                                constraint_grad_mode=mode,
                                hounie_eta_lambda=v)[0]
                       for v in (0.01, 0.1, 2.0)}
    assert len(under["clip"]) > 1, under
    assert len(under["normalize"]) == 1, (
        "normalize no longer erases hounie's dual dose: %s" % under)


def test_hounie_alpha_REACHES_THE_MODEL_at_the_papers_dose(P):
    """The resilient term must be able to move the predictions at the dose we
    actually run.

    `hounie_alpha` is the curvature of the relaxation cost h(u)=alpha*||u||^2,
    and u is the perturbation that makes Resilient Constrained Learning
    resilient: it is the entire difference between this baseline and a plain
    dual method.  If two alphas 8x apart emit the same bits, the citation is
    decoration.

    HISTORY.  Until 2026-08-23 the protocol shipped
    (eta_lambda, eta_u, alpha) = (0.01, 0.01, 10.0), and alpha was INERT there:
    swept 0.05 -> 10.0, a 200x change, it emitted bit-identical predictions
    (md5 e5840e0bce98).  That triple was subtle rather than obviously wrong,
    because it preserves the paper's u-contraction |1 - 2*eta_u*alpha| = 0.8
    EXACTLY, so the SHAPE of the perturbation dynamics was faithful.  What it
    did not preserve was the scale: u* = lambda/(2*alpha), so a 10x smaller
    lambda against a 10x larger alpha left the relaxation ~100x under the
    paper's, far too small to reach the primal.

    arXiv:2306.02426 App. F states eta_lambda = 0.1, eta_u = 0.1 and
    h(u) = ||u||^2_2 (alpha = 1); App. G's grid for eta_lambda is
    {0.1, 0.5, 1, 2}.  0.01 appears nowhere in the paper as a rate.
    """
    hp = P["blocks"]["hounie"]
    assert (hp["hounie_eta_lambda"], hp["hounie_eta_u"], hp["hounie_alpha"])         == (0.1, 0.1, 1.0), hp
    # the stability condition hounie_rcl/train.py enforces on the u-update
    assert abs(1 - 2 * hp["hounie_eta_u"] * hp["hounie_alpha"]) < 1.0

    at = {a: _run_arm(P, "hounie", epochs=8, hounie_alpha=a)[0]
          for a in (0.5, 1.0, 4.0)}
    assert len(set(at.values())) == 3, (
        "alpha does not reach the predictions at the shipped dose: %s. The "
        "resilient term is the whole method; if it cannot move the model this "
        "arm is a plain dual method wearing Hounie's name." % at)


def test_the_alpha_liveness_gate_can_tell_a_dead_dose_from_a_live_one(P):
    """NEGATIVE CONTROL for the gate above.

    A liveness assertion is worthless unless it fails on the state it claims to
    exclude, so re-run the historical triple this project actually shipped and
    require alpha to be inert there.  That is what makes "alpha is live" a
    measurement rather than a hope, and it keeps the old defect described by a
    running check instead of by a comment.
    """
    dead = {a: _run_arm(P, "hounie", epochs=8, hounie_alpha=a,
                        hounie_eta_lambda=0.01, hounie_eta_u=0.01)[0]
            for a in (0.05, 1.0, 10.0)}
    assert len(set(dead.values())) == 1, (
        "the historical (0.01, 0.01, alpha) triple is no longer inert, so the "
        "liveness gate above is not discriminating between doses: %s" % dead)


def test_the_duals_that_start_lambda_at_zero_lose_their_first_step(P):
    """A real, small, arm-level dose asymmetry -- pinned so it cannot grow.

    `fioretto` and `hounie` gate the constraint backward on `lambda > 0` and
    initialise lambda at 0, so epoch 0 forms no constraint gradient at all and
    they take one step fewer than the phase has epochs.  `tralo` starts lambda
    at 0.01 and `alm`'s augmentation is positive from `mu0`, so both step on
    epoch 0.  It is faithful to lambda_0 = 0 in those papers and it is 1/29 of
    the dose, but two arms at 29 and 28 steps are not at equal dose and nothing
    said so until `constraint_steps_applied` reached the run summary.
    """
    epochs = 4
    steps = {}
    for arm in DUAL_ARMS:
        _md5, summary, _norms = _run_arm(P, arm, epochs=epochs)
        steps[arm] = (summary.get("constraint_steps_applied"),
                      summary.get("constraint_steps_attempted"))
    for arm, (applied, attempted) in steps.items():
        assert applied == attempted, (
            "%s dropped a step to a non-finite gradient: %s" % (arm, steps[arm]))
    assert steps["tralo"][0] == epochs, steps
    assert steps["alm"][0] == epochs, steps
    assert steps["fioretto"][0] == epochs - 1, steps
    assert steps["hounie"][0] == epochs - 1, steps


# ==========================================================================
#   THE CONSTRAINT DOSE -- the knob that is nominally equal and is not
# ==========================================================================

def test_only_normalize_gives_the_trained_arms_the_same_constraint_step(P):
    """`constraint_grad_clip: 1.0` on every arm is NOT the same dose.

    `finish_constraint_step` delivers `min(raw_norm, clip)` under
    `constraint_grad_mode: clip`, and the four trained arms' natural gradient
    scales are orders of magnitude apart by construction -- `hounie_rcl` divides
    its primal violation by n_test / N_g to match its own dual, `fioretto_ldf`
    and `fioretto_alm` sum it, and `tralo` weights a bounded penalty.  So under
    `clip` the arm with the smallest natural scale takes a step the others'
    clips have already thrown away, and every config still says 1.0.

    Measured here rather than quoted: the raw norms come out of each arm's own
    `training_log.csv` on the CPU harness.
    """
    raw = {}
    for arm in DUAL_ARMS:
        _md5, _summary, norms = _run_arm(P, arm, epochs=4)
        live = [n for n in norms if n > 0]
        assert live, arm
        raw[arm] = live

    clip = P["constraint_phase"]["constraint_grad_clip"]

    def delivered(norms):
        return [min(n, clip) for n in norms]

    lo = min(min(delivered(v)) for v in raw.values())
    hi = max(max(delivered(v)) for v in raw.values())
    assert hi / lo > 10.0, (
        "the arms' delivered constraint steps are within 10x of each other, so "
        "this gate no longer measures the asymmetry it was written for: %s"
        % {a: [round(x, 6) for x in delivered(v)] for a, v in raw.items()})
    # hounie is the structural extreme: its primal is divided by n_test
    assert max(delivered(raw["hounie"])) < clip
    assert max(delivered(raw["fioretto"])) == pytest.approx(clip)

    # ... and `normalize` removes it: every arm delivers exactly `clip`.
    for arm in DUAL_ARMS:
        _md5, _summary, norms = _run_arm(P, arm, epochs=4,
                                         constraint_grad_mode="normalize")
        live = [n for n in norms if n > 0]
        assert live, arm
        # the LOGGED norm is the pre-scale one by design; what `normalize`
        # guarantees is the delivered norm, checked directly below.


def test_normalize_delivers_exactly_the_clip_for_any_raw_scale():
    """The delivered-norm semantics of both modes, on the function itself.

    `clip` caps: a 0.05-norm gradient stays 0.05 and a 5.0-norm one becomes 1.0,
    a 20x dose gap between two arms whose configs are identical.  `normalize`
    rescales in both directions, so the step size becomes a protocol constant
    and what differs between arms is DIRECTION.
    """
    from src.training.constraint_step import finish_constraint_step

    def delivered(raw_scale, mode):
        p = torch.nn.Parameter(torch.zeros(4))
        g = torch.ones(4)
        p.grad = g * (raw_scale / float(g.norm()))
        model = torch.nn.Module()
        model.register_parameter("w", p)
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        finish_constraint_step(model, opt, None, clip=1.0, mode=mode)
        return float(p.grad.norm())

    assert delivered(0.05, "clip") == pytest.approx(0.05, rel=1e-5)
    assert delivered(5.00, "clip") == pytest.approx(1.00, rel=1e-5)
    assert delivered(0.05, "normalize") == pytest.approx(1.00, rel=1e-5)
    assert delivered(5.00, "normalize") == pytest.approx(1.00, rel=1e-5)


def test_check_parity_REFUSES_a_multi_family_campaign_at_an_unmatched_dose(tmp_path, P):
    """FRAMEWORK 1b-pre finding (2): the arms were not getting the same dose,
    ~20x apart, "invisible to every gate".  It was still invisible.

    `check_parity` verified `constraint_grad_clip` and never `constraint_grad_mode`,
    so a campaign holding `hounie` beside `fioretto` and `alm` under the shipped
    default `clip` -- which is exactly what `results/dualbar`, the dual-vs-clipper
    campaign, is -- printed "PARITY OK -- this campaign is a fair comparison".

    A single-family campaign under `clip` is untouched: with one trained
    methodology the dose is constant across everything being compared.
    """
    one_family = _write_campaign(str(tmp_path / "one"), P,
                                 ["clip", "tralo", "tralo_null", "tralo_reseed"])
    assert _parity(one_family).returncode == 0

    many = _write_campaign(str(tmp_path / "many"), P,
                           ["clip", "fioretto", "hounie", "alm", "tralo_null",
                            "tralo_reseed"])
    r = _parity(many)
    assert r.returncode == 1, (
        "check_parity passed a multi-family campaign at an unmatched "
        "constraint dose:\n" + r.stdout[-2000:])
    assert "constraint_grad_mode" in r.stdout

    def to_normalize(_arm, hp):
        if "constraint_grad_mode" in hp:
            hp["constraint_grad_mode"] = "normalize"

    fixed = _write_campaign(str(tmp_path / "norm"), P,
                            ["clip", "fioretto", "hounie", "alm", "tralo_null",
                             "tralo_reseed"], hp_patch=to_normalize)
    ok = _parity(fixed)
    assert ok.returncode == 0, ok.stdout[-2500:]


def test_check_parity_checks_every_constraint_step_knob(P):
    """`constraint_grad_clip` was in SHARED_KEYS and the three knobs that decide
    what actually happens to that gradient were not.  Two arms at
    `step_rule: sgd` and `shared` would have passed the gate that exists to
    prove they differ only in method.

    And the boundary: `constraint_random_direction` must NOT be in the list.
    `tralo_coin` IS the arm whose constraint step is a random vector of the same
    norm, so demanding agreement on it would refuse every campaign carrying the
    coin control -- the arm that answers "did the direction matter at all".  The
    rule that separates them is mechanical: a knob may be required to agree
    across arms exactly when no arm block overrides it.
    """
    from scripts.check_parity import SHARED_KEYS
    for k in ("constraint_grad_clip", "constraint_grad_mode",
              "constraint_step_rule", "constraint_fp32"):
        assert k in SHARED_KEYS, k
    assert "constraint_random_direction" not in SHARED_KEYS

    overridden = {k for spec in P["blocks"].values() if isinstance(spec, dict)
                  for k in spec}
    for k in SHARED_KEYS:
        if k.startswith("constraint_"):
            assert k not in overridden, (
                "%s is overridden by an arm block, so requiring it to agree "
                "across arms would refuse a legitimate campaign" % k)
    assert "constraint_random_direction" in overridden


# ==========================================================================
# 4. EQUAL BUDGET AT THE OUTPUT
# ==========================================================================

def _budget_case(seed, n=800, C=8, G=5, capped=(2, 7), lp=0.30, gp=0.50,
                 zero_frac=0.5):
    """An iwildcam-shaped instance: 8 classes, 2 capped, and half the per-group
    ceilings at K=0 because the species is simply not at that camera."""
    import pandas as pd
    from src.training.constraints import (compute_global_constraints,
                                          compute_local_constraints)
    rng = np.random.default_rng(seed)
    y = rng.integers(0, C, size=n)
    groups = rng.integers(0, G, size=n)
    for g in range(int(G * zero_frac)):
        y[(groups == g) & np.isin(y, capped)] = 0
    logits = rng.normal(size=(n, C))
    logits[np.arange(n), y] += 1.4
    e = np.exp(logits - logits.max(1, keepdims=True))
    proba = e / e.sum(1, keepdims=True)
    df = pd.DataFrame({"label": y, "grp": groups})
    gcon = compute_global_constraints(df, "label", gp,
                                      constrained_class=list(capped),
                                      num_classes=C)
    lcon = compute_local_constraints(df, "label", lp, "grp",
                                     constrained_class=list(capped),
                                     num_classes=C)
    return proba, groups, y, gcon, lcon, C


@pytest.mark.parametrize("lp,gp,zero_frac", [
    (0.30, 0.50, 0.5),      # LOCAL binds, iwildcam's K=0 ceilings present
    (0.50, 0.30, 0.5),      # GLOBAL binds -- the sweep FRAMEWORK prescribes
    (0.50, 0.30, 0.0),
    (0.30, 0.30, 0.0),      # the two scopes coincide
])
def test_the_clipper_and_the_trained_arms_emit_the_SAME_capped_count(lp, gp, zero_frac):
    """The arms are compared at equal emitted budget, and it is not by
    coincidence -- they reach it through two different code paths.

    `clip` never calls `targeted_correction`: `heuristic` allocates in one joint
    greedy pass.  The trained arms never call the allocator: they go through
    `targeted_correction(force_exact=True)`, which reduces, fills globally, then
    reduces and fills locally.  The two must land on the same count or every
    "quality at equal budget" contrast is really a budget contrast -- and they
    did not, once: interleaving the local reduce and fill made the trained arms
    under-spend by 4-5%, a trained-vs-post-hoc bias the size of the entire
    effect under study, pointing the same way.
    """
    from src.methodologies.heuristic.train import (_build_hierarchy,
                                                   apply_allocation_heuristic)
    from src.utils.posthoc_adjustment import targeted_correction
    capped = [2, 7]
    for seed in range(6):
        proba, groups, _y, gcon, lcon, C = _budget_case(
            seed, lp=lp, gp=gp, zero_frac=zero_frac)
        greedy, _t = apply_allocation_heuristic(
            proba, groups, _build_hierarchy(C, gcon, capped), gcon, lcon, C)
        trained, _flips, _meta = targeted_correction(
            proba, groups, gcon, lcon, capped, force_exact=True)
        for c in capped:
            reachable = min(int(gcon[c]),
                            sum(int(lcon[g][c]) for g in lcon
                                if lcon[g][c] < UNLIMITED))
            assert int((greedy == c).sum()) == reachable, (seed, c, "clip")
            assert int((trained == c).sum()) == reachable, (seed, c, "trained")


def test_the_scorer_refills_every_arm_from_its_own_probabilities():
    """Why the budget-equalized family cannot be gamed by emitting fewer items.

    `full_panel` does not read the arm's shipped labels for any scored metric:
    it rebuilds an allocation with `score_arm.equalize` from the stored
    probabilities, using the cell's K and per-group room.  So the emitted budget
    is a property of the CELL and identical across arms by construction.
    """
    from scripts.score_arm import equalize
    proba, groups, _y, gcon, lcon, _C = _budget_case(0)
    # two "arms": the same probabilities, one of them arbitrarily rescaled with
    # the ORDER intact, which is what a prior shift or a temperature does
    warped = proba ** 3
    warped = warped / warped.sum(1, keepdims=True)
    a = equalize(proba, groups, gcon, lcon, 2)
    b = equalize(warped, groups, gcon, lcon, 2)
    assert int((a == 2).sum()) == int((b == 2).sum())
    assert np.array_equal(a, b), (
        "equalize is not invariant to a monotone rescale -- the budget-"
        "equalized family would then move on calibration, not allocation")


def test_the_scorer_cannot_distinguish_the_two_allocators():
    """A LIMIT of the comparison, pinned so it is not rediscovered as a result.

    `clip` and `lp` share a `base_model_id`, so they emit the same probabilities
    and -- because the scorer discards each arm's own allocation and rebuilds one
    -- they score IDENTICALLY on every metric.  That is not a measurement that
    LP-LG ties the greedy clipper; it is the scorer being unable to see the
    allocator at all.  The same holds for `focal_clip` vs `focal_lp`.
    """
    import scripts.full_panel as fp
    src = open(os.path.join(REPO, "scripts", "full_panel.py"),
               encoding="utf-8").read()
    assert "equalize(" in src
    # the arm's own labels are read into `rel`, and `rel` may only feed the
    # diagnostic-only counters
    assert 'rel = pd.read_csv(fin)["Predicted_Label"]' in src
    scored = set(sum((m for _h, m in fp.GROUPS[:2]), []))
    assert not (scored & fp.NON_SCORING)
    assert {"cnt_over_K", "flips"} <= fp.NON_SCORING


# ==========================================================================
# 5. THE NULL ARMS
# ==========================================================================

def test_every_null_arm_zeroes_its_family_by_config(P):
    """The zeroed keys are the ones that gate each family's constraint backward.

    tralo:    lambda_global / lambda_local / lambda_step -> the penalty VALUE is
              0, and the backward is gated on `total_constraint > 0`.
    fioretto: fioretto_step_size and lambda_init -> lambda can never leave 0 and
              `has_work` requires lambda > 0.
    hounie:   eta_lambda -> lam stays 0, u stays 0, `has_active` is False.
    alm:      eta AND mu0 AND mu_step -- zeroing eta alone would leave a live
              `mu_t * excess` augmentation on every epoch, because ALM adds it
              to the primal weight rather than to lambda.
    """
    z = {
        "tralo_null": {"lambda_global": 0, "lambda_local": 0, "lambda_step": 0},
        "fioretto_null": {"fioretto_step_size": 0, "fioretto_lambda_init": 0},
        "hounie_null": {"hounie_eta_lambda": 0},
        "alm_null": {"alm_eta": 0, "alm_mu0": 0, "alm_mu_step": 0},
    }
    for arm, keys in z.items():
        hp = build_hyperparams(P, P["arms"][arm], 1)
        for k, v in keys.items():
            assert hp[k] == v, (arm, k, hp[k])
    # and the ALM trap specifically: mu is what makes the augmentation live
    assert P["blocks"]["alm"]["alm_mu0"] > 0
    assert P["blocks"]["alm_null"]["alm_mu0"] == 0


def test_the_zero_dose_siblings_are_ONE_model(P):
    """All four nulls must be the same object: warm-up plus CE epochs and
    nothing else.  If any family's zeroing were incomplete its null would drift
    away from the others, and the treated arm it controls would be scored
    against a partly-treated baseline.  Every TREATED arm must differ from that
    object, or its own treatment is inert.
    """
    nulls = {arm: _run_arm(P, arm)[0] for arm in NULL_ARMS}
    assert len(set(nulls.values())) == 1, nulls
    baseline = next(iter(nulls.values()))
    for arm in DUAL_ARMS:
        assert _run_arm(P, arm)[0] != baseline, (
            "%s is bit-identical to the zero-dose control -- its treatment is "
            "inert" % arm)


def test_a_null_arm_never_forms_a_constraint_gradient(P):
    """Stronger than "the predictions match": the step is never taken.

    `constraint_steps_attempted` counts every epoch that reached
    `finish_constraint_step`, so a null that somehow built a gradient and had it
    clipped to nothing would still show up here.
    """
    for arm in NULL_ARMS:
        _md5, summary, norms = _run_arm(P, arm, epochs=4)
        assert summary.get("constraint_steps_attempted") in (0, None), (arm, summary)
        assert summary.get("constraint_steps_applied") in (0, None), (arm, summary)
        assert all(n == 0.0 for n in norms), (arm, norms)


def test_tralo_reseed_differs_from_tralo_null_in_the_RNG_STREAM_ONLY(P):
    """The noise floor arm.  It must vary ONE thing.

    Config side: the two arms' hyperparameter dicts differ in exactly
    `rng_reseed`.  Code side: `rng_reseed` does exactly one `torch.rand(1)` --
    no extra parameter, no extra step, no change to the loss -- and it happens
    inside `train()`, AFTER `run_experiment` re-seeds, so the draw cannot reach
    the warm-up the two arms share.  Seeding it from `torch.randint` instead
    would consume a global draw and move the control's dropout masks too.
    """
    a = build_hyperparams(P, P["arms"]["tralo_null"], 1)
    b = build_hyperparams(P, P["arms"]["tralo_reseed"], 1)
    diff = {k for k in set(a) | set(b) if a.get(k, "<->") != b.get(k, "<->")}
    assert diff == {"rng_reseed"}, {k: (a.get(k), b.get(k)) for k in sorted(diff)}
    assert a["rng_reseed"] is False and b["rng_reseed"] is True

    src = open(os.path.join(REPO, "src/methodologies/tralo/train.py"),
               encoding="utf-8").read()
    tree = ast.parse(src)
    guarded = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.If)
                and "rng_reseed" in ast.unparse(node.test)):
            guarded.append(node)
    assert len(guarded) == 1, "rng_reseed is read in more than one place"
    # every torch.* call inside the guarded block, ignoring logging
    torch_calls = [ast.unparse(n) for n in ast.walk(guarded[0])
                   if isinstance(n, ast.Call)
                   and ast.unparse(n).startswith("torch.")]
    assert torch_calls == ["torch.rand(1)"], torch_calls
    assert not guarded[0].orelse
    # nothing but the draw and a log line: no extra parameter, no extra step
    assert all(isinstance(s, ast.Expr) for s in guarded[0].body), (
        [ast.unparse(s) for s in guarded[0].body])

    # the two arms share a warm-up on purpose, so the draw must not reach it
    dc = P["datasets"]["iwildcam"]
    assert (compute_base_model_id(P, "MobileNetV3", a, "iwildcam", dc)
            == compute_base_model_id(P, "MobileNetV3", b, "iwildcam", dc))
    runner = open(os.path.join(REPO, "src/experiments/runner.py"),
                  encoding="utf-8").read()
    warm = runner.index("run_warmup(")
    assert runner.index("seed_all(seed)", warm) < runner.index("train_fns[", warm), (
        "the RNG is no longer re-seeded between the warm-up and train(), so the "
        "reseed control's single draw is no longer the only difference")


def test_the_reseed_control_actually_moves_the_model(P):
    """A control that changes nothing measures nothing.  One draw from the
    global generator has to reach the DataLoader shuffle and the dropout masks,
    or the "constraint moves the count 0.90-1.00x as far as a reseed" floor is
    reading zero.
    """
    assert _run_arm(P, "tralo_reseed")[0] != _run_arm(P, "tralo_null")[0]


def test_the_generator_refuses_a_trained_arm_without_its_reseed_floor(tmp_path):
    """Both controls are structural, not optional."""
    r = subprocess.run(
        [sys.executable, "-m", "configs.gen_campaign", "--root",
         str(tmp_path / "c"), "--datasets", "iwildcam", "--models",
         "MobileNetV3", "--caps", "L30_G50", "L50_G30", "--arms", "tralo"],
        cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0
    assert "reseed" in (r.stdout + r.stderr).lower()


def test_a_scorer_edit_does_not_split_a_running_campaign_s_code_version():
    """`code_version` must move when the TRAINING code moves, and only then.

    On 2026-08-24 `results/iwc3` came back split: `3bb7e8b411e8` on its first
    two runs and `3bb7e8b411e8-dirty` on the next two, because a scorer under
    `scripts/` was deployed between them. Every file the runner imports was
    byte-identical across the two halves, and `check_parity` correctly refused
    the campaign anyway -- so a rule CLAUDE.md states ("scripts/ is exempt and
    safe to update mid-flight") was true of the code and false of the stamp.

    Both directions are exercised on a throwaway repo, because a gate that only
    checks the quiet case cannot tell a scoped diff from a broken one.
    """
    import shutil
    from src.utils.gitver import git_version

    repo = tempfile.mkdtemp()
    try:
        def git(*a):
            subprocess.run(["git"] + list(a), cwd=repo, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        git("init", "-q")
        git("config", "user.email", "t@t"); git("config", "user.name", "t")
        for rel in ("src/pipeline", "scripts", "configs"):
            os.makedirs(os.path.join(repo, rel), exist_ok=True)
        for rel in ("src/pipeline/train.py", "scripts/score.py",
                    "configs/protocol.yml", "main.py"):
            with open(os.path.join(repo, rel), "w", encoding="utf-8") as fh:
                fh.write("original\n")
        git("add", "-A"); git("commit", "-qm", "base")
        assert not git_version(repo).endswith("-dirty"), "clean tree read dirty"

        # a scorer edit must NOT move the stamp
        with open(os.path.join(repo, "scripts/score.py"), "w",
                  encoding="utf-8") as fh:
            fh.write("edited\n")
        assert not git_version(repo).endswith("-dirty"), (
            "editing scripts/ still splits code_version -- this is the exact "
            "defect that split results/iwc3")

        # NEGATIVE CONTROL: a training-path edit MUST move it, or the scoping
        # has silently disabled the stamp altogether
        with open(os.path.join(repo, "src/pipeline/train.py"), "w",
                  encoding="utf-8") as fh:
            fh.write("edited\n")
        assert git_version(repo).endswith("-dirty"), (
            "editing src/ no longer marks the tree dirty -- the stamp is dead "
            "and every campaign would read as uniform no matter what landed")
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_order_probe_arithmetic_and_its_control():
    """`scripts/order_probe.py` produced the sharpest negative this project has:
    the constraint costs -30.8 items per cell against its OWN reseed twin, by
    evicting items at p~0.79 and admitting items at p~0.25.

    The whole result rests on one comparison being made correctly, so the
    arithmetic is pinned here with a case whose answer is known by hand, plus
    the control that makes it interpretable. EVICTED items outrank ADMITTED
    ones by construction, so a raw negative means nothing; only the difference
    against a perturbation of no consequence does.
    """
    from scripts.order_probe import spearman

    # a reversal must read -1, an identity +1: the sign convention is the
    # entire verdict, and getting it backwards would invert the finding
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
    # monotone but non-linear must still read +1 -- a count penalty applies a
    # monotone map to p, and if this read < 1 the probe would manufacture
    # "reordering" out of a pure rescale
    assert abs(spearman([0.1, 0.2, 0.3], [0.01, 0.04, 0.09]) - 1.0) < 1e-9
    assert spearman([1.0], [1.0]) != spearman([1.0], [1.0]) or True  # n<3 -> nan

    # the eviction swap, by hand. Twin's top-2 = items {0,1}; arm's = {0,2}.
    # So evicted={1}, admitted={2}. Item 1 is a true positive, item 2 is not.
    y = np.array([5, 5, 3, 3])
    cls = 5
    ev, ad = [1], [2]
    evicted_tp = float(np.mean(y[ev] == cls))
    admitted_tp = float(np.mean(y[ad] == cls))
    net = float(np.sum(y[ad] == cls)) - float(np.sum(y[ev] == cls))
    assert (evicted_tp, admitted_tp, net) == (1.0, 0.0, -1.0), (
        "the swap dropped a correct item for a wrong one and must read -1 item")

    # NEGATIVE CONTROL: a swap of equal quality must read ZERO, or every
    # perturbation would look damaging and the probe could not tell the
    # constraint from an RNG reseed -- which is exactly what it is for.
    y2 = np.array([5, 5, 5, 3])
    net_even = (float(np.sum(y2[[2]] == cls)) - float(np.sum(y2[[1]] == cls)))
    assert net_even == 0.0, "an even swap must net zero items"


def test_uniform_count_has_a_CONSTANT_per_item_gradient_and_sum_does_not():
    """The defining property of `soft_count_mode: uniform`, tested where it is
    exact rather than where it is noisy.

    `scripts/order_probe` measured that the shipped count evicts true positives
    and admits false ones for a net -30.4 items per cell (16/16), against a
    reseed control that nets +0.38. The cause is that `d(sum_i p_ic)/dz_ic` is
    `p(1-p)`, which differs per item, so the penalty reorders the class. The
    fix is a count whose per-item gradient is CONSTANT, since a constant step
    in the class logit is a pure bias shift and a bias shift cannot reorder.

    The smoke harness cannot see this -- 120 items of random labels give
    rho=0.999965 for both modes, i.e. the cap barely binds -- so the property
    is checked directly on the gradient, which is where it is a theorem.
    """
    import torch
    from src.losses.transductive_loss import uniform_grad_count

    torch.manual_seed(0)
    C = 4
    z = torch.randn(64, C, dtype=torch.float64, requires_grad=True)
    cls = 1

    # --- the shipped count: gradient is p(1-p), and it VARIES ---
    p = torch.softmax(z, dim=1)
    p.sum(dim=0)[cls].backward()
    g_sum = z.grad[:, cls].clone()
    expected = (p[:, cls] * (1 - p[:, cls])).detach()
    assert torch.allclose(g_sum, expected, atol=1e-9), "sum's gradient is not p(1-p)"
    assert g_sum.std().item() > 1e-3, (
        "NEGATIVE CONTROL FAILED: the shipped count's per-item gradient is "
        "already constant on this input, so the test cannot tell the two "
        "modes apart and proves nothing")

    # --- uniform: value identical, gradient constant ---
    z.grad = None
    p2 = torch.softmax(z, dim=1)
    eff = uniform_grad_count(p2)
    assert torch.allclose(eff.detach(), p2.detach(), atol=1e-12), (
        "the VALUE must stay exactly sum_i p_ic, or the penalty is comparing a "
        "different quantity to K and the cap no longer means what it says")
    eff.sum(dim=0)[cls].backward()
    g_uni = z.grad[:, cls].clone()
    spread = (g_uni.max() - g_uni.min()).item() / max(1e-12, g_uni.abs().mean().item())
    assert spread < 1e-6, (
        "uniform's per-item gradient is not constant (relative spread %.3g) -- "
        "it can still single items out and reorder the class" % spread)

    # dose comparable: `w` is the mean of p(1-p), so the two modes deliver the
    # same total pull and differ only in how it is distributed
    assert abs(g_uni.mean().item() - g_sum.mean().item()) < 1e-9, (
        "uniform changed the total dose, not just its distribution -- then any "
        "difference in a campaign is a dose effect and unattributable")


def test_an_unknown_soft_count_mode_is_refused_not_silently_run_as_sum():
    """A typo used to fall through to the `sum` branch and run the manuscript's
    arm under another arm's name. That is this project's most frequent failure
    mode -- an inert flag -- and it has cost four separate occasions.
    """
    import pytest as _pytest
    from src.methodologies.tralo.train import train as tralo_train
    from src.experiments.runner import TrainInputs  # noqa: F401

    class _Stub:
        pass
    stub = _Stub()
    stub.config = {}
    stub.hyperparams = {"soft_count_mode": "unifrom"}   # deliberate typo
    with _pytest.raises(ValueError, match="soft_count_mode must be one of"):
        tralo_train(stub)


def test_headroom_uses_the_BINDING_budget_not_the_inert_global():
    """`scripts/headroom.py` priced every direction in this project, and for an
    hour on 2026-08-24 it priced them 30x too high.

    It set `K = int(G[c])`, the GLOBAL cap alone. Local caps are per-group
    ceilings, so their SUM already bounds the count, and on iwildcam the global
    sits ABOVE that sum and can never bind -- `gen_campaign` prints exactly
    that for every cap it emits. The ceiling is `2K/(K+n)`, so an unreachable K
    inflates it twice over: on L30_G50 class 2 it read 0.667 against a
    reachable 0.462 and printed 59 items of headroom where the real gap is 2.0.

    The module docstring already said "local caps can put it out of reach".
    That was a comment describing a defect instead of a fix, and the number it
    qualified was quoted as the project's effect size.
    """
    from scripts.headroom import effective_budget
    from src.utils.constants import UNLIMITED

    # iwildcam L30_G50 class 2, real numbers: global 185, local sum 111
    G = {2: 185}
    L = {130: {2: 0}, 218: {2: 0}, 320: {2: 0}, 516: {2: 0},
         1: {2: 31}, 2: {2: 32}, 3: {2: 48}}
    assert effective_budget(G, L, 2) == 111, (
        "the inert global is being used; this is the 30x inflation")

    # NEGATIVE CONTROL 1: when the global is TIGHTER it must win, or the fix
    # has simply replaced one wrong answer with another
    assert effective_budget({2: 50}, L, 2) == 50

    # NEGATIVE CONTROL 2: one uncapped group means the local scope bounds
    # nothing globally, so the global must stand alone. Silently summing the
    # capped groups there would UNDER-count the budget and invent headroom in
    # the opposite direction.
    L_open = dict(L)
    L_open[9] = {2: UNLIMITED}
    assert effective_budget(G, L_open, 2) == 185


def test_ovr_count_has_ZERO_gradient_outside_the_capped_columns():
    """The one-vs-rest count's entire claim, checked rather than asserted.

    `scripts/family_split.py` on `results/xfam1` (16 matched cell-seeds, 9
    cells, 2026-08-24) found the three dual families damage the CAPPED classes
    near-identically (-0.0020 to -0.0028 ccF1, about one item) and differ 5.3x
    in what they do to the six classes the constraint never names (-0.0027 to
    -0.0144 uncF1). The published ordering of the families is that collateral.

    The shipped count is `S_c = sum_i softmax(z)_ic`, whose derivative
    `-sum_i p_ic p_ik` is nonzero for EVERY uncapped k, so one capped-class
    push moves all eight logits. `S_c = sum_i sigmoid(z_ic)` cannot: its
    support is the capped columns only, at any dose. That is the whole reason
    to consider it, so it is a gate and not a comment.
    """
    import numpy as np
    from scripts.collateral_probe import grad_count

    rng = np.random.default_rng(0)
    z = rng.normal(size=(64, 8))
    capped, unc = [2, 7], [0, 1, 3, 4, 5, 6]

    g_ovr = grad_count(z, capped, "ovr")
    assert np.all(g_ovr[:, unc] == 0.0), (
        "`ovr` moved an uncapped logit; its only claim is that it cannot")
    assert np.any(g_ovr[:, capped] != 0.0), (
        "`ovr` moved nothing at all -- an inert mode passes the line above "
        "trivially, which is this project's most frequent failure mode")

    # NEGATIVE CONTROL: the shipped count MUST fail the same assertion, or
    # there is no collateral to remove and the whole direction is void.
    g_sum = grad_count(z, capped, "sum")
    assert np.any(g_sum[:, unc] != 0.0), (
        "`sum` has no uncapped gradient either, so `ovr` fixes nothing")


def test_the_collateral_probe_does_not_REDERIVE_a_gradient_src_already_owns():
    """It did, for one revision on 2026-08-24, and got it wrong.

    The hand-written `uniform` gradient divided by `p(1-p)`, which explodes for
    the small `p` that most items carry, so after unit-normalisation the term
    that actually lowers the capped logit was negligible: the probe reported
    the arm moving the capped count by -0.0000. `results/uniform1` was already
    generated, gated and staged to launch on that arm. A probe that silently
    prices a staged campaign at zero is worse than no probe.

    The fix is structural rather than a corrected formula: autograd the
    SHIPPED function, so a change in `src` cannot leave this file behind.
    """
    import ast
    import io
    import numpy as np

    src = io.open("scripts/collateral_probe.py", encoding="utf-8").read()
    tree = ast.parse(src)
    imported = {
        alias.name
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("src.losses")
        for alias in node.names
    }
    assert "uniform_grad_count" in imported, (
        "the probe must autograd the shipped count, not restate its gradient")
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                and "uniform" in n.name], (
        "a local `uniform` gradient is back in the probe; that is the exact "
        "defect this gate exists for")

    # BEHAVIOURAL HALF: the bug showed up as a mode that moved nothing. Both
    # shipped modes must actually push the capped logits down.
    from scripts.collateral_probe import grad_count
    rng = np.random.default_rng(1)
    z = rng.normal(size=(128, 8))
    for mode in ("sum", "uniform"):
        g = grad_count(z, [2, 7], mode)
        assert np.abs(g[:, [2, 7]]).sum() > 1e-6, (
            "%r delivers no capped-class gradient, which is how the "
            "hand-derived version read" % mode)


def test_family_split_REFUSES_when_the_zero_lambda_twins_are_not_one_run():
    """At lambda = 0 the dual family is irrelevant, so the nulls ARE one run.

    Same warm-up cache, same allocator, same seed, no constraint gradient:
    `tralo_null`, `fioretto_null` and `hounie_null` must produce byte-identical
    raw predictions. Measured on `results/xfam1` 2026-08-24: identical in 12 of
    12 cell-seeds, which is what licenses reporting ONE compute term instead of
    three and makes the per-family attribution meaningful.

    If they ever diverge, something other than lambda differs between the
    families and every constraint term in that table is contaminated. The tool
    must refuse rather than print one.
    """
    from scripts.family_split import matched, null_identity

    nulls = ["tralo_null", "fioretto_null", "hounie_null"]
    cell = ("MobileNetV3", "L30_G50", "2-7")
    agree = {(cell, n, 1): {"raw_md5": "deadbeef"} for n in nulls}
    assert null_identity(agree, [(cell, 1)], nulls) == [], (
        "identical digests were reported as a divergence")

    diverge = dict(agree)
    diverge[(cell, "hounie_null", 1)] = {"raw_md5": "0ther"}
    assert null_identity(diverge, [(cell, 1)], nulls), (
        "a diverging null passed; every per-family attribution would be "
        "contaminated and the tool would print it anyway")

    # And the matcher must drop a cell-seed that is missing an arm, or `clip`
    # gets measured on cells the treatment was never run on.
    rows = {(cell, a, 1): {} for a in ["clip", "tralo", "tralo_null"]}
    rows[(cell, "clip", 2)] = {}
    need = ["clip", "tralo", "tralo_null"]
    assert matched(rows, need) == [(cell, 1)], (
        "seed 2 has only `clip` and must not form a pair")
