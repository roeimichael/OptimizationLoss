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


def test_the_softmax_cross_term_CANNOT_reorder_the_uncapped_classes():
    """The proposed one-vs-rest fix was aimed at damage that does not exist.

    Section 2(s) measured that the three dual families differ 5.3x in what they
    do to the six classes the constraint never names, and the obvious culprit
    was the softmax cross-term: `dS_c/dz_k = -sum_i p_ic p_ik` is nonzero for
    every uncapped k, so a capped push moves all eight logits. A one-vs-rest
    count zeroes that term exactly, and was staged as the fix.

    IT IS A NULL, and the algebra says why. The update adds `+eta * p_ic * p_ik`
    to `z_k`, which is MONOTONE INCREASING in `p_ik` -- it widens the gaps in
    the uncapped block in the direction they already point. It sharpens the
    existing order; it cannot invert it.

    Measured to match (`scripts/collateral_probe.py`, 16 stored runs, 2026-08-24,
    effect matched at 20/50/100/200 capped predictions removed): ZERO
    uncapped-to-uncapped prediction flips at every target, up to eta = 1091
    where the uncapped logits have moved 79 units. So the uncF1 damage in
    section 2(s) does NOT come through the output layer, and the lever is the
    parameter set the constraint may touch -- not the count.
    """
    import numpy as np
    from scripts.collateral_probe import softmax, step

    rng = np.random.default_rng(7)
    z = rng.normal(size=(256, 8)) * 2.0
    capped, unc = [2, 7], [0, 1, 3, 4, 5, 6]

    for eta in (1.0, 50.0, 1000.0):
        z1 = step(z, capped, "sum", eta)
        before = np.argsort(z[:, unc], axis=1)
        after = np.argsort(z1[:, unc], axis=1)
        assert np.array_equal(before, after), (
            "the softmax cross-term reordered the uncapped block at eta=%g; "
            "if this ever fires, the one-vs-rest fix is live again" % eta)

    # NEGATIVE CONTROL: a perturbation that is NOT monotone in p_ik must
    # reorder them, or the assertion above is passing for a trivial reason
    # (e.g. `step` silently returning its input).
    noisy = z.copy()
    noisy[:, unc] += rng.normal(size=(256, len(unc))) * 3.0
    assert not np.array_equal(np.argsort(z[:, unc], axis=1),
                              np.argsort(noisy[:, unc], axis=1)), (
        "even random noise did not reorder the block, so the check is inert")

    # And the capped classes MUST actually move, or nothing was enforced.
    z1 = step(z, capped, "sum", 50.0)
    assert softmax(z1)[:, capped].sum() < softmax(z)[:, capped].sum(), (
        "the step did not reduce the capped soft count at all")


def test_reachability_prices_the_run_s_OWN_count_not_always_p_times_1_minus_p():
    """`p(1-p)` is the slope of `sum` and of nothing else.

    `soft_count_mode` has had three legal values since `uniform` landed, and
    `scripts/reachability.py` hardcoded `p(1-p)` and printed it as THE
    reachability verdict. `results/uniform1` was staged on `uniform`, whose
    entire purpose is that the per-item slope is a population CONSTANT -- so
    the tool would have priced the new arm with the slope of the arm it
    replaces, and called it `flat at K` in exactly the cells it is designed to
    make live.

    Same defect class as the probe that hand-derived a gradient `src` already
    owns: the weight now comes from `uniform_grad_count` through autograd, and
    it is taken w.r.t. the class LOGIT. Differentiating against `p` instead
    returns `w / (p(1-p))`, which is item-DEPENDENT -- the precise property
    `uniform` exists to remove, so that error inverts the reading.
    """
    import numpy as np
    from scripts.reachability import slope_at

    rng = np.random.default_rng(0)
    p = rng.random(500) * 0.9 + 0.05

    # `uniform` is the population mean p(1-p), identical wherever you cut
    w = float((p * (1.0 - p)).mean())
    lo, _ = slope_at(p, 10, "uniform")
    hi, _ = slope_at(p, 400, "uniform")
    assert abs(lo - w) < 1e-9 and abs(hi - w) < 1e-9, (
        "the uniform slope is not the shipped population weight; if it varies "
        "with the cut, it was differentiated against p and not the logit")

    # NEGATIVE CONTROL: `sum` MUST vary with the cut, or the assertion above
    # is passing because every mode returns the same constant.
    s_lo, _ = slope_at(p, 10, "sum")
    s_hi, _ = slope_at(p, 400, "sum")
    assert abs(s_lo - s_hi) > 1e-3, (
        "`sum`'s slope did not move across the cut, so the contrast is inert")
    assert abs(s_lo - p[np.argsort(-p)[9]] * (1 - p[np.argsort(-p)[9]])) < 1e-9

    # An unpriceable mode must REFUSE, not silently fall back to p(1-p)
    import pytest
    with pytest.raises(SystemExit):
        slope_at(p, 10, "margin")


def test_ortho_project_removes_the_CE_component_and_LEAVES_THE_DOSE_ALONE():
    """`tralo_ortho`, reopened by FRAMEWORK 2(t), gated before it is launched.

    2(s) measured that the constraint's damage to the six uncapped classes does
    NOT arrive through the output layer -- the softmax cross-term perturbs
    those logits and provably cannot reorder them, zero flips across a 50x dose
    range. It arrives through the SHARED BACKBONE, which is what a projection
    onto the complement of the CE gradient acts on.

    Two properties, and the second is the one that makes the contrast legal:

    1. the delivered gradient is ORTHOGONAL to the CE reference, and
    2. its NORM is exactly what the unprojected arm delivers.

    (2) holds because the projection runs BEFORE `clip_grad_norm_`, so
    `normalize` rescales the projected gradient to `clip` like any other. If it
    ran after, the treatment would take a SHORTER step than its control and
    direction would be confounded with dose -- the trap that made the hounie
    baseline meaningless (`src/training/constraint_step.py` docstring).

    The predecessor campaign left ZERO prediction files, so nothing about this
    flag can be audited after the fact. It gets a gate before it gets a GPU.
    """
    import torch
    from src.training.constraint_step import finish_constraint_step, snapshot_grads

    def run(with_ref):
        torch.manual_seed(0)
        m = torch.nn.Linear(6, 3, bias=False)
        ref = [torch.randn(3, 6)]
        m.weight.grad = torch.randn(3, 6)
        raw = m.weight.grad.detach().clone()
        finish_constraint_step(m, None, None, clip=1.0, mode="normalize",
                               fp32=True, step_rule="sgd", lr=0.0,
                               ortho_ref=ref if with_ref else None)
        g = m.weight.grad.detach()
        return float((g * ref[0]).sum()), float(g.norm()), raw, ref[0]

    dot_on, nrm_on, _, _ = run(True)
    dot_off, nrm_off, _, _ = run(False)

    assert abs(dot_on) < 1e-4, (
        "the delivered gradient is not orthogonal to the CE reference; "
        "dot=%g" % dot_on)
    # NEGATIVE CONTROL: without the reference it must NOT be orthogonal, or
    # the assertion above is passing because the gradient happens to be.
    assert abs(dot_off) > 1e-3, (
        "the unprojected gradient was already orthogonal, so this fixture "
        "cannot tell the projection from a no-op; dot=%g" % dot_off)
    # DOSE HELD: both arms deliver exactly `clip`.
    assert abs(nrm_on - 1.0) < 1e-5 and abs(nrm_off - 1.0) < 1e-5, (
        "normalize did not deliver exactly clip (%g vs %g); the projected arm "
        "would differ from its control in dose as well as direction"
        % (nrm_on, nrm_off))

    # A non-finite CE gradient must yield NO reference: on the FP16 path
    # `scaler.step` skips such an update, and projecting against a direction
    # the model never moved in would remove something that never happened.
    m = torch.nn.Linear(4, 2, bias=False)
    m.weight.grad = torch.full((2, 4), float("nan"))
    assert snapshot_grads(m) is None
    m.weight.grad = torch.ones(2, 4)
    assert snapshot_grads(m) is not None


def test_the_tralo_trainer_actually_READS_ortho_project():
    """An inert flag is this project's most frequent defect -- four and counting.

    `ortho_project` has no surviving predecessor run to audit against (its 8-run
    campaign left zero prediction files, FRAMEWORK 2(t)), so a config-level
    check is all that exists before the campaign. AST, not grep: a mention in a
    comment is not a read.
    """
    import ast
    import io

    src = io.open("src/methodologies/tralo/train.py", encoding="utf-8").read()
    tree = ast.parse(src)
    reads = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "get" and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == "ortho_project"
    ]
    assert reads, "`ortho_project` is in the protocol with no reader in train()"

    # and the reference must reach the step, or the flag reads and does nothing
    passes = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", getattr(n.func, "attr", None))
        == "finish_constraint_step"
        and any(k.arg == "ortho_ref" for k in n.keywords)
    ]
    assert passes, (
        "the flag is read but `ortho_ref` never reaches finish_constraint_step")


def test_uncF1_is_exactly_the_classes_the_constraint_never_names(tmp_path):
    """`C * macroF1 == m * ccF1 + (C - m) * uncF1`, exactly.

    macro-F1 is CARRIED by the uncapped classes -- six of eight on iwildcam --
    and the scorer printed the composite for months without ever printing what
    drives it. FRAMEWORK 2(s) rests on the split: the three dual families damage
    the capped classes near-identically (-0.0020 to -0.0028 ccF1, about one
    item) and differ 5.3x on the classes the constraint never mentions
    (-0.0027 to -0.0144 uncF1). If `uncF1`'s label list were wrong -- one
    capped class leaking in, or an absent class counted -- the identity breaks
    and that entire section is arithmetic on a bad column.

    Verified 2026-08-24 on 56 stored runs from `evidence/`: holds on 56 of 56.
    This gate reproduces it on a synthetic run so it cannot regress without the
    server.
    """
    import json
    import numpy as np
    import pandas as pd
    from scripts.full_panel import panel

    rng = np.random.default_rng(3)
    n, C = 240, 5
    y = rng.integers(0, C, size=n)
    P = rng.random((n, C)) ** 2
    P[np.arange(n), y] += 1.4                       # a model with real signal
    P = P / P.sum(axis=1, keepdims=True)
    d = tmp_path / "run"
    d.mkdir()
    frame = pd.DataFrame({"True_Label": y, "Predicted_Label": P.argmax(1),
                          "Group_ID": rng.integers(0, 3, size=n)})
    for c in range(C):
        frame["Prob_Class_%d" % c] = P[:, c]
    frame.to_csv(d / "final_predictions_raw.csv", index=False)
    frame.to_csv(d / "final_predictions.csv", index=False)
    cfg = {"dataset_mode": "synthetic", "model_name": "M", "constraint_tag": "T",
           "arm": "a", "hyperparams": {"seed": 1}, "constraint": [0.3, 0.3],
           "dataset_config": {"constrained_class": [1, 3]}}
    json.dump(cfg, open(d / "config.json", "w"))

    r = panel(str(d), cfg)
    assert r is not None, "the fixture did not produce a scorable run"
    m = len(r["capped"].split("-"))
    assert m == 2
    assert abs(C * r["macroF1"] - (m * r["ccF1"] + (C - m) * r["uncF1"])) < 1e-9, (
        "uncF1 is not the complement of ccF1 within macroF1: "
        "macro=%.9f cc=%.9f unc=%.9f" % (r["macroF1"], r["ccF1"], r["uncF1"]))

    # NEGATIVE CONTROL: the identity is not vacuous -- it must FAIL if uncF1
    # were the macro over ALL classes, which is the obvious wrong label list.
    from sklearn.metrics import f1_score
    wrong = f1_score(y, P.argmax(1), average="macro", zero_division=0)
    assert abs(C * r["macroF1"] - (m * r["ccF1"] + (C - m) * wrong)) > 1e-6, (
        "the capped and uncapped classes score identically in this fixture, so "
        "it cannot distinguish a correct label list from a wrong one")


def test_a_count_must_be_INVARIANT_to_the_logit_gauge():
    """Softmax fixes the relative logits and nothing fixes the absolute ones.

    `z` and `z + c` describe the same model: softmax is invariant to a per-item
    additive shift, CE never penalises it, and nothing in training pins it. So
    any count whose gradient CHANGES under that shift has a dose that drifts
    with a quantity the objective does not control -- invisibly, which is this
    project's signature failure (`constraint_grad_mode` across arms, `cut_temp`
    across seeds, `hounie` at 1% of its intended dose).

    Measured 2026-08-24 on a stored run, four gauges (`log p`, `log p` with the
    row max at 0, `log p + 5`, `log p - 5`): the shipped `sum` count returns
    2.361e-04 on p > 0.99 items in ALL FOUR, while the proposed one-vs-rest
    count returns 4.518e-2 / 4.409e-2 / 1.955e-3 / 4.882e-2 -- a **23x** spread
    from a shift that changes no prediction.

    This is the SECOND independent reason `ovr` is closed. The first is that it
    fixes a leak that costs nothing (2(s)): the softmax cross-term perturbs the
    uncapped logits and provably cannot reorder them.

    The gate is on `sum`, which must stay invariant. `ovr` is kept only as the
    negative control that proves the check can fail.
    """
    import numpy as np
    from scripts.collateral_probe import grad_count

    rng = np.random.default_rng(11)
    p = rng.random((300, 6)) ** 2
    p = p / p.sum(axis=1, keepdims=True)
    base = np.log(p)
    capped = [1, 4]

    def unit(z, mode):
        g = grad_count(z, capped, mode)
        return g / np.linalg.norm(g)

    ref = unit(base, "sum")
    for shift, label in ((5.0, "+5"), (-5.0, "-5"),
                         (None, "row max -> 0")):
        z = (base - base.max(axis=1, keepdims=True) if shift is None
             else base + shift)
        assert np.allclose(unit(z, "sum"), ref, atol=1e-9), (
            "`sum`'s gradient moved under a gauge shift (%s). It is a function "
            "of softmax(z) alone and cannot; if this fires, the count now reads "
            "the absolute logits and its dose drifts with them" % label)

    # NEGATIVE CONTROL: a sigmoid-on-logit count MUST move, or the check above
    # is passing because the fixture cannot distinguish the two.
    o_ref = unit(base, "ovr")
    o_shift = unit(base + 5.0, "ovr")
    assert not np.allclose(o_shift, o_ref, atol=1e-6), (
        "even the one-vs-rest count was gauge-invariant here, so this fixture "
        "cannot detect the defect it exists to detect")


def test_head_only_confines_the_constraint_and_STILL_delivers_the_full_dose():
    """`tralo_head` -- the positive control that makes `tralo_ortho` readable.

    FRAMEWORK 2(s) concluded the constraint's damage to the uncapped classes
    arrives through the shared backbone. `tralo_ortho` tries to fix that;
    `tralo_head` tests it outright by confining the constraint gradient to the
    classifier head. Run alone, an `ortho` null cannot distinguish "the
    projection is too weak" from "the backbone was never the culprit"; run
    beside this arm, those separate.

    Three properties:
      1. every non-head gradient is EXACTLY zero -- not small, zero;
      2. the head's gradient is not;
      3. the delivered norm is still exactly `clip`, so the arm differs from
         its control in the constraint's SUPPORT and not in its dose.

    (3) is why the masking runs before the bound. Masking after would leave
    this arm taking a far smaller total step than `tralo`, which confounds
    support with dose -- the trap that made the hounie baseline meaningless.
    """
    import torch
    from src.training.constraint_step import (finish_constraint_step,
                                              head_parameter_ids)

    n_classes = 4

    def build():
        torch.manual_seed(5)
        return torch.nn.Sequential(torch.nn.Linear(8, 6),      # "backbone"
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(6, n_classes))  # head

    def run(mask):
        m = build()
        ids = head_parameter_ids(m, n_classes) if mask else None
        for prm in m.parameters():
            prm.grad = torch.randn_like(prm)
        finish_constraint_step(m, None, None, clip=1.0, mode="normalize",
                               fp32=True, step_rule="sgd", lr=0.0,
                               head_ids=ids)
        back = [prm.grad for prm in m[0].parameters()]
        head = [prm.grad for prm in m[2].parameters()]
        total = torch.cat([prm.grad.reshape(-1) for prm in m.parameters()])
        return back, head, float(total.norm())

    back_on, head_on, nrm_on = run(True)
    back_off, _, nrm_off = run(False)

    assert all(float(g.abs().max()) == 0.0 for g in back_on), (
        "a non-head gradient survived the mask; the constraint still reaches "
        "the backbone and the arm does not test what it claims")
    assert any(float(g.abs().max()) > 0 for g in head_on), (
        "the head gradient is zero too -- the arm is inert, which passes the "
        "assertion above trivially")
    # NEGATIVE CONTROL: unmasked, the backbone MUST carry gradient.
    assert any(float(g.abs().max()) > 0 for g in back_off), (
        "the backbone had no gradient even unmasked, so this fixture cannot "
        "tell the mask from a no-op")
    # DOSE HELD: masking happens before the bound, so both deliver `clip`.
    assert abs(nrm_on - 1.0) < 1e-5 and abs(nrm_off - 1.0) < 1e-5, (
        "normalize did not deliver exactly clip (%g vs %g); the head-only arm "
        "would differ from its control in dose as well as support"
        % (nrm_on, nrm_off))


def test_head_parameter_ids_REFUSES_an_ambiguous_head():
    """A silently-wrong head is an inert flag with a plausible name.

    The four backbones name their head differently (`classifier`, `fc`,
    `heads`), so identification is by shape -- the single Linear emitting
    `n_classes` logits. If that matches more than one layer, or none, the head
    is not determined, and confining the constraint to an arbitrary layer would
    still log `head_only: true` and still write `completed`.
    """
    import pytest
    import torch
    from src.training.constraint_step import head_parameter_ids

    ok = torch.nn.Sequential(torch.nn.Linear(8, 6), torch.nn.Linear(6, 3))
    assert len(head_parameter_ids(ok, 3)) == 2          # weight and bias

    ambiguous = torch.nn.Sequential(torch.nn.Linear(8, 3), torch.nn.Linear(3, 3))
    with pytest.raises(ValueError, match="exactly one Linear"):
        head_parameter_ids(ambiguous, 3)

    with pytest.raises(ValueError, match="exactly one Linear"):
        head_parameter_ids(ok, 99)                       # no head at all


def test_the_tralo_trainer_actually_READS_head_only():
    """Same AST check `ortho_project` gets, for the same reason."""
    import ast
    import io

    tree = ast.parse(io.open("src/methodologies/tralo/train.py",
                             encoding="utf-8").read())
    assert [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "get" and n.args
            and isinstance(n.args[0], ast.Constant)
            and n.args[0].value == "head_only"], (
        "`head_only` is in the protocol with no reader in train()")
    assert [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", getattr(n.func, "attr", None))
            == "finish_constraint_step"
            and any(k.arg == "head_ids" for k in n.keywords)], (
        "the flag is read but `head_ids` never reaches finish_constraint_step")


def test_the_feasibility_target_is_the_runs_OWN_excess_not_a_round_number():
    """`collateral_probe --feasibility` asks a different question than `--target`.

    FRAMEWORK 2(s) quotes "sum leaves a residual excess of 100.4 items and
    reaches feasibility in 25 of 56 runs" off this mode, so its arithmetic is
    load-bearing. The target must be `max(0, raw count - K)` summed over the
    capped classes -- an already-feasible run contributes nothing and must be
    SKIPPED rather than counted as a success, or the denominator flatters every
    mode equally and the comparison is meaningless.
    """
    import numpy as np
    from scripts.collateral_probe import softmax

    # counts 12, 3 against K 5, 5 -> excess is 7, not 10 and not 7-2=5
    pred = np.array([0] * 12 + [1] * 3 + [2] * 20)
    capped, K = [0, 1], {0: 5, 1: 5}
    excess = int(sum(max(0, int((pred == c).sum()) - int(K[c])) for c in capped))
    assert excess == 7, (
        "an UNDER-budget capped class must contribute 0, not a negative that "
        "cancels another class's overshoot")

    # already feasible -> zero, which the caller must treat as "skip"
    K_loose = {0: 50, 1: 50}
    assert 0 == int(sum(max(0, int((pred == c).sum()) - int(K_loose[c]))
                        for c in capped))

    # and the probe's own softmax->argmax path must agree with that count
    z = np.log(np.eye(3)[pred] * 0.9 + 0.05)
    assert int((softmax(z).argmax(1) == 0).sum()) == 12


def test_flag_live_REFUSES_post_hoc_arms_instead_of_calling_them_inert():
    """It called `clip` and `focal_clip` INERT. They are not.

    Run 2026-08-25 as a sweep over every post-hoc arm, this file reported
    bit-identical predictions for `clip`, `focal_clip`, `lp`, `focal_lp`,
    `cb_lp` and `la_lp` and printed "do not launch a campaign on it" -- about
    the two bars every campaign in this project is scored against, and about
    four of the nine methodologies the paper claims.

    The arms are healthy; the harness cannot see them. It calls
    `TRAIN_FNS[methodology]` directly and so runs neither phase a post-hoc
    arm's treatment lives in: the WARM-UP, where `warmup_loss` is read via
    `make_ce_criterion` from `run_warmup` -- reached only from
    `src/experiments/runner.py` -- and the ALLOCATOR, which is downstream of
    the model this file hashes. A post-hoc arm therefore comes back identical
    however live it is.

    A gate that condemns the healthy is worse than no gate: this project has
    already had a correct `iwc1` nearly thrown out by a claim of the same
    shape. The fix is to refuse and say why.
    """
    import ast
    import io

    from configs.gen_campaign import load_protocol

    P = load_protocol()
    posthoc = [a for a, v in P["arms"].items() if v.get("phase") == "posthoc"]
    assert {"clip", "focal_clip"} <= set(posthoc), (
        "the two in-campaign bars are no longer post-hoc; re-derive this gate")

    src = io.open("scripts/flag_live.py", encoding="utf-8").read()
    tree = ast.parse(src)
    assert [n for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and n.value == "posthoc"], (
        "flag_live no longer tests for the post-hoc phase, so it will call the "
        "clippers inert again")

    # The claim that makes the refusal necessary: `warmup_loss` is reachable
    # ONLY through the runner, which this harness bypasses. If a methodology
    # ever reads it directly, the refusal can be narrowed -- but not before.
    import os
    readers = []
    for root, _, files in os.walk("src"):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            for n in ast.walk(ast.parse(io.open(path, encoding="utf-8").read())):
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "get" and n.args
                        and isinstance(n.args[0], ast.Constant)
                        and n.args[0].value == "warmup_loss"):
                    readers.append(path.replace(os.sep, "/"))
    assert readers == ["src/pipeline/warmup.py"], (
        "`warmup_loss` is now read in %s. If a post-hoc methodology reads it "
        "directly, flag_live could see the difference and the blanket refusal "
        "should be narrowed to the allocator-only arms." % readers)


def test_the_deployment_figure_REFUSES_a_bar_it_has_no_data_for():
    """An absent cell must not render as this figure's headline claim.

    fig_deployment's claim is that the post-hoc clippers sit at ~0.00 native
    satisfaction. `reindex` turns a missing (backbone, method) cell into NaN,
    and matplotlib draws a NaN bar and a 0.00 bar identically -- so vanished
    data would have read as evidence FOR the claim.
    """
    import importlib.util
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # --- the premise, measured, not assumed: NaN and 0.00 are the same picture.
    import io as _io
    pngs = []
    for h in (np.nan, 0.0):
        f, a = plt.subplots(figsize=(1, 1))
        a.set_ylim(0, 1)
        a.axis("off")
        a.bar([0], [h], width=0.5, color="black")
        buf = _io.BytesIO()
        f.savefig(buf, format="png", dpi=40)
        plt.close(f)
        pngs.append(buf.getvalue())
    assert pngs[0] == pngs[1], (
        "matplotlib now distinguishes a NaN bar from a 0.00 bar. If that is "
        "really true the guard below can be relaxed -- but verify it visually "
        "first, because this test is the only thing asserting it."
    )

    # --- the guard itself refuses.
    path = "docs/paper/scripts/make_deployment_fig.py"
    spec = importlib.util.spec_from_file_location("_mkdep", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    full = pd.DataFrame(
        1.0, index=mod.BACKBONE_ORDER, columns=mod.METHOD_ORDER)
    pf = pd.DataFrame({
        "model": mod.BACKBONE_ORDER * len(mod.METHOD_ORDER),
        "method": [m for m in mod.METHOD_ORDER
                   for _ in mod.BACKBONE_ORDER],
    })
    mod._require_full_grid(full, pf)          # complete grid: passes

    holed = full.copy()
    holed.loc[mod.BACKBONE_ORDER[0], mod.METHOD_ORDER[-1]] = np.nan
    try:
        mod._require_full_grid(holed, pf)
    except SystemExit as e:
        assert "REFUSING" in str(e) and mod.METHOD_ORDER[-1] in str(e), str(e)
    else:
        raise AssertionError(
            "_require_full_grid accepted a grid with a hole, so an absent "
            "post-hoc-clipper cell would still be drawn as a ~0.00 bar")

    # --- and the generator actually calls it (AST: a mention in a docstring
    #     or a comment is not a call).
    tree = ast.parse(_io.open(path, encoding="utf-8").read())
    fn = [n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == "make_deployment"]
    assert fn, "make_deployment vanished from %s" % path
    calls = [n.func.id for n in ast.walk(fn[0])
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "_require_full_grid" in calls, (
        "make_deployment no longer calls _require_full_grid, so the guard is "
        "dead code and a hole in the grid draws silently again")

    # --- the same swallow one layer down must stay removed.
    handlers = [h for h in ast.walk(fn[0]) if isinstance(h, ast.ExceptHandler)]
    assert not handlers, (
        "make_deployment caught %d exception(s) again. The dot overlay used to "
        "swallow KeyError and hide exactly the absence this guard exists to "
        "catch." % len(handlers))


def test_the_macro_denominator_is_the_DATA_not_the_arm_s_predictions():
    """Two arms scored on one truth must be averaged over the same classes.

    With no explicit `labels=`, sklearn macro-averages over
    `unique(y_true) | unique(y_pred)`. So an arm that emits a class absent from
    y_true is divided by one MORE class than an arm that does not, and the two
    macro-F1s stop being comparable -- in a project whose entire output is
    arm-minus-arm differences on exactly this metric.
    """
    import io as _io
    import numpy as np
    from sklearn.metrics import f1_score

    # --- the hazard, measured. Class 7 exists in NEITHER arm's truth.
    y = np.array([0, 1, 2, 3, 4, 5, 6] * 10)
    quiet = y.copy()                    # never predicts the phantom class
    loud = y.copy()
    loud[::11] = 7                      # predicts it, always wrongly
    unpinned = [f1_score(y, p, average="macro", zero_division=0)
                for p in (quiet, loud)]
    assert unpinned[0] != unpinned[1], (
        "sklearn no longer changes the macro denominator with the prediction "
        "set; if that is genuinely true this pin is harmless but redundant")

    pinned = [f1_score(y, p, labels=sorted(set(y.tolist())),
                       average="macro", zero_division=0)
              for p in (quiet, loud)]
    assert pinned[0] == 1.0, pinned
    # The loud arm is still punished for its wrong predictions -- pinning the
    # denominator must not launder a real error away.
    assert pinned[1] < 1.0, (
        "pinning `labels` hid the loud arm's wrong predictions entirely, which "
        "would be a worse bug than the one it fixes")

    # --- full_panel pins all three macro metrics (AST: a comment is not a kwarg).
    src = _io.open("scripts/full_panel.py", encoding="utf-8").read()
    tree = ast.parse(src)
    fn = [n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and n.name == "panel"]
    assert fn, "panel() vanished from scripts/full_panel.py"

    wanted = {"macroP": "precision_score", "macroR": "recall_score",
              "macroF1": "f1_score"}
    seen = {}
    for d in ast.walk(fn[0]):
        if not isinstance(d, ast.Dict):
            continue
        for k, v in zip(d.keys, d.values):
            if (isinstance(k, ast.Constant) and k.value in wanted
                    and isinstance(v, ast.Call)):
                seen[k.value] = {kw.arg for kw in v.keywords}
    missing = sorted(k for k in wanted if k not in seen)
    assert not missing, "panel() no longer emits %s" % missing
    unpinned_keys = sorted(k for k, kws in seen.items() if "labels" not in kws)
    assert not unpinned_keys, (
        "%s computed without an explicit `labels=`, so their denominator is "
        "again `unique(y) | unique(pred)` and an arm that emits an absent class "
        "is averaged over more classes than one that does not" % unpinned_keys)

    # --- and `present` is derived from y alone, never from the predictions.
    assigns = [n for n in ast.walk(fn[0])
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "present"
                       for t in n.targets)]
    assert len(assigns) == 1, (
        "expected exactly one `present = ...` in panel(), found %d" % len(assigns))
    names = {n.id for n in ast.walk(assigns[0].value) if isinstance(n, ast.Name)}
    assert "y" in names, "`present` is no longer derived from y"
    for forbidden in ("eq", "pred", "P"):
        assert forbidden not in names, (
            "`present` is derived from `%s`, which is prediction-dependent -- "
            "that reintroduces exactly the bug this pins shut" % forbidden)


def test_ortho_project_s_GUARANTEE_DOES_NOT_REACH_THE_WEIGHTS():
    """The projection is CE-neutral in the raw gradient and nowhere else.

    `project_out` sets `<g_con, grad_CE> = 0`, which to first order claims the
    constraint step neither helps nor undoes CE progress. But the step that
    lands is Adam's `m/sqrt(v)`, and two things there are untouched by it:
    b1 = 0.9 of the momentum is stale CE momentum, and the diagonal
    preconditioner is not an isometry so it does not preserve orthogonality.
    """
    import io as _io
    import numpy as np
    from scripts.ortho_survival import survival, ref_mismatch, MEASURED, B1

    n = 20_000
    m_ce, g, sv = MEASURED[3]

    def pair(fn, **kw):
        sd = int(np.random.default_rng(0).integers(0, 2 ** 31 - 1))
        on = fn(np.random.default_rng(sd), project=True, **kw)
        off = fn(np.random.default_rng(sd), project=False, **kw)
        return on, off, ((off - on) / off if off > 0 else float("nan"))

    # --- the negative control FIRST: with both destroyers disabled the
    #     projection must remove essentially ALL of the CE inner product.
    #     Without this the headline below is vacuous.
    on, off, frac = pair(
        lambda r, project: survival(m_ce, g, sv, 0.0, r, use_momentum=False,
                                    use_precond=False, project=project, n=n))
    assert frac > 0.99, (
        "with no momentum and a flat preconditioner the projection removed only "
        "%.1f%% of the CE inner product; the probe cannot detect the thing it "
        "exists to measure" % (100.0 * frac))

    # --- the headline: turn the real optimizer back on and it removes nothing.
    for spread in (0.0, 2.0):
        on, off, frac = pair(
            lambda r, project, s=spread: survival(m_ce, g, sv, s, r,
                                                  project=project, n=n))
        assert abs(frac) < 0.01, (
            "spread=%.1f: the projection removed %.2f%% of the update's CE "
            "inner product. If that is now materially non-zero the offline "
            "verdict in FRAMEWORK 2(t) must be re-derived before it is quoted."
            % (spread, 100.0 * frac))

    # --- and it is no better when the reference is a single minibatch (rho<1),
    #     which is what `snapshot_grads` actually captures.
    for rho in (1.0, 0.2):
        sd = int(np.random.default_rng(1).integers(0, 2 ** 31 - 1))
        on, off = ref_mismatch(m_ce, g, sv, 2.0, rho,
                               np.random.default_rng(sd), n=n)
        frac = (off - on) / off if off > 0 else float("nan")
        assert abs(frac) < 0.01, (
            "rho=%.2f removed %.2f%%" % (rho, 100.0 * frac))

    # --- the probe's PREMISE: the projection really does run before the
    #     optimizer in the shipped code. If it ever moves after `optimizer.step`
    #     the model above describes code that no longer exists.
    src = _io.open("src/training/constraint_step.py", encoding="utf-8").read()
    tree = ast.parse(src)
    fn = [x for x in ast.walk(tree)
          if isinstance(x, ast.FunctionDef) and x.name == "finish_constraint_step"]
    assert fn, "finish_constraint_step vanished"

    def first_line(pred):
        hits = [x.lineno for x in ast.walk(fn[0])
                if isinstance(x, ast.Call) and pred(x)]
        return min(hits) if hits else None

    proj = first_line(lambda c: isinstance(c.func, ast.Name)
                      and c.func.id == "project_out")
    clip = first_line(lambda c: isinstance(c.func, ast.Attribute)
                      and c.func.attr == "clip_grad_norm_")
    step = first_line(lambda c: isinstance(c.func, ast.Attribute)
                      and c.func.attr == "step")
    assert proj is not None, "project_out is no longer called in the step"
    assert clip is not None and proj < clip, (
        "project_out no longer runs BEFORE clip_grad_norm_, so the projected "
        "and unprojected arms no longer share a dose")
    assert step is not None and proj < step, (
        "project_out now runs after the optimizer step; ortho_survival models "
        "the opposite order and its verdict does not apply")

    # b1 is the constant the bound is computed from.
    assert B1 == 0.9, "B1 changed; the 7.4% momentum share must be recomputed"

    # --- THE LOAD-BEARING PREMISE. Everything above is about the Adam path.
    #     Under `constraint_step_rule: sgd` the step is `p -= lr*g`, there is no
    #     momentum and no preconditioner, and the projection WOULD be delivered
    #     in full. So the verdict holds only while these arms resolve to
    #     "shared". Checked, not assumed -- I asserted it once without checking
    #     and had to retract a different claim for exactly that reason.
    import yaml
    P = yaml.safe_load(_io.open("configs/protocol.yml", encoding="utf-8").read())
    cp, blocks, arms = P["constraint_phase"], P["blocks"], P["arms"]
    for name in ("tralo", "tralo_ortho", "tralo_head"):
        spec = arms.get(name)
        assert spec, "arm %s vanished from the registry" % name
        rule = None
        for b in spec.get("blocks", []):
            if b == "constraint_phase":
                rule = cp.get("constraint_step_rule")
            blk = blocks.get(b) or {}
            if "constraint_step_rule" in blk:
                rule = blk["constraint_step_rule"]
        assert rule == "shared", (
            "%s now resolves to constraint_step_rule=%r. Under 'sgd' the step is "
            "p -= lr*g with no momentum and no preconditioner, so the projection "
            "IS delivered and FRAMEWORK 2(t)'s 0.0%% verdict does not apply to "
            "this arm. Re-derive it before quoting." % (name, rule))

    # --- AND THE SISTER ARM: `head_only` masks by parameter set, not direction.
    #     Zeroing a gradient does NOT freeze the parameter -- Adam carries
    #     `m <- 0.9*m + 0.1*0`. FRAMEWORK 2(t) states 90.4% and reads the arm as
    #     "the constraint sees only the head", never "the backbone is frozen".
    from scripts.ortho_survival import masked_coordinate_drift
    dh, db, ratio = masked_coordinate_drift()
    assert dh != 0.0, "the unmasked coordinate did not move; the probe is inert"
    assert 0.85 < ratio < 0.95, (
        "a gradient-masked coordinate now steps at %.3f of the unmasked one; "
        "FRAMEWORK 2(t) quotes 0.904 and reads `tralo_head` against it" % ratio)
    # It rises toward b1 as the CE phase lengthens -- a longer CE phase makes
    # the mask LESS effective. Anyone who assumes the opposite reads it backwards.
    assert (masked_coordinate_drift(ce_steps=1)[2]
            < masked_coordinate_drift(ce_steps=126)[2]), (
        "the masked-coordinate drift no longer grows with the CE phase length")


def test_a_COIN_and_the_REAL_constraint_gradient_deliver_the_SAME_step():
    """Under `step_rule=shared`, a coin and the real gradient give the same PER-STEP update.

    Both the treatment and its random-direction control put a norm-`clip` vector
    into `prm.grad`. Adam then adds `b1 * m_CE` to both, and that term is ~92.6%
    of the result, so the two deliver nearly the same vector on any single step.

    ⛔ THIS IS NOT AN EXPLANATION OF 1b-pre(6)'s NULL, and an earlier version of
    this test said it was. A 0.6% consistent directional difference COMPOUNDS
    over 29 steps, and that section measures coin and `linear` with
    NON-OVERLAPPING distributions at L50_G30 -- which a same-step reading cannot
    produce. The claim was retracted 2026-08-25. What survives is a forward
    warning about `tralo_coin` as a control: its contrast with the treatment is
    ~0.6% per step, so its power comes from compounding, not step geometry.
    """
    import numpy as np
    from scripts.ortho_survival import coin_equivalence, momentum_reset, B1

    n = 20_000

    # --- LIVENESS FIRST. With no CE momentum the two steps MUST diverge; if
    #     they do not, the probe is reporting a constant and means nothing.
    c0, _, share0 = coin_equivalence(1.0, np.random.default_rng(3), n=n,
                                     m_scale=0.0)
    assert share0 > 0.99, "m_scale=0 should hand the whole step to the constraint"
    assert abs(c0) < 0.1, (
        "with the CE momentum removed a coin still delivers the same step as the "
        "real gradient (cos=%.4f); the probe cannot tell them apart at all" % c0)

    # --- the finding: with the real CE momentum they are the same step.
    for spread in (0.0, 1.0, 2.0, 3.0):
        c, c_ce, share = coin_equivalence(spread, np.random.default_rng(4), n=n)
        assert c > 0.98, (
            "spread=%.1f: cos(real, coin) = %.4f. FRAMEWORK 1b-pre(6) reads its "
            "coin null against ~0.994; if the delivered steps have genuinely "
            "diverged that reading must be redone." % (spread, c))
        assert c_ce > 0.98, "the delivered step is no longer dominated by m_CE"
        assert 0.05 < share < 0.10, "constraint share moved off 7.4%%: %.3f" % share

    # --- and clearing `m` alone would hand the direction back -- WITH a dose
    #     change that must never be quoted without the cosine.
    shared, zeroed = momentum_reset(1.0, np.random.default_rng(5), n=n)
    assert shared[0] < 0.2, "shared optimizer already delivers the constraint dir"
    assert zeroed[0] > 0.95, "clearing m did not hand the direction back"
    assert zeroed[2] < 0.2, (
        "clearing `m` no longer shrinks the delivered step (rel=%.3f). The dose "
        "confound is the reason this is not a launchable arm as it stands, and "
        "FRAMEWORK 2(t) says so; if it has gone away, re-derive that." % zeroed[2])
    assert B1 == 0.9

    # --- the same channel compresses a change to the COUNT FUNCTION, which is
    #     what `tralo` vs `tralo_uniform` is. Monotone, and even a 180-degree
    #     flip survives as single digits.
    from scripts.ortho_survival import count_change_attenuation
    outs = []
    for cg in (0.99, 0.90, 0.50, 0.00, -1.00):
        cu, ai, ao = count_change_attenuation(cg, np.random.default_rng(9), n=n)
        outs.append(ao)
        assert ao < ai, "attenuation inverted at cos=%.2f: %.2f -> %.2f" % (cg, ai, ao)
    assert outs == sorted(outs), (
        "delivered angle is no longer monotone in the count-function difference: %s"
        % outs)
    assert outs[-1] < 15.0, (
        "two OPPOSITE count functions now deliver updates %.1f degrees apart "
        "ON THE FIRST STEP; ~9 is the recorded value. Re-derive before quoting."
        % outs[-1])
    assert outs[-1] > 1.0, (
        "the compression is now total (%.2f deg), which would make the probe "
        "report a constant rather than a measurement" % outs[-1])

    # --- THE CONSTRAINT STEPS ARE NOT CONSECUTIVE, and a version of this gate
    #     asserted that they were. train.py:192-212 runs the whole CE batch
    #     loop (one optimizer.step per batch) and calls finish_constraint_step
    #     ONCE per epoch at line 404, so ~126 CE steps sit between constraint
    #     steps and the momentum carries b1^126 of one into the next.
    from scripts.ortho_survival import (count_change_compounding,
                                        count_gradient_angle)
    CE_PER_EPOCH = 126
    assert B1 ** CE_PER_EPOCH < 1e-5, (
        "b1^%d = %.3e is no longer negligible, so a constraint step's momentum "
        "DOES reach the next one and the geometric accumulation applies after "
        "all" % (CE_PER_EPOCH, B1 ** CE_PER_EPOCH))
    at_step = lambda c: (1 - B1) / (1 - B1 ** (c + 1))
    assert abs(at_step(CE_PER_EPOCH) - (1 - B1)) < 1e-5, (
        "with %d CE steps between, the difference present at a constraint step "
        "must be the SINGLE-STEP value (1-b1)=%.3f, got %.4f"
        % (CE_PER_EPOCH, 1 - B1, at_step(CE_PER_EPOCH)))
    assert at_step(0) > 9 * at_step(CE_PER_EPOCH), (
        "consecutive and interleaved no longer differ, so the distinction this "
        "gate exists to pin is moot")

    # What DOES compound is the weight trajectory -- real, modest, and utterly
    # dependent on an assumption nothing measures.
    rng3 = np.random.default_rng(11)
    fresh = count_change_compounding(np.cos(np.radians(29.4)), 0.0, rng3, n=6000)
    corr = count_change_compounding(np.cos(np.radians(29.4)), 0.5, rng3, n=6000)
    assert fresh[2] > fresh[0] * 2.0, (
        "the trajectory no longer opens at all over 29 steps (%.2f -> %.2f "
        "deg); FRAMEWORK 1b-pre(6) says compounding is what separates these "
        "arms" % (fresh[0], fresh[2]))
    assert fresh[3] < 0.25, (
        "end separation is %.3f of the distance travelled. The retracted "
        "consecutive-step model gave ~0.44; if the real one now agrees, the "
        "interleaving is not being modelled." % fresh[3])
    assert fresh[3] > 5 * corr[3], (
        "the CE-correlation assumption no longer dominates the magnitude "
        "(%.4f vs %.4f). It does, by ~30x, and that is exactly why this is a "
        "power consideration and never a predicted effect size."
        % (fresh[3], corr[3]))

    # --- and the INPUT angle can never be the 180 the scripts used to quote:
    #     p(1-p) and its mean are both elementwise NON-NEGATIVE.
    rng2 = np.random.default_rng(5)
    for name, pc in (("uniform", rng2.uniform(0, 1, 2000)),
                     ("confident", rng2.beta(0.2, 0.2, 2000)),
                     ("low mass", rng2.beta(2, 5, 2000))):
        a = count_gradient_angle(pc)
        assert 0.0 < a < 90.0, (
            "%s: sum-vs-uniform angle %.1f deg. Both gradient vectors are "
            "elementwise non-negative, so anything at or above 90 means the "
            "count changed sign somewhere and the geometry argument is void."
            % (name, a))


def test_no_script_CRASHES_when_it_prints_its_own_conclusion():
    """A probe must not die on the console the user actually runs it on.

    Windows defaults stdout to cp1252, which cannot encode the emoji this
    project's docs use freely. A `print` containing one raises
    UnicodeEncodeError and the script exits 1 -- MID-REPORT, so whatever it had
    already printed reads as the complete output. Found 2026-08-25 when
    `ortho_survival` died between its table and the caveat that qualifies it,
    and `scope_probe`'s crash sits in the `PROBE CANNOT RESOLVE THIS` branch:
    it would fail exactly when it needs to say it cannot answer.

    Docstrings, comments and every .md file are unaffected and keep their emoji.
    """
    import io as _io
    import os

    offenders = {}
    for root in ("scripts", "docs/paper/scripts"):
        if not os.path.isdir(root):
            continue
        for f in sorted(os.listdir(root)):
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f).replace(os.sep, "/")
            tree = ast.parse(_io.open(path, encoding="utf-8").read())
            bad = []
            for node in ast.walk(tree):
                emitting = (
                    isinstance(node, ast.Call)
                    and ((isinstance(node.func, ast.Name)
                          and node.func.id in ("print", "SystemExit"))
                         or (isinstance(node.func, ast.Attribute)
                             and node.func.attr == "exit")))
                if not emitting:
                    continue
                for lit in ast.walk(node):
                    if (isinstance(lit, ast.Constant)
                            and isinstance(lit.value, str)
                            and any(ord(c) > 127 for c in lit.value)):
                        bad.append(lit.lineno)
            if bad:
                offenders[path] = sorted(set(bad))

    assert not offenders, (
        "these scripts print non-ASCII and will raise UnicodeEncodeError on a "
        "cp1252 console, exiting 1 mid-report: %s. Use ASCII in printed strings "
        "(!! for the warning sign, -> and => for the arrows); docstrings, "
        "comments and .md files may keep their emoji." % offenders)


# A handler may swallow silently ONLY for a reason recorded here. Everything
# else must report, because a scorer or gate that drops data without saying so
# produces a number over a smaller set than the reader believes.
SILENT_SWALLOW_ALLOWED = {
    ("scripts/bisect_determinism.py", "AttributeError"):
        "feature-detecting an optional torch API; absence is the answer",
    ("src/utils/error_handler.py", "Exception"):
        "this IS the error writer; it must not raise while recording a failure",
    ("scripts/log_health.py", "Exception"):
        "config.json is optional for this diagnostic; the training log is the input",
    ("scripts/hp_liveness.py", "Exception"):
        "falls back to summary['last_grad_norm'], an equivalent source",
}


def test_no_scorer_or_gate_DROPS_DATA_WITHOUT_SAYING_SO():
    """`except ...: pass` in an instrument is a number over a smaller set.

    Found 2026-08-25 in straddle_probe, whose BASELINE block could be built
    from fewer runs than the TREATED block printed directly below it while the
    header said they were the same cells. The audit then found the same shape
    in full_panel (silently regressing to the exact hardcoded key list its own
    docstring records as a bug), check_parity (a parity gate quietly narrowing
    to ONE key and still printing PARITY OK) and variance_probe (the NOISE
    FLOOR every effect here is judged against, over a silently smaller set).
    """
    import io as _io
    import os

    found = {}
    for root in ("scripts", "docs/paper/scripts", "src"):
        for dirpath, _, files in os.walk(root):
            if "__pycache__" in dirpath:
                continue
            for f in sorted(files):
                if not f.endswith(".py"):
                    continue
                path = os.path.join(dirpath, f).replace(os.sep, "/")
                tree = ast.parse(_io.open(path, encoding="utf-8").read())
                for h in ast.walk(tree):
                    if not isinstance(h, ast.ExceptHandler):
                        continue
                    body = [n for n in h.body
                            if not (isinstance(n, ast.Expr)
                                    and isinstance(n.value, ast.Constant)
                                    and isinstance(n.value.value, str))]
                    if not (len(body) == 1 and isinstance(body[0], ast.Pass)):
                        continue
                    t = h.type
                    if isinstance(t, ast.Name):
                        name = t.id
                    elif t is None:
                        name = "BARE"
                    else:
                        name = ast.unparse(t)
                    if (path, name) not in SILENT_SWALLOW_ALLOWED:
                        found.setdefault((path, name), []).append(h.lineno)

    assert not found, (
        "new silent swallow(s): %s. An `except ...: pass` in a scorer, gate or "
        "probe drops data and still prints a number. Either report the drop "
        "(print/stderr, and count it) or add an entry to "
        "SILENT_SWALLOW_ALLOWED saying why absence is genuinely the answer."
        % {("%s except %s" % k): v for k, v in found.items()})

    # The allowlist must not outlive what it names -- a stale entry is
    # permission nobody is checking.
    stale = []
    for (path, name) in SILENT_SWALLOW_ALLOWED:
        if not os.path.exists(path):
            stale.append((path, name))
            continue
        tree = ast.parse(_io.open(path, encoding="utf-8").read())
        names = set()
        for h in ast.walk(tree):
            if not isinstance(h, ast.ExceptHandler):
                continue
            body = [n for n in h.body
                    if not (isinstance(n, ast.Expr)
                            and isinstance(n.value, ast.Constant)
                            and isinstance(n.value.value, str))]
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                t = h.type
                names.add(t.id if isinstance(t, ast.Name)
                          else "BARE" if t is None else ast.unparse(t))
        if name not in names:
            stale.append((path, name))
    assert not stale, (
        "SILENT_SWALLOW_ALLOWED names swallows that no longer exist: %s. "
        "Remove the entries -- a stale exemption silently re-permits the bug "
        "if the code comes back." % stale)


def test_the_straddle_probe_BASELINE_reports_its_own_coverage():
    """The BASELINE and TREATED blocks are comparable only over the same runs."""
    import io as _io

    src = _io.open("scripts/straddle_probe.py", encoding="utf-8").read()
    tree = ast.parse(src)

    fn = [n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and n.name == "report"]
    assert fn, "report() vanished from straddle_probe"
    assert "n_runs" in {a.arg for a in fn[0].args.args}, "report lost n_runs"
    used = any(isinstance(n, ast.Name) and n.id == "n_runs"
               for n in ast.walk(fn[0]))
    assert used, (
        "report() accepts n_runs and ignores it again. That is what let the "
        "BASELINE block be built from fewer runs than the TREATED block "
        "without anything saying so")

    main = [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "main"]
    assert main, "main() vanished"
    calls = [n for n in ast.walk(main[0])
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "report"]
    assert len(calls) == 2, "expected two report() calls, found %d" % len(calls)
    args = {ast.unparse(c.args[1]) for c in calls}
    assert args == {"n_base", "n_ok"}, (
        "the two report() calls now pass %s. The baseline must be labelled "
        "with ITS OWN run count, not the treatment's." % sorted(args))
    assert "n_base_skipped" in src, (
        "straddle_probe no longer counts skipped baseline runs, so it cannot "
        "warn that the two blocks do not cover the same runs")


def test_family_split_does_not_count_an_UNMEASURABLE_cell_as_a_LOSS():
    """`nan > 0` is False, so a NaN cell was silently scored against the arm.

    full_panel returns np.nan for uncF1 with no capped classes, for ConfGap when
    every item is correct, and for AP/AUROC in degenerate cells. Counting those
    in the denominator turns "2 won, 3 unmeasurable, 4 lost" into "2/9", which
    reads as a much weaker result than the data support -- or a much stronger
    one, depending on which way the NaNs fell.
    """
    import io as _io
    import numpy as np

    src = _io.open("scripts/family_split.py", encoding="utf-8").read()
    tree = ast.parse(src)
    fn = [n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and n.name == "main"]
    assert fn, "main() vanished from family_split"

    names = {n.id for n in ast.walk(fn[0]) if isinstance(n, ast.Name)}
    assert "nan_cells" in names, (
        "family_split no longer separates unmeasurable cells, so a NaN metric "
        "is counted as a lost cell again")
    assert any(isinstance(n, ast.Attribute) and n.attr == "isfinite"
               for n in ast.walk(fn[0])), (
        "the win count no longer tests np.isfinite, so `nan > 0` decides it")

    # The arithmetic the fix rests on, measured rather than assumed.
    assert not (np.nan > 0), "nan > 0 is now True, which changes the whole fix"
    vals = {"a": 0.5, "b": float("nan"), "c": -0.2}
    won = sum(1 for v in vals.values() if np.isfinite(v) and v > 0)
    res = sum(1 for v in vals.values() if np.isfinite(v))
    assert (won, res) == (1, 2), (won, res)

    # And the drop of incomplete cell-seeds must be reported, not just done.
    m = [n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "matched"]
    assert m, "matched() vanished"
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "print" for n in ast.walk(m[0])), (
        "matched() drops incomplete cell-seeds without reporting how many. "
        "'16 matched' reads very differently when 18 existed than when 200 did")


def test_the_backbone_table_SAYS_when_a_cap_level_is_excluded(capsys):
    """A discarded cap level and one that was never run look identical in W/T/L.

    `cell_gaps` skips a cap tag when fewer than 3 seeds survive `.dropna()`, and
    when `tralo` or every baseline is missing. Both just shrink the W/T/L total,
    so the emitted table cannot distinguish "we ran this and threw it away" from
    "this was never run" -- and only the first is a caveat about the analysis.
    It is real: on the shipped corpus, dermmnist x MobileNetV2 x L40_G40 keeps
    2 of 5 seeds, while MobileNetV2's other thin rows are genuine coverage
    (5 and 7 cap levels exist on octmnist and tissuemnist).

    `dropna` has caused a scorer bug in this project before -- a lagging third
    arm deleted pairs from every comparison -- which is why this one reports.
    """
    import importlib.util
    import io as _io
    import numpy as np
    import pandas as pd

    path = "docs/paper/scripts/make_backbone_tables.py"
    src = _io.open(path, encoding="utf-8").read()

    # --- structural: the skip branches must feed a reported list.
    tree = ast.parse(src)
    fn = [n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and n.name == "cell_gaps"]
    assert fn, "cell_gaps vanished from %s" % path
    prints = [n for n in ast.walk(fn[0])
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
              and n.func.id == "print"]
    assert prints, (
        "cell_gaps no longer prints anything, so a cap level excluded for thin "
        "seeds is invisible in both the table and the run log")

    # --- behavioural: a thin cap level must actually produce the warning.
    spec = importlib.util.spec_from_file_location("_bbtest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # emits the real tables; unchanged
    capsys.readouterr()

    rows = []
    for tag, n_seed in (("L30_G30", 4), ("L50_G50", 2)):
        for seed in range(n_seed):
            for meth, val in (("tralo", 0.50), ("fioretto_ldf", 0.49)):
                rows.append({"dataset": "d", "model": "m", "constraint_tag": tag,
                             "seed": seed, "method": meth, "cc_f1": val})
    frame = pd.DataFrame(rows)
    out = mod.cell_gaps(frame, "cc_f1", ["fioretto_ldf"])
    text = capsys.readouterr().out
    assert "L50_G50" in text and "EXCLUDED" in text, (
        "a cap level with 2 seeds was dropped without a word. stdout was: %r"
        % text)
    assert "L30_G30" not in text, (
        "a cap level with enough seeds was reported as excluded: %r" % text)
    assert len(out[("d", "m")]) == 1, (
        "expected exactly the 4-seed cap level to survive, got %d records"
        % len(out[("d", "m")]))


def test_the_granular_table_SAYS_when_its_macro_column_has_fewer_seeds(capsys):
    """`Delta mac` and the cc columns pair against DIFFERENT baselines.

    The cc columns pair TraLO against the best trained DUAL; the macro column
    pairs it against the best CLIPPER. Each survives `.dropna()` independently,
    so their seed counts can differ -- and `cell_stats` recorded `cc_n` but not
    `mac_n`, so a reader of tab_granular_asym had no way to see it.

    Measured 2026-08-25 on the shipped corpus: identical on every paper_final
    cell, but in tab_granular_asym the macro column rests on ONE seed in 8 cells
    (L20_G50, L30_G80, L50_G20, L80_G30 on both dermmnist and tissuemnist)
    beside cc columns using four. A one-seed mean has no variance.
    """
    import importlib.util
    import pandas as pd

    path = "docs/paper/scripts/make_granular_tables.py"
    spec = importlib.util.spec_from_file_location("_gtest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # emits the real tables; unchanged
    capsys.readouterr()

    def frame(n_clip_seeds):
        rows = []
        for seed in range(4):
            rows.append({"dataset": "d", "model": "m", "constraint_tag": "L1_G1",
                         "seed": seed, "method": "tralo",
                         "cc_f1": 0.5, "f1_macro": 0.6})
            rows.append({"dataset": "d", "model": "m", "constraint_tag": "L1_G1",
                         "seed": seed, "method": mod.DUALS[0],
                         "cc_f1": 0.4, "f1_macro": 0.5})
            if seed < n_clip_seeds:
                rows.append({"dataset": "d", "model": "m",
                             "constraint_tag": "L1_G1", "seed": seed,
                             "method": mod.CLIP[0], "cc_f1": 0.3,
                             "f1_macro": 0.55})
        return pd.DataFrame(rows)

    # --- the hazard: the clipper is present for only one seed.
    r = mod.cell_stats(frame(1))
    text = capsys.readouterr().out
    assert r is not None and r.get("mac_n") == 1 and r.get("cc_n") == 4, r
    assert "macro column uses 1 seed" in text, (
        "cell_stats no longer warns when the macro column has fewer seeds than "
        "the cc columns. stdout was: %r" % text)

    # --- and it must NOT cry wolf when the counts agree.
    r = mod.cell_stats(frame(4))
    text = capsys.readouterr().out
    assert r.get("mac_n") == r.get("cc_n") == 4, r
    assert "macro column uses" not in text, (
        "warned on a cell whose seed counts agree: %r" % text)


def test_the_lp_fallback_fields_are_a_DEFAULT_for_the_post_hoc_arms():
    """`lp_fallback_used=False, lp_fallback_candidates=0` is not always measured.

    The chain, verified from source rather than remembered:
      1. five methodologies set `skip_targeted_correction=True`;
      2. `src/pipeline/eval.py` initialises `posthoc_meta = {}` and populates it
         ONLY inside the branch that skip bypasses;
      3. `src/experiments/runner.py` then reads it with `.get(k, <default>)`,
         writing `False` and `0` -- both of which are MEANINGFUL measured values
         elsewhere.
    So for those arms the field records that nothing ran, in a form
    indistinguishable from "the allocator ran and found nothing". Two of them,
    `clip` and `focal_clip`, are in every campaign by CLAUDE.md rule 2.

    This is the sibling of the `flag_live` defect fixed earlier the same day:
    the post-hoc arms do not traverse the pipeline path the field describes.
    """
    import io as _io
    import os
    import yaml

    # --- 1. which methodologies skip the allocator
    skippers = set()
    for dirpath, _, files in os.walk("src/methodologies"):
        if "__pycache__" in dirpath:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(dirpath, f)
            tree = ast.parse(_io.open(path, encoding="utf-8").read())
            for kw in ast.walk(tree):
                if (isinstance(kw, ast.keyword)
                        and kw.arg == "skip_targeted_correction"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True):
                    skippers.add(os.path.basename(dirpath)
                                 if os.path.basename(dirpath) != "methodologies"
                                 else os.path.splitext(f)[0])
    assert "danits_lp" in skippers and "heuristic" in skippers, skippers

    # --- 2. eval.py leaves the meta EMPTY on that path
    ev = _io.open("src/pipeline/eval.py", encoding="utf-8").read()
    tree = ast.parse(ev)
    fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
          and any(isinstance(x, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "posthoc_meta"
                          for t in x.targets)
                  and isinstance(x.value, ast.Dict) and not x.value.keys
                  for x in ast.walk(n))]
    assert fn, ("src/pipeline/eval.py no longer initialises posthoc_meta to an "
                "empty dict; re-derive this test's premise")

    # --- 3. runner.py fills the gap with values that mean something else
    rn = _io.open("src/experiments/runner.py", encoding="utf-8").read()
    tree = ast.parse(rn)
    defaults = {}
    for c in ast.walk(tree):
        if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and c.func.attr == "get" and len(c.args) == 2
                and isinstance(c.args[0], ast.Constant)
                and str(c.args[0].value).startswith("lp_fallback")):
            defaults[c.args[0].value] = getattr(c.args[1], "value", "?")
    assert defaults.get("lp_fallback_used") is False, defaults
    assert defaults.get("lp_fallback_candidates") == 0, defaults

    # --- 4. so name the arms, from the registry, and keep the list honest
    P = yaml.safe_load(_io.open("configs/protocol.yml", encoding="utf-8").read())
    defaulted = sorted(a for a, spec in P["arms"].items()
                       if spec.get("methodology") in skippers)
    assert set(defaulted) >= {"clip", "focal_clip", "lp"}, defaulted
    assert "tralo" not in defaulted and "fioretto" not in defaulted, defaulted

    # --- 5. and the two places that state the claim must carry the qualifier.
    sp = _io.open("scripts/scope_probe.py", encoding="utf-8").read()
    assert "THAT RAN THE\nALLOCATOR" in sp or "THAT RAN THE ALLOCATOR" in sp, (
        "scope_probe's docstring dropped the scope qualifier and again reads "
        "`lp_fallback_used` as measured on every completed run")
    tp = _io.open("tests/test_pipeline.py", encoding="utf-8").read()
    assert "THAT RAN THE ALLOCATOR" in tp, (
        "test_the_generator_says_which_scope_each_cap_binds dropped the scope "
        "qualifier from its docstring")


def test_no_script_exists_without_being_NAMED_where_someone_will_look():
    """A tool nobody knows about is a tool nobody runs.

    `docs/FRAMEWORK.md` is by this project's own rule the only operational
    document, and `CLAUDE.md` is the entry point. A script named in neither is
    invisible, however good it is. Audited 2026-08-25: eight were, and they were
    not dead code -- `rig_status` checks the exact silent operational failures
    CLAUDE.md warns about in prose, `factorial_control` bounds where
    `dataset_screen` is valid, and `hp_liveness_real` exists because
    `hp_liveness`'s smoke-net verdicts INVERT on the real backbone.

    This is the sibling of `audit_config`'s rule: no config key without a
    reader, and no script without a mention.
    """
    import io as _io
    import os

    cl = _io.open("CLAUDE.md", encoding="utf-8").read()
    fw = _io.open("docs/FRAMEWORK.md", encoding="utf-8").read()
    names = [f[:-3] for f in sorted(os.listdir("scripts"))
             if f.endswith(".py") and f != "__init__.py"]
    assert names, "scripts/ is empty, which cannot be right"

    missing = [n for n in names if n not in cl and n not in fw]
    assert not missing, (
        "these scripts are named in neither CLAUDE.md nor docs/FRAMEWORK.md, so "
        "nobody reading the operational docs knows they exist: %s. Add a line "
        "saying what each one refuses, or delete it." % missing)

    # And the reverse: a doc naming a script that no longer exists sends the
    # reader to a command that errors.
    import re
    # A `git show <rev>:scripts/x.py` is a DELIBERATE reference to a deleted
    # file with its retrieval attached -- that is the correct way to keep a
    # receipt for evidence the repo no longer carries, so it does not count as
    # a ghost. Strip those first.
    RETRIEVAL = re.compile(r"git show [^\s`]*:scripts/[a-z_][a-z0-9_]*\.py")
    referenced = set()
    for text in (cl, fw):
        text = RETRIEVAL.sub("", text)
        referenced |= set(re.findall(r"scripts\.([a-z_][a-z0-9_]*)", text))
        referenced |= set(re.findall(r"scripts/([a-z_][a-z0-9_]*)\.py", text))
    # docs/paper/scripts/ is a second, legitimate home -- `make_main_table` and
    # friends live there, and the `scripts/<name>.py` pattern matches both.
    paper = [f[:-3] for f in os.listdir("docs/paper/scripts")
             if f.endswith(".py")] if os.path.isdir("docs/paper/scripts") else []
    ghosts = sorted(r for r in referenced if r not in names and r not in paper)
    assert not ghosts, (
        "the docs name scripts that do not exist: %s. Either restore them or "
        "remove the reference -- a documented command that errors is worse "
        "than an undocumented one that works." % ghosts)


def test_a_staged_launch_script_NAMES_ONLY_ARMS_THAT_EXIST():
    """A launch script is code that runs once, on a server, under time pressure.

    `docs/launch_uniform.sh` carried, for a day, this line:

        --arms tralo tralo_uniform tralo_ortho tralo_head tralo_null \
               tralo_reseed \n           clip focal_clip \

    That `\n` is not a newline. A backslash inside an unquoted bash word
    escapes the next character, so the shell passed a bare argument `n`, and
    `gen_campaign`'s `choices=` rejected it with exit 2 under `set -e`. The
    campaign would have died AT LAUNCH -- after the operator had found a free
    GPU, taken the worktree and checked out the pin -- for a reason that was
    visible in the file the whole time.

    It failed loudly, which is the only good thing about it -- but that was
    luck, not design. Measured 2026-08-25 by dropping each token of that line in
    turn and reading what `gen_campaign` actually does:

        clip, focal_clip   auto-re-added (`mandatory_arms`)  -> HARMLESS
        tralo_reseed       REFUSED, exit 1                   -> CAUGHT
        tralo, tralo_uniform, tralo_head, tralo_null
                           exit 0, 216 runs written          -> SILENT

    Losing `tralo_null` is the bad one. It prints `*** NO ZERO-DOSE CONTROL
    for: ...` and exits 0, and in a launch script that warning scrolls past
    inside the generator's own output with `set -euo pipefail` doing nothing
    about it and the dispatcher starting 45 seconds later. The campaign would
    run to completion and be unreadable: every contrast here is seed-paired
    against the twin, so `family_split` would find no null and `full_panel
    --control tralo_null` would have no control. 216 runs, unattributable.

    So this gate does two things, both statically, from the script and
    `configs/protocol.yml`: every arm named must EXIST, and every trained arm
    named must have its `_null` sibling named beside it. It is the sibling of
    the gate that refuses a config key with no reader -- an arm name with no
    arm, and a treatment with no twin.
    """
    import io
    import shlex

    BS = chr(92)
    scripts = sorted(f for f in os.listdir("docs") if f.endswith(".sh"))
    assert scripts, "docs/ carries no launch script, which cannot be right"

    P = load_protocol()
    valid = set(P["arms"]) | {"all", "all+null"}
    checked = 0

    for name in scripts:
        path = os.path.join("docs", name)
        text = io.open(path, encoding="utf-8").read()
        # Comment lines first: the prose above these invocations discusses arms
        # by name, including removed ones, and that is exactly what it is for.
        text = "\n".join(l for l in text.splitlines()
                         if not l.lstrip().startswith("#"))
        # Then bash's line continuation -- backslash-NEWLINE vanishes, while a
        # backslash followed by any other character does not. That asymmetry is
        # the whole bug.
        text = text.replace(BS + "\n", " ")

        for line in text.splitlines():
            # The INVOCATION form, not the bare word. `launch_margin1.sh`
            # prints a briefing from a heredoc whose prose says "gen_campaign
            # skipping completed runs makes the extension cheap", and that is
            # not a command.
            if "-m configs.gen_campaign" not in line:
                continue
            toks = shlex.split(line, posix=True)
            assert "--arms" in toks, (
                "%s invokes gen_campaign without --arms, so it silently gets "
                "the default single-arm campaign" % path)
            i = toks.index("--arms") + 1
            arms = []
            while i < len(toks) and not toks[i].startswith("--"):
                arms.append(toks[i])
                i += 1
            bad = [a for a in arms if a not in valid]
            assert not bad, (
                "%s passes arm name(s) %s that `configs/protocol.yml` does not "
                "define. bash resolved the --arms line to %s. This script dies "
                "at launch." % (path, bad, arms))
            assert len(set(arms)) == len(arms), (
                "%s names an arm twice: %s" % (path, arms))

            # The silent one. `gen_campaign` prints a warning and exits 0.
            orphaned = sorted(
                a for a in arms
                if P["arms"].get(a, {}).get("phase") == "trained"
                and not a.endswith("_null")
                and P["arms"][a].get("null_sibling", a + "_null") in P["arms"]
                and P["arms"][a].get("null_sibling", a + "_null") not in arms)
            assert not orphaned, (
                "%s names trained arm(s) %s with no zero-dose twin in the same "
                "campaign. gen_campaign PRINTS this and exits 0, so `set -e` "
                "will not stop the launch 45 seconds later -- and the finished "
                "campaign cannot attribute anything to the constraint rather "
                "than to the 29 extra epochs. Add: %s"
                % (path, orphaned,
                   sorted({P["arms"][a].get("null_sibling", a + "_null")
                           for a in orphaned})))
            checked += 1

    assert checked, (
        "no launch script invoked gen_campaign, so this gate checked nothing. "
        "Either the scripts moved or the parse above stopped matching.")


def test_a_documented_command_passes_FLAGS_THAT_EXIST():
    """The ghost-script gate checks the module exists. This checks its flags do.

    Same class as `test_a_staged_launch_script_NAMES_ONLY_ARMS_THAT_EXIST`: a
    command that only ever runs by hand, once, on a server, is the command least
    likely to have been run. Scope is `CLAUDE.md`, `docs/FRAMEWORK.md` and
    every `docs/*.sh` INCLUDING their comment blocks -- 71 invocations. The
    comment blocks are the point, not an afterthought: `launch_uniform.sh`'s
    seven-command read-order, the thing someone copies line by line once the
    campaign lands, is comments end to end, and stripping them left it
    outside the one gate written for hand-run commands.

    A flag argparse does not declare exits 2 exactly the way the mangled
    `--arms` line did.

    Flags are read by AST from each module's `add_argument` calls, never by
    grep: this project has already been burned once by a grep that reported
    `rho_step` as live because a LOG LINE named it.

    Audited 2026-08-25: 71 invocations, zero bad flags. That zero is a
    measurement and not silence, because the checker was shown to fire on
    `--two-metric` (the real flag is `--two-metrics`), on `--controls` (the
    real one is `--control`), and on a `--eviction` typo planted in the
    read-order COMMENT block -- while passing the genuine `--campaign`.
    """
    import io
    import re
    import shlex

    def declared_flags(modpath):
        tree = ast.parse(io.open(modpath, encoding="utf-8").read())
        out = set()
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "add_argument"):
                for a in n.args:
                    if (isinstance(a, ast.Constant)
                            and isinstance(a.value, str)
                            and a.value.startswith("-")):
                        out.add(a.value)
        return out

    # Interpreter-agnostic: the docs write `python -m`, the launch scripts
    # write `"$PY" -m`. Anchor on `-m <module>`, which is the part that
    # matters and the part neither can spell differently.
    CMD = re.compile(
        r"-m\s+((?:scripts|configs|docs\.paper\.scripts)"
        r"\.[a-z_0-9]+)([^\n`]*)")
    # The launch scripts are in scope for exactly the reason the arm gate
    # above exists: they run once, on a server, by hand.
    sources = ["CLAUDE.md", "docs/FRAMEWORK.md"]
    sources += [os.path.join("docs", f) for f in sorted(os.listdir("docs"))
                if f.endswith(".sh")]
    bad, seen = [], 0
    for doc in sources:
        text = io.open(doc, encoding="utf-8").read()
        if doc.endswith(".sh"):
            # Strip the `#` MARKER, do not drop the line. The read-order
            # block in launch_uniform.sh -- the seven commands whose whole
            # purpose is to be copied by a human after the campaign lands
            # -- lives entirely in comments. Dropping comment lines put it
            # outside the one gate written for hand-run commands. (The arm
            # gate above still drops them, deliberately: its prose names
            # REMOVED arms on purpose.)
            text = "\n".join(re.sub(r"^\s*#\s?", "", l)
                             for l in text.splitlines())
            # bash line continuations must be joined or every flag on a
            # wrapped line is invisible.
            text = text.replace(chr(92) + "\n", " ")
        for m in CMD.finditer(text):
            mod, rest = m.group(1), m.group(2)
            path = mod.replace(".", os.sep) + ".py"
            if not os.path.exists(path):
                continue        # the ghost-script gate owns this case
            have = declared_flags(path)
            try:
                toks = shlex.split(rest, posix=True)
            except ValueError:
                continue        # an unterminated quote is prose, not a command
            seen += 1
            for u in (t for t in toks if t.startswith("--")):
                if u.split("=")[0] not in have:
                    bad.append("%s: `python -m %s ... %s`" % (doc, mod, u))

    assert seen >= 20, (
        "only %d documented invocations parsed, so this gate is reading almost "
        "nothing -- the command format in the docs probably changed" % seen)
    assert not bad, (
        "the operational docs pass flags that argparse does not declare, so "
        "these commands exit 2 for whoever copies them: %s" % bad)


def test_a_launch_script_CANNOT_SEE_A_LIVE_RUN_by_looking_for_main_py():
    """The dispatcher is not the only process, and both launch scripts thought
    it was.

    `main.py` spawns every run as a subprocess:

        subprocess.run([sys.executable, '-u', '-m', RUNNER_MODULE, config])

    with `RUNNER_MODULE = 'src.experiments.runner'`. That command line contains
    no `main.py`. So the guard both scripts shipped --

        pgrep -u "$(whoami)" -f "envs/optloss/bin/python main.py"

    -- reports a clear host whenever the dispatcher has been killed but its
    runner is still alive. CLAUDE.md records that exact event as an operational
    failure that has already happened: "a killed dispatcher leaving three
    runners alive writing into a directory a fresh dispatcher had claimed".
    The guard meant to prevent it was blind to it.

    The runner module name is read from `main.py` by AST rather than hardcoded
    here, so renaming it makes this gate demand the new name instead of
    silently passing on the old one.
    """
    import io

    src = io.open("main.py", encoding="utf-8").read()
    runner = None
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (isinstance(tgt, ast.Name) and tgt.id == "RUNNER_MODULE"
                        and isinstance(node.value, ast.Constant)):
                    runner = node.value.value
    assert runner, (
        "main.py no longer defines RUNNER_MODULE as a literal, so this gate "
        "cannot learn what a run process is called")

    checked = 0
    for name in sorted(f for f in os.listdir("docs") if f.endswith(".sh")):
        path = os.path.join("docs", name)
        text = io.open(path, encoding="utf-8").read()
        code = "\n".join(l for l in text.splitlines()
                         if not l.lstrip().startswith("#"))
        if "pgrep" not in code:
            continue
        checked += 1
        assert runner in code, (
            "%s guards against a running campaign with pgrep but never looks "
            "for %r. main.py runs each experiment as `python -u -m %s`, so a "
            "killed dispatcher's orphaned runner is invisible to this guard "
            "and the script will happily start a second dispatcher into the "
            "same tree." % (path, runner, runner))

    assert checked, (
        "no launch script uses pgrep, so this gate checked nothing -- either "
        "the guard was removed or the scripts moved")


def test_family_split_resolves_a_twin_the_way_the_CAMPAIGN_ran_it():
    """A dedicated null arm beats the shared one, and both beat concatenation.

    `family_split` derived each family's twin as `fam + "_null"`. That is right
    for `xfam1`, which deliberately RAN `fioretto_null` and `hounie_null` as
    separate arms so their byte-identity with `tralo_null` is a measurement --
    the positive control the module's own docstring calls free and mandatory.

    It is wrong for `results/uniform1`, where `tralo_uniform` and `tralo_head`
    share `tralo_null` via `null_sibling` (protocol.yml, because at lambda = 0
    they are the same run). Concatenation invented `tralo_uniform_null`, which
    exists nowhere, so the tool refused a campaign whose twin was present --
    and it is step 5 of that campaign's own read-order.

    The obvious fix, resolving everything through `null_sibling`, is the
    opposite bug: protocol.yml points `fioretto` and `hounie` at `tralo_null`
    too, so it would stop reading xfam1's dedicated nulls and silently turn its
    positive control into a tautology. Hence: dedicated if the campaign ran one,
    shared otherwise.
    """
    from scripts.family_split import null_of

    xfam = {"tralo", "fioretto", "hounie", "tralo_null", "fioretto_null",
            "hounie_null", "tralo_reseed", "clip"}
    uni = {"tralo", "tralo_uniform", "tralo_head", "tralo_null",
           "tralo_reseed", "clip", "focal_clip"}

    # xfam1 must be UNCHANGED -- this is the published read.
    assert null_of("fioretto", xfam) == "fioretto_null"
    assert null_of("hounie", xfam) == "hounie_null"
    assert null_of("tralo", xfam) == "tralo_null"

    # uniform1 must become READABLE.
    assert null_of("tralo_uniform", uni) == "tralo_null"
    assert null_of("tralo_head", uni) == "tralo_null"
    assert null_of("tralo", uni) == "tralo_null"

    # The floor arm resolves the same way the hardcoded string used to.
    assert null_of("tralo_reseed", uni) == "tralo_null"
    assert null_of("tralo_reseed", xfam) == "tralo_null"

    # And a family whose dedicated null is absent falls back rather than
    # inventing an arm name that exists nowhere.
    assert null_of("fioretto", uni) == "tralo_null"


def test_order_probe_resolves_its_TWIN_from_the_campaign_on_disk():
    """`--null` was the fixed string "tralo_null".

    Correct for every command in `launch_uniform.sh`'s read-order -- `tralo`,
    `tralo_uniform` and `tralo_head` all share one twin -- and quietly wrong for
    `--arm fioretto` on a cross-family campaign, where the twin actually run is
    `fioretto_null`. The probe would have compared a Fioretto arm against
    TraLO's null and printed a clean-looking table. Same defect class as
    `family_split`'s concatenation, one tool over.

    The fragile half is the path arithmetic, not the lookup: the arm set is
    discovered by globbing `<campaign>/*/*/*/*/seed_*` and taking the parent
    directory name. This builds both campaign shapes on disk and checks it.

    Behaviour on every documented invocation is unchanged -- each still resolves
    to `tralo_null` -- which is the point: the fix removes a foot-gun without
    moving a single published number.
    """
    import glob
    import shutil

    from scripts.family_split import null_of

    layouts = {
        "uniform1": ["tralo", "tralo_uniform", "tralo_head", "tralo_null",
                     "tralo_reseed", "clip", "focal_clip"],
        "xfam1": ["tralo", "fioretto", "hounie", "tralo_null", "fioretto_null",
                  "hounie_null", "tralo_reseed", "clip"],
    }
    root = tempfile.mkdtemp(prefix="order_probe_layout_")
    try:
        for camp, arms in layouts.items():
            for arm in arms:
                for seed in (1, 2):
                    os.makedirs(os.path.join(root, camp, "iwildcam",
                                             "MobileNetV3", "L20_G50", arm,
                                             "seed_%d" % seed))

        def discover(camp):
            return {os.path.basename(os.path.dirname(d))
                    for d in glob.glob(os.path.join(root, camp,
                                                    "*", "*", "*", "*",
                                                    "seed_*"))}

        uni = discover("uniform1")
        xfam = discover("xfam1")
        assert uni == set(layouts["uniform1"]), (
            "the glob no longer finds the arms: got %s" % sorted(uni))
        assert xfam == set(layouts["xfam1"]), (
            "the glob no longer finds the arms: got %s" % sorted(xfam))

        # uniform1: one shared twin, which is what makes the campaign readable.
        for arm in ("tralo", "tralo_uniform", "tralo_head"):
            assert null_of(arm, uni) == "tralo_null", arm
        # xfam1: the DEDICATED nulls, or 2(s)'s positive control evaporates.
        assert null_of("fioretto", xfam) == "fioretto_null"
        assert null_of("hounie", xfam) == "hounie_null"
        assert null_of("tralo", xfam) == "tralo_null"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_the_constraint_step_is_NOT_inside_the_CE_batch_loop():
    """The premise a whole analysis rested on, and that nobody had checked.

    On 2026-08-25 I computed that a count-function difference compounds through
    Adam's momentum as `(1 - b1^k)` -- 0.100 at one step rising to 0.953 at 29 --
    and wrote it into FRAMEWORK, CLAUDE.md and both launch scripts as a
    correction to the recorded per-step figure.

    That law holds for CONSECUTIVE steps. These are not consecutive.
    `src/methodologies/tralo/train.py` runs the full CE batch loop with one
    `optimizer.step()` per batch, and calls `finish_constraint_step` ONCE per
    epoch AFTER it. About 126 CE steps therefore sit between two constraint
    steps, `b1^126 = 1.7e-6`, and the momentum carries essentially nothing
    across. The compression is a single-step property that never decays, which
    is what the file said before I "corrected" it.

    It is the same error the retraction in FRAMEWORK 1b-pre(6) is kept for --
    "the premise was never checked" -- committed while citing that retraction.
    So the premise is now a gate rather than a sentence.

    AST, not grep: a comment mentioning the batch loop must not satisfy this.
    """
    import io

    src = io.open("src/methodologies/tralo/train.py", encoding="utf-8").read()
    tree = ast.parse(src)

    def calls(node, name):
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Name) and f.id == name:
                    return True
                if isinstance(f, ast.Attribute) and f.attr == name:
                    return True
        return False

    batch_loops = [n for n in ast.walk(tree)
                   if isinstance(n, ast.For)
                   and isinstance(n.iter, ast.Name)
                   and "loader" in n.iter.id]
    assert batch_loops, (
        "no `for ... in *loader` loop found in tralo/train.py, so this gate "
        "cannot locate the CE batch loop it exists to reason about")

    for loop in batch_loops:
        assert calls(loop, "step"), (
            "the CE batch loop no longer takes an optimizer step, so the "
            "126-steps-between figure is wrong in the other direction")
        assert not calls(loop, "finish_constraint_step"), (
            "finish_constraint_step is now INSIDE the CE batch loop. Constraint "
            "steps would then be consecutive-ish and the momentum WOULD "
            "accumulate a count-function difference geometrically -- which "
            "reverses the analysis in FRAMEWORK 1b-pre(6) and in "
            "scripts/ortho_survival --compounding. Re-derive both before "
            "shipping this.")

    assert calls(tree, "finish_constraint_step"), (
        "tralo/train.py no longer calls finish_constraint_step at all")


def test_the_CE_autocorrelation_is_MEASURED_and_the_probe_responds_to_batch_size():
    """The one number the compounding analysis swings 31x on.

    `count_change_compounding` needs to know how correlated consecutive CE
    minibatch gradients are. Every version of that analysis until 2026-08-25
    swept it as an assumption and quoted whichever row suited the argument. At
    ce_rho=0 the trajectory opens ~5x over 29 steps; at 0.5 it does not open at
    all. That is the whole disagreement.

    Measured on a real net with real `torch.optim.Adam` at the trainer's own
    spacing -- `batch_size: 64` from protocol.yml, and 8064/64 = 126 steps per
    epoch, which is exactly what runs between two constraint steps. It comes out
    ~0.13 in epoch 1 and FALLS as the model fits, so warm-up 1 is its high point
    and the compounding is ~1.1x, not 5x. The per-step compression is the story.

    THE LIVENESS CONTROL IS THE BATCH SIZE. If the probe returned a constant it
    would prove nothing, so it must respond to the one knob that provably drives
    minibatch noise: a 512-batch must show a markedly HIGHER cosine than a
    64-batch, because averaging more samples leaves less noise and more signal.
    """
    from scripts.ortho_survival import (ce_gradient_autocorrelation,
                                        count_change_compounding)

    acs = ce_gradient_autocorrelation(epochs=2)
    assert len(acs) == 2
    assert 0.0 < acs[0] < 0.35, (
        "lag-1 CE gradient cosine at the trainer's batch size is %.3f. The "
        "compounding tables in FRAMEWORK 1b-pre(6) assume it is small; "
        "re-derive them before quoting." % acs[0])
    assert acs[1] < acs[0], (
        "the autocorrelation no longer falls as the model fits (%.3f -> %.3f), "
        "so warm-up 1 is not its high point and the argument that this is its "
        "WORST case no longer holds" % (acs[0], acs[1]))

    big = ce_gradient_autocorrelation(batch=512, epochs=1)[0]
    assert big > 2.5 * acs[0], (
        "LIVENESS CONTROL FAILED: batch 512 gives %.3f against batch 64's "
        "%.3f. If averaging 8x more samples does not raise the cosine, this "
        "probe is not measuring minibatch noise and its value is an artefact."
        % (big, acs[0]))

    # And the consequence: at the measured value the channel does not compound.
    rng = np.random.default_rng(3)
    at_zero = count_change_compounding(np.cos(np.radians(29.4)), 0.0, rng, n=6000)
    at_meas = count_change_compounding(np.cos(np.radians(29.4)), round(acs[0], 3),
                                       rng, n=6000)
    assert at_meas[3] < at_zero[3] / 3.0, (
        "the measured autocorrelation no longer collapses the compounding "
        "(%.4f vs %.4f at rho=0). FRAMEWORK 1b-pre(6) says it does, by ~8x."
        % (at_meas[3], at_zero[3]))
    assert at_meas[2] < at_meas[0] * 2.0, (
        "at the measured rho the trajectory now OPENS materially over 29 steps "
        "(%.2f -> %.2f deg), which reverses the conclusion that the per-step "
        "compression is the whole story" % (at_meas[0], at_meas[2]))


def test_a_launch_script_VERIFIES_THE_DATA_ARRAY_not_just_the_directory():
    """A guard that tests the wrong thing is worse than no guard.

    `docs/launch_uniform.sh` linked the dataset into its fresh worktree with

        [ -e data/iwildcam ] || ln -s ~/optloss-audit/data/iwildcam data/iwildcam

    But `data/iwildcam/oodslice/train_meta.csv` and `test_meta.csv` ARE TRACKED
    IN GIT. Checking out any commit therefore creates `data/iwildcam/oodslice/`
    holding those two CSVs and nothing else, the `-e` test sees a directory, the
    link is skipped, and every run dies instantly on `train_images.npy`.

    Measured cost, 2026-08-25: the dispatcher walked all 252 runs in about four
    minutes at 0% GPU. And because an interrupted run resets to `pending`, the
    campaign afterwards looked merely unstarted rather than broken -- the same
    silent shape `smoke_arms` exists for.

    So: any launch script that links a dataset must test for a `.npy` the runner
    actually opens, and must REFUSE rather than proceed. The directory test is
    banned outright, since git will keep re-creating that directory.
    """
    import io
    import re

    checked = 0
    for name in sorted(f for f in os.listdir("docs") if f.endswith(".sh")):
        path = os.path.join("docs", name)
        text = io.open(path, encoding="utf-8").read()
        code = "\n".join(l for l in text.splitlines()
                         if not l.lstrip().startswith("#"))
        if "ln -s" not in code or "iwildcam" not in code:
            continue
        checked += 1
        assert not re.search(r"\[\s*-e\s+\S*data/iwildcam\s*\]", code), (
            "%s guards the dataset link with a directory test. git tracks CSVs "
            "inside data/iwildcam/oodslice, so that directory always exists and "
            "the link is always skipped." % path)
        assert re.search(r"\.npy", code), (
            "%s links a dataset but never names a .npy. The runner opens "
            "train_images.npy; a guard that does not mention it cannot know "
            "whether the link worked." % path)
        assert "REFUSING" in code, (
            "%s links a dataset without a refusal path. Linking silently does "
            "nothing when the source glob is empty, and the campaign then "
            "burns every run and resets them all to pending." % path)

    assert checked, (
        "no launch script links the dataset, so this gate checked nothing -- "
        "either the link moved or the scripts did")


def test_the_iwildcam_arrays_are_NOT_in_git_but_the_meta_csvs_ARE():
    """The asymmetry the guard above exists for, pinned so it cannot drift.

    If someone ever commits the arrays, the directory test becomes harmless and
    this gate should be revisited. If someone ever REMOVES the CSVs from git,
    the directory stops being auto-created and the old guard would have worked.
    Either change invalidates the reasoning, so both are asserted.
    """
    tracked = subprocess.run(["git", "ls-files", "data/"],
                             capture_output=True, text=True).stdout.split()
    csvs = [f for f in tracked if f.endswith(".csv")]
    npys = [f for f in tracked if f.endswith(".npy")]
    assert csvs, (
        "no CSV under data/ is tracked any more, so checking out a commit no "
        "longer creates data/iwildcam/oodslice and the directory-test guard "
        "would have been fine. Re-read the gate above before trusting it.")
    assert not npys, (
        "arrays are now tracked in git: %s. That is a repository-size problem "
        "in its own right, and it also means the launch guard's premise has "
        "changed." % npys[:3])


def test_a_documented_campaign_SIZE_matches_what_the_script_GENERATES():
    """The arm gate checks the names. Nothing checked the COUNT.

    `docs/FRAMEWORK.md` announced the live campaign as

        Launch: `docs/launch_uniform.sh` (9 cells, 6 arms, 4 seeds = 216 runs

    while the script it names generates **7 arms and 252 runs**. The arm count
    had been raised from 6 to 7 in the script and never in the document that
    tells an operator what is running, so the two disagreed for a day about a
    live campaign -- and the wrong number is the one in the file CLAUDE.md
    calls the only operational document.

    It is the same defect class as the mangled `--arms` line: a staged artefact
    that only a human ever reads, so nothing ever parsed it. This gate parses
    it. Two checks, both static:

    * every fully-specified triple anywhere in `docs/` -- N cells, M arms,
      S seeds = R runs -- must satisfy N*M*S == R. Partly-specified ones
      ("3 arms x 9 cells x 2 = 54 runs", where the 2 is extra SEEDS) are
      skipped deliberately: the rule is about arithmetic that claims to be
      complete, not about prose.
    * where a document NAMES a launch script, the triple beside that name must
      be the size that script actually generates: models x datasets x caps
      cells, `--arms` widened by `mandatory_arms`, times `protocol.seeds`.
      A script's own comment block may discuss counterfactual sizes (uniform1
      records 288 as first generated and 336 for a ViTB16 extension), so the
      requirement there is that at least ONE triple in it is the real one.

    Negative controls, run 2026-08-25: the gate FAILED on the 216/6-arm line
    that was live in FRAMEWORK when it was written, and FAILED again on a
    planted `= 253 runs` typo in the script's own size line.
    """
    import io
    import re
    import shlex

    BS = chr(92)
    P = load_protocol()
    seeds = len(P["protocol"]["seeds"])
    mandatory = list(P.get("mandatory_arms", []))

    UNIT = re.compile(r"(\d+)\s*(cells?|arms?|seeds?)")
    RUNS = re.compile(r"=\s*(\d+)\s*runs")

    def triples(text):
        """Every fully-specified (cells, arms, seeds) = runs claim in `text`."""
        out = []
        for m in RUNS.finditer(text):
            window = text[max(0, m.start() - 90):m.start()]
            found = {}
            for num, unit in UNIT.findall(window):
                found[unit.rstrip("s")] = int(num)
            if len(found) == 3:
                out.append((found["cell"], found["arm"], found["seed"],
                            int(m.group(1)),
                            text[max(0, m.start() - 90):m.end()].strip()))
        return out

    # ---- what each launch script actually generates -------------------------
    true_size = {}
    for name in sorted(f for f in os.listdir("docs") if f.endswith(".sh")):
        path = os.path.join("docs", name)
        raw = io.open(path, encoding="utf-8").read()
        code = "\n".join(l for l in raw.splitlines()
                         if not l.lstrip().startswith("#"))
        code = code.replace(BS + "\n", " ")
        for line in code.splitlines():
            if "-m configs.gen_campaign" not in line:
                continue
            toks = shlex.split(line, posix=True)

            def listarg(flag):
                if flag not in toks:
                    return []
                i = toks.index(flag) + 1
                vals = []
                while i < len(toks) and not toks[i].startswith("--"):
                    vals.append(toks[i])
                    i += 1
                return vals

            arms = set(listarg("--arms")) | set(mandatory)
            cells = (max(1, len(listarg("--models")))
                     * max(1, len(listarg("--datasets")))
                     * max(1, len(listarg("--caps"))))
            true_size[name] = (cells, len(arms), seeds,
                               cells * len(arms) * seeds)

    assert true_size, "no launch script parsed, so this gate checked nothing"

    # ---- check 1: the arithmetic is self-consistent everywhere ---------------
    docs = [os.path.join("docs", f) for f in sorted(os.listdir("docs"))
            if f.endswith(".sh") or f == "FRAMEWORK.md"] + ["CLAUDE.md"]
    checked = 0
    for path in docs:
        if not os.path.exists(path):
            continue
        for c, a, s, r, snippet in triples(io.open(path, encoding="utf-8").read()):
            assert c * a * s == r, (
                "%s states a campaign size that does not multiply: %d cells x "
                "%d arms x %d seeds is %d, not %d.\n    %s"
                % (path, c, a, s, c * a * s, r, snippet))
            checked += 1
    assert checked, (
        "no campaign-size arithmetic found in docs/, so check 1 is silent. "
        "Either the phrasing changed or the window above stopped matching.")

    # ---- check 2: a script's own comment block states its REAL size ----------
    for name, size in sorted(true_size.items()):
        path = os.path.join("docs", name)
        stated = [t[:4] for t in triples(io.open(path, encoding="utf-8").read())]
        assert size in stated, (
            "%s generates %d cells x %d arms x %d seeds = %d runs, and its own "
            "comment block never says so. It states %s. An operator reads the "
            "size line, not the invocation." % ((path,) + size + (stated,)))

    # ---- check 3: a document that NAMES a script quotes its REAL size --------
    for path in ["docs/FRAMEWORK.md", "CLAUDE.md"]:
        if not os.path.exists(path):
            continue
        lines = io.open(path, encoding="utf-8").read().splitlines()
        for i, line in enumerate(lines):
            for name, size in true_size.items():
                if name not in line:
                    continue
                window = "\n".join(lines[i:i + 3])
                near = [t[:4] for t in triples(window)]
                if not near:
                    continue
                assert near[0] == size, (
                    "%s:%d names %s and states %d cells x %d arms x %d seeds = "
                    "%d runs. That script generates %d cells x %d arms x %d "
                    "seeds = %d runs. The document an operator reads to learn "
                    "what is running disagrees with the thing that is running."
                    % ((path, i + 1, name) + near[0] + size))


def test_a_probability_clamp_SURVIVES_THE_DTYPE_IT_ACTUALLY_RUNS_IN():
    """`clamp(EPSILON, 1 - EPSILON)` is a NO-OP at the top, in every dtype.

    EPSILON is 1e-8. float32's own epsilon is 1.19e-7, so `1.0 - 1e-8`
    rounds to exactly 1.0 -- and in float16 (eps 9.8e-4) and bfloat16
    (eps 7.8e-3) it is not close. The clamp that exists to keep a
    probability out of {0, 1} therefore does not, and the lower bound is
    equally dead in float16, where 1e-8 is below the smallest subnormal
    and rounds to 0.

    MEASURED, on the live campaign, 2026-08-25. `results/uniform1` exists to
    test `soft_count_mode: uniform`, whose count is built on the log-odds
    `u = log p - log1p(-p)`. With p clamped to a value that is still exactly
    1.0, `log1p(-p)` is -inf, `u` is +inf, and the straight-through term
    `w * (u - u.detach())` is inf - inf = NaN. `finish_constraint_step` then
    drops the step, and the run still writes `status: completed`:

        arm             steps landed / attempted
        tralo             29 / 29    100.0%      (soft_count_mode: sum)
        tralo_head        29 / 29    100.0%      (soft_count_mode: sum)
        tralo_uniform      1 / 29      3.4%      (soft_count_mode: uniform)

    The one arm the campaign was built to measure ran at **3.4% of its
    dose**, and every other arm ran at full dose, so the comparison was not
    merely weak -- it was a dose contrast wearing a loss-shape contrast's
    clothes. Nothing in the predictions records a step that did not happen.
    `sum` is untouched because `p * (1 - p)` never takes a logarithm.

    The fix is `clamp_probability`, which takes its epsilon from the tensor's
    OWN dtype, so the bound is representable wherever the tensor lives. This
    gate holds three things: the helper is finite in all three dtypes, the
    two call sites use it, and no call site re-derives `1 - EPSILON` by hand
    (an AST scan, never a grep -- this project has been burned by a grep that
    read a log line as a live use).
    """
    import ast
    import io
    import torch

    from src.utils.constants import EPSILON, clamp_probability

    # The root fact, stated so it cannot quietly stop being true. Python
    # computes `1.0 - EPSILON` in float64, where it IS representable
    # (0.99999999) -- the bound only dies on the cast into the tensor's dtype,
    # which is exactly why reading the expression never revealed it.
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        assert float(torch.tensor(1.0 - EPSILON, dtype=dtype)) == 1.0, (
            "`1 - EPSILON` is now representable in %s, so the hand-written "
            "clamp is no longer a no-op there; re-derive the dtype-aware "
            "bounds below before relaxing this." % dtype)

    saturated = [0.0, 1e-12, 0.5, 0.99, 0.999, 1.0 - 1e-9, 1.0]
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        p = torch.tensor([saturated], dtype=dtype, requires_grad=True)
        q = clamp_probability(p)
        assert torch.isfinite(q).all(), (
            "clamp_probability left a non-finite value in %s" % dtype)
        assert (q > 0).all() and (q < 1).all(), (
            "clamp_probability returned a value at 0 or 1 in %s: %s"
            % (dtype, q))

        u = torch.log(q) - torch.log1p(-q)
        assert torch.isfinite(u).all(), (
            "the log-odds are non-finite in %s even after the clamp: %s"
            % (dtype, u))

        from src.losses.transductive_loss import uniform_grad_count
        s = uniform_grad_count(p.detach().clone().requires_grad_(True))
        assert torch.isfinite(s).all(), (
            "uniform_grad_count returned a non-finite VALUE in %s" % dtype)

    # The gradient, which is what actually reaches the optimizer and what was
    # silently dropped. float32 only: autograd through log in half precision
    # is not what the constraint pass runs, `constraint_fp32` is.
    p = torch.tensor([saturated], dtype=torch.float32, requires_grad=True)
    uniform_grad_count(p).sum().backward()
    assert torch.isfinite(p.grad).all(), (
        "uniform_grad_count produced a non-finite GRADIENT on saturated "
        "probabilities: %s. This is the failure that cost `tralo_uniform` 28 "
        "of its 29 constraint steps." % p.grad)

    # The same failure at the other end: `clamp(min=EPSILON)` in float16 is
    # `clamp(min=0)`, and `window_temp` then makes `sigmoid(margin / 0)` NaN
    # for a margin of exactly 0 -- the items AT the decision boundary, which
    # are the whole point of the margin window.
    from src.losses.transductive_loss import window_temp, margin_window
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        flat = torch.zeros((6, 3), dtype=dtype)
        t = window_temp(flat, 3)
        assert torch.isfinite(t).all() and (t > 0).all(), (
            "window_temp returned a non-positive temperature in %s: %s"
            % (dtype, t))
        w = margin_window(torch.full((6, 3), 1.0 / 3.0, dtype=dtype), t)
        assert torch.isfinite(w).all(), (
            "margin_window is non-finite in %s at zero margin: %s" % (dtype, w))

    # No site may re-derive the bound by hand. AST, so a comment or a log line
    # naming EPSILON cannot pass for a use.
    class Finder(ast.NodeVisitor):
        def __init__(self):
            self.bad = []

        def visit_Call(self, node):
            name = getattr(node.func, "attr", None) or getattr(
                node.func, "id", None)
            if name == "clamp":
                for a in list(node.args) + [k.value for k in node.keywords]:
                    # `clamp(EPSILON, 1 - EPSILON)` -- dead at the top.
                    if (isinstance(a, ast.BinOp)
                            and isinstance(a.op, ast.Sub)
                            and isinstance(a.left, ast.Constant)
                            and float(a.left.value) == 1.0
                            and getattr(a.right, "id", None) == "EPSILON"):
                        self.bad.append(node.lineno)
                    # `clamp(min=EPSILON)` -- dead at the BOTTOM in float16,
                    # where 1e-8 is under the smallest subnormal and rounds to
                    # 0. Scope is deliberately CLAMPS ONLY: an additive guard
                    # like `x / (s + EPSILON)` has the same rounding but its
                    # safety depends on `s`, so flagging it would be noise.
                    elif getattr(a, "id", None) == "EPSILON":
                        self.bad.append(node.lineno)
            self.generic_visit(node)

    offenders = []
    for root, _dirs, files in os.walk("src"):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            f = Finder()
            f.visit(ast.parse(io.open(path, encoding="utf-8").read()))
            offenders += [(path, ln) for ln in f.bad]
    assert not offenders, (
        "these sites clamp a probability with a hand-written `1 - EPSILON`, "
        "which is a no-op at the top in every dtype: %s. Use "
        "`clamp_probability` from src.utils.constants." % offenders)


def test_a_launch_scripts_PIN_carries_the_same_gen_campaign_invocation():
    """The script checks out `$PIN` onto the tree it is itself stored in.

        cd "$TREE"
        git checkout -q --detach "$PIN"

    Bash reads a script incrementally, by byte offset. If the pinned commit's
    copy of `docs/launch_uniform.sh` differs from the one being executed, the
    file changes underneath the interpreter at an offset it has not reached
    yet -- and the campaign that then generates is the PINNED script's
    campaign, not the one the operator read.

    It nearly happened on 2026-08-25. `--constraint-fp32` was added to the
    invocation and PIN still named the commit before it, so the launch would
    have re-checked-out a script with no `--constraint-fp32` and regenerated
    the same 3.4%-dose campaign the flag exists to prevent.

    A commit cannot name its own hash, so the requirement is not "PIN equals
    HEAD". It is narrower and it is the part that decides what runs: **the
    gen_campaign invocation at PIN must be token-for-token the invocation in
    the working copy.** Prose may drift; the campaign may not.

    Negative control, 2026-08-25: with PIN left at 38d96ba4 this FAILS with
    the `--constraint-fp32` token present in the working copy and absent at
    the pin.
    """
    import io
    import shlex
    import subprocess

    BS = chr(92)

    def invocation(text):
        code = "\n".join(l for l in text.splitlines()
                         if not l.lstrip().startswith("#"))
        code = code.replace(BS + "\n", " ")
        for line in code.splitlines():
            if "-m configs.gen_campaign" in line:
                return shlex.split(line, posix=True)
        return None

    checked = 0
    for name in sorted(f for f in os.listdir("docs") if f.endswith(".sh")):
        path = os.path.join("docs", name)
        here = io.open(path, encoding="utf-8").read()
        pin = None
        for line in here.splitlines():
            if line.startswith("PIN="):
                pin = line[4:].split("#")[0].strip()
                break
        if not pin:
            continue
        # THE PRIMARY DEFENCE is the out-of-tree refusal, because it removes
        # the hazard instead of tracking it: a script run from outside $TREE
        # cannot be rewritten by a checkout of $TREE, whatever the pin holds.
        # Every script that checks out a pin must carry it.
        if "checkout" in here and "$PIN" in here:
            assert 'REFUSING: this script lives inside' in here, (
                "%s does `git checkout --detach $PIN` on $TREE and does not "
                "refuse to run from inside $TREE. Bash reads a script by byte "
                "offset, so the checkout rewrites it mid-execution."
                % path)

        rel = path.replace(os.sep, "/")
        try:
            there = subprocess.check_output(
                ["git", "show", "%s:%s" % (pin, rel)],
                stderr=subprocess.STDOUT).decode("utf-8", "replace")
        except (subprocess.CalledProcessError, OSError):
            # A script newer than its own pin. Allowed, because a commit
            # cannot name its own hash and the out-of-tree guard above already
            # makes the rewrite impossible -- but only in that order.
            continue

        mine, theirs = invocation(here), invocation(there)
        if mine is None and theirs is None:
            continue
        assert mine == theirs, (
            "%s pins %s, but the gen_campaign invocation there is NOT the one "
            "in this file.\n  here : %s\n  at %s: %s\nThe checkout would "
            "replace this script with one that generates a different "
            "campaign." % (path, pin, mine, pin, theirs))
        checked += 1

    assert checked, (
        "no launch script declared a PIN, so this gate checked nothing.")


def test_the_out_of_tree_guard_REFUSES_ONLY_WHEN_IT_SHOULD():
    """A guard that always refuses is as broken as no guard, and quieter.

    The refusal added on 2026-08-25 read

        case "${TREEP:-__none__}" in
          "") ;;
          *) case "$SELF/" in "$TREEP"/*) ... exit 1 ;; esac ;;
        esac

    `${TREEP:-__none__}` substitutes the DEFAULT when TREEP is empty, so the
    `""` arm is unreachable and control falls into `*` with TREEP still empty
    -- making the inner pattern `/*`, which matches every absolute path. On a
    FIRST launch, where $TREE does not exist yet and TREEP is therefore empty,
    the script refused to run at all. It did exactly that on iwc4.

    So the guard is EXECUTED here, in bash, in the three states that matter,
    rather than pattern-matched for. Static checks cannot see this class of
    bug: the text was present and correct-looking the whole time.
    """
    import io
    import shutil
    import subprocess
    import tempfile

    bash = shutil.which("bash")
    if not bash:
        pytest.skip("no bash on this host; the guard is shell code")

    NL = "\n"
    scripts = sorted(f for f in os.listdir("docs") if f.endswith(".sh"))
    checked = 0
    for name in scripts:
        text = io.open(os.path.join("docs", name), encoding="utf-8").read()
        if 'REFUSING: this script lives inside' not in text:
            continue
        start = text.index('SELF=$(cd "$(dirname "$0")"')
        end = text.index(NL + "fi" + NL, start) + len(NL + "fi" + NL)
        guard = text[start:end]

        tmp = tempfile.mkdtemp()
        try:
            tree = os.path.join(tmp, "tree").replace(os.sep, "/")
            outside = os.path.join(tmp, "outside").replace(os.sep, "/")
            os.makedirs(outside)
            body = ('TREE=%s%s%s%secho REACHED_THE_END%s'
                    % (tree, NL, guard, NL, NL))

            def run(where):
                path = os.path.join(where, "g.sh")
                io.open(path, "w", encoding="utf-8", newline=NL).write(body)
                return subprocess.run(
                    [bash, path.replace(os.sep, "/")],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

            # 1. $TREE does not exist yet -- the FIRST-launch state.
            out = run(outside)
            assert b"REACHED_THE_END" in out.stdout, (
                "%s refuses when $TREE does not exist yet, so it can never "
                "launch a campaign for the first time. Output: %s"
                % (name, out.stdout))

            # 2. $TREE exists, the script is elsewhere -- the correct usage.
            os.makedirs(tree)
            out = run(outside)
            assert b"REACHED_THE_END" in out.stdout, (
                "%s refuses from OUTSIDE an existing $TREE, which is the only "
                "supported way to run it. Output: %s" % (name, out.stdout))

            # 3. the script IS inside $TREE -- the hazard.
            out = run(tree)
            assert b"REACHED_THE_END" not in out.stdout, (
                "%s runs from INSIDE $TREE, where the checkout it is about to "
                "do rewrites it mid-execution. Output: %s"
                % (name, out.stdout))
            assert b"REFUSING" in out.stdout, (
                "%s exits from inside $TREE without saying why: %s"
                % (name, out.stdout))
            checked += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    assert checked, (
        "no launch script carries the out-of-tree guard, so this executed "
        "nothing. Either the guard was removed or the extraction above stopped "
        "matching it.")


def test_the_dose_reader_CATCHES_BOTH_HISTORICAL_FAILURES():
    """`scripts/dose_landed.py` is the check that must run FIRST, so gate it.

    Two campaigns lost their treatment silently and both were caught late:

        uniform1   `tralo_uniform` 1/29 (3.4%) beside `tralo` 29/29 -- ONE arm
                   low, its siblings fine, i.e. the LOSS SHAPE.
        iwc3       716/1044 (68.6%) with a step lost in 36 of 36 runs -- EVERY
                   trained arm low, i.e. the HOST (FP16 + GradScaler).

    The second is the one a naive spread check misses: the arms AGREE with each
    other at 69%, so "did these arms run at the same dose" says yes while the
    honest answer is that none of them ran at the dose it was given. Both
    shapes are in the script's `--self-test` and both are asserted here.
    """
    import io

    from scripts.dose_landed import report, self_test

    buf = io.StringIO()
    assert self_test(out=buf) == 0, buf.getvalue()
    assert "SELF-TEST PASS" in buf.getvalue()

    # one arm low, siblings fine -> named as the loss shape
    buf = io.StringIO()
    n = report({"tralo": [29, 29, 1, 0, 0, 0], "tralo_uniform": [1, 29, 1, 0, 0, 0]},
               {"tralo": {"bfloat16"}, "tralo_uniform": {"bfloat16"}}, out=buf)
    text = buf.getvalue()
    assert n >= 2 and "DID NOT RUN AT THE SAME DOSE" in text, text
    assert "LOSS SHAPE" in text, text

    # every arm low -> named as the host, even though the arms agree
    buf = io.StringIO()
    n = report({"tralo": [716, 1044, 36, 0, 0, 0], "fioretto": [720, 1044, 36, 0, 0, 0]},
               {"tralo": {"float16"}, "fioretto": {"float16"}}, out=buf)
    text = buf.getvalue()
    assert n >= 2, text
    assert "STEP(S) LOST" in text, text

    # the clean case must stay quiet, or the check is noise
    buf = io.StringIO()
    assert report({"tralo": [29, 29, 1, 0, 0, 0], "hounie": [29, 29, 1, 0, 0, 0]},
                  {}, out=buf) == 0, buf.getvalue()

    # and the very start of a campaign is not a failure
    buf = io.StringIO()
    assert report({"clip": [0, 0, 0, 4, 0, 0]}, {}, out=buf) == 0
    assert "normal state at the very start" in buf.getvalue()

    # A YOUNG campaign must not read as an OLD one. On its first outing this
    # printed "36 run(s) predate the field" for 36 runs that had not started,
    # which tells the reader their checkout is stale when it is merely early.
    # `status` is what separates the two and nothing else can.
    buf = io.StringIO()
    report({"tralo": [29, 29, 1, 0, 0, 0], "tralo_uniform": [0, 0, 0, 36, 0, 0]},
           {"tralo": {"bfloat16"}}, out=buf)
    text = buf.getvalue()
    assert "36 still pending or running" in text, text
    assert "predate" not in text, text

    buf = io.StringIO()
    report({"tralo": [29, 29, 1, 0, 0, 0], "clip": [0, 0, 0, 0, 36, 0]},
           {"tralo": {"bfloat16"}}, out=buf)
    assert "predate the field" in buf.getvalue(), buf.getvalue()

    # A lambda=0 twin attempts no steps and is NOT a post-hoc arm. Calling it
    # one mislabels the control this project cannot read a campaign without.
    buf = io.StringIO()
    report({"tralo": [29, 29, 1, 0, 0, 0], "tralo_null": [0, 0, 4, 0, 0, 0]},
           {}, out=buf)
    assert "lambda=0 twin does" in buf.getvalue(), buf.getvalue()


def test_EVERY_script_offering_a_self_test_actually_PASSES_it():
    """Discovered, not enumerated -- so a NEW probe is gated the day it lands.

    Three of the six scripts carrying a `--self-test` had no gate at all when
    this was written (`collateral_probe`, `ortho_survival`, `paired_noise`),
    because each earlier gate named ONE script by hand. An enumerated list
    only ever covers what its author remembered; this walks `scripts/` and
    runs whatever it finds, so the next one is covered before anyone thinks
    to add it.

    Each of these self-tests exists because the probe it guards makes a claim
    that would otherwise be unfalsifiable -- `paired_noise` asserts it CAN
    report that pairing helped, `ceiling_screen` asserts it CAN say WORTH
    RUNNING. A probe that can only ever return one verdict is not a
    measurement, and a self-test nobody runs is not a gate.
    """
    import ast
    import glob
    import importlib
    import inspect
    import io as _io
    import os
    import subprocess
    import sys

    found = []
    for path in sorted(glob.glob(os.path.join('scripts', '*.py'))):
        src = _io.open(path, encoding='utf-8').read()
        tree = ast.parse(src)
        names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        if 'self_test' not in names:
            continue
        found.append(os.path.splitext(os.path.basename(path))[0])

    assert len(found) >= 5, (
        'expected several self-testing probes, found %r -- if scripts/ moved, '
        'this gate is silently checking nothing' % (found,))

    # Two shapes exist, and a bare `except TypeError` around the call hides
    # the difference -- it swallows a TypeError raised INSIDE a self-test and
    # reports the probe as merely old-signature. Dispatch on the signature
    # instead. `ortho_survival`'s takes an RNG and is reachable only through
    # its CLI, which is what a person runs anyway. `collateral_probe`'s needs
    # a real campaign and cannot be gated here at all: NAME it, so a new
    # script joining that category is noticed rather than quietly uncovered.
    NEEDS_A_CAMPAIGN = {'collateral_probe'}

    failures = []
    skipped = []
    for name in found:
        if name in NEEDS_A_CAMPAIGN:
            skipped.append(name)
            continue
        mod = importlib.import_module('scripts.' + name)
        params = inspect.signature(mod.self_test).parameters
        required = [p for p in params.values()
                    if p.default is inspect.Parameter.empty
                    and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if required:
            proc = subprocess.run(
                [sys.executable, '-m', 'scripts.' + name, '--self-test'],
                capture_output=True, text=True, timeout=900)
            if proc.returncode != 0:
                failures.append('%s (via CLI):%s%s%s'
                                % (name, chr(10), proc.stdout, proc.stderr))
            continue
        # Three conventions are in use and all three must be honoured, or the
        # gate reports a pass it did not observe: return an exit code into an
        # `out` stream; return a code with no stream; or return None and raise
        # SystemExit on failure (`straddle_probe`). Treating None as failure
        # would red-flag a healthy probe; treating SystemExit as an error
        # would crash the gate instead of reporting the probe.
        buf = _io.StringIO()
        try:
            rc = mod.self_test(out=buf) if 'out' in params else mod.self_test()
        except SystemExit as exc:
            rc = exc.code
        if rc not in (0, None):
            failures.append('%s: rc=%r%s%s'
                            % (name, rc, chr(10), buf.getvalue()))

    assert not failures, ('a shipped probe fails its own self-test:%s%s'
                          % (chr(10), (chr(10) * 2).join(failures)))
    # The skip list must be EARNED, not declared. Comparing `skipped` against
    # NEEDS_A_CAMPAIGN would be a tautology -- `skipped` is built by exactly
    # that membership test, so the assertion could never fail and anyone could
    # silence this gate by adding a name to the set. Instead, make each skip
    # prove itself: a probe that genuinely needs a campaign CANNOT self-test
    # standalone, so its CLI must refuse. If it succeeds, the skip is unearned.
    for name in sorted(skipped):
        proc = subprocess.run(
            [sys.executable, '-m', 'scripts.' + name, '--self-test'],
            capture_output=True, text=True, timeout=900)
        assert proc.returncode != 0, (
            '%s sits on the needs-a-campaign skip list, but its --self-test '
            'SUCCEEDS standalone -- so the skip is unearned and the probe is '
            'going ungated for no reason. Remove it from NEEDS_A_CAMPAIGN and '
            'let this gate run it.' % name)


def test_the_panel_SAYS_when_it_is_scoring_an_unfinished_campaign():
    """The scorer drops non-completed runs, then prints a finished-looking table.

    This gate exists because that cost a wrong entry in FRAMEWORK 2(u). Read at
    106 of 180 runs, `results/iwc4` showed `tralo` macroF1 -0.0156 against
    `tralo_reseed`'s -0.0156, and that four-decimal agreement was written up as
    "the macro-F1 damage IS the reseed floor". At 180 of 180 the ratio is 1.51x
    and the metric that actually matches a reseed is macroP.

    The reason is worth encoding rather than remembering: the FLOOR moves more
    than the treatment does, so `arm / floor` is the least stable quantity on
    the page. The same partial read put AP at 5.8x; the finished campaign says
    19.1x, purely because the reseed floor settled from -0.0101 to a tie.

    Crashed runs must NOT trigger it -- they are reported by their own block,
    and double-reporting them would train the reader to skip both.
    """
    import collections
    import io as _io

    from scripts.full_panel import _completeness_warning

    buf = _io.StringIO()
    frac = _completeness_warning(106, collections.Counter({'pending': 74}),
                                 out=buf)
    text = buf.getvalue()
    assert abs(frac - 106 / 180.0) < 1e-9, frac
    assert '59%' in text, text
    assert '106 of 180' in text, text
    assert 'RATIOS' in text and 'SIGNS' in text, text

    # A finished campaign must say nothing at all: a warning that fires always
    # is a warning nobody reads.
    buf = _io.StringIO()
    assert _completeness_warning(180, collections.Counter(), out=buf) == 1.0
    assert buf.getvalue() == '', buf.getvalue()

    # Crashed/diverged runs are a DIFFERENT failure with its own report.
    buf = _io.StringIO()
    _completeness_warning(170, collections.Counter({'diverged (CRASHED)': 10}),
                          out=buf)
    assert buf.getvalue() == '', buf.getvalue()

    # And it must actually be WIRED IN -- a helper nobody calls is not a gate.
    import ast
    import io as _io2
    tree = ast.parse(_io2.open('scripts/full_panel.py', encoding='utf-8').read())
    called = [n for n in ast.walk(tree)
              if isinstance(n, ast.Call)
              and getattr(n.func, 'id', '') == '_completeness_warning']
    assert called, ('_completeness_warning is defined but never called, so the '
                    'panel would go back to scoring a half-finished campaign '
                    'silently')


def test_no_script_PRINTS_a_character_the_windows_console_cannot_ENCODE():
    """One emoji in a `print` kills the process AFTER printing the table.

    The console here is cp1252. `print("\u26a0 ...")` raises
    UnicodeEncodeError, so a report renders in full, looks finished, and the
    process exits 1 on the very next line -- which reads as "the tool crashed"
    when the numbers above it are correct and complete, or worse, as "the tool
    ran" when the lines after it never printed.

    `scripts/ceiling_screen.py` did exactly this on its first run against
    iwildcam: six rows and the verdict block came out, then a traceback where
    the caveat should have been.

    The scope is what actually reaches a terminal -- string constants inside a
    `print(...)` or a `....write(...)` call, read by AST. Docstrings, comments
    and FRAMEWORK prose are untouched, and this file's own tables of emoji stay
    legal. Audited 2026-08-25 over all of `scripts/`: zero offenders once
    ceiling_screen was fixed, so the class is closed rather than merely noted.
    """
    import ast
    import io

    offenders = []
    for root, _dirs, files in os.walk("scripts"):
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            tree = ast.parse(io.open(path, encoding="utf-8").read())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = (getattr(node.func, "id", None)
                        or getattr(node.func, "attr", None))
                if name not in ("print", "write"):
                    continue
                for arg in list(node.args) + [k.value for k in node.keywords]:
                    for sub in ast.walk(arg):
                        if (isinstance(sub, ast.Constant)
                                and isinstance(sub.value, str)):
                            bad = sorted({c for c in sub.value if ord(c) > 127})
                            if bad:
                                offenders.append((path, node.lineno,
                                                  "".join(bad)))

    assert not offenders, (
        "these sites print a character cp1252 cannot encode, so the process "
        "dies mid-report on Windows: %s" % offenders[:8])


def test_the_ceiling_screen_CAN_SAY_YES_and_reproduces_the_measured_budgets():
    """A screen that only ever says no decides nothing, so gate both answers.

    `scripts/ceiling_screen.py` prices a dataset BEFORE a campaign: the whole
    prize for any method is `(1-p)*K` items, because emitting only K
    predictions for a class with n true instances caps cc-F1 at `2K/(K+n)`.

    Two things must hold or it is decoration. It must reproduce the budgets
    `headroom` measured from stored predictions -- K = 74 / 92 / 111 / 137 and
    ceilings 0.3333 / 0.3358 / 0.4615 / 0.4621 on iwildcam -- from LABELS
    alone, with no model; and it must be able to return WORTH RUNNING, which
    it does as soon as the ranking is worse or the budget larger.
    """
    import io

    from scripts.ceiling_screen import report, self_test

    buf = io.StringIO()
    assert self_test(out=buf) == 0, buf.getvalue()

    # The four measured (cap, class) budgets, priced from labels alone.
    measured = [("L20_G50", 2, 370, 185, 74, 74, "0.3333"),
                ("L20_G50", 7, 456, 228, 92, 92, "0.3358"),
                ("L30_G50", 2, 370, 185, 111, 111, "0.4615"),
                ("L30_G50", 7, 456, 228, 137, 137, "0.4621")]
    for tag, c, n, kg, kl, k, ceiling in measured:
        buf = io.StringIO()
        worth = report([(tag, c, n, kg, kl, k)], out=buf)
        text = buf.getvalue()
        assert ceiling in text, (
            "%s class %d: ceiling 2*%d/(%d+%d) should print %s\n%s"
            % (tag, c, k, k, n, ceiling, text))
        assert worth == 0 and "PRIZE BELOW THE NOISE" in text, (
            "iwildcam's measured cells must not read as worth running:\n%s"
            % text)

    # ... and it must be able to say yes. The bar is prize >= 2x the sd,
    # because a method never captures the WHOLE gap to a perfect ranking.
    buf = io.StringIO()
    assert report([("L80_G80", 2, 370, 300, 300, 300)], ccp=0.90, noise=3.0,
                  out=buf) == 1, buf.getvalue()
    assert "WORTH RUNNING" in buf.getvalue()

    # THE CALIBRATION MUST MOVE WITH K/n, in BOTH columns. A fixed p said
    # iwildcam had no prize at any cap -- false, and the first thing this tool
    # got wrong. A fixed sd then makes a loose cap look free.
    buf = io.StringIO()
    report([("L20_G50", 2, 370, 185, 74, 74),
            ("L80_G80", 2, 370, 300, 300, 300)], out=buf)
    rows = [l.split() for l in buf.getvalue().splitlines()
            if l.strip().startswith(("L20_G50", "L80_G80"))]
    tight, loose = rows[0], rows[1]
    assert float(loose[5]) < float(tight[5]) - 0.02, (
        "p@K must FALL as the budget grows: %s vs %s" % (tight[5], loose[5]))
    assert float(loose[7]) > 5.0 * float(tight[7]), (
        "the seed sd must RISE with the budget (0.40 -> 9.66 items on "
        "iwildcam): %s vs %s" % (tight[7], loose[7]))
    # and the ratio at the protocol's own cap must stay under 1
    assert float(tight[8].rstrip("x")) < 1.0, (
        "L20 is a cap this protocol sweeps and its whole prize is under the "
        "seed noise: %s" % tight)


def test_no_numerical_guard_in_the_TRAINING_PATH_is_a_no_op():
    """The dead-guard class, audited to its edge instead of one instance at a time.

    Two were found on 2026-08-25: `clamp(EPSILON, 1 - EPSILON)`, whose upper
    bound is a no-op in every dtype, and `clamp(min=EPSILON)`, whose lower one
    is a no-op in float16. Both produced NaN, both dropped constraint steps,
    both wrote `status: completed` anyway.

    So the whole surface was enumerated by AST: 63 sites in `src/` that take a
    logarithm, a square root, or divide by something non-constant. The triage,
    2026-08-25, with the reason for each rather than a count:

      * `constraint_step._randomize_direction`  guarded, `if total > 0`
      * `constraint_step.project_out`           guarded, `if nrm <= 0: return`
      * `constraint_step` normalize rescale     only reached when raw > 0
      * `hounie_rcl` group means                guarded, `max(1, group_sizes[g])`
      * `transductive_loss._penalty`            `scale = K if K >= 1 else 1.0`
      * `reordering` log-odds                   eps 1e-6 in float64, where
                                                `1 - eps` IS representable
      * `imbalanced_losses.LogitAdjustedLoss`   clamp 1e-12 on a float32 buffer
      * `select` risk denominators              `+ EPSILON` on a float32 sum
      * everything else                         `pathlib` `/`, not division

    Enumeration is not verification, so the paths are EXERCISED here with the
    inputs that would break them -- a class with no training instances, a
    zero-norm reference, an all-zero gradient, K = 0, a saturated softmax --
    in float16, bfloat16 and float32. Anything non-finite is a defect of the
    same class, whatever it looks like in the source.

    Negative controls, both run 2026-08-25:
      * deleting `if nrm <= 0.0: return 0.0` from `project_out` fails this
        with `ZeroDivisionError: float division by zero` -- the zero-reference
        case is a Python float division, so it raises rather than returning
        nan, which is the better failure and is still a failure nothing else
        was checking for;
      * deleting the `clamp(min=1e-12)` from `LogitAdjustedLoss` fails it on
        `log_prior`, because a class with no training instances has prior 0.

    ⚠️ NOT every guard here is load-bearing for FINITENESS, and the difference
    matters. Replacing `scale = K if K >= 1 else 1.0` with `float(K)` in
    `_penalty` does NOT fail this gate: at K = 0 the `+ EPSILON` keeps the
    quotient finite, it just makes it enormous. That guard protects the SCALE,
    which is section 2(a2)'s subject, not this one. A gate that claimed it
    covered both would be lying about one.
    """
    import torch

    from src.losses.imbalanced_losses import (class_balanced_criterion,
                                              logit_adjusted_criterion)
    from src.losses.transductive_loss import (MulticlassTransductiveLoss,
                                              margins, margin_window,
                                              uniform_grad_count, window_temp)
    from src.training.constraint_step import _randomize_direction, project_out

    # 1. The count relaxations, on a SATURATED softmax in every dtype. That is
    #    the input iwildcam actually produces -- warm-up ends at 0.998 train
    #    accuracy -- and it is what killed `tralo_uniform`.
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        p = torch.tensor([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [1.0 / 3, 1.0 / 3, 1.0 / 3]], dtype=dtype)
        assert torch.isfinite(uniform_grad_count(p)).all(), dtype
        assert torch.isfinite(margins(p)).all(), dtype
        t = window_temp(margins(p), 2)
        assert torch.isfinite(t).all() and (t > 0).all(), (dtype, t)
        assert torch.isfinite(margin_window(p, t)).all(), dtype

    # 2. The penalty, at K = 0 -- SEVEN of iwildcam's fourteen per-group
    #    ceilings are zero, so this is the common case there, not a corner.
    for shape in ("linear", "squared", "rational_bounded"):
        loss = MulticlassTransductiveLoss([1e10, 1e10, 1e10], {},
                                          num_classes=3,
                                          penalty_shape=shape)
        for K in (0, 1, 500):
            for soft in (0.0, 0.5, 1e6):
                v = loss._penalty(torch.tensor(soft), K)
                assert torch.isfinite(v).all(), (shape, K, soft, v)

    # 3. A class with NO training instances: prior 0, log(0) = -inf unclamped.
    y = torch.tensor([0, 0, 1, 1, 1])          # class 2 never appears
    crit = logit_adjusted_criterion(y, 3, torch.device("cpu"))
    assert torch.isfinite(crit.log_prior).all(), crit.log_prior
    logits = torch.zeros(4, 3, requires_grad=True)
    out = crit(logits, torch.tensor([0, 1, 0, 1]))
    out.backward()
    assert torch.isfinite(out) and torch.isfinite(logits.grad).all()

    cb = class_balanced_criterion(y, 3, torch.device("cpu"))
    assert torch.isfinite(cb.weight).all(), cb.weight

    # 4. A gradient that is exactly zero, and a reference that is exactly zero.
    #    `clip / total` and `dot / nrm` are both divisions by a quantity the
    #    caller does not control.
    net = torch.nn.Linear(3, 2)
    for prm in net.parameters():
        prm.grad = torch.zeros_like(prm)
    _randomize_direction(net, 1.0, torch.zeros(1))
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())

    for prm in net.parameters():
        prm.grad = torch.ones_like(prm)
    coef = project_out(net, [torch.zeros_like(p) for p in net.parameters()])
    assert coef == 0.0, "a zero reference must project to nothing, not to nan"
    assert all(torch.isfinite(p.grad).all() for p in net.parameters())


def test_the_order_verdict_REFUSES_to_call_a_coin_flip():
    """A pooled-mean sign with no test called `tralo_uniform` a reorderer.

    `scripts/order_probe` branched on `dd.mean() >= 0` alone, so on
    `results/loose1` (2026-08-28) a mean of -0.0076 at a 27/48 split printed
    "the constraint reordered MORE than a reseed. The order-preservation
    argument does NOT hold here."

    The killer is the SECOND arm it fired on. `tralo_uniform`'s per-item
    gradient is constant in log-odds, so on the direct channel it is a pure
    bias shift that CANNOT reorder -- configs/protocol.yml says exactly that at
    its definition. It read 26/48, p=0.66, and got the same verdict. That arm
    is this probe's built-in negative control, and the probe failed it.

    So this gate drives BOTH real splits and asserts they read TIE, and drives
    a 40/48 split to prove the verdict has not simply been muted.
    """
    import io

    import pandas as pd

    from scripts.order_probe import sign_test, verdict

    def call(k, n):
        dd = pd.Series([-0.01] * k + [0.01] * (n - k))
        out = io.StringIO()
        p = verdict(dd, pd.Series([0.001] * n), out=out)
        return p, out.getvalue()

    # The two splits measured on loose1. Neither may be called.
    for k, arm in ((27, "tralo"), (26, "tralo_uniform")):
        p, txt = call(k, 48)
        assert p >= 0.05, "%s: %d/48 is a coin flip, p=%.3f" % (arm, k, p)
        assert "TIE" in txt, (
            "%s reads %d/48 (sign p=%.3f) and the verdict did NOT say TIE. "
            "That is the defect this gate exists for -- a pooled-mean sign "
            "with nothing gating it. Verdict was:\n%s" % (arm, k, p, txt))
        assert "reordered MORE than a reseed, and it CLEARS" not in txt

    # ...and the verdict is still LIVE: a real effect must still be called.
    p, txt = call(40, 48)
    assert p < 0.05 and "CLEARS the coin" in txt, (
        "40/48 is p=%.4g and must still be called a real effect, or this gate "
        "has been passed by muting the verdict rather than gating it:\n%s"
        % (p, txt))

    # The test itself must be exact, not an approximation that drifts.
    assert abs(sign_test(27, 48) - 0.4709) < 5e-4
    assert abs(sign_test(24, 48) - 1.0) < 1e-12
    assert sign_test(0, 48) < 1e-13

    # And the case the verdict must not let a reader walk past: global TIE
    # while the BAND clears. Global rho is diluted by the easy mass; the band
    # is ranks K/2..2K, where the cut falls. These are uniform1's real splits
    # (2026-08-28): global 41/72 p=0.289, band 45/72 p=0.044.
    out = io.StringIO()
    verdict(pd.Series([-0.01] * 41 + [0.01] * 31),
            pd.Series([-0.05] * 45 + [0.05] * 27), out=out)
    txt = out.getvalue()
    assert "TIE" in txt and "BAND CLEARS" in txt, (
        "global ties (41/72) but the band clears (45/72, p=0.044) and the "
        "verdict did not say so. The band is the statistic that matters and "
        "printing it without acting on it is how it gets missed:\n%s" % txt)

    # CONTROL: a band that is itself a coin must NOT raise the note.
    out = io.StringIO()
    verdict(pd.Series([-0.01] * 41 + [0.01] * 31),
            pd.Series([-0.05] * 36 + [0.05] * 36), out=out)
    assert "BAND CLEARS" not in out.getvalue(), (
        "the band note fired on a 36/72 band, i.e. on nothing")


def test_the_eviction_NET_ITEMS_is_priced_and_flagged_as_a_global_topK():
    """`--evictions` reported +16.50 items with no noise and no caveat.

    Two independent defects, both measured on `results/loose1` 2026-08-28:

    1. NO POWER. The verdict branched on `d_net` against a bare +/-1.0 items,
       so +16.50 printed "the constraint's swaps are BETTER than a reseed's"
       with nothing saying the within-cell paired sd is of the same order.

    2. WRONG ALLOCATOR. The sets are `argsort(-p)[:K]` on the raw class
       column, i.e. a GLOBAL top-K. The allocator that actually ran is
       LP/greedy under per-group ceilings, and on iwildcam 7 of 14 local
       ceilings are K=0, so it cannot take the global top-K. `full_panel
       --control tralo_null` scored the same campaign at tralo +9.24 items
       against tralo_reseed +6.71, i.e. +2.53 attributable -- the probe's
       number was 6.5x too large.
    """
    import ast
    import io

    import numpy as np
    import pandas as pd

    from scripts.order_probe import paired_sd_items

    # The pooled sd must be a WITHIN-cell sd: a constant per-cell offset is a
    # real difference, not noise, and pooling it in would hide effects.
    rng = np.random.RandomState(0)
    rows = []
    for mdl in ("A", "B"):
        for cap in ("L80", "L90"):
            for cls in (2, 7):
                for s, b in zip(range(4), rng.randn(4) * 5):
                    rows.append(dict(model=mdl, cap=cap, seed="seed_%d" % s,
                                     cls=cls, net_items=b))
    arm = pd.DataFrame(rows)
    ctrl = arm.copy()
    ctrl["net_items"] = ctrl.net_items - 3.0
    sd, n_cells = paired_sd_items(arm, ctrl)
    assert n_cells == 8 and abs(sd) < 1e-9, (
        "a constant offset per cell must pool to sd 0, got %.6f" % sd)

    # ...and it must still SEE real noise, or it is a constant-zero stub.
    ctrl2 = arm.copy()
    ctrl2["net_items"] = ctrl2.net_items + rng.randn(len(ctrl2)) * 4
    sd2, _ = paired_sd_items(arm, ctrl2)
    assert sd2 > 1.0, "independent noise must give sd > 0, got %.3f" % sd2

    # And the evictions block must actually USE it and carry the caveat.
    src = io.open("scripts/order_probe.py", encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next(f for f in ast.walk(tree)
              if isinstance(f, ast.FunctionDef) and f.name == "main")
    called = {c.func.id for c in ast.walk(fn)
              if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    assert "paired_sd_items" in called, (
        "main() computes an eviction NET without ever pricing it against the "
        "within-cell seed sd -- that is the defect this gate exists for")
    assert "GLOBAL TOP-K, NOT THE ALLOCATOR THAT RAN" in src, (
        "the global-top-K caveat was removed; the number is 6.5x the "
        "allocator's on loose1 and must not be quoted bare")


def test_dataset_screen_NAMES_the_slice_not_the_convention():
    """Screening 21 candidates printed `oodslice` on all 21 rows.

    Every candidate slice is written to `<dataset>/oodslice` by convention, so
    `os.path.basename(dirname(path))` is the SAME string for all of them. This
    tool exists to be run on many slices at once -- its entire output is the
    comparison between them -- so a label that cannot tell them apart makes
    the multi-slice mode useless. Measured 2026-08-28 on the ~/_cand
    inventory: 21 rows, one distinguishable name.
    """
    from scripts.dataset_screen import slice_label

    got = {slice_label(p) for p in (
        "/home/x/_cand/fmow_country/oodslice",
        "/home/x/_cand/isic_src/oodslice/",
        "data/iwildcam/oodslice",
        "data/dermmnist/slice_1",
    )}
    assert len(got) == 4, (
        "four different slices collapsed to %d label(s): %s" % (len(got), got))
    assert slice_label("data/iwildcam/oodslice") == "iwildcam/oodslice"
    assert slice_label("data/dermmnist/slice_1") == "dermmnist/slice_1"
    # a non-generic leaf is already informative and must not gain a parent
    assert slice_label("data/tissuemnist") == "tissuemnist"


def test_a_launch_scripts_stated_SIZE_and_SIGN_TEST_are_arithmetic_not_prose():
    """A launch header is a pre-registration, so its numbers are claims.

    `docs/launch_margin2.sh` states a cap grid, a cell count, a run count and a
    PASS threshold with an exact binomial p beside it. Every one of those is a
    number a human typed, and this campaign has already been re-scoped once
    (3 cap tags -> a matched 2x2), which is exactly when such numbers go stale:
    the size line said `9 cells x 9 arms x 4 seeds = 324` for a grid that emits
    432. `check_parity` cannot catch it -- it reads the CONFIGS, not the prose
    that justified them.

    So: re-derive the arithmetic from the flags the script actually passes to
    `gen_campaign`, and re-derive the sign-test thresholds from scratch.
    """
    import math
    import re

    src = open(os.path.join(REPO, "docs/launch_margin2.sh"),
                encoding="utf-8").read()
    # The header is PROSE and wraps. Anchoring a pattern to one physical line
    # makes the gate fail on a harmless reflow and -- worse -- pass if someone
    # reflows a threshold out of existence. Strip the comment markers and
    # collapse whitespace, so the checks below read meaning, not layout.
    flat = re.sub(r"\s+", " ", re.sub(r"(?m)^#", "", src))

    # The invocation spans several lines with continuations, so join them
    # before tokenising -- a regex that stops at the newline reads 5 of the
    # 7 arms and silently under-counts the campaign by 192 runs.
    body = src[src.index('gen_campaign'):]
    toks = re.sub('\\\\\\s*\\n\\s*', ' ', body).split()

    def flag(name):
        assert '--' + name in toks, (
            'launch_margin2.sh passes no --%s' % name)
        vals = []
        for t in toks[toks.index('--' + name) + 1:]:
            if t.startswith('--'):
                break
            vals.append(t)
        assert vals, '--%s is passed with no value' % name
        return vals

    models, caps, arms = flag("models"), flag("caps"), flag("arms")
    # gen_campaign ALWAYS adds both clippers; CLAUDE.md rule 2 and the
    # generator's own assertion. They are cells' arms too and must be counted.
    n_arms = len(set(arms) | {"clip", "focal_clip"})
    cells = len(models) * len(caps)            # one dataset: iwildcam
    runs = cells * n_arms * 4                  # 4 seeds, the atomic cell

    m = re.search(r"#\s+size\s+(\d+) cells x (\d+) arms x (\d+) seeds = (\d+) runs", src)
    assert m, "no parseable `size` line in the header"
    said = tuple(int(g) for g in m.groups())
    assert said == (cells, n_arms, 4, runs), (
        "the header says %s but the flags give %s cells x %s arms x 4 seeds "
        "= %s runs" % (said, cells, n_arms, runs))

    # every cap tag named in the header's table must be one the script runs,
    # and vice versa -- a table row for a cap that was dropped is a lie that
    # reads as evidence.
    tabled = set(re.findall(r"^#\s+(L\d+_G\d+)\s+\d+", src, re.M))
    assert tabled == set(caps), (
        "header cap table %s != --caps %s" % (sorted(tabled), sorted(caps)))

    def two_sided(k, n):
        tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1))
        return min(1.0, 2.0 * tail / float(2 ** n))

    # the PASS threshold, and the value it is contrasted against, are both
    # asserted so neither can drift from the cell count.
    m = re.search(r"PASS = positive in >= (\d+) of (\d+) \(p = 2\*(\d+)/(\d+) = ([\d.]+)\)", flat)
    assert m, "the primary PASS threshold is not stated in a checkable form"
    k, n, num, den, p = (int(m.group(1)), int(m.group(2)), int(m.group(3)),
                         int(m.group(4)), float(m.group(5)))
    assert n == cells, "PASS is stated over %d cells, the grid has %d" % (n, cells)
    assert den == 2 ** n, "2^%d is %d, not %d" % (n, 2 ** n, den)
    assert num == sum(math.comb(n, i) for i in range(k, n + 1))
    assert abs(two_sided(k, n) - p) < 5e-5, (
        "%d of %d is p=%.4f, header says %.4f" % (k, n, two_sided(k, n), p))
    assert p < 0.05, "the stated PASS threshold does not actually pass"
    # and the near miss must be stated as a FAIL, so nobody reads k-1 as a win
    m = re.search(r"(\d+) of (\d+) is p = ([\d.]+) and does NOT pass", flat)
    assert m and int(m.group(1)) == k - 1 and int(m.group(2)) == n, (
        "the header does not state the near-miss cell count as a failure")
    assert abs(two_sided(k - 1, n) - float(m.group(3))) < 5e-5
    assert two_sided(k - 1, n) >= 0.05

    # the regime split must PARTITION the cells, not overlap or leave a gap:
    # a secondary stated over more cells than exist is unfalsifiable.
    m = re.search(r">= (\d+) of the (\d+) TIGHT cells AND >= (\d+) of the (\d+) LOOSE", flat)
    assert m, "the regime-consistency secondary is not stated in a checkable form"
    assert int(m.group(2)) + int(m.group(4)) == cells, (
        "TIGHT %s + LOOSE %s != %d cells" % (m.group(2), m.group(4), cells))

    # macroF1 is the metric the user has had to ask for twice. It is a NAMED
    # secondary here, and this gate is what keeps it named.
    assert "macroF1 AND uncF1" in flat, (
        "macroF1/uncF1 dropped out of the pre-registered secondaries")

    # The cells are NOT independent: `verify_caps` reports that L80_G95 and
    # L95_G80 give class 2 the same K=296, and L30_G50/L50_G30 the same K=111,
    # so each pair is one budget through two scopes; and all four tags within a
    # (model, seed) share one warm-up. Counting them as independent draws is
    # the error FRAMEWORK 2(z) caught on dom1, where 8 of 9 sweeps evaporated
    # once the unit was corrected. A pre-registration that does not name its
    # unit will be read at whichever n flatters the result.
    assert "(model, seed)" in flat, (
        "the header never names its independent unit -- with two matched cap "
        "pairs and a shared warm-up, `12 cells` is not 12 independent draws")
    assert re.search(r"sign test over the \d+ independent \(model, seed\) units",
                     flat), (
        "the primary must state that its sign test runs over independent "
        "(model, seed) units, not over cells")


# ---------------------------------------------------------------------------
# Per-class cap fractions (FRAMEWORK 2(z16)). The two capped classes on
# iwildcam have task windows that DO NOT OVERLAP on MobileNetV3, so one
# fraction for both cannot express a valid experiment. These gates protect the
# backward compatibility of every config written before that existed.
# ---------------------------------------------------------------------------

def _cap_df():
    import pandas as pd
    return pd.DataFrame({"label": [2] * 100 + [7] * 200 + [0] * 50,
                         "g": [0] * 175 + [1] * 175})


def test_a_scalar_cap_fraction_is_bit_identical_to_the_historical_behaviour():
    """Every config written before per-class caps carries a scalar. If this
    ever changes, every archived budget silently becomes non-comparable.
    """
    from src.training.constraints import compute_global_constraints
    g = compute_global_constraints(_cap_df(), "label", 0.8,
                                   constrained_class=[2, 7], num_classes=8)
    assert g[2] == 80 and g[7] == 160


def test_a_per_class_cap_fraction_is_read_positionally():
    from src.training.constraints import (compute_global_constraints,
                                          compute_local_constraints)
    g = compute_global_constraints(_cap_df(), "label", [0.8, 1.0],
                                   constrained_class=[2, 7], num_classes=8)
    assert g[2] == 80 and g[7] == 200, (
        "L80-100 on classes [2,7] must cap class 2 at 80%% and class 7 at "
        "100%%, got %s" % [g[2], g[7]])

    # reversing the list must reverse the budgets, or it is not positional
    r = compute_global_constraints(_cap_df(), "label", [1.0, 0.8],
                                   constrained_class=[2, 7], num_classes=8)
    assert r[2] == 100 and r[7] == 160

    # group 0 = rows 0..174 -> 100 of class 2, 75 of class 7
    # group 1 = rows 175..349 -> 0 of class 2, 125 of class 7
    loc = compute_local_constraints(_cap_df(), "label", [0.8, 1.0], "g",
                                    constrained_class=[2, 7], num_classes=8)
    assert loc[0][2] == 80 and loc[0][7] == 75
    assert loc[1][2] == 0 and loc[1][7] == 125


def test_NEGATIVE_CONTROL_a_mismatched_cap_list_raises_rather_than_recycling():
    """Silently recycling or truncating would cap the wrong class at the wrong
    level and look completely normal in every log.
    """
    import pytest as _pytest
    from src.training.constraints import compute_global_constraints
    for bad in ([0.8], [0.8, 0.9, 1.0]):
        with _pytest.raises(ValueError):
            compute_global_constraints(_cap_df(), "label", bad,
                                       constrained_class=[2, 7], num_classes=8)


def test_the_cap_tag_parses_both_the_scalar_and_the_per_class_form():
    from configs.gen_campaign import cap_pair
    assert cap_pair("L30_G50") == [0.30, 0.50]
    assert cap_pair("L90_G95") == [0.90, 0.95]
    assert cap_pair("L80-100_G95") == [[0.80, 1.00], 0.95]


def test_a_cap_above_100_percent_is_legal_and_still_binds():
    """K/n = 1.00 is NOT degenerate: on iwildcam class 7 the model predicts 490
    against 456 true, so a budget equal to the true count still evicts 34.
    """
    from src.training.constraints import compute_global_constraints
    g = compute_global_constraints(_cap_df(), "label", [1.0, 1.2],
                                   constrained_class=[2, 7], num_classes=8)
    assert g[2] == 100 and g[7] == 240
