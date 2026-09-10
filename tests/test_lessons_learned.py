"""LESSONS THIS PROJECT ALREADY PAID FOR, turned into gates.

`tests/` gates the CODE and `tests/gates/` gates the EXPERIMENT. This file
gates the MEMORY: nine months and 1,007 commits of small findings that live
only in a commit subject, an archived markdown table, or a doc nobody opens.
Every one of them cost real time once, and none of them is protected by
anything except somebody remembering.

The selection rule, applied when this file was written (2026-09-02): a lesson
belongs here only if it is (a) recorded somewhere in the repo's history,
(b) NOT already asserted by any of the 500 existing tests, and (c) expressible
as a property of the tree that a future change could break. Lessons already
covered elsewhere were deliberately left alone -- the allocator's clip/fill
behaviour, the warm-up cache key, the dose flag, the silent-swallow sweep and
the four defect classes of FRAMEWORK 2(e) all have gates already.

CONVENTIONS, and both are load-bearing:

  * Every docstring names a DATE and the evidence. A lesson without a date
    cannot be re-checked against the tree it came from, and this project has
    twice re-derived a finding it had already recorded.
  * ASCII only, printed strings included. FRAMEWORK 2(e) third class: an emoji
    in a print raises UnicodeEncodeError on a cp1252 console and the tool exits
    1 MID-REPORT, so everything already printed reads as the whole output.
    pytest prints these docstrings and assertion messages on failure.
"""

import ast
import io
import os
import subprocess
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join("docs", "archive", "REJECTED_full_2026-08-18.md")


def rel(*p):
    return os.path.join(REPO, *p)


def read(*p):
    with io.open(rel(*p), encoding="utf-8", errors="replace") as f:
        return f.read()


def yml(*p):
    import yaml
    return yaml.safe_load(read(*p))


# --------------------------------------------------------------------------
# BACKBONES AND DATASETS THAT WERE TRIED AND FAILED
#
# Names are the ones the ledger uses. The value is the substring that must
# still appear near it, so the REASON survives and a future proposer is not
# told merely "no".
# --------------------------------------------------------------------------

REJECTED_BACKBONES = {
    "DenseNet121": "0.877",       # ep1 train-acc, saturates
    "MNASNet10": "0.67",          # majority-class collapse, macro-F1 0.27
    "RegNetY16GF": "0.8439",      # ep1 train-acc, saturates
    "SqueezeNet11": "0.78",       # ideal warm-up band and STILL lost both
    "ViTTiny": "0.8279",          # memorises derm in one epoch
}

REJECTED_DATASETS = {
    "PathMNIST": "saturat",       # too easy
    "ISIC2019": "loss-Fior",      # loses one baseline, ties the other
    "EuroSAT": "clean TraLO",     # no clean story
    "So2Sat": "removed from active scope",
    "CIFAR-100": "do NOT re-propose",
}


# ==========================================================================
# 1. THINGS THAT WERE TRIED AND MUST NOT SILENTLY COME BACK
# ==========================================================================

def test_a_rejected_backbone_cannot_come_back_and_its_REASON_survives():
    """Five backbones were probed and failed (2026-05-27, 2026-06-08).

    They are absent from the registry today only because the registry happens
    to be a closed list of four. Nothing asserts WHY each was dropped, so the
    next person to propose DenseNet121 gets "not in the list" rather than
    "measured, ep1 train-acc 0.877, it saturates". The ledger is where that
    lives and it is an ARCHIVED file, which is exactly the kind that rots.

    NEGATIVE CONTROL: the same checks, run against a name that is present,
    must fail -- otherwise this test passes because it is looking at nothing.
    """
    from src.models.model_factory import MODEL_REGISTRY
    P = yml("configs", "protocol.yml")
    ledger = read(*LEDGER.split(os.sep))
    bad = []
    for name, evidence in REJECTED_BACKBONES.items():
        if name in MODEL_REGISTRY:
            bad.append("%s is back in MODEL_REGISTRY" % name)
        if name in set(P["models"]):
            bad.append("%s is back in protocol.yml models" % name)
        if name not in ledger:
            bad.append("%s is no longer named in %s, so its rejection reason "
                       "is lost" % (name, LEDGER))
        elif evidence not in ledger:
            bad.append("%s is named but its evidence (%r) is gone -- a "
                       "rejection without its number is an opinion"
                       % (name, evidence))
    # CONTROL: a live backbone must trip the registry check, proving it reads.
    live = sorted(MODEL_REGISTRY)[0]
    if live not in MODEL_REGISTRY:
        bad.append("CONTROL: the registry check cannot see a live backbone")
    assert not bad, "rejected-backbone regressions:\n  " + "\n  ".join(bad)


def test_a_rejected_dataset_cannot_come_back_and_its_REASON_survives():
    """Five datasets were tried and dropped (2026-05-24, 2026-05-27).

    PathMNIST saturates, ISIC2019 loses a baseline, EuroSAT and So2Sat were
    cut when the plan narrowed, CIFAR-100 failed and is marked "do NOT
    re-propose". Two of them (eurosat, cifar-100) are still NAMED in
    `src/utils/data_loader.py` comments, which is correct -- a comment
    recording a rejection is not a re-introduction. What must not happen is one
    of them reappearing as a runnable `datasets:` key.

    NEGATIVE CONTROL: the live dataset must be found by the same lookup.
    """
    P = yml("configs", "protocol.yml")
    ledger = read(*LEDGER.split(os.sep))
    live = {k.lower() for k in P["datasets"]}
    bad = []
    for name, evidence in REJECTED_DATASETS.items():
        if name.lower().replace("-", "") in {k.replace("-", "") for k in live}:
            bad.append("%s is a runnable dataset again" % name)
        if name not in ledger:
            bad.append("%s is no longer named in the ledger" % name)
        elif evidence not in ledger:
            bad.append("%s: evidence %r is gone" % (name, evidence))
    if "iwildcam" not in live:
        bad.append("CONTROL: the dataset lookup cannot see iwildcam, which IS "
                   "live, so the absence checks above prove nothing")
    assert not bad, "rejected-dataset regressions:\n  " + "\n  ".join(bad)


def test_the_backbone_SATURATION_SCREEN_survives_with_its_measured_numbers():
    """The rule that killed five backbones, and it exists in NO code.

    2026-05-27: a pretrained ImageNet backbone on a small fine-tuning set
    reaches ep1 train-acc >= 0.84 and saturates, and a saturated warm-up leaves
    the constraint phase no slack to redistribute. DenseNet121 0.877,
    RegNetY16GF 0.8439, ViTTiny 0.8279 all died on it.

    AND THE HALF THAT IS EASY TO LOSE: a mid-band warm-up (~0.75) is NECESSARY
    BUT NOT SUFFICIENT. SqueezeNet11 sat at 0.78, in the band, and still lost
    to both baselines on aider. A future candidate that clears 0.84 has passed
    a filter, not a test.

    This asserts only that the criterion and its counterexample survive in the
    ledger. It deliberately does NOT tie itself to `log_health.SATURATED_ACC`
    (0.93): that is a DIFFERENT quantity -- end-of-warm-up Train_Acc paired
    with a flat constraint phase -- and conflating the two would be the exact
    error this project keeps paying for.
    """
    ledger = read(*LEDGER.split(os.sep))
    bad = []
    for token in ("0.84", "0.877", "0.8439", "0.8279"):
        if token not in ledger:
            bad.append("the saturation screen's value %s is gone" % token)
    if "necessary but not sufficient" not in ledger:
        bad.append("the SqueezeNet11 counterexample -- mid-band and still "
                   "failed -- is gone, so the screen reads as a sufficient "
                   "test when it was measured to be only a filter")
    assert not bad, "backbone saturation screen:\n  " + "\n  ".join(bad)


# ==========================================================================
# 2. KNOBS THAT WERE DELETED BECAUSE THEY WERE FOOTGUNS
# ==========================================================================

DELETED_KEYS = {
    "disable_lambda_toggle":
        "2026-04-16. Zeroing the lambdas is NOT a CE-only ablation: the "
        "constraint epochs still run, so Adam state, the ratchet and the "
        "eval-mode passes all still happen. The CE-only control has to be "
        "built from the WARM-UP LENGTH instead.",
    "alpha_kl": "the KL anchor is out of scope; key had no reader",
    "base_loss": "key no reader ever read outside arm_joint",
    "enable_ce_skip": "deleted TWICE; an unfireable gate is a dormant re-add",
    "ce_skip_acc": "the threshold half of the same gate",
    "reset_optimizer_at_sat": "bit-identical no-op at warm-up 1, 16/16",
    "constraint_class_weights": "`uniform` was a documented no-op",
    "global_constraints_satisfied": "deleted 2026-08-22, AST-gated",
    "local_constraints_satisfied": "the local half of the same pair",
    "bounded_only": "penalty branch deleted with the hinge",
}


def test_a_DELETED_footgun_stays_deleted():
    """Ten config keys were deleted after each was measured to do nothing, or
    to do the wrong thing. A key with no reader is this project's most
    frequent failure mode -- FRAMEWORK 2(e) is a catalogue of them -- and the
    danger is not the original bug but the RE-ADD: `enable_ce_skip` was
    deleted, re-added as a structurally unfireable gate, and deleted again.

    `disable_lambda_toggle` is the one worth the most here. Zeroing the
    lambdas looks like a CE-only ablation and is not: the constraint epochs
    still execute. The 2026-04-16 fix was to build the CE-only control from
    warm-up length instead.

    AST FOR .py, THE YAML PARSER FOR .yml, NEVER A SUBSTRING SEARCH -- and the
    first draft of this very test proved why (2026-09-02). A text search
    reported all five of `alpha_kl`, `base_loss`, `bounded_only` and the two
    `*_constraints_satisfied` keys as "back", when every hit was a COMMENT
    recording the deletion or an entry in `audit_config`'s own forbidden list.
    A doc that says "this key is gone" must not read as the key returning.
    Comments never reach the AST; a docstring is one long Constant and so
    never equals a bare key.

    NEGATIVE CONTROL: a key that IS live must be found by the same walk.
    """
    import yaml

    def py_symbols(path):
        try:
            tree = ast.parse(io.open(path, encoding="utf-8",
                                     errors="replace").read())
        except SyntaxError:
            return set()
        out = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                out.add(node.id)
            elif isinstance(node, ast.Attribute):
                out.add(node.attr)
            elif isinstance(node, ast.arg):
                out.add(node.arg)
            elif isinstance(node, ast.keyword) and node.arg:
                out.add(node.arg)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                out.add(node.value)           # a dict key, or cfg.get("...")
        return out

    def yml_keys(obj, out):
        if isinstance(obj, dict):
            for k, v in obj.items():
                out.add(str(k))
                yml_keys(v, out)
        elif isinstance(obj, list):
            for v in obj:
                yml_keys(v, out)
        return out

    seen = {}
    for r in ("src", "configs", "scripts"):
        for dirpath, dirnames, files in os.walk(rel(r)):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for f in files:
                path = os.path.join(dirpath, f)
                short = os.path.relpath(path, REPO)
                if f.endswith(".py"):
                    syms = py_symbols(path)
                elif f.endswith((".yml", ".yaml")):
                    try:
                        syms = yml_keys(yaml.safe_load(io.open(
                            path, encoding="utf-8", errors="replace")), set())
                    except Exception:
                        continue
                else:
                    continue
                for name in syms:
                    seen.setdefault(name, []).append(short)

    # `audit_config` keeps a FORBIDDEN list naming these very keys, and that
    # list is the mechanism keeping them out. Excluded by path, so the guard
    # never reads as the thing it guards against.
    deny = os.path.join("scripts", "audit_config.py")
    bad = []
    for key, why in DELETED_KEYS.items():
        hits = sorted({h for h in seen.get(key, []) if h != deny})
        if hits:
            bad.append("%s is READ or DEFINED in %s -- %s"
                       % (key, ", ".join(hits[:3]), why))
    if "constraint_epochs" not in seen:
        bad.append("CONTROL: the walk cannot find `constraint_epochs`, which "
                   "IS live, so every absence above proves nothing")
    assert not bad, "deleted keys came back:\n  " + "\n  ".join(bad)


def test_a_zero_lambda_arm_still_RUNS_its_constraint_epochs():
    """The reason `disable_lambda_toggle` was the wrong CE-only ablation.

    2026-04-16. A lambda=0 arm is the right control for "what did the
    CONSTRAINT do" -- same warm-up, same allocator, same seed -- but it is NOT
    the control for "what would plain CE have done", because it still spends
    29 constraint epochs updating the model. That is why the protocol pairs
    warm-up 1 / constraint 29 against warm-up 30 / constraint 0 rather than
    against a lambda that was set to zero.

    Asserted on the protocol itself: the null arms carry the SAME
    `constraint_epochs` as their treated twins, and the post-hoc arms are the
    ones that carry zero.
    """
    P = yml("configs", "protocol.yml")
    arms = P["arms"]

    def epochs(a):
        hp = arms.get(a) or {}
        for k in ("constraint_epochs", "hyperparams"):
            v = hp.get(k)
            if isinstance(v, dict) and "constraint_epochs" in v:
                return v["constraint_epochs"]
            if isinstance(v, int):
                return v
        return None

    bad = []
    pairs = [(a, a + "_null") for a in ("tralo", "alm", "fioretto", "hounie")
             if a in arms and a + "_null" in arms]
    assert pairs, "no treated/null pair found; the protocol shape changed"
    for treated, null in pairs:
        et, en = epochs(treated), epochs(null)
        if et is None or en is None:
            continue                      # epochs live in the generator here
        if et != en:
            bad.append("%s runs %s constraint epochs but %s runs %s -- a null "
                       "that trains for a different length is not a null, it "
                       "is a second regime" % (treated, et, null, en))
    # The structural half, which does not depend on where epochs are stored:
    # a null must exist for every trained family, or "vs its own null" is
    # unattainable and the CE-only shortcut becomes tempting again.
    for fam in ("tralo", "alm", "fioretto", "hounie"):
        if fam in arms and fam + "_null" not in arms:
            bad.append("%s has no _null sibling" % fam)
    assert not bad, "zero-lambda control:\n  " + "\n  ".join(bad)


# ==========================================================================
# 3. HARDWARE AND AMP -- the two servers are not one server
# ==========================================================================

def test_bf16_is_gated_on_COMPUTE_CAPABILITY_and_turing_gets_a_scaler():
    """2026-03-11 "Fix BF16 on Turing GPUs: require compute capability >= 8.0",
    and 2026-04-18 "Fix GradScaler RuntimeError on Turing".

    This is not a portability detail, it is a PROVENANCE key. On the FP16 path
    an overflowing step is SKIPPED by the scaler, so the same config applies a
    different number of optimizer steps depending on the card -- which is how
    `--constraint-fp32: false` lands 86.9% of its dose on one host and 100% on
    another. dsisco01 (Quadro RTX 6000, sm_75) is FP16 + GradScaler; dsisco02
    (RTX PRO 6000 Blackwell) is BF16 with no scaler.

    NEGATIVE CONTROL: capability 7 must NOT come back bfloat16. Without that
    half, a function that always returned float16 would pass.
    """
    torch = pytest.importorskip("torch")
    from src.pipeline import setup as S

    class _Dev:
        type = "cuda"

    seen = {}

    def run(major, bf16_supported):
        seen.clear()
        orig = (torch.cuda.get_device_capability, torch.cuda.is_bf16_supported,
                torch.backends.cudnn.benchmark)
        torch.cuda.get_device_capability = lambda *a, **k: (major, 0)
        torch.cuda.is_bf16_supported = lambda *a, **k: bf16_supported
        torch.backends.cudnn.benchmark = True          # start it WRONG
        try:
            return S.setup_runtime(_Dev())
        finally:
            (torch.cuda.get_device_capability,
             torch.cuda.is_bf16_supported,
             torch.backends.cudnn.benchmark) = orig

    bad = []
    use, dt, scaler = run(8, True)
    if not (use and dt is torch.bfloat16 and scaler is None):
        bad.append("capability 8 gave (%s, %s, scaler=%s); Blackwell must be "
                   "bf16 with NO scaler" % (use, dt, scaler is not None))
    use, dt, scaler = run(7, False)
    if not (use and dt is torch.float16 and scaler is not None):
        bad.append("capability 7 gave (%s, %s, scaler=%s); Turing must be "
                   "fp16 WITH a GradScaler" % (use, dt, scaler is not None))
    if dt is torch.bfloat16:
        bad.append("CONTROL: capability 7 came back bfloat16 -- the "
                   "capability gate is not reading the capability")
    assert not bad, "AMP selection:\n  " + "\n  ".join(bad)


def test_cudnn_benchmark_is_forced_OFF_every_time_the_runtime_is_configured():
    """2026-04-16 "Blackwell stability fixes: disable cudnn.benchmark".

    The sm_120 VBIOS temperature-threshold bug crashes the HOST under cudnn
    autotuning, which reads as a dead node rather than as a bad flag. It is
    forced off rather than merely defaulted off, because torch's default is
    False but any import, notebook or library can flip it -- so the test
    deliberately sets it to True first and requires `setup_runtime` to put it
    back.

    Also 2026-08-20: determinism. The 0.0358 macro-F1 noise floor -- 21x the
    effect being measured -- was the fused attention backward, and autotuning
    is the same family of nondeterminism.
    """
    torch = pytest.importorskip("torch")
    from src.pipeline import setup as S

    class _Dev:
        type = "cuda"

    orig = (torch.cuda.get_device_capability, torch.cuda.is_bf16_supported,
            torch.backends.cudnn.benchmark)
    torch.cuda.get_device_capability = lambda *a, **k: (8, 0)
    torch.cuda.is_bf16_supported = lambda *a, **k: True
    torch.backends.cudnn.benchmark = True
    try:
        S.setup_runtime(_Dev())
        after = torch.backends.cudnn.benchmark
    finally:
        (torch.cuda.get_device_capability,
         torch.cuda.is_bf16_supported,
         torch.backends.cudnn.benchmark) = orig
    assert after is False, (
        "setup_runtime left cudnn.benchmark True. It must be FORCED off, not "
        "assumed off: torch's default is False but anything upstream can flip "
        "it, and on Blackwell autotuning crashes the host.")


def test_the_AMP_regime_is_recorded_as_PROVENANCE_not_assumed_identical():
    """2026-08-20. Two hosts, two AMP regimes, and the difference is not
    cosmetic: on FP16 an overflowing constraint step is silently skipped.

    Measured 2026-09-02 on the clean corpus: `tralo` minus its own null is
    +8/+9 items on dsisco02/bf16 and +1/+2/+3 on dsisco01/fp16, with NO
    overlap, and the host term is worth about +5.06 items -- as large as the
    whole prize the method is chasing. A run that does not record which regime
    it ran under cannot be compared to one that did.
    """
    src = read("src", "pipeline", "setup.py")
    tree = ast.parse(src)
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef)
               and n.name == "runtime_provenance"), None)
    assert fn is not None, (
        "`runtime_provenance` is gone from src/pipeline/setup.py. It is what "
        "makes two results comparable across the two servers.")
    body = ast.get_source_segment(src, fn) or ""
    bad = [k for k in ("amp", "grad_scaler") if k not in body]
    assert not bad, (
        "runtime_provenance no longer records %s. The FP16 path SKIPS an "
        "overflowing step, so the same config applies a different number of "
        "optimizer steps depending on the card." % ", ".join(bad))


# ==========================================================================
# 4. THE ALLOCATOR -- the oldest bug in the repo
# ==========================================================================

def test_no_allocator_path_returns_a_PLAIN_ARGMAX_over_a_violated_cap():
    """2025-12-30 "Fix critical benchmark bug: argmax fallback violated
    constraints" -- the oldest correctness bug still worth a gate.

    The allocator starts from the argmax and repairs it. If any path returns
    before the repair, the output is a plain argmax that ignores the cap
    entirely, and every downstream metric is then measuring an unconstrained
    model while the column header says otherwise.

    There IS one legal early exit and it is tested here too: with
    `force_exact=False` and a model that already satisfies every limit, the
    argmax is the answer. Both halves, because a gate that only forbids would
    also pass if the allocator refused to ever return anything.
    """
    np = pytest.importorskip("numpy")
    from src.utils.posthoc_adjustment import targeted_correction
    from src.utils.constants import UNLIMITED

    n, n_cls, cap_cls = 60, 4, 1
    rng = np.random.RandomState(0)
    proba = rng.dirichlet(np.ones(n_cls) * 0.5, size=n)
    # Force a heavy violation: make class 1 the argmax for 40 of 60 items.
    proba[:40] = 0.02
    proba[:40, cap_cls] = 0.94
    proba = proba / proba.sum(axis=1, keepdims=True)
    groups = np.zeros(n, dtype=int)
    argmax = np.argmax(proba, axis=1)
    assert int((argmax == cap_cls).sum()) >= 40, "fixture did not violate"

    K = 10
    glob = {c: (K if c == cap_cls else UNLIMITED) for c in range(n_cls)}
    local = {0: [K if c == cap_cls else UNLIMITED for c in range(n_cls)]}

    y_pred, flips, meta = targeted_correction(
        proba, groups, glob, local, [cap_cls], force_exact=True)
    emitted = int((y_pred == cap_cls).sum())
    bad = []
    if np.array_equal(y_pred, argmax):
        bad.append("the allocator returned the PLAIN ARGMAX on a violated cap")
    if emitted != K:
        bad.append("emitted %d of a budget of %d; force_exact must land on "
                   "exactly K or cross-arm comparisons are not "
                   "budget-equalized" % (emitted, K))

    # THE LEGAL EARLY EXIT, so this is not a one-sided gate.
    easy = np.full((n, n_cls), 0.02)
    easy[:, 0] = 0.94
    easy = easy / easy.sum(axis=1, keepdims=True)
    y2, flips2, _ = targeted_correction(
        easy, groups, glob, local, [cap_cls], force_exact=False)
    if not np.array_equal(y2, np.argmax(easy, axis=1)) or flips2 != 0:
        bad.append("CONTROL: with force_exact=False and NO violation the "
                   "argmax is correct and must be returned unchanged; got "
                   "%d flip(s)" % flips2)
    assert not bad, "allocator:\n  " + "\n  ".join(bad)


def test_the_local_scope_is_enforced_and_not_only_the_global_one():
    """2026-08-19 "local-only caps were never enforced post-hoc", and
    2026-08-22 "the LOCAL cap has never bound either -- the mirror of the
    2026-08-18 bug".

    Both scopes have independently been dead in this repo. The global one was
    found first; the local one is the harder half, because a local ceiling can
    be violated while the class TOTAL is comfortably under its global budget,
    so every global check passes. On iwildcam this is not hypothetical: 7 of 14
    per-group ceilings are ZERO, and a zero ceiling binds regardless of sum
    slack.
    """
    np = pytest.importorskip("numpy")
    from src.utils.posthoc_adjustment import targeted_correction
    from src.utils.constants import UNLIMITED

    n_cls, cap_cls = 4, 1
    # Two groups. Group 0 may emit 2; group 1 may emit ZERO. The GLOBAL budget
    # is 20, far above the 12 the model wants, so no global check can fire.
    groups = np.array([0] * 10 + [1] * 10)
    proba = np.full((20, n_cls), 0.02)
    proba[:, cap_cls] = 0.94
    proba = proba / proba.sum(axis=1, keepdims=True)
    glob = {c: (20 if c == cap_cls else UNLIMITED) for c in range(n_cls)}
    local = {0: [2 if c == cap_cls else UNLIMITED for c in range(n_cls)],
             1: [0 if c == cap_cls else UNLIMITED for c in range(n_cls)]}

    y_pred, _, _ = targeted_correction(
        proba, groups, glob, local, [cap_cls], force_exact=True)
    g0 = int((y_pred[groups == 0] == cap_cls).sum())
    g1 = int((y_pred[groups == 1] == cap_cls).sum())
    total = g0 + g1
    bad = []
    if g0 > 2:
        bad.append("group 0 emitted %d against a ceiling of 2" % g0)
    if g1 > 0:
        bad.append("group 1 emitted %d against a ceiling of ZERO -- a zero "
                   "ceiling binds regardless of global slack, and half of "
                   "iwildcam's per-group ceilings are zero" % g1)
    if total > 20:
        bad.append("CONTROL: the global budget of 20 was itself exceeded "
                   "(%d), so this fixture is not testing the local scope in "
                   "isolation" % total)
    assert not bad, "local scope:\n  " + "\n  ".join(bad)


# ==========================================================================
# 5. THE INSTRUMENTS THEMSELVES
# ==========================================================================

def test_every_script_that_offers_a_self_test_actually_PASSES_it():
    """2026-08-25 "the out-of-tree guard refused unconditionally on a first
    launch" -- a guard that can never pass is not a guard.

    SUBJECTS = 43 modules under `scripts/` and `configs/` carry `--self-test`.
    Each is the only thing standing between that tool and a silently wrong
    number, and NOTHING runs them together: they are invoked by hand, one at a
    time, when someone remembers. On 2026-09-02 a broken self-test fixture in
    `deployed_h2h` survived precisely because there was no sweep.

    This is the sweep. It also enforces the discovery half: a module that
    advertises `--self-test` in its argparse must actually implement it.

    !! THE COUNT IN THIS DOCSTRING IS NOW ASSERTED, BECAUSE IT ROTTED.
    It read "Twenty-two" until 2026-09-10 while the real number was 42, and
    CLAUDE.md repeated it. A stale subject count tells a reader the sweep is
    narrower than it is, which is the same failure as a stale test count
    telling them their checkout is incomplete. Bump SUBJECTS deliberately when
    a module gains or loses `--self-test`; do not silently widen the >= 20
    floor below, which only catches the sweep losing its subjects wholesale.
    """
    mods = []
    for r in ("scripts", "configs"):
        for f in sorted(os.listdir(rel(r))):
            if not f.endswith(".py"):
                continue
            if '"--self-test"' in read(r, f):
                mods.append("%s.%s" % (r, f[:-3]))
    assert len(mods) >= 20, (
        "only %d module(s) advertise --self-test; the sweep has lost its "
        "subjects" % len(mods))
    import re as _re
    claimed = int(_re.search(
        r"SUBJECTS = (\d+) modules",
        test_every_script_that_offers_a_self_test_actually_PASSES_it.__doc__
    ).group(1))
    assert claimed == len(mods), (
        "this docstring claims %d self-test modules and there are %d. It read "
        "'Twenty-two' for weeks against a real 42 (2026-09-10). Update the "
        "docstring AND CLAUDE.md's sweep line together." % (claimed, len(mods)))
    failed = []
    for m in mods:
        p = subprocess.run([sys.executable, "-m", m, "--self-test"],
                           cwd=REPO, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=300)
        if p.returncode != 0:
            tail = [ln for ln in ((p.stdout or "") + (p.stderr or "")).splitlines()
                    if "FAIL" in ln or "Error" in ln][:2]
            failed.append("%s (rc=%d) %s" % (m, p.returncode, " | ".join(tail)))
    assert not failed, (
        "%d of %d self-tests FAIL:\n  %s"
        % (len(failed), len(mods), "\n  ".join(failed)))


# ---------------------------------------------------------------------------
# BATCH 2 -- house rules and defect CLASSES, mined 2026-09-02 from the archived
# audits, the AAAI-era index and the commit history. Batch 1 gated things that
# were REMOVED (a backbone, a dataset, a config key); these gate things that
# are still present and can silently rot.
# ---------------------------------------------------------------------------

# Prose `---` per manuscript, counted 2026-09-02 by `_em_dashes` below.
#
# A RATCHET, not a zero. The house rule "NEVER `---` em-dashes" is recorded in
# the AAAI-era index and has been restated since; the 2026-07-31 audit counted
# 20 across the manuscripts and nothing gated it, so it grew. Rewriting 47
# places in the paper of record is a separate editorial job, and `main.tex` is
# the professor's file and MUST NOT be edited at all -- so the enforceable
# property is that the number never goes UP.
#
# It also fails when a count goes DOWN, with a message saying to lower the
# baseline. A ratchet that is not re-tightened stops being one.
EM_DASH_BASELINE = {
    "main.tex": 23,                  # the professor's file -- never edit
    "main_edited_by_roei.tex": 47,   # the paper of record
    "main_rev.tex": 83,
    "main_clean.tex": 74,
}
# `main_old.tex` (pre-TMLR history, baseline 23) was DELETED 2026-09-02 in the
# stale-docs sweep: nothing read it and CLAUDE.md listed it as not live. Git
# history is its archive. Removing the row is the deliberate declaration the
# `missing` branch of this test exists to force.

# The live manuscripts, per CLAUDE.md. The other three are snapshots: a fix
# applied to them has no effect on anything anyone reads.
LIVE_TEX = ("main.tex", "main_edited_by_roei.tex")


def _em_dashes(text):
    """Count PROSE `---`, excluding comments and rule separators.

    A LaTeX comment line (`% ------------`) is a section divider, not an
    em-dash, and counting those made an earlier pass report every file as
    hopeless. A run of four or more hyphens is likewise a rule.
    """
    import re
    n = 0
    for line in text.splitlines():
        if re.match(r"^\s*%", line):
            continue
        body = line.split("%")[0]
        n += len(re.findall(r"(?<!-)---(?!-)", body))
    return n


def test_the_em_dash_house_rule_does_not_get_WORSE():
    """The oldest standing house rule in the project, ungated until now.

    "NEVER `---` em-dashes" is recorded in the AAAI-era house rules and was
    restated after the TMLR pivot. The 2026-07-31 audit (FINDING 7) counted 20
    occurrences and filed it as an open note; by 2026-09-02 the paper of record
    alone carried 47. Nothing checked, so it drifted for five weeks.

    This is a RATCHET with recorded baselines rather than an assertion of zero,
    because `main.tex` is the professor's file and must never be edited, and
    rewriting 47 sites in the paper of record is an editorial task, not a test
    fixture. New prose cannot add to the count.
    """
    up, down, missing = [], [], []
    for name, base in sorted(EM_DASH_BASELINE.items()):
        path = os.path.join("docs", "paper", name)
        if not os.path.exists(rel(path)):
            missing.append(name)
            continue
        n = _em_dashes(read(path))
        if n > base:
            up.append("%s: %d, was %d (+%d)" % (name, n, base, n - base))
        elif n < base:
            down.append("%s: %d, baseline still says %d" % (name, n, base))
    assert not missing, (
        "manuscript(s) gone, so the ratchet cannot hold: %s. If a file was "
        "renamed or deleted on purpose, update EM_DASH_BASELINE."
        % ", ".join(missing))
    assert not up, (
        "prose `---` INCREASED, against the standing house rule:\n  %s\n"
        "Use an en-dash, a comma or a full stop. This is the check that did "
        "not exist between 2026-07-31 (20 occurrences) and 2026-09-02 (47 in "
        "the paper of record alone)." % "\n  ".join(up))
    assert not down, (
        "prose `---` DECREASED -- good, now tighten the ratchet by lowering "
        "the baseline:\n  %s" % "\n  ".join(down))


def test_one_method_has_ONE_name_in_the_live_manuscripts():
    """The LP allocator has been called three things (2026-07 -> 2026-09).

    `danits_lp` is the code key, "Danits-LP" was the original paper name,
    it was renamed to "Shifman-LP", and the live manuscripts now say "LP-LG".
    The 2026-07-31 audit item 7b found "Shifman" still in a paper file while
    the body and tables had moved to LP-LG; a reader hitting both takes them
    for two baselines.

    Gates the two LIVE manuscripts only. `HANDOFF_TRACK_B.tex` is a historical
    handoff and keeps the name it was written with -- rewriting a record of
    what was said at the time would be a worse defect than the inconsistency.

    Also ties the display name to the implementation: if the arm key
    disappears, the paper is naming a method with nothing behind it.
    """
    stale = []
    for name in LIVE_TEX:
        text = read("docs", "paper", name)
        for old in ("Shifman", "Danits"):
            if old in text:
                stale.append("%s still says %s; the live name is LP-LG"
                             % (name, old))
        if "LP-LG" not in text:
            stale.append("%s never says LP-LG, so the rename did not land"
                         % name)
    arms = yml("configs", "protocol.yml").get("arms", {})
    # `danits_lp` is the METHODOLOGY key and the package name; `lp` is the arm
    # that selects it. Three names for one thing, which is the whole point of
    # this test -- so check the link, not either name alone.
    lp_arms = sorted(a for a, v in arms.items()
                     if v.get("methodology") == "danits_lp")
    assert lp_arms, (
        "the manuscripts name LP-LG but no declared arm selects the "
        "`danits_lp` methodology, so the paper describes a method this tree "
        "cannot run. Declared arms: %s" % ", ".join(sorted(arms)))
    assert not stale, "one method, three names:\n  " + "\n  ".join(stale)


def test_a_dataset_whose_GROUPS_ARE_AN_INDEX_cannot_carry_a_local_constraint():
    """Why octmnist and tissuemnist could never have tested the thesis.

    Found 2026-08-28 while screening candidate datasets: both built their
    `synth_group` as `np.arange(len(y)) % 3`, so every group is an i.i.d. draw
    from one distribution and the LOCAL scope is empty BY CONSTRUCTION. Two of
    the original three datasets could not answer the question they were run to
    answer, and `rxrx1` fails the same way for a subtler reason -- every siRNA
    appears in every experiment by design.

    The general rule, and the expensive half of the lesson: a dataset famous
    for DOMAIN SHIFT is not automatically one with PER-GROUP LABEL SHIFT, and
    only the second is usable here.

    Both directions, because a screen that cannot say YES is not a screen:
    an index grouping must read as noise, and a real held-out-group shift must
    read as far above it.
    """
    import tempfile
    import pandas as pd
    from scripts.dataset_screen import _synthetic, novelty_items

    out = {}
    for kind in ("dead", "live"):
        d = _synthetic(os.path.join(tempfile.mkdtemp(), kind), kind)
        tr = pd.read_csv(os.path.join(d, "train_meta.csv"))
        te = pd.read_csv(os.path.join(d, "test_meta.csv"))
        out[kind] = novelty_items(tr, te, "location", n_null=120, seed=0)

    # The bar is `excess < 0`, not `z < 3`, and the difference is the whole
    # point. Measured on this fixture: the RAW deviation is 36.5 items and the
    # simulated sampling-noise null is 87.4, so an index grouping manufactures
    # FEWER apparent novel items than pure binomial noise does. A gate written
    # on z alone passes even when the null subtraction is deleted -- it reads
    # +1.2 -- which is exactly the "62x the seed noise" error the screen's own
    # docstring records, and it would let it back in.
    d = out["dead"]
    assert d["net_raw"] < d["net_null"], (
        "an INDEX grouping (`i %% n`) produced a raw deviation of %.0f items "
        "against a sampling-noise null of %.0f. It must not even reach the "
        "null: every group is an i.i.d. draw from one distribution."
        % (d["net_raw"], d["net_null"]))
    assert d["net_items"] < 0 and d["net_z"] < 3.0, (
        "an INDEX grouping scored %+.0f items, z=%.1f -- it must come out at "
        "or below zero once the sampling-noise null is subtracted. Either that "
        "subtraction is gone (the raw deviation IS a large positive number, "
        "which is how dermmnist was once scored at 62x the seed noise), or the "
        "synthetic 'dead' fixture stopped being i.i.d. This is the check that "
        "would have saved octmnist and tissuemnist."
        % (d["net_items"], d["net_z"]))
    assert out["live"]["net_z"] > 6.0, (
        "LIVENESS: a real per-group label shift with groups held out entire "
        "scored only z=%.1f, %.0f items. A screen that cannot detect the "
        "iwildcam shape would reject every candidate dataset, which is not a "
        "null -- it is a broken instrument."
        % (out["live"]["net_z"], out["live"]["net_items"]))


def _tiny_cache_config():
    return {
        "model_name": "MobileNetV3",
        "hyperparams": {"dropout": 0.2},
        "code_version": "abc123",
        "run_code_version": "abc123",
        "data_fingerprint": "fp-1",
    }


@pytest.mark.parametrize("cache_regime,run_regime,reused", [
    ("torch.bfloat16|scaler=False", "torch.bfloat16|scaler=False", True),
    ("torch.bfloat16|scaler=False", "torch.float16|scaler=True", False),
])
def test_a_cached_warm_up_never_crosses_the_AMP_regime(
        tmp_path, monkeypatch, cache_regime, run_regime, reused):
    """dsisco01 and dsisco02 share ONE NFS home and ONE model cache (2026-09).

    dsisco01 is FP16 + GradScaler, dsisco02 is BF16. The FP16 path SKIPS an
    overflowing optimizer step and the BF16 path does not, so the same config
    takes a different number of steps on the two servers -- the weights are
    not the same warm-up. `base_model_id` hashes hyperparameters, not the
    runtime, so without this check the second host silently loads the first
    host's model and the campaign becomes a regime mix that `check_parity`
    gate 4c cannot see (it reads each RUN's recorded runtime, never the
    cache's).

    Both directions: a matching regime MUST still be reused, or the fix would
    just be "retrain everything", which is not a fix.
    """
    import torch
    from src.models import get_model
    import src.training.model_cache as mc

    monkeypatch.setenv("OPTLOSS_MODEL_CACHE", str(tmp_path))
    cfg = _tiny_cache_config()

    monkeypatch.setattr(mc, "_amp_regime", lambda: cache_regime)
    model = get_model(cfg["model_name"], n_classes=8,
                      dropout=cfg["hyperparams"]["dropout"], pretrained=False)
    mc.save_to_cache(model, "id-amp", cfg)

    monkeypatch.setattr(mc, "_amp_regime", lambda: run_regime)
    got = mc.load_from_cache("id-amp", cfg, 8, torch.device("cpu"))

    if reused:
        assert got is not None, (
            "LIVENESS: the cache refused a warm-up from its OWN regime (%s). "
            "That retrains every model on disk and the check is worthless."
            % run_regime)
    else:
        assert got is None, (
            "a warm-up trained under %s was handed to a run under %s. One "
            "host skips overflowing optimizer steps and the other does not, "
            "so these are two different models sharing one cache key."
            % (cache_regime, run_regime))


@pytest.mark.parametrize("cache_regime,run_regime,why", [
    (None, "torch.float16|scaler=True", "the CACHE predates the field"),
    ("torch.bfloat16|scaler=False", None, "THIS PROCESS cannot determine its"),
])
def test_a_cache_check_that_cannot_RUN_says_so_instead_of_passing(
        tmp_path, monkeypatch, caplog, cache_regime, run_regime, why):
    """The and-chained-guard defect class, found and fixed 2026-09-02.

    The AMP guard was written `if want and got and got != want:`, which is
    and-chained on its OWN inputs: whenever either side was missing -- a cache
    predating the field, or `_amp_regime()` hitting its bare `except` and
    returning None -- the comparison did not happen AT ALL, with no message.
    The run then reused a cross-host warm-up and every downstream artefact
    looked clean.

    The same shape was in the `data_fingerprint` guard beside it. Two of three;
    the third (`run_code_version`) already degraded explicitly and logged, which
    is what made the other two visible.

    The lesson is NOT "invalidate when unsure" -- refusing every cache whose
    regime is unknown would retrain the whole cache in exactly the environment
    least able to tell whether that was needed. It is that a check which cannot
    run must SAY it did not run. Silence and a pass are indistinguishable.
    """
    import logging
    import torch
    from src.models import get_model
    import src.training.model_cache as mc

    monkeypatch.setenv("OPTLOSS_MODEL_CACHE", str(tmp_path))
    cfg = _tiny_cache_config()

    monkeypatch.setattr(mc, "_amp_regime", lambda: cache_regime)
    model = get_model(cfg["model_name"], n_classes=8,
                      dropout=cfg["hyperparams"]["dropout"], pretrained=False)
    mc.save_to_cache(model, "id-skip", cfg)

    monkeypatch.setattr(mc, "_amp_regime", lambda: run_regime)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="src.training.model_cache"):
        got = mc.load_from_cache("id-skip", cfg, 8, torch.device("cpu"))

    assert got is not None, (
        "an UNKNOWN regime invalidated the cache. That is the over-correction "
        "the fix explicitly avoids: it would retrain every warm-up on disk.")
    said = [r.getMessage() for r in caplog.records
            if "AMP" in r.getMessage() or "amp_regime" in r.getMessage()]
    assert said, (
        "%s AMP regime, so the FP16-vs-BF16 check could not run -- and the "
        "cache was reused with NO message. A guard and-chained on its own "
        "inputs skips silently; it must announce that it did not run.\n"
        "logged instead: %s"
        % (why, [r.getMessage() for r in caplog.records] or "nothing"))


# ---------------------------------------------------------------------------
# BATCH 3 -- mined 2026-09-02 from the COMMIT HISTORY (1,009 commits, 174 of
# them defect fixes). Batch 1 gated what was removed and batch 2 what can rot;
# these three are settings whose whole value is that they are EXACTLY right,
# and each was arrived at by discarding a plausible near-miss.
# ---------------------------------------------------------------------------


def test_deterministic_algorithms_is_STRICT_because_warn_only_takes_the_other_branch():
    """The 21x noise floor, and why the obvious setting did not fix it
    (2026-08-20, commit 5836d9ba).

    Three IDENTICAL runs -- same arm, seed, config, GPU, back to back -- spread
    0.0358 macro-F1 against a 0.0017 headline effect. 21x. Measured WITH
    `cudnn.deterministic`, `benchmark=False` and `CUBLAS_WORKSPACE_CONFIG`
    already set, so none of those was the answer. Bisecting stage by stage:
    model init identical, batch order identical, forward loss at step 0
    identical, GRADIENTS at step 0 different in all four processes. The fused
    SDPA attention backward.

    THE TRAP, and it is the reason this test exists rather than a comment:
    `warn_only=True` is NOT a softer version of the setting. PyTorch reads
    `deterministicAlgorithmsWarnOnly()` INSIDE the attention backward and takes
    the nondeterministic branch when it is true. So the flag reads as enabled,
    logs nothing, and the floor stays. Flipping it to False gives one hash
    across four processes at a 5.5% cost (54.70s -> 57.72s per 126 steps);
    disabling the fused kernels instead costs 62.97s.

    Every measurement in this project is priced against a noise floor. If this
    silently reverts, the floor returns to 21x the effect and nothing else in
    the suite would notice.
    """
    import torch
    from src.pipeline.setup import seed_all, runtime_provenance

    src = read("src", "pipeline", "setup.py")
    assert "warn_only=False" in src, (
        "`use_deterministic_algorithms` is no longer called with "
        "warn_only=False. warn_only=True is not a milder setting -- PyTorch "
        "takes the NONDETERMINISTIC branch in the attention backward when it "
        "is true, which is the 0.0358 macro-F1 floor against a 0.0017 effect.")

    # Strict mode makes an op with no deterministic implementation RAISE, so
    # it must not leak out of this test into the rest of the suite.
    was_det = torch.are_deterministic_algorithms_enabled()
    was_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    was_cudnn = torch.backends.cudnn.deterministic
    try:
        seed_all(1)
        assert torch.are_deterministic_algorithms_enabled(), (
            "seed_all left deterministic algorithms OFF")
        assert not torch.is_deterministic_algorithms_warn_only_enabled(), (
            "deterministic algorithms are in WARN-ONLY mode, which is the "
            "nondeterministic branch, not a strict one")
        assert torch.backends.cudnn.deterministic is True, (
            "cudnn.deterministic is off; it is not sufficient on its own but "
            "it is still part of the regime that was measured")
    finally:
        torch.use_deterministic_algorithms(was_det, warn_only=was_warn)
        torch.backends.cudnn.deterministic = was_cudnn

    # `seed_all(None)` would skip all seven settings silently while the run
    # still writes `seed_N/` in its path, so it must raise rather than return.
    with pytest.raises(ValueError):
        seed_all(None)

    prov = runtime_provenance(torch.device("cpu"))
    for key in ("deterministic", "deterministic_warn_only",
                "cudnn_deterministic", "cublas_workspace_config"):
        assert key in prov, (
            "runtime_provenance no longer records `%s`. The runs that first "
            "showed the 21x floor could not say which determinism regime "
            "produced them, which is why this is recorded per run." % key)


def test_CUBLAS_WORKSPACE_CONFIG_is_set_BEFORE_torch_is_imported():
    """An env var that is read once, at CUDA init (2026-08-20, and it is a
    PLACEMENT property, not a presence one).

    `torch.use_deterministic_algorithms(True)` raises on every cuBLAS matmul
    unless CUBLAS_WORKSPACE_CONFIG is set, and cuBLAS reads it when the handle
    is created -- so setting it after `import torch` has already initialised
    CUDA is a no-op that still looks correct in `os.environ`. Both entry points
    set it at module top, above the torch import, and a reorder-safe import
    sorter or a routine tidy-up would silently break it.

    Line numbers via AST, because `import torch` also appears inside functions
    and in comments.
    """
    bad = []
    for path in ("main.py", os.path.join("src", "experiments", "runner.py")):
        tree = ast.parse(read(path))
        env_line = torch_line = None
        for node in ast.walk(tree):
            if (env_line is None and isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "setdefault"
                    and any(isinstance(a, ast.Constant)
                            and a.value == "CUBLAS_WORKSPACE_CONFIG"
                            for a in node.args)):
                env_line = node.lineno
            if torch_line is None and isinstance(node, ast.Import):
                if any(a.name == "torch" or a.name.startswith("torch.")
                       for a in node.names):
                    torch_line = node.lineno
        if env_line is None:
            bad.append("%s never sets CUBLAS_WORKSPACE_CONFIG; "
                       "use_deterministic_algorithms then raises on every "
                       "cuBLAS matmul" % path)
        elif torch_line is not None and env_line > torch_line:
            bad.append("%s sets CUBLAS_WORKSPACE_CONFIG at line %d, AFTER "
                       "`import torch` at line %d. cuBLAS reads it when the "
                       "handle is created, so this is a silent no-op"
                       % (path, env_line, torch_line))
    assert not bad, "\n  ".join([""] + bad)


# (backbone, the modules its torchvision head must STILL contain after the
# builder has touched it). MobileNetV3 is the only one whose pretrained head
# carries a projection worth keeping; the other three end in a bare Linear, so
# rebuilding their head throws nothing away.
HEAD_SHAPE = {
    "MobileNetV3": {"projection": (960, 1280), "activation": "Hardswish"},
    "MobileNetV2": {"projection": None, "activation": None},
    "RegNetY400MF": {"projection": None, "activation": None},
    "ViTB16": {"projection": None, "activation": None},
}


@pytest.mark.parametrize("backbone", sorted(HEAD_SHAPE))
def test_a_backbone_replaces_ONLY_its_final_layer_and_keeps_ONE_dropout(backbone):
    """MobileNetV3 threw away its pretrained projection, biasing the HEADLINE
    (2026-08-19, commit 05097fcb).

    MobileNetV2, RegNetY400MF and ViTB16 keep the pretrained backbone and
    replace only the final layer. MobileNetV3 rebuilt its ENTIRE classifier --
    including the 960->1280 projection -- from random, to avoid the original
    head's double dropout.

    That is worse than a fairness gap between backbones. The projection is
    trained during warm-up ONLY, and the protocol gives trained arms ONE
    warm-up epoch against the post-hoc arms' thirty. So on the headline
    backbone the trained arms began from a materially worse model than the
    baseline they are measured against -- a bias pointing straight at the
    comparison the paper makes.

    The double dropout is avoided by setting the EXISTING Dropout's p, not by
    adding a second one. Checked with pretrained=False, because the structure
    is torchvision's either way: a rebuilt head has no Hardswish and no
    960->1280 Linear, so this distinguishes them without downloading weights.
    """
    import torch.nn as nn
    from src.models import get_model

    p = 0.3
    model = get_model(backbone, n_classes=8, dropout=p, pretrained=False)

    heads = [m for name, m in model.named_modules()
             if name.endswith(("classifier", "heads", "fc"))]
    assert heads, "%s exposes no recognisable head" % backbone
    head = heads[-1]
    layers = list(head.modules())

    drops = [m for m in layers if isinstance(m, nn.Dropout)]
    assert len(drops) == 1, (
        "%s has %d Dropout layers in its head, not 1. The double dropout is "
        "the defect the MobileNetV3 rebuild was introduced to avoid, and "
        "rebuilding was a worse cure than the disease."
        % (backbone, len(drops)))
    assert abs(drops[0].p - p) < 1e-9, (
        "%s ignored the configured dropout p=%.2f and kept %.2f -- the fix is "
        "to SET the existing Dropout's p, not to add another one"
        % (backbone, p, drops[0].p))

    linears = [m for m in layers if isinstance(m, nn.Linear)]
    assert linears, "%s head has no Linear" % backbone
    assert linears[-1].out_features == 8, (
        "%s final layer emits %d classes, not 8"
        % (backbone, linears[-1].out_features))

    want = HEAD_SHAPE[backbone]
    if want["projection"]:
        a, b = want["projection"]
        kept = [m for m in linears
                if (m.in_features, m.out_features) == (a, b)]
        assert kept, (
            "%s no longer keeps its pretrained %d->%d projection. Rebuilding "
            "the whole classifier discards a layer that only warm-up trains, "
            "and trained arms get ONE warm-up epoch against the post-hoc "
            "arms' thirty -- so the loss lands entirely on the treated side "
            "of the headline comparison. Mutate the head in place: set the "
            "existing Dropout's p and replace head[-1] only."
            % (backbone, a, b))
        assert any(type(m).__name__ == want["activation"] for m in layers), (
            "%s head has no %s, so it is not torchvision's head any more -- "
            "it was rebuilt" % (backbone, want["activation"]))


def test_normalize_mode_discards_the_gradient_MAGNITUDE_in_both_directions():
    """`normalize` is why two dual arms collapse into one (2026-09-02).

    `finish_constraint_step(mode="normalize")` must deliver a step of norm
    EXACTLY `clip`, scaling UP when the raw norm is below the bound and down
    when it is above. That is the whole reason `fioretto_alm` and
    `fioretto_ldf` are not two baselines on this corpus: at any fixed model
    state both build a weight vector proportional to `relu(S_j - K_j)`, so they
    differ only in a scalar, and a mode that deletes the scalar deletes the
    difference. Measured at cos = 1.0000 in 192 of 192 stored states
    (`scripts/dual_cone_probe.py`), and the deployed contrast
    `|alm - fioretto|` sits at 0.83x the RNG floor. FRAMEWORK 2(z28).

    The NEGATIVE CONTROL is `mode="clip"`, which must NOT scale a small
    gradient up. If both modes normalised, the test would pass while proving
    nothing.
    """
    import torch
    from src.training.constraint_step import finish_constraint_step

    def deliver(mode, scale):
        m = torch.nn.Linear(4, 3, bias=False)
        opt = torch.optim.SGD(m.parameters(), lr=0.0)   # lr=0: measure, do not move
        m.weight.grad = torch.full_like(m.weight, scale)
        finish_constraint_step(m, opt, None, clip=1.0, mode=mode, fp32=True)
        return float(torch.linalg.vector_norm(m.weight.grad))

    small, big = 1e-4, 10.0
    assert abs(deliver("normalize", small) - 1.0) < 1e-5, (
        "mode=normalize did not scale a SMALL constraint gradient up to clip. "
        "Then the delivered dose still depends on each arm's own scale, and "
        "the 20x hounie/fioretto dose gap that normalize exists to remove is "
        "back. FRAMEWORK 2(z28).")
    assert abs(deliver("normalize", big) - 1.0) < 1e-5, (
        "mode=normalize did not scale a LARGE constraint gradient down to clip")

    assert deliver("clip", small) < 0.5, (
        "NEGATIVE CONTROL FAILED: mode=clip scaled a small gradient UP. clip "
        "must only shrink, or it is normalize under another name and the "
        "corpus's two modes are one mode.")
    assert abs(deliver("clip", big) - 1.0) < 1e-5, (
        "mode=clip did not cap a large gradient at clip")


def test_the_two_fioretto_arms_build_PROPORTIONAL_weights_at_a_fixed_state():
    """ALM's dual rule is LDF's rule times a scalar (2026-09-02).

    LDF accumulates `lambda_j = T * step * relu(r_j)`. ALM accumulates
    `lambda_j = T * eta * relu(r_j)` and adds `mu_T * relu(r_j)`. At a FIXED
    model state -- where `r_j` does not move -- both are a scalar times
    `relu(r_j)`, so their constraint gradients are the same ray and
    `constraint_grad_mode: normalize` makes them the same step. The paper
    claims them as two independent duals; on this corpus they are one.
    FRAMEWORK 2(z28).

    Gated here rather than only in the probe because the probe is not run in
    CI, and the claim is about the SHIPPED update rules: if either arm's rule
    changes, this must go red.
    """
    import numpy as np
    from scripts.dual_cone_probe import DEFAULT_CFG, arm_weights

    soft = np.array([420.0, 60.0, 310.0, 15.0])
    hard = np.array([400.0, 55.0, 300.0, 10.0])
    K = np.array([100.0, 40.0, 0.0, 30.0])       # one slack scope, one K=0
    sizes = np.array([2943.0, 400.0, 400.0, 220.0])

    W = arm_weights(soft, hard, K, sizes, 29, DEFAULT_CFG)
    a, b = W["fioretto_alm"][-1], W["fioretto_ldf"][-1]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos > 1 - 1e-12, (
        "fioretto_alm and fioretto_ldf no longer build proportional weights "
        "(cos=%.12f). Either a dual rule changed -- in which case they are now "
        "two baselines and 2(z28) must be revised -- or this test's model of "
        "one of them drifted from src/methodologies/." % cos)

    # NEGATIVE CONTROL: tralo must NOT be proportional to them, or the check
    # is measuring a bug in arm_weights rather than a property of the rules.
    t = W["tralo"][-1]
    cos_t = float(t @ b / (np.linalg.norm(t) * np.linalg.norm(b)))
    assert cos_t < 0.99, (
        "NEGATIVE CONTROL FAILED: tralo's weights are proportional to "
        "fioretto's too (cos=%.6f), so arm_weights is collapsing every arm "
        "and the cos=1.0 above proves nothing." % cos_t)


def test_a_uniform_step_in_LOG_ODDS_is_not_a_bias_shift_and_can_reorder():
    """`uniform_grad_count`'s founding claim, refuted in LOGIT space (2026-09-02).

    The docstring used to argue: `u_c = z_c - log sum_{k!=c} e^{z_k}` gives
    `du_c/dz_c = 1` exactly, therefore a uniform step in u is a pure bias shift
    on the class logit, therefore it cannot reorder.

    The identity is true and the conclusion does not follow. `du_c/dz_c = 1` is
    only the DIAGONAL. The off-diagonal is `du_c/dz_j = -p_j/(1-p_c)`, which is
    nonzero and varies per item, so a step of equal size in every item's u still
    moves the other logits by item-dependent amounts.

    `scripts/bias_shift_probe.py` already refutes the claim in PARAMETER space.
    This is the same refutation one level earlier, and it needs no model at all
    -- which is why it belongs in the catalogue: the parameter-space argument
    can be argued about, this one cannot.
    """
    import torch

    torch.manual_seed(0)
    z = torch.randn(64, 8, dtype=torch.float64, requires_grad=True)
    p = torch.softmax(z, dim=1)
    u = torch.log(p) - torch.log1p(-p)

    J = torch.zeros(64, 8, 8, dtype=torch.float64)
    for c in range(8):
        J[:, c, :] = torch.autograd.grad(u[:, c].sum(), z, retain_graph=True)[0]

    diag = torch.diagonal(J, dim1=1, dim2=2)
    assert float((diag - 1).abs().max()) < 1e-10, (
        "du_c/dz_c is no longer exactly 1, so the log-odds coordinate is not "
        "doing what uniform_grad_count assumes")

    off = J.clone()
    for c in range(8):
        off[:, c, c] = 0.0
    pd = p.detach()
    want = torch.zeros_like(off)
    for c in range(8):
        for j in range(8):
            if j != c:
                want[:, c, j] = -pd[:, j] / (1 - pd[:, c])
    assert float((off - want).abs().max()) < 1e-9, (
        "du_c/dz_j is no longer -p_j/(1-p_c); the algebra this lesson rests on "
        "has changed")

    assert float(off.abs().max()) > 1e-2, (
        "the off-diagonals of du/dz came out ZERO. If that were true a uniform "
        "step in u WOULD be a pure bias shift and uniform_grad_count's "
        "original claim would stand. It is not true; this assertion exists so "
        "the claim cannot quietly return to the docstring.")

    per_item = off.abs().amax(dim=(1, 2))
    assert float(per_item.max() - per_item.min()) > 1e-2, (
        "the off-diagonal coupling is CONSTANT across items, which would make "
        "it a shift after all. Measured spread was 0.30 on 2026-09-02.")


def test_a_quadrature_sd_is_within_sqrt2_of_the_truth_in_EITHER_direction():
    """The gloss that discounted this repo's own power (2026-09-03).

    `scripts/paper_rows.py` builds `sd = sqrt(sa^2 + sb^2)`, the rho = 0
    quadrature, and its comment told every reader the true noise ran "6-12x"
    higher so `seeds_needed` was a LOWER BOUND. That is impossible. For any
    correlation

        sd(A - B) = sqrt(sa^2 + sb^2 - 2 rho sa sb) <= sa + sb
                  <= sqrt(2) sqrt(sa^2 + sb^2)

    so the worst under-statement is 41%, and positive correlation -- which is
    what sharing a warm-up produces -- makes it an OVER-statement instead. The
    6-12x came from FRAMEWORK 2(v), which compares the paired difference sd to
    ONE ARM's sd, a quantity the quadrature already contains in `sa`.

    Empirically too: sd(treated)/sd(null) over 73 cells of the clean corpus is
    median 0.78, range 0.13-2.65, ZERO above 6x.

    NEGATIVE CONTROL in the same test: the bound must be TIGHT, i.e. actually
    reached at rho = -1, or it is a vacuous inequality that proves nothing.
    """
    import math
    import random

    rng = random.Random(20260903)
    worst = 0.0
    for _ in range(20000):
        sa = rng.uniform(0.01, 10.0)
        sb = rng.uniform(0.01, 10.0)
        rho = rng.uniform(-1.0, 1.0)
        var = sa * sa + sb * sb - 2.0 * rho * sa * sb
        true = math.sqrt(max(0.0, var))
        quad = math.sqrt(sa * sa + sb * sb)
        worst = max(worst, true / quad)
    assert worst <= math.sqrt(2.0) + 1e-9, (
        "found a (sa, sb, rho) where the true paired sd exceeds the quadrature "
        "by %.4f > sqrt(2). If that were possible the '6-12x lower bound' "
        "gloss could stand." % worst)

    # NEGATIVE CONTROL: the bound is REACHED, so it is not vacuous. rho = -1
    # with sa == sb gives exactly sqrt(2).
    sa = sb = 3.0
    tight = math.sqrt(sa * sa + sb * sb + 2.0 * sa * sb) / math.sqrt(
        sa * sa + sb * sb)
    assert abs(tight - math.sqrt(2.0)) < 1e-12, (
        "the sqrt(2) bound is not attained (%.6f), so this test would pass "
        "for a bound that is merely loose" % tight)

    # and a claim of 6x must be REJECTED by the same arithmetic
    assert worst < 6.0, "a 6x underestimate was reachable, which it must not be"


def test_the_resolved_bar_is_a_t_of_4_on_df_3_so_survivors_need_a_chance_rate():
    """Why "1 of 158 rows resolves" is not a finding (2026-09-03).

    `paper_rows.py` writes `resolved = abs(d) >= 2.0 * sd`, where `d` is a
    difference of n-seed MEANS and `sd` is a PER-SEED sd. Those are different
    scales: in t units the bar is

        t = d / (sd / sqrt(n)) = 2 * d / sd >= 2 * 2 = 4     at n = 4

    on df = n - 1 = 3. P(abs(t_3) >= 4) = 0.0280, so over 158 strict-task rows
    the GLOBAL NULL already delivers ~4.4 "resolved" rows. The corpus delivers
    1. So the observed count is BELOW chance and the honest statement is that
    nothing resolves beyond it -- not that one effect survived.

    NEGATIVE CONTROL: reading the same bar on the wrong df (6) gives 1.12
    expected, which would make 1 look like the expectation rather than a
    shortfall. The two readings must differ, or the df does not matter and this
    lesson is empty.
    """
    from scipy import stats

    p3 = 2.0 * stats.t.sf(4.0, 3)
    p6 = 2.0 * stats.t.sf(4.0, 6)
    exp3 = 158 * p3
    exp6 = 158 * p6

    assert abs(p3 - 0.0280) < 5e-4, (
        "P(|t_3| >= 4) = %.4f, not the 0.0280 this lesson was written from" % p3)
    assert exp3 > 4.0, (
        "the chance expectation over 158 rows at df=3 came out %.2f; the "
        "lesson is that it EXCEEDS the 1 row observed" % exp3)
    assert exp3 > 1.0, (
        "chance expectation %.2f does not exceed the single observed survivor, "
        "so 'below chance' would be the wrong reading" % exp3)

    # NEGATIVE CONTROL: df is load-bearing. At df=6 the expectation drops to
    # near 1 and the same count would read as ordinary rather than short.
    assert exp6 < 2.0 < exp3, (
        "df did not change the verdict (df=3 -> %.2f, df=6 -> %.2f), so the "
        "n=4 design is not what makes this bar weak" % (exp3, exp6))


def test_two_cap_levels_share_a_warm_up_so_a_CELL_is_not_an_independent_unit():
    """The cache key that made the paper's headline p inadmissible (2026-09-03).

    `configs/gen_campaign.compute_base_model_id` hashes the backbone, the
    dataset identity and `protocol.yml: warmup_identity_keys`. The CAP appears
    in none of them, so `L30_G30` and `L40_G40` at the same (backbone, seed)
    load the SAME cached warm-up and differ only in the constraint epochs that
    follow. Six tight-cap cells were therefore three warm-up models, and the
    paper's six-cell sign test printed p=0.031 against a 0.5^3 = 0.125 floor.

    NEGATIVE CONTROL in the same test: `seed` MUST be in the key. If it were
    not, every seed would share one model too and the lesson would be about a
    key that distinguishes nothing.
    """
    import re

    proto = read("configs", "protocol.yml")
    m = re.search(r"^warmup_identity_keys:\s*\n((?:\s*-\s*\S+\s*\n)+)",
                  proto, re.M)
    assert m, "warmup_identity_keys not found in configs/protocol.yml"
    keys = [ln.strip().lstrip("-").strip() for ln in m.group(1).splitlines()
            if ln.strip()]

    cap_like = [k for k in keys
                if any(t in k.lower() for t in
                       ("cap", "constraint_tag", "local_pct", "global_pct",
                        "budget", "constrained_class"))]
    assert not cap_like, (
        "a cap-identifying key %r is now in warmup_identity_keys. If that is "
        "deliberate then cap levels NO LONGER share a warm-up, two cap levels "
        "become two units, and FRAMEWORK 2(z33) plus the paper's unit counts "
        "must be revisited -- do not simply delete this assertion." % cap_like)

    # NEGATIVE CONTROL: the key must still separate seeds, or it separates
    # nothing and this lesson is vacuous.
    assert "seed" in keys, (
        "`seed` is not in warmup_identity_keys, so every seed would load one "
        "cached model. Then the cap check above proves nothing, because the "
        "key would be failing to distinguish everything, not just the cap.")

    # and the source really does build the key from that list, not from hp
    gen = read("configs", "gen_campaign.py")
    assert 'for k in P["warmup_identity_keys"]' in gen, (
        "compute_base_model_id no longer reads warmup_identity_keys, so this "
        "test is checking a list nothing consumes")

    # the paper of record must not re-assert the refuted sentence
    paper = read("docs", "paper", "main_edited_by_roei.tex")
    # and it must still disclose the same-lesion leakage it was measured to
    # have. FRAMEWORK 2(o): slice_1, the split every derm result uses, has
    # 776/2003 test images (38.7%) sharing a lesion_id with a training image
    # and 150/223 melanoma (67.3%) -- melanoma being the CAPPED class. A
    # manuscript that reports DermMNIST without saying so is reporting
    # inflated absolute numbers silently.
    if "DermMNIST" in paper or "DermaMNIST" in paper:
        assert "Same-lesion leakage" in paper, (
            "the paper of record reports DermMNIST but no longer carries the "
            "same-lesion leakage disclosure. Paired comparisons survive the "
            "leak (both arms share the split); ABSOLUTE numbers do not, and "
            "67.3% of the capped class is leaked. FRAMEWORK 2(o).")

    assert "cells are the independent units" not in paper, (
        "the paper of record again says 'cells are the independent units'. "
        "The warm-up cache key says otherwise (above); the unit is the "
        "(backbone, seed) warm-up. FRAMEWORK 2(z33).")


def test_normalize_deletes_the_rival_duals_hyperparameters(tmp_path):
    """What actually separates TraLO from the rival family (2026-09-03).

    `finish_constraint_step` under mode="normalize" rescales the constraint
    gradient to exactly `constraint_grad_clip`, so only the DIRECTION of the
    weight vector is delivered. Every dual rule is built from linear maps and
    max(0, .), both POSITIVELY HOMOGENEOUS, so scaling the residual leaves
    their direction untouched. TraLO's `lambda * pen'(S)` saturates and does
    not.

    Consequences the docs now rest on:
      * `fioretto_ldf` is HYPERPARAMETER-FREE here -- its single knob
        multiplies the whole accumulation and factors out of the direction, so
        a 10,000x change delivers a bit-identical update. "The LDF baseline was
        mis-tuned" is not a possible criticism.
      * `hounie_alpha`, the parameter that IS Hounie-RCL, is direction-inert at
        a fixed state at any value.

    NEGATIVE CONTROL in the same test: TraLO's direction MUST move under the
    same rescaling, or homogeneity is not what is being measured and the whole
    argument is an artefact of the harness.
    """
    import numpy as np

    from scripts.dual_cone_probe import DEFAULT_CFG, arm_weights

    def cosang(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 1e-15 and nb > 1e-15 else 1.0

    rng = np.random.default_rng(20260903)
    n = 8
    K = np.floor(rng.uniform(20, 400, n))
    excess = rng.normal(40, 50, n)
    sizes = rng.uniform(50, 3000, n)

    def W(scale, cfg=None):
        soft = K + scale * excess
        hard = np.maximum(0.0, soft)
        return {a: V[-1] for a, V in
                arm_weights(soft, hard, K, sizes, 29, cfg or DEFAULT_CFG).items()}

    ref = W(1.0)
    homog = {}
    for c in (0.25, 4.0, 16.0):
        cur = W(c)
        for arm in ref:
            homog.setdefault(arm, []).append(cosang(cur[arm], ref[arm]))

    for arm in ("fioretto_ldf", "fioretto_alm", "hounie_rcl"):
        worst = min(homog[arm])
        assert worst > 1.0 - 1e-9, (
            "%s is no longer positively homogeneous (min cos %.9f under a "
            "residual rescale). Its rule must be built only from linear maps "
            "and max(0, .); if that changed, FRAMEWORK 2(z35) needs redoing."
            % (arm, worst))

    # NEGATIVE CONTROL: tralo MUST move, or the harness is rescaling nothing.
    assert min(homog["tralo"]) < 0.99, (
        "tralo's direction did NOT move under a 16x residual rescale (min cos "
        "%.6f). pen' is supposed to saturate; if it no longer does, TraLO has "
        "become scale-free like the duals and 2(z35)'s claimed structural "
        "difference is gone." % min(homog["tralo"]))

    # the LDF knob must be exactly inert, at a fixed state
    lo = arm_weights(K + excess, np.maximum(0.0, K + excess), K, sizes, 29,
                     dict(DEFAULT_CFG, fioretto_step_size=0.0005))
    hi = arm_weights(K + excess, np.maximum(0.0, K + excess), K, sizes, 29,
                     dict(DEFAULT_CFG, fioretto_step_size=5.0))
    c = cosang(lo["fioretto_ldf"][-1], hi["fioretto_ldf"][-1])
    assert c > 1.0 - 1e-9, (
        "a 10,000x change in fioretto_step_size moved the delivered direction "
        "(cos %.9f). It multiplies the whole accumulation and must factor out."
        % c)

    # and hounie_alpha, the parameter that IS the method
    lo = arm_weights(K + excess, np.maximum(0.0, K + excess), K, sizes, 29,
                     dict(DEFAULT_CFG, hounie_alpha=0.01))
    hi = arm_weights(K + excess, np.maximum(0.0, K + excess), K, sizes, 29,
                     dict(DEFAULT_CFG, hounie_alpha=1000.0))
    c = cosang(lo["hounie_rcl"][-1], hi["hounie_rcl"][-1])
    assert c > 1.0 - 1e-9, (
        "hounie_alpha moved the direction at a fixed state (cos %.9f), which "
        "contradicts 2(z35). Good news if real -- but check the rule first."
        % c)


def test_a_K0_ceiling_sits_past_the_penalty_peak_and_carries_almost_no_pull():
    """Half the iwildcam local constraints are inert in the OBJECTIVE (2026-09-03).

    `pen(E) = E/(E+scale) + rho*e^2/(1+e^2)`, `e = E/scale`,
    `scale = max(K, 1)`. Its derivative is NON-MONOTONE once rho >~ 1: it peaks
    at ~57.5% over budget and decays beyond. At `K == 0` the scale is pinned to
    1, so the peak lands at a SOFT COUNT of ~0.58 -- while a real iwildcam
    camera group's soft count (`sum_i p_ic` over hundreds of items) is tens to
    hundreds. Every K==0 ceiling is therefore permanently on the far decaying
    tail, and since `normalize` keeps only the direction of the SUM, those
    terms round out of the delivered update.

    CLAUDE.md's "7 of 14 ceilings are K=0, so the LOCAL scope constrains the
    output at every cap level" is true of the ALLOCATOR (it must emit nothing
    there) and false of the TRAINED objective.

    NEGATIVE CONTROL in the same test: a constraint at ~57% over budget must
    get MUCH more pull, or the shape is not the non-monotone one and the whole
    argument is void.
    """
    import torch

    from src.losses.transductive_loss import MulticlassTransductiveLoss

    def dpen(soft, K, rho):
        """Autograd through the SHIPPED penalty -- no re-derivation."""
        loss = MulticlassTransductiveLoss(
            global_constraints=[1e10] * 8, local_constraints={},
            num_classes=8, initial_rho=float(rho))
        t = torch.tensor(float(soft), dtype=torch.float64, requires_grad=True)
        loss._penalty(t, float(K)).backward()
        return float(t.grad)

    rho = 0.5 + 3.431 * 28          # end of the shipped ramp

    # the peak for a K==0 ceiling is at a soft count near 0.58, not near 0
    grid = [i * 0.01 for i in range(1, 500)]
    vals = [dpen(x, 0.0, rho) for x in grid]
    peak_at = grid[max(range(len(vals)), key=lambda i: vals[i])]
    assert 0.4 < peak_at < 0.8, (
        "the K==0 penalty peak moved to a soft count of %.3f; 2(z36) is "
        "written around ~0.58 and would need redoing" % peak_at)

    # a realistic group expects tens of items, and is then all but unconstrained
    peak = max(vals)
    for soft, bar in ((25.0, 1e-3), (100.0, 1e-4), (400.0, 1e-5)):
        frac = dpen(soft, 0.0, rho) / peak
        assert frac < bar, (
            "a K==0 group with soft count %.0f carries %.3g of the peak pull, "
            "above the %.0e this lesson records. If the shape changed, the "
            "'half the local constraints are inert' claim must be re-measured."
            % (soft, frac, bar))

    # NEGATIVE CONTROL: a barely-violated ceiling must be pulled hard, or the
    # shape is not non-monotone and nothing above means anything.
    healthy = dpen(63.0, 40.0, rho)          # 1.57x over budget
    starved = dpen(400.0, 0.0, rho)
    assert healthy / starved > 1e4, (
        "a constraint at 1.57x its budget gets only %.1fx the pull of one at "
        "400x over. The penalty is then roughly monotone in the violation and "
        "2(z36)'s starvation argument does not apply." % (healthy / starved))


def test_no_test_in_this_file_states_a_lesson_without_a_DATE():
    """The convention that makes this catalogue re-checkable (2026-09-02).

    A lesson recorded without a date cannot be traced back to the tree that
    produced it, and this project has twice re-derived a finding it had
    already written down. Every test here names the year it came from.

    It also enforces the ASCII rule on THIS file, for the reason in the module
    docstring: pytest prints these strings, and on a cp1252 console a non-ASCII
    character in a failure message raises UnicodeEncodeError mid-report.
    """
    src = read("tests", "test_lessons_learned.py")
    tree = ast.parse(src)
    bad = []
    import re
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith("test_"):
            continue
        doc = ast.get_docstring(node) or ""
        if not doc.strip():
            bad.append("%s has no docstring" % node.name)
        elif not re.search(r"\b20\d\d\b", doc):
            bad.append("%s names no date, so its lesson cannot be traced"
                       % node.name)
    try:
        src.encode("ascii")
    except UnicodeEncodeError as exc:
        bad.append("this file is not ASCII (%s); pytest prints these strings "
                   "and a cp1252 console will die mid-report" % exc)
    assert not bad, "catalogue conventions:\n  " + "\n  ".join(bad)


def test_the_dual_arms_UPDATE_THEIR_MULTIPLIERS_BEFORE_THE_PRIMAL_STEP():
    """2026-09-03, FIXED THE SAME DAY IT WAS FOUND. `dose_landed` on the live
    `vitdual1` read

        alm  29.00   tralo  29.00   fioretto  28.00   hounie  28.00

    attempted constraint steps per run, with every arm landing 100% of what it
    ATTEMPTED -- so nothing looked wrong from any other angle and only the
    denominators differed. `fioretto_ldf` and `hounie_rcl` ran the epoch as
    `CE -> counts -> violations -> PRIMAL step -> dual update`, and both
    initialise their multipliers at exactly zero, so epoch 0's primal gate
    ("is any lambda > 0") was False and no backward ran.

    A 3.4% dose gap in the only phase the comparison is about is not
    apples-to-apples, and the four-dual head-to-head was that campaign's entire
    purpose. The campaign was DISCARDED and relaunched rather than caveated.

    THE FIX IS AN ORDERING, NOT A HYPERPARAMETER. The dual block now runs
    BEFORE the primal gate: same violations (computed on the pre-step model
    either way), same step size, `lambda_0 = 0` untouched, no new knob. For
    hounie Steps 3 and 4 moved together, so Step 4 still reads the lambda that
    Step 3 just wrote (the deliberate Gauss-Seidel its own comment documents).
    Both orders of the alternating primal/dual scheme are conventional.

    THIS TEST GUARDS THE ORDERING, which is the thing a later tidy-up would
    silently undo. The BEHAVIOUR -- every trained arm attempting every
    constraint epoch -- is asserted end to end by
    `tests/gates/test_g4_grid.py::test_every_trained_arm_ATTEMPTS_every_constraint_epoch`,
    which was mutation-tested against the pre-fix code and reported
    `fioretto attempted 1 ... expected 2`.

    `alm` is deliberately NOT changed: it starts at `lambda = 0` too and always
    attempted 29, because its augmented term carries `mu * violation**2`, which
    is nonzero at `lambda = 0`. That contrast is what proved the cause was the
    MULTIPLIER and not the dual family, so it is asserted here too.
    """
    fails = []
    ARMS = {
        "fioretto_ldf": ("# ---- Step 3: subgradient dual update",
                         "has_work = ("),
        "hounie_rcl": ("# ---- Step 3: dual ascent on lambda",
                       "has_active = ("),
    }
    for arm, (dual_marker, primal_gate) in ARMS.items():
        src = io.open(rel("src", "methodologies", arm, "train.py"),
                      encoding="utf-8").read()
        if dual_marker not in src or primal_gate not in src:
            fails.append("%s: cannot find the dual block or the primal gate, "
                         "so the ordering this lesson guards is unverifiable "
                         "-- re-derive it rather than deleting this test" % arm)
            continue
        if src.index(dual_marker) > src.index(primal_gate):
            fails.append("%s updates its multipliers AFTER the primal gate "
                         "again. With lambda_0 = 0 that makes epoch 0 take no "
                         "constraint step, and this arm silently drops to 28 "
                         "of 29 against alm and tralo" % arm)

    # The premise: the init really is zero, which is WHY the ordering matters.
    # If the init ever becomes nonzero the ordering stops being load-bearing
    # and this lesson needs re-deriving rather than quietly passing.
    import yaml
    proto = yaml.safe_load(io.open(rel("configs", "protocol.yml"),
                                   encoding="utf-8").read())
    inits = _all_values(proto, "fioretto_lambda_init")
    if not inits or any(float(v) != 0.0 for v in inits):
        fails.append("`fioretto_lambda_init` is no longer uniformly 0.0 (%s). "
                     "The ordering fix was needed BECAUSE the init is zero; "
                     "re-derive this lesson" % inits)
    hounie = io.open(rel("src", "methodologies", "hounie_rcl", "train.py"),
                     encoding="utf-8").read()
    if "u_g = {c: 0.0" not in hounie:
        fails.append("hounie's multiplier no longer initialises at 0.0 -- "
                     "re-derive this lesson")

    # alm is the control that identified the cause, and it must stay unchanged:
    # nonzero force at lambda = 0 via the augmented term.
    alm = io.open(rel("src", "methodologies", "fioretto_alm", "train.py"),
                  encoding="utf-8").read()
    if "mu" not in alm:
        fails.append("fioretto_alm no longer carries its augmented term, so "
                     "the control that isolated the multiplier as the cause "
                     "is gone")

    assert not fails, "%d dual-dose defects:\n  - %s" % (
        len(fails), "\n  - ".join(fails))


def _all_values(node, key):
    """Every value stored under `key` anywhere in a nested dict/list."""
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                out.append(v)
            out.extend(_all_values(v, key))
    elif isinstance(node, list):
        for v in node:
            out.extend(_all_values(v, key))
    return out


def test_no_shipped_module_references_an_UNDEFINED_NAME():
    """LESSON, 2026-08 and again 2026-09-04: an undefined name ships silently.

    The first time, three arms went out with an undefined name in `train()`.
    They burned all 29 constraint epochs, died, were reset to `pending`, and
    the campaign came back looking merely UNFINISHED -- with `audit_config`
    and `check_parity` both green. `tests/gates/test_g4_grid.py` gates the
    SYMPTOM (a crash log beside a pending config). This gates the CAUSE, which
    is statically decidable and costs a second.

    It recurred immediately on 2026-09-04 while wiring the dead-arm filter
    into the scorers: `full_panel`, `cell_table` and `deployed_h2h` each
    called `quarantine.drop_dead_runs` with no module-level import. Every one
    of them PARSED, imported, and passed `--self-test`; the NameError fires
    only on the branch that a partially-quarantined campaign reaches, which is
    the branch that exists to prevent a wrong number. A gate that fires only
    when the guard is needed is the worst possible failure mode.

    NEGATIVE CONTROL: a module with a deliberate undefined name must be
    reported, or this test cannot see the defect it is named for.
    """
    pyflakes = pytest.importorskip("pyflakes")
    del pyflakes
    NL = chr(10)

    def undefined(path):
        r = subprocess.run([sys.executable, "-m", "pyflakes", path],
                           capture_output=True, text=True, cwd=REPO)
        return [L for L in (r.stdout + r.stderr).split(NL)
                if "undefined name" in L]

    targets = []
    for sub in ("scripts", "configs", "src"):
        for dirpath, _d, files in os.walk(rel(sub)):
            targets += [os.path.join(dirpath, f)
                        for f in files if f.endswith(".py")]
    targets.append(rel("main.py"))

    bad = []
    for path in sorted(targets):
        bad += undefined(path)

    # NEGATIVE CONTROL, in a temp file so nothing shipped is touched.
    import tempfile
    fd, ctrl = tempfile.mkstemp(suffix=".py", text=True)
    os.close(fd)
    try:
        io.open(ctrl, "w", encoding="utf-8").write(
            "def f():" + NL + "    return not_a_real_name" + NL)
        if not undefined(ctrl):
            bad.append("NEGATIVE CONTROL: pyflakes did not flag a deliberate "
                       "undefined name, so this test detects nothing")
    finally:
        os.unlink(ctrl)

    assert not bad, (
        "%d undefined name(s) in shipped code. Each is a NameError that fires "
        "only on the branch that reaches it:" + NL + "  %s"
    ) % (len(bad), (NL + "  ").join(bad))


# ---------------------------------------------------------------------------
# A SCORER MUST RUN IN A PINNED WORKTREE (2026-09-05).
#
# Campaign worktrees are pinned at the commit their configs were generated
# from, and `src/` is FROZEN there while the campaign runs. So a `src/` in a
# campaign checkout can predate a name a scorer wants. That is not a bug to fix
# by updating `src/` -- updating it splits `code_version` and voids the
# campaign.
#
# It happened: `deployed_h2h` grew a shared floor bar and imported it from
# `sensitivity_screen`, which imports `src.training.constraints`. On
# `optloss-domb`, whose pinned `src/` has no `cap_fraction_for`, the arm-vs-arm
# scorer stopped importing at all -- in three worktrees at once, on a tool
# whose entire job is deciding whether a number may be quoted. The constant
# moved to `scripts/floors.py`, which reaches nothing.
#
# These three are the pinned-worktree class: reached for when judging whether a
# result is real, so they must import in EVERY checkout at EVERY commit.
# `sensitivity_screen` is deliberately NOT in the list -- it genuinely needs
# `src` to derive cap fractions, and it is run on the live tree.
# ---------------------------------------------------------------------------

MUST_IMPORT_WITHOUT_SRC = ("scripts.quarantine", "scripts.pred_integrity",
                           "scripts.deployed_h2h", "scripts.floors")

_BLOCK_SRC = (
    "import sys\n"
    "class Block:\n"
    "    def find_module(self, name, path=None):\n"
    "        if name == 'src' or name.startswith('src.'):\n"
    "            return self\n"
    "        return None\n"
    "    def load_module(self, name):\n"
    "        raise ImportError('src UNAVAILABLE: pinned worktree')\n"
    "sys.meta_path.insert(0, Block())\n"
    "import importlib\n"
    "importlib.import_module(sys.argv[1])\n"
)


@pytest.mark.parametrize("mod", MUST_IMPORT_WITHOUT_SRC)
def test_the_scorer_IMPORTS_in_a_worktree_whose_src_is_old(mod):
    """2026-09-05: `deployed_h2h` imported a shared bar from
    `sensitivity_screen`, which reaches `src.training.constraints`. A
    campaign worktree is pinned and its `src/` is frozen, so on
    `optloss-domb` the arm-vs-arm scorer stopped importing entirely -- in
    three worktrees at once. Fixed by moving the constant to
    `scripts/floors.py`, which reaches nothing.
    """
    r = subprocess.run([sys.executable, "-c", _BLOCK_SRC, mod],
                       cwd=REPO, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, (
        "%s cannot import when `src` is unavailable, so it is unrunnable in a "
        "pinned campaign worktree. Do NOT fix this by updating `src/` there -- "
        "that splits code_version. Move the shared thing into a module that "
        "reaches nothing, as scripts/floors.py does.%s%s"
        % (mod, chr(10), r.stderr[-600:]))


def test_NEGATIVE_CONTROL_the_src_block_really_blocks():
    """2026-09-05: a module that DOES need `src` must fail under the probe.

    Without this the test above passes whenever the blocker silently does
    nothing, which is the shape of a gate that has never been shown to work.
    """
    r = subprocess.run([sys.executable, "-c", _BLOCK_SRC,
                        "scripts.sensitivity_screen"],
                       cwd=REPO, capture_output=True, text=True, timeout=180)
    assert r.returncode != 0, (
        "`sensitivity_screen` imports `src.training.constraints` and must fail "
        "under the blocker. It did not, so the blocker is inert and the checks "
        "above prove nothing.")


# ---------------------------------------------------------------------------
# LESSON (2026-09-09): a campaign on a DIFFERENT DATASET is an independent
# unit BY CONSTRUCTION, and `bcn1mn3` sat unread because nothing said so.
#
# `MEASURED_UNITS` exists because two iwildcam campaigns can share a warm-up
# and hand out a free replicate to a sign test; every entry there was verified
# by md5 of `final_predictions_raw.csv`. That test is a SAMPLE. For a
# cross-dataset pair it is also unnecessary: `compute_base_model_id` puts the
# dataset in the id's PREFIX, so the caches cannot collide and the warm-ups
# were trained on different data.
#
# This gate pins that, so the D1 entry is a checked claim rather than a
# comment. If the id scheme ever drops the dataset, the entry becomes unsafe
# and this goes red.
# ---------------------------------------------------------------------------

def test_a_cross_dataset_campaign_cannot_share_a_warm_up_by_construction():
    """2026-09-09: a different DATASET is an independent unit by construction.

    `MEASURED_UNITS` verified every entry by md5 because two iwildcam
    campaigns really can share a warm-up. For a cross-dataset pair md5 is
    a sample of something the id scheme already settles:
    `compute_base_model_id` returns `model_dataset_hash`, so a bcn model
    id begins `MobileNetV3_bcn_` and can never collide with
    `MobileNetV3_iwildcam_`. Pinned here so the D1 ledger entry is a
    checked claim, and so dropping the dataset from the id goes red.
    """
    from configs.gen_campaign import compute_base_model_id

    P = {"warmup_identity_keys": ["warmup_epochs", "seed"]}
    hp = {"warmup_epochs": 1, "seed": 1}
    dc_i = {"data_dir": "data/iwildcam/oodslice", "num_classes": 8}
    dc_b = {"data_dir": "data/bcn/oodslice", "num_classes": 8}

    iw = compute_base_model_id(P, "MobileNetV3", hp, "iwildcam", dc_i)
    bc = compute_base_model_id(P, "MobileNetV3", hp, "bcn", dc_b)

    assert iw != bc, "two datasets produced the SAME base_model_id: %s" % iw
    assert iw.startswith("MobileNetV3_iwildcam_"), iw
    assert bc.startswith("MobileNetV3_bcn_"), bc
    # the dataset is in the PREFIX, so no hash collision can ever join them
    assert iw.split("_")[1] != bc.split("_")[1]

    # NEGATIVE CONTROL: same dataset AND same identity keys MUST collide, or
    # the warm-up cache would never be shared and the whole unit ledger --
    # which exists to catch exactly that sharing -- would be pointless.
    again = compute_base_model_id(P, "MobileNetV3", dict(hp), "iwildcam", dict(dc_i))
    assert again == iw, (
        "the same warm-up produced two ids (%s vs %s); arms that should share "
        "a cached model would each retrain one" % (iw, again))

    # NEGATIVE CONTROL: a different data_dir under the SAME dataset name must
    # still separate them -- that is what stops a re-sliced dataset silently
    # loading the old slice's model.
    other_slice = compute_base_model_id(
        P, "MobileNetV3", hp, "iwildcam",
        {"data_dir": "data/iwildcam/othersplit", "num_classes": 8})
    assert other_slice != iw, "data_dir is not in the warm-up identity"


def test_the_unit_ledger_licenses_the_completed_bcn_campaign():
    """2026-09-09: `bcn1mn3` is COMPLETE and was contributing nothing.

    228 runs, 4 seeds, and L80/L90 are verified task cells -- yet it was
    absent from `MEASURED_UNITS`, so every row read `UNVERIFIED`. Same
    defect class as the `add_seeds` pooling bug and `shape1`'s third
    stream: runs bought, executed, then not read. FRAMEWORK 2(z66).
    """
    from scripts.paper_rows import MEASURED_UNITS

    assert ("bcn1mn3", "MobileNetV3") in MEASURED_UNITS, (
        "bcn1mn3 is complete and independent by construction; leaving it out "
        "silently discards 228 runs from the unit count.")
    label = MEASURED_UNITS[("bcn1mn3", "MobileNetV3")]
    clash = [k for k, v in MEASURED_UNITS.items()
             if v == label and k != ("bcn1mn3", "MobileNetV3")]
    assert not clash, (
        "bcn1mn3 shares unit label %r with %s. A cross-dataset campaign cannot "
        "share a warm-up, so sharing a label would UNDER-count units."
        % (label, clash))


# `docs/paper/` is the disjoint dermmnist generation (WHICH_CORPUS.md), where
# the figure is in context and must stay.
_DERM_PRIZE_SKIP = ("docs/paper", "docs/archive", ".git")
_DERM_QUALIFIER = ("dermmnist", "derm ")


def test_the_dermmnist_prize_figure_never_travels_unqualified():
    """`1.9-9.9 items` is a dermmnist number and it was live in EIGHT places.

    2026-09-10. dermmnist is removed and leaks 38.7% of its test set; the
    only runnable dataset's prize is 0.0-1.0 items at the tight caps and
    11.7-21.2 per cell (3.5-12.0 per class) at the task caps. FRAMEWORK 2(z30)
    already recorded the defect and claimed the fix was to caveat the
    DEFINITION SITE "so the caveat travels with the print" -- it does not
    travel, and the sweep found the figure still bare in `cell_table`'s
    docstring (attributed to iwildcam BY NAME), `dataset_screen`,
    `family_split`'s printed line, PLAYBOOK's scoring rule, two FRAMEWORK
    entries, CLAUDE.md's probe block, and -- the one that changes a verdict --
    `frozen_head_probe --headroom-items`, whose DEFAULT was 9.9.

    The rule is not "never write it": it is in context wherever the entry is
    about dermmnist. The rule is that the qualifier travels with it. So a
    line carrying the figure must name dermmnist within the same paragraph.

    NEGATIVE CONTROL below: a synthetic bare occurrence must fail, or this
    test would pass on a repo that had regressed.
    """
    import re

    def bare(text, path):
        """Occurrences of the figure with no dermmnist qualifier nearby."""
        hits = []
        for m in re.finditer(r"1\.9-9\.9|1\.9 to 9\.9", text):
            near = text[max(0, m.start() - 400):m.end() + 400].lower()
            if not any(q in near for q in _DERM_QUALIFIER):
                hits.append("%s:%d" % (path, text[:m.start()].count("\n") + 1))
        return hits

    bad = []
    for base, dirs, files in os.walk(REPO):
        rel = os.path.relpath(base, REPO).replace("\\", "/")
        if any(rel == s or rel.startswith(s + "/") for s in _DERM_PRIZE_SKIP):
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in (".git", "results", "evidence")]
        for f in files:
            if not f.endswith((".md", ".py")):
                continue
            p = os.path.join(base, f)
            try:
                txt = io.open(p, encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                continue
            bad += bare(txt, os.path.relpath(p, REPO).replace("\\", "/"))

    assert not bad, (
        "the dermmnist prize figure appears with no dermmnist qualifier "
        "within 400 chars at: %s. On iwildcam the prize is 0.0-1.0 (tight) / "
        "11.7-21.2 per cell (task). FRAMEWORK 2(z30), (o)." % bad)

    # NEGATIVE CONTROL: the detector must fire on a bare occurrence.
    assert bare("the whole prize is 1.9-9.9 items", "synthetic.md"), (
        "the detector does not fire on a bare figure, so the sweep above "
        "proves nothing")
    # ...and must NOT fire when the qualifier is present.
    assert not bare("1.9-9.9 items, a dermmnist figure", "synthetic.md"), (
        "the detector fires even when dermmnist is named, so it would force "
        "the figure out of the entries where it is correct")


def test_the_probe_headroom_default_is_the_runnable_datasets_prize():
    """`frozen_head_probe --headroom-items` GATES a verdict, and its default
    was the removed dataset's number (2026-09-10).

    `if res > args.headroom_items` is what prints "AND THAT IS COARSER THAN
    THE ENTIRE QUESTION". A default of 9.9 understated the question on
    iwildcam, whose per-class task-cap top is 12.0. A stale docstring is a
    reading error; a stale DEFAULT is a wrong verdict.
    """
    import argparse
    import scripts.frozen_head_probe as fhp

    src = io.open(os.path.join(REPO, "scripts", "frozen_head_probe.py"),
                  encoding="utf-8").read()
    tree = ast.parse(src)
    got = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--headroom-items"):
            continue
        for kw in node.keywords:
            if kw.arg == "default":
                got = ast.literal_eval(kw.value)
    assert got == 12.0, (
        "--headroom-items defaults to %r. 9.9 is dermmnist's top; iwildcam's "
        "per-class task-cap top is 12.0, and this value decides whether the "
        "probe warns that it cannot see the question at all." % (got,))


def _shadowed_imports(src):
    """{name: (import_line, rebind_line)} for module-level imports rebound later.

    MODULE LEVEL ONLY. A function that names a local `json` is fine and
    routine; a module that imports `capped_classes` and then defines
    `def capped_classes(...)` has silently replaced the module for the whole
    file, and every attribute access on it is an AttributeError at run time.
    """
    tree = ast.parse(src)
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                imported.setdefault(a.asname or a.name.split(".")[0],
                                    node.lineno)
    out = {}
    for node in tree.body:                       # module level only
        names = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        for n in names:
            if n in imported:
                out[n] = (imported[n], node.lineno)
    return out


def test_no_module_import_is_shadowed_by_a_local_definition():
    """`scripts/order_probe.py` imported `capped_classes` and then defined a
    function of the same name 86 lines later.

    From `2dd84549` (2026-09-09) to 2026-09-10 that made
    `capped_classes.assert_single_dataset(...)` an **AttributeError on every
    `--campaign` invocation**: the probe parsed, imported, passed `audit_config`,
    `doc_commands`, `dead_code` and the whole suite, and was unrunnable on every
    real input. It is the same shape as the three scorers that shipped
    `quarantine.` with no module-level import, which is why
    `tests/test_scorers_run_end_to_end.py` exists -- but that file drives the
    SCORERS, and `order_probe` is a probe, so nothing executed this path.

    An AttributeError rather than a NameError is what made it invisible: the
    name resolves, to the wrong object. Only running it, or this, finds it.
    """
    fails = []
    for sub in ("scripts", "configs", "src"):
        for dirpath, _dirs, files in os.walk(os.path.join(REPO, sub)):
            if "__pycache__" in dirpath:
                continue
            for f in sorted(files):
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dirpath, f)
                try:
                    hits = _shadowed_imports(io.open(p, encoding="utf-8").read())
                except SyntaxError:
                    continue
                for name, (imp, reb) in sorted(hits.items()):
                    fails.append(
                        "%s: `%s` imported at line %d and REBOUND at line %d "
                        "-- every attribute access on it is an AttributeError"
                        % (os.path.relpath(p, REPO).replace("\\", "/"),
                           name, imp, reb))
    assert not fails, "shadowed module imports:\n  " + "\n  ".join(fails)


def test_the_shadowed_import_detector_actually_detects():
    """NEGATIVE CONTROLS for the 2026-09-10 gate above.

    A gate that has never failed has never been shown to work, and that one is
    a whole-tree scan -- the shape that reads green when it is broken, because
    "no shadowed imports found" and "the detector found nothing" print the
    same. Mutation-tested 2026-09-10 by restoring `order_probe`'s original
    `from scripts import capped_classes`: the scan goes red and the alias
    control stays green.
    """
    bad = ("from scripts import capped_classes\n"
           "\n"
           "def capped_classes(run_dir):\n"
           "    return []\n")
    assert "capped_classes" in _shadowed_imports(bad), (
        "the exact 2026-09-09 defect is not detected")

    # NEGATIVE CONTROL 1: a LOCAL name of the same spelling is routine and
    # must not fire -- the module object is untouched outside that frame.
    local = ("import json\n"
             "\n"
             "def f():\n"
             "    json = 1\n"
             "    return json\n")
    assert not _shadowed_imports(local), (
        "a function-local rebinding must NOT fire")

    # NEGATIVE CONTROL 2: an aliased import is not shadowed by a function of
    # the ORIGINAL name.
    alias = ("from scripts import capped_classes as cc\n"
             "\n"
             "def capped_classes(run_dir):\n"
             "    return []\n")
    assert not _shadowed_imports(alias), (
        "an aliased import is not shadowed by the original spelling")

    # NEGATIVE CONTROL 3: a module-level ASSIGNMENT over an import counts too,
    # and is the same defect with a different statement type.
    assign = "import math\n\nmath = 3\n"
    assert "math" in _shadowed_imports(assign), (
        "a module-level assignment over an import must fire")


# Tools whose main path an EMPTY CAMPAIGN ROOT cannot exercise. Each needs a
# reason, because the default is coverage: a name added here is a decision to
# leave a tool unexecuted, which is what 2026-09-09 cost.
EMPTY_ROOT_EXEMPT = {
    "rig_status": "shells out to nvidia-smi/ps and reads the live hosts",
    "run_campaign": "the step gate; it launches and resets real runs",
    "add_seeds": "writes configs into results/",
    "quarantine": "--apply --execute mutates the registry on disk",
    "bias_shift_probe": "pure algebra, no path argument at all",
    "capped_classes": "a library; its CLI is --self-test only",
    "ortho_survival": "pure algebra, no path argument at all",
    "dead_code": "walks the repo, not a campaign",
    "doc_commands": "walks the docs, not a campaign",
    "stale_figures": "walks the docs, not a campaign",
    "stale_provenance": "walks the docs, not a campaign",
    "campaign_state": "walks the docs, not a campaign (2(z83))",
    # OK: `paper_rows`: its reason WAS a ticket ("needs a file fixture") and the
    # ticket sat there, so the one scorer whose output decides what reaches a
    # manuscript had no end-to-end test at all. It is still exempt from the
    # ROOT smoke -- it takes `--cells <csv>` and an empty directory says
    # nothing about it -- but the exemption is now a COVERAGE CLAIM instead of
    # a promise: see `test_paper_rows_runs_END_TO_END_against_a_real_cell_table_csv`
    # and the two refusal tests beside it (2026-09-10).
    "paper_rows": "takes --cells <csv>, not a root; covered end-to-end by "
                  "test_paper_rows_runs_END_TO_END_against_a_real_cell_table_csv",
    #
    # `step_dose` stays, and its reason is now a MEASURED one rather than a
    # ticket: `main()` calls `src.pipeline.data.load_data`, which needs the
    # gitignored 3.0 GB + 443 MB `.npy` arrays, and `get_model`, which pulls
    # pretrained weights. No fixture makes that runnable on a laptop. Its
    # `--self-test` measures the real quantity on a small linear model in
    # process, which is the stronger check available here.
    "step_dose": "main() needs load_data (3.0 GB gitignored arrays) and "
                 "pretrained weights; --self-test covers the measurement",
}

FATAL_ON_AN_EMPTY_ROOT = ("AttributeError", "NameError", "UnboundLocalError",
                          "ImportError", "IndentationError", "TypeError")


def _root_argv(src, root):
    """The one flag that points a tool at a campaign, or None."""
    tree = ast.parse(src)
    flags, positional = [], []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args):
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                (flags if a0.value.startswith("-")
                 else positional).append(a0.value)
    for f in ("--campaign", "--root"):
        if f in flags:
            return [f, root]
    if "--glob" in flags:
        return ["--glob", os.path.join(root, "*")]
    return [root] if positional else None


def _gated_tools():
    """{module: source} for every `scripts/*.py` carrying a `--self-test`."""
    out = {}
    for f in sorted(os.listdir(os.path.join(REPO, "scripts"))):
        if not f.endswith(".py") or f == "__init__.py":
            continue
        src = io.open(os.path.join(REPO, "scripts", f),
                      encoding="utf-8").read()
        if '"--self-test"' in src or "'--self-test'" in src:
            out[f[:-3]] = src
    return out


def test_every_gated_tool_fails_CLEANLY_on_an_EMPTY_campaign_root():
    """2026-09-10: `order_probe` was unrunnable on every real input for a day.

    `scripts/order_probe.py` imported the module `capped_classes` and then
    defined a function of the same name, so `capped_classes.assert_single_dataset`
    -- called BEFORE the glob, on line 384 -- raised `AttributeError` on every
    `--campaign` invocation. Every static gate was green, and `--self-test` was
    green because it drives `verdict` and `sign_test` and never enters `main`.
    FRAMEWORK 2(z81).

    **A `--self-test` that never enters `main` tests the helpers, not the tool.**
    41 modules carry one and, before this, SEVEN had ever been executed the way
    a person executes them -- the six scorers in
    `tests/test_scorers_run_end_to_end.py` plus `order_probe` once it had a
    fixture.

    This is the cheap half of the coverage: point every tool at an EMPTY
    campaign root and require it to fail like a tool, not like a bug. It
    asserts nothing about the numbers -- `tests/test_scorers_run_end_to_end.py`
    does that on a real fixture -- only that the entry point is reachable and
    the refusal is deliberate. It would have caught the 2026-09-09 defect on
    the day it landed, because that call sits before any file is read.
    """
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    tools = _gated_tools()
    unknown = sorted(set(EMPTY_ROOT_EXEMPT) - set(tools))
    assert not unknown, (
        "EMPTY_ROOT_EXEMPT names modules that no longer carry a --self-test: "
        "%s. Remove them, or the exemption list rots into a place where a "
        "tool can hide." % unknown)

    root = tempfile.mkdtemp(prefix="smoke_empty_root_")
    jobs = []
    unclassified = []
    for name, src in sorted(tools.items()):
        if name in EMPTY_ROOT_EXEMPT:
            continue
        argv = _root_argv(src, root)
        if argv is None:
            unclassified.append(name)
            continue
        jobs.append((name, argv))
    assert not unclassified, (
        "these tools take no campaign root and are not in EMPTY_ROOT_EXEMPT: "
        "%s. Classify each -- either give it a root-shaped flag or write down "
        "why it cannot have one." % unclassified)

    def run(job):
        name, argv = job
        try:
            r = subprocess.run(
                [sys.executable, "-m", "scripts." + name] + argv,
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=120, cwd=REPO,
                env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        except subprocess.TimeoutExpired:
            return "%s: hung for 120s on an EMPTY root" % name
        out = (r.stdout or "") + (r.stderr or "")
        bad = [f for f in FATAL_ON_AN_EMPTY_ROOT if f + ":" in out]
        if bad:
            return ("%s %s: %s -- the entry point is broken, not the input\n"
                    "        %s" % (name, argv, ", ".join(bad),
                                    out.strip()[-400:]))
        return None

    try:
        with ThreadPoolExecutor(max_workers=4) as ex:
            fails = [f for f in ex.map(run, jobs) if f]
    finally:
        shutil.rmtree(root, ignore_errors=True)
    assert not fails, ("tools that do not survive an empty root:\n  "
                       + "\n  ".join(fails))


def test_the_empty_root_smoke_would_have_caught_the_2026_09_10_defect():
    """NEGATIVE CONTROL for the 2026-09-10 gate above.

    That gate is a whole-tree sweep, the shape that reads green when it is
    broken -- "every tool passed" and "no tool ran" print the same.

    So reproduce the defect in a throwaway module and require the same
    subprocess check to see it. It must fire on a shadowed import and stay
    quiet on a tool that merely refuses an empty root, which is the normal and
    correct behaviour of all 33.
    """
    import shutil
    import textwrap

    pkg = tempfile.mkdtemp(prefix="smoke_control_")
    try:
        d = os.path.join(pkg, "faketools")
        os.makedirs(d)
        io.open(os.path.join(d, "__init__.py"), "w", encoding="utf-8").write("")
        io.open(os.path.join(d, "broken.py"), "w", encoding="utf-8").write(
            textwrap.dedent("""
                import argparse
                import json


                def json(x):            # shadows the import, exactly as 2(z81)
                    return x


                def main():
                    ap = argparse.ArgumentParser()
                    ap.add_argument("--campaign")
                    a = ap.parse_args()
                    json.dumps({"root": a.campaign})
                    return 0


                if __name__ == "__main__":
                    raise SystemExit(main())
            """))
        io.open(os.path.join(d, "healthy.py"), "w", encoding="utf-8").write(
            textwrap.dedent("""
                import argparse
                import os


                def main():
                    ap = argparse.ArgumentParser()
                    ap.add_argument("--campaign")
                    a = ap.parse_args()
                    if not os.listdir(a.campaign):
                        print("no runs under %s" % a.campaign)
                        return 1
                    return 0


                if __name__ == "__main__":
                    raise SystemExit(main())
            """))
        empty = os.path.join(pkg, "empty")
        os.makedirs(empty)

        def out_of(mod):
            r = subprocess.run(
                [sys.executable, "-m", "faketools." + mod,
                 "--campaign", empty],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=60, cwd=pkg,
                env=dict(os.environ, PYTHONIOENCODING="utf-8"))
            return (r.stdout or "") + (r.stderr or "")

        broken = out_of("broken")
        assert any(f + ":" in broken for f in FATAL_ON_AN_EMPTY_ROOT), (
            "the 2(z81) shape is not detected by the empty-root check:\n%s"
            % broken)
        healthy = out_of("healthy")
        assert not any(f + ":" in healthy for f in FATAL_ON_AN_EMPTY_ROOT), (
            "NEGATIVE CONTROL: a clean refusal must NOT read as a defect:\n%s"
            % healthy)
    finally:
        shutil.rmtree(pkg, ignore_errors=True)


# ---------------------------------------------------------------------------
# LESSON 32 (2026-09-10): A RUN-STATE TABLE CAN ONLY BE AUDITED FOR THE ROWS
# IT CONTAINS.
#
# `price1` was launched (task #78, COMPLETED), is named four times in
# FRAMEWORK -- twice as "before `price1` launched" -- and 2(z29), the entry
# holding the whole coin result, says it is "being re-priced by `price1`".
# Its state is recorded in NO authority: not quarantine.REGISTRY, not
# COVERAGE section 0, not MISSION's LIVE or LANDED tables, not CLAUDE.md's
# archive list. Nothing said whether it ran, died, or landed.
#
# That is 2(z72)'s defect inverted. A STALE row is visible to anyone who
# re-reads the block; an ABSENT row is visible to nobody, because there is no
# block to re-read. The first audit found 18 such campaigns, 10 of them
# carrying a past-execution verb. FRAMEWORK 2(z83).
# ---------------------------------------------------------------------------


def test_no_campaign_is_discussed_without_a_recorded_state():
    """2026-09-10: `price1` was launched and no doc recorded that it existed."""
    from scripts import campaign_state

    _, unrecorded = campaign_state.scan()
    orphans = [(n, len(ex)) for n, _cnt, ex, _hits in unrecorded if ex]
    assert not orphans, (
        "these campaigns carry a PAST-EXECUTION verb and no recorded state: "
        "%s\nGive each a row in MISSION 0-RUNNING's campaign-state ledger, or "
        "a marker in quarantine.REGISTRY. Run "
        "`python -m scripts.campaign_state --all` for the lines." % orphans)


def test_the_campaign_state_audit_actually_detects_an_orphan():
    """2026-09-10: the negative control -- the gate above must be able to fail.

    A gate that has never failed has never been shown to work, and this one
    reads five separate authorities, so "it passes" is weak evidence on its
    own.
    """
    import shutil

    from scripts import campaign_state

    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "docs"))
        doc = os.path.join(root, "docs", "FRAMEWORK.md")
        io.open(doc, "w", encoding="utf-8").write(
            "# f\n\n`zzz9` was launched on dsisco01 and landed 48 runs\n"
            "`qqq9` would be generated by `--root results/qqq9`\n")
        _, unrec = campaign_state.scan(("docs/FRAMEWORK.md",), root=root)
        flagged = [n for n, _c, ex, _h in unrec if ex]
        assert flagged == ["zzz9"], (
            "the detector must flag the campaign with an execution verb and "
            "ONLY that one, got %s" % flagged)
        # NEGATIVE CONTROL inside the control: a name with no execution verb
        # is a proposal, not an orphan, and must not be reported as a defect.
        assert "qqq9" in [n for n, _c, _e, _h in unrec]
        assert "qqq9" not in flagged
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# LESSON 33 (2026-09-10): THE argsort AUDIT IS PER-FILE AND THE DEFECT IS PER
# CALL SITE.
#
# The allocator emits top-`k_g` WITHIN each group. A tool that rebuilds the
# cut with `argsort(-p)[:K]` reads the GLOBALLY K-th item instead, and the two
# sets differ. NINE sites have now been found, one at a time, by hand:
#
#   1 order_probe --evictions (2026-08-28, DISCLOSED not fixed)
#   2 the task window        2(z16)
#   3 the cap screen         2(z28)
#   4 the fmow window        2(z59)
#   5 paired_noise           2(z63)
#   6 cut_gap                2(z64)
#   7 step_direction_probe   2(z79)
#   8 order_probe band+Jaccard 2(z80)
#   9 score_scan prec@K+Jaccard 2(z84)  <- found BY THIS GATE'S ENUMERATION
#
# 2(z80) is the lesson: the audit walked FILES and cleared `order_probe` while
# listing only one of its two sites. #9 was found the first time the sites
# were enumerated mechanically instead of read.
#
# So this gate does not try to judge whether a site is correct -- it cannot,
# and guessing is how 2(z64) got a fixtured, mutation-tested, WRONG direction.
# It requires every site to be CLASSIFIED, and turns red on any site that is
# new or whose sorted expression changed. Classifying is the human's job; not
# noticing is what this prevents.
# ---------------------------------------------------------------------------

# `sort` is in the list because the fix to site 1 introduced an
# `np.sort(pn)[::-1][K - 1]` -- a global cut the argsort-only registry
# could not see. A registry that tracks one spelling of "take the K-th
# ranked item" and not another is the per-FILE mistake again, one level
# down. (2026-09-10)
SORTS = ("argsort", "argpartition", "topk", "nlargest", "sort",
         "partition", "quantile")
# !! THE TARGET LIST IS THE GATE'S BLIND SPOT, AND IT HAS BEEN WIDENED
# TWICE IN ONE DAY (2026-09-10). `sort` joined it because the fix to site 1
# introduced an `np.sort(pn)[::-1][K - 1]` an argsort-only registry could
# not see; `partition` and `quantile` joined it hours later because
# `straddle_probe.cut_score` located its cut with `np.partition(s, -K)[-K]`
# and the freshly-built gate passed that file CLEAN. That site was the
# ELEVENTH (FRAMEWORK 2(z85)).
# !! WHAT IS DELIBERATELY EXCLUDED, AND WHY -- a list nobody maintains is
# not a gate. `max`/`min` occur 416 times and are overwhelmingly bounds and
# clamps; `median` 29 times as a report statistic; `argmax` 50 times over
# the CLASS axis, which is the model's decision and not an item budget at
# all. `searchsorted` is a genuine cut spelling and occurs ZERO times --
# checked, so it need not be listed. Add a spelling here the moment one
# appears; do not add one that would flood the registry with non-cuts.

# Verdicts, and each is a claim somebody checked:
#   DEPLOYED    cuts per group, or uses the allocator's own selection
#   GREEDY-ROOM global ORDER, but every take is gated on that group's room --
#               a different rule from per-group top-k and NOT a naive global K
#   GLOBAL-KEPT a global reading retained ON PURPOSE beside a correct one, so
#               an old figure reproduces. Never to be read as a direction
#   GLOBAL-OPEN a known-wrong global reading that is disclosed, not yet fixed
#   NOT-A-CUT   no budget is involved -- rank correlation, kNN, a surrogate
#               loss, fixture construction
#   SELF-TEST   inside a --self-test fixture
ARGSORT_SITES = {
    "scripts/bias_shift_probe.py::kendall_tau::argsort(-a)":
        "NOT-A-CUT rank correlation, pure algebra, no budget",
    "scripts/bias_shift_probe.py::run::argsort(-z0)":
        "NOT-A-CUT compares two orderings; no K anywhere",
    "scripts/bias_shift_probe.py::run::argsort(-z1)":
        "NOT-A-CUT the other half of the same comparison",
    "scripts/cut_gap.py::measure::argsort(-P[:, cls])":
        "GLOBAL-KEPT p_K_glob, retained so the docstring table reproduces; "
        "p_K itself is per-group and budget-weighted (2(z64))",
    "scripts/frozen_head_probe.py::_cut::topk(K)":
        "NOT-A-CUT surrogate loss on a frozen head; no allocator",
    "scripts/frozen_head_probe.py::_matroid_topk::topk(K)":
        "DEPLOYED the global stage runs AFTER the per-group mask, and the "
        "-inf survivors are trimmed; gated by the polytope-vs-greedy test",
    "scripts/frozen_head_probe.py::_matroid_topk::topk(cap)":
        "DEPLOYED explicitly `for idx, cap in zip(group_idx_list, group_caps)`",
    "scripts/frozen_head_probe.py::make_synthetic::argsort(capped_mass[pool])":
        "NOT-A-CUT builds the synthetic fixture",
    "scripts/frozen_head_probe.py::pauc_loss::topk(m)":
        "NOT-A-CUT pAUC surrogate",
    "scripts/graph_probe.py::knn_affinity::argpartition(-S)":
        "NOT-A-CUT picks kNN neighbours, not a budget",
    "scripts/graph_probe.py::prec_at_K::argsort(-col)":
        "SELF-TEST synthetic two-class fixture with no groups at all",
    "scripts/order_probe.py::band_per_group::argsort(-p[where])":
        "DEPLOYED the 2(z80) fix -- sorts within one group's members",
    "scripts/order_probe.py::main::argsort(-pa)":
        "GLOBAL-KEPT the retained band_glob reading, printed beside the "
        "per-group one with its band size (2(z80))",
    "scripts/order_probe.py::main::argsort(-pn)":
        "GLOBAL-KEPT as above, the null arm",
    "scripts/order_probe.py::main::argsort(-pn)#2":
        "GLOBAL-KEPT as above, second use in the same function",
    "scripts/order_probe.py::main::argsort(-pr)":
        "GLOBAL-KEPT as above, the reseed arm",
    "scripts/order_probe.py::self_test::argsort(-pg)":
        "SELF-TEST two-group fixture",
    "scripts/order_probe.py::self_test::argsort(-pg)#2":
        "SELF-TEST the negative control on the same fixture",
    "scripts/paired_noise.py::load_arm::argsort(-p[idx])":
        "DEPLOYED the 2(z63) fix -- `idx` is one group's members",
    "scripts/reachability.py::concentration::argsort(np.abs(m))":
        "NOT-A-CUT sorts gradient magnitudes, no budget",
    "scripts/score_arm.py::equalize::argsort(-y_proba[:, cls])":
        "GREEDY-ROOM fills to exactly K in global order but SKIPS any item "
        "whose group has no room left, which is the post-hoc clipper's own "
        "rule. Not the per-group top-k rule, and deliberately so -- this is "
        "what makes full_panel allocator-blind",
    "scripts/score_scan.py::main::argsort(-prob[:, c])":
        "GLOBAL-KEPT site 9 (2(z84)): this WAS the primary reading and its "
        "prec@K and Jaccard were computed on a set no run deployed. The "
        "deployed reading is now `pa == c`, exact, and this is retained "
        "beside it so the old figures reproduce",
    "scripts/step_direction_probe.py::cut_band::argsort(-z)":
        "GLOBAL-KEPT the retained reading from the 2(z79) fix",
    "scripts/step_direction_probe.py::cut_band::argsort(-z[where])":
        "DEPLOYED the 2(z79) fix -- one group's members",
    "scripts/step_direction_probe.py::self_test::argsort(-z3)":
        "SELF-TEST fixture",
    "scripts/step_direction_probe.py::self_test::argsort(-zA)":
        "SELF-TEST fixture",
    "scripts/step_direction_probe.py::self_test::argsort(-zz)":
        "SELF-TEST fixture",
    "scripts/task_window.py::select_local::argsort(-pr[m])":
        "DEPLOYED the 2(z16) fix -- `m` masks one group",
    "scripts/task_window.py::sweep::argsort(-pr)":
        "GLOBAL-KEPT labelled `GLOBAL top-K (diagnostic only)` in the source, "
        "printed beside the per-group `select_local` result",
    "src/losses/transductive_loss.py::margins::topk(proba)":
        "NOT-A-CUT the differentiable soft count; the loss has no allocator",
    "src/methodologies/heuristic/train.py::apply_allocation_heuristic::argsort(flat)":
        "GREEDY-ROOM THE ALLOCATOR ITSELF. Global order over (item, class) "
        "pairs, every take gated on `_has_room(groups[idx], ci)`",
    "src/methodologies/heuristic/train.py::apply_allocation_heuristic::"
    "argsort(probs[candidates, class_idx])":
        "DEPLOYED sorts within the unassigned candidates, room-gated",
    "src/utils/posthoc_adjustment.py::_fallback_lp::argsort(-y_proba[g_not_c, c])":
        "DEPLOYED `g_not_c` is one group's non-c items",
    "src/utils/posthoc_adjustment.py::_fallback_lp::argsort(-y_proba[not_c, c])":
        "DEPLOYED the GLOBAL scope's own fallback, which is global by "
        "definition -- the global cap is a single budget over all items",
    "src/utils/posthoc_adjustment.py::targeted_correction::argsort(-y_proba[candidates, c])":
        "DEPLOYED sorts within the candidate subset",
    "src/utils/posthoc_adjustment.py::targeted_correction::argsort(-y_proba[local_not_c, c])":
        "DEPLOYED `local_not_c` is one group's non-c items",
    "src/utils/posthoc_adjustment.py::targeted_correction::argsort(y_proba[indices, c])":
        "DEPLOYED sorts within the passed index subset",
    "src/utils/posthoc_adjustment.py::targeted_correction::argsort(y_proba[local_c, c])":
        "DEPLOYED `local_c` is one group's c-labelled items",
    # --- np.sort sites. `sort` joined the target list on 2026-09-10 because
    # the fix to site 1 introduced an `np.sort(pn)[::-1][K - 1]` the
    # argsort-only registry could not see. `list.sort()` (no positional arg) is
    # filtered out -- 14 of them sort report rows and are never a cut.
    "scripts/cut_gap.py::per_group_cut::sort(p[idx])":
        "DEPLOYED the 2(z64) fix -- `idx` is one group's members",
    "scripts/cut_gap.py::self_test::sort(p)":
        "SELF-TEST fixture",
    "scripts/cut_gap.py::self_test::sort(pg)":
        "SELF-TEST fixture",
    "scripts/cut_gap.py::self_test::sort(pg2)":
        "SELF-TEST fixture",
    "scripts/frozen_head_probe.py::stratified_halves::sort(fit_idx)":
        "NOT-A-CUT sorts INDICES to make a reproducible split",
    "scripts/frozen_head_probe.py::stratified_halves::sort(held_idx)":
        "NOT-A-CUT the other half of the same split",
    "scripts/order_probe.py::evictions::sort(pn)":
        "GLOBAL-KEPT `p_cut_glob`, retained beside the per-group cut so the "
        "2026-08-28 figures reproduce (2(z84))",
    "scripts/order_probe.py::evictions::sort(pn[gn == g])":
        "DEPLOYED the 2(z84) fix -- one group's members, budget-weighted",
    "scripts/reachability.py::concentration::sort(np.abs(m))":
        "NOT-A-CUT sorts gradient magnitudes",
    "scripts/reachability.py::slope_at::sort(p_col)":
        "DEPLOYED the caller decides the scope, and `slope_per_group` is now "
        "the primary caller (2(z84), the TENTH site). The global call is "
        "retained and LABELLED `glob` in the output",
    "scripts/sensitivity_screen.py::gradient_at_cut::sort(pr)":
        "GLOBAL-KEPT and CORRECT: this is `p_bd`, the argmax DECISION "
        "BOUNDARY, which is a property of the model over the whole test set "
        "and not a per-group budget. `p_cut` beside it IS per-group, via "
        "`select_local`, budget-weighted. The source says so at the line",
    "scripts/step_direction_probe.py::group_tau::sort(z[where])":
        "DEPLOYED the 2(z79) fix -- one group's members",
    "scripts/step_direction_probe.py::self_test::sort(zz)":
        "SELF-TEST fixture",
    "scripts/step_direction_probe.py::weightings::sort(d)":
        "NOT-A-CUT picks the window TEMPERATURE by item count, not a budget",
    "scripts/step_direction_probe.py::weightings::sort(z)":
        "GLOBAL-KEPT the fallback tau when the caller passes none; the "
        "per-group caller supplies `group_tau` (2(z79))",
    "src/losses/transductive_loss.py::cut_params::sort((col - t).abs())":
        "NOT-A-CUT the LOSS's own window temperature. `soft_count_mode: cut` "
        "is REJECTED (MISSION knob ledger) so this is not in the shipped path",
    "src/losses/transductive_loss.py::cut_params::sort(col)":
        "NOT-A-CUT as above -- the loss sees a training batch, not the "
        "allocator's groups",
    "src/losses/transductive_loss.py::window_temp::sort(m.abs())":
        "NOT-A-CUT margin-window temperature, same rejected family",
    # --- straddle_probe. The registry held ZERO entries for this file until
    # 2026-09-10: `np.partition` and `np.quantile` were not in SORTS, so the
    # gate that had just found sites 9 and 10 walked past site 11 (2(z85)).
    "scripts/straddle_probe.py::cut_score::partition(scores)":
        "GLOBAL-KEPT the pre-2026-09-10 reading, retained and printed in a "
        "column labelled `glob`. `straddle_grouped` is primary",
    "scripts/straddle_probe.py::topk_per_group::argsort(-scores[m])":
        "DEPLOYED `m` is one group's members and `k` its own budget -- the "
        "shuffled CONTROL's selection rule, held on the same per-group "
        "budgets as the real reading (2(z85))",
    "scripts/straddle_probe.py::measured_delta::quantile(d)":
        "NOT-A-CUT a quantile of the per-item |treated - null| DISPLACEMENT. "
        "It sets the delta LADDER, never a selection; no budget is involved",
}


def _sort_call_sites(root):
    """Every sort-on-scores CALL SITE, keyed by module + function + expression.

    Keyed by the sorted EXPRESSION rather than the line number, so the registry
    survives edits above it -- and so changing what is sorted turns the gate
    red, which is the case that matters.
    """
    out = []
    for base in ("scripts", "src", "configs"):
        for dirpath, _dirs, files in os.walk(os.path.join(root, base)):
            if "__pycache__" in dirpath:
                continue
            for f in sorted(files):
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dirpath, f)
                rel = os.path.relpath(p, root).replace("\\", "/")
                try:
                    tree = ast.parse(io.open(p, encoding="utf-8").read())
                except SyntaxError:
                    continue
                stack, found = [], []

                class V(ast.NodeVisitor):
                    def visit_FunctionDef(self, n):
                        stack.append(n.name)
                        self.generic_visit(n)
                        stack.pop()
                    visit_AsyncFunctionDef = visit_FunctionDef

                    def visit_Call(self, n):
                        nm = (getattr(n.func, "attr", None)
                              or getattr(n.func, "id", None))
                        if nm in SORTS:
                            # `list.sort()` takes no positional argument and
                            # is never a cut -- 14 of them (sorting rows for a
                            # report) would otherwise flood the registry and
                            # make it the kind of list nobody reads.
                            if nm == "sort" and not n.args:
                                self.generic_visit(n)
                                return
                            try:
                                a = ast.unparse(n.args[0]) if n.args else "-"
                            except Exception:
                                a = "-"
                            found.append((stack[-1] if stack else "<module>",
                                          nm, a[:44]))
                        self.generic_visit(n)

                V().visit(tree)
                seen = {}
                for fn, nm, a in found:
                    k = "%s::%s::%s(%s)" % (rel, fn, nm, a)
                    seen[k] = seen.get(k, 0) + 1
                    out.append(k if seen[k] == 1 else "%s#%d" % (k, seen[k]))
    return sorted(out)


def test_every_sort_on_scores_is_classified_per_CALL_SITE():
    """2026-09-10: the audit was per-FILE and cleared a file half-read (2(z80))."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    found = set(_sort_call_sites(root))
    known = set(ARGSORT_SITES)

    new = sorted(found - known)
    assert not new, (
        "%d sort-on-scores call site(s) are not classified:\n  %s\n\n"
        "The allocator emits top-k_g WITHIN each group, so `argsort(-p)[:K]` "
        "reads a different set. ELEVEN sites have carried that defect. Add each "
        "to ARGSORT_SITES with one of DEPLOYED / GREEDY-ROOM / GLOBAL-KEPT / "
        "GLOBAL-OPEN / NOT-A-CUT / SELF-TEST and a reason somebody checked."
        % (len(new), "\n  ".join(new)))

    # ROT: a registry entry for a site that no longer exists is a stale claim,
    # and it is how an exemption list becomes a place things hide.
    gone = sorted(known - found)
    assert not gone, (
        "ARGSORT_SITES names %d call site(s) that no longer exist:\n  %s\n"
        "Delete them, or the registry is describing code that is gone."
        % (len(gone), "\n  ".join(gone)))


def test_the_call_site_registry_would_have_caught_the_2026_09_10_defect():
    """2026-09-10: the negative control -- a new global cut must turn it red."""
    import shutil

    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "scripts"))
        io.open(os.path.join(root, "scripts", "newtool.py"), "w",
                encoding="utf-8").write(
            "import numpy as np\n\n\n"
            "def measure(prob, c, K):\n"
            "    return np.argsort(-prob[:, c])[:K]\n")
        found = _sort_call_sites(root)
        assert found == ["scripts/newtool.py::measure::argsort(-prob[:, c])"], found
        assert set(found) - set(ARGSORT_SITES) == set(found), (
            "a brand-new tool's cut must be UNCLASSIFIED, i.e. must turn the "
            "gate red until somebody writes down what it does")

        # NEGATIVE CONTROL: renaming the sorted expression must change the key,
        # because that is exactly the edit that silently changes what is cut.
        io.open(os.path.join(root, "scripts", "newtool.py"), "w",
                encoding="utf-8").write(
            "import numpy as np\n\n\n"
            "def measure(prob, c, K, where):\n"
            "    return np.argsort(-prob[where, c])[:K]\n")
        assert _sort_call_sites(root) == [
            "scripts/newtool.py::measure::argsort(-prob[where, c])"], (
            "the key must follow the EXPRESSION, not the line")

        # NEGATIVE CONTROL: a file with no sort at all yields nothing, so the
        # gate cannot pass by finding zero sites everywhere.
        io.open(os.path.join(root, "scripts", "newtool.py"), "w",
                encoding="utf-8").write("def measure():\n    return 1\n")
        assert _sort_call_sites(root) == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# LESSON 34 (2026-09-10): THE TOOL THAT SAYS WHAT MAY BE WRITTEN HAD NEVER
# BEEN EXECUTED AGAINST A FILE.
#
# `test_every_gated_tool_fails_CLEANLY_on_an_EMPTY_campaign_root` points 33
# root-shaped tools at an empty directory. `paper_rows` was EXEMPT from it,
# with the reason "takes --cells <csv>; needs a file fixture" -- i.e. the
# exemption was a ticket, and the ticket sat there. So the one scorer whose
# output decides what reaches a manuscript had no end-to-end test at all: its
# `--self-test` exercises `build()` in-process and never enters `main`, which
# is exactly the gap 2(z81) exploited to leave `order_probe` unrunnable for a
# day with every gate green.
#
# This runs the real CLI as a subprocess against a real CSV, and pins the
# three refusals that are the whole point of the tool: a hard-quarantined
# campaign must be REFUSED, a PARTIAL campaign's dead arms must be DROPPED
# while its live arms survive, and a CSV that is not a cell_table must be
# named rather than raising.
# ---------------------------------------------------------------------------

CELL_COLS = ("campaign", "dataset", "model", "cap", "arm", "n_seeds", "seeds",
             "items_per_001", "ccF1", "ccF1_sd")


def _cell_csv(path, rows):
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(CELL_COLS) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in CELL_COLS) + "\n")


def _cell(campaign, arm, ccF1, seeds="1|2|3|4", model="MobileNetV2",
          cap="L80_G95"):
    return {"campaign": campaign, "dataset": "iwildcam", "model": model,
            "cap": cap, "arm": arm, "n_seeds": len(seeds.split("|")),
            "seeds": seeds, "items_per_001": 0.15, "ccF1": ccF1,
            "ccF1_sd": 0.004}


def _run_paper_rows(*argv):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(
        [sys.executable, "-m", "scripts.paper_rows"] + list(argv),
        cwd=root, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=120,
        env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def test_paper_rows_runs_END_TO_END_against_a_real_cell_table_csv():
    """2026-09-10: it was EXEMPT from the smoke and its self-test never entered main."""
    import shutil

    d = tempfile.mkdtemp()
    try:
        csv_path = os.path.join(d, "cells.csv")
        out_path = os.path.join(d, "rows.csv")
        # `taskwin2` is live: not in quarantine.REGISTRY at all.
        _cell_csv(csv_path, [
            _cell("taskwin2", "tralo", 0.5120),
            _cell("taskwin2", "tralo_null", 0.5060),
            _cell("taskwin2", "tralo_reseed", 0.5058),
            _cell("taskwin2", "clip", 0.5090),
        ])
        code, body = _run_paper_rows("--cells", csv_path, "--out", out_path)
        assert code == 0, body
        assert os.path.exists(out_path), "no --out file was written:\n%s" % body
        written = io.open(out_path, encoding="utf-8").read().strip().splitlines()
        assert len(written) > 1, "header only, no rows:\n%s" % body
        # It emits per (cell, CONTRAST). A lambda=0 twin is a REFERENCE and
        # never a subject, so read the `arm` column -- a substring test on the
        # whole line matches the reference column too and asserts nothing.
        head = written[0].split(",")
        i_arm, i_ref = head.index("arm"), head.index("ref")
        subjects = {ln.split(",")[i_arm] for ln in written[1:]}
        refs = {ln.split(",")[i_ref] for ln in written[1:]}
        assert "tralo" in subjects, subjects
        assert not {a for a in subjects if a.endswith(("_null", "_reseed"))}, (
            "a lambda=0 twin was emitted as a SUBJECT row: %s" % subjects)
        assert {"tralo_null", "tralo_reseed", "clip"} <= refs, (
            "the three contrasts vs clip / own null / reseed floor did not all "
            "resolve, so this fixture is not exercising the tool: %s" % refs)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_paper_rows_REFUSES_a_hard_quarantined_campaign_and_DROPS_partial_arms():
    """2026-09-10: the refusals are the tool's purpose and were never executed."""
    import shutil

    from scripts import quarantine

    # Read the authorities rather than hardcoding names, so this test cannot
    # drift from the registry it is checking.
    hard = [n for n, e in quarantine.REGISTRY.items()
            if e.get("scorable") is False]
    partial = [(n, sorted(e["dead_arms"])) for n, e in quarantine.REGISTRY.items()
               if e.get("scorable") is not False and e.get("dead_arms")]
    assert hard and partial, (
        "the registry has no hard and/or no PARTIAL entry, so this test would "
        "assert nothing: hard=%s partial=%s" % (hard[:3], partial[:3]))
    dead_camp = sorted(hard)[0]
    part_camp, dead_arms = sorted(partial)[0]

    d = tempfile.mkdtemp()
    try:
        # (a) a hard-quarantined campaign must be REFUSED, exit 1.
        p1 = os.path.join(d, "dead.csv")
        _cell_csv(p1, [_cell(dead_camp, "tralo", 0.51),
                       _cell(dead_camp, "tralo_null", 0.50),
                       _cell(dead_camp, "clip", 0.505)])
        code, body = _run_paper_rows("--cells", p1)
        assert code == 1, "a quarantined campaign was scored:\n%s" % body
        assert "REFUSING" in body and dead_camp in body, body

        # NEGATIVE CONTROL: --allow-quarantined must let it through, or the
        # refusal is a wall rather than a gate.
        code, body = _run_paper_rows("--cells", p1, "--allow-quarantined")
        assert code == 0, (
            "--allow-quarantined did not re-admit the campaign:\n%s" % body)

        # (b) a PARTIAL campaign: the DEAD arm's rows go, the live ones stay.
        p2 = os.path.join(d, "partial.csv")
        out2 = os.path.join(d, "partial_out.csv")
        rows = [_cell(part_camp, "tralo", 0.5120),
                _cell(part_camp, "tralo_null", 0.5060),
                _cell(part_camp, "clip", 0.5090)]
        for a in dead_arms:
            rows.append(_cell(part_camp, a, 0.5100))
        _cell_csv(p2, rows)
        code, body = _run_paper_rows("--cells", p2, "--out", out2)
        assert code == 0, body
        assert "PARTIAL QUARANTINE" in body, body
        written = io.open(out2, encoding="utf-8").read()
        for a in dead_arms:
            assert ",%s," % a not in written, (
                "a dead arm (%s) reached the paper rows:\n%s" % (a, written))
        assert ",tralo," in written, (
            "the PARTIAL drop took the LIVE arms with it -- that would delete "
            "an independent unit to describe a defect in two arms:\n%s"
            % written)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_paper_rows_NAMES_a_csv_that_is_not_a_cell_table():
    """2026-09-10: the negative control -- a wrong file must not raise."""
    import shutil

    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, "wrong.csv")
        io.open(p, "w", encoding="utf-8").write("a,b,c\n1,2,3\n")
        code, body = _run_paper_rows("--cells", p)
        assert code != 0, "a non-cell_table CSV was accepted:\n%s" % body
        assert "not a cell_table" in body, (
            "it failed, but did not SAY what was wrong -- which is the whole "
            "difference between a refusal and a traceback:\n%s" % body)
        for bad in FATAL_ON_AN_EMPTY_ROOT:
            assert bad + ":" not in body, (
                "refused with a %s traceback rather than a message:\n%s"
                % (bad, body))

        # NEGATIVE CONTROL: an EMPTY file is a different failure and must also
        # be named, not raise IndexError on rows[0].
        p2 = os.path.join(d, "empty.csv")
        io.open(p2, "w", encoding="utf-8").write("")
        code, body = _run_paper_rows("--cells", p2)
        assert code != 0 and "empty" in body.lower(), body
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_reachability_reads_the_cut_PER_GROUP_and_the_two_readings_differ():
    """2026-09-10: site 10 -- slope_at got the whole column, so K was global."""
    import numpy as np
    import pandas as pd

    from scripts.reachability import slope_at, slope_per_group

    # Two groups. A: 20 items, the allocator took 2, and they sit at p ~ 0.99
    # where p(1-p) is ~0.01 -- unreachable. B: 20 items, the allocator took 10,
    # cutting at p ~ 0.5 where the slope is at its maximum. Globally the top-12
    # is dominated by A's confident items, so the global reading says "flat"
    # about a cell in which one whole group's cut is live.
    pa = np.linspace(0.999, 0.980, 20)
    pb = np.linspace(0.600, 0.100, 20)
    p_col = np.concatenate([pa, pb])
    groups = np.array(["A"] * 20 + ["B"] * 20)
    pred = np.array([2] * 2 + [0] * 18 + [2] * 10 + [0] * 10)
    dep = pd.DataFrame({"Predicted_Label": pred, "Group_ID": groups})

    g_slope, g_p, n_g = slope_per_group(p_col, dep, 2, "sum")
    slope, p = slope_at(p_col, 12, "sum")
    assert n_g == 2, n_g
    assert g_slope != slope, (
        "the per-group and global readings coincide on a fixture built so "
        "they cannot: %.6f vs %.6f" % (g_slope, slope))
    assert g_slope > slope, (
        "on THIS fixture the deep group's live cut must lift the weighted "
        "slope above the global one: %.6f vs %.6f" % (g_slope, slope))
    # NEGATIVE CONTROL: no Group_ID must yield NaN so the caller falls back to
    # the global reading LABELLED as global, never silently.
    bad = pd.DataFrame({"Predicted_Label": pred})
    s2, p2, n2 = slope_per_group(p_col, bad, 2, "sum")
    assert n2 == 0 and s2 != s2 and p2 != p2, (s2, p2, n2)
    # NEGATIVE CONTROL: one group only -- the two readings must then AGREE,
    # because a single group's cut IS the global cut.
    one = pd.DataFrame({"Predicted_Label": pred,
                        "Group_ID": np.array(["A"] * 40)})
    s3, p3, n3 = slope_per_group(p_col, one, 2, "sum")
    assert n3 == 1 and abs(s3 - slope) < 1e-12, (s3, slope)
