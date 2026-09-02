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

    Twenty-two modules under `scripts/` and `configs/` carry `--self-test`.
    Each is the only thing standing between that tool and a silently wrong
    number, and NOTHING runs them together: they are invoked by hand, one at a
    time, when someone remembers. On 2026-09-02 a broken self-test fixture in
    `deployed_h2h` survived precisely because there was no sweep.

    This is the sweep. It also enforces the discovery half: a module that
    advertises `--self-test` in its argparse must actually implement it.
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
    "main_old.tex": 23,
}

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
