"""STAGE 3 -- THE MODEL, THE WARM-UP CACHE AND THE OPTIMISER PLUMBING.

All of it decidable BEFORE a training step runs, on CPU, in seconds. The five
measured incidents, cited again at each test:

  * the four backbones are a CLOSED list, `ViTB16` the headline fixed a priori
    2026-08-20 (FRAMEWORK 1-pre);
  * `base_model_id` must split on everything the warm-up optimizes -- it has
    failed four times, memorably by omitting the ImageNet-norm state;
  * `prm.grad` IS NOT THE DELIVERY MECHANISM, ADAM IS (FRAMEWORK 2(u),
    `scripts/ortho_survival.py`): `ortho_project` delivers 0.0% of its promised
    CE-neutrality in 16/16 conditions, and a MASKED coordinate still steps at
    90.4%, so `head_only` freezes nothing;
  * Adam's `(1-b1^k)` law is for CONSECUTIVE steps; at ~126 CE steps between
    constraint steps the multiplier is `(1-b1)/(1-b1^(c+1))` = 0.1000, forever;
  * FP16 + GradScaler SKIPS an overflowing step: `constraint_fp32: false` lands
    86.9% of the dose over 189 runs, `true` lands 15284/15284 over 532.

Every test collects all failures and reports once, and every gate carries a
NEGATIVE CONTROL in the same test -- conftest rules 2 and 3.
"""
import copy
import logging
import os
import re

import pytest
import torch

try:                          # pytest puts `tests/` on sys.path (gates/ is a pkg)
    from gates.conftest import (rel, read, load_yaml, report,
                                CLAIMED_BACKBONES, HEADLINE_BACKBONE)
except ImportError:           # ... unless tests/ ever becomes a package itself
    from tests.gates.conftest import (rel, read, load_yaml, report,
                                      CLAIMED_BACKBONES, HEADLINE_BACKBONE)

pytestmark = pytest.mark.stage3_model

B1 = 0.9                      # Adam beta1, everywhere in this pipeline
N_CLASSES = 8                 # iwildcam species
DELETED_BACKBONES = ("ShuffleNetV2", "TinyCNN", "SmallCNN", "MediumCNN")
# (head params, total params) at n_classes=8, as protocol.yml documents them
# beside `tralo_head`.
HEAD_SIZES = {"MobileNetV3": (10248, 4212280), "MobileNetV2": (10248, 2234120),
              "RegNetY400MF": (3528, 3906672), "ViTB16": (6152, 85804808)}
# Both sides of the split in one hp dict: every declared warm-up key, and the
# constraint-phase keys that must NOT split it -- or `tralo_null` stops sharing
# `tralo`'s warm-up and stops being its null.
WARMUP_HP = {"lr": 1e-4, "dropout": 0.3, "batch_size": 64, "warmup_epochs": 1,
             "pretrained": True, "class_weighted_ce": False, "seed": 1,
             "warmup_loss": "ce", "focal_alpha": 0.25, "focal_gamma": 2.0,
             "cb_beta": 0.999, "logit_adjust_tau": 1.0}
CONSTRAINT_HP = {"constraint_epochs": 29, "lambda_global": 0.01,
                 "lambda_local": 0.01, "lambda_step": 0.05, "initial_rho": 0.5,
                 "constraint_grad_clip": 1.0, "constraint_grad_mode": "clip",
                 "constraint_fp32": True, "lr_constraint": 1e-4,
                 "head_only": True, "soft_count_mode": "uniform",
                 "constraint_random_direction": True}
DC = {"data_dir": "data/iwildcam/oodslice", "num_classes": N_CLASSES}
_NEGATED = re.compile(r"\b(not|never|cannot|no|nothing|without)\b")


def _bump(v):
    """Perturb a value without changing its type."""
    if isinstance(v, bool):
        return not v
    return v + 1 if isinstance(v, (int, float)) else str(v) + "_x"


def _build(name, k=N_CLASSES):
    """A backbone on CPU with pretrained=False -- no download, no network."""
    from src.models import get_model
    return get_model(name, n_classes=k, dropout=0.3, pretrained=False)


def _freeze_claims(text):
    """(lineno, line) for every line asserting something is frozen. A line that
    NAMES freezing to DENY it is the correct description of `head_only` -- the
    measured backbone drift is 90.4% -- and must not be flagged."""
    return [(i, ln.strip()) for i, ln in enumerate(text.splitlines(), 1)
            if ("freez" in ln.lower() or "frozen" in ln.lower())
            and not _NEGATED.search(ln.lower())]


def test_the_backbone_list_is_closed_and_ViTB16_is_the_headline():
    """CLAUDE.md "Backbones", FRAMEWORK 1. Four backbones and no fifth;
    ShuffleNetV2 and the small CNNs were deleted 2026-08-18 BECAUSE they appear
    in no `.tex`, which is the property gated here."""
    from src.models.model_factory import MODEL_REGISTRY, get_model
    P, bad = load_yaml("configs", "protocol.yml"), []
    if set(MODEL_REGISTRY) != set(CLAIMED_BACKBONES):
        bad.append("registry is %s, claimed %s"
                   % (sorted(MODEL_REGISTRY), sorted(CLAIMED_BACKBONES)))
    if set(P["models"]) != set(CLAIMED_BACKBONES):
        bad.append("protocol.yml models: %s" % sorted(P["models"]))
    if HEADLINE_BACKBONE not in MODEL_REGISTRY:
        bad.append("headline %s is not registered" % HEADLINE_BACKBONE)
    tex = [f for f in os.listdir(rel("docs", "paper")) if f.endswith(".tex")]
    for name in DELETED_BACKBONES:
        if name in MODEL_REGISTRY:
            bad.append("%s is back in the registry" % name)
        bad += ["%s appears in docs/paper/%s" % (name, f) for f in tex
                if name.lower() in read("docs", "paper", f).lower()]
    # NEGATIVE CONTROL, all three checks: the factory REFUSES a deleted name,
    # the set comparison SEES an injected one, and the .tex scan read something.
    try:
        get_model("ShuffleNetV2", n_classes=N_CLASSES)
        bad.append("CONTROL: get_model accepted ShuffleNetV2")
    except ValueError:
        pass
    if set(list(MODEL_REGISTRY) + ["ShuffleNetV2"]) == set(CLAIMED_BACKBONES):
        bad.append("CONTROL: the closed-list check cannot see an extra backbone")
    if not tex:
        bad.append("CONTROL: no .tex found, the manuscript scan tested nothing")
    report(bad, "backbone-list defects")


def test_base_model_id_splits_on_the_warm_up_and_on_nothing_else():
    """A key that changes what the WARM-UP optimizes but is absent from
    `warmup_identity_keys` makes the second arm silently load the first arm's
    cached model -- four occurrences. The control runs BOTH ways: a
    constraint-phase key must NOT split the id, or the null stops being null."""
    from configs.gen_campaign import compute_base_model_id as bmid
    P, bad = load_yaml("configs", "protocol.yml"), []
    hp = dict(WARMUP_HP, **CONSTRAINT_HP)
    base = bmid(P, "ViTB16", hp, "multiclass", DC)
    declared = list(P["warmup_identity_keys"])
    for k in declared:
        if k not in hp:                     # a gate on this test's own table
            bad.append("declared key %r absent from WARMUP_HP, never tested" % k)
        elif bmid(P, "ViTB16", dict(hp, **{k: _bump(hp[k])}), "multiclass", DC) == base:
            bad.append("%s does NOT change base_model_id" % k)
    bad += ["%s changes the warm-up but is undeclared" % k
            for k in WARMUP_HP if k not in declared]
    if bmid(P, "MobileNetV3", hp, "multiclass", DC) == base:
        bad.append("model_name does NOT change base_model_id")
    if bmid(P, "ViTB16", hp, "single_class", DC) == base:
        bad.append("dataset_mode does NOT change base_model_id")
    for k in ("data_dir", "num_classes"):
        if bmid(P, "ViTB16", hp, "multiclass", dict(DC, **{k: _bump(DC[k])})) == base:
            bad.append("%s does NOT change base_model_id" % k)
    # NEGATIVE CONTROL: the constraint phase runs after the warm-up, so none of
    # its knobs may split the cache.
    bad += ["CONTROL: constraint key %s splits the warm-up cache" % k
            for k in CONSTRAINT_HP
            if bmid(P, "ViTB16", dict(hp, **{k: _bump(hp[k])}), "multiclass", DC) != base]
    report(bad, "base_model_id identity defects")


def test_the_smoke_net_liveness_verdict_does_not_transfer():
    """`scripts/hp_liveness_real.py`. On the smoke net the clip never engages,
    so lambda/rho read LIVE and `constraint_grad_clip` reads INERT; on ViTB16
    both invert. The smoke tool must state its own invalidity, and the real
    probe must move constraint-phase knobs only -- a probe on a warm-up key
    would compare a model against itself."""
    from scripts.hp_liveness_real import PROBES
    warm = set(load_yaml("configs", "protocol.yml")["warmup_identity_keys"])
    bad = []
    if "do NOT transfer" not in read("scripts", "hp_liveness.py"):
        bad.append("hp_liveness.py no longer says its magnitude verdicts do not "
                   "transfer to a real backbone")
    real = read("scripts", "hp_liveness_real.py").lower()
    bad += ["hp_liveness_real no longer records %r" % p
            for p in ("smoke", "invert", "clip bound") if p not in real]
    bad += ["probe %r moves warm-up key(s) %s" % (lab, sorted(set(ov) & warm))
            for lab, ov in PROBES if set(ov) & warm]
    if not PROBES:
        bad.append("PROBES is empty, the sweep above tested nothing")
    # NEGATIVE CONTROL: the same predicate must catch a probe on `dropout`.
    if not set({"dropout": 0.6}) & warm:
        bad.append("CONTROL: the warm-up-key check cannot see a dropout probe")
    report(bad, "liveness-probe defects")


def test_the_cache_refuses_a_warm_up_from_another_regime_commit_or_slice(
        tmp_path, monkeypatch, caplog):
    """`src/training/model_cache.py`. `base_model_id` hashes hyperparameters,
    not code and not data, so a cache written before a change to what the
    warm-up optimizes is silently wrong -- how the pre-ImageNet-norm caches
    survived a norm change. Three later gates catch it: the AMP regime
    (dsisco01 FP16 vs dsisco02 BF16, ONE shared NFS cache dir), the data
    fingerprint, and the runner's commit stamp. Each must refuse or SAY it
    could not run; the identical payload is the negative control."""
    from src.training import model_cache as mc
    monkeypatch.setenv("OPTLOSS_MODEL_CACHE", str(tmp_path))
    monkeypatch.setattr(mc, "get_model", lambda *a, **k: torch.nn.Linear(4, N_CLASSES))
    monkeypatch.setattr(mc, "_amp_regime", lambda: "float16|scaler=True")
    bmid = "MobileNetV3_multiclass_deadbeef"
    cfg = {"model_name": "MobileNetV3", "hyperparams": {"dropout": 0.3},
           "code_version": "cafe1", "run_code_version": "beef1",
           "data_fingerprint": "fp1"}
    good = {"model_state_dict": torch.nn.Linear(4, N_CLASSES).state_dict(),
            "base_model_id": bmid, "code_version": "cafe1",
            "run_code_version": "beef1", "data_fingerprint": "fp1",
            "amp_regime": "float16|scaler=True"}
    drop, bad = object(), []
    for label, patch, want_load, want_log in [
            ("identical (NEGATIVE CONTROL)", {}, True, None),
            ("id mismatch", {"base_model_id": "other"}, False, None),
            ("BF16 cache on an FP16 host", {"amp_regime": "bf16|scaler=False"},
             False, "AMP regime"),
            ("slice changed under the path", {"data_fingerprint": "fp2"},
             False, "fingerprint"),
            ("trained by another commit", {"run_code_version": "beef2"},
             False, "run_code_version"),
            ("no runner stamp, generator differs",
             {"run_code_version": drop, "code_version": "cafe2"}, False, "code_version"),
            ("no runner stamp, generator agrees", {"run_code_version": drop},
             True, "falling back"),
            ("cache predates amp_regime", {"amp_regime": drop}, True, "predates")]:
        payload = dict(good)
        for k, v in patch.items():
            payload.pop(k, None) if v is drop else payload.__setitem__(k, v)
        torch.save(payload, mc.get_cache_path(bmid))
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="src.training.model_cache"):
            got = mc.load_from_cache(bmid, cfg, N_CLASSES, torch.device("cpu"))
        if (got is not None) != want_load:
            bad.append("%s: loaded=%s, wanted %s" % (label, got is not None, want_load))
        if want_log and want_log not in caplog.text:
            bad.append("%s: said nothing about %r" % (label, want_log))
    # NEGATIVE CONTROL for the SILENCE, the defect these log lines fix: when
    # this process cannot name its own AMP regime the check does not run, and it
    # must announce that rather than pass quietly.
    torch.save(dict(good), mc.get_cache_path(bmid))
    monkeypatch.setattr(mc, "_amp_regime", lambda: None)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="src.training.model_cache"):
        mc.load_from_cache(bmid, cfg, N_CLASSES, torch.device("cpu"))
    if "DID NOT RUN" not in caplog.text:
        bad.append("CONTROL: an undeterminable AMP regime skipped the check "
                   "SILENTLY -- the exact shape of the old defect")
    report(bad, "warm-up cache defects")


def test_head_only_identifies_exactly_one_head_on_all_four_backbones():
    """`head_parameter_ids`. The backbones name the head differently
    (`classifier`, `fc`, `heads`), so the rule is "the one Linear with
    out_features == n_classes" and it must REFUSE on ambiguity: a wrong choice
    confines the constraint to the wrong parameters while every config and log
    still read `head_only: true`. Sizes from protocol.yml's `tralo_head`."""
    from src.training.constraint_step import head_parameter_ids
    bad = []
    for name in CLAIMED_BACKBONES:
        try:
            model = _build(name)
        except Exception as exc:            # a download attempt, not a defect
            pytest.skip("%s could not be built offline (%s) -- gate NOT run"
                        % (name, type(exc).__name__))
        try:
            ids = head_parameter_ids(model, N_CLASSES)
        except ValueError as exc:
            bad.append("%s: %s" % (name, exc))
            continue
        head = sum(p.numel() for p in model.parameters() if id(p) in ids)
        total = sum(p.numel() for p in model.parameters())
        if (head, total) != HEAD_SIZES[name]:
            bad.append("%s head/total %d/%d, protocol.yml documents %d/%d"
                       % ((name, head, total) + HEAD_SIZES[name]))
        if len(ids) != 2:
            bad.append("%s head is %d tensors, want weight + bias" % (name, len(ids)))
    # NEGATIVE CONTROL, both directions of ambiguity. MobileNetV3's penultimate
    # Linear is 960 -> 1280, so at n_classes=1280 TWO Linears match; and a
    # backbone asked for a class count it does not emit matches none.
    for label, model, k in (("two matching Linears", _build("MobileNetV3", 1280), 1280),
                            ("no matching Linear", _build("MobileNetV2"), 3)):
        try:
            head_parameter_ids(model, k)
            bad.append("CONTROL: %s did not raise -- the constraint would land "
                       "on an arbitrary layer" % label)
        except ValueError:
            pass
    report(bad, "head-identification defects")


def test_masking_a_gradient_does_not_freeze_the_parameter():
    """FRAMEWORK 2(u), `scripts/ortho_survival.py`. `head_only` zeroes the
    backbone's grad, but Adam carries `m <- 0.9*m + 0.1*0`, so a masked
    coordinate still steps at 90.4% of an unmasked one at 126 CE steps/epoch.
    The arm is "the constraint sees only the head", NEVER "the backbone is
    frozen", and no docstring may say otherwise."""
    from scripts.ortho_survival import masked_coordinate_drift
    bad = []
    ratio = masked_coordinate_drift(ce_steps=126)[2]
    if not 0.88 <= ratio <= 0.92:
        bad.append("masked/unmasked ratio at 126 CE steps is %.4f, measured "
                   "0.904" % ratio)
    prev = -1.0
    for steps in (1, 3, 126):               # more CE steps => LESS effective
        r = masked_coordinate_drift(ce_steps=steps)[2]
        if r <= prev:
            bad.append("ratio not rising with CE steps: %d -> %.4f" % (steps, r))
        prev = r
    # NEGATIVE CONTROL: with no stale momentum a mask DOES freeze the
    # coordinate, so 0.904 is the momentum and not an artefact of the harness.
    zero = masked_coordinate_drift(ce_steps=0)[2]
    if abs(zero) > 1e-9:
        bad.append("CONTROL: at 0 CE steps the masked coordinate moved (%.3g), "
                   "so this is not isolating momentum" % zero)
    for parts in (("src", "training", "constraint_step.py"),
                  ("configs", "protocol.yml")):
        bad += ["%s:%d claims freezing: %s" % ("/".join(parts), i, ln[:60])
                for i, ln in _freeze_claims(read(*parts))]
    # NEGATIVE CONTROL for the prose scan, both ways.
    if not _freeze_claims("head_only freezes the backbone in the constraint phase."):
        bad.append("CONTROL: the prose scan cannot see an un-negated freezing "
                   "claim, so its silence above means nothing")
    if _freeze_claims("zeroing a gradient does not freeze the parameter."):
        bad.append("CONTROL: the prose scan flags a NEGATED line, so it fires on "
                   "every correct description of this arm")
    report(bad, "head_only / masking defects")


def test_the_ortho_projection_does_not_survive_adam():
    """FRAMEWORK 2(u), `scripts/ortho_survival.py`. `project_out` sets
    `<g_con, ref> = 0` exactly, which to first order is a claim of
    CE-neutrality. Adam voids it twice: 92.6% of the momentum is stale CE the
    projection never touches, and `sqrt(v)` is not an isometry. Measured
    removal 0.0% in 16/16. The negative control is plain SGD
    (`constraint_step_rule: sgd`), which DOES preserve the zero -- so the loss
    is Adam's, not the harness's."""
    from src.training.constraint_step import project_out
    from scripts.ortho_survival import removal_fraction
    torch.manual_seed(0)
    n = 64
    ref, g = torch.randn(n), torch.randn(n)
    mod = torch.nn.Linear(n, 1, bias=False)
    mod.weight.grad = g.view(1, n).clone()
    project_out(mod, [ref.view(1, n)])
    g_proj = mod.weight.grad.view(-1).clone()

    def adam_delta(grad, ce_steps=126):
        p = torch.nn.Parameter(torch.zeros(n))
        opt = torch.optim.Adam([p], lr=1e-3)
        for _ in range(ce_steps):           # the CE phase builds the momentum
            opt.zero_grad()
            p.grad = ref.clone()
            opt.step()
        before = p.detach().clone()
        opt.zero_grad()
        p.grad = grad.clone()
        opt.step()
        return p.detach() - before

    def cos(a, b):
        return float(torch.dot(a, b) / (a.norm() * b.norm()))

    bad, scale = [], float(g.norm() * ref.norm())
    if abs(float(torch.dot(g_proj, ref))) / scale > 1e-5:
        bad.append("project_out did not zero <g, ref> at the GRADIENT level")
    if abs(float(torch.dot(g, ref))) / scale < 1e-3:
        bad.append("CONTROL: the raw gradient was already orthogonal to ref")
    removed = 1.0 - abs(cos(adam_delta(g_proj), ref)) / abs(cos(adam_delta(g), ref))
    if abs(removed) > 0.05:
        bad.append("the projection removed %.1f%% of the delivered CE alignment; "
                   "measured is 0.0%%" % (100 * removed))
    # NEGATIVE CONTROL: p -= lr*g keeps the guarantee exactly.
    if abs(cos(-1e-3 * g_proj, ref)) > 1e-5:
        bad.append("CONTROL: plain SGD also destroyed the orthogonality, so the "
                   "finding is not attributable to Adam")
    if not removal_fraction(1.4) < 0.08 < removal_fraction(0.01):
        bad.append("removal_fraction is not monotone in |m_CE|/|g_con|: %.4f at "
                   "1.4, %.4f at 0.01"
                   % (removal_fraction(1.4), removal_fraction(0.01)))
    report(bad, "ortho_project survival defects")


def test_the_constraint_step_multiplier_is_the_single_step_value_forever():
    """FRAMEWORK 2(u), `ortho_survival.count_change_compounding`. The retracted
    analysis applied `(1-b1^k)`, the accumulation over CONSECUTIVE steps.
    `tralo/train.py` calls `finish_constraint_step` ONCE per epoch with the
    whole CE batch loop between, so c = ~126 and the difference present at a
    constraint step is `(1-b1)/(1-b1^(c+1))` = 0.1000, forever. The negative
    control is c=0, where the same formula returns 1.000."""
    def present(c):
        return (1 - B1) / (1 - B1 ** (c + 1))

    bad = []
    if abs(B1 ** 126 - 1.7e-6) > 1e-7:
        bad.append("b1^126 is %.3g, documented as 1.7e-6" % B1 ** 126)
    if abs(present(126) - 0.1000) > 1e-4:
        bad.append("multiplier at c=126 is %.4f, not 0.1000" % present(126))
    if abs(present(126) - (1 - B1)) > 1e-6:
        bad.append("the c=126 multiplier is not the single-step value")
    if abs(present(0) - 1.0) > 1e-9:
        bad.append("CONTROL: at c=0 the multiplier is %.4f, not 1.000 -- the "
                   "formula is not responding to c" % present(0))
    retracted = 1 - B1 ** 29
    if not 9.0 < retracted / present(126) < 10.0:
        bad.append("CONTROL: the retracted (1-b1^k) law is %.4f at k=29, only "
                   "%.1fx the correct value" % (retracted, retracted / present(126)))
    src = read("src", "methodologies", "tralo", "train.py")   # the 126 is real
    if src.count("finish_constraint_step(") != 1:
        bad.append("tralo/train.py calls finish_constraint_step %d times; the "
                   "arithmetic assumes one per epoch"
                   % src.count("finish_constraint_step("))
    report(bad, "Adam accumulation defects")


def test_a_non_finite_constraint_gradient_is_detected_not_silently_dropped(protocol):
    """FRAMEWORK 2(u), the sixth defect class: the treatment that reports
    `completed` and never landed. FP16 + GradScaler SKIPS an overflowing step,
    so `constraint_fp32: false` landed 4684/5393 = 86.9% over 189 runs while
    `true` landed 15284/15284 over 532. So the step must RETURN whether it
    applied, the fp32 path must bypass the scaler, and `gen_campaign` must
    refuse a trained campaign without the flag."""
    from src.training.constraint_step import (snapshot_grads, constraint_backward,
                                              finish_constraint_step)
    from configs.gen_campaign import fp32_gate
    bad = []
    mod = torch.nn.Linear(4, 3)
    opt = torch.optim.Adam(mod.parameters(), lr=1e-3)
    for p in mod.parameters():
        p.grad = torch.full_like(p, float("nan"))
    if snapshot_grads(mod) is not None:
        bad.append("snapshot_grads built a reference from non-finite grads")
    before = [p.detach().clone() for p in mod.parameters()]
    if finish_constraint_step(mod, opt, None, 1.0)[1]:
        bad.append("finish_constraint_step reported a NaN step as APPLIED")
    if any(not torch.equal(p.detach(), b) for p, b in zip(mod.parameters(), before)):
        bad.append("a NaN constraint step moved the parameters")
    # NEGATIVE CONTROL: a finite gradient must land, and be seen to land.
    for p in mod.parameters():
        p.grad = torch.ones_like(p)
    if snapshot_grads(mod) is None:
        bad.append("CONTROL: snapshot_grads rejected a finite gradient")
    before = [p.detach().clone() for p in mod.parameters()]
    if not finish_constraint_step(mod, opt, None, 1.0)[1] or all(
            torch.equal(p.detach(), b) for p, b in zip(mod.parameters(), before)):
        bad.append("CONTROL: a finite constraint step did not land, so the check "
                   "above cannot tell a dropped step from a taken one")

    class FakeScaler:                       # records whether the scaler was used
        def __init__(self):
            self.scaled = 0

        def scale(self, loss):
            self.scaled += 1
            return loss

    for fp32, want in ((True, 0), (False, 1)):   # fp32 bypasses the CE loss scale
        s = FakeScaler()
        w = torch.nn.Parameter(torch.ones(3))
        constraint_backward((w * w).sum(), s, fp32)
        if s.scaled != want:
            bad.append("constraint_backward(fp32=%s) used the scaler %d time(s), "
                       "wanted %d" % (fp32, s.scaled, want))

    P = copy.deepcopy(protocol)
    trained = [a for a, s in P["arms"].items() if s.get("phase") != "posthoc"][:1]
    posthoc = [a for a, s in P["arms"].items() if s.get("phase") == "posthoc"][:1]
    P["constraint_phase"]["constraint_fp32"] = False
    try:
        fp32_gate(P, type("A", (), {"allow_fp16_constraint": False})(), trained)
        bad.append("gen_campaign accepted a trained campaign with "
                   "constraint_fp32: false -- ~13%% of the dose")
    except SystemExit:
        pass
    # NEGATIVE CONTROL, three ways the refusal must stay quiet.
    for label, flag, arms in (("fp32 on", True, trained),
                              ("post-hoc only", False, posthoc),
                              ("explicit override", False, trained)):
        P["constraint_phase"]["constraint_fp32"] = flag
        args = type("A", (), {"allow_fp16_constraint": label == "explicit override"})()
        try:
            fp32_gate(P, args, arms)
        except SystemExit:
            bad.append("CONTROL: fp32_gate refused a legal campaign (%s)" % label)
    if not trained or not posthoc:
        bad.append("CONTROL: protocol.yml has no %s arm, that case was skipped"
                   % ("trained" if not trained else "post-hoc"))
    report(bad, "constraint-dose defects")


# ==========================================================================
#   THE DE-SATURATION KNOB. `--pretrained` overrides `core.pretrained` for a
#   pilot that asks whether an unsaturated model can separate the arms at all.
#   Source: configs/gen_campaign.build_hyperparams, 2026-09-04.
# ==========================================================================
def test_the_pretrained_override_splits_the_warm_up_cache_and_is_off_by_default():
    """A knob that changes what the warm-up optimises MUST change
    `base_model_id`, or the second regime silently loads the first one's
    cached model. That has happened four times in this project, which is why
    `warmup_identity_keys` exists.

    `pretrained` was already in that list before this flag was added, so the
    split is inherited rather than newly asserted -- but it is asserted here
    anyway, because the flag is what makes the key reachable from the command
    line and a later tidy-up of the list would now break a campaign design
    rather than only a latent property.

    NEGATIVE CONTROLS, both directions:
      * omitting the flag must leave the hyperparameters BYTE-IDENTICAL, or
        every future campaign silently changed the day this landed;
      * flipping it must change nothing EXCEPT `pretrained` itself, or the
        override is dragging an unrelated key along with it;
      * and the two regimes must NOT collide in the cache, or the pilot's
        two halves are one half measured twice.
    """
    try:
        from configs.gen_campaign import (build_hyperparams,
                                          compute_base_model_id)
    except ImportError:                                       # pragma: no cover
        pytest.skip("configs/ is frozen at a commit predating the flag")

    P = load_yaml(rel("configs", "protocol.yml"))
    dc = {"data_dir": "data/iwildcam/oodslice", "num_classes": N_CLASSES}
    bad = []

    for arm in ("tralo", "tralo_null", "clip", "alm", "fioretto"):
        spec = P["arms"][arm]
        base = build_hyperparams(P, spec, 1)
        same = build_hyperparams(P, spec, 1, pretrained=None)
        off = build_hyperparams(P, spec, 1, pretrained=False)

        if base != same:
            bad.append("%s: omitting --pretrained changed the hyperparameters, "
                       "so every existing campaign design just moved" % arm)
        diff = {k for k in set(base) | set(off) if base.get(k) != off.get(k)}
        if diff != {"pretrained"}:
            bad.append("%s: the override touched %s, expected only "
                       "{'pretrained'}" % (arm, sorted(diff)))
        if off.get("pretrained") is not False:
            bad.append("%s: --pretrained false did not reach this arm (%r). "
                       "A flag reaching only the trained arms would leave the "
                       "post-hoc baseline as the only one with ImageNet "
                       "features" % (arm, off.get("pretrained")))

        for model in CLAIMED_BACKBONES:
            a = compute_base_model_id(P, model, base, "iwildcam", dc)
            b = compute_base_model_id(P, model, off, "iwildcam", dc)
            if a == b:
                bad.append(
                    "%s/%s: the two pretraining regimes share base_model_id "
                    "%s, so the second one loads the first one's cached "
                    "warm-up and the pilot measures one model twice"
                    % (model, arm, a))

    # 🛑 THE ARGPARSE PATH, END TO END, AND THIS IS THE HALF THAT
    # ACTUALLY FAILED. The first version of the flag carried
    # `type=lambda v: v.lower()`, so `--pretrained false` arrived as the STRING
    # "false" -- and `bool("false")` is True. It emitted 48 configs at
    # `pretrained: True` while every assertion above passed, because they call
    # `build_hyperparams` with a real Python bool and never touch the parser.
    # A flag is only live once the value a USER types reaches the config.
    import configs.gen_campaign as gc

    class _A:                       # the shape argparse produces, nothing more
        def __init__(self, v):
            self.pretrained = v

    for typed, want in (("false", False), ("true", True), (None, None)):
        got = gc._pretrained(_A(typed))
        if got is not want:
            bad.append("--pretrained %r parsed to %r, expected %r; "
                       "bool('false') is True and that is how flag six went "
                       "inert" % (typed, got, want))
    try:
        gc._pretrained(_A("False"))
    except ValueError:
        pass
    else:
        bad.append("_pretrained accepted 'False' silently; anything argparse "
                   "did not validate must RAISE, not be guessed at")

    # And the whole way through to a hyperparameter dict, which is the thing
    # that is actually written to disk.
    hp_off = build_hyperparams(P, P["arms"]["tralo"], 1,
                               pretrained=gc._pretrained(_A("false")))
    if hp_off.get("pretrained") is not False:
        bad.append("the typed string 'false' reached the config as %r"
                   % hp_off.get("pretrained"))

    # `pretrained` must still be a DECLARED warm-up identity key. If it is ever
    # dropped from that list the split above vanishes silently.
    if "pretrained" not in (P.get("warmup_identity_keys") or []):
        bad.append("`pretrained` left warmup_identity_keys; the override no "
                   "longer splits the cache and the pilot is unrunnable")

    report(bad, "pretraining-override failures")
