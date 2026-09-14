"""Maintained behavioral regression fixtures."""

import logging

import pytest
from .conftest import load_yaml, rel, read, report

import torch

pytestmark = pytest.mark.stage3_model

B1 = 0.9

N_CLASSES = 8

WARMUP_HP = {
    "lr": 0.0001,
    "dropout": 0.3,
    "batch_size": 64,
    "warmup_epochs": 1,
    "pretrained": True,
    "class_weighted_ce": False,
    "seed": 1,
    "warmup_loss": "ce",
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
}

CONSTRAINT_HP = {
    "constraint_epochs": 29,
    "lambda_global": 0.01,
    "lambda_local": 0.01,
    "lambda_step": 0.05,
    "initial_rho": 0.5,
    "constraint_grad_clip": 1.0,
    "constraint_grad_mode": "clip",
    "constraint_fp32": True,
    "lr_constraint": 0.0001,
}

DC = {"data_dir": "data/iwildcam/oodslice", "num_classes": N_CLASSES}


def _bump(v):
    if isinstance(v, bool):
        return not v
    return v + 1 if isinstance(v, (int, float)) else str(v) + "_x"


def test_base_model_id_splits_on_the_warm_up_and_on_nothing_else():
    from configs.gen_campaign import compute_base_model_id as bmid

    (P, bad) = (load_yaml("configs", "protocol.yml"), [])
    hp = dict(WARMUP_HP, **CONSTRAINT_HP)
    base = bmid(P, "ViTB16", hp, "multiclass", DC)
    declared = list(P["warmup_identity_keys"])
    for k in declared:
        if k not in hp:
            bad.append("declared key %r absent from WARMUP_HP, never tested" % k)
        elif bmid(P, "ViTB16", dict(hp, **{k: _bump(hp[k])}), "multiclass", DC) == base:
            bad.append("%s does NOT change base_model_id" % k)
    bad += [
        "%s changes the warm-up but is undeclared" % k
        for k in WARMUP_HP
        if k not in declared
    ]
    if bmid(P, "MobileNetV3", hp, "multiclass", DC) == base:
        bad.append("model_name does NOT change base_model_id")
    if bmid(P, "ViTB16", hp, "single_class", DC) == base:
        bad.append("dataset_mode does NOT change base_model_id")
    for k in ("data_dir", "num_classes"):
        if bmid(P, "ViTB16", hp, "multiclass", dict(DC, **{k: _bump(DC[k])})) == base:
            bad.append("%s does NOT change base_model_id" % k)
    bad += [
        "CONTROL: constraint key %s splits the warm-up cache" % k
        for k in CONSTRAINT_HP
        if bmid(P, "ViTB16", dict(hp, **{k: _bump(hp[k])}), "multiclass", DC) != base
    ]
    report(bad, "base_model_id identity defects")


def test_the_cache_refuses_a_warm_up_from_another_regime_commit_or_slice(
    tmp_path, monkeypatch, caplog
):
    from src.training import model_cache as mc

    monkeypatch.setenv("OPTLOSS_MODEL_CACHE", str(tmp_path))
    monkeypatch.setattr(mc, "get_model", lambda *a, **k: torch.nn.Linear(4, N_CLASSES))
    monkeypatch.setattr(mc, "_amp_regime", lambda: "float16|scaler=True")
    bmid = "MobileNetV3_multiclass_deadbeef"
    cfg = {
        "model_name": "MobileNetV3",
        "hyperparams": {"dropout": 0.3},
        "code_version": "cafe1",
        "run_code_version": "beef1",
        "data_fingerprint": "fp1",
    }
    good = {
        "model_state_dict": torch.nn.Linear(4, N_CLASSES).state_dict(),
        "base_model_id": bmid,
        "code_version": "cafe1",
        "run_code_version": "beef1",
        "data_fingerprint": "fp1",
        "amp_regime": "float16|scaler=True",
    }
    (drop, bad) = (object(), [])
    for label, patch, want_load, want_log in [
        ("identical (NEGATIVE CONTROL)", {}, True, None),
        ("id mismatch", {"base_model_id": "other"}, False, None),
        (
            "BF16 cache on an FP16 host",
            {"amp_regime": "bf16|scaler=False"},
            False,
            "AMP regime",
        ),
        (
            "slice changed under the path",
            {"data_fingerprint": "fp2"},
            False,
            "fingerprint",
        ),
        (
            "trained by another commit",
            {"run_code_version": "beef2"},
            False,
            "run_code_version",
        ),
        (
            "no runner stamp, generator differs",
            {"run_code_version": drop, "code_version": "cafe2"},
            False,
            "code_version",
        ),
        (
            "no runner stamp, generator agrees",
            {"run_code_version": drop},
            True,
            "falling back",
        ),
        ("cache predates amp_regime", {"amp_regime": drop}, True, "predates"),
    ]:
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
    torch.save(dict(good), mc.get_cache_path(bmid))
    monkeypatch.setattr(mc, "_amp_regime", lambda: None)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="src.training.model_cache"):
        mc.load_from_cache(bmid, cfg, N_CLASSES, torch.device("cpu"))
    if "DID NOT RUN" not in caplog.text:
        bad.append(
            "CONTROL: an undeterminable AMP regime skipped the check SILENTLY -- the exact shape of the old defect"
        )
    report(bad, "warm-up cache defects")


def test_the_constraint_step_multiplier_is_the_single_step_value_forever():

    def present(c):
        return (1 - B1) / (1 - B1 ** (c + 1))

    bad = []
    if abs(B1**126 - 1.7e-06) > 1e-07:
        bad.append("b1^126 is %.3g, documented as 1.7e-6" % B1**126)
    if abs(present(126) - 0.1) > 0.0001:
        bad.append("multiplier at c=126 is %.4f, not 0.1000" % present(126))
    if abs(present(126) - (1 - B1)) > 1e-06:
        bad.append("the c=126 multiplier is not the single-step value")
    if abs(present(0) - 1.0) > 1e-09:
        bad.append(
            "CONTROL: at c=0 the multiplier is %.4f, not 1.000 -- the formula is not responding to c"
            % present(0)
        )
    retracted = 1 - B1**29
    if not 9.0 < retracted / present(126) < 10.0:
        bad.append(
            "CONTROL: the retracted (1-b1^k) law is %.4f at k=29, only %.1fx the correct value"
            % (retracted, retracted / present(126))
        )
    src = read("src", "methodologies", "tralo", "train.py")
    if src.count("finish_constraint_step(") != 1:
        bad.append(
            "tralo/train.py calls finish_constraint_step %d times; the arithmetic assumes one per epoch"
            % src.count("finish_constraint_step(")
        )
    report(bad, "Adam accumulation defects")


def test_a_non_finite_constraint_gradient_is_detected_not_silently_dropped(protocol):
    from src.training.constraint_step import constraint_backward, finish_constraint_step

    for finite in (False, True):
        model = torch.nn.Linear(4, 3)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        for p in model.parameters():
            p.grad = torch.ones_like(p) if finite else torch.full_like(p, float("nan"))
        before = [p.detach().clone() for p in model.parameters()]
        (_, applied) = finish_constraint_step(model, opt, None, 1.0)
        changed = any(
            (not torch.equal(p, b) for (p, b) in zip(model.parameters(), before))
        )
        assert applied == changed == finite

    class Scaler:
        calls = 0

        def scale(self, loss):
            self.calls += 1
            return loss

    for fp32 in (False, True):
        scaler = Scaler()
        p = torch.nn.Parameter(torch.ones(3))
        constraint_backward(p.square().sum(), scaler, fp32)
        assert scaler.calls == int(not fp32)
        assert torch.equal(p.grad, torch.full((3,), 2.0))


def test_the_pretrained_override_splits_the_warm_up_cache_and_is_off_by_default():
    try:
        from configs.gen_campaign import build_hyperparams, compute_base_model_id
    except ImportError:
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
            bad.append(
                "%s: omitting --pretrained changed the hyperparameters, so every existing campaign design just moved"
                % arm
            )
        diff = {k for k in set(base) | set(off) if base.get(k) != off.get(k)}
        if diff != {"pretrained"}:
            bad.append(
                "%s: the override touched %s, expected only {'pretrained'}"
                % (arm, sorted(diff))
            )
        if off.get("pretrained") is not False:
            bad.append(
                "%s: --pretrained false did not reach this arm (%r). A flag reaching only the trained arms would leave the post-hoc baseline as the only one with ImageNet features"
                % (arm, off.get("pretrained"))
            )
    for model in P["models"]:
        a = compute_base_model_id(P, model, base, "iwildcam", dc)
        b = compute_base_model_id(P, model, off, "iwildcam", dc)
        if a == b:
            bad.append(
                "%s/%s: the two pretraining regimes share base_model_id %s, so the second one loads the first one's cached warm-up and the pilot measures one model twice"
                % (model, arm, a)
            )
    import json
    import tempfile
    from pathlib import Path
    from test_lean_protocol import generate

    with tempfile.TemporaryDirectory() as root:
        for typed, want in (("false", False), ("true", True)):
            dest = Path(root) / typed
            result = generate(dest, "--pretrained", typed)
            assert result.returncode == 0, result.stderr
            configs = [json.loads(p.read_text()) for p in dest.rglob("config.json")]
            assert configs and all(
                (c["hyperparams"]["pretrained"] is want for c in configs)
            )
        assert generate(Path(root) / "invalid", "--pretrained", "False").returncode != 0
    if "pretrained" not in (P.get("warmup_identity_keys") or []):
        bad.append(
            "`pretrained` left warmup_identity_keys; the override no longer splits the cache and the pilot is unrunnable"
        )
    report(bad, "pretraining-override failures")
