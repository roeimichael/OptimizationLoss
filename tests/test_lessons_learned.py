"""Maintained behavioral regression fixtures."""

import ast

import io

import os

import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rel(*p):
    return os.path.join(REPO, *p)


def read(*p):
    with io.open(rel(*p), encoding="utf-8", errors="replace") as f:
        return f.read()


def test_a_zero_lambda_arm_still_RUNS_its_constraint_epochs():
    from lean_fixtures import protocol_with_nulls
    from configs.gen_campaign import build_hyperparams

    P = protocol_with_nulls()
    arms = P["arms"]

    def epochs(a):
        return build_hyperparams(P, arms[a], 1)["constraint_epochs"]

    bad = []
    pairs = [
        (a, a + "_null")
        for a in ("tralo", "alm", "fioretto", "hounie")
        if a in arms and a + "_null" in arms
    ]
    assert pairs, "no treated/null pair found; the protocol shape changed"
    for treated, null in pairs:
        (et, en) = (epochs(treated), epochs(null))
        if et is None or en is None:
            continue
        if et != en:
            bad.append(
                "%s runs %s constraint epochs but %s runs %s -- a null that trains for a different length is not a null, it is a second regime"
                % (treated, et, null, en)
            )
    for fam in ("tralo", "alm", "fioretto", "hounie"):
        if fam in arms and fam + "_null" not in arms:
            bad.append("%s has no _null sibling" % fam)
    assert not bad, "zero-lambda control:\n  " + "\n  ".join(bad)


def test_bf16_is_gated_on_COMPUTE_CAPABILITY_and_turing_gets_a_scaler():
    torch = pytest.importorskip("torch")
    from src.pipeline import setup as S

    class _Dev:
        type = "cuda"

    seen = {}

    def run(major, bf16_supported):
        seen.clear()
        orig = (
            torch.cuda.get_device_capability,
            torch.cuda.is_bf16_supported,
            torch.backends.cudnn.benchmark,
        )
        torch.cuda.get_device_capability = lambda *a, **k: (major, 0)
        torch.cuda.is_bf16_supported = lambda *a, **k: bf16_supported
        torch.backends.cudnn.benchmark = True
        try:
            return S.setup_runtime(_Dev())
        finally:
            (
                torch.cuda.get_device_capability,
                torch.cuda.is_bf16_supported,
                torch.backends.cudnn.benchmark,
            ) = orig

    bad = []
    (use, dt, scaler) = run(8, True)
    if not (use and dt is torch.bfloat16 and (scaler is None)):
        bad.append(
            "capability 8 gave (%s, %s, scaler=%s); Blackwell must be bf16 with NO scaler"
            % (use, dt, scaler is not None)
        )
    (use, dt, scaler) = run(7, False)
    if not (use and dt is torch.float16 and (scaler is not None)):
        bad.append(
            "capability 7 gave (%s, %s, scaler=%s); Turing must be fp16 WITH a GradScaler"
            % (use, dt, scaler is not None)
        )
    if dt is torch.bfloat16:
        bad.append(
            "CONTROL: capability 7 came back bfloat16 -- the capability gate is not reading the capability"
        )
    assert not bad, "AMP selection:\n  " + "\n  ".join(bad)


def test_cudnn_benchmark_is_forced_OFF_every_time_the_runtime_is_configured():
    torch = pytest.importorskip("torch")
    from src.pipeline import setup as S

    class _Dev:
        type = "cuda"

    orig = (
        torch.cuda.get_device_capability,
        torch.cuda.is_bf16_supported,
        torch.backends.cudnn.benchmark,
    )
    torch.cuda.get_device_capability = lambda *a, **k: (8, 0)
    torch.cuda.is_bf16_supported = lambda *a, **k: True
    torch.backends.cudnn.benchmark = True
    try:
        S.setup_runtime(_Dev())
        after = torch.backends.cudnn.benchmark
    finally:
        (
            torch.cuda.get_device_capability,
            torch.cuda.is_bf16_supported,
            torch.backends.cudnn.benchmark,
        ) = orig
    assert after is False, (
        "setup_runtime left cudnn.benchmark True. It must be FORCED off, not assumed off: torch's default is False but anything upstream can flip it, and on Blackwell autotuning crashes the host."
    )


def test_the_AMP_regime_is_recorded_as_PROVENANCE_not_assumed_identical():
    src = read("src", "pipeline", "setup.py")
    tree = ast.parse(src)
    fn = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "runtime_provenance"
        ),
        None,
    )
    assert fn is not None, (
        "`runtime_provenance` is gone from src/pipeline/setup.py. It is what makes two results comparable across the two servers."
    )
    body = ast.get_source_segment(src, fn) or ""
    bad = [k for k in ("amp", "grad_scaler") if k not in body]
    assert not bad, (
        "runtime_provenance no longer records %s. The FP16 path SKIPS an overflowing step, so the same config applies a different number of optimizer steps depending on the card."
        % ", ".join(bad)
    )






def test_a_dataset_whose_GROUPS_ARE_AN_INDEX_cannot_carry_a_local_constraint():
    import pandas as pd
    from scripts.dataset_screen import _synthetic, novelty_items

    out = {}
    for kind in ("dead", "live"):
        d = _synthetic(os.path.join(tempfile.mkdtemp(), kind), kind)
        tr = pd.read_csv(os.path.join(d, "train_meta.csv"))
        te = pd.read_csv(os.path.join(d, "test_meta.csv"))
        out[kind] = novelty_items(tr, te, "location", n_null=120, seed=0)
    d = out["dead"]
    assert d["net_raw"] < d["net_null"], (
        "an INDEX grouping (`i %% n`) produced a raw deviation of %.0f items against a sampling-noise null of %.0f. It must not even reach the null: every group is an i.i.d. draw from one distribution."
        % (d["net_raw"], d["net_null"])
    )
    assert d["net_items"] < 0 and d["net_z"] < 3.0, (
        "an INDEX grouping scored %+.0f items, z=%.1f -- it must come out at or below zero once the sampling-noise null is subtracted. Either that subtraction is gone (the raw deviation IS a large positive number, which is how dermmnist was once scored at 62x the seed noise), or the synthetic 'dead' fixture stopped being i.i.d. This is the check that would have saved octmnist and tissuemnist."
        % (d["net_items"], d["net_z"])
    )
    assert out["live"]["net_z"] > 6.0, (
        "LIVENESS: a real per-group label shift with groups held out entire scored only z=%.1f, %.0f items. A screen that cannot detect the iwildcam shape would reject every candidate dataset, which is not a null -- it is a broken instrument."
        % (out["live"]["net_z"], out["live"]["net_items"])
    )


def _tiny_cache_config():
    return {
        "model_name": "MobileNetV3",
        "hyperparams": {"dropout": 0.2},
        "code_version": "abc123",
        "run_code_version": "abc123",
        "data_fingerprint": "fp-1",
        "cache_identity": {"release_id": "fresh", "data_id": "all-six-files"},
    }


@pytest.mark.parametrize(
    "cache_regime,run_regime,reused",
    [
        ("torch.bfloat16|scaler=False", "torch.bfloat16|scaler=False", True),
        ("torch.bfloat16|scaler=False", "torch.float16|scaler=True", False),
    ],
)
def test_a_cached_warm_up_never_crosses_the_AMP_regime(
    tmp_path, monkeypatch, cache_regime, run_regime, reused
):
    import torch
    from src.models import get_model
    import src.training.model_cache as mc

    monkeypatch.setenv("OPTLOSS_MODEL_CACHE", str(tmp_path))
    cfg = _tiny_cache_config()
    monkeypatch.setattr(mc, "_amp_regime", lambda: cache_regime)
    model = get_model(
        cfg["model_name"],
        n_classes=8,
        dropout=cfg["hyperparams"]["dropout"],
        pretrained=False,
    )
    mc.save_to_cache(model, "id-amp", cfg)
    monkeypatch.setattr(mc, "_amp_regime", lambda: run_regime)
    got = mc.load_from_cache("id-amp", cfg, 8, torch.device("cpu"))
    if reused:
        assert got is not None, (
            "LIVENESS: the cache refused a warm-up from its OWN regime (%s). That retrains every model on disk and the check is worthless."
            % run_regime
        )
    else:
        assert got is None, (
            "a warm-up trained under %s was handed to a run under %s. One host skips overflowing optimizer steps and the other does not, so these are two different models sharing one cache key."
            % (cache_regime, run_regime)
        )




def test_deterministic_algorithms_is_STRICT_because_warn_only_takes_the_other_branch():
    import torch
    from src.pipeline.setup import seed_all, runtime_provenance

    src = read("src", "pipeline", "setup.py")
    assert "warn_only=False" in src, (
        "`use_deterministic_algorithms` is no longer called with warn_only=False. warn_only=True is not a milder setting -- PyTorch takes the NONDETERMINISTIC branch in the attention backward when it is true, which is the 0.0358 macro-F1 floor against a 0.0017 effect."
    )
    was_det = torch.are_deterministic_algorithms_enabled()
    was_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    was_cudnn = torch.backends.cudnn.deterministic
    try:
        seed_all(1)
        assert torch.are_deterministic_algorithms_enabled(), (
            "seed_all left deterministic algorithms OFF"
        )
        assert not torch.is_deterministic_algorithms_warn_only_enabled(), (
            "deterministic algorithms are in WARN-ONLY mode, which is the nondeterministic branch, not a strict one"
        )
        assert torch.backends.cudnn.deterministic is True, (
            "cudnn.deterministic is off; it is not sufficient on its own but it is still part of the regime that was measured"
        )
    finally:
        torch.use_deterministic_algorithms(was_det, warn_only=was_warn)
        torch.backends.cudnn.deterministic = was_cudnn
    with pytest.raises(ValueError):
        seed_all(None)
    prov = runtime_provenance(torch.device("cpu"))
    for key in (
        "deterministic",
        "deterministic_warn_only",
        "cudnn_deterministic",
        "cublas_workspace_config",
    ):
        assert key in prov, (
            "runtime_provenance no longer records `%s`. The runs that first showed the 21x floor could not say which determinism regime produced them, which is why this is recorded per run."
            % key
        )


def test_CUBLAS_WORKSPACE_CONFIG_is_set_BEFORE_torch_is_imported():
    bad = []
    for path in ("main.py", os.path.join("src", "experiments", "runner.py")):
        tree = ast.parse(read(path))
        env_line = torch_line = None
        for node in ast.walk(tree):
            if (
                env_line is None
                and isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and (node.func.attr == "setdefault")
                and any(
                    (
                        isinstance(a, ast.Constant)
                        and a.value == "CUBLAS_WORKSPACE_CONFIG"
                        for a in node.args
                    )
                )
            ):
                env_line = node.lineno
            if torch_line is None and isinstance(node, ast.Import):
                if any(
                    (
                        a.name == "torch" or a.name.startswith("torch.")
                        for a in node.names
                    )
                ):
                    torch_line = node.lineno
        if env_line is None:
            bad.append(
                "%s never sets CUBLAS_WORKSPACE_CONFIG; use_deterministic_algorithms then raises on every cuBLAS matmul"
                % path
            )
        elif torch_line is not None and env_line > torch_line:
            bad.append(
                "%s sets CUBLAS_WORKSPACE_CONFIG at line %d, AFTER `import torch` at line %d. cuBLAS reads it when the handle is created, so this is a silent no-op"
                % (path, env_line, torch_line)
            )
    assert not bad, "\n  ".join([""] + bad)


HEAD_SHAPE = {
    "MobileNetV3": {"projection": (960, 1280), "activation": "Hardswish"},
    "MobileNetV2": {"projection": None, "activation": None},
    "RegNetY400MF": {"projection": None, "activation": None},
    "ViTB16": {"projection": None, "activation": None},
}


@pytest.mark.parametrize("backbone", sorted(HEAD_SHAPE))
def test_a_backbone_replaces_ONLY_its_final_layer_and_keeps_ONE_dropout(backbone):
    import torch.nn as nn
    from src.models import get_model

    p = 0.3
    model = get_model(backbone, n_classes=8, dropout=p, pretrained=False)
    heads = [
        m
        for (name, m) in model.named_modules()
        if name.endswith(("classifier", "heads", "fc"))
    ]
    assert heads, "%s exposes no recognisable head" % backbone
    head = heads[-1]
    layers = list(head.modules())
    drops = [m for m in layers if isinstance(m, nn.Dropout)]
    assert len(drops) == 1, (
        "%s has %d Dropout layers in its head, not 1. The double dropout is the defect the MobileNetV3 rebuild was introduced to avoid, and rebuilding was a worse cure than the disease."
        % (backbone, len(drops))
    )
    assert abs(drops[0].p - p) < 1e-09, (
        "%s ignored the configured dropout p=%.2f and kept %.2f -- the fix is to SET the existing Dropout's p, not to add another one"
        % (backbone, p, drops[0].p)
    )
    linears = [m for m in layers if isinstance(m, nn.Linear)]
    assert linears, "%s head has no Linear" % backbone
    assert linears[-1].out_features == 8, "%s final layer emits %d classes, not 8" % (
        backbone,
        linears[-1].out_features,
    )
    want = HEAD_SHAPE[backbone]
    if want["projection"]:
        (a, b) = want["projection"]
        kept = [m for m in linears if (m.in_features, m.out_features) == (a, b)]
        assert kept, (
            "%s no longer keeps its pretrained %d->%d projection. Rebuilding the whole classifier discards a layer that only warm-up trains, and trained arms get ONE warm-up epoch against the post-hoc arms' thirty -- so the loss lands entirely on the treated side of the headline comparison. Mutate the head in place: set the existing Dropout's p and replace head[-1] only."
            % (backbone, a, b)
        )
        assert any((type(m).__name__ == want["activation"] for m in layers)), (
            "%s head has no %s, so it is not torchvision's head any more -- it was rebuilt"
            % (backbone, want["activation"])
        )


def test_normalize_mode_discards_the_gradient_MAGNITUDE_in_both_directions():
    import torch
    from src.training.constraint_step import finish_constraint_step

    def deliver(mode, scale):
        m = torch.nn.Linear(4, 3, bias=False)
        opt = torch.optim.SGD(m.parameters(), lr=0.0)
        m.weight.grad = torch.full_like(m.weight, scale)
        finish_constraint_step(m, opt, None, clip=1.0, mode=mode, fp32=True)
        return float(torch.linalg.vector_norm(m.weight.grad))

    (small, big) = (0.0001, 10.0)
    assert abs(deliver("normalize", small) - 1.0) < 1e-05, (
        "mode=normalize did not scale a SMALL constraint gradient up to clip. Then the delivered dose still depends on each arm's own scale, and the 20x hounie/fioretto dose gap that normalize exists to remove is back. FRAMEWORK 2(z28)."
    )
    assert abs(deliver("normalize", big) - 1.0) < 1e-05, (
        "mode=normalize did not scale a LARGE constraint gradient down to clip"
    )
    assert deliver("clip", small) < 0.5, (
        "NEGATIVE CONTROL FAILED: mode=clip scaled a small gradient UP. clip must only shrink, or it is normalize under another name and the corpus's two modes are one mode."
    )
    assert abs(deliver("clip", big) - 1.0) < 1e-05, (
        "mode=clip did not cap a large gradient at clip"
    )


def test_a_K0_ceiling_sits_past_the_penalty_peak_and_carries_almost_no_pull():
    import torch
    from src.losses.transductive_loss import MulticlassTransductiveLoss

    def dpen(soft, K, rho):
        loss = MulticlassTransductiveLoss(
            global_constraints=[10000000000.0] * 8,
            local_constraints={},
            num_classes=8,
            initial_rho=float(rho),
        )
        t = torch.tensor(float(soft), dtype=torch.float64, requires_grad=True)
        loss._penalty(t, float(K)).backward()
        return float(t.grad)

    rho = 0.5 + 3.431 * 28
    grid = [i * 0.01 for i in range(1, 500)]
    vals = [dpen(x, 0.0, rho) for x in grid]
    peak_at = grid[max(range(len(vals)), key=lambda i: vals[i])]
    assert 0.4 < peak_at < 0.8, (
        "the K==0 penalty peak moved to a soft count of %.3f; 2(z36) is written around ~0.58 and would need redoing"
        % peak_at
    )
    peak = max(vals)
    for soft, bar in ((25.0, 0.001), (100.0, 0.0001), (400.0, 1e-05)):
        frac = dpen(soft, 0.0, rho) / peak
        assert frac < bar, (
            "a K==0 group with soft count %.0f carries %.3g of the peak pull, above the %.0e this lesson records. If the shape changed, the 'half the local constraints are inert' claim must be re-measured."
            % (soft, frac, bar)
        )
    healthy = dpen(63.0, 40.0, rho)
    starved = dpen(400.0, 0.0, rho)
    assert healthy / starved > 10000.0, (
        "a constraint at 1.57x its budget gets only %.1fx the pull of one at 400x over. The penalty is then roughly monotone in the violation and 2(z36)'s starvation argument does not apply."
        % (healthy / starved)
    )


def test_a_cross_dataset_campaign_cannot_share_a_warm_up_by_construction():
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
    assert iw.split("_")[1] != bc.split("_")[1]
    again = compute_base_model_id(P, "MobileNetV3", dict(hp), "iwildcam", dict(dc_i))
    assert again == iw, (
        "the same warm-up produced two ids (%s vs %s); arms that should share a cached model would each retrain one"
        % (iw, again)
    )
    other_slice = compute_base_model_id(
        P,
        "MobileNetV3",
        hp,
        "iwildcam",
        {"data_dir": "data/iwildcam/othersplit", "num_classes": 8},
    )
    assert other_slice != iw, "data_dir is not in the warm-up identity"


def test_read_run_counts_EVERY_class_not_only_the_capped_ones(tmp_path):
    import csv
    import json
    from scripts import deployed_h2h

    d = tmp_path / "seed_1"
    d.mkdir()
    json.dump(
        {
            "dataset_config": {"constrained_class": [2], "num_classes": 8},
            "hyperparams": {
                "constraint_epochs": 29,
                "constraint_fp32": True,
                "constraint_grad_mode": "normalize",
            },
        },
        open(d / "config.json", "w"),
    )
    rows = [(2, 2)] * 3 + [(2, 7)] * 1 + [(7, 2)] * 2 + [(5, 5)] * 2 + [(5, 7)] * 1
    with open(d / "final_predictions.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["Predicted_Label", "True_Label", "Group_ID"])
        for pred, true in rows:
            wr.writerow([pred, true, 0])
    rec = deployed_h2h.read_run(str(d))
    assert rec is not None, "the fixture must look like a finished run"
    assert rec["per"] == {2: dict(TP=3, K=4, n=5)}, rec["per"]
    assert 5 in rec["all_per"], (
        "`all_per` dropped the uncapped class, so macro-F1 sees exactly what cc-F1 sees and the one independent metric is a duplicate: %r"
        % (rec["all_per"],)
    )
    assert rec["all_per"][5] == dict(TP=2, K=3, n=2), rec["all_per"][5]
    assert rec["all_per"][2] == rec["per"][2]
    capped_only = deployed_h2h.macrof1({2: rec["all_per"][2]})
    assert abs(capped_only - 2 * 3 / 9.0) < 1e-12, capped_only
    assert abs(deployed_h2h.macrof1(rec["all_per"]) - capped_only) > 1e-06, (
        "macro-F1 over every class equals the capped-only figure, so the uncapped channel is inert"
    )
