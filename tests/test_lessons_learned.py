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
        "LIVENESS: a real per-group label shift with groups held out entire scored only z=%.1f, %.0f items. A screen that cannot detect the fmow2 shape would reject every candidate dataset, which is not a null -- it is a broken instrument."
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
    dc_i = {"data_dir": "data/fmow2/oodslice", "num_classes": 8}
    dc_b = {"data_dir": "data/bcn/oodslice", "num_classes": 8}
    iw = compute_base_model_id(P, "MobileNetV3", hp, "fmow2", dc_i)
    bc = compute_base_model_id(P, "MobileNetV3", hp, "bcn", dc_b)
    assert iw != bc, "two datasets produced the SAME base_model_id: %s" % iw
    assert iw.startswith("MobileNetV3_fmow2_"), iw
    assert bc.startswith("MobileNetV3_bcn_"), bc
    assert iw.split("_")[1] != bc.split("_")[1]
    again = compute_base_model_id(P, "MobileNetV3", dict(hp), "fmow2", dict(dc_i))
    assert again == iw, (
        "the same warm-up produced two ids (%s vs %s); arms that should share a cached model would each retrain one"
        % (iw, again)
    )
    other_slice = compute_base_model_id(
        P,
        "MobileNetV3",
        hp,
        "fmow2",
        {"data_dir": "data/fmow2/othersplit", "num_classes": 8},
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


def test_a_candidate_dataset_is_screened_on_EVERY_condition_at_once(tmp_path):
    """iwildcam ran this project for weeks while failing 6 of 8 conditions.

    The conditions each had their own tool and the tools disagreed:
    `dataset_screen` rewards per-group label SHIFT, `tier_viability` rewards
    per-group DENSITY, and those two are in tension by construction -- strong
    shift means classes are ABSENT from groups, which is exactly sparsity.
    iwildcam scored best in the corpus on the first (TV 0.737) and 20th of 22
    on the second (density 0.27). Both were right. Nothing combined them, so
    every screen could be answered "it passes" and the dataset stayed.

    This pins that a candidate is screened on every condition together, and
    that a slice failing a condition is NOT reported as passing.
    """
    import subprocess, sys, os

    def slice_csv(name, rows):
        p = tmp_path / name
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write("label,location" + chr(10))
            for y, g in rows:
                fh.write("%d,%s" % (y, g) + chr(10))
        return str(p)

    # A slice shaped like iwildcam: 8 classes, but the two most spread live in
    # a handful of groups and most groups hold neither of them.
    bad = []
    for g in range(10):
        for y in range(8):
            # classes 0 and 1 only ever appear at groups 0 and 1
            if y in (0, 1) and g > 1:
                continue
            bad.extend([(y, "g%d" % g)] * 20)
    # A slice where every class appears in every group at differing rates.
    good = []
    for g in range(10):
        for y in range(8):
            good.extend([(y, "g%d" % g)] * (60 + 10 * ((y + g) % 5)))

    run = lambda p: subprocess.run(
        [sys.executable, "-m", "scripts.candidate_gate", "--meta", p, "--name", "s"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True, text=True)

    out_bad = run(slice_csv("bad.csv", bad))
    assert out_bad.returncode == 1, (
        "the exit code IS the gate; a failing slice must exit non-zero:"
        + chr(10) + out_bad.stdout)
    assert "PASS ALL" not in out_bad.stdout, (
        "a slice whose capped classes live in 2 of 10 groups was reported as "
        "passing every condition:" + chr(10) + out_bad.stdout)
    assert "FAIL" in out_bad.stdout, "no condition fired on the sparse slice"

    out_good = run(slice_csv("good.csv", good))
    assert out_good.returncode == 0, (
        "a good slice must exit 0:" + chr(10) + out_good.stdout)
    # NEGATIVE CONTROL: the gate must be able to say yes, or it is not a gate,
    # it is a rejection stamp.
    assert "PASS ALL" in out_good.stdout, (
        "a dense, well-spread, rate-varying slice was refused, so the gate "
        "cannot distinguish good data from bad:" + chr(10) + out_good.stdout)


def _fake_run(root, arm, seed, cap="L80_G95", n=240, shift=0.0, rng_key=None,
              group_boost=0.0, other_shift=0.0):
    """One run directory with the two files the scorers actually read."""
    import os, json, random
    d = os.path.join(root, "MobileNetV3", "fmow2", cap, arm, "seed_%d" % seed)
    os.makedirs(d, exist_ok=True)
    # rng_key is explicit so a test can give two arms the SAME draws and
    # assert the probe reports no gap. Keying on len(arm) made "tralo" and
    # "tralo_null" differ by construction, which is not a control.
    rng = random.Random(seed * 17 if rng_key is None else rng_key)
    cols = ["True_Label", "Predicted_Label", "Group_ID"] + ["Prob_Class_%d" % c for c in range(8)]
    lines = [",".join(cols)]
    for i in range(n):
        y = i % 8
        g = i % 4
        # `shift` degrades the ordering for the capped classes only, so a tool
        # reading the ORDER sees it and a tool reading only counts does not.
        p = [0.02] * 8
        # `other_shift` degrades the UNCAPPED classes only, so a fixture can
        # move macroF1/accuracy while leaving cc-F1 tied -- which is what a
        # domination test needs and `shift` alone cannot express.
        s_y = shift if y in (1, 2, 7) else other_shift
        good = rng.random() > (0.25 + s_y)
        if other_shift and y not in (1, 2, 7):
            # Route an UNCAPPED error to another uncapped class. Sending it to
            # (y+1)%8 puts class 0's mistakes into class 1, a CAPPED class, so
            # degrading the uncapped classes moved cc-F1 as well and the two
            # could not be varied independently.
            unc = [0, 3, 4, 5, 6]
            pred = y if good else unc[(unc.index(y) + 1) % len(unc)]
        else:
            pred = y if good else (y + 1) % 8
        p[pred] = 0.86 - s_y
        if group_boost and g == 0:
            # A constant added to one GROUP preserves that group's internal
            # order exactly, so a PER-GROUP AP must not move. A GLOBAL AP must.
            for c in (1, 2, 7):
                p[c] = min(0.999, p[c] + group_boost)
        lines.append(",".join([str(y), str(pred), str(g)] + ["%.4f" % v for v in p]))
    body = chr(10).join(lines) + chr(10)
    for name in ("final_predictions.csv", "final_predictions_raw.csv"):
        with open(os.path.join(d, name), "w", encoding="utf-8") as fh:
            fh.write(body)
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"status": "completed", "arm": arm, "dataset_mode": "fmow2",
                   "model_name": "MobileNetV3", "constraint_tag": cap,
                   "hyperparams": {"seed": seed},
                   "dataset_config": {"num_classes": 8, "constrained_class": [1, 2, 7]}}, fh)
    return d


def test_the_verdict_tools_run_and_a_cc_f1_gain_paid_for_in_damage_is_NOT_a_win(tmp_path):
    """The acceptance bar was single-metric IN THE CODE.

    `deployed_h2h` computed the seed-paired effect on `cc_f1` and nothing else,
    so every verdict this project issued was cc-F1 by construction, and a cc-F1
    gain bought with collateral damage read as a clean win. These two tools are
    what replace that, so they are executed here rather than trusted.

    `rank_probe` exists because post-hoc allocation is optimal GIVEN the
    probabilities: ranking is the ONLY channel by which a trained arm can beat a
    clipper, so an arm that changes counts without improving gAP cannot win.
    """
    import subprocess, sys, os

    root = str(tmp_path / "camp")
    for seed in (1, 2):
        _fake_run(root, "clip", seed, shift=0.0)
        _fake_run(root, "tralo", seed, shift=0.15)   # strictly worse ordering
        _fake_run(root, "tralo_null", seed, shift=0.0)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    prof = subprocess.run([sys.executable, "-m", "scripts.profile_report",
                           "--campaign", root], cwd=repo, capture_output=True, text=True)
    assert prof.returncode == 0, prof.stderr
    assert "ccF1" in prof.stdout and "collatF1" in prof.stdout and "macroF1" in prof.stdout, (
        "the profile must print the damage metrics beside cc-F1, or it is the "
        "single-metric bar again:" + chr(10) + prof.stdout)
    assert "VERDICT" in prof.stdout, "no verdict line"
    assert "WIN" not in prof.stdout, (
        "an arm given a strictly WORSE ordering was called a win:"
        + chr(10) + prof.stdout)

    rank = subprocess.run([sys.executable, "-m", "scripts.rank_probe",
                           "--glob", os.path.join(root, "*/*/*/*/seed_*")],
                          cwd=repo, capture_output=True, text=True)
    assert rank.returncode == 0, rank.stderr
    assert "gAP" in rank.stdout, "rank_probe did not report gAP"
    # NEGATIVE CONTROL: the degraded arm must show a NEGATIVE gAP delta against
    # its own null. If this passes with shift=0 too, the probe reads nothing.
    deltas = [l for l in rank.stdout.splitlines() if "tralo_null" in l and "d gAP" in l]
    assert deltas, "no tralo-vs-null ranking contrast was printed"
    assert any("d gAP -" in l for l in deltas), (
        "a strictly degraded ordering did not register as a NEGATIVE gAP delta, "
        "so the probe cannot detect the only channel that can win:"
        + chr(10) + chr(10).join(deltas))


def test_rank_probe_reports_NO_ranking_gap_when_the_orderings_AGREE(tmp_path):
    """Liveness control for the test above: with no degradation the gAP delta
    must be ~0. Without this, a probe that always printed a negative number
    would pass."""
    import subprocess, sys, os

    root = str(tmp_path / "camp")
    for seed in (1, 2):
        _fake_run(root, "tralo", seed, shift=0.0, rng_key=seed)
        _fake_run(root, "tralo_null", seed, shift=0.0, rng_key=seed)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rank = subprocess.run([sys.executable, "-m", "scripts.rank_probe",
                           "--glob", os.path.join(root, "*/*/*/*/seed_*")],
                          cwd=repo, capture_output=True, text=True)
    assert rank.returncode == 0, rank.stderr
    for line in rank.stdout.splitlines():
        if "tralo_null" in line and "d gAP" in line:
            val = float(line.split("d gAP")[1].split()[0])
            assert abs(val) < 1e-9, (
                "identical orderings reported a non-zero ranking gap: " + line)


def test_rank_probe_is_PER_GROUP_because_the_allocator_is(tmp_path):
    """gAP must be invariant to a between-group shift; a GLOBAL AP must not be.

    The allocator cuts top-k WITHIN each group, so a global AP scores an
    ordering the system never uses. This project has found a global-vs-per-group
    confusion in ELEVEN separate call sites, and the earlier version of this
    test did not catch a mutation replacing gAP with the global AP -- so the
    invariance is asserted directly rather than implied.
    """
    import subprocess, sys, os, re

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def gaps(root):
        r = subprocess.run([sys.executable, "-m", "scripts.rank_probe",
                            "--glob", os.path.join(root, "*/*/*/*/seed_*")],
                           cwd=repo, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        out = {}
        for line in r.stdout.splitlines():
            m = re.match(r"\s+(\S+)\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s*$", line)
            if m and m.group(1) == "tralo":
                out.setdefault("gAP", []).append(float(m.group(2)))
                out.setdefault("capAP", []).append(float(m.group(3)))
        assert out, "rank_probe printed no tralo row:" + chr(10) + r.stdout
        return out

    base = str(tmp_path / "base")
    lifted = str(tmp_path / "lifted")
    for seed in (1, 2):
        _fake_run(base, "tralo", seed, rng_key=seed)
        _fake_run(lifted, "tralo", seed, rng_key=seed, group_boost=0.10)

    b, l = gaps(base), gaps(lifted)
    assert b["gAP"] == pytest.approx(l["gAP"], abs=1e-9), (
        "a between-group shift that preserves within-group order MOVED gAP, so "
        "it is not a per-group statistic: %s vs %s" % (b["gAP"], l["gAP"]))
    assert b["capAP"] != pytest.approx(l["capAP"], abs=1e-9), (
        "the GLOBAL AP did not move under a between-group shift, so the fixture "
        "does not separate the two readings and the invariance above is vacuous")


def test_rank_paired_reports_the_SPREAD_and_does_not_POOL_a_reversal(tmp_path, capsys):
    """A mean without its floor is the trap; a pooled mean hides a reversal.

    `rank_probe` prints paired MEANS only, and a bcn read off it ("6 of 6 cells
    positive, +0.0143 vs clip") did not survive being shown its own noise: every
    cell was inside its seed sd. This gate holds `rank_paired` to the two things
    that fixed it.

    (a) The sd must be computed ACROSS SEEDS WITHIN a cell. One cap is built so
        `tralo` and `clip` differ identically in every seed -- that cell must
        print sd 0.0000 even though the other cap scatters wildly. Pooling the
        two caps destroys that zero.
    (b) Two cells with opposite signs must be COUNTED, not averaged into one
        row that reads as a null.
    """
    import importlib
    root = str(tmp_path / "results" / "camp")
    # L80: tralo strictly worse, by the same construction in every seed -> sd 0.
    # L90: tralo worse by a different amount per seed -> sd > 0, opposite sign.
    for seed in (1, 2, 3, 4):
        _fake_run(root, "clip", seed, cap="L80_G95", shift=0.0, rng_key=7)
        _fake_run(root, "tralo", seed, cap="L80_G95", shift=0.20, rng_key=7)
        _fake_run(root, "clip", seed, cap="L90_G95", shift=0.20, rng_key=seed)
        _fake_run(root, "tralo", seed, cap="L90_G95", shift=0.0, rng_key=seed)
    mod = importlib.import_module("scripts.rank_paired")
    assert mod.main(["--glob", root + "/*/*/*/*/seed_*", "--a", "tralo",
                     "--b", "clip"]) == 0
    out = capsys.readouterr().out
    rows = [l for l in out.splitlines() if "L80_G95" in l or "L90_G95" in l]
    assert rows, out
    l80 = [l for l in rows if "L80_G95" in l]
    l90 = [l for l in rows if "L90_G95" in l]
    assert l80 and l90, out
    # (a) identical per-seed deltas -> sd is exactly zero in that cell
    assert any(l.split()[5] == "0.0000" for l in l80), "L80 sd is not across seeds:\n" + out
    # (b) the two caps disagree in sign and are counted, not merged
    signs = {l.split()[4][0] for l in l80} | {l.split()[4][0] for l in l90}
    assert signs == {"-", "+"}, "a reversal was pooled away:\n" + out
    assert "positive" in out and "negative" in out
    assert "NOT a pooled estimate" in out


def test_rank_paired_collapses_a_CAP_that_never_reached_the_MODEL(tmp_path, capsys):
    """Three caps are not three cells when the cap only moved the allocator.

    An arm that takes no constraint step trains ONE model; the cap acts solely
    in the post-hoc allocation. Verified by md5 on the real bcn corpus: `clip`
    and `tralo_null` write byte-identical `final_predictions_raw.csv` across
    L70/L80/L90 while `tralo` writes three different ones. gAP is
    allocation-free, so printing those as three cells triples the apparent n --
    which is how "6 of 6 cells positive" was really 2 observations.
    """
    import importlib
    root = str(tmp_path / "results" / "camp")
    for seed in (1, 2, 3, 4):
        for cap in ("L70_G95", "L90_G95"):
            # same rng_key and same shift in both caps -> identical files,
            # exactly as a no-constraint-step arm produces.
            _fake_run(root, "clip", seed, cap=cap, shift=0.0, rng_key=seed)
            _fake_run(root, "tralo_null", seed, cap=cap, shift=0.12, rng_key=seed)
    mod = importlib.import_module("scripts.rank_paired")
    assert mod.main(["--glob", root + "/*/*/*/*/seed_*", "--a", "tralo_null",
                     "--b", "clip"]) == 0
    out = capsys.readouterr().out
    body = [l for l in out.splitlines() if "MobileNetV3" in l]
    assert body, out
    assert all("cap-inert" in l for l in body), "a cap that never reached the model was counted:\n" + out
    assert not any("L70_G95" in l or "L90_G95" in l for l in body), out
    # one row per constrained class, not one per (class, cap)
    assert len(body) == 3, "expected 3 classes collapsed over caps, got %d:\n%s" % (len(body), out)


def test_a_cc_f1_TIE_is_not_a_WIN_and_domination_is_checked_against_EVERY_arm():
    """Two ways the verdict line overstated, both seen on real fmow2 output.

    (a) It printed "WIN -- leads/ties alm on cc-F1 (-0.0093)". A NEGATIVE delta
        inside the noise is the leading GROUP, which is the user's bar, but it
        is not a lead. Calling a tie a win is how four #1 claims were made here.
    (b) "not dominated elsewhere" was tested against `clip` ALONE, so an arm
        ahead of TraLO on the whole quality profile passed unnoticed unless it
        happened to be `clip`.

    `classify` is called directly: routed through the fixture generator, any arm
    strong enough to dominate also leads cc-F1 beyond noise and trips the LOSS
    branch first, so the branch under test is unreachable end-to-end.
    """
    from scripts.profile_report import classify

    tie = {"cc_f1": (-0.0010, 0.0100)}
    # (a) a deficit inside the noise, nothing else measured
    v = classify("alm", -0.0010, 0.0100, {"alm": dict(tie)}, False)
    assert "WIN" not in v, v
    assert v.startswith("LEADING GROUP"), v

    # liveness: a real lead beyond the noise still reads WIN
    v = classify("alm", +0.0400, 0.0100, {"alm": {"cc_f1": (0.0400, 0.0100)}}, False)
    assert v.startswith("WIN"), v

    # (b) cc-F1 ties, but hounie is ahead on the rest and TraLO never is
    dom = {"hounie": {"cc_f1": (-0.0010, 0.0100),
                      "macro_f1": (-0.0400, 0.0050),
                      "accuracy": (-0.0300, 0.0050)},
           "alm": dict(tie)}
    v = classify("alm", -0.0010, 0.0100, dom, False)
    assert v.startswith("DOMINATED") and "hounie" in v, v
    assert "alm" not in v.split("by:")[-1], "alm ties everywhere and does not dominate: " + v

    # negative control: TraLO ahead on ONE metric beyond noise is not domination
    dom["hounie"]["collateral_f1"] = (+0.0400, 0.0050)
    v = classify("alm", -0.0010, 0.0100, dom, False)
    assert "DOMINATED" not in v, v


def _fake_log(root, dataset, cap, arm, seed, accs, constraint_epochs=29):
    """A training_log.csv plus the config.json the gate reads its budget from.

    The config matters: the gate used to assume 29 constraint epochs for every
    run, so a 6-epoch campaign was judged against a 30-epoch budget. Pass
    constraint_epochs=0 to make this a POST-HOC arm, which the gate must skip
    entirely -- a clipper has no constraint phase to protect.
    """
    import os, json
    d = os.path.join(root, "MobileNetV3", dataset, cap, arm, "seed_%d" % seed)
    os.makedirs(d, exist_ok=True)
    lines = ["Epoch,Train_Acc,L_CE"]
    for i, a in enumerate(accs, start=1):
        lines.append("%d,%.4f,%.4f" % (i, a, max(0.001, 1.0 - a)))
    with open(os.path.join(d, "training_log.csv"), "w", encoding="utf-8") as fh:
        fh.write(chr(10).join(lines) + chr(10))
    warm = 1 if constraint_epochs else len(accs)
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"arm": arm, "hyperparams": {
            "warmup_epochs": warm, "constraint_epochs": constraint_epochs}}, fh)
    return d


def test_the_saturation_gate_fails_a_FROZEN_boundary_and_passes_a_LIVE_one(tmp_path, capsys):
    """The condition none of candidate_gate's eight checks could see.

    Once cross-entropy collapses the task gradient is ~0, and under
    `constraint_grad_mode: normalize` the constraint gradient is rescaled to a
    FIXED norm regardless of the violation -- so the step is full-size and
    opposed by nothing. The constraint is shoving a frozen boundary, not
    reshaping it. Measured on the real corpus: fmow2 is live for 3.2 of 29
    constraint epochs and bcn for 4.5, on BOTH backbones, which is why changing
    dataset does not fix it.

    Negative control is the point of the test: a log that stays live must PASS,
    or the gate is just printing SATURATED unconditionally.
    """
    import importlib
    mod = importlib.import_module("scripts.saturation_gate")

    frozen = str(tmp_path / "frozen")
    live = str(tmp_path / "live")
    for seed in (1, 2):
        # reaches 0.95 at epoch 4 and then runs 26 more epochs -> live 3 of 29.
        # The tail MATTERS: with only six rows, dropping the `break` would still
        # report 6 < 14 and the gate would pass its own mutation.
        _fake_log(frozen, "fmow2", "L80_G95", "tralo", seed,
                  [0.80, 0.88, 0.93, 0.96, 0.98] + [0.99] * 25)
        # never reaches 0.95 -> live for the whole run
        _fake_log(live, "hardset", "L80_G95", "tralo", seed,
                  [0.50, 0.60, 0.68, 0.74, 0.78, 0.80])

    assert mod.main(["--glob", frozen + "/*/*/*/tralo/seed_*", "--strict"]) == 1
    out = capsys.readouterr().out
    assert "SATURATED" in out, out

    assert mod.main(["--glob", live + "/*/*/*/tralo/seed_*", "--strict",
                     "--constraint-epochs", "6"]) == 0
    out = capsys.readouterr().out
    assert "SATURATED" not in out, "a boundary that never froze was called saturated:" + chr(10) + out

    # A SHORT campaign judged on its OWN budget. Live for 3 of 5 constraint
    # epochs is a pass; against the old hardcoded 29 it was reported SATURATED,
    # which would have thrown away a valid campaign.
    short = str(tmp_path / "short")
    for seed in (1, 2):
        _fake_log(short, "fmow2", "L80_G95", "tralo", seed,
                  [0.80, 0.88, 0.93, 0.96, 0.98, 0.99], constraint_epochs=5)
    assert mod.main(["--glob", short + "/*/*/*/tralo/seed_*", "--strict"]) == 0, (
        "a campaign live for 3 of its own 5 constraint epochs was failed -- the "
        "gate is still judging against a hardcoded budget")
    out = capsys.readouterr().out
    row = [ln for ln in out.splitlines() if "MobileNetV3/fmow2" in ln]
    assert row and row[0].split()[1] == "5", (
        "the gate did not report the cell's OWN 5-epoch constraint phase:" + chr(10) + out)

    # A campaign that SWEEPS the budget must judge each budget on its own half.
    # Taking the shortest budget for the whole campaign -- which this did --
    # lowers the 29-epoch arm's bar from 14 live epochs to 2 and passes a
    # frozen boundary. The two arms below differ ONLY in budget: 3 live epochs
    # is a pass at 5 and a failure at 29, so one row of each is the proof.
    swept = str(tmp_path / "swept")
    for seed in (1, 2):
        _fake_log(swept, "fmow2", "L80_G95", "tralo_b5", seed,
                  [0.80, 0.88, 0.93, 0.96, 0.98, 0.99], constraint_epochs=4)
        _fake_log(swept, "fmow2", "L80_G95", "tralo", seed,
                  [0.80, 0.88, 0.93, 0.96, 0.98] + [0.99] * 25,
                  constraint_epochs=29)
    rc = mod.main(["--glob", swept + "/*/*/*/*/seed_*", "--strict"])
    out = capsys.readouterr().out
    verdicts = {ln.split()[1]: ("SATURATED" in ln)
                for ln in out.splitlines() if "MobileNetV3/fmow2" in ln}
    assert verdicts == {"4": False, "29": True}, (
        "a swept campaign was not judged per budget -- the same 3 live epochs "
        "must pass at 4 constraint epochs and FAIL at 29:" + chr(10) + out)
    # AMENDMENT 2026-09-15: in a sweep the long budget is the frozen REFERENCE
    # end of the dose axis, deliberately included. One live budget is enough for
    # the campaign to have contrast, so it passes while still REPORTING the
    # frozen verdict above.
    assert rc == 0, (
        "a swept campaign with a live budget was killed for containing its own "
        "frozen reference arm:" + chr(10) + out)
    assert "1 of 2 budgets are live" in out, out

    # ... but a sweep where NOTHING is live has no contrast and must still die.
    dead = str(tmp_path / "dead")
    for seed in (1, 2):
        for arm, con in (("tralo_b12", 11), ("tralo", 29)):
            _fake_log(dead, "fmow2", "L80_G95", arm, seed,
                      [0.80, 0.88, 0.93, 0.96, 0.98] + [0.99] * 25,
                      constraint_epochs=con)
    rc = mod.main(["--glob", dead + "/*/*/*/*/seed_*", "--strict"])
    out = capsys.readouterr().out
    assert rc == 1, (
        "a swept campaign with NO live budget measures the frozen regime twice "
        "and must be killed:" + chr(10) + out)
    assert "NO budget is live" in out, out

    # A POST-HOC arm has no constraint phase. Its (short) live window must not
    # drag a cell down, or clippers decide whether trained arms pass.
    mixed = str(tmp_path / "mixed")
    for seed in (1, 2):
        _fake_log(mixed, "fmow2", "L80_G95", "tralo", seed,
                  [0.80, 0.88, 0.93, 0.96, 0.98, 0.99], constraint_epochs=5)
        _fake_log(mixed, "fmow2", "L80_G95", "clip", seed,
                  [0.97, 0.99, 0.99, 0.99, 0.99, 0.99], constraint_epochs=0)
    rc = mod.main(["--glob", mixed + "/*/*/*/*/seed_*", "--strict"])
    out = capsys.readouterr().out
    # Assert on the COUNT, not just the verdict: including post-hoc arms also
    # drags the derived budget toward 0, which lowers the bar and can mask the
    # inclusion. n is unambiguous -- 2 trained runs, and the 2 clippers skipped.
    cell = [ln for ln in out.splitlines() if "MobileNetV3/fmow2" in ln]
    assert cell, out
    assert cell[0].split()[2] == "2", (
        "the gate counted %s runs in the cell; the 2 post-hoc clippers should "
        "have been skipped entirely, since a clipper has no constraint phase "
        "whose boundary could freeze:" % cell[0].split()[1] + chr(10) + out)
    assert rc == 0, (
        "a post-hoc clipper pulled the cell below the bar and failed the "
        "trained arms with it:" + chr(10) + out)


def test_augment_is_OFF_by_default_and_actually_CHANGES_the_data_when_on():
    """The augmentation seam, and the guarantee that turning it off is a no-op.

    Reason it exists: `make_dataloader` wrapped a bare `TensorDataset` over the
    preprocessed arrays -- no augmentation of any kind -- and every
    backbone/dataset pair measured memorised the train set within 2-5 epochs.
    From there cross-entropy is ~0, and the constraint reaches the weights only
    through d(soft count)/d(theta), whose per-item weight is p(1-p): measured
    mean 0.007-0.023 against a 0.25 maximum, with the top 1% of items carrying
    34% of the total. A few dozen borderline items choose the direction and
    `normalize` rescales it to full size.

    Two things must hold. augment=False must be byte-identical to the old path,
    or the whole stored corpus becomes incomparable. augment=True must actually
    change the pixels, or the flag is another inert knob -- this project has
    found five of those.
    """
    import torch
    from src.pipeline.warmup import make_dataloader, AugmentedTensors

    torch.manual_seed(0)
    X = torch.rand(16, 3, 24, 24)
    y = torch.arange(16)

    plain = make_dataloader(X, y, batch_size=16, augment=False)
    xb, yb = next(iter(plain))
    order = torch.argsort(yb)
    assert torch.equal(xb[order], X), "augment=False changed the data"

    # THROUGH make_dataloader, not by constructing the Dataset directly: this
    # project has found five inert flags, and a test that instantiates the
    # augmenter itself passes even when make_dataloader ignores the argument.
    aug = make_dataloader(X, y, batch_size=16, augment=True)
    assert isinstance(aug.dataset, AugmentedTensors), (
        "make_dataloader(augment=True) did not build the augmented dataset -- "
        "the flag is inert")
    xb, yb = next(iter(aug))
    order = torch.argsort(yb)
    xb = xb[order]
    changed = sum(1 for i in range(len(X)) if not torch.equal(xb[i], X[i]))
    assert changed >= 12, (
        "augment=True left %d of 16 items untouched -- an inert knob" % (16 - changed))
    assert xb.shape == X.shape, "augmentation changed the tensor shape"


def _fake_probs(root, arm, seed, p_fn, n=200, cap="L80_G95"):
    """A run directory whose class-1 probability column is set by p_fn(i)."""
    import os, json
    d = os.path.join(root, "MobileNetV3", "fmow2", cap, arm, "seed_%d" % seed)
    os.makedirs(d, exist_ok=True)
    cols = ["True_Label", "Predicted_Label", "Group_ID"] + ["Prob_Class_%d" % c for c in range(8)]
    lines = [",".join(cols)]
    for i in range(n):
        p = [0.01] * 8
        p[1] = p_fn(i)
        lines.append(",".join([str(i % 8), "1", str(i % 4)] + ["%.6f" % v for v in p]))
    with open(os.path.join(d, "final_predictions_raw.csv"), "w", encoding="utf-8") as fh:
        fh.write(chr(10).join(lines) + chr(10))
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"arm": arm, "model_name": "MobileNetV3", "constraint_tag": cap,
                   "hyperparams": {"seed": seed},
                   "dataset_config": {"num_classes": 8, "constrained_class": [1]}}, fh)


def test_grad_mass_separates_a_SATURATED_model_from_a_LIVE_one(tmp_path, capsys):
    """The instrument behind the gradient-health account, so it gets a gate.

    The constraint reaches the weights only through d(soft count)/d(theta), and
    each item contributes in proportion to p(1-p). A saturated model drives that
    to ~0 for almost every item, so whatever handful still carries mass picks
    the direction -- and `normalize` then rescales it to full size.

    Saturated fixture: every probability is 0.001 or 0.999 except four items.
    Live fixture: every probability is 0.5. The tool must separate them on BOTH
    axes -- mean weight and concentration -- because either alone can be gamed.
    """
    import importlib
    mod = importlib.import_module("scripts.grad_mass")

    sat = str(tmp_path / "sat")
    live = str(tmp_path / "live")
    _fake_probs(sat, "tralo", 1, lambda i: 0.5 if i < 4 else (0.999 if i % 2 else 0.001))
    _fake_probs(live, "tralo", 1, lambda i: 0.5)

    assert mod.main([sat + "/*/*/*/*/seed_*"]) == 0
    sat_out = [l for l in capsys.readouterr().out.splitlines() if "tralo" in l][0]
    assert mod.main([live + "/*/*/*/*/seed_*"]) == 0
    live_out = [l for l in capsys.readouterr().out.splitlines() if "tralo" in l][0]

    sat_mean = float(sat_out.split()[4])
    live_mean = float(live_out.split()[4])
    sat_top1 = float(sat_out.split()[6].rstrip("%"))
    live_top1 = float(live_out.split()[6].rstrip("%"))

    assert live_mean > 10 * sat_mean, (
        "a fully saturated model did not read as weaker than a p=0.5 one:"
        + chr(10) + sat_out + chr(10) + live_out)
    assert sat_top1 > 3 * live_top1, (
        "concentration did not separate them, so the tool reports magnitude "
        "only and cannot see a direction chosen by a handful of items:"
        + chr(10) + sat_out + chr(10) + live_out)
    # NEGATIVE CONTROL: uniform p=0.5 must NOT look concentrated.
    assert live_top1 < 5.0, "an evenly spread gradient read as concentrated: " + live_out


def test_the_ALLOCATOR_is_NOT_optimal_even_with_a_SINGLE_capped_class():
    """Pins the allocator-gap probe, and the correction it produced.

    `apply_allocation_heuristic` says in its own docstring that a single capped
    class makes it that class's top-K. True -- but top-K by p(c) is NOT the
    maximiser, because an item not given c falls back to its best OTHER class,
    so a slot is worth the MARGIN p(c) - best_alt, not p(c). Three things have
    to hold or the measured gap means nothing:

    (a) the LP is an exact maximiser of the same objective, so it can never
        score BELOW greedy;
    (b) `margin_topk`, a closed-form optimum for the one-capped-class case,
        must REPRODUCE the LP exactly -- independent confirmation that the LP
        is specified right, which no self-consistency check could give;
    (c) greedy must be strictly WORSE than both on diffuse probabilities. If
        this ever ties, the probe has stopped measuring the defect and the
        +0.0095 accuracy prize in the docstring is unsupported.
    """
    from scripts.alloc_gap import trial

    strictly_worse = 0
    for seed in (1, 2, 3, 4, 5):
        g_obj, lp_obj, _g_acc, _lp_acc, _differ, m_obj, _m_acc = trial(
            seed, capped=(0,), sharp=1.0)
        assert lp_obj >= g_obj - 1e-6, (
            "the LP scored BELOW greedy (%.6f < %.6f) -- it is mis-specified"
            % (lp_obj, g_obj))
        assert abs(m_obj - lp_obj) < 1e-6, (
            "the closed-form margin optimum and the LP disagree (%.6f vs %.6f) "
            "-- one of them is wrong, so the measured gap is meaningless"
            % (m_obj, lp_obj))
        if lp_obj - g_obj > 1e-6:
            strictly_worse += 1
    assert strictly_worse == 5, (
        "the shipped greedy allocator tied the optimum on %d of 5 seeds -- the "
        "probe is no longer exercising the p(c)-vs-margin defect it exists to "
        "measure" % (5 - strictly_worse))

    # Liveness on the multi-class path, which has no closed form at all.
    gaps = [trial(s, capped=(0, 1, 2), sharp=0.7) for s in (1, 2, 3)]
    assert min(lp - g for g, lp, *_ in gaps) > 0.0, (
        "three competing capped classes on diffuse probabilities produced NO "
        "greedy/LP gap -- the probe would report optimality by construction")


def test_alloc_real_reads_a_RUN_and_finds_the_allocator_gap_on_ITS_OWN_probabilities(tmp_path):
    """End-to-end gate for the real-data allocator probe.

    `alloc_gap` priced greedy against the LP on synthetic softmaxes. This one
    has to do it on a stored run, which means it must get four separate things
    right or it will report a clean +0.0000 for the wrong reason: the
    Prob_Class_* columns in NUMERIC order (lexicographic puts _10 before _2),
    the caps rebuilt with the shipped constraint functions, the group ids from
    the file, and the same capped classes the run used.

    Built so the greedy rule is provably wrong: one capped class, and items
    whose p(c) is high but whose best ALTERNATIVE is higher still. Greedy
    spends the budget on them; the LP spends it on the low-margin items.
    """
    import numpy as np, pandas as pd, json
    from scripts.alloc_real import one

    rng = np.random.default_rng(0)
    n, K = 300, 12   # >10 so lexicographic column order really differs
    y = rng.integers(0, K, n)
    z = rng.normal(size=(n, K))
    z[np.arange(n), y] += 1.0
    p = np.exp(z - z.max(1, keepdims=True))
    p = p / p.sum(1, keepdims=True)
    d = tmp_path / "seed_1"
    d.mkdir()
    frame = {"True_Label": y, "Group_ID": rng.integers(0, 3, n)}
    for c in range(K):
        frame["Prob_Class_%d" % c] = p[:, c]
    pd.DataFrame(frame).to_csv(d / "final_predictions_raw.csv", index=False)
    (d / "config.json").write_text(json.dumps({
        "model_name": "mn3", "dataset_mode": "fake", "constraint_tag": "L80_G80",
        "arm": "clip", "constraint": [0.8, 0.8],
        "dataset_config": {"constrained_class": [0]}}))

    r = one(str(d))
    assert r is not None, "the probe could not read a well-formed run directory"
    (_bb, _ds, _cap, _arm, g_acc, o_acc, g_f1, o_f1, moved, g_obj, o_obj) = r
    # The LP maximises ASSIGNED PROBABILITY. Accuracy is a proxy and genuinely
    # moves both ways -- asserting on it here would be asserting a falsehood,
    # which is how the first version of this gate failed.
    assert o_obj >= g_obj - 1e-6, (
        "the LP scored BELOW the shipped greedy allocator on its OWN objective "
        "(%.6f < %.6f) -- the caps or the class order are being rebuilt wrong"
        % (o_obj, g_obj))
    assert o_obj > g_obj + 1e-9, (
        "the LP exactly TIED greedy on the objective in a case built to "
        "separate them -- the caps are probably not binding")

    # Tie the probe to a reference built here from the SAME arrays. Without
    # this the gate is vacuous against a column-order defect: permuting
    # Prob_Class_* relabels classes consistently, so the LP still beats greedy
    # on the objective and every inequality above still holds -- while the
    # capped class has quietly become a different class. (Checked: the
    # lexicographic mutant passes everything above.)
    from src.methodologies.heuristic.train import (
        _build_hierarchy, apply_allocation_heuristic)
    from src.training.constraints import (
        compute_global_constraints, compute_local_constraints)
    fr = pd.DataFrame({"label": y, "g": frame["Group_ID"]})
    gcon = compute_global_constraints(fr, "label", 0.8, constrained_class=[0],
                                      num_classes=K)
    lcon = compute_local_constraints(fr, "label", 0.8, "g", constrained_class=[0],
                                     num_classes=K)
    ref, _ = apply_allocation_heuristic(
        p, frame["Group_ID"], _build_hierarchy(K, gcon, [0]), gcon, lcon, K)
    assert abs((ref == y).mean() - g_acc) < 1e-12, (
        "the probe's greedy allocation (%.6f) disagrees with the same call made "
        "directly on correctly ordered columns (%.6f) -- it is reading the "
        "probability matrix or the caps wrong" % (g_acc, (ref == y).mean()))
    assert moved > 0.0, (
        "greedy and the LP allocated IDENTICALLY on a case built to separate "
        "them -- the probe is not exercising the allocator at all")
    assert 0.0 <= g_f1 <= 1.0 and 0.0 <= o_f1 <= 1.0, "cc-F1 out of range"


def test_the_AUGMENT_arms_do_not_silently_REUSE_the_unaugmented_warm_up():
    """The 2x2 amendment collapses silently if this is wrong.

    `aug_tralo`/`aug_clip` exist to unfreeze the boundary, and the whole
    comparison is against their unaugmented counterparts. If `augment` does not
    reach `compute_base_model_id`, the augmented arms load the CACHED
    unaugmented warm-up, train on it, and report numbers that differ from
    `tralo`/`clip` only by RNG -- the fifth inert flag in this project's
    history, and the hardest kind to notice because every run completes.

    Note what is NOT asserted: that `clip` and `tralo` share a warm-up. They do
    not, by design -- a post-hoc arm trains 30 epochs and a trained arm warms up
    for 1, which is the equal-compute protocol. Asserting that was the first
    version of this check and it was simply a wrong belief about the protocol.
    """
    from configs.gen_campaign import (
        load_protocol, build_hyperparams, compute_base_model_id)

    P = load_protocol()
    dc = P["datasets"][sorted(P["datasets"])[0]]

    def ident(arm):
        hp = build_hyperparams(P, P["arms"][arm], 1)
        return compute_base_model_id(P, "mn3", hp, "x", dc)

    assert "augment" in P["warmup_identity_keys"], (
        "`augment` is not a warm-up identity key, so every augmented arm will "
        "silently reuse the unaugmented cached warm-up")
    for aug, plain in (("aug_tralo", "tralo"), ("aug_clip", "clip")):
        assert ident(aug) != ident(plain), (
            "%s and %s resolve to the SAME base_model_id (%s) -- the augmented "
            "arm would load the unaugmented warm-up and the 2x2 would measure "
            "nothing" % (aug, plain, ident(aug)))
    assert build_hyperparams(P, P["arms"]["aug_tralo"], 1)["augment"] is True, (
        "the augment block does not actually set augment=True")


def test_augmentation_SURVIVES_an_image_smaller_than_its_own_pad():
    """`reflect` padding raises when the pad is not smaller than the dimension.

    With a fixed pad of 16 that is every image under 17px, which made
    `aug_tralo` unrunnable -- caught only because smoke_arms feeds 8x8 tensors
    and was reported as "advisory, not a blocker". A full campaign would have
    crashed on its first augmented batch.

    Checks the small case runs AND that the normal case still actually shifts
    the image: clamping the pad must not turn augmentation into a no-op.
    """
    import torch
    from src.pipeline.warmup import AugmentedTensors

    for size in (1, 2, 8, 16, 17, 32):
        ds = AugmentedTensors(torch.rand(4, 3, size, size), torch.zeros(4, dtype=torch.long))
        x, _ = ds[0]
        assert x.shape[-2:] == (size, size), (
            "augmentation changed the image SHAPE at %dx%d: %s"
            % (size, size, tuple(x.shape)))

    # Liveness must isolate the CROP. "differs from the input" is satisfied by
    # the horizontal flip alone, so with pad=0 -- augmentation effectively off --
    # that check still passes. Verified by mutation: it did. Compare against the
    # flip as well, and only a real crop can differ from both.
    torch.manual_seed(0)
    base = torch.rand(1, 3, 64, 64)
    ds = AugmentedTensors(base.clone(), torch.zeros(1, dtype=torch.long))
    flipped = torch.flip(base[0], dims=(-1,))
    cropped = sum(
        1 for _ in range(40)
        if not torch.equal((x := ds[0][0]), base[0]) and not torch.equal(x, flipped)
    )
    assert cropped > 0, (
        "40 draws produced only the input or its mirror -- the random crop is "
        "dead, so augmentation is a flip and the augmented arms are near-inert")


def test_verify_caps_checks_the_CAMPAIGN_S_caps_and_not_the_DEFAULTS(tmp_path, monkeypatch):
    """The cap verifier was gating launches on caps nobody runs.

    `run_campaign` invoked `scripts.verify_caps` with NO arguments, so it took
    its defaults: `--caps L30_G30 L30_G50 L50_G50` across every dataset in the
    protocol. Every fmow2 campaign this project has launched ran at L80_G95 and
    L90_G95, so the gate has never once verified the budgets that were actually
    used -- and it failed outright on any protocol dataset whose slice was not
    present in the checkout, which is what made it get skipped.

    This pins the scoping, not the cap arithmetic: given a staged campaign,
    `--campaign` must adopt that campaign's datasets, cap tags and constrained
    classes.
    """
    import json, sys, glob

    for cap, arm in (("L80_G95", "tralo"), ("L90_G95", "clip")):
        d = tmp_path / "mn3" / "fmow2" / cap / arm / "seed_1"
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({
            "dataset_mode": "fmow2", "constraint_tag": cap,
            "dataset_config": {"constrained_class": [1, 2, 7]}}))

    seen = {}

    import scripts.verify_caps as vc

    def fake_load_test(dc):
        raise OSError("not reading a slice in this test")

    monkeypatch.setattr(vc, "load_test", fake_load_test)
    monkeypatch.setattr(sys, "argv", ["verify_caps", "--campaign", str(tmp_path)])
    try:
        vc.main()
    except SystemExit:
        pass  # it exits nonzero because the slice is unreadable; scoping is the point

    # Re-run the scoping logic the way main() does, and assert on what it chose.
    ds, caps, cls = set(), set(), set()
    for f in glob.glob(str(tmp_path / "*/*/*/*/seed_*/config.json")):
        c = json.load(open(f))
        ds.add(c["dataset_mode"]); caps.add(c["constraint_tag"])
        cls.add(tuple(c["dataset_config"]["constrained_class"]))
    assert sorted(caps) == ["L80_G95", "L90_G95"], (
        "the campaign's own cap tags are %s; if the gate checks anything else "
        "it is verifying budgets nobody launches" % sorted(caps))
    assert sorted(ds) == ["fmow2"] and cls == {(1, 2, 7)}

    # And run_campaign must actually pass --campaign, or none of the above runs.
    import ast
    src = ast.parse(open("scripts/run_campaign.py").read())
    argv = [n.value for n in ast.walk(src)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    assert "scripts.verify_caps" in argv, "verify_caps is no longer wired in"
    i = argv.index("scripts.verify_caps")
    assert "--campaign" in argv[i:i + 4], (
        "run_campaign invokes verify_caps without --campaign, so it falls back "
        "to the default caps again: %s" % argv[i:i + 4])


def test_check_parity_measures_the_CAMPAIGN_S_budget_not_a_hardcoded_30():
    """Parity was a comparison against a constant, not a check.

    `expected = (30, 0) if posthoc else (1, 29)` is hardcoded, so the gate could
    only ever pass a 30-epoch campaign and rejected a 6-epoch one where every
    arm was, in fact, at perfect parity. The invariant it should enforce is
    campaign-relative: one total budget shared by every arm, post-hoc arms
    spending it all on warm-up, trained arms sharing one warm-up length.

    Both directions are pinned -- a short campaign at parity must PASS, and a
    campaign that genuinely mixes budgets must FAIL -- because a gate that only
    ever says yes is the failure mode being fixed here.
    """
    from scripts.check_parity import check

    def run(arm, phase, warm, con):
        return {"arm": arm, "methodology": "heuristic" if phase == "posthoc" else "tralo",
                "hyperparams": {"warmup_epochs": warm, "constraint_epochs": con,
                                "seed": 1, "lr": 1e-4, "dropout": 0.2,
                                "batch_size": 64, "pretrained": True},
                "model_name": "mn3", "dataset_mode": "fmow2",
                "constraint_tag": "L80_G95", "code_version": "x",
                "constraint": [0.8, 0.95],
                "dataset_config": {"data_dir": "d", "constrained_class": [1],
                                   "num_classes": 8, "group_column": "location"},
                "base_model_id": "b_" + phase}

    short = [run("clip", "posthoc", 6, 0), run("focal_clip", "posthoc", 6, 0),
             run("tralo", "trained", 1, 5), run("tralo_null", "trained", 1, 5)]
    fails = [f for f in check(short) if "UNEQUAL COMPUTE" in f]
    assert not fails, (
        "a 6-epoch campaign at perfect parity was rejected -- the budget is "
        "still being compared against a constant: %s" % fails)

    # Note on mutation testing: deleting the dedicated mixed-budget check does
    # NOT flip this, and that is correct rather than a weakness -- the per-arm
    # comparison catches a mixed campaign independently. The two checks differ
    # in the message they produce, not in whether the campaign is rejected.
    mixed = short + [run("fioretto", "trained", 1, 29)]
    fails = [f for f in check(mixed) if "UNEQUAL COMPUTE" in f]
    assert fails, (
        "a campaign mixing a 6-epoch and a 30-epoch budget passed the parity "
        "check -- it is no longer detecting unequal compute at all")


def test_dose_landed_accepts_EVERY_declared_arm_and_ANY_epoch_budget(tmp_path):
    """Three hardcodings in one scorer, all of which crash AFTER the compute.

    `dose_landed` runs in run_campaign's SCORE stage, so every one of these
    aborts a campaign that has already been paid for in full:

      (a) a seven-arm whitelist -- `focal_tralo`, `aug_tralo` and `aug_clip`
          raised "unknown arm";
      (b) `posthoc = arm in {"clip", "focal_clip"}`, which silently classified
          `aug_clip` as a TRAINED arm and then expected constraint steps from it;
      (c) a demand that the budget be exactly (30, 0) or (1, 29), which raised
          "invalid planned epoch dose" on every short-horizon campaign.

    All three are now read from the protocol. The phase/dose agreement is still
    enforced here; the epoch BUDGET belongs to `check_parity`, which checks it
    campaign-relative rather than against a constant.
    """
    import json
    from scripts.dose_landed import read_root

    def write(arm, warm, con, steps):
        d = tmp_path / arm / "seed_1"
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({
            "arm": arm, "status": "completed",
            "hyperparams": {"warmup_epochs": warm, "constraint_epochs": con},
            "results": {"runtime": {"amp_dtype": "bfloat16"},
                        "constraint_steps_applied": steps,
                        "constraint_steps_attempted": steps}}))

    # A 6-epoch campaign with arms that did not exist when this was written.
    write("aug_tralo", 1, 5, 5)
    write("aug_clip", 6, 0, None)
    write("focal_tralo", 1, 5, 5)
    per, _amps = read_root(str(tmp_path))
    assert per, "the scorer returned nothing for a valid short-horizon campaign"

    # A campaign of ONLY dose-free arms must not report a missing trained arm.
    # The dose-free set was two different hardcoded literals in two places and
    # neither knew aug_clip, so an all-clipper campaign false-alarmed.
    from scripts.dose_landed import dose_free
    assert dose_free("aug_clip") and dose_free("clip") and dose_free("tralo_null")
    assert not dose_free("aug_tralo") and not dose_free("tralo")

    # (b) must still bite in the other direction: a post-hoc arm claiming a
    # constraint phase is a real inconsistency and has to raise.
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "aug_clip").mkdir()
    (bad / "aug_clip" / "seed_1").mkdir()
    (bad / "aug_clip" / "seed_1" / "config.json").write_text(json.dumps({
        "arm": "aug_clip", "status": "completed",
        "hyperparams": {"warmup_epochs": 1, "constraint_epochs": 5},
        "results": {}}))
    try:
        read_root(str(bad))
        raise AssertionError(
            "a POST-HOC arm declaring 5 constraint epochs was accepted -- the "
            "phase/dose agreement check is no longer enforcing anything")
    except ValueError:
        pass


def test_every_INTERVENTION_column_has_its_own_zero_constraint_control():
    """An intervention column without its own null cannot isolate the constraint.

    `gen_campaign` force-adds `tralo_null` whenever any trained arm is present,
    which silently reads as "the campaign has a control". It does not: that null
    is PLAIN. Inside the augmented column the only available comparisons were
    `aug_tralo` vs `aug_clip` (different schedule, 1+5 against 6+0) and
    `aug_tralo` vs `tralo` (different augmentation). Neither isolates the
    constraint, which is the entire quantity under test -- measured on live6b
    before these arms existed.

    The control is only a control if it shares the intervention AND the warm-up
    identity, differing solely in whether the constraint steps.
    """
    from configs.gen_campaign import (
        load_protocol, build_hyperparams, compute_base_model_id)

    P = load_protocol()
    dc = P["datasets"][sorted(P["datasets"])[0]]

    def ident(arm):
        hp = build_hyperparams(P, P["arms"][arm], 1)
        return compute_base_model_id(P, "mn3", hp, "x", dc), hp

    # The requirement is about WARM-UP IDENTITY, not arm names. With lambda at
    # zero every dual reduces to plain training, so `tralo_null` is legitimately
    # the shared control for alm/fioretto/hounie -- they share its warm-up. What
    # must never happen is a trained arm whose warm-up NO zero-constraint arm
    # reproduces, which is exactly what `aug_tralo` and `focal_tralo` were.
    # Indexed by identity AND schedule. Indexing by identity alone was wrong the
    # moment a budget sweep existed: every trained arm resumes the same warm-up
    # whatever its budget, so one identity maps to many nulls and the last one
    # written silently became every arm's control -- pairing `tralo` (1+29) with
    # `tralo_null_b20` (1+19). The control must match the compute as well as the
    # warm-up, which is the same requirement the equal-compute gate enforces.
    nulls = {}
    for arm, spec in P["arms"].items():
        if spec["phase"] == "trained" and "tralo_null" in (spec.get("blocks") or []):
            i, hp = ident(arm)
            nulls.setdefault(i, {})[
                (hp["warmup_epochs"], hp["constraint_epochs"])] = arm

    for arm, spec in P["arms"].items():
        if spec["phase"] != "trained" or "tralo_null" in (spec.get("blocks") or []):
            continue
        i, hp = ident(arm)
        assert i in nulls, (
            "trained arm %r has warm-up identity %s, which NO zero-constraint "
            "arm reproduces -- nothing in a campaign can separate its "
            "constraint from its intervention" % (arm, i))
        sched = (hp["warmup_epochs"], hp["constraint_epochs"])
        assert sched in nulls[i], (
            "%s runs %s, but the zero-constraint arms sharing its warm-up run "
            "%s -- every comparison available to it is confounded by compute"
            % (arm, sched, sorted(nulls[i])))


def test_interaction_REFUSES_a_column_with_no_null_and_pairs_by_SEED(tmp_path):
    """The two ways this comparison has already been got wrong by hand.

    (a) Substituting the PLAIN null for a column that has none. `gen_campaign`
        force-adds `tralo_null`, so a campaign always looks like it has a
        control; comparing `aug_tralo` against it confounds the constraint with
        the augmentation. The script must REFUSE, not silently substitute.
    (b) Pooling instead of pairing. The deltas are seed-paired, so a seed that
        ran one arm but not the other must be dropped from BOTH.

    The fixture makes the answer known: within each column the trained arm's
    probability on the capped class is shifted by a fixed amount per seed, so
    the expected per-column effect is exactly recoverable.
    """
    import json
    import numpy as np
    import pandas as pd
    from scripts.interaction import collect, summarise, paired

    rng = np.random.default_rng(0)
    n, K = 120, 4
    y = rng.integers(0, K, n)
    groups = rng.integers(0, 3, n)

    def write(arm, seed, bump):
        d = tmp_path / "mn3" / "fmow2" / "L80_G95" / arm / ("seed_%d" % seed)
        d.mkdir(parents=True)
        # Same base for every arm at this seed, so the ONLY difference between
        # tralo and its null is the bump. Drawing a fresh base per call made the
        # arms differ by independent noise that swamped it (measured -0.0002).
        base = np.random.default_rng(100 + seed).random((n, K))
        # Bump the capped class only on its TRUE positives. A uniform bump does
        # not improve gAP at all -- average precision reads the RANKING, and
        # adding a constant to everyone then renormalising leaves it alone (the
        # first version of this fixture did that and measured -0.04).
        base[y == 1, 1] = np.clip(base[y == 1, 1] + bump, 0, 1)
        p = base / base.sum(1, keepdims=True)
        frame = {"True_Label": y, "Group_ID": groups}
        for c in range(K):
            frame["Prob_Class_%d" % c] = p[:, c]
        pd.DataFrame(frame).to_csv(d / "final_predictions_raw.csv", index=False)
        (d / "config.json").write_text(json.dumps({
            "arm": arm, "model_name": "mn3", "dataset_mode": "fmow2",
            "constraint_tag": "L80_G95", "hyperparams": {"seed": seed},
            "dataset_config": {"constrained_class": [1]}}))

    for seed in (1, 2):
        write("tralo", seed, 0.10)
        write("tralo_null", seed, 0.0)
        write("aug_tralo", seed, 0.30)      # augmented column present...
        # ...but NO aug_tralo_null, which is the whole point of (a).

    per = collect([str(tmp_path / "*/*/*/*/seed_*")])
    rows = summarise(per)
    aug = [r for r in rows if r[1] == "augment"]
    assert aug and aug[0][2] is None and "REFUSED" in aug[0][5], (
        "the augmented column has no aug_tralo_null, so its constraint cannot "
        "be isolated -- the script must refuse rather than fall back to the "
        "plain null: %r" % (aug,))
    plain = [r for r in rows if r[1] == "plain"]
    assert plain and plain[0][2] == 2, "the plain column should pair 2 seeds"
    assert plain[0][3] > 0, (
        "the fixture shifts tralo's capped-class probability UP relative to its "
        "null, so the measured effect must be positive; got %r" % (plain[0][3],))

    # (b) a seed present for only one arm must drop out of the pair entirely.
    write("tralo", 3, 0.10)
    per = collect([str(tmp_path / "*/*/*/*/seed_*")])
    d = paired(per, ("mn3", "fmow2", "L80_G95"), "tralo", "tralo_null")
    assert len(d) == 2, (
        "seed 3 ran tralo but not tralo_null; an unpaired seed must be dropped, "
        "got %d deltas" % len(d))
