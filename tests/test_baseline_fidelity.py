"""Maintained behavioral regression fixtures."""

import ast

import copy

import csv

import hashlib

import io

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

from configs.gen_campaign import build_hyperparams, cap_pair, compute_base_model_id

from lean_fixtures import protocol_with_nulls as load_protocol

from src.utils.constants import UNLIMITED

DUAL_ARMS = ("tralo", "fioretto", "hounie", "alm")

NULL_ARMS = ("tralo_null", "fioretto_null", "hounie_null", "alm_null")


@pytest.fixture(scope="module")
def P():
    return load_protocol()


def _run_arm(P, arm, epochs=3, return_probabilities=False, **overrides):
    from scripts.smoke_arms import make_inputs
    from src.experiments.runner import TRAIN_FNS

    tmp = tempfile.mkdtemp()
    (inputs, _g, _l) = make_inputs(P, arm, tmp, seed=1)
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
        col = next(
            (c for c in ("grad_norm", "Grad_Norm") if rows and c in rows[0]), None
        )
        if col:
            norms = [float(r[col]) for r in rows if r[col] not in ("", None)]
    return (proba if return_probabilities else md5, out.summary, norms)


def _write_campaign(
    root,
    P,
    arms,
    caps=("L30_G50", "L50_G30"),
    seeds=(1, 2),
    hp_patch=None,
    dataset="iwildcam",
    model="MobileNetV3",
):
    arms = sorted(set(arms) | {"clip", "focal_clip", "tralo_null"})
    dc = P["datasets"][dataset]
    for arm in arms:
        for tag in caps:
            for seed in seeds:
                hp = build_hyperparams(P, P["arms"][arm], seed)
                if hp_patch:
                    hp_patch(arm, hp)
                path = os.path.join(root, model, dataset, tag, arm, "seed_%d" % seed)
                os.makedirs(path, exist_ok=True)
                cfg = {
                    "methodology": P["arms"][arm]["methodology"],
                    "model_name": model,
                    "constraint": cap_pair(tag),
                    "constraint_tag": tag,
                    "dataset_mode": dataset,
                    "dataset_config": dc,
                    "hyperparams": hp,
                    "base_model_id": compute_base_model_id(P, model, hp, dataset, dc),
                    "arm": arm,
                    "exp_name": "%s_%s_%d" % (arm, tag, seed),
                    "status": "pending",
                    "code_version": "aaaaaaaaaaaa",
                }
                json.dump(cfg, open(os.path.join(path, "config.json"), "w"), indent=2)
    return root


def _parity(root):
    return subprocess.run(
        [sys.executable, "-m", "scripts.check_parity", str(root)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


def test_the_protocol_pins_lr_constraint_to_lr(P):
    assert P["core"]["lr"] == P["constraint_phase"]["lr_constraint"]
    for block, spec in P["blocks"].items():
        if "lr_constraint" in spec:
            assert spec["lr_constraint"] == P["core"]["lr"], (
                "block %r sets lr_constraint %s against core.lr %s"
                % (block, spec["lr_constraint"], P["core"]["lr"])
            )


def test_an_unequal_lr_constraint_detunes_29_of_30_TRAINED_EPOCHS(P):

    def optimizer_lr_args(path):
        tree = ast.parse(open(os.path.join(REPO, path), encoding="utf-8").read())
        out = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "make_optimizer"
            ):
                out.append(ast.unparse(node.args[1]))
        return out

    assert optimizer_lr_args("src/methodologies/tralo/train.py") == ["lr_constraint"]
    assert optimizer_lr_args("src/methodologies/dual_common.py") == ["lr"]
    dual = ast.parse(
        open(
            os.path.join(REPO, "src/methodologies/dual_common.py"), encoding="utf-8"
        ).read()
    )
    setup = next(
        (
            n
            for n in ast.walk(dual)
            if isinstance(n, ast.FunctionDef) and n.name == "dual_setup"
        )
    )
    assert [a.arg for a in setup.args.args][3] == "lr", (
        "dual_setup's third positional is no longer the learning rate"
    )
    for meth in ("fioretto_ldf", "hounie_rcl", "fioretto_alm"):
        src = open(
            os.path.join(REPO, "src/methodologies", meth, "train.py"), encoding="utf-8"
        ).read()
        tree = ast.parse(src)
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "dual_setup"
        ]
        assert calls, meth
        assert ast.unparse(calls[0].args[3]) == "lr_c", meth
        assigns = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "lr_c"
        ]
        assert assigns and "lr_constraint" in ast.unparse(assigns[0].value), meth
    tralo = open(
        os.path.join(REPO, "src/methodologies/tralo/train.py"), encoding="utf-8"
    ).read()
    assert 'pg["lr"] = lr_constraint' in tralo


def test_the_generator_refuses_an_unequal_lr_constraint(tmp_path, P):
    trapped = copy.deepcopy(P)
    trapped["constraint_phase"]["lr_constraint"] = 5e-06
    proto = tmp_path / "trap_protocol.yml"
    proto.write_text(yaml.safe_dump(trapped), encoding="utf-8")
    argv = [
        "--root",
        str(tmp_path / "camp"),
        "--datasets",
        "iwildcam",
        "--models",
        "MobileNetV3",
        "--caps",
        "L30_G50",
        "L50_G30",
        "--arms",
        "tralo",
        "tralo_null",
    ]
    bad = subprocess.run(
        [sys.executable, "-m", "configs.gen_campaign", "--protocol", str(proto)] + argv,
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0, "the generator emitted an LR-trapped campaign"
    assert "lr_constraint" in bad.stdout + bad.stderr
    ok = subprocess.run(
        [sys.executable, "-m", "configs.gen_campaign"] + argv,
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stdout[-1500:] + ok.stderr[-1500:]


def test_check_parity_REFUSES_the_lr_trap(tmp_path, P):

    def trap(arm, hp):
        if "lr_constraint" in hp:
            hp["lr_constraint"] = 5e-06

    root = _write_campaign(
        str(tmp_path / "trapped"), P, ["clip", "tralo", "tralo_null"], hp_patch=trap
    )
    r = _parity(root)
    assert r.returncode == 1, (
        "check_parity passed a campaign whose trained arms train at 5e-6 while the clipper trains at 1e-4:\n"
        + r.stdout[-2000:]
    )
    assert "lr_constraint" in r.stdout
    clean = _write_campaign(str(tmp_path / "clean"), P, ["clip", "tralo", "tralo_null"])
    ok = _parity(clean)
    assert ok.returncode == 0, ok.stdout[-2500:]


def test_every_arm_gets_the_same_optimizer_epochs(P):
    total = P["protocol"]["total_epochs"]
    for arm, spec in P["arms"].items():
        hp = build_hyperparams(P, spec, 1)
        assert hp["warmup_epochs"] + hp["constraint_epochs"] == total, arm
        if spec["phase"] == "posthoc":
            assert (hp["warmup_epochs"], hp["constraint_epochs"]) == (total, 0), arm
        else:
            assert hp["warmup_epochs"] == P["protocol"]["trained_warmup"], arm


def test_no_trained_arm_can_early_stop_out_of_its_constraint_budget(P):
    from src.pipeline.config import validate_hyperparams

    for arm in DUAL_ARMS:
        hp = build_hyperparams(P, P["arms"][arm], 1)
        assert "stable_count_threshold" not in hp
        hp["stable_count_threshold"] = 1
        with pytest.raises(ValueError, match="stable_count_threshold"):
            validate_hyperparams(P["arms"][arm]["methodology"], hp)


def test_the_warmup_cache_key_covers_everything_the_warmup_reads():
    from scripts.audit_config import WARMUP_PATH, WARMUP_EXTRA, _keys_in, _walk

    paths = []
    for f in WARMUP_PATH:
        paths += _walk(f) if os.path.isdir(f) else [f]
    read = _keys_in(paths) | WARMUP_EXTRA
    proto = yaml.safe_load(
        open(os.path.join(REPO, "configs/protocol.yml"), encoding="utf-8")
    )
    declared = set(proto["warmup_identity_keys"])
    assert not read - declared, sorted(read - declared)
    assert read - {"warmup_loss"} - declared == set()
    assert read & {"warmup_loss"}, (
        "the audit no longer sees warmup_loss on the warm-up path, so this gate would pass on a protocol that omitted it"
    )


def test_the_cap_never_reaches_the_warmup_so_a_shared_cache_is_legitimate():
    proto = yaml.safe_load(
        open(os.path.join(REPO, "configs/protocol.yml"), encoding="utf-8")
    )
    assert "constraint" not in proto["warmup_identity_keys"]
    src = open(os.path.join(REPO, "src/utils/data_loader.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    tainted = {
        "constrained_class",
        "local_percent",
        "global_percent",
        "global_con",
        "local_con",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = {getattr(t, "id", None) for t in node.targets}
            if names & {"X_train", "y_train"}:
                used = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
                assert not used & tainted, (
                    "the training split is built from %s -- the cap has reached the warm-up"
                    % sorted(used & tainted)
                )


def test_the_four_trained_arms_share_one_warmup_and_the_clippers_do_not(tmp_path, P):
    ids = {}
    for arm in (
        "clip",
        "focal_clip",
        "tralo",
        "tralo_null",
        "fioretto",
        "hounie",
        "alm",
    ):
        hp = build_hyperparams(P, P["arms"][arm], 1)
        dc = P["datasets"]["iwildcam"]
        ids[arm] = compute_base_model_id(P, "MobileNetV3", hp, "iwildcam", dc)
    assert (
        len({ids[a] for a in ("tralo", "tralo_null", "fioretto", "hounie", "alm")}) == 1
    )
    assert ids["clip"] != ids["focal_clip"], (
        "focal_clip shares clip's cached model and is therefore a second clip"
    )
    assert ids["clip"] != ids["tralo"], "warm-up 30 and warm-up 1 share a cache"


def _inline_defaults(paths):
    hp_names = {"hp", "hyperparams", "hparams"}
    found = []
    for path in paths:
        full = os.path.join(REPO, path)
        for node in ast.walk(ast.parse(open(full, encoding="utf-8").read())):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and (node.func.attr == "get")
                and (len(node.args) == 2)
            ):
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
    (vals, base) = (set(), None)
    for block, spec in P["blocks"].items():
        if isinstance(spec, dict) and key in spec:
            vals.add(json.dumps(spec[key]))
    for section in ("core", "constraint_phase", "chunked"):
        if key in P.get(section, {}):
            vals.add(json.dumps(P[section][key]))
            base = json.dumps(P[section][key])
    return (vals, base)


def test_no_inline_default_disagrees_with_the_protocol(P):

    def offenders(pairs):
        bad = []
        for key, default, path, line in pairs:
            if key in ("constraint_grad_mode", "constraint_fp32"):
                for spec in P["arms"].values():
                    if spec["phase"] == "trained":
                        assert (
                            build_hyperparams(P, spec, 1)[key]
                            == P["constraint_phase"][key]
                        )
                continue
            (vals, _base) = _protocol_values(P, key)
            if not vals:
                continue
            if json.dumps(default) not in vals:
                bad.append((key, default, sorted(vals), path, line))
        return bad

    stand_in = [("hounie_eta_lambda", 0.5, "src/methodologies/hounie_rcl/train.py", 0)]
    assert offenders(stand_in), (
        "the checker cannot see a default that disagrees with the protocol -- it would pass on the very defect it exists to catch. If 0.5 has since become a protocol value, move the stand-in, do not delete it."
    )
    live = _inline_defaults(
        [
            "src/methodologies/tralo/train.py",
            "src/methodologies/fioretto_ldf/train.py",
            "src/methodologies/hounie_rcl/train.py",
            "src/methodologies/fioretto_alm/train.py",
            "src/methodologies/dual_common.py",
            "src/methodologies/heuristic/train.py",
        ]
    )
    assert live, "the AST walker found no inline defaults at all -- it is broken"
    assert not offenders(live), offenders(live)


def test_the_core_knobs_are_emitted_on_every_arm_so_no_default_can_fire(P):
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
    must_be_required = {
        "src/methodologies/fioretto_ldf/train.py": ["fioretto_step_size"],
        "src/methodologies/hounie_rcl/train.py": ["hounie_eta_lambda", "hounie_eta_u"],
        "src/methodologies/fioretto_alm/train.py": [
            "alm_eta",
            "alm_mu0",
            "alm_mu_step",
        ],
        "src/methodologies/tralo/train.py": ["lambda_global", "lambda_local"],
    }
    for path, keys in must_be_required.items():
        defaults = {k for (k, _d, _p, _l) in _inline_defaults([path])}
        assert not defaults & set(keys), "%s defaults %s instead of requiring it" % (
            path,
            sorted(defaults & set(keys)),
        )
        src = open(os.path.join(REPO, path), encoding="utf-8").read()
        tree = ast.parse(src)
        required = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "_required"
                and node.args
                and isinstance(node.args[1], ast.Constant)
            ):
                required.add(node.args[1].value)
        subscripts = {
            n.slice.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Subscript)
            and isinstance(n.slice, ast.Constant)
            and isinstance(n.slice.value, str)
        }
        for k in keys:
            assert k in required or k in subscripts, "%s: %s" % (path, k)


def test_the_ALM_augmentation_is_LIVE_so_alm_is_not_a_second_fioretto(
    P, tmp_path, monkeypatch
):
    import importlib
    from scripts.smoke_arms import make_inputs

    def capture(arm, mu, eta, initial):
        (inputs, _, _) = make_inputs(P, arm, tmp_path)
        inputs.model = torch.nn.Linear(2, 2, bias=False).double()
        with torch.no_grad():
            inputs.model.weight.zero_()
        inputs.X_test = torch.tensor(
            [[1.0, 0.0]] * 2 + [[0.0, 1.0]] * 6, dtype=torch.float64
        )
        inputs.X_train = inputs.X_test.clone()
        inputs.y_train = torch.zeros(8, dtype=torch.long)
        inputs.group_ids = np.array([0] * 2 + [1] * 6)
        inputs.num_classes = 2
        inputs.constrained_classes = [1]
        inputs.global_con = [UNLIMITED, UNLIMITED]
        inputs.local_con = {0: [UNLIMITED, 0], 1: [UNLIMITED, 0]}
        inputs.config["dataset_config"]["num_classes"] = 2
        inputs.hyperparams.update(
            constraint_epochs=1,
            constraint_fp32=True,
            constraint_chunk_size=3,
            constraint_grad_mode="normalize",
            lr_constraint=0.0,
            alm_mu0=mu,
            alm_mu_step=0.0,
            alm_eta=eta,
            fioretto_step_size=eta,
            fioretto_lambda_init=initial,
        )
        trainer = importlib.import_module(
            "src.methodologies." + P["arms"][arm]["methodology"] + ".train"
        )
        original = trainer.finish_constraint_step
        gradients = []

        def observe(model, *args, **kwargs):
            raw = model.weight.grad.detach().clone()
            result = original(model, *args, **kwargs)
            gradients.append((raw, model.weight.grad.detach().clone()))
            return result

        with monkeypatch.context() as patch:
            patch.setattr(trainer, "ce_epoch", lambda *args: ([0.0], 0.0))
            patch.setattr(trainer, "finish_constraint_step", observe)
            trainer._train_constraints(inputs.model, inputs, inputs.device)
        return gradients

    live = capture("alm", mu=0.01, eta=0.0, initial=0.0)
    dead = capture("alm", mu=0.0, eta=0.0, initial=0.0)
    assert len(live) == 1
    assert dead == []
    torch.testing.assert_close(
        live[0][0],
        torch.tensor([[-0.005, -0.045], [0.005, 0.045]], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    ldf = capture("fioretto", mu=0.0, eta=0.005, initial=0.2)
    off = capture("alm", mu=0.0, eta=0.005, initial=0.2)
    on = capture("alm", mu=0.01, eta=0.005, initial=0.2)
    assert len(ldf) == len(off) == len(on) == 1
    for expected, actual in zip(ldf[0], off[0]):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        off[0][0],
        torch.tensor([[-0.1025, -0.3225], [0.1025, 0.3225]], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        on[0][0],
        torch.tensor([[-0.1075, -0.3675], [0.1075, 0.3675]], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.linalg.vector_norm(on[0][1] - off[0][1]) > 0.01


def test_neither_grad_mode_puts_the_duals_at_a_COMPARABLE_dose(P):
    raw = {}
    for arm in DUAL_ARMS:
        (_md5, _s, norms) = _run_arm(
            P, arm, epochs=6, constraint_grad_mode="clip", constraint_fp32=True
        )
        raw[arm] = max((n for n in norms if n > 0))
    clip = P["constraint_phase"]["constraint_grad_clip"]
    assert raw["hounie"] < clip, raw
    assert raw["fioretto"] > clip and raw["alm"] > clip, raw
    assert min(raw["fioretto"], raw["alm"]) / raw["hounie"] > 100.0, raw


def test_hounie_alpha_moves_predictions_in_the_historical_clip_fp32_fixture(P):
    hp = P["blocks"]["hounie"]
    assert (hp["hounie_eta_lambda"], hp["hounie_eta_u"], hp["hounie_alpha"]) == (
        0.1,
        0.1,
        1.0,
    ), hp
    assert abs(1 - 2 * hp["hounie_eta_u"] * hp["hounie_alpha"]) < 1.0
    at = {
        a: _run_arm(
            P,
            "hounie",
            epochs=8,
            hounie_alpha=a,
            constraint_grad_mode="clip",
            constraint_fp32=True,
            return_probabilities=True,
        )[0]
        for a in (0.5, 1.0, 4.0)
    }
    tolerance = 2 * np.finfo(np.float32).eps
    for a, b in ((0.5, 1.0), (1.0, 4.0), (0.5, 4.0)):
        difference = float(np.max(np.abs(at[a] - at[b])))
        assert difference > tolerance, (a, b, difference, tolerance)


def test_the_alpha_liveness_gate_can_tell_a_dead_dose_from_a_live_one(P):
    dead = {
        a: _run_arm(
            P,
            "hounie",
            epochs=8,
            hounie_alpha=a,
            constraint_grad_mode="clip",
            constraint_fp32=True,
            return_probabilities=True,
            hounie_eta_lambda=0.01,
            hounie_eta_u=0.01,
        )[0]
        for a in (0.05, 1.0, 10.0)
    }
    tolerance = 2 * np.finfo(np.float32).eps
    for a, b in ((0.05, 1.0), (1.0, 10.0), (0.05, 10.0)):
        difference = float(np.max(np.abs(dead[a] - dead[b])))
        assert difference <= tolerance, (a, b, difference, tolerance)


def test_every_dual_arm_TAKES_EVERY_CONSTRAINT_STEP(P):
    epochs = 4
    steps = {}
    for arm in DUAL_ARMS:
        (_md5, summary, _norms) = _run_arm(P, arm, epochs=epochs)
        steps[arm] = (
            summary.get("constraint_steps_applied"),
            summary.get("constraint_steps_attempted"),
        )
    for arm, (applied, attempted) in steps.items():
        assert applied == attempted, (
            "%s dropped a step to a non-finite gradient: %s" % (arm, steps[arm])
        )
    for arm in DUAL_ARMS:
        assert steps[arm][0] == epochs, (
            "%s took %d of %d constraint steps -- the arms are not at equal dose, which is the defect this test exists to prevent: %s"
            % (arm, steps[arm][0], epochs, steps)
        )


def test_only_normalize_gives_the_trained_arms_the_same_constraint_step(P):
    raw = {}
    for arm in DUAL_ARMS:
        (_md5, _summary, norms) = _run_arm(P, arm, epochs=4)
        live = [n for n in norms if n > 0]
        assert live, arm
        raw[arm] = live
    clip = P["constraint_phase"]["constraint_grad_clip"]

    def delivered(norms):
        return [min(n, clip) for n in norms]

    lo = min((min(delivered(v)) for v in raw.values()))
    hi = max((max(delivered(v)) for v in raw.values()))
    assert hi / lo > 10.0, (
        "the arms' delivered constraint steps are within 10x of each other, so this gate no longer measures the asymmetry it was written for: %s"
        % {a: [round(x, 6) for x in delivered(v)] for (a, v) in raw.items()}
    )
    assert max(delivered(raw["hounie"])) < clip
    assert max(delivered(raw["fioretto"])) == pytest.approx(clip)
    for arm in DUAL_ARMS:
        (_md5, _summary, norms) = _run_arm(
            P, arm, epochs=4, constraint_grad_mode="normalize"
        )
        live = [n for n in norms if n > 0]
        assert live, arm


def test_normalize_delivers_exactly_the_clip_for_any_raw_scale():
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

    assert delivered(0.05, "clip") == pytest.approx(0.05, rel=1e-05)
    assert delivered(5.0, "clip") == pytest.approx(1.0, rel=1e-05)
    assert delivered(0.05, "normalize") == pytest.approx(1.0, rel=1e-05)
    assert delivered(5.0, "normalize") == pytest.approx(1.0, rel=1e-05)


def _budget_case(seed, n=800, C=8, G=5, capped=(2, 7), lp=0.3, gp=0.5, zero_frac=0.5):
    import pandas as pd
    from src.training.constraints import (
        compute_global_constraints,
        compute_local_constraints,
    )

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
    gcon = compute_global_constraints(
        df, "label", gp, constrained_class=list(capped), num_classes=C
    )
    lcon = compute_local_constraints(
        df, "label", lp, "grp", constrained_class=list(capped), num_classes=C
    )
    return (proba, groups, y, gcon, lcon, C)


@pytest.mark.parametrize(
    "lp,gp,zero_frac",
    [(0.3, 0.5, 0.5), (0.5, 0.3, 0.5), (0.5, 0.3, 0.0), (0.3, 0.3, 0.0)],
)
def test_the_clipper_and_the_trained_arms_emit_the_SAME_capped_count(lp, gp, zero_frac):
    from src.methodologies.heuristic.train import (
        _build_hierarchy,
        apply_allocation_heuristic,
    )
    from src.utils.posthoc_adjustment import targeted_correction

    capped = [2, 7]
    for seed in range(6):
        (proba, groups, _y, gcon, lcon, C) = _budget_case(
            seed, lp=lp, gp=gp, zero_frac=zero_frac
        )
        (greedy, _t) = apply_allocation_heuristic(
            proba, groups, _build_hierarchy(C, gcon, capped), gcon, lcon, C
        )
        (trained, _flips, _meta) = targeted_correction(
            proba, groups, gcon, lcon, capped, force_exact=True
        )
        for c in capped:
            reachable = min(
                int(gcon[c]),
                sum((int(lcon[g][c]) for g in lcon if lcon[g][c] < UNLIMITED)),
            )
            assert int((greedy == c).sum()) == reachable, (seed, c, "clip")
            assert int((trained == c).sum()) == reachable, (seed, c, "trained")


def test_every_null_arm_zeroes_its_family_by_config(P):
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
    assert P["blocks"]["alm"]["alm_mu0"] > 0
    assert P["blocks"]["alm_null"]["alm_mu0"] == 0


def test_the_zero_dose_siblings_are_ONE_model(P):
    nulls = {arm: _run_arm(P, arm)[0] for arm in NULL_ARMS}
    assert len(set(nulls.values())) == 1, nulls
    baseline = next(iter(nulls.values()))
    for arm in DUAL_ARMS:
        assert _run_arm(P, arm)[0] != baseline, (
            "%s is bit-identical to the zero-dose control -- its treatment is inert"
            % arm
        )


def test_a_null_arm_never_forms_a_constraint_gradient(P):
    for arm in NULL_ARMS:
        (_md5, summary, norms) = _run_arm(P, arm, epochs=4)
        assert summary.get("constraint_steps_attempted") in (0, None), (arm, summary)
        assert summary.get("constraint_steps_applied") in (0, None), (arm, summary)
        assert all((n == 0.0 for n in norms)), (arm, norms)


def test_headroom_uses_the_BINDING_budget_not_the_inert_global():
    from scripts.headroom import effective_budget
    from src.utils.constants import UNLIMITED

    G = {2: 185}
    L = {
        130: {2: 0},
        218: {2: 0},
        320: {2: 0},
        516: {2: 0},
        1: {2: 31},
        2: {2: 32},
        3: {2: 48},
    }
    assert effective_budget(G, L, 2) == 111, (
        "the inert global is being used; this is the 30x inflation"
    )
    assert effective_budget({2: 50}, L, 2) == 50
    L_open = dict(L)
    L_open[9] = {2: UNLIMITED}
    assert effective_budget(G, L_open, 2) == 185


def test_the_lp_fallback_fields_are_a_DEFAULT_for_the_post_hoc_arms():
    import io as _io
    import os
    import yaml

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
                if (
                    isinstance(kw, ast.keyword)
                    and kw.arg == "skip_targeted_correction"
                    and isinstance(kw.value, ast.Constant)
                    and (kw.value.value is True)
                ):
                    skippers.add(
                        os.path.basename(dirpath)
                        if os.path.basename(dirpath) != "methodologies"
                        else os.path.splitext(f)[0]
                    )
    assert "heuristic" in skippers, skippers
    ev = _io.open("src/pipeline/eval.py", encoding="utf-8").read()
    tree = ast.parse(ev)
    fn = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and any(
            (
                isinstance(x, ast.Assign)
                and any(
                    (
                        isinstance(t, ast.Name) and t.id == "posthoc_meta"
                        for t in x.targets
                    )
                )
                and isinstance(x.value, ast.Dict)
                and (not x.value.keys)
                for x in ast.walk(n)
            )
        )
    ]
    assert fn, (
        "src/pipeline/eval.py no longer initialises posthoc_meta to an empty dict; re-derive this test's premise"
    )
    rn = _io.open("src/experiments/runner.py", encoding="utf-8").read()
    tree = ast.parse(rn)
    defaults = {}
    for c in ast.walk(tree):
        if (
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and (c.func.attr == "get")
            and (len(c.args) == 2)
            and isinstance(c.args[0], ast.Constant)
            and str(c.args[0].value).startswith("lp_fallback")
        ):
            defaults[c.args[0].value] = getattr(c.args[1], "value", "?")
    assert defaults.get("lp_fallback_used") is False, defaults
    assert defaults.get("lp_fallback_candidates") == 0, defaults
    P = yaml.safe_load(_io.open("configs/protocol.yml", encoding="utf-8").read())
    defaulted = sorted(
        (a for (a, spec) in P["arms"].items() if spec.get("methodology") in skippers)
    )
    assert set(defaulted) == {"clip", "focal_clip"}, defaulted
    assert "tralo" not in defaulted and "fioretto" not in defaulted, defaulted


def test_the_constraint_step_is_NOT_inside_the_CE_batch_loop():
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

    batch_loops = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.For)
        and isinstance(n.iter, ast.Name)
        and ("loader" in n.iter.id)
    ]
    assert batch_loops, (
        "no `for ... in *loader` loop found in tralo/train.py, so this gate cannot locate the CE batch loop it exists to reason about"
    )
    for loop in batch_loops:
        assert calls(loop, "step"), (
            "the CE batch loop no longer takes an optimizer step, so the 126-steps-between figure is wrong in the other direction"
        )
        assert not calls(loop, "finish_constraint_step"), (
            "finish_constraint_step is now INSIDE the CE batch loop. Constraint steps would then be consecutive-ish and the momentum WOULD accumulate a count-function difference geometrically -- which reverses the analysis in FRAMEWORK 1b-pre(6) and in scripts/ortho_survival --compounding. Re-derive both before shipping this."
        )
    assert calls(tree, "finish_constraint_step"), (
        "tralo/train.py no longer calls finish_constraint_step at all"
    )


def test_a_probability_clamp_SURVIVES_THE_DTYPE_IT_ACTUALLY_RUNS_IN():
    from src.utils.constants import clamp_probability, clamp_denominator

    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        p = torch.tensor([0.0, 1e-12, 0.5, 1.0], dtype=dtype, requires_grad=True)
        q = clamp_probability(p)
        assert ((q > 0) & (q < 1)).all()
        u = torch.log(q) - torch.log1p(-q)
        assert torch.isfinite(u).all()
        denominator = clamp_denominator(torch.zeros(3, dtype=dtype))
        assert torch.isfinite(denominator).all() and (denominator > 0).all()
    u.sum().backward()
    assert torch.isfinite(p.grad).all()


def test_the_dose_reader_CATCHES_BOTH_HISTORICAL_FAILURES():
    from scripts.dose_landed import report, self_test

    buf = io.StringIO()
    assert self_test(out=buf) == 0, buf.getvalue()
    assert "SELF-TEST PASS" in buf.getvalue()
    buf = io.StringIO()
    n = report(
        {"tralo": [29, 29, 1, 0, 0, 0], "tralo_uniform": [1, 29, 1, 0, 0, 0]},
        {"tralo": {"bfloat16"}, "tralo_uniform": {"bfloat16"}},
        out=buf,
    )
    text = buf.getvalue()
    assert n >= 2 and "DID NOT RUN AT THE SAME DOSE" in text, text
    assert "LOSS SHAPE" in text, text
    buf = io.StringIO()
    n = report(
        {"tralo": [716, 1044, 36, 0, 0, 0], "fioretto": [720, 1044, 36, 0, 0, 0]},
        {"tralo": {"float16"}, "fioretto": {"float16"}},
        out=buf,
    )
    text = buf.getvalue()
    assert n >= 2, text
    assert "STEP(S) LOST" in text, text
    buf = io.StringIO()
    assert (
        report(
            {"tralo": [29, 29, 1, 0, 0, 0], "hounie": [29, 29, 1, 0, 0, 0]}, {}, out=buf
        )
        == 0
    ), buf.getvalue()
    buf = io.StringIO()
    assert report({"clip": [0, 0, 0, 4, 0, 0]}, {}, out=buf) == 0
    assert "normal state at the very start" in buf.getvalue()
    buf = io.StringIO()
    report(
        {"tralo": [29, 29, 1, 0, 0, 0], "tralo_uniform": [0, 0, 0, 36, 0, 0]},
        {"tralo": {"bfloat16"}},
        out=buf,
    )
    text = buf.getvalue()
    assert "36 still pending or running" in text, text
    assert "predate" not in text, text
    buf = io.StringIO()
    report(
        {"tralo": [29, 29, 1, 0, 0, 0], "clip": [0, 0, 0, 0, 36, 0]},
        {"tralo": {"bfloat16"}},
        out=buf,
    )
    assert "predate the field" in buf.getvalue(), buf.getvalue()
    buf = io.StringIO()
    report(
        {"tralo": [29, 29, 1, 0, 0, 0], "tralo_null": [0, 0, 4, 0, 0, 0]}, {}, out=buf
    )
    assert "lambda=0 twin does" in buf.getvalue(), buf.getvalue()


def test_no_numerical_guard_in_the_TRAINING_PATH_is_a_no_op():
    from src.losses.transductive_loss import MulticlassTransductiveLoss
    from src.training.constraint_step import finish_constraint_step

    loss = MulticlassTransductiveLoss([10000000000.0] * 3, {}, num_classes=3)
    for K in (0, 1, 500):
        for soft in (0.0, 0.5, 1000000.0):
            value = torch.tensor(soft, requires_grad=True)
            penalty = loss._penalty(value, K)
            penalty.backward()
            assert torch.isfinite(penalty) and torch.isfinite(value.grad)
    net = torch.nn.Linear(3, 2)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    before = [p.detach().clone() for p in net.parameters()]
    for prm in net.parameters():
        prm.grad = torch.zeros_like(prm)
    (norm, applied) = finish_constraint_step(net, opt, None, 1.0, mode="normalize")
    assert norm == 0 and (not applied)
    assert all((torch.equal(a, b) for (a, b) in zip(before, net.parameters())))


def test_dataset_screen_NAMES_the_slice_not_the_convention():
    from scripts.dataset_screen import slice_label

    got = {
        slice_label(p)
        for p in (
            "/home/x/_cand/fmow_country/oodslice",
            "/home/x/_cand/isic_src/oodslice/",
            "data/iwildcam/oodslice",
            "data/dermmnist/slice_1",
        )
    }
    assert len(got) == 4, "four different slices collapsed to %d label(s): %s" % (
        len(got),
        got,
    )
    assert slice_label("data/iwildcam/oodslice") == "iwildcam/oodslice"
    assert slice_label("data/dermmnist/slice_1") == "dermmnist/slice_1"
    assert slice_label("data/tissuemnist") == "tissuemnist"


def _cap_df():
    import pandas as pd

    return pd.DataFrame(
        {"label": [2] * 100 + [7] * 200 + [0] * 50, "g": [0] * 175 + [1] * 175}
    )


def test_a_scalar_cap_fraction_is_bit_identical_to_the_historical_behaviour():
    from src.training.constraints import compute_global_constraints

    g = compute_global_constraints(
        _cap_df(), "label", 0.8, constrained_class=[2, 7], num_classes=8
    )
    assert g[2] == 80 and g[7] == 160


def test_a_per_class_cap_fraction_is_read_positionally():
    from src.training.constraints import (
        compute_global_constraints,
        compute_local_constraints,
    )

    g = compute_global_constraints(
        _cap_df(), "label", [0.8, 1.0], constrained_class=[2, 7], num_classes=8
    )
    assert g[2] == 80 and g[7] == 200, (
        "L80-100 on classes [2,7] must cap class 2 at 80%% and class 7 at 100%%, got %s"
        % [g[2], g[7]]
    )
    r = compute_global_constraints(
        _cap_df(), "label", [1.0, 0.8], constrained_class=[2, 7], num_classes=8
    )
    assert r[2] == 100 and r[7] == 160
    loc = compute_local_constraints(
        _cap_df(), "label", [0.8, 1.0], "g", constrained_class=[2, 7], num_classes=8
    )
    assert loc[0][2] == 80 and loc[0][7] == 75
    assert loc[1][2] == 0 and loc[1][7] == 125


def test_NEGATIVE_CONTROL_a_mismatched_cap_list_raises_rather_than_recycling():
    import pytest as _pytest
    from src.training.constraints import compute_global_constraints

    for bad in ([0.8], [0.8, 0.9, 1.0]):
        with _pytest.raises(ValueError):
            compute_global_constraints(
                _cap_df(), "label", bad, constrained_class=[2, 7], num_classes=8
            )


def test_the_cap_tag_parses_both_the_scalar_and_the_per_class_form():
    from configs.gen_campaign import cap_pair

    assert cap_pair("L30_G50") == [0.3, 0.5]
    assert cap_pair("L90_G95") == [0.9, 0.95]
    assert cap_pair("L80-100_G95") == [[0.8, 1.0], 0.95]


def test_a_cap_above_100_percent_is_legal_and_still_binds():
    from src.training.constraints import compute_global_constraints

    g = compute_global_constraints(
        _cap_df(), "label", [1.0, 1.2], constrained_class=[2, 7], num_classes=8
    )
    assert g[2] == 100 and g[7] == 240
