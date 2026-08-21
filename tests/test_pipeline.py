"""Regression tests for the invariants this project has broken before.

Every test here corresponds to a defect that actually shipped. The point is not
coverage -- it is that the specific ways this pipeline has produced wrong
numbers are now mechanically checked.

    python -m pytest tests -q

Runs in a few seconds on CPU and needs no dataset.
"""
import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.gen_campaign import (build_hyperparams, cap_pair,      # noqa: E402
                                  compute_base_model_id, load_protocol)
from src.losses.transductive_loss import MulticlassTransductiveLoss  # noqa: E402
from src.methodologies.heuristic.train import (                      # noqa: E402
    _build_hierarchy, apply_allocation_heuristic, verify_allocation)
from src.training.constraints import (compute_global_constraints,    # noqa: E402
                                      compute_local_constraints)
from src.utils.constants import UNLIMITED                            # noqa: E402
from scripts.full_panel import equalize_multi
from src.experiments.runner import TRAIN_FNS
from src.losses.transductive_loss import margin_window
from src.methodologies.select.train import selective_loss
from src.models import get_model
from src.training.constraint_step import finish_constraint_step
from src.utils.data_loader import _load_imagery_data as load_data
from src.utils.posthoc_adjustment import targeted_correction
import ast
import pandas as pd
import pathlib
import re
import torch.nn as nn
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- the loss --

def _reference_penalty(soft, K, rho=0.5):
    """The penalty as it stood before the K=0 fix. For K >= 1 the current code
    must match this exactly -- the fix must not move any existing result."""
    E = F.relu(soft - K)
    e = E / (K + 1e-8)
    return E / (E + K + 1e-8) + rho * (e ** 2) / (1 + e ** 2 + 1e-8)


def _loss(global_constraints, local_constraints=None, rho=0.5):
    return MulticlassTransductiveLoss(
        global_constraints=global_constraints,
        local_constraints=local_constraints or {},
        num_classes=len(global_constraints), initial_rho=rho)


@pytest.mark.parametrize("K", [1, 2, 5, 20, 67, 250])
@pytest.mark.parametrize("soft", [0.0, 0.5, 7.5, 20.0, 100.0, 1e4])
def test_penalty_unchanged_for_positive_K(K, soft):
    """max(K,1) is the identity for K >= 1: value AND gradient."""
    crit = _loss([K, UNLIMITED])
    a = torch.tensor(soft, requires_grad=True)
    b = torch.tensor(soft, requires_grad=True)
    crit._penalty(a, K).backward()
    _reference_penalty(b, K).backward()
    assert torch.allclose(crit._penalty(torch.tensor(soft), K),
                          _reference_penalty(torch.tensor(soft), K), atol=0, rtol=0)
    assert a.grad.item() == b.grad.item()


def test_K0_constraint_has_a_gradient():
    """A group with no true instance of the capped class gets K=0 legitimately.
    It used to sit pinned at the penalty's own bound: a nonzero CONSTANT with
    exactly zero gradient, permanently unsatisfiable, holding the ratchet gate
    open for every other constraint."""
    crit = _loss([0, UNLIMITED])
    soft = torch.tensor(12.0, requires_grad=True)
    crit._penalty(soft, 0).backward()
    assert soft.grad.item() > 1e-6, "K=0 must push the count down"
    old = torch.tensor(12.0, requires_grad=True)
    _reference_penalty(old, 0).backward()
    assert old.grad.item() == 0.0, "the old form really was inert (guards the test)"


def test_penalty_is_zero_at_and_below_the_budget():
    crit = _loss([10, UNLIMITED])
    for soft in (0.0, 5.0, 9.99, 10.0):
        assert crit._penalty(torch.tensor(soft), 10).item() == 0.0


def test_empty_constraints_stay_connected_to_the_graph():
    """An all-UNLIMITED scope must still return an autograd-connected zero on
    the right device, or .backward() raises on a run with no capped class."""
    crit = _loss([UNLIMITED, UNLIMITED])
    counts = torch.zeros(2, requires_grad=True)
    total = crit.compute_global_from_counts(counts)
    assert total.item() == 0.0
    assert total.requires_grad, "an all-UNLIMITED scope returned a detached zero"
    total.backward()                       # must not raise
    assert counts.grad is not None

    # An empty local scope has no count tensor to hang the graph on, so its
    # zero is a constant -- but it must be on the MODULE's device, or adding it
    # to a CUDA global term raises. It used to be a bare CPU tensor(0.0).
    local = crit.compute_local_from_counts({})
    assert local.item() == 0.0
    assert local.device == crit.global_constraints.device
    # and the sum must stay differentiable as long as either scope is live
    assert (crit.compute_global_from_counts(counts) + local).requires_grad


# ------------------------------------------------------------- the budgets --

def _frame(labels, groups):
    return pd.DataFrame({"label": np.asarray(labels), "grp": np.asarray(groups)})


def test_local_and_global_percentages_are_independent():
    df = _frame([1] * 100 + [0] * 100, [0] * 50 + [1] * 50 + [0] * 50 + [1] * 50)
    local_pct, global_pct = cap_pair("L50_G30")
    assert (local_pct, global_pct) == (0.5, 0.3)
    g = compute_global_constraints(df, "label", global_pct,
                                   constrained_class=1, num_classes=2)
    loc = compute_local_constraints(df, "label", local_pct, "grp",
                                    constrained_class=1, num_classes=2)
    assert g[1] == 30                                  # 30% of 100
    assert sorted(v[1] for v in loc.values()) == [25, 25]   # 50% of 50, twice


def test_K_zero_from_a_nonzero_count_is_refused():
    """A percentage that rounds a real class to zero would vanish silently."""
    df = _frame([1] + [0] * 99, [0] * 100)
    with pytest.raises(ValueError, match="K=0"):
        compute_global_constraints(df, "label", 0.3,
                                   constrained_class=1, num_classes=2)


def test_K_zero_from_an_absent_class_is_allowed():
    """count == 0 is a real, tightest-possible budget, not a config error."""
    df = _frame([0] * 50 + [1] * 50, [0] * 50 + [1] * 50)
    loc = compute_local_constraints(df, "label", 0.5, "grp",
                                    constrained_class=1, num_classes=2)
    assert loc[0][1] == 0 and loc[1][1] == 25


def test_several_capped_classes_get_independent_budgets():
    df = _frame([0] * 40 + [1] * 60 + [2] * 100, list(range(2)) * 100)
    g = compute_global_constraints(df, "label", 0.5,
                                   constrained_class=[0, 2], num_classes=3)
    assert g[0] == 20 and g[2] == 50 and g[1] == UNLIMITED


# ----------------------------------------------------------- the allocator --

def _probs(n, k, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, k))
    return np.exp(z) / np.exp(z).sum(1, keepdims=True)


@pytest.mark.parametrize("seed", range(6))
def test_single_capped_class_is_exactly_top_K(seed):
    n, k, K = 200, 4, 30
    p, g = _probs(n, k, seed), np.zeros(200, dtype=int)
    gcon = [UNLIMITED] * k
    gcon[2] = K
    lcon = {0: [UNLIMITED] * k}
    y, _ = apply_allocation_heuristic(
        p, g, _build_hierarchy(k, gcon, [2]), gcon, lcon, k)
    chosen = set(np.where(y == 2)[0])
    assert chosen == set(np.argsort(p[:, 2])[::-1][:K])


@pytest.mark.parametrize("seed", range(6))
def test_several_capped_classes_are_not_starved(seed):
    """Filling classes one at a time let each take its full budget from the
    whole pool, so the last capped class got nothing: [40, 40, 40, 0]."""
    n, k = 120, 4
    p, g = _probs(n, k, seed), np.zeros(120, dtype=int)
    gcon, lcon = [40] * k, {0: [40] * k}
    y, _ = apply_allocation_heuristic(
        p, g, _build_hierarchy(k, gcon, list(range(k))), gcon, lcon, k)
    counts = np.bincount(y, minlength=k)
    assert counts.min() > 0, "a capped class was starved: %s" % counts
    assert not verify_allocation(y, g, gcon, lcon, k)


def test_a_local_cap_binds_without_a_global_cap():
    """The leftover pass gated the local check on the GLOBAL cap, so a class
    capped only locally was never blocked locally."""
    n, k = 200, 4
    p = _probs(n, k, 11)
    g = np.arange(n) % 2
    gcon = [UNLIMITED] * k
    lcon = {0: [UNLIMITED, 5, UNLIMITED, UNLIMITED],
            1: [UNLIMITED, 5, UNLIMITED, UNLIMITED]}
    y, _ = apply_allocation_heuristic(
        p, g, _build_hierarchy(k, gcon, [1]), gcon, lcon, k)
    for grp in (0, 1):
        assert int((y[g == grp] == 1).sum()) <= 5
    assert not verify_allocation(y, g, gcon, lcon, k)


def test_infeasible_instance_does_not_dump_everything_into_class_zero():
    """argmax of an all -1 vector returns 0, so every impossible item was
    assigned class 0 far past its cap, silently."""
    n, k = 100, 4
    p, g = _probs(n, k, 7), np.zeros(100, dtype=int)
    gcon, lcon = [10] * k, {0: [10] * k}
    y, _ = apply_allocation_heuristic(
        p, g, _build_hierarchy(k, gcon, list(range(k))), gcon, lcon, k)
    counts = np.bincount(y, minlength=k)
    assert counts.max() < n / 2, "collapsed onto one class: %s" % counts


# ------------------------------------------------------- the generator gate --

def _gen(tmp, *extra):
    cmd = [sys.executable, "-m", "configs.gen_campaign", "--root", str(tmp),
           "--datasets", "dermmnist", "--arms", "tralo"] + list(extra)
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)


def test_generator_refuses_a_single_cap_level(tmp_path):
    r = _gen(tmp_path, "--caps", "L30_G30")
    assert r.returncode == 1 and "two cap levels" in r.stdout + r.stderr


def test_generator_refuses_an_out_of_range_capped_class(tmp_path):
    r = _gen(tmp_path, "--caps", "L30_G30", "L50_G30", "--constrained-class", "9")
    assert r.returncode == 1 and "out of range" in r.stdout + r.stderr


def test_generator_refuses_a_repeated_capped_class(tmp_path):
    r = _gen(tmp_path, "--caps", "L30_G30", "L50_G30",
             "--constrained-class", "4", "4")
    assert r.returncode == 1 and "repeats a class" in r.stdout + r.stderr


def test_generator_refuses_to_mix_capped_classes_in_one_root(tmp_path):
    """A completed run is never reset, so a second pass with a different capped
    class would leave both in one cell and the scorer would pool them."""
    assert _gen(tmp_path, "--caps", "L30_G30", "L50_G30").returncode == 0
    cfg_path = next(iter(sorted(tmp_path.rglob("config.json"))))
    cfg = json.loads(cfg_path.read_text())
    cfg["status"] = "completed"
    cfg_path.write_text(json.dumps(cfg))
    r = _gen(tmp_path, "--caps", "L30_G30", "L50_G30",
             "--constrained-class", "4", "5")
    assert r.returncode == 1 and "already holds a run" in r.stdout + r.stderr


def test_mandatory_clippers_are_always_added(tmp_path):
    assert _gen(tmp_path, "--caps", "L30_G30", "L50_G30").returncode == 0
    arms = {p.parts[-3] for p in tmp_path.rglob("config.json")}
    assert {"clip", "focal_clip"} <= arms


# -------------------------------------------------- the warm-up cache identity

def _bid(P, arm, seed=1, **over):
    dc = dict(P["datasets"]["dermmnist"])
    hp = build_hyperparams(P, P["arms"][arm], seed)
    hp.update(over)
    return compute_base_model_id(P, "MobileNetV3", hp, "dermmnist", dc)


def test_arms_differing_only_in_the_allocator_share_a_warm_up():
    P = load_protocol()
    assert _bid(P, "clip") == _bid(P, "lp")
    assert _bid(P, "focal_clip") == _bid(P, "focal_lp")
    assert _bid(P, "tralo") == _bid(P, "fioretto") == _bid(P, "alm")


def test_a_different_warm_up_objective_does_not_share_a_model():
    """clip and focal_clip once hashed identically, so focal_clip silently
    loaded clip's model and became a second clip."""
    P = load_protocol()
    assert _bid(P, "clip") != _bid(P, "focal_clip")
    assert _bid(P, "clip") != _bid(P, "cb_lp") != _bid(P, "la_lp")


@pytest.mark.parametrize("key,value", [
    ("lr", 0.5), ("dropout", 0.9), ("batch_size", 7), ("warmup_epochs", 3),
    ("pretrained", False), ("class_weighted_ce", True), ("seed", 99),
])
def test_every_warm_up_key_moves_the_hash(key, value):
    P = load_protocol()
    assert _bid(P, "tralo") != _bid(P, "tralo", **{key: value})


def test_constraint_phase_keys_do_not_move_the_hash():
    """They do not change what the warm-up optimizes, so the cache must be
    shared -- that sharing is the whole point."""
    P = load_protocol()
    base = _bid(P, "tralo")
    for key, value in (("constraint_epochs", 5), ("lambda_step", 0.9),
                       ("initial_rho", 9.0), ("enable_checkpoint_restore", True)):
        assert _bid(P, "tralo", **{key: value}) == base


# ------------------------------------------------------------- the lifecycle --

def test_a_diverged_result_is_not_recorded_as_completed(tmp_path):
    """An all-NaN model still scores like a degenerate but healthy classifier;
    `completed` means the dispatcher never revisits it."""
    from src.pipeline.io import save_results_to_config
    cfg = {"hyperparams": {}, "status": "pending"}
    save_results_to_config(cfg, tmp_path, {"accuracy": float("nan"), "f1": 0.3})
    assert cfg["status"] == "diverged" and "accuracy" in cfg["diverged_keys"][0]
    cfg2 = {"hyperparams": {}, "status": "pending"}
    save_results_to_config(cfg2, tmp_path, {"accuracy": 0.8, "f1": 0.3})
    assert cfg2["status"] == "completed"


def test_rerunning_does_not_turn_the_old_header_into_a_data_row(tmp_path):
    """A crashed run is reset to pending and re-dispatched into the same dir;
    rows are appended, so df["Epoch"].max() returned the STRING 'Epoch'."""
    from src.training.logging import log_progress_to_csv, write_csv_header
    p = tmp_path / "training_log.csv"
    for run in range(2):
        write_csv_header(str(p), num_classes=2)
        for epoch in range(3):
            log_progress_to_csv(str(p), epoch, 0.5, 0.9, num_classes=2)
    df = pd.read_csv(p)
    assert pd.api.types.is_numeric_dtype(df["Epoch"])
    assert df["Epoch"].max() == 3


def test_the_csv_reports_the_real_local_lambda_column():
    """Lambda_Local was passed as a literal 0.0 and read 0.0000 forever."""
    from src.training.logging import build_csv_header
    header = build_csv_header(num_classes=2)
    assert "Lambda_Local" in header
    assert "Grad_Norm" in header and "L_KL" not in header


# --------------------------------------------------------- the backbone heads

@pytest.mark.parametrize("name", ["MobileNetV3", "MobileNetV2",
                                  "RegNetY400MF", "ViTB16"])
def test_backbones_keep_their_pretrained_weights(name):
    """Every backbone must replace ONLY its final layer. MobileNetV3 -- the
    headline backbone -- used to rebuild its whole classifier, discarding the
    pretrained 960->1280 projection. That projection is trained during warm-up
    only, and trained arms get ONE warm-up epoch against the post-hoc arms'
    thirty, so it biased the headline comparison on the headline backbone."""
    torch.manual_seed(0)
    a = get_model(name, input_dim=None, n_classes=7, dropout=0.3, pretrained=True)
    torch.manual_seed(999)
    b = get_model(name, input_dim=None, n_classes=7, dropout=0.3, pretrained=True)
    named_a, named_b = dict(a.named_parameters()), dict(b.named_parameters())
    differ = [k for k in named_a
              if not torch.equal(named_a[k], named_b[k])]
    # only the randomly-initialised final layer may differ between two seeds
    assert len(differ) <= 2, (
        "%s re-initialises %d tensors from random, not just the final layer: %s"
        % (name, len(differ), differ[:6]))


def test_mobilenetv3_has_exactly_one_dropout_in_its_head():
    """The whole reason the classifier was rebuilt was to avoid a double
    dropout. Reusing the pretrained head must not reintroduce one."""
    m = get_model("MobileNetV3", input_dim=None, n_classes=7,
                  dropout=0.3, pretrained=False)
    head = m.backbone.classifier
    drops = [l for l in head if isinstance(l, nn.Dropout)]
    assert len(drops) == 1 and drops[0].p == 0.3


# ------------------------------------------------------ the failure lifecycle

def test_a_repeatedly_failing_run_stops_being_re_dispatched(tmp_path):
    """A config that fails deterministically used to reset to `pending` and be
    picked up again by every subsequent dispatch, forever, with nothing on disk
    saying why."""
    from src.utils.filesystem_manager import (MAX_FAILURES,
                                              get_experiments_by_status,
                                              save_config_to_path,
                                              update_experiment_status)
    exp = tmp_path / "MobileNetV3" / "derm" / "L30_G30" / "tralo" / "seed_1"
    exp.mkdir(parents=True)
    save_config_to_path({"status": "pending", "hyperparams": {"seed": 1},
                         "arm": "tralo"}, exp)
    for i in range(1, MAX_FAILURES + 1):
        update_experiment_status(str(exp), "pending", count_failure=True)
        cfg = json.loads((exp / "config.json").read_text())
        assert cfg["failures"] == i
    assert cfg["status"] == "failed"
    buckets = get_experiments_by_status(str(tmp_path))
    assert not buckets["pending"] and len(buckets["blocked"]) == 1


def test_a_diverged_run_is_not_re_dispatched(tmp_path):
    from src.utils.filesystem_manager import (get_experiments_by_status,
                                              save_config_to_path)
    exp = tmp_path / "M" / "d" / "L30_G30" / "tralo" / "seed_1"
    exp.mkdir(parents=True)
    save_config_to_path({"status": "diverged", "hyperparams": {"seed": 1},
                         "arm": "tralo"}, exp)
    buckets = get_experiments_by_status(str(tmp_path))
    assert not buckets["pending"] and len(buckets["blocked"]) == 1


def test_an_interrupted_run_still_resets_to_pending(tmp_path):
    """Load-bearing: this is what makes overnight re-dispatch idempotent.
    `running` must NOT be terminal."""
    from src.utils.filesystem_manager import (get_experiments_by_status,
                                              save_config_to_path)
    exp = tmp_path / "M" / "d" / "L30_G30" / "tralo" / "seed_1"
    exp.mkdir(parents=True)
    save_config_to_path({"status": "running", "hyperparams": {"seed": 1},
                         "arm": "tralo"}, exp)
    buckets = get_experiments_by_status(str(tmp_path))
    assert len(buckets["pending"]) == 1 and not buckets["blocked"]


# --------------------------------------------- protocol values are not optional

@pytest.mark.parametrize("key", ["lr_constraint", "constraint_epochs",
                                 "stable_count_threshold"])
def test_a_missing_protocol_value_raises_instead_of_using_the_trap(key, tmp_path):
    """These inline defaults WERE the retracted values: lr_constraint 1e-5
    against the protocol's 1e-4 (an unequal lr_constraint fabricated a -16.7 pp
    finding), constraint_epochs 150 against 29, and stable_count_threshold 5
    against 31 -- low enough that the early stop would actually fire, so an arm
    would stop training partway and still be scored at 'equal compute'."""
    import scripts.smoke_arms as sm
    P = load_protocol()
    inputs, _, _ = sm.make_inputs(P, "tralo", str(tmp_path))
    inputs.hyperparams.pop(key, None)
    with pytest.raises(KeyError, match=key):
        TRAIN_FNS["tralo"](inputs)


def test_deleted_danits_helpers_are_really_gone():
    """They were reachable only through __init__ re-exports, which is why the
    AST reachability pass reported them as live. The manuscript claims two
    post-hoc clippers -- a greedy threshold (that is the `clip` arm) and LP-LG
    with 'an identity misclassification cost rather than the general cost
    matrix' -- so none of the three was in scope."""
    import src.methodologies.danits_lp as d
    assert set(d.__all__) == {"solve_lp_assignment", "LPResult"}
    for gone in ("solve_greedy_assignment", "build_psi_phi_from_percentages",
                 "build_priority_cost_matrix", "describe_cost_matrix"):
        assert not hasattr(d, gone), "%s came back" % gone


# --------------------------------------------------- the gates must be able to fail

def test_parity_catches_two_arms_sharing_one_warm_up_with_different_objectives(tmp_path):
    """Occurrence 5 of the inert-flag failure: clip and focal_clip hashed
    identically, so focal_clip loaded clip's model and silently became a second
    clip. This gate used to print the sharing groups and ask a human to look."""
    r = subprocess.run(
        [sys.executable, "-m", "configs.gen_campaign", "--root", str(tmp_path),
         "--datasets", "dermmnist", "--models", "MobileNetV3",
         "--caps", "L30_G30", "L50_G30", "--arms", "clip", "focal_clip"],
        cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0
    clip_id = next(json.loads(p.read_text())["base_model_id"]
                   for p in sorted(tmp_path.rglob("config.json"))
                   if json.loads(p.read_text())["arm"] == "clip")
    for p in sorted(tmp_path.rglob("config.json")):
        cfg = json.loads(p.read_text())
        if cfg["arm"] == "focal_clip":
            cfg["base_model_id"] = clip_id
            p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, "-m", "scripts.check_parity", str(tmp_path)],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 1
    # gate 4 now names the offending key rather than saying "objectives",
    # because it checks all twelve warm-up-identity keys, not just this one
    assert "DIFFERENT warmup_loss" in r.stdout


def test_verify_caps_fails_when_it_cannot_read_a_slice(tmp_path):
    """It printed 'CAP CHECK OK -- every cap tag produces a real integer budget
    on every dataset' having opened no file at all."""
    r = subprocess.run(
        [sys.executable, "-m", "scripts.verify_caps", "--datasets", "dermmnist"],
        cwd=str(tmp_path), capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": REPO})
    assert r.returncode == 1, "a gate that cannot fail is not a gate"


def test_the_scorer_pairs_on_the_capped_class(tmp_path):
    """pivot_table's default aggfunc averaged two capped-class settings into one
    pair, so a +0.40 cell and a -0.40 cell collapsed to an exact tie while the
    header still printed two cells."""
    import scripts.full_panel as fp
    src = io.io.open(os.path.join(REPO, "scripts", "full_panel.py"),
                      encoding="utf-8").read() if False else open(
        os.path.join(REPO, "scripts", "full_panel.py"), encoding="utf-8").read()
    key = src.split("key = [")[1].split("]")[0]
    assert '"capped"' in key, "the capped class is not in the pairing key"
    with pytest.raises(ValueError, match="pairing key is missing"):
        fp._one(pd.Series([0.4, -0.4]))
    assert fp._one(pd.Series([0.4])) == 0.4


# ------------------------------------------------------------- the verdict rule

def _panel_verdict(tmp_path, n_better_cells, n_tied_cells, metric="AP"):
    """Build a synthetic campaign with a KNOWN answer and read the verdict."""
    cells = [("dermmnist", "MobileNetV3", "L30_G30"),
             ("dermmnist", "MobileNetV3", "L50_G30"),
             ("octmnist", "MobileNetV3", "L30_G30"),
             ("octmnist", "MobileNetV3", "L50_G30"),
             ("tissuemnist", "MobileNetV3", "L30_G30"),
             ("tissuemnist", "MobileNetV3", "L50_G30"),
             ("dermmnist", "MobileNetV2", "L30_G30"),
             ("dermmnist", "MobileNetV2", "L50_G30")]
    N, K = 200, 4
    for i, (ds, model, cap) in enumerate(cells[:n_better_cells + n_tied_cells]):
        for arm, boost in (("clip", 0.0),
                           ("tralo", 1.5 if i < n_better_cells else 0.0)):
            for seed in (1, 2, 3, 4):
                rng = np.random.default_rng(1000 * i + seed)
                y = rng.integers(0, K, size=N)
                z = rng.normal(size=(N, K))
                z[np.arange(N), y] += boost
                P = np.exp(z) / np.exp(z).sum(1, keepdims=True)
                d = tmp_path / model / ds / cap / arm / ("seed_%d" % seed)
                d.mkdir(parents=True, exist_ok=True)
                cols = {"True_Label": y, "Predicted_Label": P.argmax(1),
                        "Group_ID": rng.integers(0, 3, size=N)}
                for c in range(K):
                    cols["Prob_Class_%d" % c] = P[:, c]
                for f in ("final_predictions_raw.csv", "final_predictions.csv"):
                    pd.DataFrame(cols).to_csv(d / f, index=False)
                (d / "config.json").write_text(json.dumps(
                    {"arm": arm, "methodology": "x", "model_name": model,
                     "dataset_mode": ds, "constraint_tag": cap,
                     "constraint": [0.5, 0.3], "status": "completed",
                     "dataset_config": {"constrained_class": 1, "num_classes": K},
                     "hyperparams": {"seed": seed}}))
    r = subprocess.run([sys.executable, "-m", "scripts.full_panel",
                        "--campaign", str(tmp_path), "--control", "clip"],
                       cwd=REPO, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.strip().startswith(metric + " "):
            return line
    return r.stdout + r.stderr


def test_a_win_with_majority_ties_is_not_reported_as_a_loss(tmp_path):
    """stats.wilcoxon DISCARDS zero differences, but the majority test counted
    them in its denominator. With 3 cells strictly better and 5 bit-identical
    the old rule printed '*** LOSS' on a +0.19 delta at p=0.0022. The bug is
    asymmetric -- it can only turn a win into a loss -- and partial ties are the
    normal state here."""
    line = _panel_verdict(tmp_path, n_better_cells=3, n_tied_cells=5)
    assert "LOSS" not in line, line
    assert "3/0" in line, line          # 3 better, 0 worse, rest tied


def test_a_clean_win_is_still_called_a_win(tmp_path):
    line = _panel_verdict(tmp_path, n_better_cells=6, n_tied_cells=2)
    assert "*** WIN" in line, line


def test_a_bit_identical_arm_is_a_dead_flag_not_a_direction(tmp_path):
    """scipy returns p=1.0 for n<=12 and NaN for n>=16 on all-zero differences,
    so an inert arm printed 'lean loss' x13. Identical output means the
    treatment did nothing -- the project's most frequent failure, five
    occurrences -- and must never render as a direction."""
    line = _panel_verdict(tmp_path, n_better_cells=0, n_tied_cells=8)
    assert "DEAD FLAG" in line, line
    assert "loss" not in line.lower(), line


def test_unattainable_significance_is_declared_not_called(tmp_path):
    """At n=4 the exact two-sided Wilcoxon floor is 0.125, so 'p=0.125, lean
    loss' was the arm being un-callable, not a settled tie."""
    line = _panel_verdict(tmp_path, n_better_cells=4, n_tied_cells=0)
    assert "NOT CALLABLE" in line, line


def test_the_scorer_skips_runs_that_are_not_completed(tmp_path):
    """It ignored `status` entirely, and regenerating a campaign overwrites a
    non-completed config in place while leaving the OLD predictions on disk --
    so the previous code's predictions get scored as the new code's result."""
    _panel_verdict(tmp_path, 6, 2)
    for p in sorted(tmp_path.rglob("config.json")):
        cfg = json.loads(p.read_text())
        if cfg["arm"] == "tralo":
            cfg["status"] = "diverged"
            p.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, "-m", "scripts.full_panel",
                        "--campaign", str(tmp_path), "--control", "clip"],
                       cwd=REPO, capture_output=True, text=True)
    assert "skipped" in r.stdout and "diverged" in r.stdout


def test_the_audit_sees_keys_read_through_the_required_helper():
    """_required(hp, "lr_constraint", float) passes the config as an argument.
    The walker understood subscripts and dict methods only, so every key read
    that way was invisible in BOTH directions -- emitting one looked
    HALLUCINATED, omitting one produced no SILENT flag."""
    from scripts.audit_config import per_methodology_reads
    reads = per_methodology_reads()
    for meth in ("tralo", "fioretto_ldf", "hounie_rcl", "fioretto_alm"):
        assert "lr_constraint" in reads[meth], meth
        assert "stable_count_threshold" in reads[meth], meth
        assert "enable_checkpoint_restore" in reads[meth], meth


# ---------------------------------------------------------------------------
# A read must not mutate the model's train/eval mode.
#
# compute_prediction_statistics is called immediately after model.eval() at the
# end of every trained run and used to return the model in train() mode. Every
# current caller re-asserts eval(), so it was harmless -- but the duals' own
# comments say eval() during a test-set pass is what stops BN running stats
# updating from test data, "a data-leakage source that flips a few borderline
# samples and corrupts the lambda update". A read that silently arms that is
# the shape of defect this project keeps finding.
# ---------------------------------------------------------------------------
def test_metrics_helpers_restore_the_callers_mode():
    from src.training import metrics

    model = nn.Sequential(nn.Linear(4, 3), nn.Dropout(0.5))
    X = torch.randn(20, 4)
    groups = torch.zeros(20, dtype=torch.long)
    loader = [(torch.randn(8, 4), torch.zeros(8, dtype=torch.long))]

    for mode in (True, False):
        model.train(mode)
        metrics.compute_prediction_statistics(model, X, groups, num_classes=3)
        assert model.training is mode, (
            "compute_prediction_statistics left the model in %s mode after "
            "being called in %s mode" % (model.training, mode))

        model.train(mode)
        metrics.get_predictions_with_probabilities(model, X)
        assert model.training is mode

        model.train(mode)
        metrics.compute_train_accuracy(model, loader, torch.device("cpu"))
        assert model.training is mode


# ---------------------------------------------------------------------------
# A misconfigured or swapped slice must FAIL, not produce a plausible number.
#
# Before these guards: four independent np.load calls with no length comparison
# (train died late inside TensorDataset with an unlabelled AssertionError, test
# never raised at all -- the chunked loops key off len(X_test) and would score
# fewer items than the labels describe); and num_classes / constrained_class
# were never checked against the labels, so pointing data_dir at the wrong
# dataset gave K=0 on an absent class, a log warning, and a complete run.
# ---------------------------------------------------------------------------
def _write_slice(d, n_train=12, n_test=8, n_classes=4, capped=2,
                 truncate_test_labels=False, drop_capped=False):
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0)
    ytr = np.array([i % n_classes for i in range(n_train)])
    yte = np.array([i % n_classes for i in range(n_test)])
    if drop_capped:                       # a slice that lacks the capped class
        ytr = np.where(ytr == capped, (capped + 1) % n_classes, ytr)
        yte = np.where(yte == capped, (capped + 1) % n_classes, yte)
    np.save(os.path.join(d, "train_images.npy"),
            rng.random((n_train, 3, 8, 8)).astype("float32"))
    np.save(os.path.join(d, "train_labels.npy"), ytr)
    np.save(os.path.join(d, "test_images.npy"),
            rng.random((n_test, 3, 8, 8)).astype("float32"))
    np.save(os.path.join(d, "test_labels.npy"),
            yte[:-1] if truncate_test_labels else yte)
    pd.DataFrame({"label": yte, "grp": [i % 2 for i in range(n_test)]}).to_csv(
        os.path.join(d, "test_meta.csv"), index=False)


def _cfg(d, n_classes=4, capped=2):
    return {"dataset_mode": "dermmnist", "dataset_config": {
        "data_dir": d, "num_classes": n_classes, "constrained_class": capped,
        "group_column": "grp", "target_column": "label"},
        "constraint": [0.5, 0.5]}


def test_loader_refuses_mismatched_image_and_label_counts(tmp_path):
    d = str(tmp_path / "bad_len")
    _write_slice(d, truncate_test_labels=True)
    with pytest.raises(ValueError, match="test_images.npy has 8 rows"):
        load_data(_cfg(d))


def test_loader_refuses_a_slice_whose_labels_exceed_num_classes(tmp_path):
    d = str(tmp_path / "wrong_ds")
    _write_slice(d, n_classes=6)                 # 6 real classes...
    with pytest.raises(ValueError, match="num_classes is 4"):
        load_data(_cfg(d, n_classes=4))          # ...config says 4


def test_loader_refuses_a_slice_missing_the_capped_class(tmp_path):
    d = str(tmp_path / "no_capped")
    _write_slice(d, drop_capped=True)
    with pytest.raises(ValueError, match="does not occur in this slice"):
        load_data(_cfg(d))


def test_loader_accepts_a_well_formed_slice(tmp_path):
    """The guards must not fire on good data."""
    d = str(tmp_path / "good")
    _write_slice(d)
    out = load_data(_cfg(d))
    assert out is not None


# ---------------------------------------------------------------------------
# Pin the penalty's gradient SHAPE.
#
# Above rho ~ 1 the gradient is non-monotone in the violation: near-zero at the
# boundary, peaking at u = 1/sqrt(3) = 57.7% overshoot, then decaying toward
# zero. At rho=100 a constraint violated 8x over budget gets ~167x less pull
# than one violated 58% over. This is the PUBLISHED formula (manuscript Eq. 4),
# not a bug, so the test pins it rather than forbidding it -- if someone changes
# the shape, that has to be a deliberate act with the paper updated, and this
# test is where they find out.
# ---------------------------------------------------------------------------
def test_penalty_gradient_is_non_monotone_above_rho_one():

    K = 67.0

    def grad_at(rho, violation):
        soft = torch.tensor(K + violation, requires_grad=True, dtype=torch.double)
        Kt = torch.tensor(K, dtype=torch.double)
        E = torch.relu(soft - Kt)
        S = Kt
        eps = 1e-8
        e = E / (S + eps)
        (E / (E + S + eps) + rho * (e ** 2) / (1 + e ** 2 + eps)).backward()
        return float(soft.grad)

    # rho starts at 0.5 and rho_step is derived as (100 - 0.5)/29 = 3.43,
    # so rho is 3.93 after ONE constraint epoch of twenty-nine.
    assert abs((100.0 - 0.5) / 29 - 3.431) < 0.001

    # at the initial rho the gradient is monotonically decreasing -- correct
    g0 = [grad_at(0.5, f * K) for f in (0.001, 0.3, 1.0, 8.0)]
    assert g0 == sorted(g0, reverse=True), "rho=0.5 should still be monotone"

    # one epoch in, it is not
    edge, peak, deep = (grad_at(3.93, 0.001 * K), grad_at(3.93, 0.577 * K),
                        grad_at(3.93, 8.0 * K))
    assert peak > 2.5 * edge, "expected a hump above the boundary value"
    assert peak > 100 * deep, "expected the deep violation to be starved"

    # u = 1/sqrt(3) is the peak of the QUADRATIC TERM alone, so it is the
    # rho -> infinity limit, not a general truth. It holds at rho=100...
    import math
    at_analytic = grad_at(100.0, (1 / math.sqrt(3)) * K)
    for f in (0.2, 0.4, 0.8, 1.5):
        assert grad_at(100.0, f * K) <= at_analytic + 1e-12, (
            "peak should be at u = 1/sqrt(3), not at %.2f" % f)

    # ...and NOT at the rho the runs actually operate at. The rational term's
    # own slope is decreasing, which pulls the combined peak left, to u ~ 0.53.
    # An earlier version of this claim said "analytically" with no caveat.
    us = [i / 500 for i in range(1, 5001)]
    gs = [grad_at(3.93, u * K) for u in us]
    peak_u = us[gs.index(max(gs))]
    assert 0.50 < peak_u < 0.56, (
        "at rho=3.93 the peak should sit left of 1/sqrt(3), got u=%.3f" % peak_u)


def test_a_deep_violation_is_starved_by_a_milder_one_sharing_the_clip():
    """The terms compete: `_sum` adds them, then ONE clip normalizes the total.

    So the relative weight of two capped scopes is set entirely by the penalty
    shape, and the shape hands almost everything to whichever scope sits nearer
    its own peak. dermmnist has 3 groups, so every run already carries 4 terms.
    """

    K, rho, eps = 67.0, 100.0, 1e-8

    def shares(u_a, u_b):
        gs = []
        for u in (u_a, u_b):
            soft = torch.tensor(K * (1 + u), requires_grad=True, dtype=torch.double)
            Kt = torch.tensor(K, dtype=torch.double)
            E = torch.relu(soft - Kt)
            e = E / (Kt + eps)
            (E / (E + Kt + eps) + rho * (e ** 2) / (1 + e ** 2 + eps)).backward()
            gs.append(float(soft.grad))
        tot = sum(g * g for g in gs)
        return [g * g / tot for g in gs]

    mild, deep = shares(0.577, 8.0)
    assert mild > 0.999, "the milder violation should take essentially all of it"
    assert deep < 0.001, "the 8x violation should be starved"
    # symmetric: it is the violation DEPTH, not the position in the sum
    deep2, mild2 = shares(8.0, 0.577)
    assert abs(mild - mild2) < 1e-9 and abs(deep - deep2) < 1e-9
    assert abs(sum(shares(0.577, 0.577))/2 - 0.5) < 1e-9


def test_loader_detects_a_permuted_train_split(tmp_path):
    """A permutation of images vs labels is invisible to a length check.

    This was written up as an undetectable accepted risk on the reasoning that
    no train_meta.csv exists. That was wrong -- all three prep scripts write
    one, with a `label` column, and it is on disk in every slice. The redundant
    signal needed to catch this was there the whole time.
    """
    d = str(tmp_path / "permuted")
    _write_slice(d)
    y = np.load(os.path.join(d, "train_labels.npy"))
    # same length, same multiset, different order -- exactly what a length
    # check cannot see
    pd.DataFrame({"label": y[::-1]}).to_csv(
        os.path.join(d, "train_meta.csv"), index=False)
    with pytest.raises(ValueError, match="not row-aligned"):
        load_data(_cfg(d))


def test_loader_accepts_an_aligned_train_meta(tmp_path):
    d = str(tmp_path / "aligned")
    _write_slice(d)
    y = np.load(os.path.join(d, "train_labels.npy"))
    pd.DataFrame({"label": y}).to_csv(
        os.path.join(d, "train_meta.csv"), index=False)
    assert load_data(_cfg(d)) is not None


def test_the_null_arm_really_delivers_no_constraint():
    """tralo_null is the control that makes "nothing happened" falsifiable.

    Everything else about it matches tralo -- warm-up length, epoch count,
    optimizer restart, transductive passes, allocator. If its lambdas were not
    exactly zero it would be a weak treatment rather than a control, and the
    whole point of the arm would be lost silently.
    """
    proto = yaml.safe_load(open("configs/protocol.yml", encoding="utf-8"))
    blk = proto["blocks"]["tralo_null"]
    for key in ("lambda_global", "lambda_local", "lambda_step"):
        assert blk[key] == 0.0, "%s must be exactly 0.0, got %r" % (key, blk[key])

    # and the arm must otherwise be tralo: same phase, same shared block
    null_arm, real_arm = proto["arms"]["tralo_null"], proto["arms"]["tralo"]
    assert null_arm["phase"] == real_arm["phase"] == "trained"
    assert null_arm["methodology"] == real_arm["methodology"] == "tralo"
    assert "constraint_phase" in null_arm["blocks"]

    # zero lambda must make the summed penalty exactly zero, not merely small:
    # the trainer gates pass 2 on `total_constraint > 0`, so a 1e-30 residue
    # would still run a constraint step.
    from src.losses.transductive_loss import MulticlassTransductiveLoss as L
    total = torch.tensor(0.0)
    for soft, K in ((500.0, 67.0), (67.0, 67.0), (0.0, 67.0)):
        st = torch.tensor(soft)
        Kt = 67.0
        E = torch.relu(st - Kt)
        e = E / (Kt + 1e-8)
        pen = E / (E + Kt + 1e-8) + 0.5 * (e ** 2) / (1 + e ** 2 + 1e-8)
        total = total + 0.0 * pen
    assert float(total) == 0.0
    assert not bool(total > 0), "pass 2 would still run on a nonzero residue"
    assert L is not None


def test_every_trained_arm_reports_reordering():
    """The diagnostic must not reach one arm only.

    It was written inside tralo/train.py and reached only TraLO -- the exact
    shape of the CE-skip asymmetry that produced a 0.22 cc-F1 artifact, and of
    the focal keys that were live in one arm and dead in another. The guard is
    structural: every trainer that runs a constraint phase reaches the SAME two
    functions in the SAME module, and the scorer reads the field.

    Reached DIRECTLY (tralo) or through `src/training/dual_arm.py`, which the
    three duals share. Following the import chain rather than grepping one file
    is what lets the shared tail exist at all -- and it still fails if an arm
    grows a private copy or drops the summary field, because the module that
    calls `reordering_report` is also the module that must write "reordering".
    """

    def _reaches_reordering(src, seen):
        """Does this source call reordering_report and emit the summary key --
        here, or in a src.training module it imports?"""
        if "reordering_report(" in src and '"reordering"' in src:
            return True
        for line in src.splitlines():
            line = line.strip()
            if not line.startswith("from src.training."):
                continue
            mod = line.split()[1]
            path = mod.replace(".", os.sep) + ".py"
            if path in seen or not os.path.exists(path):
                continue
            seen.add(path)
            if _reaches_reordering(io.open(path, encoding="utf-8").read(), seen):
                return True
        return False

    trained = ["tralo", "fioretto_ldf", "hounie_rcl", "fioretto_alm"]
    for m in trained:
        path = os.path.join("src", "methodologies", m, "train.py")
        src = io.open(path, encoding="utf-8").read()
        assert "reordering_report" not in src or "def reordering_report" not in src, (
            "%s must use the shared diagnostic, not a private copy" % m)
        assert _reaches_reordering(src, set()), (
            "%s never reaches reordering_report / never puts it in the summary" % m)

    # it has to survive to disk, and outside config["results"] -- a NaN tau on
    # a constant score column would otherwise mark the run `diverged`
    runner = io.open(os.path.join("src", "experiments", "runner.py"),
                      encoding="utf-8").read()
    assert "config['reordering']" in runner
    results_blk = runner[runner.index("save_results_to_config(config"):]
    assert "reordering" not in results_blk[:results_blk.index("})")]

    # and the scorer must actually read it
    panel = io.open(os.path.join("scripts", "full_panel.py"), encoding="utf-8").read()
    assert 'cfg.get("reordering")' in panel
    assert "_reordering_check(rows)" in panel


def test_the_documented_test_count_is_the_real_one(request):
    """CLAUDE.md and FRAMEWORK.md both quote this number. Both were wrong.

    CLAUDE.md said 75, FRAMEWORK.md said 96 in three places, and pytest
    collected 107. A reader uses the number to decide whether their checkout is
    complete, so a stale one says "you are missing tests" to someone who is not.
    """


    # Only meaningful when the whole suite was collected. Running a single node
    # id collects 1, which would fail the guard on every targeted run.
    # `-k` is an OPTION, not a positional arg, so scanning config.args never
    # saw it: a targeted `-k` run collected 1 test and failed this guard on
    # `n > 1` instead of skipping. Read the option itself.
    if (request.config.option.keyword
            or any("::" in a for a in request.config.args)):
        pytest.skip("subset run: the collected count is not the suite count")
    n = request.session.testscollected or len(request.session.items)
    assert n > 1

    claimed = {}
    for path in ("CLAUDE.md", "docs/FRAMEWORK.md"):
        txt = io.open(path, encoding="utf-8").read()
        for m in re.finditer(r"(\d+)\s+(?:regression\s+)?tests", txt):
            claimed.setdefault(path, set()).add(int(m.group(1)))

    wrong = {path: sorted(v - {n}) for path, v in claimed.items() if v - {n}}
    assert not wrong, (
        "pytest collects %d, but the docs claim %s. Update them, or the count "
        "tells a reader their checkout is incomplete." % (n, wrong))


NULL_SIBLINGS = [
    # (null arm, its treated parent, the keys that must be exactly zero)
    ("tralo_null", "tralo", ("lambda_step", "lambda_global", "lambda_local")),
    ("fioretto_null", "fioretto", ("fioretto_step_size",)),
    ("hounie_null", "hounie", ("hounie_eta_lambda",)),
    ("alm_null", "alm", ("alm_eta", "alm_mu0", "alm_mu_step")),
]


@pytest.mark.parametrize("null,parent,zeroed", NULL_SIBLINGS,
                         ids=[n for n, _, _ in NULL_SIBLINGS])
def test_every_trained_arm_has_a_working_zero_dose_sibling(null, parent, zeroed):
    """Without it, an arm-vs-clip delta cannot be attributed to the constraint.

    For a long time only tralo had one, so "was it the constraint or was it the
    regime" was falsifiable for exactly one of the four trained arms while the
    other three were reported against `clip` anyway.

    Zeroing is a DIFFERENT knob in each arm and two of them have a trap, so the
    keys are named per arm rather than pattern-matched.
    """
    proto = yaml.safe_load(open("configs/protocol.yml", encoding="utf-8"))
    blk = proto["blocks"][null]
    for key in zeroed:
        assert blk[key] == 0.0, "%s.%s must be exactly 0.0, got %r" % (
            null, key, blk[key])

    a_null, a_parent = proto["arms"][null], proto["arms"][parent]
    assert a_null["methodology"] == a_parent["methodology"]
    assert a_null["phase"] == a_parent["phase"] == "trained"
    assert "constraint_phase" in a_null["blocks"]


def test_alm_null_zeroes_the_augmentation_not_just_the_multiplier():
    """The ALM weight is `lambda + mu_t * r+`, mu_t = mu0 + mu_step * epoch.

    Zeroing alm_eta and alm_mu_step but leaving mu0 at 0.01 gives a live
    augmentation of mu0 * excess on every epoch -- a weak treatment wearing a
    control's name, which is worse than having no control.
    """
    blk = yaml.safe_load(open("configs/protocol.yml", encoding="utf-8"))["blocks"]["alm_null"]
    for epoch in (0, 14, 28):
        mu_t = blk["alm_mu0"] + blk["alm_mu_step"] * epoch
        assert mu_t == 0.0, "mu_t is %r at epoch %d, so the weight is not zero" % (
            mu_t, epoch)


def test_hounie_null_does_not_trip_hounie_s_own_stability_guard():
    """eta_u must NOT be zeroed even though it looks like a dose knob.

    hounie_rcl/train.py refuses `abs(1 - 2*eta_u*alpha) >= 1.0`, and eta_u = 0
    gives exactly 1.0, so a null built by zeroing every eta would raise before
    training a single epoch. eta_u moves only the slack u, which cannot reach
    the primal once lam is pinned at its 0.0 init.
    """
    blk = yaml.safe_load(open("configs/protocol.yml", encoding="utf-8"))["blocks"]["hounie_null"]
    factor = abs(1.0 - 2.0 * blk["hounie_eta_u"] * blk["hounie_alpha"])
    assert factor < 1.0, (
        "hounie_null would raise its own stability check: factor %.3f" % factor)
    assert blk["hounie_eta_lambda"] == 0.0


def _load_panel():
    """full_panel.py is a script, not a package module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_full_panel", os.path.join("scripts", "full_panel.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_full_panel"] = mod
    argv, sys.argv = sys.argv, ["full_panel"]
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    finally:
        sys.argv = argv
    return mod


def test_equalize_multi_respects_every_capped_class_not_just_the_first():
    """The bug its docstring describes: caps 2..n silently violated.

    `equalize` in score_arm.py takes top-K for ONE class and argmaxes the rest
    with no budget check, so a multi-capped-class campaign scored "equalized"
    metrics on an allocation that broke every cap after the first.
    """

    panel = _load_panel()
    UNLIM = 10 ** 10
    rng = np.random.default_rng(0)
    n, n_cls = 60, 4
    proba = rng.random((n, n_cls))
    proba = proba / proba.sum(axis=1, keepdims=True)
    gids = np.array([i % 3 for i in range(n)])
    capped = [1, 2]
    glob_c = [UNLIM, 10, 8, UNLIM]
    loc = {g: [UNLIM, 4, 3, UNLIM] for g in (0, 1, 2)}

    y = np.asarray(panel.equalize_multi(proba, gids, glob_c, loc, capped))
    assert y.shape == (n,)
    for c in capped:
        assert int((y == c).sum()) <= glob_c[c], (
            "class %d global cap violated: %d > %d"
            % (c, int((y == c).sum()), glob_c[c]))
        for g, bounds in loc.items():
            got = int((y[gids == g] == c).sum())
            assert got <= bounds[c], (
                "class %d local cap violated in group %s: %d > %d"
                % (c, g, got, bounds[c]))


def test_equalize_multi_matches_the_single_class_path():
    """Its docstring promises "with one capped class this is exactly the old
    behaviour" -- i.e. top-K by probability. The single-class results the
    project still stands on were produced by that path."""

    panel = _load_panel()
    UNLIM = 10 ** 10
    rng = np.random.default_rng(7)
    n, n_cls = 40, 3
    proba = rng.random((n, n_cls))
    proba = proba / proba.sum(axis=1, keepdims=True)
    gids = np.zeros(n, dtype=int)
    K = 9
    y = np.asarray(panel.equalize_multi(proba, gids, [UNLIM, K, UNLIM],
                                        {0: [UNLIM, UNLIM, UNLIM]}, [1]))
    chosen = set(np.where(y == 1)[0].tolist())
    topk = set(np.argsort(-proba[:, 1])[:K].tolist())
    assert len(chosen) == K
    assert chosen == topk, "single-class path is not top-K by probability"


def test_signflip_is_the_exact_permutation_and_hits_its_floor():
    """At D datasets the exact two-sided floor is 2^(1-D): 0.25 at three.

    This is the number that says no campaign this project can run reaches
    p<0.05 on the generalization unit, so it has to be exact, not approximate.
    """
    panel = _load_panel()
    for x in ([0.1, 0.2, 0.3], [3.0, 0.001, 99.0], [-0.4, -0.4, -0.4]):
        p, d = panel._signflip_p(x)
        assert d == 3 and abs(p - 0.25) < 1e-12, (x, p, d)
    assert panel._signflip_p([0.4])[0] == 1.0
    assert abs(panel._signflip_p([0.5, 0.5])[0] - 0.5) < 1e-12
    # a mixed-sign set must NOT reach the floor
    assert panel._signflip_p([0.1, -0.2, 0.3])[0] > 0.25
    # 2^(1-D) for a same-signed set, at every D it can reach
    for D in range(1, 7):
        p, d = panel._signflip_p([1.0] * D)
        assert d == D and abs(p - 2.0 ** (1 - D)) < 1e-12


def test_bh_monotonicity_and_that_aliases_do_not_widen_the_family():
    """Two things the scorer's q-values depend on.

    BH is a step-up procedure: the raw p*m/i can be non-monotone in the sorted
    order and must be made monotone by a backward cumulative minimum. And ccP /
    ccR must stay OUT of the family -- letting all three in widens the callable
    threshold by 2.54x.
    """
    panel = _load_panel()
    assert panel.BH_ALIAS_OF == {"ccP": "ccF1", "ccR": "ccF1"}
    assert "ccF1" not in panel.BH_ALIASES

    pv = [0.04, 0.01, 0.03]
    finite = sorted((v, str(i)) for i, v in enumerate(pv))
    q = {}
    for i, (p_, m) in enumerate(finite, 1):
        q[m] = min(1.0, p_ * len(finite) / i)
    for i in range(len(finite) - 2, -1, -1):
        q[finite[i][1]] = min(q[finite[i][1]], q[finite[i + 1][1]])
    got = [q[str(i)] for i in range(3)]
    # raw: p=0.01 -> 0.03 (i=1), p=0.03 -> 0.045 (i=2), p=0.04 -> 0.04 (i=3).
    # The raw sequence is NOT monotone -- 0.045 sits above the 0.04 that follows
    # it -- and the backward cumulative minimum pulls it down to 0.04. Without
    # that pass the middle hypothesis would carry a LARGER q than a weaker one.
    assert abs(got[1] - 0.03) < 1e-12
    assert abs(got[2] - 0.04) < 1e-12, "0.045 should be pulled down to 0.04"
    assert abs(got[0] - 0.04) < 1e-12
    ordered = [q[m] for _, m in finite]
    assert ordered == sorted(ordered), "q must be non-decreasing in sorted p"


def _panel_run(tmp_path, constrained_class, n=12, n_cls=3):
    """A minimal scorable run directory: config.json + the two prediction CSVs."""


    d = tmp_path / "ds" / "M" / "L30_G30" / "arm" / "seed_1"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    proba = rng.random((n, n_cls))
    proba = proba / proba.sum(axis=1, keepdims=True)
    frame = {
        "True_Label": [i % n_cls for i in range(n)],
        "Predicted_Label": list(proba.argmax(axis=1)),
        "Group_ID": [i % 2 for i in range(n)],
    }
    for c in range(n_cls):
        frame["Prob_Class_%d" % c] = proba[:, c]
    df = pd.DataFrame(frame)
    df.to_csv(d / "final_predictions_raw.csv", index=False)
    df.to_csv(d / "final_predictions.csv", index=False)
    cfg = {
        "status": "completed",
        "arm": "arm",
        "constraint": [0.3, 0.3],
        "constraint_tag": "L30_G30",
        "model_name": "M",
        "dataset_mode": "ds",
        "dataset_config": {"num_classes": n_cls,
                           "constrained_class": constrained_class},
        "hyperparams": {"seed": 1},
    }
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return d, cfg


def test_panel_refuses_a_run_with_no_capped_class(tmp_path):
    """`constrained_class: null` must raise with a reason, not label a cell "None".

    full_panel.py carried two more inline scalar-or-list copies after the other
    three were unified. The one building the `capped` cell key did not raise on
    None at all -- it produced the literal string "None" as the cell label, so
    the run would have been paired and scored under a cell that does not exist.
    """
    panel_mod = _load_panel()
    d, cfg = _panel_run(tmp_path, None)
    with pytest.raises(ValueError) as exc:
        panel_mod.panel(str(d), cfg)
    assert "constrained_class" in str(exc.value)


@pytest.mark.parametrize("cc,expect", [(1, "1"), ([1], "1"), ([1, 2], "1-2")])
def test_panel_labels_the_capped_cell_the_same_way_for_scalar_and_list(
        tmp_path, cc, expect):
    """A scalar and a one-element list are the same campaign and must pair.

    If they produced different `capped` labels they would land in different
    cells and every pair would silently vanish from the comparison -- the
    failure that once made in-flight campaigns read as ties.
    """
    panel_mod = _load_panel()
    d, cfg = _panel_run(tmp_path, cc)
    r = panel_mod.panel(str(d), cfg)
    assert r is not None, "a well-formed run must be scorable"
    assert r["capped"] == expect


@pytest.mark.parametrize("pct", [0.30, 0.50])
def test_targeted_correction_spends_the_whole_reachable_budget(pct):
    """The allocator every TRAINED arm uses must fill to exactly K.

    Nothing in this suite called targeted_correction before, and that is
    precisely why a regression shipped: a global-budget check added to the
    local FILL phase, while 3a and 3b were interleaved per group, blocked the
    fill for every group processed before the reductions that free the room.
    The trained arms then under-spent the capped-class budget by ~4-5% while
    the clippers -- which never call this function -- filled to exactly K.

    An asymmetry that size is the size of the entire effect under study, and it
    pointed the same way, so it would have read as a real loss for the trained
    arms. Assert the invariant force_exact=True actually promises.
    """


    n, n_cls, n_grp, capped = 2000, 7, 7, [4]
    for seed in range(8):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, n_cls, n)
        g = rng.integers(0, n_grp, n)
        logits = rng.normal(size=(n, n_cls))
        logits[:, capped[0]] += 0.8          # over-predict the capped class
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        proba = e / e.sum(axis=1, keepdims=True)

        df = pd.DataFrame({"label": y, "grp": g})
        gcon = compute_global_constraints(df, "label", pct,
                                          constrained_class=capped,
                                          num_classes=n_cls)
        lcon = compute_local_constraints(df, "label", pct, "grp",
                                         constrained_class=capped,
                                         num_classes=n_cls)
        y_pred, _, meta = targeted_correction(proba, g, gcon, lcon, capped)

        c = capped[0]
        # local caps are per-GROUP ceilings, so their sum also bounds the count
        reachable = min(gcon[c], sum(lcon[gid][c] for gid in lcon))
        got = int((y_pred == c).sum())
        assert got == reachable, (
            "seed %d pct %s: filled %d of a reachable %d -- the trained arms "
            "would score against clippers that fill to exactly K"
            % (seed, pct, got, reachable))

        # and it must still be FEASIBLE, which is what the global check exists
        # for -- filling to the budget must not overshoot any cap
        assert got <= gcon[c], "global cap violated"
        for gid in lcon:
            in_g = int((y_pred[g == gid] == c).sum())
            assert in_g <= lcon[gid][c], (
                "local cap violated in group %s: %d > %d"
                % (gid, in_g, lcon[gid][c]))


def test_every_training_log_gets_a_header_not_just_tralo_s(tmp_path):
    """A headerless log is silently mis-parsed, not loudly broken.

    write_csv_header was called by exactly one of the ten arms -- tralo. Every
    other arm appended rows to a file with no header line, so pandas took the
    FIRST LOGGED EPOCH as the column names and mislabelled every column after
    it. TraLO's training log was machine-readable and the baselines it is
    compared against were not.

    The header is now written by the writer on first row, so this asserts the
    property for a caller that never mentions headers at all.
    """
    from src.training.logging import build_csv_header, log_progress_to_csv

    path = tmp_path / "training_log.csv"
    local = {0: [10 ** 10, 5, 10 ** 10], 1: [10 ** 10, 4, 10 ** 10]}
    for epoch in range(3):
        log_progress_to_csv(
            str(path), epoch, ce_loss=0.5, train_acc=0.9,
            constraints=[10 ** 10, 12, 10 ** 10], num_classes=3,
            local_constraints=local, grad_norm=1.5, lambda_global=0.25)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    expected = build_csv_header(3, local)
    assert lines[0].split(",") == expected, (
        "first line is not the header -- pandas would eat epoch 1 as column names")
    assert len(lines) == 1 + 3, "expected a header plus one row per epoch"

    df = pd.read_csv(path)
    assert len(df) == 3, "an epoch was consumed by the header"
    assert "Grad_Norm" in df.columns and "Lambda_Global" in df.columns
    assert float(df["Grad_Norm"].iloc[0]) == 1.5
    assert float(df["Lambda_Global"].iloc[0]) == 0.25


def test_constraint_phase_reaches_every_trained_arm_and_no_posthoc_one():
    """A shared knob must reach all four trained arms, or none of them.

    This is the structure the CE-saturation gate violated: its flag was
    declared by TraLO alone, so a campaign ran it off for TraLO and on for
    both duals -- a 0.22 cc-F1 artifact against a 0.019-0.031 margin. Any
    shared constraint knob therefore lives in `constraint_phase`, which every
    trained arm includes and no post-hoc arm does, so one assignment cannot
    reach one arm and not another.
    """
    proto = yaml.safe_load(open("configs/protocol.yml", encoding="utf-8"))
    cp = proto["constraint_phase"]

    trained = [a for a, spec in proto["arms"].items()
               if spec.get("phase") == "trained"]
    assert len(trained) >= 4
    for arm in trained:
        spec = proto["arms"][arm]
        if spec.get("constraint_step") is False:
            # An arm that takes NO constraint step (`select`: a selection head,
            # no dual, no gradient clip) cannot miss a gate it does not have.
            # Handing it `constraint_phase` would emit eight keys nothing reads
            # -- the exact defect audit_config exists to catch -- so the
            # exemption is declared, and then CHECKED: it may not carry a single
            # constraint_phase key, which is what stops "constraint_step: false"
            # from becoming a way to smuggle dead keys past both gates.
            declared = set()
            for b in spec["blocks"]:
                declared |= set(proto["blocks"].get(b, {}))
            leaked = declared & (set(cp) - {"lr_constraint", "constraint_chunk_size"})
            assert not leaked, (
                "%s declares constraint_step: false but still carries %s" %
                (arm, sorted(leaked)))
            continue
        assert "constraint_phase" in spec["blocks"], (
            "%s is a trained arm that does NOT include constraint_phase, so a "
            "shared constraint knob would silently miss it. If it genuinely "
            "takes no constraint step, declare `constraint_step: false`." % arm)
    for arm, spec in proto["arms"].items():
        if spec.get("phase") == "posthoc":
            assert "constraint_phase" not in (spec.get("blocks") or []), (
                "%s is post-hoc; emitting a constraint-phase key for it would "
                "be a key with no reader" % arm)


def test_the_bounded_shape_starves_the_deepest_violator_and_the_hinges_do_not():
    """The measured multi-class failure, pinned as a property.

    This asserts a GRADIENT property, which is why it is a unit test and not a
    reading of a training log. Counts cannot establish it: measured against a
    lambda=0 control on dermmnist with classes 2+4 capped at L30_G20, CE alone
    swings the capped counts 242 -> 227 -> 324 -> 233 over four epochs with the
    penalty identically off, so no count trajectory is attributable to the
    shape without that control.

    What the control DOES show is the consequence of the property below. Every
    shape pushed class 4 down and class 2 up -- a see-saw, because the softmax
    makes the capped classes compete and the class that should resist is the
    one this shape starves. Shape set the see-saw's size in the order this test
    fixes: class 2 moved +197 under rational_bounded, +112 under squared, +86
    under linear.

    Single-class runs cannot show any of it: their spread of relative excess
    across scopes has median 1.5x, and at equal excess the shape is exactly
    inert -- a common scalar times a fixed direction, divided out by the clip.
    """
    from src.losses.transductive_loss import MulticlassTransductiveLoss, UNLIMITED

    NC = 5
    gcon = [UNLIMITED, 44.0, 45.0, UNLIMITED, UNLIMITED]

    def pull(shape):
        L = MulticlassTransductiveLoss(gcon, {}, num_classes=NC,
                                       initial_rho=10.0, penalty_shape=shape)
        for c in (1, 2):
            L.set_lambda_per_class(c, 0.01, scope="global")
        # class 1 = the DEEP violator (410 vs 44); class 2 = mild (57 vs 45)
        soft = torch.tensor([0.0, 410.0, 57.0, 0.0, 0.0], requires_grad=True)
        L.compute_global_from_counts(soft).backward()
        return float(soft.grad[1]), float(soft.grad[2])

    deep_b, mild_b = pull("rational_bounded")
    deep_s, mild_s = pull("squared")
    deep_l, mild_l = pull("linear")

    assert deep_b < mild_b, (
        "the bounded shape is supposed to STARVE the deep violator here; if "
        "this fails the measured pathology has changed and 2a2 needs redoing")
    assert deep_s > mild_s, "squared must favour the deeper violator"
    assert deep_l > mild_l, "linear must favour the deeper violator"
    # and the reversal must be large enough to matter, not a rounding artifact
    assert (deep_s / mild_s) > 10 * (deep_b / mild_b)


def test_normalize_delivers_the_same_step_size_whatever_the_raw_norm():
    """The point of `normalize`: the dose stops depending on the arm.

    Under `clip` a gradient below the threshold passes through untouched, which
    is why hounie (raw norm 0.005-0.11 against a clip of 1.0) took a step ~20x
    smaller than tralo's on every one of its 29 epochs while both configs said
    constraint_grad_clip: 1.0.
    """

    def step(raw_scale, mode):
        m = torch.nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            m.weight.fill_(0.0)
        m.weight.grad = torch.full((1, 4), raw_scale)
        before = m.weight.detach().clone()
        finish_constraint_step(m, None, None, 1.0, mode=mode,
                               fp32=True, step_rule="sgd", lr=1.0)
        return float((m.weight.detach() - before).norm())

    # raw norm 0.02 (far below the clip) and 20.0 (far above)
    small_clip, big_clip = step(0.01, "clip"), step(10.0, "clip")
    small_nrm, big_nrm = step(0.01, "normalize"), step(10.0, "normalize")

    assert small_clip == pytest.approx(0.02, rel=1e-5), small_clip
    assert big_clip == pytest.approx(1.0, rel=1e-5), big_clip
    assert small_clip < big_clip / 10, (
        "under `clip` a below-threshold gradient keeps its own magnitude -- "
        "this is the hounie asymmetry, and it must stay reproducible")

    assert small_nrm == pytest.approx(1.0, rel=1e-5), small_nrm
    assert big_nrm == pytest.approx(1.0, rel=1e-5), big_nrm
    assert small_nrm == pytest.approx(big_nrm, rel=1e-6), (
        "under `normalize` the delivered step must be the same size no matter "
        "what the arm's natural gradient scale is -- that is the whole point")


def test_the_random_direction_control_keeps_the_dose_and_drops_the_information():
    """The control must change ONLY the direction, never the step size.

    Its whole purpose is to answer "did the penalty's direction matter?", and
    it can only answer that if the dose is held exactly. If it also changed the
    norm it would confound direction with magnitude -- the same confound that
    made the dedicated-Adam arm uninterpretable (it moved 8,900x further and
    cost AP -0.0938, and no one could say which half did it).
    """

    def step(random_direction):
        torch.manual_seed(0)
        m = torch.nn.Linear(64, 8)
        for p in m.parameters():
            p.grad = torch.randn_like(p) * 0.01      # small: normalize scales UP
        before = [p.detach().clone() for p in m.parameters()]
        finish_constraint_step(m, None, None, clip=1.0, mode="normalize",
                               step_rule="sgd", lr=1.0,
                               random_direction=random_direction)
        return torch.cat([(p.detach() - b).flatten()
                          for p, b in zip(m.parameters(), before)])

    real, rand = step(False), step(True)

    # same dose: normalize delivers exactly `clip`, times lr=1.0
    assert abs(float(real.norm()) - 1.0) < 1e-4
    assert abs(float(rand.norm()) - 1.0) < 1e-4

    # and the control must not DRAW from the global RNG: if it does, the
    # control run's dropout masks and batch order diverge from the real arm's
    # too, so it varies two things instead of one. Checked on the randomiser
    # itself -- the step() helper above seeds and samples on its own.
    from src.training.constraint_step import _randomize_direction
    m2 = torch.nn.Linear(8, 4)
    for q in m2.parameters():
        q.grad = torch.ones_like(q)
    torch.manual_seed(1234)
    before_state = torch.random.get_rng_state()
    _randomize_direction(m2, 1.0, next(iter(m2.parameters())))
    assert torch.equal(torch.random.get_rng_state(), before_state), (
        "the random-direction control consumed a global RNG draw")

    # different information: a random direction in 520 dimensions is very
    # nearly orthogonal to any fixed one
    cos = float(torch.dot(real, rand) / (real.norm() * rand.norm()))
    assert abs(cos) < 0.3, "random direction is not independent of the real one"


def test_a_non_finite_constraint_gradient_never_moves_the_weights():
    """fioretto lost 10 of 29 epochs to NaN/inf. It must lose them SAFELY."""

    for bad in (float("nan"), float("inf")):
        m = torch.nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            m.weight.fill_(1.0)
        m.weight.grad = torch.full((1, 4), bad)
        before = m.weight.detach().clone()
        raw, applied = finish_constraint_step(m, None, None, 1.0,
                                              mode="normalize", fp32=True,
                                              step_rule="sgd", lr=1.0)
        assert not applied, "a %s gradient must not take the step" % bad
        assert torch.equal(m.weight.detach(), before), (
            "weights moved on a %s gradient" % bad)


@pytest.mark.parametrize("arm", ["tralo", "fioretto_ldf", "hounie_rcl",
                                 "fioretto_alm"])
def test_no_arm_hand_rolls_its_own_constraint_step(arm):
    """All four must go through src/training/constraint_step.py.

    They each hand-rolled this block once, and the copies drifted until the
    arms were not receiving the same treatment. Measured 2026-08-20 on
    results/vit_diag seed 1, same warm-up model, every config saying
    constraint_grad_clip: 1.0 --

        tralo     raw grad norm 0.638 .. 1826.5    clip bound  6 of 7
        fioretto  raw grad norm 17,667 .. 80,827   clip bound 18 of 18
        hounie    raw grad norm 0.005 .. 0.1105    clip bound  0 of 29

    -- so hounie took its raw ~0.05-norm step while the other two took unit
    ones. A direct clip_grad_norm_ call in a trainer is how that came back, so
    it is banned here rather than merely discouraged.
    """
    src = open("src/methodologies/%s/train.py" % arm, encoding="utf-8").read()
    assert "finish_constraint_step" in src, (
        "%s does not use the shared constraint step" % arm)
    assert "clip_grad_norm_" not in src, (
        "%s calls clip_grad_norm_ directly. That is the divergence this "
        "module exists to prevent -- route it through finish_constraint_step "
        "so every arm is bounded the same way." % arm)
    assert "constraint_autocast" in src, (
        "%s does not use the shared constraint autocast, so --constraint-fp32 "
        "cannot reach it and it can still lose epochs to fp16 overflow" % arm)


def test_the_margin_count_is_a_real_count_not_a_constant():
    """The trap that killed the order-statistic version, pinned.

    Centring the window on the K-th largest probability -- the obvious way to
    put the gradient on the cut -- yields `sum_i sigma((p_i - tau)/T)`, which
    counts how many items exceed the K-th largest. That is K - 0.5 for ANY
    model, so it is a constant, `relu(s - K)` is identically zero, and no
    gradient is ever produced. It was wired into the trainer and caught here.

    The margin count must instead MOVE with the model and be able to exceed
    the budget, which is what makes a violation visible at all.
    """

    torch.manual_seed(0)
    K, cls = 8, 1
    counts, hards, os_wide, os_narrow = [], [], [], []
    for shift in (-4.0, 0.0, 4.0):
        logits = torch.randn(40, 3) + torch.tensor([0.0, shift, 0.0])
        proba = F.softmax(logits, dim=1)
        counts.append(float(margin_window(proba, 0.02)[:, cls].sum()))
        hards.append(float((proba.argmax(dim=1) == cls).sum()))
        tau = torch.topk(proba[:, cls], K).values[-1]
        os_wide.append(float(torch.sigmoid((proba[:, cls] - tau) / 0.02).sum()))
        os_narrow.append(float(torch.sigmoid((proba[:, cls] - tau) / 1e-4).sum()))
        assert abs(counts[-1] - hards[-1]) < 1.0, (
            "margin count %.2f is not tracking the hard count %d"
            % (counts[-1], hards[-1]))

    swing = max(hards) - min(hards)
    assert swing > 20 and max(counts) - min(counts) > 20, (
        "the margin count barely moved (%s) as the class went from rare to "
        "dominant -- it is not measuring the count" % counts)
    assert max(counts) > K, "the count can never exceed the budget: no violation"

    # The dead version, for the record. As T -> 0 it is EXACTLY K - 0.5 for every
    # model: it counts the items above the K-th largest. At a usable T it
    # smears wherever probabilities are packed tighter than T, but it is still
    # pinned near the budget and still blind to the count -- here it spans
    # 8.5 while the true count spans 34.
    assert all(abs(c - (K - 0.5)) < 0.01 for c in os_narrow), os_narrow
    assert max(os_wide) - min(os_wide) < swing / 3, (
        "order-statistic count spread %.1f vs true spread %.1f -- this test "
        "is supposed to show it does NOT track"
        % (max(os_wide) - min(os_wide), swing))


def test_the_margin_count_puts_its_gradient_on_the_boundary():
    """Not on the unsure items. This is the entire point of the arm.

    The plain count's per-item derivative is p(1-p), maximal at p = 0.5 and
    ~zero at the cut. The margin count's is sigma'(m/T)/T, maximal at margin 0
    -- at the items one step from flipping out of the class.
    """

    torch.manual_seed(1)
    proba = F.softmax(torch.randn(60, 4), dim=1).requires_grad_(True)
    cls = 2
    margin = proba[:, cls] - torch.cat(
        [proba[:, :cls], proba[:, cls + 1:]], dim=1).max(dim=1).values

    (g_margin,) = torch.autograd.grad(
        margin_window(proba, 0.02)[:, cls].sum(), proba, retain_graph=True)
    (g_sum,) = torch.autograd.grad(proba[:, cls].sum(), proba)

    at_cut = int(margin.detach().abs().argmin())
    deep = int(margin.detach().argmax())
    gm = g_margin[:, cls].abs()
    assert gm[at_cut] > gm[deep] * 50, (
        "the margin count weights a deeply-committed item (%.3g) comparably "
        "to one at the boundary (%.3g)" % (gm[deep], gm[at_cut]))
    # the plain count is flat in p: every item gets identical weight here, and
    # the p(1-p) shape enters only through the softmax below it.
    assert torch.allclose(g_sum[:, cls], torch.ones(60))


def test_the_windowed_count_keeps_the_exact_full_N_gradient_when_chunked():
    """The shipped construction: value is the HARD count, gradient is the window.

    Each chunk backprops through `total.detach() - chunk.detach() + chunk`,
    whose gradient reaches only that chunk's items, so the chunks sum to the
    exact full-N gradient -- this is what makes `constraint_chunk_size`
    gradient-neutral. In margin mode `total` is seeded with the HARD count, so
    the same expression is also a straight-through estimator: the penalty
    reads the true count and differentiates the window.

    That matters because a wide window over-counts (56.6 against a hard 45 at
    40 items), and a penalty reading an inflated count keeps pushing after the
    cap is already satisfied -- the overshoot scripts/flag_live measured.
    """
    from src.losses.transductive_loss import margin_window, margins, window_temp

    torch.manual_seed(0)
    logits = (torch.randn(40, 3) + torch.tensor([0.0, 2.0, 0.0])).requires_grad_(True)
    K, cls = 8, 1

    def penalty(s_):
        return F.relu(s_[cls] - K) ** 2

    with torch.no_grad():
        proba0 = F.softmax(logits, dim=1)
        T = window_temp(margins(proba0), 10)
        hard = torch.zeros(3)
        for c in range(3):
            hard[c] = float((proba0.argmax(dim=1) == c).sum())
    assert float(hard[cls]) > K, "no violation, so the penalty gradient is 0"

    # reference: the full-batch gradient of the windowed count
    full_soft = margin_window(F.softmax(logits, dim=1), T).sum(dim=0)
    st_full = hard.detach() - full_soft.detach() + full_soft
    assert torch.allclose(st_full.detach(), hard), "straight-through value is not the hard count"
    (g_full,) = torch.autograd.grad(penalty(st_full), logits)

    # chunk size 7, not a divisor of 40, so the ragged last chunk runs too
    g_chunked = torch.zeros_like(logits)
    for start in range(0, 40, 7):
        sl = slice(start, min(start + 7, 40))
        eff = margin_window(F.softmax(logits[sl], dim=1), T)
        g_soft = hard.detach() - eff.sum(dim=0).detach() + eff.sum(dim=0)
        assert torch.allclose(g_soft.detach(), hard)
        (g,) = torch.autograd.grad(penalty(g_soft), logits, allow_unused=True)
        g_chunked = g_chunked + g

    assert torch.allclose(g_chunked, g_full, atol=1e-6), (
        "chunking changed the gradient (max diff %.3g) -- the detach "
        "cancellation broke, so chunk size is no longer a free knob"
        % float((g_chunked - g_full).abs().max()))
    assert g_full.abs().sum() > 0


def test_the_window_width_is_in_items_so_the_dose_cannot_vanish():
    """A fixed T is not a fixed dose, and an empty window is a silent null.

    Measured on the stored dermmnist evidence (MobileNetV3, 4 seeds, two cap
    tags): the T holding ~20 items at the boundary spans 0.182 .. 0.502 ACROSS
    SEEDS OF ONE CELL, and T = 0.02 puts 0-3 items in the window -- a run that
    contributes nothing, reports a null, and writes `completed`. Deriving T
    from a width in items makes the dose dimensionless and non-empty by
    construction.
    """
    from src.losses.transductive_loss import margins, window_temp

    torch.manual_seed(3)
    for scale in (0.5, 2.0, 8.0):        # flat, moderate and sharp models
        proba = F.softmax(torch.randn(300, 5) * scale, dim=1)
        m = margins(proba)
        for n in (10, 40, 120):
            T = window_temp(m, n)
            for c in range(5):
                inside = int((m[:, c].abs() < T[c]).sum())
                assert abs(inside - n) <= 1, (
                    "asked for %d items in the window, got %d (scale %.1f, "
                    "class %d)" % (n, inside, scale, c))
        # and T really does have to move to achieve that
        assert float(window_temp(m, 120).max()) > float(window_temp(m, 10).max())


def test_every_trained_arm_has_a_null_sibling_the_gate_can_find():
    """The gate looks up a name; a new arm can slip past it silently.

    gen_campaign warns when a trained arm is requested without its zero-dose
    sibling, because a delta vs `clip` cannot otherwise be attributed to the
    constraint rather than to the regime. It found that sibling by appending
    `_null`, so `tralo_margin` -- the arm that most needs a control, being a
    new estimator -- resolved to `tralo_margin_null`, which does not exist,
    and the gate said nothing. Arms may now name another arm's null via
    `null_sibling`; this pins that every trained arm resolves to a real one.
    """

    from configs.gen_campaign import _null_of

    P = yaml.safe_load(io.open("configs/protocol.yml", encoding="utf-8").read())
    missing = []
    for name, spec in P["arms"].items():
        if spec["phase"] != "trained" or name.endswith("_null"):
            continue
        sib = _null_of(P, name)
        if sib not in P["arms"]:
            missing.append("%s -> %s" % (name, sib))
    assert not missing, (
        "trained arms whose null sibling does not exist, so the gate cannot "
        "warn about them: %s" % ", ".join(missing))

    # And the shared one is genuinely shared, not a copy: tralo_margin differs
    # from tralo only in where the count puts its gradient, and at lambda 0 no
    # constraint gradient is formed at all, so one null serves both.
    assert _null_of(P, "tralo_margin") == _null_of(P, "tralo") == "tralo_null"
    assert P["blocks"]["tralo_null"]["lambda_global"] == 0.0


def test_the_inert_flag_gate_can_actually_detect_an_inert_flag():
    """A gate nobody has seen fail is not known to work.

    `flag_live` exists because inert flags are this project's most frequent
    failure mode -- four occurrences, every one of which passed audit_config
    (the key had a reader) and smoke_arms (the arm ran). Comparing an arm to
    ITSELF must report inert, which both proves the detector fires and proves
    the harness is deterministic: if two runs of one arm already differ, no
    difference between two arms means anything.
    """

    same = subprocess.run(
        [sys.executable, "-m", "scripts.flag_live", "tralo", "tralo",
         "--constraint-epochs", "2"],
        capture_output=True, text=True)
    assert same.returncode == 1, (
        "an arm compared to itself was NOT reported inert -- either the "
        "detector is broken or the harness is nondeterministic, and in the "
        "second case the gate cannot distinguish a live flag from noise:\n%s"
        % same.stdout[-600:])
    assert "INERT" in same.stdout

    diff = subprocess.run(
        [sys.executable, "-m", "scripts.flag_live", "tralo", "tralo_margin",
         "--constraint-epochs", "2"],
        capture_output=True, text=True)
    assert diff.returncode == 0, (
        "soft_count_mode: margin produced bit-identical predictions to sum, "
        "so it is a fifth inert flag:\n%s" % diff.stdout[-600:])


def test_the_scorers_posthoc_list_matches_the_protocol():
    """A hardcoded copy of a protocol fact drifts, and this one is load-bearing.

    `full_panel.POSTHOC_ARMS` exists so arms with `constraint_epochs == 0` are
    not flagged for producing cap-invariant predictions -- their warm-up IS the
    run, so identical predictions across cap levels is correct, not a bug. If a
    post-hoc arm is added to protocol.yml and not here, the scorer reports a
    real arm as broken. If a TRAINED arm is listed here by mistake, the scorer
    stops reporting the one failure mode that has actually occurred (six models
    behind twelve cells).
    """

    from scripts.full_panel import POSTHOC_ARMS

    P = yaml.safe_load(io.open("configs/protocol.yml", encoding="utf-8").read())
    expected = {a for a, v in P["arms"].items() if v["phase"] == "posthoc"}
    assert POSTHOC_ARMS == expected, (
        "full_panel.POSTHOC_ARMS has drifted from configs/protocol.yml.\n"
        "  only in the scorer:  %s\n"
        "  only in the protocol: %s"
        % (sorted(POSTHOC_ARMS - expected), sorted(expected - POSTHOC_ARMS)))


def test_straight_through_closes_the_K_equals_zero_trap():
    """A group with no true instances of the capped class gets K == 0 legitimately.

    On the soft value that constraint can NEVER be satisfied: `sum_i p_ic` is
    strictly positive for any softmax, even when the model predicts the class
    for nobody in the group, so `relu(count - 0)` stays positive forever. It
    contributes nothing useful and holds the ratchet gate open for every other
    constraint, for the whole run -- a standing warning in this project.

    The hard count CAN be exactly zero, so `straight_through: true` makes that
    constraint satisfiable. This pins the difference rather than the fix, since
    the K == 0 group is created by the data, not by a setting.
    """
    torch.manual_seed(4)
    # class 2 is never the argmax, but it still carries probability mass
    logits = torch.randn(80, 5)
    others = torch.cat([logits[:, :2], logits[:, 3:]], dim=1).max(dim=1).values
    logits[:, 2] = others - 1.5      # always second best, never the argmax
    p = F.softmax(logits, dim=1)

    hard = int((p.argmax(dim=1) == 2).sum())
    soft = float(p[:, 2].sum())
    assert hard == 0, "fixture broken: class 2 is predicted for %d items" % hard
    assert soft > 0.5, "fixture broken: class 2 carries no mass (%.3f)" % soft

    K = 0
    assert float(F.relu(torch.tensor(soft) - K)) > 0.5, (
        "the soft value is satisfiable at K=0, which would mean this trap is "
        "not real")
    assert float(F.relu(torch.tensor(float(hard)) - K)) == 0.0, (
        "the hard value is NOT satisfiable at K=0 -- straight_through does not "
        "close the trap after all")


def test_margin_without_straight_through_is_refused_not_silently_reinterpreted():
    """The fourth corner of the 2x2 is a third semantics, so it must not run.

    Pass 1 accumulates the plain `sum_i p_ic` into the running total; pass 2
    windows each chunk. The detach construction then cancels a WINDOWED chunk
    out of a PLAIN total, giving value `sum_i p_ic` with a windowed gradient --
    neither `tralo` nor `tralo_margin`. And the windowed count over-counts
    (56.6 against a hard 45), so the penalty pushes past feasibility, which is
    the joint arm's measured failure mode.

    The generator cannot emit this combination; a hand-written config can.
    """
    import inspect

    from src.methodologies.tralo import train as tralo_train

    src = inspect.getsource(tralo_train.train)
    assert 'soft_count_mode: margin requires straight_through' in src, (
        "the guard against margin-without-straight-through is gone; that "
        "combination now runs and produces a third, undocumented estimator")
    # and the protocol's own arm sets both, so the guard never fires in practice
    P = yaml.safe_load(io.open("configs/protocol.yml", encoding="utf-8").read())
    blk = P["blocks"]["tralo_margin"]
    assert blk.get("soft_count_mode") == "margin" and blk.get("straight_through") is True


def test_the_allocator_does_not_fall_through_to_the_LP_when_G_is_less_than_L():
    """G < L is the prescribed sweep, and it used to break the greedy on EVERY run.

    Phase 3b filled local room without re-checking the GLOBAL budget, so local
    room got spent past the global cap, the allocation came out infeasible, and
    the run silently fell through to the small-scope LP -- a DIFFERENT algorithm
    from the greedy that `clip` keeps. An arm scored against `clip` while
    running a different allocator is not an arm-vs-arm comparison, and
    `full_panel` still reports 29% fall-through on the stored evidence.

    The fix is in the code with a comment. Nothing pinned it, and the failure
    is silent: caps still hold afterwards, so `smoke_arms --matrix` passes
    either way. This pins it on the cap tags actually being run.
    """

    from configs.gen_campaign import cap_pair

    rng = np.random.default_rng(0)
    N, C, G = 600, 7, 5
    capped = [2, 4]
    labels = rng.choice(C, size=N, p=np.array([.1, .1, .25, .1, .3, .1, .05]))
    groups = rng.integers(0, G, size=N)
    df = pd.DataFrame({"label": labels, "grp": groups})

    for tag in ("L50_G30", "L40_G30", "L30_G20"):
        loc_pct, glob_pct = cap_pair(tag)
        gcon = compute_global_constraints(df, "label", glob_pct,
                                          constrained_class=capped, num_classes=C)
        lcon = compute_local_constraints(df, "label", loc_pct, "grp",
                                         constrained_class=capped, num_classes=C)
        for trial in range(5):
            logits = rng.normal(0, 2.0, size=(N, C))
            logits[:, capped] += 1.2          # over-predict the capped classes
            e = np.exp(logits - logits.max(1, keepdims=True))
            proba = e / e.sum(1, keepdims=True)
            _, _, info = targeted_correction(proba, groups, gcon, lcon, capped)
            assert not info["lp_fallback_used"], (
                "%s trial %d fell through to the LP with %d candidates -- the "
                "greedy left the allocation infeasible, so this arm would be "
                "scored against `clip` while running a different allocator"
                % (tag, trial, info["lp_fallback_candidates"]))


def test_equalize_multi_fills_to_exactly_K_so_the_items_conversion_is_exact():
    """`full_panel` reports `items per 0.01 capF1`, and that number assumes it.

    `F1 = 2TP/(K+n)` only holds when exactly K predictions are emitted for the
    capped class. If `equalize_multi` ever fell short, the denominator would be
    `(emitted + n)` and the conversion printed after every panel would be
    quietly wrong -- in the direction of understating how many items a delta is
    worth, on a scale where the whole headroom is 2 to 10 items.

    Worth pinning because the RUNTIME allocator does NOT have this property:
    measured on the stored evidence, `targeted_correction` emits K-1 on 22 of
    88 (run, capped class) pairs -- never over, so no cap is violated, but not
    exactly K either. The scorer re-equalizes from probabilities, which is why
    the house rule is to read `full_panel` and never the stored metrics.
    """

    rng = np.random.default_rng(0)
    N, C, G = 500, 7, 4
    caps = [2, 4]
    for trial in range(6):
        logits = rng.normal(0, 2.0, size=(N, C))
        logits[:, caps] += 1.0            # over-predict the capped classes
        e = np.exp(logits - logits.max(1, keepdims=True))
        P = e / e.sum(1, keepdims=True)
        gids = rng.integers(0, G, size=N)
        glob_c = np.full(C, UNLIMITED, dtype=float)
        for c in caps:
            glob_c[c] = int(rng.integers(30, 90))
        loc = {g: np.full(C, UNLIMITED, dtype=float) for g in range(G)}

        eq = equalize_multi(P, gids, glob_c, loc, caps)
        for c in caps:
            got, want = int((eq == c).sum()), int(glob_c[c])
            assert got == want, (
                "trial %d class %d: equalize_multi emitted %d against a budget "
                "of %d, so F1 = 2TP/(K+n) no longer holds and the items "
                "conversion printed by full_panel is wrong"
                % (trial, c, got, want))

def test_budgets_reads_a_post_hoc_arm_not_just_a_trained_one(tmp_path):
    """The clipper is the bar. A budget helper blind to it drops the control.

    A post-hoc arm runs `constraint_epochs: 0`, so every row of its training
    log carries the hardcoded `Limit_Class = inf` default and the log-only
    lookup returned {}. Any caller that skips a run with no budget then reports
    the treatment with no control and calls it a comparison.
    """
    from scripts.reachability import budgets

    d = tmp_path / "clip" / "seed_1"
    d.mkdir(parents=True)
    # exactly what a post-hoc arm writes: the inf default, on every row
    pd.DataFrame({"Epoch": [0, 1], "Limit_Class4": [1e10, 1e10]}).to_csv(
        d / "training_log.csv", index=False)
    pd.DataFrame({"True_Label": [4] * 100 + [0] * 50}).to_csv(
        d / "final_predictions_raw.csv", index=False)
    (d / "config.json").write_text(json.dumps({
        "constraint": [0.3, 0.3],
        "capped_classes": [4],
        "dataset_config": {"constrained_class": [4]},
    }), encoding="utf-8")

    assert budgets(d) == {4: 30}, (
        "the cap is a fraction of the class's true count: 0.3 x 100 = 30")


def test_headroom_scores_the_control_the_way_the_scorer_does(tmp_path):
    """`achieved` must be the scorer's allocation, not the raw argmax.

    The raw argmax ignores the budget, so scoring the control on it beats the
    analytic ceiling `2K/(K+n)` and yields a NEGATIVE headroom -- which is how
    this was caught. A headroom that can go negative is measuring two different
    allocations against each other.
    """
    from scripts.headroom import f1

    rng = np.random.default_rng(0)
    n, K, cls = 200, 20, 1
    y = np.zeros(n, dtype=int)
    y[:60] = cls
    P = rng.random((n, 3))
    P[:60, cls] += 1.0                     # class 1 genuinely more probable
    P = P / P.sum(axis=1, keepdims=True)

    g = np.zeros(n, dtype=int)
    G = {c: (K if c == cls else UNLIMITED) for c in range(3)}
    eq = equalize_multi(P, g, G, {0: {c: UNLIMITED for c in range(3)}}, [cls])

    assert int((eq == cls).sum()) == K, "the scorer fills the budget exactly"
    ceiling = 2.0 * K / (K + int((y == cls).sum()))
    assert f1(y, eq, cls) <= ceiling + 1e-12, (
        "no allocation emitting exactly K can beat 2K/(K+n)")
    assert f1(y, P.argmax(1), cls) > ceiling, (
        "the raw argmax DOES beat it, because it ignores the budget -- which "
        "is exactly why headroom must not be measured against it")

def test_a_cap_that_does_not_bind_gives_the_constraint_zero_gradient():
    """A seed already under budget is its own null. Measure it PER SEED.

    The penalty is built on relu(soft_count - K). Below the budget it is
    identically zero, so the constraint contributes no gradient and the treated
    arm and its null are the same run. Averaging over such a seed dilutes a
    real effect toward zero and reports it as a tie.

    Measured on the stored evidence: tissuemnist L50_G50 class 1 runs 76 / 51 /
    34 / 18 against K=56, so it binds in ONE seed of four -- while its MEAN
    excess is -11.2 and its L30 sibling's mean excess is a healthy-looking
    +10.8 on 2-of-4 binding. The mean cannot show this; only the per-seed count
    can, which is why scripts/headroom.py keeps the counts per seed.
    """


    K = 56.0
    loss = MulticlassTransductiveLoss(global_constraints=[K, K],
                                      local_constraints={}, num_classes=2)
    for count in (76.0, 51.0, 34.0, 18.0):
        soft = torch.tensor(count, requires_grad=True)
        pen = loss._penalty(soft, K)
        if count > K:
            assert float(pen) > 0.0, "an over-budget count must be penalised"
        else:
            assert float(pen) == 0.0, (
                "count %g is under K=56, so the penalty -- and the whole "
                "constraint gradient -- is identically zero" % count)
            pen.backward()
            assert float(soft.grad) == 0.0, (
                "no gradient reaches the model from a satisfied cap")

def test_the_dermmnist_split_is_grouped_by_lesion_not_by_label():
    """The split MUST be group-aware. A label-only split leaks by construction.

    HAM10000 photographs many lesions more than once -- 10,015 images over 7,470
    lesions, 26.2% of lesions with more than one image. Splitting on the label
    alone therefore puts two photographs of the SAME lesion on opposite sides:
    measured, 38.7% of the test set and 67.3% of the melanoma test set shared a
    lesion with a training image.

    Source-level, so it runs with no dataset present -- and because the failure
    is silent: a leaky split produces better-looking numbers, not an error.
    """

    src = io.open("data/dermmnist/create_slices.py", encoding="utf-8").read()
    # AST, not grep. This file DOCUMENTS the bug by name, so a substring check
    # fires on the prose explaining why the bug is gone -- the same trap that
    # made a grep report `rho_step` as read when only a log line named it.
    tree = ast.parse(src)
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    imported = {a.name for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "StratifiedShuffleSplit" not in (called | imported), (
        "create_slices.py is back to a label-only split. That is the leak.")
    assert "StratifiedGroupKFold" in imported, "the grouped splitter must be used"
    assert "StratifiedGroupKFold" in src and "groups=groups" in src, (
        "the split must be grouped by lesion_id")
    assert "lesion_id" in src and "image_id" in src, (
        "both ids must be carried into the slice so the check is reproducible "
        "from the slice alone")
    assert "raise AssertionError" in src, (
        "a slice that shares a lesion between train and test must FAIL rather "
        "than reach disk -- nothing downstream can detect it")


def test_the_prevalence_shift_only_touches_the_test_split():
    """`--shift` must move test prevalence without creating leakage.

    It drops whole images from the TEST side only, so no training item moves and
    no lesion changes sides. The point is to break the correspondence that makes
    K inferable: under a stratified split the capped class's test count is
    recoverable from TRAINING prevalence to within about one item, so the budget
    tells the model something it could already compute.
    """

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                    "data", "dermmnist"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_slices", os.path.join(os.path.dirname(__file__), "..", "data",
                                "dermmnist", "create_slices.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    labels = np.array([4] * 200 + [0] * 800)
    test_idx = np.arange(1000)
    rng = np.random.default_rng(0)
    kept = mod.shift_test(test_idx, labels, cls=4, factor=0.5, rng=rng)

    assert (labels[kept] == 4).sum() == 100, "half the capped class must remain"
    assert (labels[kept] == 0).sum() == 800, "no OTHER class may be touched"
    assert set(kept).issubset(set(test_idx)), (
        "the shift may only REMOVE test items -- it must never add one, which "
        "is what would let a training item cross over")
    before = (labels[test_idx] == 4).mean()
    after = (labels[kept] == 4).mean()
    assert after < before, "the whole point is that test prevalence moves"

def test_the_leakage_warning_is_measured_not_remembered(tmp_path, caplog):
    """A correctness claim nobody re-checks is worse than no claim.

    The loader used to print a hardcoded caveat -- "38.7% of this test set
    (776/2003) ... share a lesion_id with a TRAINING image". True when written;
    FALSE the moment the split was fixed, and it kept printing on corrected
    data, naming a test-set size that no longer existed. So it is computed from
    the slice now: silent when clean, loud when leaking, and explicit that it
    cannot tell when `lesion_id` is absent.
    """
    import logging


    from src.utils.data_loader import _warn_lesion_leakage

    def write(d, train_les, test_les):
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"label": [0] * len(train_les), "lesion_id": train_les}
                     ).to_csv(d / "train_meta.csv", index=False)
        pd.DataFrame({"label": [0] * len(test_les), "lesion_id": test_les}
                     ).to_csv(d / "test_meta.csv", index=False)

    clean = tmp_path / "clean"
    write(clean, ["a", "b"], ["c", "d"])
    with caplog.at_level(logging.WARNING):
        _warn_lesion_leakage(str(clean))
    assert not caplog.records, "a clean slice must print nothing"

    leaky = tmp_path / "leaky"
    write(leaky, ["a", "b"], ["a", "d"])
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _warn_lesion_leakage(str(leaky))
    assert any("LEAKS" in r.getMessage() for r in caplog.records), (
        "a shared lesion must be reported")
    assert any("50.0%" in r.getMessage() for r in caplog.records), (
        "and the percentage must be MEASURED from this slice, not recalled")

    blind = tmp_path / "blind"
    blind.mkdir()
    pd.DataFrame({"label": [0]}).to_csv(blind / "train_meta.csv", index=False)
    pd.DataFrame({"label": [0]}).to_csv(blind / "test_meta.csv", index=False)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _warn_lesion_leakage(str(blind))
    assert any("CANNOT be checked" in r.getMessage() for r in caplog.records), (
        "a slice with no lesion_id must say it cannot tell, never assume clean")

def test_final_predictions_that_violate_a_cap_are_refused_not_logged(tmp_path):
    """A trained arm must not be able to ship an infeasible result.

    `heuristic` raises on its own violations, so the post-hoc arms hard-failed
    while the trained arms wrote `status: completed` with "VIOLATED by N" at
    INFO level -- an asymmetry in which arms can silently ship an infeasible
    result, in the file that decides what every scorer reads.

    And the check read `global_con` only, so a LOCAL violation was not merely
    unreported, it was never looked at. A class capped only per-group has an
    UNLIMITED global budget, so every global check on it passes vacuously.
    """

    from src.pipeline.eval import write_evaluation_outputs

    n = 12
    y_test = np.zeros(n, dtype=int)
    groups = np.array([0] * 6 + [1] * 6)
    proba = np.tile(np.array([0.6, 0.4]), (n, 1))

    def run(y_pred, global_con, local_con):
        return write_evaluation_outputs(
            tmp_path, y_test, groups,
            {"y_pred": np.asarray(y_pred), "raw_pred": np.asarray(y_pred),
             "y_proba": proba,
             "metrics": {"flips_required": 0, "raw_all_satisfied": True,
                         "raw_total_excess": 0}},
            2, global_con, local_con)

    # feasible under both scopes
    run([1] * 2 + [0] * 10, [UNLIMITED, 4], {0: [UNLIMITED, 2], 1: [UNLIMITED, 2]})

    # GLOBAL violated
    with pytest.raises(RuntimeError, match="violate"):
        run([1] * 6 + [0] * 6, [UNLIMITED, 4], None)

    # LOCAL violated while the global budget is UNLIMITED -- the case the
    # global-only check could not see even in principle.
    with pytest.raises(RuntimeError, match="local group"):
        run([1] * 5 + [0] * 7, [UNLIMITED, UNLIMITED],
            {0: [UNLIMITED, 2], 1: [UNLIMITED, 2]})

def test_the_deletion_table_does_not_claim_live_code_was_deleted():
    """FRAMEWORK is the law, so a false claim in it is a defect, not a typo.

    Section (f) listed `danits_lp`, `focal`, `class_balanced` and
    `logit_adjust` as deleted methodology packages, and `cb_beta` /
    `logit_adjust_tau` as removed keys. All four packages exist, are registered
    in TRAIN_FNS, and are among the nine methodologies the PAPER claims; both
    keys are live in protocol.yml. Anyone trusting the table would conclude
    those arms are gone.

    Rather than fix the prose and hope, the table is checked: anything it says
    was removed must actually be absent.
    """



    text = io.open("docs/FRAMEWORK.md", encoding="utf-8").read()
    start = text.index("### (f) What was DELETED FROM THE CODE")
    section = text[start:text.index(chr(10) + "### ", start + 10)]

    proto = yaml.safe_load(io.open("configs/protocol.yml", encoding="utf-8"))
    live_keys = set(proto.get("core", {})) | set(proto.get("constraint_phase", {}))
    for blk in proto.get("blocks", {}).values():
        live_keys |= set(blk)

    claimed = set()
    for row in section.splitlines():
        if not row.startswith("|") or row.startswith("| removed") or "---" in row:
            continue
        cells = row.split("|")
        # Columns 1 AND 2 -- "removed" and "was". Reading only column 1 made the
        # methodology half of this check VACUOUS: that row says "5 methodology
        # packages" in column 1 and puts the actual names in column 2, so a
        # false claim about a live package sailed through. Caught by running the
        # negative control instead of trusting a green test.
        text2 = " ".join(cells[1:3])
        claimed |= set(re.findall(r"[a-z_][a-z0-9_]{2,}", text2))

    assert claimed, "the deletion table parsed to nothing -- the check is vacuous"

    still_live = sorted(k for k in claimed if k in live_keys)
    assert not still_live, (
        "FRAMEWORK section (f) says these were removed, but they are live keys "
        "in protocol.yml: %s" % still_live)

    registered = sorted(m for m in claimed if m in TRAIN_FNS)
    assert not registered, (
        "FRAMEWORK section (f) says these methodologies were deleted, but they "
        "are registered in TRAIN_FNS: %s" % registered)


def test_the_coin_control_matches_the_delivered_step_not_the_clip_bound():
    """The coin must differ from the treatment in INFORMATION only, not dose.

    `_randomize_direction` rescaled the random gradient to exactly `clip`
    unconditionally, but under the protocol default `constraint_grad_mode: clip`
    the treatment delivers min(raw, clip). So on every epoch where the clip did
    not bind, the control took a LARGER step than the thing it controls -- 20x
    for hounie, whose raw norms are 0.005-0.11 against clip 1.0 -- and the bias
    runs in the direction that flatters the treatment.
    """


    def delivered(mode, target_raw, coin):
        torch.manual_seed(0)
        m = nn.Linear(64, 8, bias=False)
        for p in m.parameters():
            p.grad = torch.ones_like(p)
            p.grad.mul_(target_raw / float(p.grad.norm()))
        opt = torch.optim.SGD(m.parameters(), lr=0.0)
        finish_constraint_step(m, opt, None, 1.0, mode=mode, step_rule="sgd",
                               lr=0.0, random_direction=coin)
        return sum(float(p.grad.pow(2).sum()) for p in m.parameters()) ** 0.5

    for mode in ("clip", "normalize"):
        for raw in (0.05, 0.5, 5.0):
            t = delivered(mode, raw, False)
            c = delivered(mode, raw, True)
            assert abs(c - t) < 1e-5, (
                "coin is dosed %.4f but the treatment delivers %.4f at "
                "mode=%s raw=%.2f (%.1fx) -- the control varies dose as well "
                "as information" % (c, t, mode, raw, c / max(t, 1e-12)))


def test_coverage_targets_uses_a_whole_test_budget_not_the_smallest_group():
    """tau's numerator and denominator must live in the same scope.

    `local_con[g][c]` bounds group g alone; `g.mean()` covers the whole batch.
    Taking min across groups put the SMALLEST group's budget over the whole
    test set -- 9/2004 instead of 67/2004 on derm L50_G30, a 7.4x
    over-tightening -- and made tau move with the LOCAL tag while the global
    cap was unchanged, so a G<L sweep would sweep the smallest group.
    """
    from src.methodologies.select.train import coverage_targets
    from src.training.constraints import UNLIMITED

    n = 2004
    loc = {g: [UNLIMITED] * 7 for g in ("a", "b", "c")}
    loc["a"][4], loc["b"][4], loc["c"][4] = 75.0, 28.0, 9.0   # sum 112

    g_tight = [UNLIMITED] * 7
    g_tight[4] = 67.0
    assert abs(coverage_targets(g_tight, loc, [4], n, 7)[4] - 67 / n) < 1e-9, (
        "global 67 is tighter than the local sum 112, so tau must be 67/n")

    # Local-only: the global is UNLIMITED, and the SUM of the locals is the
    # whole-test ceiling. Reading global_con alone would give tau = 1.0 here.
    assert abs(coverage_targets([UNLIMITED] * 7, loc, [4], n, 7)[4]
               - 112 / n) < 1e-9, "local-only must fall back to the local SUM"

    # The smallest group's budget (9) must never be the numerator.
    for g in (g_tight, [UNLIMITED] * 7):
        assert coverage_targets(g, loc, [4], n, 7)[4] > 9.0 / n + 1e-9, (
            "tau collapsed onto the smallest group's budget")


def test_the_selection_arm_actually_threads_its_running_coverage_estimate():
    """Inert-flag gate for `cov_ema`: it must set the coverage term's VALUE.

    The stabilised coverage term takes its value from a running estimate and
    its gradient from the current batch. Asserting only that the argument
    "changes the loss" is too weak -- a broken construction that merely ADDED
    the estimate (`cov_ema + cov`, no `- cov.detach()`) would pass that. So
    this pins the actual property: with cov_ema = X the penalty must equal
    cov_weight * (X - tau)^2 exactly, i.e. the value is the ESTIMATE and not
    this batch's coverage.
    """



    tau, w, X = 0.03, 32.0, 0.20
    g = torch.full((64,), 0.5, requires_grad=True)
    probs = torch.full((64, 7), 1 / 7.0)
    y = torch.zeros(64, dtype=torch.long)

    bare, cov, _b = selective_loss(g, probs, y, 4, tau, w)
    with_ema, cov2, _b2 = selective_loss(g, probs, y, 4, tau, w, cov_ema=X)
    assert abs(cov - 0.5) < 1e-6 and abs(cov2 - 0.5) < 1e-6

    # The ONLY difference between the two is the coverage term's value.
    delta = float(with_ema) - float(bare)
    expected = w * ((X - tau) ** 2 - (cov - tau) ** 2)
    assert abs(delta - expected) < 1e-4, (
        "coverage term used %.6f, expected the estimate %.2f -> %.6f"
        % (delta, X, expected))

    # ...and the gradient must still come from THIS batch, not the estimate.
    with_ema.backward()
    assert g.grad is not None and float(g.grad.abs().sum()) > 0, (
        "no gradient reaches g -- the detach construction broke the graph")

    # ...and the training loop must actually pass it.
    src = io.open("src/methodologies/select/train.py", encoding="utf-8").read()
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "selective_loss"]
    assert calls, "no call to selective_loss found in the arm"
    assert all(len(c.args) >= 7 or any(k.arg == "cov_ema" for k in c.keywords)
               for c in calls), (
        "selective_loss is called without cov_ema -- the stabilisation is inert")


def test_the_selective_risk_is_centred_so_it_does_not_only_push_coverage_down():
    """An UNcentred risk makes every item's gradient positive.

    Normalising by the expected covered mass instead of g.sum() fixes the
    variance but removes the centring the ratio estimator had for free:
    d risk / d g_i = per_i / (n*tau) > 0 for every item, so the risk term
    degenerates into a pure "cover nothing" force. Measured on a synthetic:
    equilibrium coverage falls from 0.74*tau to 0.60*tau, and undershooting
    the budget is the one regime where the two-allocator confound bites.
    A selective risk must pull EASY items in and push hard ones out.
    """


    g = torch.full((64,), 0.5, requires_grad=True)
    # All 64 ARE the capped class; the model is confident on the first half and
    # wrong on the second, so the two halves differ in per-item LOSS, which is
    # the quantity the selective risk is supposed to sort on. (Flipping the
    # LABEL instead gives both halves the same loss and the baseline cancels
    # everything -- a test that then passes on any implementation.)
    probs = torch.zeros(64, 7)
    probs[:, 4] = torch.cat([torch.full((32,), 0.95), torch.full((32,), 0.05)])
    probs[:, 0] = 1 - probs[:, 4]
    y = torch.full((64,), 4, dtype=torch.long)

    # cov_weight 0 isolates the RISK term from the coverage pull.
    loss, _cov, _b = selective_loss(g, probs, y, 4, 0.03, 0.0)
    loss.backward()
    easy, hard = g.grad[:32], g.grad[32:]
    assert float(easy.mean()) < 0 < float(hard.mean()), (
        "risk gradient is not centred: easy items %.4g, hard items %.4g -- "
        "both signs must not be the same, or the term only pushes coverage "
        "down" % (float(easy.mean()), float(hard.mean())))


def test_no_methodology_reads_the_test_LABELS_except_to_count_them():
    """`y_test` is handed to every train(); only discipline keeps it unread.

    `TrainInputs` carries the true test labels so the pipeline can derive K and
    score afterwards, and every methodology receives them. The transductive
    setting permits reading the COUNT (that is the declared assumption) and
    nothing else. An arm that trains on test inputs -- self-training, pseudo
    -labelling -- is exactly where an accidental read would be easy to write
    and impossible to notice, so this is a gate rather than a convention.
    """

    offenders = []
    for path in sorted(pathlib.Path("src/methodologies").rglob("*.py")):
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        # Every `len(...)` argument is an allowed context; collect them first.
        allowed = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "len"):
                for a in node.args:
                    allowed.update(id(n) for n in ast.walk(a))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and node.attr == "y_test"
                    and id(node) not in allowed):
                offenders.append("%s:%d" % (path.as_posix(), node.lineno))

    assert not offenders, (
        "these methodology modules read the TEST LABELS outside a len() call, "
        "which is label leakage, not transduction: %s" % offenders)


def test_the_two_fioretto_arms_initialise_both_multiplier_scopes_alike():
    """ALM claims to differ from LDF in the DUAL UPDATE only.

    `fioretto_lambda_init` reached ALM's global and local multipliers but only
    LDF's global one -- LDF's locals were hardcoded to 0.0. Sweeping the key
    would then have changed the dual rule AND the local initialisation at once,
    so the arm-vs-arm delta would be attributable to neither. Dormant at the
    protocol's 0.0, which is exactly why only a gate catches it.
    """

    for mod in ("fioretto_ldf", "fioretto_alm"):
        path = "src/methodologies/%s/train.py" % mod
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        assigns = [n for n in ast.walk(tree)
                   if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Subscript)
                           and getattr(t.value, "id", None) == "lambda_l"
                           for t in n.targets)]
        init = [n for n in assigns if isinstance(n.value, ast.Constant)]
        assert not init, (
            "%s initialises lambda_l to the literal %r instead of the "
            "fioretto_lambda_init value the other arm uses"
            % (mod, [n.value.value for n in init]))


def test_alm_gates_its_constraint_pass_on_the_weights_the_loss_uses():
    """A treatment that logs itself and never runs is this repo's failure mode.

    ALM's chunk loss weights each term by `lambda + aug`. `has_work` decided
    whether to run that pass at all, and consulted `lambda` only -- so with the
    multipliers pinned at 0 and the augmentation climbing, the pass was skipped
    on every epoch while `training_log.csv` wrote a rising mu_t.
    """

    src = io.open("src/methodologies/fioretto_alm/train.py",
                   encoding="utf-8").read()
    tree = ast.parse(src)
    node = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.Assign)
                 and any(getattr(t, "id", None) == "has_work" for t in n.targets)),
                None)
    assert node is not None, "has_work assignment not found"
    names = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
    for w in ("lambda_g", "lambda_l", "aug_g", "aug_l"):
        assert w in names, (
            "has_work ignores %s, but the constraint loss weights terms by "
            "lambda + aug -- the augmentation is unreachable whenever the "
            "multipliers are 0" % w)


def test_no_f_string_placeholder_survives_in_a_plain_string_literal():
    """A `{name}` in a NON-f string prints the braces instead of the value.

    `main.py`'s dispatcher header printed the literal "{completed} done,
    {failed} failed" for an unknown length of time -- an implicit-concatenation
    slip where the first fragment lost its `f` prefix and the second kept it.
    The dispatcher banner is how a run's progress is read at a glance, so this
    is a silent instrument failure, not a cosmetic one.

    Scoped to the two places the slip can actually reach a reader: a plain
    fragment implicitly concatenated INTO an f-string, and a bare literal
    handed straight to `print`. Docstrings are excluded -- this repo's carry
    real algebra, and `(1 - beta) / (1 - beta^n)` is not a formatting bug.
    """

    PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_.\[\]']*\}")

    def bad(node):
        return [c for c in ([node] if isinstance(node, ast.Constant)
                            else getattr(node, "values", []))
                if isinstance(c, ast.Constant) and isinstance(c.value, str)
                and PLACEHOLDER.search(c.value)]

    offenders = []
    roots = ([pathlib.Path("main.py")] + sorted(pathlib.Path("src").rglob("*.py"))
             + sorted(pathlib.Path("scripts").rglob("*.py")))
    for path in roots:
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            # (a) a plain fragment glued into an f-string: the exact slip.
            if isinstance(node, ast.JoinedStr):
                hits = bad(node)
            # (b) a bare literal handed to print().
            elif (isinstance(node, ast.Call)
                  and getattr(node.func, "id", None) == "print"):
                hits = [h for a in node.args for h in bad(a)]
            else:
                continue
            offenders += ["%s:%d %r" % (path.as_posix(), node.lineno,
                                        h.value[:60]) for h in hits]
    assert not offenders, (
        "these string literals contain an f-string placeholder but are not "
        "f-strings, so they print the braces: %s" % offenders)


def test_the_scorer_says_a_skipped_run_CRASHED_rather_than_never_started(tmp_path):
    """A dead treatment arm must not read as "campaign merely unfinished".

    The dispatcher resets an interrupted run to `pending`, which makes a run
    that CRASHED indistinguishable from one that never started. Measured
    2026-08-21 on `results/dosefix`: all 8 `tralo` runs died of CUDA OOM in the
    transductive forward while every control completed -- the lambda=0 arm
    skips that pass entirely, so it never allocates the chunk that OOMs -- and
    the panel reported a clean comparison between controls with no hint that
    the treatment was absent. The tell is an error_log.json beside the config.
    """
    import json as _json

    run = tmp_path / "ds" / "mdl" / "L50_G30" / "tralo" / "seed_1"
    run.mkdir(parents=True)
    (run / "config.json").write_text(
        _json.dumps({"status": "pending", "arm": "tralo"}), encoding="utf-8")
    (run / "error_log.json").write_text(
        _json.dumps({"exception_type": "OutOfMemoryError"}), encoding="utf-8")

    out = subprocess.run(
        [sys.executable, "-m", "scripts.full_panel",
         "--campaign", str(tmp_path), "--control", "clip"],
        capture_output=True, text=True).stdout

    assert "CRASHED" in out, (
        "a run with an error_log.json was reported as merely not-completed:\n%s"
        % out[-1500:])
    assert "OutOfMemoryError" in out, "the exception type was not surfaced"
    assert "tralo" in out and "NO scorable run" in out, (
        "the scorer did not say which arm contributed nothing")


def test_two_cap_tags_that_produce_the_same_budget_are_one_cap_level():
    """House rule 4 is about BUDGETS, not tag spellings.

    `gen_campaign` refuses a single-cap campaign by comparing tag strings, which
    any two distinct spellings satisfy. Measured 2026-08-21 on `results/dosefix`:
    L40_G30 and L50_G30 both bind on the GLOBAL scope (local sums 82 and 103
    against a global 62), so class 2 gets K=62 and class 4 K=67 in BOTH cells --
    one budget level wearing two tags, and a per-cell count over them
    double-counts a single measurement.
    """
    from scripts.verify_caps import duplicate_budget_tags

    # the real dosefix numbers
    dup = duplicate_budget_tags({2: {"L40_G30": 62, "L50_G30": 62},
                                 4: {"L40_G30": 67, "L50_G30": 67}})
    assert dup == [(2, 62, ["L40_G30", "L50_G30"]),
                   (4, 67, ["L40_G30", "L50_G30"])], dup

    # a genuine sweep must come back clean
    assert duplicate_budget_tags({2: {"L50_G30": 62, "L50_G20": 41}}) == []

    # ...and a partial collision must be reported for the colliding class only
    dup = duplicate_budget_tags({2: {"a": 10, "b": 10}, 4: {"a": 10, "b": 20}})
    assert dup == [(2, 10, ["a", "b"])], dup


def test_no_autocast_banned_op_is_reachable_from_an_arm():
    """CUDA autocast BANS a few ops; calling one under AMP is a hard error.

    `select` called F.binary_cross_entropy. Every GPU run died in 11 seconds --
    "binary_cross_entropy and BCELoss are unsafe to autocast" -- wrote a
    header-only training_log.csv, and was reset to `pending` by the dispatcher,
    so the campaign looked merely unfinished while burning the card.

    ⚠️ THE BAN IS DYNAMIC, NOT LEXICAL, which is why this test is a blanket ban
    rather than a scan of `with autocast` bodies. The offending call sat in
    `selective_loss`, a module-level function that autocast never encloses
    lexically -- it is only ever CALLED from inside the block. A lexical gate
    written that way passed while the bug was present; this one was checked
    against the real defect.

    ⚠️ And the obvious fix is the wrong one. `binary_cross_entropy_with_logits`
    IS autocast-safe, but it applies a SIGMOID to its input, and our argument is
    a SOFTMAX probability, not that class's logit -- swapping it in silences the
    crash while quietly computing a different loss. Write the BCE out by hand on
    the already-clamped probability instead; it is bit-identical (verified, max
    abs diff 0.0) and autocast-safe.

    `scripts/smoke_arms.py` cannot cover this: it runs on cpu, where autocast is
    a no-op.
    """

    BANNED = {"binary_cross_entropy", "BCELoss"}

    offenders = []
    for path in sorted(pathlib.Path("src").rglob("*.py")):
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = getattr(node.func, "attr", getattr(node.func, "id", ""))
                if fn in BANNED:
                    offenders.append("%s:%d %s"
                                     % (path.as_posix(), node.lineno, fn))

    assert not offenders, (
        "CUDA autocast bans these ops and every training path runs under AMP, "
        "so the arm dies on its first GPU batch: %s. Write the loss out by "
        "hand on the clamped probability -- NOT the _with_logits variant, which "
        "computes a different quantity." % offenders)


def test_the_selective_risks_centring_estimate_is_not_self_referential():
    """`selective_loss` must return THIS batch's estimate, not the EMA it was given.

    It returned the EMA-substituted value, and the trainer feeds the return
    straight back into its EMA -- so the update was 0.9*x + 0.1*x = x and the
    centring constant FROZE at batch 1 of epoch 1 and never moved again.

    That is not a stale log line. d risk / d g_i = (per_i - centre)/(n*tau), so a
    centre pinned to an untrained model's covered-set loss while per_item
    collapses (CE saturates by epoch 10, L_CE 0.02) flips the sign for nearly
    every item and turns the risk term into exactly the "cover everything" force
    the centring exists to prevent. The tell was that `cov_ema`, three lines
    away, IS recomputed from the batch -- the asymmetry between the two.
    """


    ema, beta, seen = None, 0.9, []
    torch.manual_seed(0)
    for b in range(5):
        g = torch.rand(64, requires_grad=True)
        # sharpen the probabilities each batch: the covered-set loss MUST move
        probs = torch.softmax(torch.randn(64, 7) * (1.0 + 2.0 * b), dim=1)
        y = torch.randint(0, 7, (64,))
        _loss, _cov, base = selective_loss(g, probs, y, 4, 0.03, 32.0,
                                           cov_ema=0.03, risk_ema=ema)
        ema = base if ema is None else beta * ema + (1 - beta) * base
        seen.append(base)

    assert len(set(round(v, 9) for v in seen)) > 1, (
        "selective_loss returned the SAME centring estimate on 5 batches whose "
        "probabilities differ by 5x -- it is echoing the EMA it was handed, so "
        "the caller's EMA can never move: %s" % seen)
    assert abs(ema - seen[0]) > 1e-6, "the caller's EMA never left batch 1"


def test_the_scorer_detects_a_run_that_collapsed_on_its_final_epoch(tmp_path):
    """The pipeline keeps the LAST epoch unconditionally, so one bad terminal
    epoch is the model that gets scored -- and when that run is the CONTROL,
    every arm appears to beat it at that seed.

    Measured on `results/dosefix`: `clip` seed 4 ended 0.9934 -> 0.9116 while
    the other seven controls ended 0.9935-1.0000, and it reversed the sign of
    the 4-seed `tralo_null` vs `clip` headline. FRAMEWORK section 9 states
    this; this test is what makes the statement checkable.
    """
    from scripts.full_panel import _terminal_collapse

    n = [0]

    def _log(accs):
        n[0] += 1
        d = tmp_path / ("r%d" % n[0])
        d.mkdir()
        rows = ["Epoch,Train_Acc"]
        rows += ["%d,%.4f" % (i, a) for i, a in enumerate(accs)]
        (d / "training_log.csv").write_text(chr(10).join(rows) + chr(10))
        return str(d)

    assert _terminal_collapse(_log([0.95, 0.98, 0.9934, 0.9116])) is not None, (
        "the detector missed the exact drop it was written for "
        "(dosefix clip seed 4, 0.9934 -> 0.9116)")

    # NEGATIVE CONTROLS -- a gate is not done until it has been shown not to
    # fire on the things it must leave alone.
    for accs, why in [
        ([0.95, 0.98, 0.9934, 0.9940], "a healthy run still improving"),
        ([0.95, 0.98, 0.9934, 0.9900], "ordinary wobble, 0.0034 < 0.02"),
        ([0.9116, 0.9934], "a run that RECOVERED -- only the last epoch is kept"),
    ]:
        assert _terminal_collapse(_log(accs)) is None, "fired on " + why
    assert _terminal_collapse(str(tmp_path / "nope")) is None, (
        "raised or fired on a missing training_log.csv")


def test_framework_section_9_does_not_still_carry_the_retracted_3_seed_number():
    """I published `tralo_null` - `clip` = -5.2 items from THREE seeds. The
    fourth reverses it (4-seed mean -0.06 items) because the `clip` control at
    that seed collapsed on its final epoch. The retraction has to sit ABOVE the
    superseded table or the next reader quotes the dead number.
    """
    txt = io.open(os.path.join(REPO, "docs", "FRAMEWORK.md"),
                  encoding="utf-8").read()
    assert "-0.0188" in txt, "the 3-seed table vanished; keep it as superseded"
    head = txt.split("-0.0188")[0]
    assert "RETRACTED AT 4 SEEDS" in head, (
        "FRAMEWORK section 9 shows the 3-seed ccF1 table with no retraction "
        "above it -- a reader quotes the first number they see.")
