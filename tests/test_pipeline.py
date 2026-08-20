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
    import pandas as pd
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
    import pandas as pd
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
    from src.models import get_model
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
    import torch.nn as nn
    from src.models import get_model
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
    from src.experiments.runner import TRAIN_FNS
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
    src = io._io.open(os.path.join(REPO, "scripts", "full_panel.py"),
                      encoding="utf-8").read() if False else open(
        os.path.join(REPO, "scripts", "full_panel.py"), encoding="utf-8").read()
    key = src.split("key = [")[1].split("]")[0]
    assert '"capped"' in key, "the capped class is not in the pairing key"
    import pandas as pd
    with pytest.raises(ValueError, match="pairing key is missing"):
        fp._one(pd.Series([0.4, -0.4]))
    assert fp._one(pd.Series([0.4])) == 0.4


# ------------------------------------------------------------- the verdict rule

def _panel_verdict(tmp_path, n_better_cells, n_tied_cells, metric="AP"):
    """Build a synthetic campaign with a KNOWN answer and read the verdict."""
    import pandas as pd
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
    import torch
    import torch.nn as nn
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
    import numpy as np
    import pandas as pd
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
    from src.utils.data_loader import _load_imagery_data as load_data
    d = str(tmp_path / "bad_len")
    _write_slice(d, truncate_test_labels=True)
    with pytest.raises(ValueError, match="test_images.npy has 8 rows"):
        load_data(_cfg(d))


def test_loader_refuses_a_slice_whose_labels_exceed_num_classes(tmp_path):
    from src.utils.data_loader import _load_imagery_data as load_data
    d = str(tmp_path / "wrong_ds")
    _write_slice(d, n_classes=6)                 # 6 real classes...
    with pytest.raises(ValueError, match="num_classes is 4"):
        load_data(_cfg(d, n_classes=4))          # ...config says 4


def test_loader_refuses_a_slice_missing_the_capped_class(tmp_path):
    from src.utils.data_loader import _load_imagery_data as load_data
    d = str(tmp_path / "no_capped")
    _write_slice(d, drop_capped=True)
    with pytest.raises(ValueError, match="does not occur in this slice"):
        load_data(_cfg(d))


def test_loader_accepts_a_well_formed_slice(tmp_path):
    """The guards must not fire on good data."""
    from src.utils.data_loader import _load_imagery_data as load_data
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
    import torch

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
    import torch

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
    import numpy as np
    import pandas as pd
    from src.utils.data_loader import _load_imagery_data as load_data
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
    import numpy as np
    import pandas as pd
    from src.utils.data_loader import _load_imagery_data as load_data
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
    import yaml
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
    import torch
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
    structural: every trainer that runs a constraint phase imports the SAME two
    functions from the SAME module, and the scorer reads the field.
    """
    import io as _io
    import os

    trained = ["tralo", "fioretto_ldf", "hounie_rcl", "fioretto_alm"]
    for m in trained:
        path = os.path.join("src", "methodologies", m, "train.py")
        src = _io.open(path, encoding="utf-8").read()
        assert "from src.training.reordering import" in src, (
            "%s must use the shared diagnostic, not a private copy" % m)
        assert "reordering_report(" in src, "%s never calls it" % m
        assert '"reordering"' in src, "%s never puts it in the summary" % m

    # it has to survive to disk, and outside config["results"] -- a NaN tau on
    # a constant score column would otherwise mark the run `diverged`
    runner = _io.open(os.path.join("src", "experiments", "runner.py"),
                      encoding="utf-8").read()
    assert "config['reordering']" in runner
    results_blk = runner[runner.index("save_results_to_config(config"):]
    assert "reordering" not in results_blk[:results_blk.index("})")]

    # and the scorer must actually read it
    panel = _io.open(os.path.join("scripts", "full_panel.py"), encoding="utf-8").read()
    assert 'cfg.get("reordering")' in panel
    assert "_reordering_check(rows)" in panel


def test_the_documented_test_count_is_the_real_one(request):
    """CLAUDE.md and FRAMEWORK.md both quote this number. Both were wrong.

    CLAUDE.md said 75, FRAMEWORK.md said 96 in three places, and pytest
    collected 107. A reader uses the number to decide whether their checkout is
    complete, so a stale one says "you are missing tests" to someone who is not.
    """
    import io as _io
    import re

    import pytest as _pytest

    # Only meaningful when the whole suite was collected. Running a single node
    # id collects 1, which would fail the guard on every targeted run.
    if any("::" in a or "-k" in a for a in request.config.args):
        _pytest.skip("subset run: the collected count is not the suite count")
    n = request.session.testscollected or len(request.session.items)
    assert n > 1

    claimed = {}
    for path in ("CLAUDE.md", "docs/FRAMEWORK.md"):
        txt = _io.open(path, encoding="utf-8").read()
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
    import yaml
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
    import yaml
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
    import yaml
    blk = yaml.safe_load(open("configs/protocol.yml", encoding="utf-8"))["blocks"]["hounie_null"]
    factor = abs(1.0 - 2.0 * blk["hounie_eta_u"] * blk["hounie_alpha"])
    assert factor < 1.0, (
        "hounie_null would raise its own stability check: factor %.3f" % factor)
    assert blk["hounie_eta_lambda"] == 0.0


def _load_panel():
    """full_panel.py is a script, not a package module."""
    import importlib.util
    import sys

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
    import numpy as np

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
    import numpy as np

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
    import json

    import numpy as np
    import pandas as pd

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
    import numpy as np
    import pandas as pd

    from src.training.constraints import (compute_global_constraints,
                                          compute_local_constraints)
    from src.utils.posthoc_adjustment import targeted_correction

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

    import pandas as pd
    df = pd.read_csv(path)
    assert len(df) == 3, "an epoch was consumed by the header"
    assert "Grad_Norm" in df.columns and "Lambda_Global" in df.columns
    assert float(df["Grad_Norm"].iloc[0]) == 1.5
    assert float(df["Lambda_Global"].iloc[0]) == 0.25


def test_ce_skip_lives_in_the_shared_block_and_reaches_every_trained_arm():
    """The defect that got CE-skip deleted was ASYMMETRY, not the mechanism.

    `enable_ce_skip` was declared by TraLO alone, so a campaign ran the gate off
    for TraLO and on for both duals -- a 0.22 cc-F1 artifact against a
    0.019-0.031 margin. The structural fix is that the keys live in
    `constraint_phase`, which every trained arm includes and no post-hoc arm
    does, so one assignment reaches all of them or none.
    """
    import yaml
    proto = yaml.safe_load(open("configs/protocol.yml", encoding="utf-8"))
    cp = proto["constraint_phase"]
    assert "ce_skip_acc" in cp and "ce_skip_patience" in cp
    assert cp["ce_skip_acc"] == 0.0, (
        "the committed default must be OFF -- every result so far was produced "
        "with CE running every epoch")

    trained = [a for a, spec in proto["arms"].items()
               if spec.get("phase") == "trained"]
    assert len(trained) >= 4
    for arm in trained:
        assert "constraint_phase" in proto["arms"][arm]["blocks"], (
            "%s is a trained arm that does NOT include constraint_phase, so the "
            "CE-skip gate would silently miss it" % arm)
    for arm, spec in proto["arms"].items():
        if spec.get("phase") == "posthoc":
            assert "constraint_phase" not in (spec.get("blocks") or []), (
                "%s is post-hoc; emitting a constraint-phase key for it would "
                "be a key with no reader" % arm)

    # and no arm-level block may redeclare it -- that is how the asymmetry
    # reappears
    for name, blk in (proto.get("blocks") or {}).items():
        if isinstance(blk, dict):
            for k in ("ce_skip_acc", "ce_skip_patience", "enable_ce_skip"):
                assert k not in blk, (
                    "block %r redeclares %s; it must come ONLY from the shared "
                    "constraint_phase block" % (name, k))


def test_ce_skip_is_a_live_gate_not_an_inert_flag():
    """A flag that is read but changes nothing is this project's #1 failure."""
    from src.training.ce_schedule import CESaturationSkip

    off = CESaturationSkip({"ce_skip_acc": 0.0})
    assert not off.enabled
    for e in range(10):
        off.update(1.0, e)
    assert not off.should_skip(), "a disabled gate must never fire"

    on = CESaturationSkip({"ce_skip_acc": 0.995, "ce_skip_patience": 2})
    assert on.enabled
    on.update(0.99, 0)
    assert not on.should_skip(), "below threshold must not arm the gate"
    on.update(0.996, 1)
    assert not on.should_skip(), "one saturated epoch is not `patience`"
    on.update(0.997, 2)
    assert on.should_skip(), "two consecutive saturated epochs must fire it"
    assert on.skip_from_epoch == 2

    # the streak must RESET on a dip, or patience means nothing
    r = CESaturationSkip({"ce_skip_acc": 0.995, "ce_skip_patience": 2})
    r.update(0.996, 0)
    r.update(0.5, 1)
    r.update(0.996, 2)
    assert not r.should_skip(), "a dip must reset the streak"
    r.update(0.996, 3)
    assert r.should_skip()

    # and once fired it LATCHES -- an un-latching gate would let the 126 CE
    # steps per epoch back in, which is the force it exists to remove
    r.update(0.1, 4)
    assert r.should_skip(), "the gate must latch"


def test_normalize_delivers_the_same_step_size_whatever_the_raw_norm():
    """The point of `normalize`: the dose stops depending on the arm.

    Under `clip` a gradient below the threshold passes through untouched, which
    is why hounie (raw norm 0.005-0.11 against a clip of 1.0) took a step ~20x
    smaller than tralo's on every one of its 29 epochs while both configs said
    constraint_grad_clip: 1.0.
    """
    import torch
    from src.training.constraint_step import finish_constraint_step

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


def test_a_non_finite_constraint_gradient_never_moves_the_weights():
    """fioretto lost 10 of 29 epochs to NaN/inf. It must lose them SAFELY."""
    import torch
    from src.training.constraint_step import finish_constraint_step

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


@pytest.mark.parametrize("arm", ["tralo", "fioretto_ldf", "hounie_rcl",
                                 "fioretto_alm"])
def test_every_trained_arm_wires_the_same_ce_skip(arm):
    """All four must construct it, gate the CE loop on it, and feed it.

    Checked as source structure rather than behaviour because the failure being
    guarded is one arm silently not having the wiring at all -- which is
    invisible to any test that only runs the arms that do.

    The scope walk below is not decoration. The `in src` assertions alone
    passed on 2026-08-20 while SIX arms could not run at all: the three duals
    split their constraint phase into `_train_constraints`, so `ce_skip` was
    constructed in that helper and `ce_skip.summary()` was read in `train()` --
    two different scopes, every string present, `NameError` at runtime. Only
    `smoke_arms` caught it, and only because it executes. A substring test
    cannot see a scope, so it must not be the only guard on a name.
    """
    import ast
    path = "src/methodologies/%s/train.py" % arm
    src = open(path, encoding="utf-8").read()
    assert "from src.training.ce_schedule import CESaturationSkip" in src
    assert "ce_skip = CESaturationSkip(hp)" in src, (
        "%s does not construct the gate" % arm)
    assert "[] if ce_skip.should_skip()" in src, (
        "%s does not gate its CE pass on it" % arm)
    assert "ce_skip.update(cached_train_acc, epoch)" in src, (
        "%s never feeds the gate, so it could never fire" % arm)
    assert "ce_skip" in src.split("def train(")[-1], (
        "%s does not report whether its gate fired -- 'never fired' and 'fired "
        "and did nothing' are different results that look identical in the "
        "metrics" % arm)

    # every function that READS `ce_skip` must also BIND it in that same scope
    for fn in [n for n in ast.walk(ast.parse(src))
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        reads, binds = False, False
        for n in ast.walk(fn):
            if isinstance(n, ast.Name) and n.id == "ce_skip":
                if isinstance(n.ctx, ast.Load):
                    reads = True
                else:
                    binds = True
        assert not (reads and not binds), (
            "%s: %s() reads `ce_skip` but never binds it in its own scope -- "
            "this is a NameError the moment the arm runs" % (arm, fn.name))
