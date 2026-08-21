"""Shared evaluation: inference + posthoc adjustment + Track1 metrics + per-class verification.

evaluate_with_posthoc handles the per-candidate work (no I/O, no per-class log)
so methodologies that pick between checkpoints (fioretto_ldf) can call it
multiple times cheaply. write_evaluation_outputs saves files and logs the
per-class summary; only call it once, on the final chosen result.
"""

import logging

import numpy as np

from src.training.metrics import (
    compute_flips,
    compute_metrics,
    compute_raw_constraint_satisfaction,
    get_predictions_with_probabilities,
)
from src.training.logging import save_final_predictions
from src.utils.constants import UNLIMITED
from src.utils.posthoc_adjustment import targeted_correction

log = logging.getLogger(__name__)


def evaluate_with_posthoc(model, X_test, y_test, group_ids, global_con, local_con,
                          constrained_classes, *,
                          skip_targeted_correction=False,
                          precomputed_predictions=None):
    """Inference + targeted_correction + metrics (incl. Track1).

    skip_targeted_correction=True with precomputed_predictions: caller already
    enforced limits (heuristic / danits_lp). Skip targeted_correction.

    Returns dict: y_pred, y_proba, raw_pred, metrics, adj, posthoc_meta.
    """
    model.eval()
    raw_pred, y_proba = get_predictions_with_probabilities(model, X_test)
    # Check the PROBABILITIES, not the metrics derived from them. All-NaN
    # logits argmax to class 0, which scores like a degenerate but healthy
    # classifier -- every summary number comes out finite and the run is
    # recorded `completed`, so the dispatcher never revisits it and the cell
    # silently has one fewer seed.
    if not np.isfinite(y_proba).all():
        n_bad = int((~np.isfinite(y_proba)).any(axis=1).sum())
        raise RuntimeError(
            "model produced non-finite probabilities for %d of %d test items -- "
            "it diverged. Refusing to score it: argmax of NaN is class 0, which "
            "looks like a healthy degenerate classifier."
            % (n_bad, len(y_proba)))
    adj = 0
    posthoc_meta = {}

    if skip_targeted_correction and precomputed_predictions is not None:
        y_pred = precomputed_predictions
    else:
        y_pred = raw_pred
        # Both scopes. Gating on the GLOBAL cap alone meant a class capped only
        # per-group was never adjusted post-hoc, so the arm reported whatever
        # raw argmax produced and the local caps were simply not enforced. That
        # is exactly the configuration the framework now prescribes for making
        # the global scope testable (sweep G < L), and the local-only case.
        needs_adjustment = (
            any(global_con[c] < UNLIMITED for c in constrained_classes)
            or any(bounds[c] < UNLIMITED
                   for bounds in (local_con or {}).values()
                   for c in constrained_classes))
        if needs_adjustment:
            y_pred, adj, posthoc_meta = targeted_correction(
                y_proba, group_ids, global_con, local_con, constrained_classes,
                force_exact=True)

    metrics = compute_metrics(y_test, y_pred, y_proba)
    flips = compute_flips(raw_pred, y_pred)
    raw_sat = compute_raw_constraint_satisfaction(
        raw_pred, global_con, local_con, group_ids, constrained_classes)
    metrics["flips_required"] = flips
    metrics.update(raw_sat)

    log.info("[final] acc=%.4f f1=%.4f adjusted=%d",
             metrics["accuracy"], metrics["f1_macro"], adj)

    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "raw_pred": raw_pred,
        "metrics": metrics,
        "adj": adj,
        "posthoc_meta": posthoc_meta,
    }


def write_evaluation_outputs(experiment_path, y_test, group_ids, result,
                             num_classes, global_con, local_con=None):
    """Persist final + raw predictions and log per-class + Track1 summary.

    Call once, on the final chosen evaluation (after candidate selection).
    """
    y_pred = result["y_pred"]
    raw_pred = result["raw_pred"]
    y_proba = result["y_proba"]
    metrics = result["metrics"]

    save_final_predictions(experiment_path / "final_predictions.csv",
                           y_test, y_pred, y_proba, group_ids)
    save_final_predictions(experiment_path / "final_predictions_raw.csv",
                           y_test, raw_pred, y_proba, group_ids)

    violations = []
    for c in range(num_classes):
        pred_count = int((y_pred == c).sum())
        limit = int(global_con[c]) if global_con[c] < UNLIMITED else "INF"
        over = not isinstance(limit, str) and pred_count > limit
        if over:
            violations.append("global class %d: %d > %d" % (c, pred_count, limit))
        log.info("Class %d: pred=%d limit=%s %s", c, pred_count, limit,
                 "VIOLATED by %d" % (pred_count - limit) if over else "OK")

    # BOTH scopes. This loop read `global_con` only, so a LOCAL cap violated in
    # the stored predictions was not merely unreported -- it was never looked
    # at. That matters for the class of cell the framework prescribes: a class
    # capped only per-group has an UNLIMITED global budget, so every global
    # check on it passes vacuously and the local one was the only real check.
    for group_id, bounds in (local_con or {}).items():
        mask = (np.asarray(group_ids) == group_id)
        for c in range(num_classes):
            lim = bounds[c] if c < len(bounds) else UNLIMITED
            if lim is None or (isinstance(lim, float) and np.isnan(lim)):
                continue
            if lim < UNLIMITED:
                n = int((y_pred[mask] == c).sum())
                if n > lim:
                    violations.append("local group %s class %d: %d > %d"
                                      % (group_id, c, n, int(lim)))

    # RAISE, do not log. `heuristic` already raises on its own violations, so
    # the post-hoc arms hard-failed here while the trained arms wrote
    # `status: completed` with "VIOLATED by N" at INFO level -- an asymmetry in
    # which arms can silently ship an infeasible result, in the file that
    # decides what every scorer reads. feasibility_check found zero violations
    # over 199 runs, so this should never fire; that is the argument for making
    # it fatal, not for leaving it as a log line.
    if violations:
        raise RuntimeError(
            "final predictions violate %d cap(s) AFTER post-hoc adjustment: %s. "
            "Refusing to write a run that does not satisfy its own constraints."
            % (len(violations), violations[:5]))

    log.info("[Track1] flips=%d raw_satisfied=%s excess=%d",
             metrics["flips_required"],
             metrics["raw_all_satisfied"],
             metrics["raw_total_excess"])
