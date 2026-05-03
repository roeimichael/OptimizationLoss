"""Shared evaluation: inference + posthoc adjustment + Track1 metrics + per-class verification.

evaluate_with_posthoc handles the per-candidate work (no I/O, no per-class log)
so methodologies that pick between checkpoints (fioretto_ldf) can call it
multiple times cheaply. write_evaluation_outputs saves files and logs the
per-class summary; only call it once, on the final chosen result.
"""

import logging

from src.training.metrics import (
    compute_flips,
    compute_metrics,
    compute_raw_constraint_satisfaction,
    get_predictions_with_probabilities,
)
from src.pipeline.io import save_final_predictions
from src.utils.constants import UNLIMITED
from src.utils.posthoc_adjustment import targeted_correction

log = logging.getLogger(__name__)


def evaluate_with_posthoc(model, X_test, y_test, group_ids, global_con, local_con,
                          constrained_classes, num_classes, *,
                          skip_targeted_correction=False,
                          precomputed_predictions=None,
                          label="final"):
    """Inference + targeted_correction + metrics (incl. Track1).

    skip_targeted_correction=True with precomputed_predictions: caller already
    enforced limits (heuristic / danits_lp). Skip targeted_correction.

    Returns dict: y_pred, y_proba, raw_pred, metrics, adj, posthoc_meta.
    """
    model.eval()
    raw_pred, y_proba = get_predictions_with_probabilities(model, X_test)
    adj = 0
    posthoc_meta = {}

    if skip_targeted_correction and precomputed_predictions is not None:
        y_pred = precomputed_predictions
    else:
        y_pred = raw_pred
        needs_adjustment = any(global_con[c] < UNLIMITED for c in constrained_classes)
        if needs_adjustment:
            y_pred, adj, posthoc_meta = targeted_correction(
                y_proba, group_ids, global_con, local_con, constrained_classes)

    metrics = compute_metrics(y_test, y_pred, y_proba)
    flips = compute_flips(raw_pred, y_pred)
    raw_sat = compute_raw_constraint_satisfaction(
        raw_pred, global_con, local_con, group_ids, constrained_classes)
    metrics["flips_required"] = flips
    metrics.update(raw_sat)

    log.info("[%s] acc=%.4f f1=%.4f adjusted=%d",
             label, metrics["accuracy"], metrics["f1_macro"], adj)

    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "raw_pred": raw_pred,
        "metrics": metrics,
        "adj": adj,
        "posthoc_meta": posthoc_meta,
    }


def write_evaluation_outputs(experiment_path, y_test, group_ids, result,
                             num_classes, global_con):
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

    for c in range(num_classes):
        pred_count = (y_pred == c).sum()
        limit = int(global_con[c]) if global_con[c] < UNLIMITED else "INF"
        status = ("OK" if (isinstance(limit, str) or pred_count <= limit)
                  else f"VIOLATED by {pred_count - limit}")
        log.info("Class %d: pred=%d limit=%s %s", c, pred_count, limit, status)

    log.info("[Track1] flips=%d raw_satisfied=%s excess=%d",
             metrics["flips_required"],
             metrics["raw_all_satisfied"],
             metrics["raw_total_excess"])
