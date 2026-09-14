import logging
import numpy as np
from src.training.metrics import (
    compute_flips,
    compute_metrics,
    compute_raw_constraint_satisfaction,
)
from src.training.logging import save_final_predictions
from src.utils.constants import UNLIMITED, INFERENCE_CHUNK_SIZE
from src.pipeline.campaign import DEPLOYMENT as DEPLOYMENT_PROTOCOL
from src.utils.inference import chunked_probs
from src.methodologies.heuristic.train import (
    _build_hierarchy, apply_allocation_heuristic, verify_allocation,
)

log = logging.getLogger(__name__)


def evaluate_with_posthoc(
    model,
    X_test,
    y_test,
    group_ids,
    global_con,
    local_con,
    constrained_classes,
):
    model.eval()
    y_proba = chunked_probs(model, X_test, INFERENCE_CHUNK_SIZE)
    raw_pred = y_proba.argmax(axis=1)
    if not np.isfinite(y_proba).all():
        n_bad = int((~np.isfinite(y_proba)).any(axis=1).sum())
        raise RuntimeError(
            "model produced non-finite probabilities for %d of %d test items -- it diverged. Refusing to score it: argmax of NaN is class 0, which looks like a healthy degenerate classifier."
            % (n_bad, len(y_proba))
        )
    n_classes = y_proba.shape[1]
    hierarchy = _build_hierarchy(n_classes, global_con, constrained_classes)
    y_pred, allocation_time = apply_allocation_heuristic(
        y_proba, group_ids, hierarchy, global_con, local_con or {}, n_classes)
    violations = verify_allocation(y_pred, group_ids, global_con, local_con or {}, n_classes)
    if violations:
        raise RuntimeError('deployment violates caps: %s' % violations)
    adj = compute_flips(raw_pred, y_pred)
    posthoc_meta = {'deployment_protocol': DEPLOYMENT_PROTOCOL,
                    'inference_chunk_size': INFERENCE_CHUNK_SIZE, 'allocation_time': allocation_time}
    metrics = compute_metrics(y_test, y_pred, y_proba,
                              constrained_classes=constrained_classes)
    flips = compute_flips(raw_pred, y_pred)
    raw_sat = compute_raw_constraint_satisfaction(
        raw_pred, global_con, local_con, group_ids, constrained_classes
    )
    metrics["flips_required"] = flips
    metrics.update(raw_sat)
    log.info(
        "[final] acc=%.4f f1=%.4f adjusted=%d",
        metrics["accuracy"],
        metrics["f1_macro"],
        adj,
    )
    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "raw_pred": raw_pred,
        "metrics": metrics,
        "adj": adj,
        "posthoc_meta": posthoc_meta,
    }


def write_evaluation_outputs(
    experiment_path, y_test, group_ids, result, num_classes, global_con, local_con=None
):
    y_pred = result["y_pred"]
    raw_pred = result["raw_pred"]
    y_proba = result["y_proba"]
    metrics = result["metrics"]
    violations = []
    for c in range(num_classes):
        pred_count = int((y_pred == c).sum())
        limit = int(global_con[c]) if global_con[c] < UNLIMITED else "INF"
        over = not isinstance(limit, str) and pred_count > limit
        if over:
            violations.append("global class %d: %d > %d" % (c, pred_count, limit))
        log.info(
            "Class %d: pred=%d limit=%s %s",
            c,
            pred_count,
            limit,
            "VIOLATED by %d" % (pred_count - limit) if over else "OK",
        )
    for group_id, bounds in (local_con or {}).items():
        mask = np.asarray(group_ids) == group_id
        for c in range(num_classes):
            lim = bounds[c] if c < len(bounds) else UNLIMITED
            if lim is None or (isinstance(lim, float) and np.isnan(lim)):
                continue
            if lim < UNLIMITED:
                n = int((y_pred[mask] == c).sum())
                if n > lim:
                    violations.append(
                        "local group %s class %d: %d > %d" % (group_id, c, n, int(lim))
                    )
    if violations:
        raise RuntimeError(
            "final predictions violate %d cap(s) AFTER post-hoc adjustment: %s. Refusing to write a run that does not satisfy its own constraints."
            % (len(violations), violations[:5])
        )
    save_final_predictions(
        experiment_path / 'final_predictions.csv', y_test, y_pred, y_proba, group_ids)
    save_final_predictions(
        experiment_path / 'final_predictions_raw.csv', y_test, raw_pred, y_proba, group_ids)
    log.info(
        "[Track1] flips=%d raw_satisfied=%s excess=%d",
        metrics["flips_required"],
        metrics["raw_all_satisfied"],
        metrics["raw_total_excess"],
    )
