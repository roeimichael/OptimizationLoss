# Heuristic baseline: greedy allocation on a fixed warmup model.
# Trains CE-only model, then assigns predictions via top-K constrained allocation.
# Processes constrained classes first to ensure optimal budget utilization.

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.pipeline.data import load_data
from src.pipeline.warmup import run_warmup
from src.training.metrics import compute_metrics
from src.utils.filesystem_manager import load_config_from_path
from src.pipeline.setup import seed_all
from src.pipeline.io import save_results_to_config
from src.training.logging import save_final_predictions, save_evaluation_metrics
from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)



def _build_hierarchy(num_classes, global_constraints, constrained_classes):
    constrained_sorted = sorted(constrained_classes,
                                key=lambda c: global_constraints[c])
    unconstrained = [c for c in range(num_classes) if c not in constrained_classes]
    return constrained_sorted + unconstrained


def apply_allocation_heuristic(probs: np.ndarray, groups: np.ndarray, hierarchy: List[int],
                               global_constraints: List[float], local_constraints: Dict[int, List[float]],
                               num_classes: int = 7) -> Tuple[np.ndarray, float]:
    start_time = time.time()
    n_samples, n_classes = probs.shape
    y_pred = np.full(n_samples, -1, dtype=int)
    assigned_mask = np.zeros(n_samples, dtype=bool)
    current_global = {c: 0 for c in range(n_classes)}
    current_local = {}
    argmax_preds = np.argmax(probs, axis=1)
    for class_idx in hierarchy:
        g_limit = global_constraints[class_idx]
        is_constrained = g_limit < UNLIMITED
        unassigned = np.where(~assigned_mask)[0]
        if len(unassigned) == 0:
            break
        if is_constrained:
            class_probs = probs[unassigned, class_idx]
            sorted_indices = unassigned[np.argsort(class_probs)[::-1]]
        else:
            prefer = argmax_preds[unassigned] == class_idx
            candidates = unassigned[prefer]
            if len(candidates) == 0:
                continue
            class_probs = probs[candidates, class_idx]
            sorted_indices = candidates[np.argsort(class_probs)[::-1]]
        for idx in sorted_indices:
            group_id = groups[idx]
            if group_id not in current_local:
                current_local[group_id] = {c: 0 for c in range(n_classes)}
            if is_constrained and current_global[class_idx] >= g_limit:
                break
            l_limit = local_constraints.get(group_id, [UNLIMITED] * num_classes)[class_idx]
            if l_limit is None or np.isnan(l_limit):
                l_limit = UNLIMITED
            if l_limit < UNLIMITED and current_local[group_id][class_idx] >= l_limit:
                continue
            y_pred[idx] = class_idx
            assigned_mask[idx] = True
            current_global[class_idx] += 1
            current_local[group_id][class_idx] += 1
    remaining = np.where(~assigned_mask)[0]
    for idx in remaining:
        sample_probs = probs[idx].copy()
        group_id = groups[idx]
        # Ensure group exists in current_local before checking — otherwise
        # first-time groups skipped local limit check and could overflow.
        if group_id not in current_local:
            current_local[group_id] = {c: 0 for c in range(n_classes)}
        for c in range(n_classes):
            if global_constraints[c] < UNLIMITED and current_global[c] >= global_constraints[c]:
                sample_probs[c] = -1
            if global_constraints[c] < UNLIMITED:
                l_limit = local_constraints.get(group_id, [UNLIMITED] * n_classes)[c]
                if l_limit < UNLIMITED and current_local[group_id].get(c, 0) >= l_limit:
                    sample_probs[c] = -1
        best = np.argmax(sample_probs)
        y_pred[idx] = best
        current_global[best] = current_global.get(best, 0) + 1
        current_local[group_id][best] = current_local[group_id].get(best, 0) + 1
    return y_pred, time.time() - start_time


def run_heuristic(config_path: str) -> None:
    experiment_path = Path(config_path).parent
    config = load_config_from_path(experiment_path)
    # Seed control — matches run_experiment.py so both paths produce
    # deterministic warmup models when a seed is specified.
    seed = config.get('hyperparams', {}).get('seed', None)
    seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info("Running heuristic %s on %s (model=%s, seed=%s)", config_path, device, config['model_name'], seed)
    if torch.cuda.is_available():
        log.info("GPU: %s | CUDA: %s | BF16: %s",
                 torch.cuda.get_device_name(0), torch.version.cuda,
                 torch.cuda.is_bf16_supported())
    data = load_data(config)
    X_train_tensor = data.X_train
    y_train_tensor = data.y_train
    X_test_tensor = data.X_test.to(device)
    global_con = data.global_con
    local_con = data.local_con
    num_classes = data.num_classes
    constrained_classes = data.constrained_classes
    warmup_start = time.time()
    model, _from_cache = run_warmup(
        config, num_classes, X_train_tensor, y_train_tensor, device,
    )
    warmup_time = time.time() - warmup_start
    model.eval()
    with torch.no_grad():
        chunk_size = 256
        logit_chunks = [model(X_test_tensor[i:i + chunk_size])
                        for i in range(0, len(X_test_tensor), chunk_size)]
        probs = torch.softmax(torch.cat(logit_chunks, dim=0), dim=1).cpu().numpy()
    groups_np = data.groups_test
    y_true = data.y_test
    methodology = config.get('methodology', 'heuristic')
    if methodology == 'danits_lp':
        # Paper [5] (Shifman et al. 2025) LP post-hoc with arbitrary cost matrix.
        # This branches from the SAME cached warmup as 'heuristic'
        # (via base_model_id), so all four methodologies can be compared
        # head-to-head on identical Phase-1 weights.
        from danits_research import solve_lp_assignment
        # Cost matrix preset (currently only 'identity' is wired in -- the
        # DermMNIST priority matrices were dropped during cleanup since the
        # active datasets are TissueMNIST + CIFAR-100). Identity matrix
        # minimises expected error rate subject to Psi/Phi caps and is built
        # dynamically from num_classes so it works for any dataset shape.
        cost_preset = config.get('hyperparams', {}).get('danits_cost_preset', 'identity')
        if cost_preset != 'identity':
            raise ValueError(
                f"Unknown danits_cost_preset {cost_preset!r}. Only 'identity' "
                f"is supported. To add task-specific cost matrices, extend "
                f"danits_research/cost_matrices.py.")
        omega = np.ones((num_classes, num_classes), dtype=np.float64) - np.eye(num_classes, dtype=np.float64)
        # Convert the project's (global_con, local_con) format — which uses
        # UNLIMITED=1e10 sentinel — into paper-[5] Psi/Phi (None = unbounded).
        psi_list = [int(v) if v < UNLIMITED else None for v in global_con]
        phi_dict: dict = {}
        if local_con:
            for g, bounds in local_con.items():
                phi_dict[g] = [int(v) if v < UNLIMITED else None for v in bounds]
        t_alloc = time.time()
        lp_res = solve_lp_assignment(
            y_proba=probs, groups=groups_np, cost_matrix=omega,
            psi=psi_list, phi=phi_dict, verbose=False,
        )
        exec_time = time.time() - t_alloc
        if lp_res.status != "OPTIMAL":
            raise RuntimeError(
                f"danits_lp: LP solver returned status={lp_res.status}"
            )
        y_pred = lp_res.y_pred
        log.info(
            "DANITS-LP [%s]: obj=%.4f status=%s runtime=%.3fs vars=%d constraints=%d",
            cost_preset, lp_res.objective_value, lp_res.status,
            exec_time, lp_res.num_variables, lp_res.num_constraints,
        )
    else:
        hierarchy = _build_hierarchy(num_classes, global_con, constrained_classes)
        y_pred, exec_time = apply_allocation_heuristic(
            probs, groups_np, hierarchy, global_con, local_con, num_classes)
    # Save raw argmax predictions (before heuristic/LP reallocation)
    argmax_preds = np.argmax(probs, axis=1)
    save_final_predictions(Path(experiment_path) / 'final_predictions_raw.csv',
                           y_true, argmax_preds, probs, groups_np)
    for c in range(num_classes):
        pred_count = (y_pred == c).sum()
        limit = int(global_con[c]) if global_con[c] < UNLIMITED else 'INF'
        status = 'OK' if (isinstance(limit, str) or pred_count <= limit) else f'VIOLATED by {pred_count - limit}'
        log.info("Heuristic class %d: pred=%d limit=%s %s", c, pred_count, limit, status)
    metrics = compute_metrics(y_true, y_pred, probs)
    save_final_predictions(Path(experiment_path) / 'final_predictions.csv',
                           y_true, y_pred, probs, groups_np)
    # Track 1: constraint-specific metrics
    from src.training.metrics import compute_flips, compute_raw_constraint_satisfaction
    flips = compute_flips(argmax_preds, y_pred)
    raw_sat = compute_raw_constraint_satisfaction(
        argmax_preds, global_con, local_con, groups_np, constrained_classes)
    metrics['flips_required'] = flips
    metrics.update(raw_sat)
    log.info("[Track1] flips=%d raw_satisfied=%s excess=%d",
             flips, raw_sat['raw_all_satisfied'], raw_sat['raw_total_excess'])
    metrics['warmup_time'] = float(warmup_time)
    metrics['constraint_train_time'] = 0.0  # post-hoc methods have no constraint training
    metrics['posthoc_time'] = float(exec_time)
    save_evaluation_metrics(Path(experiment_path) / 'evaluation_metrics.csv', metrics)
    save_results_to_config(config, experiment_path, {
        'accuracy': float(metrics['accuracy']),
        'precision_macro': float(metrics['precision_macro']),
        'recall_macro': float(metrics['recall_macro']),
        'f1_macro': float(metrics['f1_macro']),
        'training_time': float(warmup_time + exec_time),
        'warmup_time': float(warmup_time),
        'constraint_train_time': 0.0,
        'posthoc_time': float(exec_time),
        'samples_adjusted': int(flips),
    })
    log.info("Heuristic: acc=%.4f time=%.2fs", metrics['accuracy'], exec_time)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    try:
        run_heuristic(args.config_path)
    except Exception as e:
        log.error("Heuristic failed: %s", e, exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
