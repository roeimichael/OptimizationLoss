"""danits_lp methodology: LP post-hoc allocation (Shifman et al. 2025).

Solves a global+local LP with an arbitrary cost matrix on softmax
probabilities of the warmup model. Identity cost minimises expected
error rate subject to Psi (global) and Phi (per-group) caps.
"""

import logging
import time

import numpy as np
import torch

from src.pipeline.contracts import TrainInputs, TrainOutputs
from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)


def _infer_probs(model, X_test, chunk_size=256):
    model.eval()
    with torch.no_grad():
        chunks = [model(X_test[i:i + chunk_size])
                  for i in range(0, len(X_test), chunk_size)]
        probs = torch.softmax(torch.cat(chunks, dim=0), dim=1).cpu().numpy()
    return probs


def train(inputs: TrainInputs) -> TrainOutputs:
    from src.methodologies.danits_lp import solve_lp_assignment

    cost_preset = inputs.hyperparams.get("danits_cost_preset", "identity")
    if cost_preset != "identity":
        raise ValueError(
            f"Unknown danits_cost_preset {cost_preset!r}. Only 'identity' is "
            f"supported. To add task-specific cost matrices, extend "
            f"danits_research/cost_matrices.py.")

    device = inputs.device
    X_test = inputs.X_test.to(device)
    probs = _infer_probs(inputs.model, X_test)

    num_classes = inputs.num_classes
    omega = np.ones((num_classes, num_classes), dtype=np.float64) - np.eye(num_classes, dtype=np.float64)
    psi_list = [int(v) if v < UNLIMITED else None for v in inputs.global_con]
    phi_dict = {}
    if inputs.local_con:
        for g, bounds in inputs.local_con.items():
            phi_dict[g] = [int(v) if v < UNLIMITED else None for v in bounds]

    t_alloc = time.time()
    lp_res = solve_lp_assignment(
        y_proba=probs, groups=inputs.group_ids, cost_matrix=omega,
        psi=psi_list, phi=phi_dict, verbose=False,
    )
    exec_time = time.time() - t_alloc
    if lp_res.status != "OPTIMAL":
        raise RuntimeError(f"danits_lp: LP solver returned status={lp_res.status}")
    y_pred = lp_res.y_pred
    log.info("DANITS-LP [%s]: obj=%.4f status=%s runtime=%.3fs vars=%d constraints=%d",
             cost_preset, lp_res.objective_value, lp_res.status,
             exec_time, lp_res.num_variables, lp_res.num_constraints)

    return TrainOutputs(
        model=inputs.model,
        summary={"allocation_time": exec_time, "lp_objective": float(lp_res.objective_value)},
        skip_targeted_correction=True,
        precomputed_predictions=y_pred,
    )
