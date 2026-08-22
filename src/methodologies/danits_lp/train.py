"""danits_lp methodology: LP post-hoc allocation (Shifman et al. 2025).

Solves a global+local LP with an arbitrary cost matrix on softmax
probabilities of the warmup model. Identity cost minimises expected
error rate subject to Psi (global) and Phi (per-group) caps.
"""

import logging
import time

import numpy as np

from src.pipeline.contracts import TrainInputs, TrainOutputs
from src.utils.constants import UNLIMITED, INFERENCE_CHUNK_SIZE
from src.utils.inference import chunked_probs

log = logging.getLogger(__name__)


def train(inputs: TrainInputs) -> TrainOutputs:
    from src.methodologies.danits_lp import solve_lp_assignment

    chunk_size = int(inputs.hyperparams.get("inference_chunk_size",
                                            INFERENCE_CHUNK_SIZE))
    device = inputs.device
    X_test = inputs.X_test.to(device)
    probs = chunked_probs(inputs.model, X_test, chunk_size)

    num_classes = inputs.num_classes
    # The identity misclassification cost, which is what the manuscript claims
    # LP-LG uses ("rather than the general cost matrix"). It is the only cost
    # this arm has ever run, so it is written here rather than selected by a
    # config key with exactly one legal value.
    omega = np.ones((num_classes, num_classes), dtype=np.float64) - np.eye(num_classes, dtype=np.float64)
    psi_list = [int(v) if v < UNLIMITED else None for v in inputs.global_con]
    phi_dict = {}
    if inputs.local_con:
        for g, bounds in inputs.local_con.items():
            phi_dict[g] = [int(v) if v < UNLIMITED else None for v in bounds]

    t_alloc = time.time()
    lp_res = solve_lp_assignment(
        y_proba=probs, groups=inputs.group_ids, cost_matrix=omega,
        psi=psi_list, phi=phi_dict,
    )
    exec_time = time.time() - t_alloc
    if lp_res.status != "OPTIMAL":
        raise RuntimeError(f"danits_lp: LP solver returned status={lp_res.status}")
    y_pred = lp_res.y_pred
    log.info("DANITS-LP [identity]: obj=%.4f status=%s runtime=%.3fs vars=%d constraints=%d",
             lp_res.objective_value, lp_res.status,
             exec_time, lp_res.num_variables, lp_res.num_constraints)

    return TrainOutputs(
        model=inputs.model,
        summary={"allocation_time": exec_time, "lp_objective": float(lp_res.objective_value)},
        skip_targeted_correction=True,
        precomputed_predictions=y_pred,
    )
