"""LP-clip driver for the imbalanced-learning baselines (Track B / B1).

Protocol (revised 2026-07-23 after the smoke test): the backbone is TRAINED
with the imbalanced loss in the shared warmup phase (via hp['warmup_loss'];
see src/losses/imbalanced_losses.build_warmup_criterion), then this methodology
allocates predictions with the Shifman-LP clipper -- identical to danits_lp.
Only the training LOSS differs from the CE-warmup + LP-clip baseline
(danits_lp), so the comparison isolates exactly the imbalanced-training effect.

(The earlier fine-tune-from-CE-warmup approach was empirically a no-op: the CE
warmup is saturated, so any training loss on it is ~0 with ~0 gradients.)
"""

import logging

import numpy as np

from src.methodologies.danits_lp import solve_lp_assignment
from src.pipeline.contracts import TrainInputs, TrainOutputs
from src.utils.constants import UNLIMITED
from src.utils.inference import chunked_probs
from src.utils.constants import CONSTRAINT_CHUNK_SIZE

log = logging.getLogger(__name__)


def _lp_clip(model, inputs, device):
    """Shifman-LP allocation on the imbalanced-trained model's test-set softmax
    (identical formulation to danits_lp: identity cost, per-class psi + per-group phi)."""
    X_test = inputs.X_test.to(device)
    chunk = int(inputs.hyperparams.get("constraint_chunk_size", CONSTRAINT_CHUNK_SIZE))
    probs = chunked_probs(model, X_test, chunk)
    n = inputs.num_classes
    omega = np.ones((n, n), dtype=np.float64) - np.eye(n, dtype=np.float64)
    psi = [int(v) if v < UNLIMITED else None for v in inputs.global_con]
    phi = {g: [int(v) if v < UNLIMITED else None for v in bounds]
           for g, bounds in (inputs.local_con or {}).items()}
    res = solve_lp_assignment(y_proba=probs, groups=inputs.group_ids,
                              cost_matrix=omega, psi=psi, phi=phi)
    if res.status != "OPTIMAL":
        raise RuntimeError(f"imbalanced LP clip: solver status={res.status}")
    return res.y_pred, float(res.objective_value)


def run_imbalanced(inputs: TrainInputs, method: str) -> TrainOutputs:
    y_pred, lp_obj = _lp_clip(inputs.model, inputs, inputs.device)
    log.info("%s: LP-clip of imbalanced-trained warmup, obj=%.4f", method, lp_obj)
    return TrainOutputs(
        model=inputs.model,
        summary={"method": method, "lp_objective": lp_obj},
        skip_targeted_correction=True,
        precomputed_predictions=y_pred,
    )
