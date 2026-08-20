"""Shared seeding + AMP/cudnn runtime config.

Extracted from run_experiment.py / run_heuristic.py / run_fioretto.py
which all duplicated the same setup blocks.
"""

import logging
import os

import numpy as np
import random

import torch

log = logging.getLogger(__name__)


def seed_all(seed):
    """Seed every RNG the pipeline draws from, and pin the nondeterministic ops.

    `random` was missing, and cuDNN's conv backward is nondeterministic by
    default -- so two runs of the same config differed. With method effects at
    ~0.1 pp, that noise is the same order as the signal.

    MEASURED 2026-08-20, and the docstring above understated it badly. Running
    the SAME arm (`clip`), SAME seed, SAME config twice on ViTB16 x dermmnist:

        epoch 1   loss 0.7624  acc 0.8245   |  loss 0.7624  acc 0.8245  identical
        epoch 6   loss 0.1048  acc 0.9809   |  loss 0.1049  acc 0.9480
        epoch 30  loss 0.0221  acc 0.9939   |  loss 0.0266  acc 0.9973

        final macro-F1  0.6709 vs 0.7015    -> 0.0306 apart
        raw excess         126 vs    336

    0.0306 macro-F1 of run-to-run noise, against a headline TraLO-vs-clip
    effect of 0.0017. The noise is roughly 18x the signal. Epoch 1 is
    bit-identical and divergence starts at epoch 2, which rules out seeding and
    data order and leaves nondeterministic KERNELS.

    `cudnn.deterministic` alone does not cover them: scatter/reduction and
    several non-cuDNN CUDA kernels have nondeterministic implementations, and
    FP16's adaptive GradScaler amplifies any difference because the scale it
    picks depends on which step first overflowed.

    So `torch.use_deterministic_algorithms` is now called, with the CUBLAS
    workspace env var it requires set BEFORE the first CUDA context. warn_only
    is used deliberately: an op with no deterministic implementation should
    make a loud warning, not kill a 20-hour campaign -- but it will be visible
    in the log rather than silent, which is the state this was in before.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # cuBLAS needs this set before the first CUDA context to make its GEMM
    # reductions deterministic; without it use_deterministic_algorithms raises
    # on any matmul, which on a transformer backbone is every layer.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        det = "full"
    except Exception as exc:                       # pragma: no cover
        det = "cudnn-only (%s)" % type(exc).__name__
        log.warning("use_deterministic_algorithms unavailable: %s", exc)
    log.info("Set random seed: %d (determinism: %s)", seed, det)


def setup_runtime(device):
    """Configure cudnn + select AMP dtype/scaler for the given device.

    Returns (use_amp, amp_dtype, scaler). cudnn.benchmark is forced off
    on CUDA: Blackwell sm_120 VBIOS temp threshold bug crashes with autotuning.
    """
    if device.type != "cuda":
        return False, torch.float32, None
    torch.backends.cudnn.benchmark = False
    gpu_arch = torch.cuda.get_device_capability(0)[0]
    use_bf16 = gpu_arch >= 8 and torch.cuda.is_bf16_supported()
    if use_bf16:
        return True, torch.bfloat16, None
    return True, torch.float16, torch.amp.GradScaler("cuda")


def runtime_provenance(device):
    """What a result needs in order to be comparable to another result.

    The two servers run different AMP regimes (dsisco01 Quadro RTX 6000 =
    FP16 + GradScaler, dsisco02 RTX PRO 6000 Blackwell = BF16, no scaler), and
    on the FP16 path an overflowing step is SKIPPED -- so the same config can
    apply a different number of optimizer steps depending on the card.
    """
    use_amp, amp_dtype, scaler = setup_runtime(device)
    return {
        "device": str(device),
        "gpu_name": (torch.cuda.get_device_name(device)
                     if device.type == "cuda" else None),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "amp_enabled": bool(use_amp),
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype else None,
        "grad_scaler": scaler is not None,
    }
