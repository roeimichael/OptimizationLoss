"""Shared seeding + AMP/cudnn runtime config.

Extracted from run_experiment.py / run_heuristic.py / run_fioretto.py
which all duplicated the same setup blocks.
"""

import logging

import numpy as np
import random

import torch

log = logging.getLogger(__name__)


def seed_all(seed):
    """Seed every RNG the pipeline draws from, and pin the nondeterministic ops.

    `random` was missing, and cuDNN's conv backward is nondeterministic by
    default -- so two runs of the same config differed. With method effects at
    ~0.1 pp, that noise is the same order as the signal.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    log.info("Set random seed: %d (deterministic cudnn)", seed)


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
