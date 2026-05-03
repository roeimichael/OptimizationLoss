"""Shared seeding + AMP/cudnn runtime config.

Extracted from run_experiment.py / run_heuristic.py / run_fioretto.py
which all duplicated the same setup blocks.
"""

import logging

import numpy as np
import torch

log = logging.getLogger(__name__)


def seed_all(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.info("Set random seed: %d", seed)


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
