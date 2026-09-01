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

    MEASURED 2026-08-20 by `scripts/variance_probe.py` -- three runs of the
    SAME arm (`clip`), SAME seed, SAME config, SAME GPU, back to back:

        F1 (Macro)         0.6524 .. 0.6882   spread 0.0358   sd 0.0181
        Precision (Macro)  0.7177 .. 0.7625   spread 0.0448
        warm-up times      1178.5 / 1176.4 / 1176.4 s  (each really retrained)

    0.0358 macro-F1 of run-to-run noise against a headline TraLO-vs-clip effect
    of 0.0017 -- 21x the signal -- and that was measured WITH
    cudnn.deterministic, benchmark=False and CUBLAS_WORKSPACE_CONFIG already
    set. Averaging more seeds does not help: it shrinks the standard error, and
    this is what each draw is drawn FROM.

    WHERE IT COMES FROM (`scripts/bisect_determinism.py`, four processes each):

        model init                identical
        batch order, whole epoch  identical    <- NOT the DataLoader
        forward loss, step 0      identical
        gradients, step 0         4 processes -> 4 DIFFERENT hashes

    The backward, on the very first step, in the fused attention kernel. With
    the fused SDPA backends disabled the same four processes agree bit for bit.
    PyTorch's mem-efficient attention backward accumulates dQ across split-key
    block groups behind an atomic lock, and float addition is not associative.

    An earlier version of this docstring said epoch 1 was bit-identical and
    divergence began at epoch 2, which is what made this look like accumulated
    drift. That was a ROUNDING ARTIFACT -- the two losses were compared at 4
    decimals. At the 6 the log actually writes they are 0.762393 / 0.762403 /
    0.762397: already apart in the first epoch. Reading the log at full
    precision is what moved the diagnosis to the first backward step.

    warn_only=False IS the fix, and is not a stricter flavour of warn_only=True:
    PyTorch reads `deterministicAlgorithmsWarnOnly()` inside the attention
    backward and takes the NONdeterministic branch when it is true. Measured on
    the real path, fused attention still enabled, four processes -> one hash.
    It costs 5.5%: 126 steps run 54.70s nondeterministic, 57.72s strict, and
    62.97s if you disable the fused backends instead. Keep the fused kernel.

    The price of strict mode is that an op with no deterministic implementation
    RAISES instead of warning. At 21x that is the right trade -- a campaign that
    dies in its first minute costs less than one that finishes unreadable --
    and `scripts/smoke_arms.py` runs every arm, so the gate fires before launch
    rather than at hour 19.
    """
    if seed is None:
        # 🛑 NOT A NO-OP. Returning here skips all seven settings below,
        # including `cudnn.benchmark = False` and
        # `use_deterministic_algorithms(True)`, so the run trains
        # nondeterministically and still writes `status: completed` with
        # plausible metrics. The seed lives in the DIRECTORY name (`seed_3/`),
        # so every scorer groups it correctly and the cell looks like a normal
        # 4-seed cell. `tralo_reseed` -- the RNG-only noise floor that every
        # paired contrast is priced against -- is meaningless if this stream is
        # not controlled. `seed` is in protocol.yml's contract_keys.
        raise ValueError(
            "seed is None. Seeding would be skipped entirely and the run "
            "would be silently nondeterministic while its directory still "
            "says seed_N. Set hyperparams.seed.")
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
        torch.use_deterministic_algorithms(True, warn_only=False)
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

    The determinism keys are here because the 0.0358 noise floor could not be
    diagnosed from the artifacts of the runs that showed it: nothing recorded
    whether the run was strict or warn_only, so "was this measured before or
    after the fix" had to be reconstructed from commit dates. A run that cannot
    say which determinism regime produced it is not comparable to one that can.
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
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only":
            torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
