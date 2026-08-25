"""Project-wide numeric constants. Single source of truth for sentinels.

Importing site:
    from src.utils.constants import UNLIMITED

Previously these values were redeclared in 7+ modules with the dangerous
quirk that `metrics.py` declared a local UNLIMITED=1e9 while everyone
else used 1e10 -- meaning a constraint set to UNLIMITED was correctly
skipped by the loss but incorrectly registered as ACTIVE by the metric
layer. The audit that found it (2026-04-26 S3) did not survive the doc
purge, so the record is here and nowhere else; do not replace this with
a pointer.

⚠️ The same mismatch came back as bare `1e9` literals in four analysis
scripts and was removed again 2026-08-23. Import the sentinel; never
re-derive the threshold.
"""

# Sentinel for "no cap on this class". Any K >= UNLIMITED is treated as
# unconstrained throughout the codebase.
UNLIMITED = 1e10

# Numerical safety floor for logarithms / divisions.
EPSILON = 1e-8


def clamp_probability(p):
    """Clamp `p` into the OPEN unit interval, in the dtype it actually lives in.

    `p.clamp(EPSILON, 1 - EPSILON)` reads as a safety floor and is not one at
    the top: EPSILON is 1e-8, float32's own epsilon is 1.19e-7, so
    `1.0 - EPSILON` rounds to exactly 1.0 -- and float16 (eps 9.8e-4) and
    bfloat16 (eps 7.8e-3) are nowhere near. The lower bound is equally dead in
    float16, where 1e-8 sits below the smallest subnormal and rounds to 0.

    It cost a campaign. `soft_count_mode: uniform` builds the log-odds
    `log p - log1p(-p)`; with p still exactly 1.0 that is +inf, the
    straight-through term is `inf - inf` = NaN, `finish_constraint_step` drops
    the update, and the run writes `status: completed` anyway. Measured on
    `results/uniform1` 2026-08-25: `tralo_uniform` landed **1 of 29** steps
    while `tralo` and `tralo_head` -- same everything, `soft_count_mode: sum`,
    no logarithm -- landed 29 of 29.

    So take the epsilon from the tensor. `torch.finfo(dtype).eps` is by
    definition the smallest step representable next to 1.0, so `1 - eps` is
    always a real number strictly below 1 in that dtype, and `eps` is always
    strictly above 0.

    ⚠️ This makes the bound COARSE in low precision, and that is the honest
    reading rather than a side effect: at bfloat16 the log-odds saturate at
    +-4.85 no matter how confident the model is. If an arm depends on the
    log-odds it should run under `constraint_fp32: true`, where the same
    quantity reaches +-15.9. The clamp stops the NaN; only fp32 gives the arm
    its resolution back.
    """
    import torch

    eps = max(EPSILON, float(torch.finfo(p.dtype).eps))
    return p.clamp(eps, 1.0 - eps)

# Default rows per NO-GRAD forward pass over the test set when a config does
# not set `inference_chunk_size`. Lived in tralo/train.py, where danits_lp
# could not see it and hardcoded its own 256 instead.
#
# NAMED FOR THE KNOB IT ACTUALLY BACKS. It was `CONSTRAINT_CHUNK_SIZE` until
# 2026-08-22, which was the name of a DIFFERENT knob: `constraint_chunk_size`
# bounds a gradient-carrying BACKWARD pass and sits at 128 because 256 OOMs on
# ViTB16 under `constraint_fp32`. Nothing ever read this constant as that
# knob's default -- the four constraint-phase arms use `_required`, which has
# no default at all -- so the old name only created the risk of "fixing" it to
# 128 and silently halving the allocators' inference chunk. The lowercase gate
# in tests could not see it: it compared lowercase key names and this is
# uppercase.
#
# It must equal `chunked.inference_chunk_size` in configs/protocol.yml, and a
# test asserts that. `clip` and `focal_clip` do not carry the `chunked` block,
# so they fall back to THIS value while every other allocator reads the
# protocol's -- if the two drift, one knob has two values depending on the arm.
INFERENCE_CHUNK_SIZE = 256
