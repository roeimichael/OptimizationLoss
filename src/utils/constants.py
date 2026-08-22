"""Project-wide numeric constants. Single source of truth for sentinels.

Importing site:
    from src.utils.constants import UNLIMITED

Previously these values were redeclared in 7+ modules with the dangerous
quirk that `metrics.py` declared a local UNLIMITED=1e9 while everyone
else used 1e10 -- meaning a constraint set to UNLIMITED was correctly
skipped by the loss but incorrectly registered as ACTIVE by the metric
layer. See docs/AUDIT_FINDINGS_2026-04-26.md S3.
"""

# Sentinel for "no cap on this class". Any K >= UNLIMITED is treated as
# unconstrained throughout the codebase.
UNLIMITED = 1e10

# Numerical safety floor for logarithms / divisions.
EPSILON = 1e-8

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
