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

# Default rows per forward pass over the test set when a config does not
# set `constraint_chunk_size`. Lived in tralo/train.py, where danits_lp
# could not see it and hardcoded its own 256 instead.
CONSTRAINT_CHUNK_SIZE = 256
