"""Bars that more than one scorer reads. Nothing else belongs here.

WHY THIS FILE EXISTS (2026-09-05). `MIN_FLOOR_OBS` lived in
`sensitivity_screen`, and when `deployed_h2h` grew the same guard it imported
the constant from there rather than restating it -- correctly, because a second
literal is free to drift from the first.

That import broke `deployed_h2h` in three worktrees. `sensitivity_screen`
imports `src.training.constraints`, and a campaign worktree is PINNED at the
commit its configs were generated from, so its `src/` can predate a name the
scorer needs (`cap_fraction_for` on `optloss-domb`). `src/` is frozen while a
campaign runs and MUST NOT be updated to fix it.

So the rule this file encodes: **a scorer that has to run in a pinned worktree
may not import anything that reaches `src/`.** `deployed_h2h`, `quarantine` and
`pred_integrity` are in that class -- they are the tools you reach for when
deciding whether a number may be quoted, and they have to work in every
checkout, at every commit. A shared bar therefore lives somewhere that drags
nothing with it.

`tests/test_lessons_learned.py` gates it: those scorers must import with `src`
made unavailable.
"""

import re

# THE FLOOR NEEDS OBSERVATIONS BEFORE IT IS A FLOOR. The spread between arms is
# estimated from every arm PAIR in a cell; the RNG floor comes from the
# `_null`/`_reseed` pairs only, and most campaigns here carry exactly ONE such
# pair -- so at 4 seeds the floor is a median of FOUR numbers whose
# order-statistic confidence interval is the entire sample range. Comparing a
# well-estimated median against a badly-estimated one certifies cells that are
# pure noise.
#
# Measured live on 2026-09-05: at ONE completed seed the floor came back 0.0,
# and every spread clears zero. `deployed_h2h` named a #1 off one seed until it
# grew this guard.
#
# Below this many observations the honest verdict is that the floor is
# unmeasured, not that the spread beat it.
MIN_FLOOR_OBS = 8

# WHICH ARMS ARE lambda=0 RNG STREAMS. Second thing to live here, for the same
# reason as the first, and it was already drifting.
#
# `<fam>_null`, `<fam>_reseed`, `<fam>_reseed2`, ... all carry `lambda_step: 0.0`
# via the `<fam>_null` block and differ only in the RNG draw, so ANY TWO of them
# bound the same family's RNG-only noise and every within-family PAIR is a floor
# observation.
#
# 🛑 THE DRIFT, found 2026-09-07. `deployed_h2h` was fixed on 2026-09-06 to read
# every stream; `sensitivity_screen` kept its own test, `arm.endswith("_reseed")`,
# which is **False for `tralo_reseed2`**. Two consequences, both silent and both
# in the same direction -- toward calling noise a result:
#   * the third stream was NOT excluded from the cross-arm SPREAD, so a pure-RNG
#     arm widened the spread it is supposed to define the floor for;
#   * the floor itself paired only `<fam>_null` with `<fam>_reseed`, so of the 12
#     observations a three-stream family yields it saw 4, and 4 < MIN_FLOOR_OBS.
# `sensitivity_screen.classify` prints, in its own UNDER-POWERED message, the
# advice to buy observations with `tralo_reseed2` -- and could not read the arm
# it was recommending. `dualprop1` is the first campaign that took that advice.
_STREAM = re.compile(r"^(?P<fam>.+?)_(?:null|reseed\d*)$")


def stream_family(arm):
    """The family `arm` is a lambda=0 RNG stream OF, or None if it is not one.

    ⚠️ `<fam>_lam0` is NOT a stream and must not match: it keeps `lambda_step`
    and takes real constraint steps.
    """
    m = _STREAM.match(arm)
    return m.group("fam") if m else None


def is_lambda0_stream(arm):
    """Is `arm` any family's lambda=0 RNG stream?"""
    return stream_family(arm) is not None


def stream_pairs(arms):
    """Every within-family pair of lambda=0 streams, as (a, b) tuples.

    A family holding ONE stream yields C(1,2) = 0 pairs and contributes nothing
    -- which is the whole point, and is why the COUNT of streams is not the
    count of observations.
    """
    fams = {}
    for a in sorted(arms):
        fam = stream_family(a)
        if fam is not None:
            fams.setdefault(fam, []).append(a)
    out = []
    for fam in sorted(fams):
        got = fams[fam]
        for i, a in enumerate(got):
            for b in got[i + 1:]:
                out.append((a, b))
    return out

