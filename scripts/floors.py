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
