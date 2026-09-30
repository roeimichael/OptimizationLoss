# Boundary-study scorer arithmetic amendment — 1 October 2026

The immutable MobileNetV3 runner release `4334855f81c70596ff64c08bfa6c884c67e08e8a`
finished all twelve fixed seeds 6401–6412 with exit code zero. The first
complete-block independent scorer, release
`3aef5348d74b3229a77b69f30ca4e2e8f7b43616`, stopped in its label-blind
gate before opening development labels. Its preserved failure receipt is
`C:/Users/roeym/.codex/rebuild-audit-20260922/fmow_boundary_full_score_gate_failure_20260930T2226Z.json`.
The first difference was seed 6409, epoch 6, cap divisor 20, Netherlands:
the recorded PHR dual before the step was `11.25182819366455`, while the
scorer's Python-double projection predicted `11.251827049255372`. The
`1.14440918e-6` gap exceeded the existing `1e-6` continuity tolerance.

This is a scorer arithmetic discrepancy, not a training retry or a change to
the study intervention. The runner computes soft residuals and projected
duals with FP32 tensors. The scorer had converted saved soft counts to Python
doubles, projected in double precision, and carried that reconstructed state
across epochs. A separate read-only FP32 replay from the saved PHR side
probability tensors matched the logged residuals, dual-before and dual-after
vectors exactly for all 144 seed/cap/epoch records in the finished block.

The amended independent scorer still recounts hard/soft calls, residuals,
penalties, boundary decisions, doses, source/data/config and artifact hashes
before label access. For PHR it additionally computes the after-residuals
directly from the saved FP32 side tensor in the runner's reduction and scope
order, projects the multiplier in FP32, and requires bit-exact finite-float
JSON vectors for residual-after, dual-after and next-epoch dual-before. It
rejects JSON booleans even where Python would otherwise compare them equal to
`0.0` or `1.0`. No tolerance is widened. A six-epoch synthetic artifact
replay and small dual/residual mutations exercise the full audit path; the
recorded seed-6409 multiplier sequence is also a regression fixture.

The ViT scorer imports this same boundary audit, so a new scorer-only release
must be used for its finished block as well. The old failed scorer attempt and
all immutable runner artifacts remain preserved. The amended scorer will
write new output filenames and must pass the entire label-blind gate before
either study's development metrics are accepted. No seeds, model weights,
configuration, caps, cohorts or primary contrasts change.
