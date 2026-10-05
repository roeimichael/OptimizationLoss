# Prospective knee pilot uncertainty correction

Before any scientific seed claim or model-quality evaluation in the approved
four-seed pilot, source review found that the new scorer reported a point
interval and p=0 for four identical nonzero seed deltas. This conflicts with
`docs/FRAMEWORK.md`, which requires an unavailable interval when empirical
variance is zero. The error was reproduced with fictitious positive, negative
and zero deltas; both original failing JUnit records are preserved offhost.

The scorer now retains every delta, its mean and seed SD, and reports a null
interval and null p-value with an explicit reason for zero variance. A valid
nonzero-variance contrast retains the prespecified paired Student t(3) method.
Holm retains all ten declared contrasts in its family even when some tests are
unavailable; unavailable tests never acquire a reported adjusted p-value.
No outcome-dependent replacement test or inference method is introduced.

Eleven new focused cases cover the zero-variance boundary, small nonzero
variance, closed-form t(3) arithmetic, full-family Holm adjustment and invalid
inputs. Together with the thirteen directly affected scorer checks, all 24
passed locally. The prior scorer fixture's incorrect zero-variance expectations
were corrected and its complete-panel assertion now checks unavailable
inference explicitly. This strengthens the protocol boundary; it does not
relax a launch gate. Both-host validation and source-byte receipts belong to
the new immutable release, separately from the completed earlier suites.

This is a software correction using fictitious targets. No actual development
target, prior scientific score, completed model or scientific seed was accessed
or replayed. Training, groups, caps, recipe, controls and the ten contrasts are
unchanged. The scientific CLIs remain closed pending real campaign gates and
certified budget room. The full CPU baseline remains at cc21dd1c, not this
focused release, and the two previously unexecuted CUDA checks remain open.
