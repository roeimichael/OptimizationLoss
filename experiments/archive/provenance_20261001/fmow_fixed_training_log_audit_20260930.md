# Read-only training-log audit of the completed fmow2 fixed-dose study

Checked 2026-09-30 05:31–05:36 UTC from the immutable
`/home/dsi/michaer8/tralo-rebuild/runs/fmow-local-fixed-dose-20260928/full6200_6211`
run root on dsisco02. This reads all 12 `retrain1/events.jsonl` files and
`summary.json` records, without re-running a fit or reading labels. The
independent offline metric result remains
`fmow_local_fixed_dose_result_20260930.md`; this note adds log-level mechanism
checks and does not select a new setting.

## Training trajectory

Every seed 6200–6211 ran six epochs and selected **epoch 1** by the unchanged
early-stop rule. Across the 12 seeds, mean training loss fell from 0.9260 at
epoch 1 to 0.0984 at epoch 6, while mean stop-country loss rose from 1.2302
to 2.1272. This is evidence of a widening train/stop loss gap under this recipe,
not proof of a specific cause or of patient-level generalization. The fixed
ensemble still includes all six PTO snapshots because its window is
`max(1, best_epoch-2)..last_epoch`; every arm shares that window. Do not
post-select a shorter window from these viewed outcomes.

## Constraint-step dose and raw calls

At each cap there were 72 of 72 applied side-step opportunities, six per seed.
The actual full-parameter displacement was 0.1 in every logged step, with no
nonnegative directional derivative among logged active scopes. Yet the step
strongly overshot the pooled raw-call ceiling:

| Pooled cap | Mean raw class-1 calls before | Mean after | Mean soft count before | Mean after | Raw-call range after |
| --- | ---: | ---: | ---: | ---: | ---: |
| G=167 | 238.54 | 12.03 | 233.20 | 15.01 | 0–36 |
| G=83 | 238.54 | 13.78 | 233.20 | 16.15 | 1–34 |

These figures summarize the saved **training logs**, not post-hoc allocated
predictions. Some local raw scopes remained infeasible despite pooled collapse,
as separately recorded in the completed result. The common final allocator
still enforces pooled and country quotas for every arm. Therefore raw-count
overshoot is a mechanistic warning, not by itself the measured cause of the
cc-F1 loss. The independent scorer found more correct exits than entries in
the selected class-1 set (443 versus 244 at G=167; 347 versus 222 at G=83),
which directly supports adverse allocated-ranking changes at this fixed dose.

## Implication and limits

The 0.1 L2 radius was a predeclared stress dose, and these logs show it
typically suppresses far more raw class-1 calls than either pooled cap requires.
The completed negative result is evidence against this particular fixed-dose
direction, not against every local constraint objective. A smaller or adaptive
dose would be a **new exploratory method**, not a correction to these runs.
It needs a separately fixed, label-free rule and independent evaluation; the
repeatedly viewed fmow2 development countries cannot confirm it. The draft
PHR-ALM direction comparison uses the same 0.1 maximum dose to isolate the
direction, so it should be interpreted as a stress-dose mechanism comparison,
not a search for a deployable optimum.
