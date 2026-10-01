# ViT pilot gate event-stream correction — 1 October 2026

The immutable ViT runner release `0f12bde246ccfabc64d549046550e9bd44cfda9a`
completed both fixed seed-6600 jobs with exit code 0. The step-on ran for
1,746.0344576407224 seconds and its matched step-off reference for
429.5516574680805 seconds on dsisco02 GPU0 in FP32. Both top-level
`events.jsonl` files contain one successful `completed` event; each nested
`retrain1/events.jsonl` ends with `training_completed`, as specified by the
runner. The first independent label-blind gate falsely stopped before
scoring with `expected one completed event, got 0`. Its failure receipt is
`C:/Users/roeym/.codex/rebuild-audit-20260922/vit_v2_pilot6600_gate_failure_20260930T2107Z.json`.

The scorer audited the top-level `completed` events in `_receipt`, then
returned the nested training events for epoch and snapshot checks. The gate
mistakenly queried that returned training stream for `completed` when
estimating cost. This amendment changes only that duration lookup to read the
already validated top-level event files, matching the existing MobileNetV3
pilot gate. A regression fixture now keeps training and top-level streams
separate and rejects a missing top-level completion event. The two pilot
roots, runner source, settings, artifacts, results and failed gate receipt
remain untouched. No development labels were read by the failing gate.

Deploy the tested scorer as a **new immutable release**. Re-run the complete
independent label-blind gate against the original two roots. Only if PTO
trajectory equality, source/data/artifact hashes, quotas, gradient/dose and
allocator checks, and projected total at most 24 GPU-hours including the
0.5-hour failed-attempt reserve all pass may the separate exploratory pilot
score be computed and the registered full seeds 6601–6612 be dispatched once.
The pilot's development score cannot select settings or cancel a passing
fixed block. Preserve every negative or failed diagnostic.
