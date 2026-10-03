# Calibrated tabular pilot gate: scorer control-arm correction

The cached ISIC2020/MobileNetV3 calibrated pilot, seed 6890, finished all six
arms and exited 0 at 2026-10-02 07:57:28 UTC under immutable runner release
`034ae537bd6eb0a474eff286ea5225471a260e2c`. Its guarded queue stopped
before claiming seeds 6891–6894 because the independent label-blind gate raised
`KeyError: 'pto'`. The failed gate and queue logs are preserved offhost in
`isic_calibrated_pilot6890_gate_failure_20261002`, alongside a SHA-256 manifest;
the completed pilot's run evidence and checkpoints are separately archived in
`tabular_calibrated_v2_backup/isic2020_6890/seed6890`.

The new calibrated config deliberately assigns step sizes only to the four
treated arms. The runner gives PTO and sham scheduled zero corrections without
looking up a step size. The scorer incorrectly looked up a calibrated step size
for those controls when checking every logged correction. This is a scorer
replay error, not evidence that a trained model or constraint failed. The
scorer now expects exactly zero proposed displacement and zero gradient for
PTO/sham, while it continues to verify the frozen arm-specific step times the
logged parameter-gradient norm for each treated arm. The model runner, data,
configs, original failure evidence, numerical tolerances and label boundary are
unchanged. A regression test covers both zero-step controls and all four
calibrated treated scales.

No development quality was scored. Immutable scorer release
`2619a5d2e0b8bdab0e4eadb6525ba249a1d9d847` was verified byte-for-byte
on both DSI hosts and re-audited the **same** completed seed 6890. The
label-blind gate passed: corrections applied 4/5, 5/5, 2/5 and 3/5 in the four
treated arms. Its projected four-seed cost is 7.123 GPU-hours, or 10.685 with
the registered 1.5 safety factor, below the 24 GPU-hour cell ceiling. The
independent receipt and log are archived offhost in
`reaudit_isic6890_2619a5d2e0b8bdab0e4eadb6525ba249a1d9d847`.

The fixed seeds 6891–6894 remain unclaimed. They require a separately guarded
continuation that authenticates the new gate, uses the original immutable runner,
checks physical GPU ownership before each seed, and never reruns the pilot. The
pilot training logs show nonzero corrections, but they do not establish
predictive benefit. Its treated stop losses often exceeded PTO's; the full
prespecified block, if completed, must report that evidence without selecting
settings from it.

The first continuation script was prepared before the VPN outage. Its 24-hour
cell guard mistakenly subtracted calendar time since the pilot launched. The
registered limit is **24 aggregate GPU-hours**, so an idle day without a GPU job
must not spend that budget. Before any fixed seed was claimed, the guard was
corrected to sum elapsed seconds from successful, seed-matched completion
receipts and reject failed or inconsistent receipts. The separate six-hour
per-job timeout and the 1.5-times pilot cost projection remain in force. This
is scheduling accounting only; no model, data, scorer, tolerance, label access,
or scientific setting changed. The original continuation script release is
preserved and was never deployed to run fixed seeds.
