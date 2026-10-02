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

No development quality was scored. A new immutable scorer release must re-audit
the **same** completed seed 6890. The fixed block remains unclaimed until an
independent label-blind gate passes and its measured cost is under the cell
ceiling. The pilot training logs already show nonzero corrections in all four
treated arms, but they do not establish predictive benefit.
