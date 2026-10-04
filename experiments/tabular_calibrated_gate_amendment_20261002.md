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

The first remote `--check-only` of the budget-corrected release exposed a
second script defect: a shell error handler was parsed into the embedded Python
preflight. Python raised `IndentationError`, yet the shell returned a false
`preflight_pass`. No seed or continuation was claimed or launched. That release
and its failed check are preserved. The handler was removed from each embedded
Python block; `set -e` now stops on a Python error. A local test compiles every
embedded Python block, and the corrected remote preflight must reject an
invalid gate and accept the authentic gate before any new seed is claimed.

On 2026-10-03, immutable script release
`4d3d0676209cd854dd8904461aee25e89b9656bd` was deployed byte-identically
to both hosts. On dsisco01, its genuine ISIC `--check-only` passed and a
deliberately invalid copy of the gate failed with a nonzero exit. The guarded
ISIC fixed-seed queue started once at 17:31:46 UTC on the pilot's physical
GPU. Seed 6891 exited zero, passed its separate label-blind gate, and was
backed up offhost; 6892 started next. No development quality has been scored.

The completed CelebA seed 6880 was independently re-audited under the same
corrected scorer on its original physical GPU after that GPU became free.
Its original `KeyError` failure remains preserved. The corrected label-blind
gate passed: all four treated arms applied five of five corrections, and the
four fixed seeds project to 13.516 GPU-hours (20.274 with the frozen 1.5
safety factor), within the 24 aggregate GPU-hour cell ceiling. The gate receipt
was copied offhost with matching SHA-256
`c73c1a2b635c0df0a3572a24aedc3694ed414ffd7386ac2bc7723e053cdf5aa1`.
The authentic CelebA `--check-only` passed, then its guarded fixed-seed queue
started once at 20:15:17 UTC on dsisco02 GPU 2. Seed 6881 was running at the
post-launch inventory. This establishes correction activity and provenance,
not predictive benefit; the pilot's treated stop losses were substantially
worse than PTO's, and no CelebA development labels have been opened.

The ISIC fixed seeds 6891–6894 all exited zero and passed separate label-blind
gates by 2026-10-04 01:00:06 UTC. The entire cell, including checkpoints and
receipts, is backed up offhost. The first complete-block score attempt under
scorer release `2619a5d2e0b8bdab0e4eadb6525ba249a1d9d847` independently
replayed all four seeds, then stopped before opening development labels with
`RuntimeError: completed cell wall exceeds fixed 24-hour ceiling`. That stale
scorer check counted the VPN outage between pilot and fixed seeds as GPU time,
contrary to the registered **24 aggregate GPU-hour** ceiling and the guarded
continuation's measured-receipt accounting. The failed score attempt remains
evidence; it produced no score file. The scorer-only correction sums successful
pilot and fixed queue durations, validates their ownership and provenance,
and rejects a sum above 24 GPU-hours. It leaves model training, data, quotas,
metric definitions, score contrasts and the label boundary unchanged.

Scorer-only release `358bfd3e8925adc7eabc2e1dbc8c53348c091840` passed
11 relevant tests on each DSI host with identical scorer SHA-256
`c34bb759d3a4e2cd97c33e61c22c907da29874517ba24acb3e0928671c2bdf8a`.
It independently replayed the complete fixed ISIC block and scored the
development pool once. The output is archived offhost as
`isic2020_mnv3_calibrated_6891_6894_score_358bfd3e8925adc7eabc2e1dbc8c53348c091840.json`
with SHA-256 `c0e60de9878300e9e14b2352d11982038564e02f8bde111beedb6933e2a13c8c`.
Successful pilot plus four fixed queue jobs used 9.059 GPU-hours by measured
receipts; the four fixed runners used 7.268 GPU-hours. Separate independent
gate/scoring replay time is additional and remains well below the 24 GPU-hour
cell ceiling. All four fixed seed backups are verified offhost.

The prespecified allocated cc-F1 contrasts, averaged over the four fixed
seeds with paired 95% t intervals, are:

| Quota | TraLO minus PTO/sham | TraLO minus PHR |
| --- | ---: | ---: |
| Level 1 | -0.0061 [-0.0707, +0.0586] | +0.0053 [-0.0790, +0.0896] |
| Level 2 | +0.0365 [-0.0496, +0.1225] | +0.1010 [-0.0436, +0.2456] |

No contrast excludes zero. Level 2 points toward a possible benefit but is
imprecise at four seeds; level 1 points slightly against TraLO versus PTO.
This is viewed-development evidence, not a held-out result or grounds to
select a setting and spend more compute on the same labels.
