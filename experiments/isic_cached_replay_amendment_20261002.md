# ISIC 2020 cached-pilot replay correction, 2 October 2026

The completed MobileNetV3 pilot seed 6830 under immutable runner
`c6b7aca395018f2d50af1150b75210faa8e76ca3` failed its independent
label-blind gate **before development labels were opened**. The first scorer
replay differed from the saved probabilities by at most 2.8014e-6, exceeding
the frozen 1e-6 absolute threshold in 96 of 9,590 values. The failed gate and
original checkpoint/prediction bytes remain preserved in
`C:/Users/roeym/.codex/rebuild-audit-20260922/isic_cached_pilot6830_failed_gate_20261002.tar`
(SHA-256 `7a21df8046140a6029d5840b7b34d4ffe0f0e0d48e15494d5f9576e4479afcd0`).
No fixed seed 6831–6834 was claimed, and no development quality was scored.

The cause is a scorer-only batch partition mismatch. The frozen runner's
`BACKBONES["mobilenet_v3_large"]` uses batches of 32 for development-pool
prediction. The old scorer replayed that backbone in batches of 16; both use
the same FP32 model, saved state, image order and exact cached pixels. As an
independent read-only diagnostic, the *same selected checkpoints* were
replayed on dsisco01 GPU1 with the runner's batch size 32, after an exclusive
GPU lock and compute-PID check. All six arms then matched the saved
probabilities **bit-for-bit** (maximum gap 0.0, 0/9,590 values over 1e-6 each).
The six-arm diagnostic is in
`C:/Users/roeym/.codex/rebuild-audit-20260922/diagnose_isic_cached_replay_batch32_20261002.jsonl`.
This directly identifies the batch partition, rather than inferring a cause
from zero hard-decision flips.

The scorer change reads the replay batch size from the frozen runner's
backbone registry. It changes neither model weights nor saved results,
constraint logic, training schedule, caps, seed IDs, private labels, scoring
metrics, or numerical tolerance. It is applicable to the same completed pilot;
there is no reason to retrain it. The new scorer release must pass native
tests, immutable both-host deployment and tracked-byte parity. It may then
run a **new exclusive label-blind gate attempt** against the same pilot,
retaining the failed attempt. A passing gate is a data/model integrity result,
not evidence that TraLO improves classification. Only the originally fixed,
conditional 6831–6834 block may be considered after the gate and measured
cost check; no pilot score may decide that.

## Re-audit outcome

The scorer-only change was committed and deployed as immutable release
`53435a48be183bb34962d0625fdcccb4bcd88672`; both hosts had identical
tracked-file hashes and passed the focused native tests and CLI check. The
exclusive re-audit of the **same** seed 6830 ran on dsisco01 GPU1 and failed
at a later, independent label-blind gate. It did not open development labels.
The cached pilot made **zero applied constraint corrections in all four treated
arms** across five opportunities each. Every attempt was rejected by the
original 0.10 parameter-displacement veto: proposed displacements ranged
0.902–2.220 for level-1 TraLO, 3.750–20.789 for level-1 PHR, 1.391–2.028
for level-2 TraLO, and 0.340–3.082 for level-2 PHR. The PTO and sham arms
correctly made zero scheduled corrections. The pilot took 6,382 seconds;
four analogous seeds would project about 7.09 aggregate GPU-hours *before*
any changed method or contention allowance, but cost feasibility cannot
override an inactive-arm failure.

The new failed-gate launch, completion, log and unchanged pilot summary were
copied with SHA-256 manifest to
`C:/Users/roeym/.codex/rebuild-audit-20260922/isic_cached_pilot6830_reaudit_53435a48`.
The original probability-replay failure is also retained. Fixed seeds
6831–6834 remain **unclaimed and unscored**. Any next pilot must be a new,
prospectively specified dose condition with fresh seed IDs and release,
analytic/real-image gradient checks, matched controls, and a finite ceiling;
it cannot be presented as a continuation of the original six-epoch condition.
