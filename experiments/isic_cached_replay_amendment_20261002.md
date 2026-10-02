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
