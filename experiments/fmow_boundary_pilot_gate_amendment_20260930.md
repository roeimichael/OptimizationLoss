# Boundary pilot scorer amendment, 2026-09-30

The immutable `4334855f81c70596ff64c08bfa6c884c67e08e8a` MobileNetV3 pilot
seed 6400 and its same-host reference both exited 0. The first independent
label-blind gate stopped before any development-label score or full-block
dispatch. Its accepted PHR probe logged pooled soft count
`168.03691079160637`, while the saved accepted side probabilities recount to
`168.03689575195312`, an absolute difference of `0.000015039653249004914`.
The gate had required `1e-5` agreement for separate FP32 reductions.

Inspection of all 24 accepted joint/PHR records in the pilot training events
found a maximum logged probe-to-final pooled-count difference of
`0.00001582734148541931`; five exceeded `1e-5`. Hard counts agreed in the
first failing record. The discrepancy is at the scale of one FP32 unit in the
last place near a count of 168. The runner already logs the final side count
separately and the scorer independently recounts its saved probabilities.
This is a scorer tolerance failure by itself; feasibility remains unverified
until the amended gate completes. The initial gate failure and empty stdout receipt
remain preserved in the DSI verification directory; the completed seed roots
are immutable and must not be rerun.

The amended scorer allows `1e-4` absolute replay drift only when comparing
the accepted probe's soft counts with final saved-side counts. This remains
tighter than its existing `1e-3` artifact recount bound. It then **independently
reapplies every acceptance predicate to the final saved-side counts** so a
near-threshold probe cannot pass if replay drift reverses the violation
decision. The hard-count, country partition, accepted radius, dose,
probe-decision arithmetic, PTO identity, source/data hashes, and cost gates
remain. Regression tests accept a `2e-5` replay difference with a stable
decision, reject `1e-3` drift, and reject a `5e-5` replay difference that
reverses feasibility. The scorer change
requires a new immutable source release and a complete rerun of the label-
blind gate before pilot scoring or full-block dispatch.
