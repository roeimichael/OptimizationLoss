# Approved knee pilot: public-input and snapshot preparation stage

The user approved the four-seed knee comparison in
`knee_snapshot_local_attribution_20261005.md` and broader overnight experimentation
and readiness simulations. That approval is recorded; another approval of the
same groups, limits or execution scope is not required.

This change adds the neutral input preparation interface and the six-arm
snapshot comparison. It reuses `knee_yuval.train_run`, the native global step,
the joint local step and both existing allocators. Historical runners and their
seed validators are unchanged. There is deliberately no production campaign
launcher in this stage.

`analysis.prepare_knee_snapshot_local` handles the authorized train/development
source roles. It never enumerates the source root or test tree, checks IDs and
train/development subject and exact RGB overlap, and preserves the existing
training-only stopping carve. It copies native RGB pixels into neutral PNGs,
stripping original metadata. Development row ordering uses sample IDs, never
grade directories. Original source hashes could fingerprint grade-bearing PNG
metadata, so they also stay in the separate trusted/private artifact. Public
rows expose IDs, subject hash groups, neutral paths and public image hashes;
development targets and original grade paths are absent. No private artifact
path/hash is referenced by the public manifest. This logical boundary does not
certify OS access controls or semantically independent patients.

`tralo.knee_snapshot_local.snapshot` saves PTO, native global, native global sham,
joint local, pooled-at-joint-radius and joint sham predictions from isolated
copies. It checks activation, radius and actual displacement parity; per-tensor
sham doses; original model/gradient/mode identity; and RNG restoration even on
failure. Every started/completed/failed arm is exposed to the caller's event
sink. All arms use the same authenticated ensemble window and can enter either
common allocator. Infeasible search exceptions remain failures, not silently
relaxed updates. Logging includes counts, residuals, gradients, displacements,
activation and joint/pooled displacement alignment. The module does not claim
to reproduce a clinical local-constraint benefit.

Local verification on the final source passed 22 new focused CPU tests. These
include hostile rehashed row/path/quota changes, private/test access denial,
label-independent development serialization, active and inactive six-arm
corrections, sham dose checks, a parameter finite-difference/full-objective
check of the streamed joint direction, and complete three-epoch CPU supervised
trajectory parity with and without side corrections. The initial local CPU
rehearsal used the earlier public source-hash interface; review then moved that
hash into the private artifact, and both the focused tests and the rehearsal
were rerun against the final interface. Earlier artifacts remain preserved.

`tools/knee_snapshot_cpu_readiness.py` performs a separate fictitious RGB
input-to-ensemble rehearsal using the actual image readers and Yuval transforms,
denies all original/private reads after preparation, checks identical supervised
fits and both common allocation policies, and performs one unfrozen CPU
MobileNetV3 task backward without pretrained weights. Its explicitly shortened
three-epoch simulation is not the scientific 75-epoch recipe. The final local
rehearsal passed; its MobileNet task gradient norm was 6.43687747199322. All
images, labels and IDs were fictitious; no scientific seed was claimed and no
CUDA context initialized. CPU timing and gradients do not certify GPU cost,
pretrained initialization, scientific dose, real data or a positive quality
effect. Server deployment/validation is recorded separately in immutable
off-host receipts after commit; this document does not predict their outcome.

Before scientific runs, finish the production training/event/scoring wrapper,
actual authorized source and public/private reader integration, independently
checked evaluation arithmetic, real-data derivative/dose/logging gates,
pretrained bytes, prospective seed freshness and exclusive claims, physical GPU
ownership and full-recipe cost certification. The current partial accounting
charge is 75.158211024 GPU hours, with certified remaining balance still null.
That unresolved budget coverage prevents GPU dispatch. The existing overnight
heartbeat continues implementation and validation within the approved scope.
No prior result, completed scientific run, source release or sealed evaluation
was replayed or overwritten by this preparation stage.
