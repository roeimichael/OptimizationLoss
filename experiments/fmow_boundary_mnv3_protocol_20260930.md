# Fixed exploratory protocol: boundary-calibrated local TraLO on fmow2 MobileNetV3

Status: fixed design and CPU implementation in preparation, **no GPU result**.
This is a new policy motivated by the preserved negative 0.1-dose studies, not
a retuning of those releases. The five development countries have been viewed
repeatedly; every outcome from this block is exploratory. The five reserved
countries and the Chen knee test remain unavailable for this study.

## Question, intervention, and controls

Does the label-free boundary rule in
[`fmow_local_boundary_policy_design_20260930.md`](fmow_local_boundary_policy_design_20260930.md)
retain class-1 ranking while decreasing pooled-plus-country positive soft
violations? The expected mechanism signature is a substantially smaller
realized step than 0.1 when a full step would collapse raw calls, with logged
nonworsening positive soft violations and preserved raw pooled-call floor. A
smaller dose alone is not a quality result. A zero step or lower cc-F1 is a
valid negative outcome.

Use the established Yuval-pipeline fmow2 MobileNetV3-Large ImageNet-V2 recipe:
FP32, full-frame RGB 224, existing ImageNet normalization, balanced training,
the existing stop-country early stopping and unchanged snapshot window. The
PTO optimizer, supervised batches and checkpoint state are identical for all
arms within a seed. At each PTO snapshot, make side copies only, evaluate the
same unlabeled 1,673-image development pool, and use two pooled caps 167 and
83. Recount Hamilton size-share country upper bounds (totals 209 and 104)
from country IDs without labels. All arms deploy via the unchanged
`local_capped_first` allocator. Preserve the full original PTO snapshots.

The five arms at each cap are:

| Arm | Side-copy rule |
| --- | --- |
| PTO / Clipper analogue | Zero step, same ensemble and local allocator. This is not the historical `clip` or `focal_clip` implementation. |
| Calibrated joint TraLO | Hard-active pooled-plus-country direction, boundary radius chosen solely by soft/hard unlabeled probes. |
| Pooled-only | Pooled constraint direction at the joint arm's realized radius; skip if joint skips. |
| Sham/null | Random direction with the joint arm's realized per-tensor displacement norms; skip if joint skips. |
| Calibrated PHR | PHR direction with fixed rho 0.5 and the same boundary acceptance rule; independent radius/activation and logged dual continuity. This is a snapshot-policy comparison, not full ALM training. |

For a nonzero joint step, record scope gradients and normalized directional
derivatives, every probe, accepted radius, actual parameter and per-tensor
displacement, pooled/country soft and hard calls before and after, and exact
side-copy probability hashes. If no radius passes, apply zero and keep an
explicit rejection record. Do not alter the main model, BatchNorm buffers,
RNG, training optimizer, stop rule or snapshot selection. If the PHR policy
rejects, its dual still advances under the registered PHR update; its side
probabilities must equal PTO. The final allocator, rather than the raw step,
enforces exact pooled and local hard caps.

## Frozen jobs, inference and cost

Pilot seed **6400** has one step-on job and one same-host, same-precision
step-off PTO reference in exclusive, non-reused roots. Only after a label-blind
pilot gate may its development score be independently recomputed; pilot scores
cannot change the policy or stop a prespecified full block. The full block is
the twelve fresh seeds **6401–6412**, each one training trajectory and both
caps. No previous seed or artifact is a replicate. The maximum allowed
projected cost, measured on the pilot host including pilot and reference, is
**8 GPU-hours**. A pilot over budget stops before the full block. Dispatch at
most three genuinely free, exclusively claimed GPUs after both-host ownership
inventory; a low utilization reading with a compute PID is occupied. A lost
SSH connection or vanished queue requires forensic inspection before any
continuation, never a duplicate run.

The primary metric is the fixed-class, actually deployed **allocated class-1
cc-F1**. For each cap, the three fixed seed-paired contrasts are joint minus
PTO, joint minus sham, and joint minus PHR. Analyze all six together with
two-sided paired Student-t 95% intervals and six-test Holm correction; show
every seed, mean and SD. A promising exploratory lead requires joint above
both PTO and sham with positive adjusted paired evidence at a cap and no
wholly negative paired interval for accuracy, macro-F1 or weighted-F1. A
numerical increase without those conditions is a tradeoff or inconclusive.
Report both caps and all arms regardless of direction. Joint-minus-PHR is
always reported, but a PHR loss cannot substitute for beating PTO and sham.
Add constrained precision/recall, class supports, feasibility, slot turnover
with correct entries/exits, activation/skip rate and measured compute. Do not
choose a cap, epoch, country, seed, radius, method or backbone after reading
the development metrics. The 12 seed pairs quantify algorithmic variability
conditional on the inspected pool, not independent geographic generalization.

## Gates before a dispatch

1. Test hand-computed scalar policy cases, conflicting scopes, upper-bound
   quota sums greater than the pooled cap, nonfinite inputs, hard-count jumps,
   nonmonotone probes and no-step fallback. Mutate a source calculation and
   verify the corresponding gate fails.
2. On real unlabeled priors, compare autograd directional derivatives with
   central finite differences; verify chunked/full inference and gradient
   parity, actual parameter displacement, all active and inactive scopes,
   exact accepted-probe replay, BN/RNG neutrality and zero-step PTO identity.
   Check the sham's per-tensor displacement and pooled-only realized radius.
3. Revalidate actual data arrays, image/label alignment, split and country
   identities, exact byte duplicates, class supports using the offline audit,
   fixed preprocessing, pinned ImageNet weight bytes and source/config hashes.
   The run loader receives no development labels; the independent scorer reads
   labels only after artifacts and label-blind gates pass. Recount country
   quotas independently.
4. Commit tested source/configs/scorer, push GitHub and DSI bare mirror, make
   a new immutable detached release and verify tracked-byte parity, native
   tests, CLI, data hashes and GPU arithmetic smoke on the intended host. Do
   not edit prior releases or resume an existing seed root.
5. The pilot must prove complete source/data/artifact identity, same-host PTO
   trajectory equality, correct applied/skipped/dual records, probability and
   allocator recount, no label access and projected cost at or below 8 hours.
   A gate failure is preserved and stops expansion. Only then dispatch all 12
   seeds and independently score the complete block once.

The old fixed-dose and PHR results are planning evidence only. No fmow2
development result from this new policy can validate a ViT or independent
dataset claim. ViT-B/16 requires its own fixed backbone recipe, weight hash,
data/gradient preflight, same-host memory and runtime pilot, and separately
declared ceiling before a complete matched block.
