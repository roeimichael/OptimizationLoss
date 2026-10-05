# ISIC 2024: one prospective protocol for review

This is a concrete **proposal**, not a frozen experiment or permission to launch.
It extends the existing image-only classification question: does training with
pooled and metadata-local upper bounds improve classification after the same
allocator, compared with matched controls? Metadata enters the constraints and
allocator. It does not enter the classifier. The user's separate question about
metadata as classifier input remains pending.

## Data and scientific interpretation

Use the complete official SLICE-3D noncommercial training archive already
staged on dsisco02, without changing to the permissive subset, dropping weak
labels, balancing classes, or searching alternative cohorts. The
[official dataset listing](https://challenge.isic-archive.com/data/) defines its
binary malignancy ground truth. The proposed target uses that official binary
definition. It is not a melanoma-only target or a new diagnosis mapping.
The [dataset descriptor](https://doi.org/10.1038/s41597-024-03743-w) distinguishes
biopsy-associated labels from clinically assumed benign labels. Retain that
uncertainty; an allocation benchmark is not proof of clinical utility or fairness.

The archive contains 401,059 images and 1,042 patient identifiers. Its SHA-256 is
`e99fd71ff396f8e5ba2d3a1a2d91ee2240bea6351f1a4a09b0d92451cd06cf39`.
The completed CPU audit verified every JPEG CRC and full RGB decode, exact
metadata/image alignment and zero byte/pixel exact duplicates. Near-duplicates
and semantic overlap with other datasets remain unverified. Do not repeat the
full decode audit or treat its pass as a classification result.

Retain the single existing patient-hash candidate, with namespace
`tralo-isic2024-patient-split-20261005`. The first eight SHA-256 bytes of
`namespace|patient_id`, interpreted big-endian modulo 10,000, assign train below
7,000, stop below 8,500, and development otherwise. This candidate has zero
patient overlap. No alternative split or namespace has been tried for this
proposal. Its cohort sizes are:

| Partition | Patients | Images |
| --- | ---: | ---: |
| Train | 689 | 275,295 |
| Stop | 184 | 63,387 |
| Development | 169 | 62,377 |

Recorded sex defines female and male local groups. Missing sex remains in the
pooled constraint, with no separate local constraint. No demographic feature
or diagnostic label may be inferred from the image to fill missing metadata.
The development metadata counts are female 20,673, male 39,735, missing 1,969.
Both locally capped groups exceed the unchanged 1,000-image support threshold.
Partitioned diagnosis counts and model development outcomes were not inspected.

## One proposed quota policy, checked without diagnosis labels

Reuse the existing ISIC2020 capacity rates as an explicit proposal: pooled
1%/2%, female and male each 1.5%/3%, rounded upward to integers. These rates are
not selected from ISIC2024 prevalence, model outcomes or an optimization over
cap choices. Their purpose is comparability with the existing scarce-capacity
benchmark. They are not measured clinical referral capacities.

| Proposed level | Pooled cap | Female cap | Male cap |
| --- | ---: | ---: | ---: |
| 1 | 624 | 311 | 597 |
| 2 | 1,248 | 621 | 1,193 |

The authentic existing quota function and an independent integer ceiling
calculation agree. Each local cap is smaller than the pooled cap, while the
two local caps sum to more than it. This verifies coupled quota geometry.
It does not establish target support, a binding correction, useful dose or
quality. The 1,969 missing-sex samples can compete under the pooled cap and
have no local ceiling; this is a disclosed policy limitation. The code still
rejects `isic2024` as an unregistered quota dataset. No policy was silently added.

The exclusive receipt is
`C:/Users/roeym/.codex/rebuild-audit-20260922/isic2024_prospective_quota_geometry_receipt_20261005.json`.
Its source records the candidate and quota-module hashes, diagnosis-label
access false and GPU use false. The existing candidate receipt and full decode
receipt are in the same audit directory.

## Adapter contract and tests to complete before training

Preparation must produce a frozen manifest and three authenticated runner row
files. Train and stop rows contain sample ID, image filename, group and binary
label. Development runner rows contain only sample ID, image filename and group.
A separate scorer-only artifact holds development labels; the runner manifest
must not reference it or expose development positive counts.

The adapter must authenticate official source hashes, join unique image IDs,
route whole patients into exactly one partition, reject malformed or unexpected
labels and groups, and recount every public support and proposed quota.
The classifier receives RGB pixels only. Preserve original images, patient
identity audit evidence, decode policy and every failed preparation attempt.
Do not adapt the split, target, thresholds or cap levels to diagnosis counts.

Before a campaign, test this real adapter's label boundary with scorer-label
access denied, input hash/path tampering, duplicate sample/patient routing and
missing metadata. Validate real-image task gradients, null neutrality, active
and slack pooled/local corrections, parameter displacement and fixed-weight
replay. Logging must distinguish attempted/applied/skipped updates, nonfinite
events, losses, soft/hard residuals, actual dose, checkpoint identity and source.
The completed eight-image CPU probe validates existing gradient infrastructure;
it does not validate an unimplemented adapter or GPU throughput.

## Bounded experiment and evidence boundary

The proposed first backbone is the existing hash-pinned ImageNet MobileNetV3,
with matched PTO, dose-matched sham, TraLO and PHR conditions at both cap levels.
Use a label-blind cost/dose preflight, then one exclusive pipeline pilot. Freeze
the final schedule, dose and source before any fixed comparison seeds. Do not
assign or claim seed identities before the protocol and dispatch gates pass.
All arms use the same pools, stopping rule, initialization and allocator.
No dose may be selected by development quality.

A fixed four-seed comparison would be a preliminary development result.
Score only its complete gated block, once, reporting both levels and every
negative. Four seeds are not a powered, external confirmation of a broad paper
claim. A subsequent confirmation needs a separately declared hypothesis,
adequate precision and untouched evaluation; sealed Chen and reserved fMoW
remain outside this proposal. Independent sampling units are patients, not
401,059 independent lesions, and repeated-seed uncertainty is not patient
generalization uncertainty.

This cohort is much larger than ISIC2020. Existing decoder speed and CPU
gradient checks do not demonstrate that its full recipe fits a six-hour job,
24-hour cell, or the finite global ceiling. Measure actual GPU task, stop,
pool replay and correction costs before projecting the pilot plus fixed block.
Do not silently subsample, shorten the recipe or enlarge compute to make it fit.
If that projection fails, preserve it as a negative feasibility gate and bring
one concrete amended design to review.

## Compute account and next decision

The latest preserved partial account is
`global_gpu_budget_gate_addendum_20261005.json`. It adds 0.431488531 GPU-hours
for eight independent fixed-seed gates and the failed cached-ISIC6830 re-audit,
without counting their training jobs twice. Its charged subtotal is
72.778787696 hours, with at most 23.221212304 hours of mathematical planning
headroom under 96. The spendable balance remains uncertified.

Original dates recovered for all 20 ViT-v2 receipts agree with their previous
7.225222774-hour duration subtotal and have no mutual overlap. Of that component,
3.013299263 hours precede 1 October UTC and 4.211923512 follow it. The ceiling's
protocol does not specify an exact UTC start. This diagnostic **credits no
hours back**: retain the whole component as a conservative planning charge until
scope is resolved. Do not call folder dates or a duration-only subtotal proof
of weekend-scoped consumption. Evidence is in
`vit_v2_original_timing_receipts_20261005.json` and
`vit_v2_budget_calendar_diagnostic_20261005.json`. Eleven remaining cost gaps,
including scoring/replays and unreceipted attempts, remain explicitly recorded
in `gpu_accounting_remaining_gaps_20261005.json`; none is assigned zero cost.

The review decision is whether to adopt this official binary target, existing
patient-hash candidate, recorded-sex groups, missing-sex policy and two explicit
capacity levels for an image-only pilot. This proposal does not approve GPU
spending or classifier metadata inputs. Even after the scientific decision,
adapter/source/label-boundary, logging, real GPU cost/dose and finite-budget gates
must pass, followed by fresh ownership checks on both hosts immediately before
dispatch. Current own GPU cells are complete; no new job was launched here.
