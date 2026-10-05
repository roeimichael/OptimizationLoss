# Broader image and metadata research, 5 October 2026

The user requested broader image datasets with a tabular local constraint and
continued testing today. The working interpretation is the existing image-only
classifier with metadata supplied separately to pooled/local constraints and
the common deployment allocator. Feeding metadata into the classifier would be
a separate model condition; the clarification is pending. No new full GPU
campaign was launched by this checkpoint.

## Completed comparisons retained

| Dataset and metadata | Completed finding | Interpretation |
| --- | --- | --- |
| fMoW, country | Cap-10 TraLO minus focal clipper cc-F1 -0.01393 [-0.01964,-0.00821] | Exploratory negative |
| ISIC 2020, sex | Calibrated MobileNetV3 levels 1/2 minus PTO allocated cc-F1 -0.0061 [-0.0707,+0.0586] / +0.0365 [-0.0496,+0.1225] | No demonstrated benefit |
| CelebA, Male attribute | Levels 1/2 minus PTO +0.000045 [-0.000140,+0.000230] / +0.000079 [-0.000951,+0.001110]; level-2 minus PHR exactly zero | No meaningful benefit |

These complete blocks were already scored and backed up. This checkpoint does
not rerun or re-score them, add seeds, select doses from their viewed outcomes,
or open the Chen knee test or reserved fMoW countries. The knee snapshot finding
and its distinct claim boundary remain in
[the paper-readiness audit](paper_claim_readiness_20261004.md).

## ISIC 2024 source and metadata gates advanced

Official source: [ISIC challenge data](https://challenge.isic-archive.com/data/).
The [primary dataset descriptor](https://doi.org/10.1038/s41597-024-03743-w)
distinguishes pathology-associated strong labels from clinically assumed benign
weak labels. That limitation must accompany any future result. Acquisition of
the complete noncommercial archive is already recorded; this checkpoint does
not substitute a different license subset.

On dsisco02 the staged archive at
`/tmp/tralo-isic2024-michaer8-20261001/ISIC_2024_Training_Input.zip` has SHA-256
`e99fd71ff396f8e5ba2d3a1a2d91ee2240bea6351f1a4a09b0d92451cd06cf39`.
The CPU audit read **all 401,059 JPEG members**, checked their CRCs, fully decoded
RGB pixels, and joined each unique image ID to the archive's embedded metadata.
All 1,042 patient IDs were present. There were **zero byte-identical JPEG pairs
and zero size-aware decoded-RGB duplicate pairs**, including across patients.
Semantic, resized, and re-encoded near-duplicates remain unverified.

The embedded table is `ISIC_2024_Training_Input/metadata.csv`, SHA-256
`b8efbcdc24391148e47cb01d3e141d60d0cbb356d53101a28b5904e5527bf569`.
An initial audit incorrectly expected patient IDs in the supplementary diagnosis
table and failed before decoding. Its source, launch and traceback were retained
off-host in `isic2024_failed_decode_preserved_20261005.json`; the failed server
directory was not changed. The corrected audit used a new exclusive directory
`audit_full_decode_20261005T060109Z`, source SHA-256
`930ee0cf95b0f0e3bb9e11261b82280a7722d6cef0cfef293164233d6317a506`.
It completed in about 66 seconds, with no CUDA or diagnosis-label access.
The byte-identical off-host receipt is
`C:/Users/roeym/.codex/rebuild-audit-20260922/isic2024_full_decode_receipt_20261005.json`,
SHA-256 `97a0bf5a6f052aecacaa1e3502e90cd36349a6c442518c19298a5f1dc4abe2af`.

One prospective patient-hash 70/15/15 diagnostic used the existing first-eight
SHA-256 bytes, big-endian modulo 10,000 rule and namespace
`tralo-isic2024-patient-split-20261005`. It did not read diagnosis labels or try
alternative namespaces. It produced:

| Candidate partition | Patients | Images | Female | Male | Missing sex |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 689 | 275,295 | 86,339 | 182,334 | 6,622 |
| Stop | 184 | 63,387 | 16,984 | 43,477 | 2,926 |
| Development | 169 | 62,377 | 20,673 | 39,735 | 1,969 |

Patient overlap is zero. Female and male candidate development groups clear the
unchanged 1,000-image support threshold. This is a **candidate**, not a frozen
cohort or cap approval. Target support in those partitions was not inspected.
The exclusive receipt is `isic2024_patient_candidate_receipt_20261005.json` in
the same off-host audit directory.

## Real-image gradient and source checks

Immutable code release `5f74ce387a48cb22e69742c3d8db71a4ead95c62` was clean on both
DSI hosts. Committed-byte hashes matched for the relevant preparation, input,
quota, backbone, gradient, runner, scorer and test modules. With CUDA hidden,
**31 focused tests passed on each host**. The existing tensor-to-scalar warning
was retained; tests do not prove scientific quality.

An eight-image ISIC 2024 CPU probe selected the first four image IDs from each
sex group in lexical order, without diagnosis labels, and used the hash-verified
existing ImageNet MobileNetV3 weight
`5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997`.
Double precision was used to compare a direct full-pool scalar objective with
the existing two-pass gradient code. Synthetic binding caps (pooled 2, each
group 1) were infrastructure fixtures, not research referral capacities.

| Probe | Parameter gradient norm | Direct-objective maximum gradient gap | Applied displacement |
| --- | ---: | ---: | ---: |
| TraLO | 28.8772768181 | 9.55e-15 | 0.00288772768 |
| PHR | 37.5455604648 | 7.94e-15 | 0.00375455605 |

Both fixed-weight probability replay gaps were zero. Both bounded corrections
applied. This does not validate GPU throughput, a clinical target, full-cohort
dose, training convergence, an ISIC 2024 adapter or classification quality.
An initial detached validation wrapper had a string-escaping syntax error;
its logs were preserved and a newly compiled wrapper ran in another exclusive
directory. No model experiment or claimed seed was rerun.

Hash-verified off-host evidence is under
`metadata_cpu_validation_dsisco01_20261005T060950Z/` and
`metadata_cpu_validation_dsisco02_20261005T060950Z/` in the audit directory.
The latter includes the real-image gradient receipt and logs.

## Hardware and finite compute

Both-host inventory at 05:51:44 UTC found all four Quadro RTX 6000 GPUs on
dsisco01 unoccupied, with zero memory use and no compute PIDs. All four Blackwell
GPUs on dsisco02 belonged to zehavid or liverty. No other user's GPU was shared
or stopped. This is a snapshot, not a reservation; both hosts must be checked
again immediately before any future GPU dispatch. Today's checks used CPU.

Six additional nonoverlapping timed smoke/gate intervals added 0.171799896 hours
to the previous account. The verified lower bound is now **72.347299165 aggregate
GPU-hours**; at most **23.652700835** remain mathematically under the unchanged
96-hour ceiling. The spendable balance is still **uncertified**. Failed gate
attempts, independent scoring/replays, cost probes and remaining possible roots
cannot be assigned zero cost. Artifact timing surveys from both hosts were
preserved to continue reconciliation. The authoritative addendum is
`global_gpu_budget_addendum_20261005.json`; it does not authorize a new booking.

## Next bounded experiment decision

ISIC 2024 is the next concrete preparation candidate: its patient IDs permit a
patient-disjoint design, its sex metadata supports local groups, and its full
decode and CPU gradient checks now pass. The proposed scientific cell retains
an image-only classifier, malignant-versus-other target, pooled and per-sex
upper bounds, missing sex global-only, identical allocation for every arm, and
matched PTO/sham/TraLO/PHR comparisons. A new dataset adapter, frozen protocol,
source/label-boundary tests, actual GPU cost/dose preflight and certified budget
are still required. None follows merely from the large support counts.

HAM10000 remains a second staged candidate. Its official source and decodes pass,
but it has lesion identifiers rather than patient identifiers, and the proposed
MEL-versus-other 60/10/30 design remains unapproved and exploratory after target
support was viewed. Preserve its original failed support gate. Do not adjust
the target or split from those counts.

This checkpoint advances breadth and validation. It establishes no new
performance win and adds no full GPU training run.

The later [prospective ISIC2024 protocol](isic2024_prospective_protocol_for_review_20261005.md)
records one explicit quota proposal, its diagnosis-blind geometry check, adapter
requirements and the additional compute-accounting evidence. It is a review
proposal and does not freeze the data design or authorize GPU dispatch.
