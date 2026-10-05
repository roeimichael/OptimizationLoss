# ISIC2024 diagnosis-blind adapter stage

This checkpoint implements the metadata portion of the prospective ISIC2024
adapter. It does not freeze the proposed experiment or create a training cohort.
The scientific decision in `isic2024_prospective_protocol_for_review_20261005.md`
remains pending, including target access, groups and caps. Classification remains
image-only; this stage prepares identity and constraint-group information.

## Contract and rationale

`analysis.prepare_isic2024_metadata` reads only the official archive's embedded
`ISIC_2024_Training_Input/metadata.csv` and ZIP member names. It checks unique
canonical image IDs, exact image/metadata joins, nonempty patient identities,
recorded sex and CSV structure. It rejects duplicate ZIP members, unsafe or
unexpected image paths, ambiguous headers and known diagnosis-bearing headers.
JPEG payloads, ground truth and diagnosis supplements are never opened. It does
not extract images, recompute the existing decode audit or infer missing sex.

The stage reproduces the already specified, unapproved patient-hash candidate
with namespace `tralo-isic2024-patient-split-20261005`, first eight SHA-256 bytes,
big-endian modulo 10,000, and boundaries 7,000/8,500. No alternative namespace,
split, cohort or quota was selected. Whole patients share a partition. Group
counts are image counts; patient counts are reported separately. Recorded sex
is retained per image, with missing values reported separately and no imputation
between images of one patient. This stage assigns no local or pooled ceilings.

Its two exclusive outputs are a label-free candidate index and
`metadata_candidate_manifest.json`. The manifest records metadata, image-member
index and output hashes, partition counts, and the explicit status
`candidate_only_not_frozen_not_training_ready`. No `manifest.json`, runner rows,
scorer labels or approved quota policy is emitted. Existing training preparation
and the quota registry continue to reject an ISIC2024 campaign.

The purpose is to validate identity routing and the read boundary independently
of diagnosis access. The alternative of extending the complete labeled cohort
preparer now would require the pending scientific/data-access decision. This
small stage can become an input to that preparer after approval; its outputs
must not be mistaken for a frozen dataset. Source archive integrity, decoding,
diagnosis support, gradients, dose, GPU cost and launch readiness are explicitly
unverified by this stage. A metadata/member-name hash is not an archive checksum
or proof of pixel identity. Synthetic fixtures do not authenticate the real
official archive.

## Independent validation

The new test file contains 20 synthetic checks. Independently calculated patient
hash buckets 3,553, 8,446 and 9,616 cover train, stop and development, including
two images of one patient. A read interceptor permits only embedded metadata:
opening a JPEG, ground-truth CSV or supplementary CSV fails the test. Fixtures
deliberately contain undecodable image bytes and forbidden diagnosis files.

Negative controls cover duplicated IDs/members, missing or extra images, nested
image paths, empty/ambiguous patients, unknown sex, malformed CSV rows, missing
or repeated headers, known target/diagnosis headers, and an empty cohort. A
permuted input preserves sorted output identities. Output hashes are recomputed
independently, an existing output cannot be overwritten, and the actual module
CLI succeeds once and refuses repetition. The runtime reader and unchanged
quota policy reject the candidate as training input.

The first test invocation failed because the new module did not yet exist.
After implementation, all 20 new checks and nine neighboring preparation,
authenticated-reader and quota checks passed locally (29 total). No existing
test was weakened. Local and remote CPU receipts, source hashes, exact immutable
release identity and actual CLI checks are preserved outside the repository
under `isic2024_metadata_stage_validation_<commit>` in
`C:/Users/roeym/.codex/rebuild-audit-20260922`. Check that receipt before claiming
remote validation: committing this note alone does not establish deployment.

## Evidence boundary and next step

Only synthetic archives have been used with this new stage. The existing real
archive, once-only decode receipt and diagnosis-blind candidate diagnostics
remain unchanged. No real diagnosis labels, development target counts, model
outputs, GPU jobs or new GPU inventory were accessed. No GPU spending is added.
Earlier empirical negatives and the partial accounting balance remain intact.

The remaining adapter work is authenticated labeled preparation with a sealed
development-label artifact, real-source/identity validation and an image reader.
That work depends on the scientific/data-access decision. Logging, gradients,
GPU cost/dose and certified aggregate budget remain gates before any campaign.
