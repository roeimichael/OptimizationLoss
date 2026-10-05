# ISIC2024 authenticated label candidate stage

This adds source authentication and label separation to the existing metadata
candidate. It does **not** approve the prospective scientific protocol or real
diagnosis access, register ISIC2024 for training, select quotas or start a job.
Every execution of this new component so far uses synthetic data.

## What the component verifies

`analysis.prepare_isic2024_labels` accepts the metadata candidate directory, a
binary-label CSV, independently supplied SHA256 pins for its metadata manifest
and label CSV, and an exclusive new output directory. A supplied hash establishes
byte identity; it cannot establish permission or trustworthy source provenance.
Real use still requires the scientific/data-access review and authenticated
official source receipts.

The metadata manifest and row bytes are authenticated before the label file is
opened. The stage rejects duplicate identities, label-bearing metadata rows,
altered patient routing, changed group categories, unexpected image paths and
incorrect partition supports. It retains the single existing patient-hash
candidate and recorded-sex groups. No alternative split or target is searched.

The label schema must contain exactly `isic_id` and `malignant`. Target values
must be the literal binary strings `0` or `1`; duplicate, missing, malformed or
unmatched image IDs fail before any output is created. Parsing uses the bytes
that were hashed, avoiding a second source-file read between authentication and
parsing. This stage does not open the image archive or decode image payloads.

Candidate train and stop rows have sample ID, image filename, group and label.
Public development rows have only ID, filename and group. Development labels
are written to a separate private artifact with its own manifest linked to the
public manifest hash. Public file references and support summaries contain no
private artifact pointer or development positive counts. The private directory
is a logical boundary, not an operating-system access-control guarantee.

Outputs retain `candidate_only_not_frozen_not_training_ready`. There is no
runtime `manifest.json`, image directory registration, decoder-policy selection,
quota registration or training recipe. The existing runtime reader therefore
refuses this candidate as a prepared campaign. No source or archive evidence is
overwritten; repeat output paths fail before source or label reads.

## Choices and alternatives

The separate candidate stage permits verification before scientific approval
without silently enabling a campaign. Reusing the existing metadata candidate
avoids a second split or metadata interpretation during label preparation.
Externally supplied source hashes and strict schemas were chosen over accepting
an unverified CSV or deriving trust from a filename. Reconsider the interface
only if authenticated official input bytes demonstrate a schema mismatch;
preserve that negative and review an explicit amendment rather than accepting
extra diagnostic fields automatically.

## Validation and limits

The local focused check exercises 23 new synthetic cases and the existing
partial-block scorer label-refusal case: 24 passed. Examples independently fix
the patient routing, label rows and public support counts. Negative controls
deny label-file reads when metadata authentication or routing fails. Changing
the synthetic development label changes its private row while leaving every
public row and support unchanged. A synthetic bridge to the existing image
reader denies all private file opens and verifies that the classifier-facing
item contains RGB pixels, an absent development target and separate metadata.
The actual candidate CLI and repeat refusal are also exercised.

Each immutable deployment must verify its own commit and all committed file
hashes, run these focused checks and a separate synthetic CLI on both DSI
hosts, and preserve the exact receipts offhost under
`isic2024_label_stage_validation_<commit>`. Those receipts identify the actual
deployed version and outcomes; this document is not a substitute for them.
The complete CPU regression baseline remains at `cc21dd1c41f332f4147a7cf298e96762b13cde3e`;
it has not been rerun for this component. The two CUDA-only checks remain open.

This is infrastructure evidence. It demonstrates neither official target
support nor a ready labeled training adapter, useful constraint dose, GPU cost,
clinical validity or empirical superiority. The weak clinically assumed benign
labels and remaining overlap limitations in the prospective protocol remain.
Real data access, protocol freeze, image/runner integration, logging and gradient
gates, GPU cost/dose and a certified finite balance must precede any campaign.
The existing completed experimental negatives and uncertainty are unchanged.
