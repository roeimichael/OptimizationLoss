# Weekend dual-constraint preparation, 1 October 2026

**Status:** prospective design. Only the already frozen knee/fMoW runs described
below may be inspected. A new dataset/backbone cell is not a GPU job until its
data, gradient, source, release, first-run, and cost gates pass. This document
does not convert an exploratory development score into a held-out result.

## Question and honest scope

Does a *training-time* pooled-plus-metadata-group constraint improve the
ranking of correct cases admitted under a fixed global capacity and explicit
group upper bounds, compared with the same trained model clipped only at
deployment, an inexact PHR-ALM training correction, and a schedule-matched
zero-correction control? A good result requires a paired gain in deployed
constrained-class F1 without being dominated in macro/weighted F1, plus valid
source/data/compute evidence. No favorable pilot score may select a cap,
backbone, group, seed, or dataset.

The current knee targeted step is a normalized soft-count gradient followed by
a hard-cap line search. It does **not** isolate the original saturating TraLO
loss. The prospective persistent fMoW runner tests a genuine training-time
pooled/country correction, but its country development labels have been viewed
repeatedly; it can diagnose mechanism, not confirm a paper claim.

## Data shortlist and local attribute

Every candidate has at least 10,000 images before splitting. The classifier
receives the image only in the primary comparison; the tabular group field
is supplied to the quota calculation and identical deployment allocator for
every arm. Feeding tabular covariates to the classifier would be a separate,
matched experiment, never an unmarked advantage for TraLO.

| Candidate | Current verified state | Target and group | Decision before GPU |
|---|---|---|---|
| fMoW country slice | Byte-audited 15,841 supervised images plus country-disjoint stop and development pools; the development countries have been scored repeatedly. | Class-1 facility category; country from image metadata. | Treat new scores as exploratory mechanism evidence. The prepared persistent MobileNetV3 comparison still needs immutable release, real-data/gradient preflight, pilot and cost gate. Reserved countries remain sealed. |
| SIIM-ISIC 2020 | Official 33,126 JPEGs, metadata and duplicate list downloaded; byte/CRC and image-name join pass. 584 training positives, 2,056 patients. | Melanoma; sex or broad anatomical site. | Patient/lesion-disjoint split; duplicate-family separation; positive support and missing-group policy; first-run regime audit. |
| CelebA | Official 202,599 aligned images, 40 attributes, identity and partition files downloaded; every official MD5 and image count passed. A frozen identity-disjoint 70/15/15 split has 140,166/31,966/30,467 images and 7,034/1,586/1,557 identities. | Smiling target; recorded Male attribute defines two local groups. | Verify filename/attribute/identity bijection, patient-like identity isolation and image decode; use only as a synthetic allocation benchmark, never as a clinical or fairness claim. |
| SLICE-3D / ISIC 2024 | Official 401,059 JPEGs, metadata and labels downloaded; exact three-way ID join passes. 393 positives, 1,042 patients. | Malignancy; sex or broad anatomical site. | Exploratory reserve only: patient-disjoint split has just 53 development positives, including 15 in the female group. Review label provenance and group support before any GPU work. Extremely low prevalence makes accuracy a poor primary metric. |
| HAM10000 | 10,015 official images are available publicly; not yet downloaded or audited. | Seven diagnoses, with melanoma as a prospectively declared capped class; sex/site group. | Verify license/access, image/metadata identity, lesion-level split, source overlap with ISIC, class/group support. It does not count as validated until these gates pass. |
| iWildCam oodslice | 20,000 train and 2,943 internal test images; labels match official train annotations, but all internal test filenames come from the official training list. Seven held-out camera locations have zero support for several of the eight species. | Camera ID. | Exclude this slice from the three-dataset local-constraint claim; its camera/species coupling leaves many class-by-group cells unevaluable. A different, independently audited wildlife protocol would be needed. |

The prospective primary dataset panel is fMoW, ISIC 2020 and CelebA. They
cover satellite, dermoscopy and face imagery, but neither the fMoW development
countries nor any viewed pilot pool can support a confirmatory paper claim.
CelebA's attribute annotations describe images, not an external operational
resource; a sex-group quota is a controlled synthetic task. ISIC 2020 and
2024 are related dermatology domains, not proof of cross-modality
generalization. Their patient identifiers are distinct within each archive,
but cross-archive duplicate/source leakage must be checked before pooling.

The official SLICE-3D descriptor states that most benign cases lack a biopsy
and are clinically assumed benign; do not describe every negative as
pathology-confirmed. Source: https://doi.org/10.1038/s41597-024-03743-w .

Together with fMoW, ISIC 2020 and CelebA form three prospective tables above
10,000 images, but no three-dataset result exists until each independently
passes its release, split, group-support and complete-block gates. Counting
downloaded images as a trained result would be wrong. The backbone panel is
MobileNetV3-Large, ViT-B/16 and ConvNeXt-Tiny where weights, preprocessing,
first-run reproducibility and memory fit are verified. A backbone comparison
is matched within a cell; a pilot score cannot select a favorable backbone.

## Exact intervention and matched controls

For unlabeled deployment images i, image model probabilities p_i(c), global
capacity K, and metadata group g_i, the soft violations are

    r_global = sum_i p_i(c) - K
    r_g = sum_{i:g_i=g} p_i(c) - K_g.

The hard deployment allocator chooses at most K class-c calls overall and at
most K_g from each group, with fixed sample-ID tie breaking. Quotas are
upper bounds, not forced referral targets. Fix K and K_g from a disclosed
operational percentage and unlabeled group sizes before opening development
labels. Retain every missing-metadata row in the pooled constraint and report
it separately. If its support is too small for a meaningful local estimate,
give it no separate local cap rather than manufacturing a tiny bucket; apply
that same rule to every arm and the deployment allocator. A 70/30 referral
share is only one possible policy, not a
fact inferred from outcomes; any such policy needs a dated protocol choice.

For a future test of TraLO's **actual saturating loss**, use one term per
pooled or metadata-group scope. With
`e_s = relu(sum_{i in s} p_i(c)-K_s)/max(K_s,1)`, define
`L_local = sum_s lambda_s [e_s/(1+e_s) + rho*e_s^2/(1+e_s^2)]`.
The image classifier receives no sex/site input in the primary comparison;
those fields index the constraints and common deployment allocator. The new
`tralo/local_bounded_penalty.py` implements this scalar and its independently
finite-difference-checked logit derivative. It is a mathematical prototype,
**not yet a trained result**. Its scope coefficients must remain meaningful
through the parameter update; normalizing away the whole gradient would again
erase the saturating loss shape and fail to isolate the novelty.

The planned arms use the same backbone, initial weights, training images,
order, augmentation, task optimizer, nominal task updates, precision,
checkpoint rule, and deployment allocator:

1. **PTO/Clipper:** supervised image training, then only the common allocator.
2. **TraLO:** persistent task training plus the specified pooled and group
   correction, with logged raw gradients and applied parameter displacement.
3. **PHR-ALM:** persistent inexact primal correction and per-scope nonnegative
   dual update, with the same correction opportunities and maximum dose.
4. **Null/sham:** the same schedule and observations with zero correction;
a radius-matched sham is retained where the method makes a side step.

For the newly prepared tabular cohorts, freeze these two capacity levels now,
before a training score is available. They are calculated only from the
unlabeled development cohort counts by `tralo/tabular_quota_policy.py`:

| Cohort | Level | Pooled cap | Female cap | Male cap | Interpretation |
|---|---:|---:|---:|---:|---|
| ISIC 2020 | 1 | 48/4,795 (1%) | 32/2,110 (1.5%) | 41/2,668 (1.5%) | Scarce specialist referral slots. |
| ISIC 2020 | 2 | 96/4,795 (2%) | 64/2,110 (3%) | 81/2,668 (3%) | Less scarce referral slots. |
| CelebA | 1 | 7,617/30,467 (25%) | 6,211/17,745 (35%) | 4,453/12,722 (35%) | Synthetic positive-call allocation. |
| CelebA | 2 | 10,664/30,467 (35%) | 7,986/17,745 (45%) | 5,725/12,722 (45%) | Less restrictive synthetic allocation. |

The 17 ISIC rows with missing sex count toward the pooled cap and are
reported separately, but have no separate 0-or-1-person local quota. Local
caps sum above the pooled cap and each local cap is below it, so both levels
can actually constrain a skewed allocator. This is a prospective operational
stress test, not a clinically endorsed referral policy or a fairness
guarantee. If the image model's probabilities make all constraints slack, we
report that failure of experimental activation instead of changing the caps
after looking at outcomes.

The primary contrast is TraLO minus its schedule-matched null, then TraLO
minus the deployment-only control. An ALM comparison alone cannot attribute a
benefit to TraLO. Report both clip and focal-clip where their supervised
training recipes have been independently validated; never substitute an old
score from another data/optimizer recipe.

## Measures, gates, and stopping

Primary quality is deployed cc-F1 for the declared capped class; alongside it
report precision, recall, macro-F1, weighted-F1, per-group confusion/counts,
feasibility, model-selection rule, elapsed GPU-hours, and per-seed paired
differences with a prespecified 95% interval. A bold best mean is not a
significance claim. At least four matched seeds and two cap levels make a
screen; a publishable comparison needs a separately frozen, sufficiently
powered block and untouched evaluation. Pilot labels are not used to choose
settings. The current knee and fMoW development pools are not untouched.

Before dispatch: hash source/config/checkpoint/data/split; verify image-label
alignment, patient/lesion overlap, exact duplicate pixels, missing groups,
group support, and a label-free quota recount. Check analytic/autograd/finite
difference gradients for active/slack/conflicting pooled/group constraints,
real-image forward/backward parity, null neutrality, dose and allocator
invariants. Record planned/attempted/applied/skipped task and correction
updates; soft/hard counts, residual/dual, raw and transformed gradient norm,
actual parameter displacement, training/stop loss, nonfinite events, host UUID
and precision. First run on two genuinely free GPUs only, inspect events and
outputs, then expand if the gate passes. SSH loss means unknown run state.

The user authorized a weekend target of roughly 24 hours of work with up to
four dsisco02 GPUs. This is a **maximum 96 aggregate GPU-hour planning ceiling**,
not an instruction to occupy another user's cards or spend hours on unvalidated
jobs. Each job needs its own exclusive root and per-job wall cap; the queue
must not claim a seed with insufficient remaining time. Record actual card
seconds and halt new dispatch at the ceiling. Use dsisco02 local `/tmp` for
large new artifacts: shared `/home` had only 4.4 GB free at 20:32 UTC. Keep
hashes and an off-host recovery copy before relying on `/tmp` alone.
The persistent fMoW runner uses an exclusive root beneath
`/tmp/tralo-weekend-michaer8-20261001/runs`; its small seed claims, cost
receipts and physical GPU leases remain on shared `/home` so a second host
cannot unknowingly reuse an identity or card. Its queue is restricted to
dsisco02. Local `/tmp` is not durable after a host reset.

## Tonight's order

1. Forensically close the timed-out knee ViT queue. Preserve seed6705/6709
   partial roots, do not reuse their IDs, and independently gate seed6708.
2. Review the already prepared persistent fMoW runner/scorer and its negative
   controls, complete focused tests and data/gradient checks, freeze a new
   immutable release, and run one gated pilot if still cost-feasible. This is
   exploratory because its development countries have been viewed.
3. Freeze ISIC 2020 patient-level and CelebA identity-level boundaries and
   broad group policies without looking at development quality. Validate
   image decode, joins, duplicates, labels, class/group support and costs.
4. Keep ISIC 2024 as a weak-label, low-positive-support reserve. Do not count
   it as a successful third trained dataset or use it merely to fill GPU time.
5. Only after first-run integrity and measured cost pass, launch bounded
   independent queues that can finish during a VPN outage. A missed weekend
   capacity target is preferable to invalid or duplicate evidence.

The weekend output is a set of exact run receipts, audited metrics if a
complete block finishes, and a clear list of failed gates and remaining work.
No claim of a TraLO win is made in advance.
