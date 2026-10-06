# Separate pooled and local derivative identities

The non-boundary `local_targeted_step` accepted a local group named `global`,
then wrote its derivative over the pooled derivative under the same dictionary
key. The common-descent guard could therefore miss pooled ascent. The boundary
path already has separate `pooled` / `local:<group>` identities and is unchanged.

New non-boundary records explicitly declare `scope_derivative_schema` as
`pooled-local-v1`. Their derivative keys are `pooled` and `local:<exact group>`.
The prefix is injective even for group names containing colons or named `pooled`.
The same complete dictionary is used for the guard and its log. Scope gradients,
weights, normalization, search, placement, caps, masks and recipe are unchanged.

The three existing TraLO side-step auditors accept this explicit schema and
retain their unambiguous legacy input contract. Unknown schemas and an active
pooled-plus-local-`global` legacy record are refused. Old receipts and reports
are not rewritten. This is not a replay or rescore of any completed campaign.

Prospectively fixed CPU regression: two scalar logits at theta zero have
probabilities .8/.6 and probability derivatives +4/-6. At pooled K=1 and local
K=0/1, the pooled gradient is -2, active local gradient +4 and normalized joint
direction -1. The independently expected derivatives are pooled +2 and local
-4. The repaired guard refuses this conflicting step before displacement;
the explicit no-descent fixed-dose control records both. Relabeling the local
group preserves the parameter displacement, counts, gradient and radius.

The original missing-pytest interpreter refusal is preserved separately from
the native RED run (10 failures, 7 passes). The new/affected local checks pass
54 nodes on the repaired source. Additional direct snapshot integration and
both-host release verification have their own original, hash-backed receipts
outside the repository. No completed full suite or actual one-epoch readiness
trajectory is replayed for this repair.

This edge case is absent from the completed country masks and approved H0/H1
groups. Fixing it does not explain their F1 differences or establish improvement.
The neighboring PHR implementation has a separate raw-name derivative log path
that still requires review; it is not covered by this repair. All scientific
seeds, data access, constraints, comparison design and GPU authorization remain
unchanged. No real input, private target, pretrained weight, model artifact,
scientific training, scorer, CUDA or GPU cost probe is used by these fixtures.
