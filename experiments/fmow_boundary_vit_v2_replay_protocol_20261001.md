# ViT-B/16 fixed-weight replay amendment, 1 October 2026

This is a repair of the already fixed fmow2 ViT-B/16 backbone comparison in
`fmow_boundary_vit_preflight_20260930.md`, not a new choice informed by a ViT
development score. No ViT development labels have been scored. The pretrained
checkpoint, 224-pixel full-frame transform, training recipe, two caps, five
side-copy arms, allocator, metrics, paired analysis and six-contrast family
remain fixed. The fmow2 development countries remain exploratory; the five
reserved countries and Chen knee test remain sealed.

## Preserved failures and numerical diagnosis

Immutable release `67141697e03b95faea5669c9d146a822b8888784` stopped in
its label-free synthetic memory smoke before claiming seed 6500. The head's
class-1 soft count was 134.5556, below cap 167, so a required active-gradient
diagnostic could not run. The named smoke-fixture amendment and immutable
release `284d638dcfa14e6fa1f8b224933e627b67e8c5f0` corrected only that
disposable diagnostic head. Its queue-owned label-free memory and real-image
preflights passed on dsisco02. Seed 6500 was then claimed and failed in epoch
one at the fixed-weight logit-replay integrity check. Its failed root,
`queue.log`, event log and preflight receipts remain immutable evidence. Do
not restart, overwrite or re-score it; do not launch its old step-off or full
block. Neither failure is an efficacy result.

The replay check compares PTO probabilities made in `torch.no_grad()` with a
side-copy gradient replay at the same weights and images. A read-only local
pretrained ViT-B/16 probe observed different attention kernels in these two
modes (`aten::_native_multi_head_attention` versus
`aten::scaled_dot_product_attention`), with a maximum probability difference
of 1.94e-7; disabling the PyTorch MHA fast path made the two outputs identical
on that CPU probe. That is a plausible cause of the server failure, not yet a
proof of its exact magnitude. The new release disables the fast path **for
ViT only, before its first PTO inference**, so both saved probabilities and
gradient replay follow the same route. The pre-existing elementwise replay
tolerance (absolute 1e-7 plus relative 1e-6) is unchanged. MobileNetV3 and
all completed releases retain their old behavior.

## Fixed fresh attempt and gates

Use the new study ID `local_boundary_vit_v2`, pilot seed **6600** step-on and
same-host FP32 step-off in separate new roots, then full seeds **6601-6612**
only if the independent label-blind gate passes. The 14 configs are frozen
before either new pilot and use training batch 16 and development chunk 8.
Never copy checkpoints or artifacts from the failed seed 6500 into these
runs. No setting, epoch, cap, seed or arm may be selected by a ViT score.

Each queue invocation on a genuinely free and exclusively locked GPU must
first run its own synthetic memory and 15-real-image numerical preflights
under the exact immutable release. Both receipts must record that MHA fast
path is disabled and that an **ordinary newly initialized eight-class head**
passes no-grad versus gradient fixed-weight probability replay on the same
unlabeled eight-image chunk before either diagnostic head is installed. Require
the unchanged elementwise tolerance, an explicit positive replay flag and a
maximum tolerance ratio at most one. Preserve a negative receipt and stop
before seed claim if any check fails. The synthetic smoke still checks actual
batch-16 training/Adam, full unlabeled 1,673-image inference, every active
pooled/country diagnostic scope, nonzero backbone gradients, all side-copy
arms at both caps, GPU capacity and PTO neutrality. The real-image preflight
still checks exact data/weight hashes, full-versus-chunked probabilities,
autograd-versus-finite-difference gradients, side-copy artifacts and unchanged
PTO weights. The independent scorer must reject missing, fabricated or
positive-looking but numerically invalid replay evidence before opening
development labels.

Before launch, run local and both-host native tests, source/config/data
identity checks, CLI and a same-card exclusive CUDA arithmetic check from a
new detached immutable release. Inventory physical GPU UUIDs, compute PIDs,
owners and commands on **both** hosts and recheck the selected card just
before claiming it. A compute PID means occupied even with low utilization.
Never share or stop another user's job. A dead queue or lost SSH connection
requires forensic inspection rather than an automatic retry.

After both pilot jobs finish, require exact PTO trajectory parity, hashes of
release/config/data/splits/artifacts, no development-label access, independent
quota/allocator recount, correct active/skip and dose/gradient/neutrality
diagnostics, and a same-host measured projection at or below the original
**24 GPU-hour ceiling** for all planned pilots, queue-owned preflights and
full runs. Reserve **0.5 GPU-hours** for the two failed v1/v1.1 attempts
(the first failed smoke itself took 0.0277 hours); the v2 projected cost must
therefore be at most 23.5 GPU-hours. This conservative reserve keeps the
cumulative study within the original ceiling and does not authorize more
compute. Pilot scoring is exploratory and cannot choose settings or cancel a
fixed block that passed its integrity and cost gates. Only after those gates
pass may the 12-seed block launch once, with complete per-seed scoring by the
independent release scorer. Report all caps, arms, secondary metrics, paired
intervals and failures, including negative findings. The ViT seed pairs are
separate from MobileNetV3 and cannot be pooled as independent confirmation.
