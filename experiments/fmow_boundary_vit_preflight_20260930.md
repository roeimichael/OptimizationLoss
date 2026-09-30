# ViT-B/16 extension: fixed preflight, no scored run yet

The user requested a transformer backbone in addition to MobileNetV3. The
current fmow2 local runner is MobileNetV3-specific, so this document fixes the
ViT choice and entry gates **before** interpreting any ViT development score.
The old fmow2 ViT runs used a different training schedule, precision and cap
policy; they are neither controls nor fresh confirmation for this study.

Use torchvision `vit_b_16` with
[`ViT_B_16_Weights.IMAGENET1K_V1`](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html)
and a newly initialized eight-way head. Preserve the fmow2 full-frame RGB
224-by-224 ImageNet mean/std transform and disclose that it differs from the
official weight preset's resized center crop. Do not change geometry after
seeing results. The factory's no-download CPU shape test is implemented. A
read-only check on 2026-09-30 found the cached `vit_b_16-c867db91.pth` on
**both** DSI hosts with matching SHA-256
`c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9`.
The frozen DSI Python environment also loaded those weights on both hosts
and exposed the expected 768-input, 1,000-class original head. The new
eight-class head has only had a no-download local CPU shape check. Actual
remote new-factory load, image transform, full gradient and GPU memory remain
to be verified. A missing or changed cached weight file is a failed
preflight, not permission to substitute random initialization or a different
weight preset.

The intended comparison has one ViT PTO trajectory per seed and the **same
five side-copy arms and two caps** as the fixed MobileNetV3 boundary protocol:
zero-step PTO/local Clipper analogue, calibrated joint TraLO, pooled-only at
the joint radius, dose-matched sham, and calibrated PHR. Keep the allocator,
unlabeled development group/size quotas, primary deployed class-1 cc-F1,
paired six-contrast family, secondary metrics and data boundary identical.
Only the backbone and an up-front, within-ViT matched training batch size may
differ. Choose that size from a label-free GPU memory smoke, before any ViT
development metric. A ViT result is a separate atomic cell and cannot be
pooled with MobileNetV3 seed deltas.

Proposed distinct ViT seed IDs are pilot **6500** step-on and same-host
step-off, then full **6501–6512** if and only if every identity, gradient,
allocator, dose and label-blind pilot gate passes. Before full dispatch, use
the measured host runtime to project all 13 step-on seeds plus the reference
at no more than **24 GPU-hours**; a larger projection stops the full block.
This ceiling is a limit, not a commitment to spend it. ViT-B/16 has about
86.6 million parameters and 17.56 GFLOPs of nominal inference versus
MobileNetV3-Large's 5.48 million and 0.22 GFLOPs under torchvision's
[model metadata](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_large.html).
The earlier 2.1-hour MobileNetV3 projection therefore cannot price ViT. A
successful one-seed pilot is not itself a backbone efficacy claim.

Before launch, record the exact pretrained checkpoint SHA-256, dependencies,
eight-class output and parameter counts; revalidate actual array and role
hashes, group/sample IDs and exact duplicates, source parity, class supports
offline, no development-label access, full/chunked probability and gradient
parity, finite-difference constraint derivative, PTO/BN/RNG neutrality, fixed
allocator recount and same-host GPU memory/runtime. Test a mutation that makes
each critical gate fail. The scored ViT block must have its own immutable
release, configs, queue roots, independent scorer and complete per-seed
artifacts. Never score fmow2 reserved countries merely to accelerate the
paper. If ViT-B/16 exceeds the ceiling, a smaller transformer (for example
DeiT-Tiny) is a **new** named protocol requiring its own pinned weights and
preflight, not a substitution inside this one.
