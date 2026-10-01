# Knee ViT replay recovery, 1 October 2026

This is a source-integrity repair to the frozen knee comparison, not a change to
the TraLO direction, Kassif PAO, allocator, model, data, caps, optimizer, or
metric. The original ViT seed 6700 pilot failed at its first TraLO constraint
pass with `constraint pass changed logits at fixed weights`. Its output and
logs remain immutable. It must never be resumed or overwritten.

The constraint code makes an inference pass without autograd and a second pass
with autograd at unchanged weights. PyTorch can dispatch the ViT multi-head
attention to its native fast path only in the first pass. A fixed-weight CPU
replay on the server produced a maximum output difference of
`3.5762786865234375e-07` with that path enabled and exact equality when it was
disabled. The existing fMoW ViT runner had already handled the same dispatch
difference. This is a source-backed failure hypothesis, not evidence that a
model quality result improved.

The new runner disables the attention fast path for ViT before model creation
and all training arms. It records the setting and a finite no-grad versus grad
replay on actual label-free development images, using the original strict
probability tolerance; a mismatch fails before training. The independent
scorer requires that receipt. No scorer tolerance is relaxed. B5 and
MobileNetV3 remain on their original immutable release and are not rerun.

Only ViT seed **6713** at cap **76** is added as a fresh integrity pilot. It
uses the identical pilot recipe, subject-stable data, pretrained weight hash,
PTO/null equality, PAO recurrence, TraLO direction, sham dose, and label-blind
gate. Its development quality must not choose a setting. Keep the original
full ViT seeds **6701–6712** and caps **54/86** fixed; dispatch them only after
the recovery pilot's label-blind gate and a measured finite cost projection.
The recovery pilot has a **4 GPU-hour** execution ceiling on one genuinely
free dsisco02 GPU. Preserve every failed attempt and all old scorer receipts.
The final analysis must identify the distinct immutable runner release for
each backbone and verify every run against its own release before comparing
paired seed results. No sealed test label is accessed.
