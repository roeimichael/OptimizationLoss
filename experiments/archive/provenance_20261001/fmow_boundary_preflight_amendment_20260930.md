# Boundary preflight scorer precision amendment, 30 September 2026

The first real-image, label-free preflight of immutable release
`c03f9795d76b24c3d3ff8fed894082d2d85705dd` ran on dsisco01 Quadro GPU0
after an exclusive analytic CUDA smoke. It used 25 actual development images
(five per country), no development labels, a deterministic diagnostic head,
and the published boundary runner/scorer. The full-versus-chunked inference
and gradient checks and the central finite-difference checks reached the
runner/scorer contract stage without raising. The independent scorer then
rejected the PHR boundary record at its **initial radius recount**:

| Quantity | Value |
| --- | ---: |
| Recomputed radius from saved FP32 probabilities | 0.04022328140070927 |
| Runner's logged radius | 0.04022328439896066 |
| Absolute difference | 0.00000000299825139 |

This is a scorer precision false failure, not a quality result or a completed
study pilot. The old preflight root and its four side probability artifacts
are retained at
`/home/dsi/michaer8/tralo-rebuild/verification/c03f9795d76b24c3d3ff8fed894082d2d85705dd/boundary-real-preflight-20260930T1700Z`.
The immutable release is untouched; no seed 6400 or 6401–6412 job ran.

The scorer now allows a bounded `max(1e-7, 1e-5 × |radius|)` recount tolerance
for the initial radius, scaled by each probe's halving index. Exact zero
remains strict. A regression fixture covers this FP32 discrepancy and the
mutation tests still reject materially changed radii. The correction requires
a new commit/release and fresh preflight root; it does not justify reusing or
overwriting the failed attempt. The preflight's final outcome remains unknown
until the corrected gate and all subsequent gates complete.
