# fmow2 PHR direction: fixed seed-6300 pilot and log audit

This is a descriptive exploratory pilot, not a setting-selection result or a
full ALM training comparison. The fixed 12-seed block 6301–6312 was launched
unchanged after the label-blind pilot gate passed. All fits used immutable
release `2f6a68eb006cf2d9cd535a7dfa51f5c09ce9850e` on dsisco02 in FP32.
The paired step-on and step-off runs both exited 0. The gate independently
verified six epochs, exact PTO snapshot parity, source/config/data/artifact
identity, active finite PHR gradients at both caps in all six epochs, the
predeclared step dose and allocator checks. It did not access development
labels. The measured projection for the pilot plus full block was 2.103
GPU-hours, below the fixed 8 GPU-hour ceiling.

The separate offline pilot scorer then evaluated the repeatedly viewed fmow2
development countries. No setting was chosen from those labels:

| Pooled cap | PTO allocated class-1 F1 | Fixed TraLO-local | PHR-local | Pooled-only dose | Sham |
| --- | ---: | ---: | ---: | ---: | ---: |
| 167 | 0.5013 | 0.4178 | 0.4178 | 0.4282 | 0.5013 |
| 83 | 0.4013 | 0.3411 | 0.3411 | 0.3478 | 0.4013 |

Equal F1 does **not** mean equal selected images. At cap 167, the PHR and fixed
TraLO allocations share 161 of 167 IDs (six swaps each way; Jaccard 0.931).
At cap 83 they share 79 of 83 (four swaps; Jaccard 0.908). Compared with PTO,
both lose 16 correct selected images at cap 167 (96 to 80) and nine at cap 83
(60 to 51). At the tighter cap, the methods also differ in which country
receives correct selections. The pilot therefore supports similar *quality at
this stress dose*, not equivalence of the directions.

The saved training traces show a strong common mechanism. Each of the 12
cap-by-epoch PHR opportunities applied the full parameter L2 displacement
of approximately 0.1. Before the side step, raw pooled class-1 calls across
the six snapshots were 186, 217, 208, 231, 243 and 256. After the PHR step,
they were 4–12 at cap 167 and 4–13 at cap 83; the fixed TraLO step produced
4–13 and 4–17 respectively. The PTO ensemble made 229 raw calls; the fixed
TraLO and PHR ensembles made only 7–8. The common allocator still filled all
167 or 83 permitted slots, so this raw collapse is a mechanism diagnostic,
not a substitute for allocated F1. The PHR penalty dropped to zero on 11 of
12 snapshots (the last tighter-cap snapshot retained 0.000408). The sham
step generally left raw calls near their starting values.

The underlying shared training fit selected epoch 1 by the unchanged stop
rule. Its training loss fell from 0.9176 to 0.0951 through epoch 6, while
stop-country loss increased from 1.3281 to 1.8653. This replicates the
widening train/stop gap observed in the prior fixed-dose block. All five pilot
arms used the same PTO trajectory and six-snapshot window.

The evidence suggests that the fixed 0.1 stress dose overwhelms the cap
boundary and obscures differences between local directions. It does not
establish a dose-response curve or prove that a smaller, label-free calibrated
step would improve held-out performance. Such a rule would be a distinct
method needing a separately fixed protocol and an independent evaluation
boundary; the viewed fmow2 development labels cannot confirm it.

Receipts in `C:/Users/roeym/.codex/rebuild-audit-20260922` are
`fmow_local_alm_6300_gate_20260930.parsed.json` and
`fmow_local_alm_6300_pilot_score_20260930.parsed.json`. The original raw
command captures are preserved alongside these parsed copies. The GPU run
roots are `pilot6300_step` and `pilot6300_ref` under
`/home/dsi/michaer8/tralo-rebuild/runs/fmow-local-alm-direction-20260930-2f6a68eb`.
