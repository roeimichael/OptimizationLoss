# Fmow2 joint-local pilot: bounded feasibility failure

The fixed `fmow_joint_local_protocol_20260928.md` pilot began on 2026-09-28
13:04:23 UTC on `dsisco02` GPU 2 (physical UUID
`GPU-5a6df0cc-27fe-9ffa-3a55-df83b918e434`) from immutable release
`e5124c7d6be87dbc223cc014658c0402aaece227`. The exclusive run is
`/home/dsi/michaer8/tralo-rebuild/runs/fmow-local-pilot-20260928/seed6099`.
An analytic CUDA smoke passed immediately beforehand; the research pilot
stopped at 13:05:32 UTC with exit code 1 during its first epoch's first
joint-local side step. Its terminal error was `joint search reached its fixed
displacement ceiling`. The source, configuration, data and pool identity were
hashed in the run's `started` event. The queued step-off reference and full
6100–6147 block were **not** launched. This is not a quality comparison.

The saved epoch-1 PTO probabilities contain 1,673 development items and eight
classes. An independent, label-free recount found 227 raw class-1 calls.
At the `N/10` policy, the pooled limit is 167, and local limits include NLD
19 and PHL 23. Raw NLD and PHL calls were 66 and 82 respectively, so those
two countries exceeded their limits by 47 and 59. At `N/20`, the pooled
limit is 83, and the country limits are tighter still. The search tested a
prespecified doubling sequence from radius 0.001 through 0.1; it did not
find a sampled radius meeting all pooled and local hard limits. The pilot
does **not** establish that no feasible radius exists between sampled points
or along another parameter direction. Per-radius hard counts were not saved,
so the precise failing scope at radius 0.1 is unknown.

The final `local_capped_first` allocator is separate from this raw-call side
step. Its code and independent tiny-case exhaustive tests enforce the pooled
and country ceilings. A label-free counterexample on the saved epoch-1
probabilities applied the same negative offset to every class-1 logit:
at offset -8, raw calls fell to one NLD item and met both cap levels, while
the final allocator selected **exactly the same items** at both levels as
before the shift. This follows because a uniform capped-class logit shift
preserves the order of capped-class probabilities. Raw feasibility alone is
therefore not a useful surrogate for improved identity of allocated items.

The fixed protocol's feasibility gate failed, so its 48-seed denominator
cannot be expanded or scored as if valid. A subsequent experiment must have
a separate, explicit protocol, release, seed block and output roots. It should
measure final allocated identity and quality against matched pooled-only,
sham and untouched controls, retain this failure, and avoid choosing settings
from the repeatedly inspected development labels.
