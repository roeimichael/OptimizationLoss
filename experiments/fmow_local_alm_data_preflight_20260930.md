# fmow2 data preflight for snapshot-PHR direction study

Read-only audit on dsisco02, 2026-09-30 05:53:51–05:53:55 UTC. The script
imported `tralo.fmow_yuval.load(root, include_pool_labels=False)` from release
`6783d8d2a4b5975566a5b74956f94d81ba101ee8`, using
`/home/dsi/michaer8/optloss-audit/data/fmow2/oodslice`. That loader rechecked
all six preregistered file SHA-256 values. This is a data audit, not a new
training run or evaluation score.

| Role | Images | Exact duplicate extra rows within role |
| --- | ---: | ---: |
| Train | 15,841 | 8 |
| Country-disjoint stop | 1,829 | 0 |
| Development | 1,673 | 0 |
| Reserved countries | 1,769 | 0 |

The audit partitioned `train_images.npy` using `roles['train']` and
`roles['stop']`, and `test_images.npy` using `roles['dev']` and its complement.
For every image row it hashed the raw uint8 RGB pixel bytes with SHA-256.
**No image hash was shared by any pair of the four roles.** Development sample
IDs were unique; its runner rows contained only `split`, `sample_id` and
`location`. The development CSV labels agreed with `test_labels.npy` at all
1,673 development indices, checked only for alignment and never scored.
Reserved-country image bytes were accessed for this integrity hash only;
reserved labels were not read, and no model predictions or metrics were
computed on reserved images.

The four duplicate image groups are confined to training indices
`[10319,10342]`, `[12716,12720]`, `[13541,13543,13553,13576]` and
`[13545,13546,13566,13578]`. They are all class 0 and each group stays in a
single country (ITA, BRA, ARG and ARG respectively). This is not cross-role
leakage, but it affects effective training sample count and must remain
disclosed. Do not deduplicate mid-study because that would change the fixed
recipe and seed comparability.

This preflight proves exact byte-duplicate separation, not patient or scene
identity separation beyond the supplied country roles. The new ALM release's
source/config/preprocessing byte parity remains a separate deployment gate;
the matched pilot must still prove PTO trajectory equality and all side-step
integrity checks before the full block.
