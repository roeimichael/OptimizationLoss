> HISTORICAL RECEIPTS — not fresh evidence after the 2026-09-14 reset. Source result trees have moved recoverably; see [the reset receipt](../2026-09-14-reset.md) for locations.

# Evidence receipts — 2026-09-13

See [the audit](../2026-09-13-repository-audit.md) for scope, limitations, fixes,
and the distinction between a software test pass and a scientific result.

These are derived receipts, not replacement raw predictions. Source campaigns
originally occupied these paths (now under `optloss-history-20260914/<worktree>/results`):

- `/home/dsi/michaer8/optloss-snap3/results/snap3`
- `/home/dsi/michaer8/optloss-score/results/clipsweep2`
- `/home/dsi/michaer8/optloss-bcn/results/bcn1vit`
- `/home/dsi/michaer8/optloss-bcn/results/bcn1vitseed`

Scoring used an isolated `/tmp/optloss-review-20260913-ZILZxX` directory on
`dsisco02`, exported from local commit `901de2de5b107436c15d4a1e417c3e820d1ff98f`,
with the working-tree `scripts/deployed_h2h.py` copied in. Its SHA-256 was
`364dff576b5e0edf550bd5b25d9f2e328cda98c13e24fedd5aa69758eb63df78`.
This includes the user's pre-existing arm filter/seed-coverage work and the
audit's numbered-control exclusion fix. No existing remote worktree was updated.
After the two review-derived reporting fixes, the final scorer's SHA-256 is
`97d50b760b532bce0325eee04dacfd0c049a3840bcda82338ffe4fb8e198a212`.
`scorer.patch` reconstructs it from the starting commit. All three final reruns
were compared programmatically against the corrected JSON receipts and their
rows were identical; `*-final.log` preserves the updated reporting as well.
`remote-environment-freeze.txt` records installed packages, not a tested portable
lockfile or proof of bitwise identity between GPU generations.

Each `*-corrected.log` is the actual CLI output; the adjacent JSON is its
machine-readable table. The ViT pooled table names both parent and extension
after ONE `--campaign` flag and explicitly limits arms to `tralo`, `fioretto`,
`hounie`, `alm`, `clip`, `focal_clip` and all three TraLO RNG streams. Thus its
L80/L90 comparisons use seeds 1–12 without the exploratory arms limited to four.
The two-seed L70 row and non-task rows are not evidence for the cap claim.

`paired-contrasts.json` retains paired differences, seeds, means and sample
standard deviations for items and official cc-F1. It was derived using
`scripts.deployed_h2h.collect`, `paired` and `ccf1` from the same gated campaigns.
No averaging across caps or datasets was performed. It contains descriptive
comparisons, not multiplicity-corrected significance or new licensed units.

`dataset-integrity.json` records server metadata SHA-256 hashes, array shapes,
class/group counts, label alignment and split-overlap checks. Image contents
were memory-mapped for shape checks, not hashed for duplicates. This is not a
complete leakage audit.

The initial staging directory lacked its server data symlink. Early diagnostic
tables therefore said `no_data`; those outputs are not the corrected receipts
here. A first ViT command repeated `--campaign`, causing argparse to read only
the extension (eight seeds). That diagnostic is likewise not the twelve-seed
result. Both mistakes were corrected before reporting the pooled result.
