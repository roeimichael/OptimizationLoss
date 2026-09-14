# Repository tracking

Git contains maintained source, tests, experiment definitions, dataset preparation
code, versioned split metadata, manuscript sources and curated research notes.
It does not contain experiment outputs, run logs, model weights, local tool
backups, numerical paper result tables, compiled PDFs or generated figures.

`.gitignore` prevents future accidental additions. Previously tracked outputs
were removed from the index with `git rm --cached`; their local files were not
deleted. The one-time removal diff is expected. Future result changes should
not appear in `git status` or commits. Avoid `git add -f` for evidence.

## Evidence recovery

Before the September 14 cleanup, 176 tracked artifact files (12,255,418 bytes)
were copied into a verified ZIP. Every file was checked by SHA-256 both in the
ZIP and at its original path after removal from tracking.

Local backup, beside the repository:

`../OptimizationLoss-git-backup-20260914/`

- `untracked-artifacts.zip`: original repository-relative paths.
- `manifest.json`: per-file size/hash, original commit and recovery instructions.
- `before-cleanup.bundle`: Git history through the pre-cleanup snapshot.

The original local commit is also retained on
`codex/backup-before-git-hygiene-20260914`. This backup branch and bundle are
local recovery material and must not be pushed wholesale. Existing published
history remains intact; this cleanup does not shrink historical Git objects.

To restore local evidence, extract the ZIP into the repository root using its
stored relative paths, then verify the hashes against the manifest. Files already
exist locally, so normally no restore is necessary. For another machine, transfer
the evidence through the project's approved storage, separately from GitHub.
Do not overwrite newer evidence without comparing it to the manifest first.

## Fresh clones and reviews

A fresh clone includes generators and provenance notes but not historical paper
data, numerical tables or compiled documents. Restore the evidence package before
regenerating those artifacts. Some tables were maintained by hand: the backup
preserves those too; do not assume all results can be regenerated from scripts.
Research reports may link to ignored local receipts; transfer the receipts when
an audit needs their underlying measurements.

The six current iwildcam/fMoW/BCN split CSVs remain tracked because they define
which samples were used. BCN's tracked metadata do not imply launch approval.
`.gitattributes` identifies these generated metadata files for GitHub to collapse
in reviews. Archived prose is marked vendored for language statistics. Active
source and tests retain normal diffs.

Large pre-reset instruction copies remain in the local `docs/archive/` tree and
the verified backup. Their contents also have predecessors in existing Git
history. Current protocol, current state, and concise historical research notes
remain tracked; local archive copies are not active instructions.

## Lean-cleanup recovery

Further recoverable moves are under `../OptimizationLoss-history-20260914/`:

| Directory | Preserved contents | Files / bytes |
|---|---|---|
| `retired-executable-docs` | Obsolete archive launchers and campaign-state code | 6 / 107,795 |
| `retired-operational-docs` | Superseded PLAYBOOK and COVERAGE | 2 / 54,023 |
| `retired-cct-metadata` | Unused CCT train/test split CSVs | 2 / 1,154,882 |

Each directory has a manifest with original paths, sizes and SHA-256 hashes.
Every destination was checked against its pre-move hash and the active source
confirmed absent. Restore into a separate directory and verify the manifest
before copying any file back; never overwrite newer evidence blindly.
The retired runtime/probe/test sources remain in Git history, starting from
`be92d3c4`, and in the blob-verified `lean-source-before-be92d3c4-blobs.zip`
beside these directories. No image arrays or model checkpoints were removed.
