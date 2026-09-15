# ARCHIVE -- QUARANTINED. DO NOT READ THIS AS CURRENT EVIDENCE.

**Everything below this directory is historical. None of it is current. None of
it establishes a result, a winner, a rejection, or a protocol.**

If you are an agent resuming work on this project: **do not cite, quote,
summarise, or reason from anything in here.** Read
[`../../RULESET.md`](../../RULESET.md), [`../MISSION.md`](../MISSION.md) and
[`../LEDGER.md`](../LEDGER.md). Those three plus
[`../FRAMEWORK.md`](../FRAMEWORK.md) are the entire live record.

## Why this is quarantined and not deleted

`AGENTS.md` requires evidence to be preserved by recoverable archival. Nothing
here is deleted, and everything is recoverable in full from git history. It is
quarantined because reading it produces **wrong answers**, for three specific
reasons:

### 1. It is contaminated with retired datasets

The paper tree was written against datasets that are no longer viable. Reference
counts of `octmnist` / `dermmnist` / `isic` / `pathmnist` / `medmnist` / `bcn` /
`lesion` in the archived paper sources:

| File | Hits |
|---|---|
| `paper/main_rev.tex` | 115 |
| `paper/main_edited_by_roei.tex` | 112 |
| `paper/main_clean.tex` | 102 |
| `paper/main.tex` | 78 |
| `paper/HANDOFF_TRACK_B.tex` | 16 |
| `paper/data/PROVENANCE.md` | 10 |

**dermMNIST is removed for leakage** -- 38.7% of its test lesions appear in
train. **BCN is blocked on integrity** -- exact duplicate pairs cross train/test
with conflicting labels. **iwildcam is retired** -- it passes 2 of 8 candidate
conditions. Every number in the paper tree was measured through one of those.

### 2. The evidence was reset on 2026-09-14

At the user's request. Historical results and folder names establish no winners
and no failures. Names like `good`, `final`, `best` or `equal` carry no
validation status whatsoever. A table in here is not a result.

### 3. Its conclusions have been superseded or refuted

Several claims in the archived documents were overturned by later work -- the
strong impossibility conjecture is false, the allocator "fix" was cancelled on
real data, and at least four first-place TraLO calls were produced by arms that
were not running the treatment.

## What was salvaged, and where it went

The one part of the archive that survived review is the theorem package from the
pre-reset theory document. It is carried forward in **`../LEDGER.md` PART 2**:
no value-level selection, the budget as a scalar gain, the four duals as one
family, provable invariance in degenerate regimes, and the conditional harm
lemma. **Read it there, not here** -- the ledger version is the reviewed one and
states what was renounced.

The disposition of every historical hypothesis is in **`../LEDGER.md` PART 4**.

## Contents

| Path | Was |
|---|---|
| `paper/` | TMLR paper sources, figure and table scripts, data provenance. Contaminated per above. `main.tex` is the professor's file and is preserved byte-identical -- **never edit it**. |
| `audits/` | Point-in-time repository audit receipts, 2026-09-13 and 2026-09-14. |
| `research/` | Loss-research shortlist and mechanism probe. Its three candidates are carried forward in `../LEDGER.md` PART 5. |
| `reset_2026-09-14/` | The instruction files as they stood at the evidence reset. |
| `cleanup_2026-09-06/`, `track_b/`, `review/`, `warmup50_tables/`, `launchers/` | Superseded operational narratives and result tables. |
| `THEORY_pre-reset_2026-09-02.md` | The pre-reset theory document. Salvage is in `../LEDGER.md` PART 2. |
| `REJECTED_2026-09-14.md` | Folded into `../LEDGER.md` PART 4. |
| `TRALO_LOSS_RESEARCH_2026-09-14.md` | Folded into `../LEDGER.md` PART 5. |

## The rule

**A live document may point INTO this directory. Nothing in here may be treated
as pointing back out.** `tests/test_archive_quarantine.py` enforces that every
markdown file here carries the ARCHIVED banner and that no live document sources
a claim from here.
