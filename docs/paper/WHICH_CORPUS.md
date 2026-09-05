# 🛑 READ THIS BEFORE QUOTING ANY NUMBER FROM THIS FOLDER

**Every manuscript in `docs/paper/` is built on a corpus that no longer exists,
and it shares NO data with any experiment running today.**

Verified 2026-09-04 by grep over all five `.tex` files and the whole of
`docs/paper/data/`:

| | manuscripts in this folder | current experiments |
|---|---|---|
| datasets | `dermmnist`, `octmnist`, `tissuemnist` (+ AIDER, EuroSAT) | **`iwildcam` only** |
| mentions of `iwildcam` | **0** | all of them |
| mentions of the MedMNIST family | all five `.tex` | 0 in `results/` |
| corpus file | `docs/paper/data/corpus/corpus_final.csv`, 7,574 rows | not derived from it |

The two are **disjoint experimental generations**. A finding on one says
nothing about the other, in either direction.

## Why the paper's datasets are gone

* **`dermmnist` -- LEAKED.** 38.7% of its test set, and 67.3% of MELANOMA,
  appears in training. Removed.
* **`octmnist`, `tissuemnist` -- DEAD BY CONSTRUCTION.** Their group variable
  is `synth_group = np.arange(len(y)) % 3`, so the groups are i.i.d. draws from
  one distribution and a per-group count constraint is empty **by
  arithmetic**. They could never have tested the thing being tested.
* AIDER and EuroSAT are not in the current backbone/dataset set at all.

None of these are runnable. The images are not on the server and the loaders
are gone.

## What this does and does not mean

* ✅ **The manuscripts are not invalidated by anything found in the iwildcam
  work.** The 2026-09-04 dead-arm audit, the dose-gap quarantine, and the
  unit-ledger corrections touch `dom1` / `dom1b` / `equaldose1` / `vitdual*`
  and **nothing in this folder**.
* ⛔ **But the manuscripts have their OWN separate problems**, documented in
  `docs/FRAMEWORK.md` and not repeated here: the warm-up-50 regime in which CE
  saturates and every method ties; the dermmnist leak above; the absence of any
  lambda=0 control arm in the corpus; and cc-F1 numbers that are partly a
  budget measurement. **Do not quote a number out of here without checking
  those.**

## Which file is live

| file | what it is |
|---|---|
| `main_edited_by_roei.tex` | ✅ **the paper of record.** Edit this one |
| `main.tex` | the professor's file. **Never edit** |
| `main_rev.tex` | the revision `main_edited_by_roei` branched from |
| `main_clean.tex` | a de-marked-up snapshot |
| `HANDOFF_TRACK_B.tex` | Track B handoff, superseded |

Only the first two are live. A fix applied to either of the others changes
nothing anyone reads.

## If you came here confused

You probably saw "MedMNIST" or "dermmnist" while discussing current results.
That is this folder leaking into a conversation about `iwildcam`. They are not
related. **`iwildcam` is the only runnable dataset**; see `CLAUDE.md` and
`docs/FRAMEWORK.md` section 2(n).
