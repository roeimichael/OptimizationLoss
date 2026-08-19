# Results provenance — what feeds what

**Read this before changing any experimental result.** It maps every number in
the paper back to the runs that produced it, and forward to every place a change
must land. Companion to `docs/PAPER_REVISION_TRACKER.md`, which tracks the open
work; this file is the wiring diagram.

---

## The chain

```
results/**/config.json + evaluation_metrics.csv     (the runs -- GONE, see below)
        |
        |  build_experiment_manifest.py                 <-- SCRIPT DOES NOT EXIST
        v
docs/paper/data/manifest/experiments.csv               (one row per run)
        |
        |  build_corpus.py [--verify]                   <-- SCRIPT DOES NOT EXIST
        v
docs/paper/data/corpus/corpus_final.csv         COMMITTED, 7,574 rows -- the
        |                                       generators' actual input
        |  docs/paper/scripts/make_*.py                 <-- COMMITTED 2026-08-19
        v
docs/paper/tables/*.tex   docs/paper/figures/*.pdf      (floats)
        |
        |  HAND-WRITTEN, NOT GENERATED  <-- the weak link
        v
docs/paper/main_edited_by_roei.tex prose        (numbers quoted in text)
```

⚠️ **Corrected 2026-08-19.** Every path above the corpus was stale: the
consolidation moved `paper/` to `docs/paper/` and deleted `docs/main.tex`, and
this file was carried forward unedited in the same commit that moved them, so it
pointed a reader into dead paths for eighteen days.

**What is true, verified by running it:**

| Hop | State |
|---|---|
| runs -> manifest | ⛔ `build_experiment_manifest.py` exists nowhere in the repo |
| manifest -> corpus | ⛔ `build_corpus.py` exists nowhere in the repo. Seven scripts read `corpus_final.csv`; **none has ever produced it** -- this file's own historical note records that, and it is still true |
| corpus -> floats | ✅ **eleven generators committed at `docs/paper/scripts/`. The EIGHT tables that have a generator regenerate BYTE-IDENTICAL to the committed `.tex`; the other three have no generator and never did, so `git diff` being empty after a regeneration says nothing about them** |
| floats -> prose | ⚠️ still manual, still the weak link |

So the corpus is a **frozen input**, not a reproducible artifact: you can verify
every float against it, and you cannot rebuild it from the runs, because the
runs are gone (`results/` is empty on both machines) and the two scripts that
built it are gone with them.

### The generators were lost twice

`.gitignore` line 104 records the first rescue -- out of the gitignored
`archive/` tree on 2026-07-31, "these are the only copies". The consolidation
then moved `paper/` to `docs/paper/` and the scripts did not come; they fell
back into `archive/`, which is gitignored wholesale. Re-homed again 2026-08-19.

**Regenerating with the archive copies would have silently reverted three
things**, all of which are now folded back in and verified:

1. **The tie-band bolding rule** in `tab_ccf1`. The shipped caption says bold
   means "within 0.005 of the best constraint-trained entry -- the tie band
   every comparison in this paper is adjudicated at -- so jointly bolded entries
   are *tied, not ranked*". The archive rule bolds the single best, which prints
   a **ranking** across the headline table where the project's own standard says
   there is a tie.
2. **`LP-LG` -> `Shifman`**, 8 occurrences across 5 generators: a baseline
   renamed throughout the paper.
3. The `resizebox` wrapper, the `tab:bbgen` caption sentence, and the width
   tuning this file listed under "known hand-edits" and never folded.

**Verify it yourself** -- this is the check, and it is cheap:

```bash
python docs/paper/scripts/make_main_table.py --two-metrics
python docs/paper/scripts/make_backbone_tables.py
python docs/paper/scripts/make_graft_table.py
python docs/paper/scripts/make_granular_tables.py
git diff --stat docs/paper/tables/          # must be EMPTY
```

⚠️ The six **figures** regenerate and run clean, but are **not byte-reproducible**:
matplotlib stamps a creation date, and two consecutive runs differ. Their
committed PDFs also predate the `LP-LG` rename, so regenerating them changes a
legend. Left alone deliberately -- that is a manuscript decision, not a cleanup.

> Historical note, and the reason this file exists: until 2026-08-13 **nothing
> wrote `corpus_final.csv`**. Seven scripts read it; none produced it. That is
> how the ALM arm came to be absent from every table with no check failing.

---

## Rule: what to do when a result changes

Applies to **every** result change — a rerun, a new arm, a corrected metric, a
campaign finishing. Work top to bottom; do not skip.

1. **Rebuild the manifest** (on the server, then copy down):
   `python paper/scripts/build_experiment_manifest.py -o /tmp/experiments_manifest.csv`
2. **Re-check coverage:** `python paper/scripts/coverage_report.py`
   Gaps must be zero, or knowingly accepted and written down.
3. **Verify before overwriting:** `python paper/scripts/build_corpus.py --verify`
   This must PASS. It proves the derivation still reproduces every shared row
   bit-for-bit. Only then run it without `--verify`.
4. **Regenerate the floats** — every generator, not only the one you think moved
   (see the table below for which read the corpus).
5. **Re-apply the structural hand-edits** listed under "Known hand-edits" below,
   or the regeneration will revert them.
6. **Propagate into the prose.** This is the step that gets forgotten. Use the
   checklist in the next section.
7. **Rebuild the PDF and look at it.** Not just `errors=0` — open the changed
   pages.
8. **Grep `\pending`** in `docs/main.tex`. None may survive into a submission.

---

## Propagating a changed number into the prose

A number rarely appears once. Before declaring a result updated, sweep all four
places it can live:

| Where | How to find it |
|---|---|
| The float itself | regenerated in step 4 |
| The float's **caption** | captions carry claims ("leads in all six cells"), and captions are inside the generator — check the generator, not the `.tex` |
| **Body prose** | `grep` the literal number, e.g. `grep -n "0\.038" docs/main.tex` |
| **Abstract, §1 contributions, Conclusion** | these restate headline numbers and are the most-often-missed; check them explicitly every time |

Then the derived claims, which do not contain the number but depend on it:

- **counts**: "in 23 of the 27 cells", "all six cells", "24 of 27"
- **superlatives**: "the best constraint-trained baseline", "the largest effect"
- **scope words**: "all three backbones", "the full grid", "every dataset"
- **significance**: sign tests, BH thresholds, bootstrap CIs — a changed set of
  comparisons changes the family size

**Worked example of the failure mode.** On 2026-08-13 Table 5 gained an isolated
hinge row (+0.032). The table was correct; two sentences quoting the *old* row
were not — one in §6 and its twin in Appendix B — and both survived a clean
build, because LaTeX cannot know a number is stale. They were caught by reading,
not by tooling. Assume the same about every future change.

---

## Which generator reads what

| Float | Generator | Reads | Needs rerun for ALM / MobileNetV2? |
|---|---|---|---|
| `tab_alm_regime` (Table 10) | `make_alm_regime_table.py` (via `analyze_alm.py`) | `corpus_final.csv` + `alm_results.csv` | **Yes** — MobileNetV2 joins when its ALM runs land |
| Table 1–2 (headline) | `make_main_table.py --two-metrics` | `corpus_final.csv` | **Yes** — and being split by metric |
| `tab_backbone_generality`, `tab_deploy_backbone` | `make_backbone_tables.py` | `corpus_final.csv` | **Yes** |
| `tab_granular_*` | `make_granular_tables.py` | `corpus_final.csv` | **Yes** |
| `tab_graft` | `make_graft_table.py` | `review_graft_2026-07.csv` | No (separate campaign) |
| `tab_ablation_complete` | *(no generator — hand-maintained)* | `ablation_complete.csv` | No |
| `fig_octmnist` | `make_octmnist_fig.py` | `corpus_final.csv` | **Yes** — plots all methods and per-backbone deltas |
| `fig_deployment` | `make_deployment_fig.py` | `corpus_final.csv` | **Yes** |
| `fig_convergence` | `make_convergence_fig.py` | `data/dynamics/` | Only if ALM is added to the convergence census |
| `fig_mechanism`, `fig_loss_shape` | `make_figs.py`, `make_loss_shape_fig.py` | `data/dynamics/`, analytic | No |
| `fig_datasets` | `make_datasets_fig.py` | images | Being removed (tracker item 8) |

All figure generators live in `docs/paper/scripts/` (re-homed 2026-08-19).

⚠️ Three tables in this list have **no generator and never did**:
`tab_ablation_complete`, `tab_deploy` and `tab_oct_backbone`. They are
hand-maintained, so "regenerate and diff" cannot check them at all -- the only
three floats in the paper where that is true.
Analysis scripts that emit LaTeX (`b5`/`b6`/`b7`) live in `src/evaluation/`.

---

## Known hand-edits that regeneration reverts

Measured 2026-08-13 by regenerating every table and diffing. **All numbers
reproduced exactly**; the drift is structural only:

**None. Folded in 2026-08-19 and verified: `git diff docs/paper/tables/` after a
full regeneration is empty.** What used to be here:

| File | Hand-edit | Folded |
|---|---|---|
| `tab_graft.tex` | `\tabcolsep` 5pt (generator emitted 2.6pt); caption/label above the tabular | ✅ |
| `tab_granular_*.tex` | float placement and `\tabcolsep` | ✅ |
| `tab_ccf1.tex` | tie-band bolding, `resizebox`, caption cross-refs | ✅ |

The standing instruction was "prefer folding these into the generators over
re-applying them by hand", and it was right: the drift sat here for six days and
then the generator holding the folded versions was lost, which is how the
tie-band rule nearly went back to printing a ranking.

**Folded in 2026-08-13, so no longer listed above:** both `tab_ccf1.tex` hand-edits (the `\resizebox{\linewidth}`
wrapper and the `tab:bbgen` caption sentence) now live in
`make_main_table.py`, so regenerating no longer reverts them.

**Fixed 2026-08-13, previously silent:** the generators emitted `Shifman`/
`Shifman-LP` while the manuscript says `LP-LG` throughout. The committed tables
had been hand-corrected and the generators had not, so any regeneration would
have renamed a baseline across the paper. Also `make_granular_tables.py` crashed
on Windows (UTF-8 delta to a cp1252 console) and could not be re-run at all.

---

## Files in `paper/data/`

| Path | What | Rebuilt by |
|---|---|---|
| `manifest/experiments.csv` | every run ever, with `config_path` | `build_experiment_manifest.py` (server) |
| `manifest/gaps.csv` | cells the target grid still lacks | `coverage_report.py --csv` |
| `corpus/corpus_final.csv` | aggregated metrics the generators read | `build_corpus.py` |
| `corpus/alm_results.csv` | the ALM arm (b3 + b3_full + b3_mnv2 + r1_almrh), one row per run | `extract_alm_results.py` (**server**), then copy down |
| `corpus/ablation_complete.csv` | component-ablation digest | hand-maintained |
| `corpus/convergence_census.csv` | epochs-to-feasibility, §5.4 | hand-maintained |
| `corpus/review_graft_2026-07.csv` | graft campaign | hand-maintained |
| `corpus/kl_sweep.csv` | KL sweep (material now struck) | hand-maintained |
| `corpus/imbalanced_baselines.csv` | Table 3: focal / class-balanced / logit-adjusted, campaign `b1` | `release_manifest_campaigns.py` |
| `corpus/native224_ham10000.csv` | Table 8 + App. D: native 224 HAM10000, campaign `b2_derm` | `release_manifest_campaigns.py` |
| `corpus/ablation_no_hinge.csv` | Table 4's hinge-removed arm, campaign `g5_hinge_oct` | `release_manifest_campaigns.py` |
| `corpus/mnv2_provenance.csv` | which campaign each MobileNetV2 cell came from | `mnv2_provenance.py` |
| `corpus/r2_seeds10.csv` | seeds 5--10 on the six tight-cap cells, all 7 methods | `extract_campaign.py` (**server**) |
| `corpus/r3_rerunvar.csv` | ten repeats of one configuration (determinism check) | `extract_campaign.py` (**server**) |
| `dynamics/` | per-epoch logs for the two dynamics figures | copied run dirs |

The three released campaigns are projections of `manifest/experiments.csv`, not
recomputations: same metrics, one row per completed run, with an `arm` column
read from each run's own directory. `verify_hinge_arm.py` documents why the
hinge arm cannot be refereed against `g5_component_ablation` (that campaign's
removal flags did not take effect: five of six arms land within 0.005 of `full`,
including `no_reset`, which the released `B_loo_ablation` block puts at +0.079).

See `README_DATA.md` for the per-file detail on the corpus CSVs.
