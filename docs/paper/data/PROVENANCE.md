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

### What `review_graft_2026-07.csv` is and is not, measured 2026-08-23

`cc_f1` in this file is `f1_for_class` -- the F1 of the ONE capped class, not a
macro over several -- so each value is exactly `2*TP/(K+n)` with integer `TP`,
`K` the emitted count and `n` the true count. That identity lets two properties
be checked from the CSV alone, with no run directories:

1. ✅ **The anti-windup arm is bit-identical to its host in all 24 cells**, on
   both `cc_f1` and `macro_f1`. The paper states this twice and explains the
   mechanism (the deployed checkpoint is selected before first feasibility, and
   restarts only act after it), and `tab_graft` prints `host` and `+restart` as
   two columns of the same six numbers. It is now enforced by
   `test_the_anti_windup_arm_is_identical_to_its_host_as_the_paper_states`,
   with a negative control that perturbs one value by 1e-9.
2. ⚠️ **The arms did NOT emit the same number of capped-class predictions.** In
   24 of 24 cells no single `(K+n) <= 6000` divides into every arm's reduced
   denominator -- in one cell the arms imply multiples of 321, 155, 40 and 161
   simultaneously, whose LCM is 64 million. So these cc-F1 values sit at each
   method's OWN operating point; they are not budget-equalized the way
   `scripts/full_panel.py` equalizes (fill to exactly K). That is the expected
   output of dual methods rather than a fault, but it means a cc-F1 gap between
   two arms here mixes allocation with quality, and it cannot be separated after
   the fact: **the raw predictions for this campaign no longer exist**
   (`evidence/` keeps predictions for `mcbar` and `multiclass` only), so the
   table cannot be re-scored at equal budget. Re-derive the property with
   `Fraction(v).limit_denominator(10**7)` on each `cc_f1`.

See `README_DATA.md` for the per-file detail on the corpus CSVs.

---

## THE FIGURES DO NOT ALL REGENERATE -- checked 2026-08-25

`CLAUDE.md` records that eight of the eleven **tables** rebuild from
`corpus_final.csv` byte-for-byte. **That was re-verified today and holds:** run
every `docs/paper/scripts/make_*.py` (with `--two-metrics` on
`make_main_table.py`) and `git diff docs/paper/tables/` is empty.

**The FIGURES are a different story, and nothing said so before.** The same run
rewrote six PDFs under `docs/paper/figures/`:

| figure | shipped | regenerated | delta |
|---|---|---|---|
| `fig_convergence` | 52,002 | 52,002 | **0** |
| `fig_datasets` | 245,169 | 245,169 | **0** |
| `fig_loss_shape` | 58,786 | 57,898 | -888 |
| `fig_octmnist` | 47,414 | 44,852 | -2,562 |
| `fig_mechanism` | 44,781 | 41,238 | -3,543 |
| `fig_deployment` | 86,202 | 81,585 | -4,617 |

**Two reproduce exactly and four do not, all four smaller.** The two that match
are what make this readable: they prove the pipeline is deterministic, so the
other four differ in CONTENT rather than in a timestamp.

🔑 **AND IT IS NOT THE DATA.** Every input those generators read is present
(`corpus_final.csv` and one `training_log.csv`, both on disk), and
`make_loss_shape_fig.py` **reads no data file at all** -- it draws the penalty
shape analytically -- yet still regenerates 888 bytes smaller. Nor is it the
toolchain: shipped and regenerated PDFs both carry
`Matplotlib v3.10.5` and the identical font subsets
(`CZRBLR+STIXGeneral-Regular`, `FHXFSG+TimesNewRomanPSMT`,
`FTXWZA+STIXGeneral-Italic`).

⇒ **The committed versions of those four figures were produced by earlier
versions of their generators.** What the paper shows is not what the scripts
now draw, and the difference has never been characterised.

🛑 **So an empty `git diff docs/paper/tables/` says nothing about the figures**,
exactly as it says nothing about `tab_ablation_complete`, `tab_deploy` and
`tab_oct_backbone`, which have no generator at all. Before any figure is
quoted, re-run its generator and look at the result; do not assume the
committed PDF is what the current code produces.

⚠️ Re-running them REWRITES the PDFs in place. Restore with
`git checkout -- docs/paper/figures/` unless the change is intended -- and the
two byte-identical ones will come back clean either way.

### WHAT differs, decompressed and read out

The PDF content streams were decompressed and their drawing operators and text
strings compared. `fig_convergence` is the control: **identical operator
profile and identical 103 text draws**, so the method sees a real difference
when there is one and none when there is not.

🛑 **THE SHIPPED FIGURES DRAW SERIES THE GENERATORS NO LONGER MENTION.**

| figure | text only in the SHIPPED pdf | text only in the CURRENT code |
|---|---|---|
| `fig_deployment` | **`joint (no edits)`**, **`global cap only`**, `MobileNetV2`, `-dataset mean (joint)` | `-dataset mean` |
| `fig_octmnist` | **`ALM`**, `MobileNetV2`, `RegNetY`, `-400MF`, ticks `0.05 / 0.10 / 0.15` | `RegNet`, ticks `0.04 / 0.08`, `0.3 / 0.5 / 0.7` |
| `fig_mechanism` | *(text identical, 135 = 135)* -- but **575 fewer line segments** | -- |
| `fig_loss_shape` | 18 fewer text draws; glyph fragments `h`, `m` | -- |

`grep -ril` over `make_deployment_fig.py` and `make_octmnist_fig.py` finds
**none** of `joint`, `ALM`, `global cap only`, `MobileNetV2` or `no edits`. The
deployment generator declares exactly one series, `label="per-dataset mean"`,
against the four the shipped PDF draws.

⇒ **`fig_deployment` as shipped shows a `joint (no edits)` arm and a
`global cap only` arm; `fig_octmnist` as shipped shows an `ALM` series. No
script in this repository draws any of them, AND THE CORPUS DOES NOT CONTAIN
THEM EITHER** -- `corpus_final.csv`'s `method` column holds exactly six values
(`danits_lp`, `fioretto_ldf`, `heuristic`, `hounie_rcl`, `tralo`,
`tralo_bounded`), which is precisely `make_deployment_fig.py`'s hardcoded
`METHOD_ORDER`. So those panels cannot be produced from the committed data by
the committed code at any setting: **their input no longer exists**, and that
is not recoverable, only recorded.

🛑 **AND THE SHIPPED FIGURE WAS DRAWN FROM A WIDER SLICE THAN THE ONE THE CODE
READS.** `make_deployment_fig.py` filters `sweep == 'paper_final'`, and that
sweep holds exactly the three backbones it names and the six methods it names:

| | in `sweep == 'paper_final'` (n=1944) |
|---|---|
| models | `MobileNetV3`, `RegNetY400MF`, `ViTB16` -- **exactly `BACKBONE_ORDER`** |
| methods | the six above -- **exactly `METHOD_ORDER`** |

So the generator drops nothing from its own slice. But the shipped PDF draws
**`MobileNetV2`**, which in this corpus lives only in *other* sweeps
(`paper_backbones` 288 rows, `blackwell_validation` 80, and eleven more) and is
reported separately by `make_backbone_tables.py` ("the 3 main-grid backbones +
MobileNetV2"). ⇒ the committed figure was built over a **different and wider
row set** than `sweep == 'paper_final'`, in addition to using series
(`joint`, `global cap only`, `ALM`) that no sweep contains at all.

⚠️ **CORRECTION, 2026-08-25.** An earlier version of this entry said the
generator "would silently drop a claimed backbone" because MobileNetV2 is in
`corpus_final.csv` but not in `BACKBONE_ORDER`. **That was wrong** -- it
compared the hardcoded list against the whole 7,574-row file instead of against
the `paper_final` slice the generator actually reads. There is no drop. The
real defect in that generator was a different one, and it is now fixed:

🔧 **FIXED 2026-08-25 -- an ABSENT bar was pixel-identical to this figure's
headline claim.** `piv` came from `groupby(...).unstack().reindex(...)`, so a
`(backbone, method)` cell missing from the corpus became `NaN`, and
`ax.bar` renders a `NaN` height and a `0.00` height to **byte-identical
pixels** (measured: both PNGs 236 bytes). This figure's entire claim is that
the post-hoc clippers sit at ~0.00 native satisfaction -- so a cell that had
simply *vanished from the data* would have drawn as an empty bar and read as
**evidence for the paper's claim**. `make_deployment_fig.py` now calls
`_require_full_grid`, which raises rather than draws, and prints the models and
methods it excludes instead of dropping them silently. The dot-overlay's
`except KeyError: pts = []`, the same swallow one layer down, is removed.
Gated by `test_the_deployment_figure_REFUSES_a_bar_it_has_no_data_for`.

🛑 **`joint` IS A REJECTED ARM** (FRAMEWORK 2: `joint_objective` holds the cap
98.8% of epochs and overfits, AP -0.067). A shipped figure displaying it is not
automatically wrong -- it may be shown *as* a negative result -- but it cannot
be checked against code that no longer produces it, and no one should assume
the panel means what the current script would draw.

⚠️ `fig_mechanism` is the subtler case: every label is identical and only the
GEOMETRY moved (-575 `l` operators). A reader diffing labels would call it
unchanged. The curves are what changed.

## The backbone tables cover fewer cap levels for MobileNetV2, and one is a DISCARD

`make_backbone_tables.py` emits `tab_backbone_generality.tex` and
`tab_deploy_backbone.tex`, and both regenerate **byte-for-byte**. Their W/T/L
triples sum to the number of cap levels behind each row, and those totals are
**not equal across rows**:

| backbone | W/T/L totals (Table A, the three datasets) |
|---|---|
| MobileNetV3 | 9 / 9 / 9 |
| RegNetY-400MF | 9 / 9 / 9 |
| ViT-B/16 | 9 / 9 / 9 |
| **MobileNetV2** | **7 / 6 / 5** |

Measured 2026-08-25, and **most of that is genuine coverage, not a defect**:
only 5 cap levels exist for `octmnist x MobileNetV2` and 7 for
`tissuemnist x MobileNetV2`, because MobileNetV2 lives in the `paper_backbones`
sweep rather than `paper_final`.

🔑 **But ONE of them is a discard, and the table cannot show the difference.**
`dermmnist x MobileNetV2 x L40_G40` **was run and was thrown away**: only 2 of
its 5 seeds survive the `.dropna()` that pairs `tralo` against the best
baseline, and `cell_gaps` requires 3. In the emitted table that is
indistinguishable from a cap level that never ran -- both simply shrink the
W/T/L total -- yet only the first is a caveat about the *analysis*.

⚠️ `dropna` has produced a scorer bug in this project before: it once ran over
ALL arms, so a lagging third arm deleted pairs from every comparison.

✅ **FIXED 2026-08-25 by making the generator SAY SO**, not by changing the
table -- both `.tex` files still regenerate byte-for-byte. `cell_gaps` now
prints every excluded cap level with its surviving seed count, and separately
reports the `no tralo / no baseline at all` branch (which removes three cap
levels from `dermmnist x ShuffleNetV2`, a row that does not reach the paper).
Gated by `test_the_backbone_table_SAYS_when_a_cap_level_is_excluded`, which
checks both directions: a thin cap level must be reported, and a healthy one
must not.

## 🛑 `tab_granular_asym`'s macro column rests on ONE seed in 8 of its cells

The cc-F1 columns pair TraLO against the best trained **dual**; the
`$\Delta$mac` column beside them pairs it against the best **clipper**. Each
survives its own `.dropna()`, so the two can be built from different seeds --
and `cell_stats` recorded `cc_n` but **not** `mac_n`, so nothing in the emitted
table disclosed it.

Measured 2026-08-25 over `corpus_final.csv`:

| selection | mismatched cells |
|---|---|
| `paper_final` / dermmnist, octmnist, tissuemnist | **0** |
| `tab_granular_asym`: dermmnist `paperv2_phase2` | **4** |
| `tab_granular_asym`: tissuemnist `g2_asym_tissue_aider` | **4** |

The 8 are exactly the four asymmetric caps -- `L20_G50`, `L30_G80`, `L50_G20`,
`L80_G30` -- on both datasets, every one of them `mac_n = 1` against `cc_n = 4`.
**A one-seed mean has no variance and no pairing power**, and it sits in a
column whose neighbours are four-seed paired means.

⚠️ **The main tables are unaffected** -- every `paper_final` cell agrees -- so
this is specific to the asymmetric-cap appendix table.

✅ `cell_stats` now records `mac_n` and prints a warning naming each cell.
`tab_granular_asym.tex` and every other emitted table **still regenerate
byte-for-byte**; the numbers are correct for what they are, and the defect was
that the table did not disclose the `n` behind one column.

🛑 **THE REMAINING DECISION IS NOT A CODE ONE.** Whether that column should
carry its `n`, be dropped, or be re-run at four seeds changes a shipped
appendix table, which is the professor's call. Gated by
`test_the_granular_table_SAYS_when_its_macro_column_has_fewer_seeds`, which
checks both directions -- it must warn on a thin cell and must not cry wolf on a
matched one.
