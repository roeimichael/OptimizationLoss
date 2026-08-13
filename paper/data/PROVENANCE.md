# Results provenance — what feeds what

**Read this before changing any experimental result.** It maps every number in
the paper back to the runs that produced it, and forward to every place a change
must land. Companion to `docs/PAPER_REVISION_TRACKER.md`, which tracks the open
work; this file is the wiring diagram.

---

## The chain

```
results/**/config.json + evaluation_metrics.csv        (the runs, on dsisco01)
        |
        |  paper/scripts/build_experiment_manifest.py   <- RUN ON THE SERVER
        v
paper/data/manifest/experiments.csv                     (one row per run, 10,938)
        |                                               carries config_path, so
        |                                               every row is traceable
        |  paper/scripts/build_corpus.py [--verify]
        v
paper/data/corpus/corpus_final.csv                      (what the generators read)
        |
        |  paper/scripts/make_*.py
        v
paper/figures/*.pdf   paper/tables/*.tex                (floats)
        |
        |  HAND-WRITTEN, NOT GENERATED  <-- the weak link
        v
docs/main.tex prose                                     (numbers quoted in text)
```

Only the last hop is manual, and that is exactly where staleness hides: a
regenerated table silently disagreeing with a sentence three sections away.
Everything above it is now reproducible and verifiable.

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

All figure generators live in `paper/scripts/` — verified repo-wide, the only
other matplotlib code is third-party reference material under `benchmarks/`.
Analysis scripts that emit LaTeX (`b5`/`b6`/`b7`) live in `src/evaluation/`.

---

## Known hand-edits that regeneration reverts

Measured 2026-08-13 by regenerating every table and diffing. **All numbers
reproduced exactly**; the drift is structural only:

| File | Hand-edit | Why |
|---|---|---|
| `tab_ccf1.tex` | `\resizebox{\linewidth}` wrapper | raw tabular overflowed by ~83pt |
| `tab_ccf1.tex` | caption sentence pointing at `tab:bbgen` | cross-reference added later |
| `tab_graft.tex` | `\tabcolsep` 5pt (generator emits 2.6pt); caption/label order | width tuning |
| `tab_granular_*.tex` | 2 cosmetic lines each | width tuning |

Prefer folding these into the generators over re-applying them by hand.

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
| `corpus/ablation_complete.csv` | component-ablation digest | hand-maintained |
| `corpus/convergence_census.csv` | epochs-to-feasibility, §5.4 | hand-maintained |
| `corpus/review_graft_2026-07.csv` | graft campaign | hand-maintained |
| `corpus/kl_sweep.csv` | KL sweep (material now struck) | hand-maintained |
| `dynamics/` | per-epoch logs for the two dynamics figures | copied run dirs |

See `README_DATA.md` for the per-file detail on the corpus CSVs.
