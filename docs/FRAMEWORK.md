# TraLO: current research protocol

Evidence reset: **2026-09-14**, at the user's request. This file replaces the
historical operational narrative. The project remains dual-constraint training.
Old experiments are reference material, not evidence for the new campaign.
Names such as `good`, `final`, `best`, or `equal` convey no validation status.

## Objective and evidence boundary

Establish whether training with TraLO's transductive constraints improves
classification under the same deployment constraints and compute budget as the
baselines. A win is a hypothesis, not a promised result. A valid negative result
is retained. Do not start a new research task to manufacture an advantage.

Use fresh, traceable runs after validation. Never mix historical and fresh runs
in an acceptance table. New seeds on repeatedly inspected labels are fresh
executions, **not an untouched test set**. Record which splits informed model,
metric, cap, and hyperparameter selection. Reserve untouched evaluation groups
when available; otherwise limit claims explicitly to exploratory evaluation.
Aggregate count information allowed by the transductive task must be disclosed
and supplied identically to every method; individual evaluation labels must not
enter gradients, checkpoint selection, or hyperparameter search.

## Metrics and comparisons

- Primary: **cc-F1**, the arithmetic mean of per-class F1 over the declared
  constrained classes, computed from the actual deployed predictions.
- Alongside it: macro-F1 over a fixed declared class set, constrained precision
  and recall, uncapped-class F1, and per-class supports. Fix zero-division behavior
  before scoring; do not silently omit difficult or absent classes.
- Report feasibility violations and compute as guardrails. Raw predicted counts,
  flips, or proximity to a cap are not classification quality. Captured true
  positives are an optional diagnostic, not the primary acceptance metric.
- Use both `clip` and `focal_clip`, rival duals `fioretto`, `hounie`, `alm`, and
  a matched zero-constraint control. Report all prespecified comparisons.
- Match data/split, architecture, initialization seeds, augmentation recipe,
  task optimizer type/nominal learning rate, deployment allocator and precision.
  Give averaging or checkpoint-selection improvements to the controls too.
- Start from the established 30-epoch compute budget: trained arms warm-up 1 plus
  constraint phase 29; post-hoc arms warm-up 30 plus 0. This is a comparability
  convention, not a theorem that these values are optimal. Change it only in an
  explicit matched protocol amendment, not implicitly per arm.
- The existing phase boundary reseeds and constructs a fresh Adam optimizer for
  the 29-epoch trained phase; post-hoc 30+0 has one continuous warm-up optimizer
  and shuffle stream. Equal task epochs are therefore **not identical optimizer
  or RNG trajectories**. `tralo_null` matches TraLO's boundary and code path,
  with no constraint updates and the same deployment clipper. Attribute a gain
  to constraint training only through that contrast as well as both conventional
  clippers and rivals. Do not silently change this schedule during cleanup.
- Baseline recipe: `constraint_fp32: true`, `constraint_grad_mode: normalize`.
  Hashes that differ do not prove a live objective: compare actual gradients.
- Atomic reported cell: dataset, backbone, cap, method, over matched seeds.
  Start with at least four seeds and at least two cap levels. Exploratory pilots
  may be smaller but cannot establish superiority. Never count caps or copied
  warm-ups as independent replication.
- Report mean, seed SD, paired differences and confidence intervals in native
  metric units. Define the resampling unit and scope of inference. Four seeds
  give weak uncertainty estimates. A reseed spread is a diagnostic, not a
  confidence interval or a proof of equivalence. Correct multiplicity for
  confirmatory comparisons; do not choose an inferential method after seeing
  which one declares a win. Bolded best means must not imply significance.
- `deployed_h2h` is the maintained deployed-prediction reporter. Historical
  panel and acceptance scorers are retired. Fresh identity, common deployment
  and fixed-class metric validation must finish before it reports a new result.
  Each report covers one complete frozen campaign. Independently dispatched
  campaign roots get separate reports; copied runs never count as extra seeds.
- Prespecified exploratory uncertainty: two-sided 95% Student-t intervals on
  within-cell seed-paired differences, conditional on the fixed inspected data
  and recipe. Show every seed delta and sample size; fewer than four seeds is
  a pilot. For fewer than two deltas or zero empirical variance, report the
  interval unavailable. Small-sample normality is unverified; these marginal
  intervals are not multiplicity-adjusted tests or automatic win verdicts.
  Formula references: [NIST](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm)
  and [SciPy paired differences](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html).

## Validation gates

1. **Software:** full regression baseline, real CLI execution, hand-computed
   loss/metric fixtures, autograd versus analytic/finite-difference checks,
   chunked/full gradient parity, AMP/nonfinite/zero-gradient paths, deterministic
   repeats within the same environment, and crash/restart behavior. Tests do not
   certify unseen data, mathematical novelty, or universal correctness.
2. **Data:** resolve actual array paths; check hashes, shapes, label alignment,
   class supports, grouping semantics, split overlap, exact image duplicates and
   available subject/source identities. Preserve missing provenance as an open
   issue. Training accuracy alone cannot diagnose test-cut saturation.
3. **Regime:** on development data, inspect actual per-group allocation cuts,
   mistakes inside the selected set, informative caps, and whether true positives
   outside it provide correctable headroom. Screen every backbone separately.
   Difficulty is necessary only insofar as it leaves a learnable cap question.
   Do not weaken a model or choose a slice because TraLO already wins there.
4. **Logs:** verify every quantity against the computation it describes. A record
   must distinguish phase/epoch, planned/attempted/applied/skipped updates, raw
   gradient versus applied parameter displacement, soft versus hard counts,
   pre- versus post-update state, per-scope integer budget/residual/multiplier/rho,
   training/development loss, nonfinite events, timing and checkpoint identity.
   Log actual values, not inferred percentages or stale cached predictions.
5. **Release:** commit tested source, record dependency/hardware versions, sync
   local and remote source by SHA-256, validate remote arrays and execute the
   relevant tests on the target environment. Git HEAD alone is not a dirty-tree
   source fingerprint. Never change campaign code after launch.
6. **First runs:** use two GPUs on one host initially. Inspect logs and outputs
   immediately; halt expansion on wrong dose, nonfinite updates, silent CPU use,
   leakage, saturation at the cap, or invalid logging. A third GPU is permitted
   only after first-run checks; total ceiling is three, not three per host.
7. **Monitoring:** before leaving a live campaign, attach a thread heartbeat.
   Read both-host processes plus new log/checkpoint progress. Notify on meaningful
   change, failure, completion, or a required decision; stay quiet otherwise.
   A heartbeat must not relaunch duplicate jobs or rewrite frozen source. SSH
   loss means monitoring is unavailable, not that a job stopped.

Current executable entry points (verify their help in the checked-out version):

```bash
python -m pytest tests -q
python -m scripts.preflight --before-launch
python -m scripts.audit_config
python -m scripts.run_campaign --root <campaign> --step verify
python -m scripts.run_campaign --root <campaign> --step launch
python -m scripts.run_campaign --root <campaign> --step firstrun
python -m scripts.run_campaign --root <campaign> --step score
python -m scripts.pred_integrity <campaign>
```

## Research changes and historical findings

Read [REJECTED.md](REJECTED.md) for evidence-qualified historical hypotheses.
An aggregate-count gradient **can** change rankings through shared model
parameters; there is no general impossibility proof here. Nor does improving
constraint satisfaction imply improving cc-F1. Price and test that link.

The current development-freeze instruction is to validate and test the retained
reference first; no new loss modification is required before those experiments.
If development resumes, propose one falsifiable TraLO modification with its derivative,
expected log signature, matched control, and failure criterion. Keep changes to
the dual-constraint direction. Separate algorithm changes from cleanup commits.
Do not repeat a failed setting without identifying what invalidated its test or
which materially different, independently justified condition is being tested.

The full former instructions remain locally in `docs/archive/reset_2026-09-14/`
and in verified recovery backups; see `docs/GIT_TRACKING.md`. They are not all
included in a fresh source-only clone.
They preserve hypotheses and provenance, including contradictory conclusions;
their commands, counts, prohibitions and winner claims are not current policy.
Use [MISSION.md](MISSION.md) for actual progress; no historical result is fresh.
