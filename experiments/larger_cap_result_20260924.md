# Larger caps: matched knee and CIFAR comparison

**Result:** more output slots raise some absolute constrained-class F1 scores,
but did not turn the TraLO count term into a repeatable benefit. This is an
exploratory development-set result from the fixed design in
[larger_cap_protocol_20260924.md](larger_cap_protocol_20260924.md), not a
held-out confirmation or an exact Kassif/Singer replication.

The knee development set has **826 images**, not 30–50. Its original caps
reserved 82 grade-3 and 16 grade-4 slots; the earlier 30–50 numbers were
TraLO's *raw* predicted counts in some seeds. Knee training has 757 grade-3
and 173 grade-4 examples among 5,778. The CIFAR development subset has
**2,000 images**, not the full dataset; ten selected classes were initially
limited to 10 slots each. We increased the caps by 1.5× to **123/24** on
knee and **15 per selected CIFAR class**. All methods within a row received
identical caps. The other classes were unrestricted.

The table gives four-seed mean constrained-class F1 points **after
`capped_first` exact-slot allocation**. This policy fills all specified
constrained slots, including when a model's raw predictions underfill them.
The 1× rows come from the earlier matched sample-aware experiment. Comparing
methods *within* a row is meaningful; comparing absolute F1 across cap rows
also changes the number of allowed predictions, so it is not a model-only gain.

| Dataset | Caps | Clipper | Sample-aware Null | Sample-aware TraLO | TraLO − Null |
|---|---|---:|---:|---:|---:|
| Knee | 82/16 | 38.04 | 43.35 | 43.17 | −0.18 |
| Knee | 123/24 | 38.24 | **44.99** | 44.23 | −0.76 |
| CIFAR subset | 10 each | **53.52** | 52.73 | 52.94 | +0.22 |
| CIFAR subset | 15 each | 60.25 | **60.47** | **60.47** | 0.00 |

At 1.5× caps, knee seed-paired TraLO − Null differences were +0.87, −1.96,
−1.96 and 0.00 points; mean −0.76, ordinary paired 95% t interval
approximately [−3.04,+1.51]. CIFAR differences were exactly 0.00 in all
four seeds for this metric. The intervals and zero differences are on the
already inspected development split and do not establish population equality.

Under **upper-bound correction**, which need not fill unused slots, the 1.5×
mean TraLO − Null constrained-class F1 differences were −0.51 on knee and
−0.43 on CIFAR. Raw predictions showed the same broad suppression pattern:
relative to Null, TraLO changed 29 knee predictions (8 wrong→right,
11 right→wrong, 10 wrong→other-wrong) and 38 CIFAR predictions
(13/16/9). After capped-first allocation, only 15 knee predictions changed
(5/5/5) and 9 CIFAR predictions changed (5/3/1). These are endpoint
comparisons between separately trained paths, not isolated causal effects of
one update. A same-state before/after constraint trace is needed for that.

The term was active: every CIFAR TraLO fit applied 15 count updates. Knee
fits applied 15, 13, 12 and 11, because with the looser caps some epochs
had no active excess. Each knee fit completed 460 supervised updates; each
CIFAR fit completed 800, with no skipped updates. Thus ``TraLO did nothing''
would be false, but the changed predictions did not provide a reliable
advantage under either reported allocation policy.

All 24 fits (two datasets × four seeds × three arms) completed on dsisco01
Quadro FP32 from immutable release
`e66df2ee0e264539f9ac5294dc47aeb38481c5d2`. The release passed 94
local and native regressions, source-byte identity on both DSI hosts, CLI and
free-GPU smoke checks. The pilot and full sets passed independent saved-file
hash, update-dose, shared warm-up/batch identity, exact-slot and scikit-learn
metric recomputation audits. Test sets were not scored. Run artifacts remain
under `/home/dsi/michaer8/tralo-rebuild/runs/larger-cap-20260924`;
local audit summaries are at
`C:/Users/roeym/.codex/rebuild-audit-20260922/larger_cap_audit_20260924.json`
and `larger_cap_flips_20260924.json` beside it. The prior-probability
offline 1×/1.5×/2× cap analysis is `cap_sensitivity_20260924.json` in the
same directory. Offline rescoring is not retraining.

Next: a cap is a task requirement, not an accuracy knob. Before scaling
further, define realistic costs/cap provenance and inspect what the count
update does to true and false candidates at the quota cutoff. A larger
CIFAR development split would reduce the granularity caused by 10–15 slots
per class; it requires a new data split and matched runs rather than
relabelling this experiment as larger-data evidence.
