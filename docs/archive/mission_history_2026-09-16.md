# ARCHIVED -- NOT CURRENT EVIDENCE

**ARCHIVED -- NOT CURRENT EVIDENCE.** This file holds run state and
pre-registrations removed from `docs/MISSION.md` on 2026-09-16 during a
documentation consolidation. Every campaign described here is finished, dead or
superseded. Its FINDINGS were merged into `docs/LEDGER.md`; what remains below is
provenance only -- the record that each reading was fixed BEFORE its numbers
landed, and the recovery procedure that was followed during a host outage.

**Do not source a claim from this file.** The live record is `docs/FRAMEWORK.md`,
`RULESET.md`, `docs/MISSION.md` and `docs/LEDGER.md`.

---

## Stage 1 of the ranking pivot -- run state, 2026-09-15

Three campaigns on dsisco01 GPUs 1, 2, 3 (`rank1_MobileNetV3`, `rank1_MobileNetV2`,
`rank1_RegNetY400MF`, 40 runs each, budget 30, fmow2), in a SEPARATE worktree
`~/optloss-rank` pinned at `338110cc` so the four completed campaigns in
`~/optloss-probe` kept their frozen `source_inventory`. 20/20 pre-launch gates.

**The pivot, and why it was not another arm.** The user authorised a staged pivot
on 2026-09-15, having accepted that the count-based question is answered NO. A
count penalty reads the MULTISET while the allocator reads the RANKS; the
literature proves post-hoc thresholding is the OPTIMUM for selection-rate
constraints, with one crack -- it is optimal only when the score is Bayes-optimal
(Woodworth et al., COLT 2017). A training-time win is permitted only by improving
the score. `rank_clip` did that: a hinge around the per-group K-th order statistic
on TRAIN labels with the budget simulated, the cut left in the graph so items
compete for the K slots. No test label entered any gradient.

**STAGE 1 GATE, pre-registered.** Does gAP(`rank_clip`) - gAP(`clip`) > 0? gAP is
allocation-free, so it isolates the SCORE, and `rank_clip` deliberately carried no
constraint at all.

- Flat in all three backbones -> the ranking channel is dead. Stop and write the
  negative result.
- Moves -> Stage 2 is justified: differentiable top-K through the allocator, plus
  `rank_tralo` / `rank_tralo_null`, which were BUILT and gated but deliberately not
  run because their result is uninterpretable until the channel is shown.

Honest prior, recorded before the seeds landed: ~60% that gAP moves, ~25% that it
helps TraLO specifically more than its null. If it lifted the clipper too, that was
still the paper -- "for budgeted deployment the training signal that matters is the
ranking at the cut, not count satisfaction".

**First-run checks owed:** `AMP: float16 + GradScaler` (dsisco01), and that
`rank_clip` is NOT byte-identical to `clip`.

### Stage 1 ranking gate -- pre-registered reading, written 2026-09-15

Arms `clip`, `focal_clip`, `rank_clip`, `aug_clip`, `aug_rank_clip`. No new scorer
was needed; `scripts/rank_paired.py` gives per-cell paired gAP with the seed sd
beside it:

```
python3 scripts/rank_paired.py --glob 'results/rank3_*/*/*/*/*/seed_*' --a rank_clip     --b clip
python3 scripts/rank_paired.py --glob 'results/rank3_*/*/*/*/*/seed_*' --a aug_rank_clip --b aug_clip
```

**The mechanism check came free and was read FIRST.** `rank_paired` marks an arm
`(cap-inert)` when its probabilities are byte-identical across L80 and L90. Both
control arms are marked that way correctly. But `rank_frac` is read from the cap
(`warmup.py:165` -- 0.8 at L80, 0.9 at L90), so a LIVE ranking loss trains two
different models and `rank_clip` cannot be cap-inert.

| what `rank_paired` shows for `rank_clip` | what it means | what follows |
|---|---|---|
| `(cap-inert)` | **AMBIGUOUS, not death.** Until `323edf44` this was guaranteed by a CACHE bug -- both caps shared one `base_model_id`. In `rank3_*` that cause is removed and the two caps carry different digests, so cap-inertness would now mean the loss really did not reach the model. | if the digests differ and the arm is still cap-inert: a sixth dead flag, discard, do NOT read gAP |
| separate L80 / L90 rows | the loss moved the model | proceed to read gAP |

**Then, and only then, the gate: is gAP(`rank_clip`) - gAP(`clip`) > 0?** Read as
cells, never pooled -- 3 backbones x 3 constrained classes = 9 cells per contrast,
with `|mean|/sd` beside each.

- **>= 6 of 9 cells positive, on both contrasts** -> the ranking channel moves the
  score. Proceed to Stage 2.
- **mixed, or <= 3 of 9 positive** -> **NOT a refutation of the ranking channel.**
  This loss fires on 8 of 139 train groups and trains a 2.3rd-of-12 order statistic
  to serve a 41st-of-363 decision, so a null is confounded with a 29x estimator
  deficit. The next move would be group-batched sampling, a relaunch and therefore
  a question for the user.

## Run state -- rank3, 2026-09-15 (superseded)

**NOTHING RUNNING. dsisco01 GPUs 1, 2, 3 free by decision, not by accident** --
Stage 1 answered, Stage 2 a compute-budget question for the user. GPU 0 was
`dvorata1`; dsisco02 GPUs 1-3 were `liverty`.

- `rank1_*` -- DEAD. 24 usable / 16 failed per campaign on the unpack defect. Its
  72 CONTROL runs are valid and were used to measure the gAP noise envelope.
- `rank2_*` -- DEAD. Stopped by explicit PID at 2/40 on the warm-up cache defect,
  before it could write wrong numbers.
- `rank3_*` -- **COMPLETE. 120/120 runs, zero failures, score gates GREEN on all
  three.** 84 distinct models, 36 shared hashes all cap-invariant controls, zero
  rank-arm collisions.

🛑 The tree stayed pinned at `323edf44`. `rank_min_group` is declared an identity
key in git but deliberately NOT deployed there: changing `configs/protocol.yml`
would have broken `rank3`'s frozen `source_inventory`.

**Stage 1 verdict: the budgeted ranking loss does not supply the "which".**
Numbers and diagnosis are in the live LEDGER. The decision table that was open at
the time:

| option | what it tests | rough cost |
|---|---|---|
| Stage 2: differentiable top-K (Petersen / Xie / Berthet) | differentiate the SELECTION itself, so the gradient stays rank-dependent through the backward pass | build + 1 campaign, ~5.5h on 3 cards |
| Fix the dose first (group-batched sampler) | whether the 8-of-139-groups deficit was the binding constraint | sampler change + 1 campaign, ~5.5h |
| Val-split stopping rule | the only candidate that could produce a POSITIVE reportable result | retrain on 83% of train, ~5.5h |
| Stop and write the negative result | the mechanism package is a coherent, well-evidenced story | 0 |

## dsisco02 outage, 2026-09-16 18:15-21:28 -- diagnosis and recovery

Resolved: the host recovered on its own (no reboot, uptime 108 days), the trainers
were never killed, and `vit_a`/`vit_b` resumed. Kept because the diagnosis held up.

**Symptoms.** ssh to dsisco02 failed at the banner exchange while `ping` succeeded
and TCP 22 was OPEN from dsisco01, so the host was up and had not rebooted -- sshd
accepted the connection and never spoke. `vit_a` and `vit_b` were frozen at 21/84
each; two samples two minutes apart showed identical completion counts, queue-log
sizes and mtime against a ~10.3 min/run rate. One cause explains all of it: an
NFS/IO hang, with sshd blocking reading `/home` for auth and the trainers blocking
writing to `/home`.

| campaign | completed | running (stuck) | pending |
|---|---|---|---|
| `vit_a` | 21 | 1 | 62 |
| `vit_b` | 21 | 1 | 62 |
| `cap_a` | 0 | 0 | 90 |
| `cap_b` | 0 | 0 | 90 |

### Recovery runbook, followed in order

1. Confirm ssh answers: `ssh dsisco02 'echo ALIVE; uptime'`.
2. 🛑 **Confirm NO trainer of ours is alive before touching any config:**
   `pgrep -u michaer8 -f 'main\.py'` and `pgrep -u michaer8 -f queue_runner`. A
   live trainer may simply have UNBLOCKED; resetting its config to `pending` while
   it still holds the directory gives two writers to one run. Either wait, or stop
   it by EXPLICIT PID (INT, then TERM/KILL), scoped by `/proc/<pid>/environ`.
3. Reset ONLY the stuck run in each campaign. `scripts/reset_crashed.py` is
   conservative by construction: a run is eligible only with NO usable result (no
   `results.accuracy`, no `training_log.csv` of >= 5 rows). Dry run first, then
   `--apply`.
4. Re-validate: `python -m src.pipeline.campaign validate --root results/vit_a`.
   🛑 Do NOT re-freeze -- the stamp `55c1be530de9` must not move.
5. Relaunch with a **NEW label** so a fresh log file is created. Never reuse the
   old label: the previous log may still be held open, and rewriting a live log is
   what orphaned `vita_vit_a.log` earlier the same day.
6. Re-queue anything parked behind it, same pattern.
7. `~/camp_status.sh` needs no edit -- it resolves logs by `*_<campaign>.log`.

### The fallback pricing, if dsisco02 had stayed down

Repriced 2026-09-16 with a measured host ratio, replacing a guess ("15-20 min/run"
for ViT on dsisco01, which was unfounded -- no ViTB16 run has ever executed on
dsisco01). Outstanding ViT work was 126 pending (`vit_a`+`vit_b`) plus 180
(`cap_a`+`cap_b`) = 306 runs:

| where | rate | GPU-hours | wall time on 2 cards |
|---|---|---|---|
| dsisco02 (when it returns) | 10.4 min | 53 | **~27 h** |
| dsisco01 (fallback) | ~32 min | 163 | **~82 h (3.4 days)** |

The fallback additionally cost stopping two `bud_*` campaigns and produced a
different (backbone, HOST) unit that cannot pool with the 42 finished ViT runs.
**Recommendation: wait for dsisco02.** It returned.
