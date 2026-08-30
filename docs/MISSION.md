# MISSION -- make TraLO mathematically the best, and prove it

**This file is the resume point.** A fresh session reads it first, then
`docs/FRAMEWORK.md` section 3(0) (the status board). It is updated at the end of
every working session. If it is stale, that is a defect -- fix it before doing
anything else.

Last updated: **2026-08-30**.

---

## 0. THE GOAL, stated so it can be failed

Make **TraLO** the best of the constrained-optimization methodologies, on the
mathematics, and show it. Not "not worse". Not "wins on one metric on one
backbone at one cap". The bar the work is held to:

| axis | required | have now | gap |
|---|---|---|---|
| **datasets** | **3** | **1** (iwildcam) | `fmow` screened + passes the factorial gate, needs ~21k images. Third TBD |
| **backbones** | **3** | 2 in `dom1` (**both MobileNet**) + RegNet landing | ViTB16 at LOOSE caps is the hole |
| **constraint pairs** | **varied**, both **equal and unequal** local:global ratios | 3, **all loose**, only 1 unequal-binding | `margin2`'s matched 2x2 (4 tags, 2 budgets x 2 scopes) closes this the moment a GPU frees |
| **consistency** | wins across **regimes**, not one | wins at L80-L95 only; **loses at L20-L50** | the central open problem |
| **metrics** | ccF1 **and** macroF1 both defensible | ccF1 +, **macroF1 NEGATIVE** | the central open problem |

🛑 **Winning only at L80/L90 is not a result.** If TraLO loses at every other
constraint pair, the claim is "TraLO helps when the constraint barely binds",
which is not the thesis.

---

## 1. WHERE WE ACTUALLY ARE (read the numbers, not the vibe)

### What is established

- **`dom1` (384 runs, complete, LOOSE caps, MobileNetV2+V3).** TraLO is #1 of
  five on ccF1 / AP / AUROC, 6/6 cells each.
- **First campaign at equal dose.** All five trained arms 100.0%; `hounie`
  672/672, which previously ran at **1%**. No earlier dual-vs-dual number is safe.
- **The four lambda=0 nulls are byte-identical 24/24**, so the compute term is
  shared exactly and arm differences are the method.

### What is NOT established, and must be said every time

| claim | reality |
|---|---|
| TraLO > fioretto | **AP 3/6 cells, p=1.00. A coin flip.** |
| TraLO > alm | 4/6 on everything, p=0.69. Not shown. |
| TraLO > hounie | 6/6 AP+AUROC, p=0.031 -- **fails BH** |
| Anything survives correction | **0 of 20 contrasts.** Structural: at 6 cells the sign floor is 0.031, so best q = 0.62 |
| macroF1 | **-0.0022, 2/6 cells -- BELOW the reseed floor.** `hounie` is the only positive arm |
| TraLO enforces better | **REFUTED.** Pulls +6.2 items vs hounie +23.4. The WEAKEST of the four |
| Constraints ever satisfied in training | **0 of 696 epochs.** The post-hoc allocator does all of it |
| dom1 is the headline | **No.** FRAMEWORK 1-pre fixed **ViTB16** a priori; dom1 has none |

### The mechanism we currently believe

TraLO improves the **capped** classes (+10.7 items ccF1) by **damaging the
uncapped** ones (uncF1 -0.0077, 1/6), and the net on macroF1 is **negative**.
The gain is concentrated where the **global** scope binds (`L95_G80`: AP +0.0439
vs fioretto) and reverses where the **local** scope binds (`L80_G95`: -0.0084) --
**at an identical budget**, so it is scope, not tightness.

---

## 2. THE KNOB LEDGER -- what has been tried on TraLO itself

✅ = keep · 🟡 = live, unresolved · ⛔ = rejected, **do not retry**

| knob | verdict | evidence |
|---|---|---|
| `soft_count_mode: sum` (shipped) | 🟡 wins LOOSE, loses TIGHT | AP +0.0253 loose / -0.0572..-0.0933 tight |
| `soft_count_mode: uniform` | 🟡 the mirror: fixes TIGHT, weak LOOSE | `uniform1` -0.0754 -> +0.0030 |
| `soft_count_mode: margin` | ❓ **NEVER RUN** -- staged in `launch_margin2.sh` | the only arm that can reorder on the direct channel |
| `tralo_st` (hard-count value fix) | ❓ **NEVER RUN** -- same campaign | isolates VALUE from PLACEMENT |
| `straight_through` | ✅ keeps count value exact | -- |
| `constraint_grad_mode: normalize` | ✅ **required** -- `clip` gives a ~20x dose spread across duals | `check_parity` refuses `clip` |
| `--constraint-fp32` | ✅ removes the FP16 skipped-step dose loss | iwc3 lost 328/1044 without it |
| `tralo_head` (head-only) | ⛔ 1.7x floor, tie uninformative; masking does not freeze the backbone (90.4% step) | |
| `tralo_ortho` (CE-orthogonal) | ⛔ delivers **0.0%** of its guarantee in 16/16 conditions | `ortho_survival` |
| `tralo_coin` (random direction) | ❓ never run -- **the control** for any placement claim | in `launch_margin2.sh` |
| penalty-shape variants | ⛔ FRAMEWORK 2 | all measured worse |
| more constraint steps | ⛔ **worse** | 2(c) |
| dedicated constraint optimizer | ⛔ | 2 |
| joint objective | ⛔ overfits, -0.067 AP | |
| undershoot hinge | ⛔ not budget-equalized; +16.3% free fill | |
| finer granularity (LLP) | ⛔ refuted | |
| KL anchor | ⛔ deleted from the pipeline | |
| `select` arm | ⛔ worst measured, -22 items | |
| `rank` / `beta` arms | ⛔ null / rejected | |

🎯 **The next knob is `margin` + `st` + `coin`** -- `docs/launch_margin2.sh`,
**432 runs, 12 cells**, re-validated 2026-08-30 (`gen_campaign` emits 432,
`check_parity` PASSES), never fired. `margin` is the only untested corner of
the count-function 2x2 and the only arm whose per-item gradient is not a
function of `p_ic` alone -- every other penalty this project ships has the form
`f(sum_i p_ic)`, whose logit gradient `f'(S) p_ic(1-p_ic)` is a monotone map
and therefore **cannot move an item across another on the direct channel**.

Its cap grid is a **matched 2x2**, which is what makes it answer the regime
question rather than just adding cells:

| tag | K (cls 2 / cls 7) | budget | what is pinned |
|---|---|---|---|
| `L30_G50` | 111 / 137 | K/n=0.30 | the DISTRIBUTION across groups |
| `L50_G30` | 111 / 137 | K/n=0.30 | only the TOTAL |
| `L80_G95` | 296 / 364 | K/n=0.80 | the DISTRIBUTION across groups |
| `L95_G80` | 296 / 365 | K/n=0.80 | only the TOTAL |

Each row-pair imposes the **same total budget through a different scope**, so
scope is isolated with tightness held fixed. ⚠️ 7 of 14 per-group ceilings are
K=0, and a zero ceiling binds however much slack the sum has -- so
"global-binding" never means the local scope is off. Say "pinned vs free
distribution", not "local vs global".
⛔ Do NOT add `L30_G30`: at `L30_G50` the global K=185 sits above the local sum
111, so the global term is INERT and the two tags are ONE cap level.

---

## 3. THE STANDING RULES THIS WORK IS HELD TO

1. **Never idle.** A campaign running is not a reason to stop; it is a reason to
   do the cheap offline work beside it.
2. **Cells, not seeds.** 4 seeds cannot resolve any of these effects (46-91
   needed). Everything rests on sign consistency across cells. **>= 9 cells** for
   a `***`, and **>= 10** if more than a couple of contrasts are tested.
3. **Pre-register ONE primary contrast** before scoring. 20 contrasts at 9 cells
   still cannot survive BH. This is the cheapest fix in the project.
4. **Always quote the `tralo_reseed` floor beside any win.** A 6/6 sweep is not
   evidence when the RNG floor also sweeps 6/6.
5. **Always quote macroF1 beside ccF1.** ccF1 alone hides the uncapped damage.
6. **Read the logs, but never compare counts across arms from
   `training_log.csv`** -- the schemas differ (76/16/15/14 cols) and trained arms'
   logged counts disagree with their predictions. Use
   `final_predictions_raw.csv`. FRAMEWORK 3(0c).
7. **md5 the raw predictions** before reading any metric (`_raw` = model, plain
   = allocator).
8. **Update this file and FRAMEWORK 3(0) at the end of every session.** A knob
   that failed goes in the ledger so it is never retried.

---

## 4. THE QUEUE -- in priority order

Work top-down. When one finishes, score it, update sections 1-2 of this file
and FRAMEWORK 3(0), then start the next.

1. 🔴 **`dom1b` -> 9 cells.** Running (dsisco01 GPU 3). On landing:
   `full_panel --campaign results/dom1 results/dom1b --control clip` and
   `--control tralo_null`. **Pre-register the primary contrast BEFORE scoring.**
   It is also the out-of-sample test of the L95_G80 scope hypothesis on a
   genuinely different architecture.
2. 🔴 **Loose-cap ViTB16 at >= 6 cells.** The pre-registered headline backbone,
   currently 2 cells at loose caps. Decides whether `dom1` is about TraLO or
   about MobileNet. **Highest-value missing run.**
3. 🔴 **`margin2`** (`docs/launch_margin2.sh`, **432 runs, 12 cells**, ready
   and re-validated 2026-08-30). The next real TraLO improvement, and the only
   queued campaign that spans TIGHT + LOOSE + both scopes at ONE code_version.
   **Pre-registration is written into the script header and is now fixed** --
   one primary (`tralo_margin` - `tralo` on AP, >= 10 of 12 cells, p=0.0386),
   with regime-consistency, the reseed floor, scope, the `tralo_coin`
   placement control and **macroF1/uncF1** as named secondaries. It may not be
   edited again now that it is queued.
   ⛔ BLOCKED ON A GPU, not on readiness -- 2026-08-30 all 4 GPUs on dsisco02
   (nirgal, zehavid) and 3 of 4 on dsisco01 are other users'; the 4th is our
   own `dom1b`.
4. 🟡 **A tight-cap campaign that can actually resolve** -- the current tight-cap
   nulls are underpowered, not negative. Either more cells or a cap where the
   prize clears the noise (2(v): K/n=0.9 needs 7 seeds, L20 needs 2607).
5. 🟡 **Unequal L:G ratios beyond L95_G80** -- e.g. `L50_G20`, `L70_G40`, to test
   the scope hypothesis at other budgets.
6. 🟢 **`fmow` images (~21k)** -- the only route to dataset #2. Needs the user's
   go-ahead for the download.

---

## 5. RESUME PROTOCOL -- what to read, in order

A fresh session with no context should do exactly this:

```bash
# 1. state of the world -- 60 seconds
cat docs/MISSION.md                      # this file: goal, ledger, queue
sed -n '/^### 3(0)/,/^### 3(1)/p' docs/FRAMEWORK.md   # the live status board

# 2. what is running RIGHT NOW
for h in dsisco01 dsisco02; do ssh $h 'nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | while IFS=, read -r u p; do echo "$(ps -o user= -p ${p// /} 2>/dev/null)"; done | sort | uniq -c'; done
ssh dsisco01 'cd ~/optloss-domb && ~/anaconda3/envs/optloss/bin/python -m scripts.rig_status'

# 3. progress of every campaign
ssh dsisco02 '~/anaconda3/envs/optloss/bin/python - <<PY
import glob,json,os,collections
seen=collections.defaultdict(collections.Counter)
for t in sorted(glob.glob(os.path.expanduser("~/optloss-*"))):
    for c in glob.glob(os.path.join(t,"results","*","*","*","*","*","seed_*","config.json")):
        p=c.split(os.sep); seen[p[p.index("results")+1]][json.load(open(c)).get("status","?")]+=1
for k,v in sorted(seen.items(), key=lambda kv:-sum(kv[1].values())):
    print("%-14s %s"%(k,dict(v)))
PY'

# 4. gates, before ANY launch
python -m pytest tests -q          # must be 402 (bump when you add one)
python -m scripts.audit_config
python -m scripts.smoke_arms
```

**Then pick up item 1 of the queue that is not already running.**

### Reading a landed campaign, in this order and no other

```bash
python -m scripts.dose_landed <root>                        # FIRST. always.
python -m scripts.full_panel --campaign <root> --control clip
python -m scripts.full_panel --campaign <root> --control tralo_null
python -m scripts.family_split --campaign <root> --families tralo fioretto hounie alm
python -m scripts.log_health <root>                         # read 3(0c) first
python -m scripts.order_probe --campaign <root> --arm tralo
```
Then: **per-cell breakdown, never a pooled digit**; the **reseed row** beside
every win; **macroF1 beside ccF1**; and an **exact sign test** with the cell
count stated.

---

## 6. WORKING IN PARALLEL

The GPU is the scarce resource; context is the other one. While a campaign runs,
delegate independent read-only analysis to subagents (the user has standing
approval for this) and keep only the conclusions:

- one agent per landed campaign that has never been scored
- one agent per offline probe that prices a direction (`ceiling_screen`,
  `paired_noise`, `dataset_screen`, `factorial_control`, `straddle_probe`)
- one agent to re-audit a defect class already found once (inert flags,
  incommensurable logs, unequal dose, pooled digits hiding per-cell reversals)

Never delegate a launch, a `git push`, or anything that writes to `src/`,
`configs/` or `main.py` while a campaign is running.
