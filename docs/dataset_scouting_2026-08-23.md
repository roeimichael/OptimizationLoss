# DATASET SCOUTING, 2026-08-23 -- the search for a NON-camera-trap dataset

Scope: find datasets that satisfy `docs/FRAMEWORK.md` 2(n). Not a proposal to run
anything. `docs/FRAMEWORK.md` remains the only operational document; this file is a
measurement record and a shopping list.

**The problem this run was pointed at.** Every dataset that has cleared 2(n) so far is a
camera-trap corpus read through the same COCO-CameraTraps schema, several by the same
authors. They buy resolution; a reviewer may fairly call the whole family ONE
generalization unit. The valuable find is a usable dataset that leaves camera traps
entirely.

---

## 0. BOTTOM LINE

1. 🟢 **ISIC 2019 is the first non-camera-trap dataset to clear stage 1**, but only in
   ONE form: **BCN_20000 only, groups = (anatomical site x age band), held out entire.**
   NET **+1705 items, z=47.7, NET/LOCAL 84.9%, 10 unseen groups** -- third on NET/LOCAL
   behind `cct` (90.6%) and `iwildcam` (88.7%), and clear of every other camera trap.
2. 🛑 **The obvious ISIC design is DEAD, and it is the one anyone would have tried.**
   Grouping by acquisition archive (Barcelona / Vienna / MSK) and holding one out gives
   **NET -141 items, z=-2.7** against a GLOBAL of +5581. With a single held-out
   institution the per-group shift and the global shift are the same object, and 2(j)
   shut the global route permanently. **A cross-hospital split of ISIC tests nothing.**
3. 🚨 **A new failure mode was found, and it inflates NET by up to 5.7x.** `dataset_screen`
   credits an unseen group with the GLOBAL training prior. That is right for an ATOMIC
   group (a camera) and **too generous for a group built as a PRODUCT of factors that
   both appear in training** -- a model that has seen (head/neck, 60s) and (upper
   extremity, 70s) can interpolate (head/neck, 70s). Section 3 measures it and
   `scripts/factorial_control.py` now gates it. **Every camera-trap dataset is unaffected
   (99.8-100.1%). The best-looking ISIC variant lost 82% of its NET.**
4. ⛔ Fitzpatrick17k, DomainNet, Office-Home and ENA24 are **rejected with measured
   numbers**, not on reasoning alone. ENA24's previously unreproduced rejection is now
   reproduced.
5. 📌 **Acquire `cct` first and `isic` second.** `cct` is ~8 GB, is 27% fetched already,
   and its wiring is committed. `isic` is 9.8 GB transient / **1.8 GB steady** and is the
   only measured non-camera-trap survivor. Together they take the dataset count from 1 to
   3 and the *independent* count from 1 to 2.

---

## 1. THE TABLE -- every candidate considered

**Provenance is marked per row.** All numbers are from `scripts.dataset_screen`, in
ITEMS. `A`, `B` and `L` are three invocations of the screen made during this run (`L` was run
locally while the ssh jump host was down, `A` and `B` on `dsisco01`); **all three contain
`iwildcam` and both server runs contain `cct`, and every one returns them
bit-identically** (+3133/3531/994 and
+2546/2810/540), so all three are on one scale and NET/LOCAL may be read across them.
Nothing here is a stored number divided into a fresh one.

| dataset / slice | modality | group variable | NET | z | LOCAL | GLOBAL | **NET/LOCAL** | unseen | run | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| `cct/oodslice` | camera trap | camera | +2546 | 75.8 | 2810 | 540 | **90.6%** | 5 | A+B | 🟢 acquired (metadata), images pending |
| `iwildcam/oodslice` | camera trap | camera | +3133 | 96.3 | 3531 | 994 | **88.7%** | 7 | A+B | 🟢 **LIVE, in use** |
| **`isic/bcn` (site x age)** | **dermoscopy** | **body site x age band** | **+1705** | **47.7** | **2009** | **1110** | **84.9%** | **10** | **B** | 🟢 **PASS -- the recommended non-camera-trap buy** |
| `idaho/oodslice` | camera trap | camera | +2291 | 77.0 | 2946 | 1262 | 77.8% | 3 | A | 🟡 pass, 4 of 8 classes are camera artefacts |
| `wcs/oodslice` | camera trap | camera | +3440 | 103.7 | 4640 | 2611 | 74.1% | 5 | A | 🟡 pass, biggest NET but 26% of it is global |
| `wellington/oodslice` | camera trap | camera | +1331 | 52.3 | 2027 | 1093 | 65.7% | 2 | A | 🟡 pass, only 2 test groups |
| `serengeti/oodslice` | camera trap | camera | +1646 | 55.9 | 2914 | 2136 | 56.5% | 5 | A | 🟡 pass, weakest differential fraction |
| `isic/siteage` (BCN+HAM) | dermoscopy | body site x age band | +2169 | 68.0 | 2204 | 464 | 98.4% | 7 | A+B | 🛑 **REJECT -- 82% of it is interpolable, see §3** |
| `isic/srcsiteage` | dermoscopy | archive x site x age | +1751 | 58.7 | 1811 | 360 | 96.7% | 10 | B | 🛑 REJECT -- 69% interpolable |
| `isic/srcage` | dermoscopy | archive x age band | +1537 | 56.1 | 2204 | 1401 | 69.7% | 7 | A+B | 🛑 REJECT -- 69% interpolable |
| `isic/srcsite` | dermoscopy | archive x body site | +1125 | 35.9 | 1777 | 1225 | 63.3% | 12 | A+B | ⛔ REJECT -- survives 72.7% but on an ANNOTATION artefact, see §4 |
| `isic/site` | dermoscopy | body site | +1488 | 33.7 | 3303 | 3223 | 45.0% | 2 | L | ⛔ REJECT -- 98% of LOCAL is global shift |
| `isic/src` | dermoscopy | acquisition archive | **-141** | **-2.7** | 5576 | 5581 | -2.5% | 1 | L | ⛔ **DEAD -- the cross-hospital split, §2** |
| `fitzpatrick17k` (atlas x skin type) | clinical derm | source atlas x Fitzpatrick type | +369 | 11.8 | 582 | 332 | 63.4% | 6 | A+B | ⛔ REJECT -- 12% of iwildcam's density, no zero ceilings, URL-scrape acquisition |
| `fitzpatrick17k` (skin type) | clinical derm | Fitzpatrick type | +181 | 4.7 | 357 | 314 | 50.7% | 2 | B | ⛔ REJECT |
| `domainnet` | object photos/renders | domain | **+57** | **1.2** | 365 | 308 | 15.6% | 2 | A+B | ⛔ **DEAD -- z below 2, the screen's own threshold** |
| `ena24` | camera trap | **none** | -- | -- | -- | -- | -- | -- | -- | ⛔ **DEAD -- no group variable exists**, §5 |

⚠️ **`isic/siteage` is the trap this table exists to expose.** On NET/LOCAL alone it is
**98.4%, the best number in the table, better than iwildcam and cct** -- and it is the
worst ISIC variant once §3's control is applied. Ranking on NET/LOCAL is necessary and,
for a factorial group, not sufficient.

---

## 2. WHY THE OBVIOUS ISIC DESIGN IS DEAD

ISIC 2019 is 25,331 dermoscopic images, 8 classes, pooled from three archives. The
archive is not a column; it is recoverable from the `lesion_id` prefix:

| archive | n | AK | BCC | BKL | DF | MEL | NV | SCC | VASC |
|---|---|---|---|---|---|---|---|---|---|
| BCN_20000 (Barcelona) | 12,413 | 5.9% | 22.6% | 9.2% | 1.0% | 23.0% | 33.9% | 3.5% | 0.9% |
| HAM10000 (Vienna) | 10,015 | 1.3% | 5.1% | 11.0% | 1.1% | 11.1% | 66.9% | 2.0% | 1.4% |
| legacy ISIC archive | 2,084 | 0 | 0 | 9.5% | 0 | 16.2% | 74.3% | 0 | 0 |
| MSK4 | 819 | 0 | 0 | 23.1% | 0 | 26.3% | 50.7% | 0 | 0 |

The class mix differs by 4-5x between the two real hospitals, which looks like exactly
what 2(n) asks for. **It is not.** Holding out one archive leaves ONE test group, and
with one group "per-group deviation" and "global deviation" are the same statistic:
GLOBAL +5581, LOCAL +5576, **NET -141 (z=-2.7)**. The screen is not failing; it is
reporting correctly that a single-institution holdout is a pure prior shift, and 2(j)
proved a prior shift cannot reorder any top-K set.

🔑 **The generalisable rule: a group variable with very few levels cannot carry
differential information however sharply its levels differ.** "Hospital A vs hospital B"
is one multiplier per class. This is the same lesson as `shift_1`, reached from the
opposite direction.

**What works instead is a WITHIN-institution subpopulation split.** Restrict to
BCN_20000, group by `anatom_site_general x age_approx` decade, hold whole cells out. Both
factors are acquisition metadata, so the group is knowable at inference without the label
-- the same property a camera id has.

---

## 3. 🚨 THE NEW CONTROL -- `scripts/factorial_control.py`

**The defect.** `dataset_screen` gives an unseen group the GLOBAL training prevalence as
its baseline. 2(n) argues for this correctly: a model that never saw camera 501 holds no
prior for it. **That argument does not survive the group being a PRODUCT.** A model that
has seen (head/neck, 60s), (head/neck, 50s) and (upper extremity, 70s) can estimate
(head/neck, 70s) far better than from the global prior, so part of what the screen scores
as novelty is information the training set already carries -- the one thing 2(n) exists
to exclude.

**The control.** Replace each unseen group's baseline with the independence (raking)
estimate `p(c|f0) * p(c|f1) / p(c)`, both marginals from TRAINING, renormalised. Whatever
survives is novelty the factor structure does not already supply.

✅ **It passes its negative control on five datasets.** An atomic group has no factors to
interpolate from, so it must be unaffected, and every one is:

| slice | group kind | NET (global base) | NET (additive base) | **survives** |
|---|---|---|---|---|
| `iwildcam` | atomic (camera) | +3130 | +3132 | **100.1%** |
| `cct` | atomic (camera) | +2546 | +2542 | **99.9%** |
| `idaho` | atomic (camera) | +2289 | +2288 | **100.0%** |
| `wcs` | atomic (camera) | +3444 | +3437 | **99.8%** |
| `serengeti` | atomic (camera) | +1645 | +1646 | **100.0%** |
| `fitzpatrick17k` atlas x type | factorial | +369 | +337 | 91.4% |
| **`isic/bcn` site x age** | **factorial** | **+1704** | **+1424** | **83.6%** |
| `isic/srcsite` | factorial | +1123 | +816 | 72.7% |
| `isic/srcsiteage` | factorial | +1749 | +539 | 30.8% |
| `isic/srcage` | factorial | +1533 | +475 | 31.0% |
| `isic/siteage` (BCN+HAM) | factorial | +2168 | **+380** | **17.6%** |

🔑 **Why BCN-only survives and BCN+HAM does not.** Pooling the two hospitals makes the
site and age MARGINALS carry almost everything -- BCN is old, head/neck, carcinoma-heavy;
HAM is young, torso, nevus-heavy -- so the product of marginals reconstructs each cell and
the interaction is nearly empty. Within one hospital the site x age interaction is
genuinely non-additive (acral melanoma in the young, actinic keratosis on the elderly
head/neck), and it survives.

⚠️ **Read the ABSOLUTE items, not only the ratio.** The raking estimate is itself fitted
on training data, so on a small training set it is noisy and can be WORSE than the global
prior. Across four seeds of `isic/bcn` the ratio swings 50.4 / 68.7 / 83.6 / **128.1%**
while the surviving item count stays **+1024 / +1077 / +1424 / +2044**. The ratio is a
direction; the item count is the number. **Budget ISIC/BCN at ~1400 items, not 1705.**

**Seed stability of the headline, four seeds:** NET +1567 / +1596 / +1705 / +2031,
NET/LOCAL 84.9-100.1%, unseen groups 8-10. Not a lucky draw.

---

## 4. ISIC/BCN IN DETAIL -- what a campaign would actually get

Reproduce with (2.5 MB of CSV, no images, no GPU):

```bash
python -m scripts.prep_isic --out data/isic/oodslice --meta-only
python -m scripts.dataset_screen data/isic/oodslice
python -m scripts.factorial_control data/isic/oodslice
```

train 8,381 / test 3,793 · 8 classes · 10 test groups, **all unseen** · imbalance 34.6x ·
rarest test class 48.

**🛑 THE PER-GROUP ZERO-CEILING STRUCTURE IS THINNER THAN IWILDCAM'S, and this is the
real weakness.** A K=0 ceiling binds regardless of sum slack, which is why 2(n) chose
iwildcam. Per-group counts over the 10 test groups:

| class | test n | groups present | **ZERO ceilings** | usable as a capped class? |
|---|---|---|---|---|
| NV | 1660 | 10 of 10 | **0** | ⛔ local scope adds nothing over a divided global cap |
| MEL | 862 | 10 of 10 | **0** | ⛔ same |
| BCC | 665 | 10 of 10 | **0** | ⛔ same |
| BKL | 274 | 10 of 10 | **0** | ⛔ same |
| **AK** | **86** | 6 of 10 | **4** | 🟢 binds, but only 86 test items |
| **SCC** | **135** | 7 of 10 | **3** | 🟢 binds, 135 test items |
| **VASC** | **48** | 7 of 10 | **3** | 🟡 binds, 48 items is very thin |
| DF | 63 | 9 of 10 | 1 | 🟡 barely binds |

Against `iwildcam`, where every class is absent from **3 to 6** of the 7 test cameras and
the two capped classes hold 370 and 456 test items, and `cct`, with 4 of 10 zero ceilings
on classes of 426 and 382. **On ISIC the classes that make the LOCAL scope bind are the
rare ones.** A campaign here would have to cap `{AK, SCC}` -- 4 of 20 per-group ceilings
at zero -- and accept 86 and 135 test items per capped class. Capping BCC or MEL instead
would make the local scope a divided-up global cap, which is the failure 2(n) flags for
`cct`'s class 5.

**✅ LEAKAGE, checked rather than assumed.**

* filename overlap `isic` x `iwildcam`: **0**
* `isic` own train/test filename overlap: **0**
* `isic` own train/test **lesion_id** overlap: **0**. This one matters: ISIC ships up to
  **31 images of the same lesion** across 11,847 lesions, so a random split reproduces the
  dermmnist 38.7% leak exactly. `prep_isic` drops train images sharing a test lesion and
  prints the count (9 images at seed 0).
* **HAM10000 content in the recommended slice: 0 images.** HAM10000 is the source of
  `dermmnist`, which this project removed after it nulled, so a BCN-only slice keeps the
  new dataset disjoint from a dataset already measured to null. ⚠️ The rejected BCN+HAM
  variant was **40.6% HAM10000**; that is a second reason not to use it.

**Disk.** `ISIC_2019_Training_Input.zip` = **9,771,618,190 B (9.77 GB)**, verified by HTTP
HEAD. `prep_isic --meta-only` costs **2.5 MB**. A full acquisition extracts only the
slice's members and writes **1.83 GB** of 224x224 uint8 npy; delete the zip and steady
state is 1.83 GB, peak ~11.6 GB. The 3.82 GB test zip is **not needed** -- ISIC 2019 test
labels were never released, and 2(n) requires us to build the split by group anyway.

---

## 5. REJECTED, one line each, grounded in the criterion

| candidate | reason |
|---|---|
| **ISIC by acquisition archive** | one held-out institution makes per-group == global; NET **-141, z=-2.7**. §2 |
| **ISIC BCN+HAM (site x age)** | 82% of NET is reconstructible from the site and age marginals; §3 |
| **ISIC archive x site** | survives the control at 72.7%, but the surviving signal is that the legacy and MSK archives **never annotated AK/BCC/DF/SCC at all** -- annotation-protocol shift, not population shift. Same objection as `idaho`'s camera-artefact classes, and worse |
| **Fitzpatrick17k** | NET +369 at n_test 3,276 = **12% of iwildcam's per-item density**; only 1 of 8 classes has any zero ceiling, so the local scope barely binds; and images are URL-scrape-only with known dead links |
| **DomainNet** | **NET +57, z=1.2 -- below the screen's own z=2 threshold.** `quickdraw` holds **exactly 350 train images for every one of its 345 classes** (verified here: the set of its per-class counts is `{350}`), so `P(y \| quickdraw)` is uniform to the item -- octmnist's `index % 3` in a new costume. And **0 of the 1380 (domain x class) cells are empty**, so there is not one K=0 ceiling to bind on |
| **Office-Home** | not screened: all 65 classes provably appear in all 4 domains (per-domain minimum images-per-class 15/39/38/23), so the label support is identical across groups and the local scope is empty by construction. The Art-vs-rest difference is group SIZE (2,427 vs ~4,400), not class prior |
| **ENA24** | ⛔ **previously "REPORTED, not reproduced" in 2(n) -- now REPRODUCED.** Downloaded `ena24.json` (3.78 MB) and checked the union of keys over all 9,676 image records: `['file_name','height','id','width']`. **No `location` on any image.** The dataset has no group variable, so the local scope does not exist. Rejection stands |
| **rxrx1** | unchanged from 2(n): every siRNA appears in every experiment by design |
| **Camelyon17** | 2 classes |
| **povertymap / globalwheat** | regression / detection |

---

## 6. RECOMMENDATION

### Acquire, in this order

**1. `cct` images -- ~8 GB, do it first.**
Highest NET/LOCAL measured anywhere (90.6%), atomic groups (survives the control at
99.9%), 5 unseen cameras, 4 of 10 per-group ceilings at zero, on classes of 426 and 382
items. The metadata is already screened, `~/_cct_chunks` holds **2.2 GB of a partial
fetch**, and the wiring is committed at `3ea7d35f`. 🛑 **Deploy the wiring only after the
last `results/iwc2` run** -- `code_version` is a git hash.

**2. `isic` (BCN, site x age) -- 9.8 GB transient, 1.8 GB steady.**
The independence buy, and the only non-camera-trap candidate that survived measurement.

### The independence argument, stated plainly

Section 0 of the FRAMEWORK prices generality in datasets, and the clustered sign-flip
floor is p=1.000 at one dataset and p=0.500 at two. **`cct` moves the count but not the
floor's honesty**: same modality, same schema, same authors, same reader as `iwildcam`, so
a reviewer may collapse the two into one unit and the effective count stays 1. `wcs`,
`idaho`, `wellington` and `serengeti` have the same problem four more times over -- they
are cheap resolution, not generality.

**`isic` is the only candidate measured here that a reviewer cannot collapse into the
camera-trap family**: different modality, different acquisition physics, different group
semantics (a patient subpopulation rather than a fixed sensor), different label space,
and a different failure mode. With `iwildcam` + `isic` the effective independent count is
**2**, which is the first time this project has had more than 1.

### And the honest caveats a campaign designer must carry

1. **ISIC's usable information is ~1400 items, not 1705**, and its per-item density is
   ~35% of iwildcam's. It is 20x dermmnist's +65, so it is a materially better bet than
   anything the original three offered -- but it is not iwildcam.
2. **Its group is factorial, not atomic.** That is a real difference in kind, it is now
   measured rather than assumed, and it will not go away with more seeds.
3. **The capped classes must be `{AK, SCC}`** or the local scope carries nothing beyond a
   divided global cap. 86 and 135 test items is thin; check `verify_caps` and
   `reachability` before committing GPU.
4. **The training set is 8,381 images against iwildcam's 20,000.** BCN_20000 holds
   12,413 usable images and the slice spends 3,793 of them on test, so a campaign here
   trains a weaker model on a shorter budget. That is a confound against `iwildcam`, not
   against ISIC's own `_null` twin, so within-dataset contrasts are unaffected -- but do
   not read an ISIC-vs-iwildcam effect size difference as a dataset property.
5. **Stage 1 is still only stage 1.** dermmnist cleared it at +65 and nulled; iwildcam
   cleared it at +3131 and its representation channel then measured NEGATIVE. Nothing
   here predicts a win -- it establishes that the question is askable on a second,
   independent dataset. `scope_probe --calibrate` on a trained model is still the gate.

---

## 7. WHAT THIS RUN COST

| item | where | size |
|---|---|---|
| ISIC 2019 ground truth + metadata CSVs | server `~/_isic`, and locally | 2.8 MB |
| `ena24.json` (to reproduce the rejection) | server, since removed by a cleanup | 3.78 MB |
| `fitzpatrick17k.csv` | local | 4.1 MB |
| DomainNet split txts (4 domains) | local | 18.6 MB |
| derived candidate slices (CSV only) | server `~/_cand` | 18 MB |
| **total downloaded** | | **~29 MB. No images, no GPU, no experiment.** |

Server disk `~`: **142 GB free at start, 159 GB free at end** (a concurrent cleanup
freed space; this run consumed under 30 MB and no GPU was touched).

⚠️ **A concurrent process deleted `~/_cand` mid-run**, taking the four regenerated
camera-trap slices with it. They were re-screened from tarballs where needed; the run-A
figures above were captured before the deletion. The `.json` annotation files are gone
again, so re-screening a camera trap means re-downloading its LILA annotation
(`~/_screen_many.sh` still holds the URLs).

---

## 8. NEW TOOLING ADDED BY THIS RUN

* `scripts/factorial_control.py` -- the §3 control. Gate any product-shaped group
  through it before quoting a NET. Atomic groups return ~100% and cost nothing.
* `scripts/prep_isic.py` -- builds the slice, `--meta-only` for the 2.5 MB screen and a
  full run to extract only the slice's members from the 9.77 GB zip. Enforces group AND
  lesion disjointness and prints the leak count.

Neither is on `src.experiments.runner`'s import path, so both are safe to deploy while a
campaign runs.
