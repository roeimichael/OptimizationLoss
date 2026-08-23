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

0. 🟢🟢 **fMoW IS THE FIND, and it beats every camera trap on the measure that matters.**
   Satellite imagery, groups = **country**, whole countries held out entire:
   **NET +2969, z=79.7, NET/LOCAL 95.8%, 10 unseen countries**, and it survives §3's
   control at **100.0%** because a country is an ATOMIC group. Its per-group binding
   structure is *better* than iwildcam's -- **11 of 20 zero ceilings** on two capped
   classes of 408 and 511 test items, against iwildcam's 7 of 14 on 370 and 456. It
   leaves BOTH camera traps AND dermatology, and the images are **1.65 GB, already
   cropped to the AOI and resized to 224x224 JPEG**, which is our pipeline's format.
   **This is the acquisition to make.**
1. 🟢 **ISIC 2019 also clears stage 1** -- the first *medical* non-camera-trap set to do
   so -- but only in
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
5. 📌 **Acquire `fmow` first, `cct` second, `isic` third.** `fmow` is **1.65 GB** and is
   the only candidate that is independent of everything we hold; `cct` is ~8 GB, 27%
   fetched, wiring committed, and buys consistency rather than independence; `isic` is
   9.8 GB transient / 1.8 GB steady. All three take the dataset count from 1 to 4 and the
   *independent* count from 1 to 3.
6. 🛑 **The cheap fMoW route the shortlist assumed does not exist.** `danielz01/fMoW` on
   HuggingFace is **gated** (`HTTP 401, x-error-code: GatedRepo`) and there is no token on
   either machine, so its byte-exact WILDS parquet is not readable anonymously. The route
   that DOES work is `jbourcier/fmow-rgb-baseline`, whose per-image JSON sidecars ship
   **separately from the images** -- `val-metadata.tar.gz`, 52 MB, parsed to 63,422 rows
   in 8 seconds.

---

## 1. THE TABLE -- every candidate considered

**Provenance is marked per row.** All numbers are from `scripts.dataset_screen`, in
ITEMS. `A`, `B`, `C` and `L` are four invocations of the screen made during this run (`L` was run
locally while the ssh jump host was down; `A`, `B` and `C` on `dsisco01`); **all four contain
`iwildcam` and all three server runs contain `cct`, and every one returns them
bit-identically** (+3133/3531/994 and
+2546/2810/540), so all four are on one scale and NET/LOCAL may be read across them.
Nothing here is a stored number divided into a fresh one.

| dataset / slice | modality | group variable | NET | z | LOCAL | GLOBAL | **NET/LOCAL** | unseen | run | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **`fmow/country`** | **satellite** | **country** | **+2969** | **79.7** | **3100** | **1087** | **95.8%** | **10** | **C** | 🟢🟢 **PASS -- the strongest candidate measured, and independent of everything we hold** |
| `fmow/country_wide` | satellite | country | +2194 | 65.0 | 2470 | 1213 | 88.8% | 12 | C | 🟢 pass (a wider group-size band; kept as the seed check) |
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

🛑 **And there is no two-archive holdout to escape into.** Only BCN and HAM contain all
eight classes; the legacy archive and MSK4 never annotated AK, BCC, DF or SCC at all. So
a test set of two archives is either {BCN, HAM}, which leaves no training data, or one
that is missing half the label space. **The archive axis of ISIC 2019 is exhausted, not
under-explored.**

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

✅ **It passes its negative control on seven datasets.** An atomic group has no factors to
interpolate from, so it must be unaffected, and every one is:

| slice | group kind | NET (global base) | NET (additive base) | **survives** |
|---|---|---|---|---|
| `iwildcam` | atomic (camera) | +3130 | +3132 | **100.1%** |
| `cct` | atomic (camera) | +2546 | +2542 | **99.9%** |
| `idaho` | atomic (camera) | +2289 | +2288 | **100.0%** |
| `wcs` | atomic (camera) | +3444 | +3437 | **99.8%** |
| `serengeti` | atomic (camera) | +1645 | +1646 | **100.0%** |
| **`fmow/country`** | **atomic (country)** | **+2968** | **+2968** | **100.0%** |
| `fmow/country_wide` | atomic (country) | +2188 | +2191 | 100.1% |
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

**And this is a DATASET property, not a slice artefact, which is worth knowing before
anyone tries to slice-shop their way out of it.** Over ALL 25 BCN (site x age) cells of
>=150 images (11,461 images), the zero structure is:

| class | n in BCN | cells present | ZERO cells of 25 |
|---|---|---|---|
| NV | 3747 | 25 | **0** |
| MEL | 2573 | 25 | **0** |
| BCC | 2727 | 24 | 1 |
| BKL | 1076 | 24 | 1 |
| **AK** | **699** | 19 | **6** |
| **SCC** | **412** | 18 | **7** |
| VASC | 106 | 18 | 7 |
| DF | 121 | 17 | 8 |

No split of this dataset can give a frequent class a K=0 ceiling, because BCN is a
carcinoma referral clinic and BCC/MEL/NV appear in essentially every subpopulation.
📌 **But the 86 AK and 135 SCC test items are a slice choice, not a ceiling**: BCN holds
699 AK and 412 SCC in total, and `prep_iwildcam`'s inherited objective maximises the
RAREST test class (VASC/DF here), not the capped ones. A campaign that wants more mass on
`{AK, SCC}` can retarget the split -- and should declare that it did, since choosing the
split by the capped classes is a design choice, not a neutral default.

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

**Disk, and the acquisition is VERIFIED without acquiring anything.**
`ISIC_2019_Training_Input.zip` = **9,771,618,190 B (9.77 GB)**, verified by HTTP HEAD.
✅ Its central directory was then read over HTTP **range requests** (3.4 MB, no images):
**25,334 entries**, laid out as `ISIC_2019_Training_Input/ISIC_0000000.jpg`, and **all
12,174 images the slice asks for are present, 0 missing**. `prep_isic.collect()` matches
on `os.path.basename(member)`, so the directory prefix is handled. There is no way for
the 9.77 GB fetch to land and then find nothing -- which is the failure `smoke_arms`
exists to catch on the training side and which nothing was catching here. `prep_isic --meta-only` costs **2.5 MB**. A full acquisition extracts only the
slice's members and writes **1.83 GB** of 224x224 uint8 npy; delete the zip and steady
state is 1.83 GB, peak ~11.6 GB. The 3.82 GB test zip is **not needed** -- ISIC 2019 test
labels were never released, and 2(n) requires us to build the split by group anyway.

---

## 4b. 🟢🟢 fMoW IN DETAIL -- the one candidate independent of everything we hold

**How it was acquired, and the shortlist's route was wrong.** 2(n) lists FMoW as blocked:
the WILDS CodaLab bundle is gone, and the popular HF mirror `EVER-Z/fMoW_rgb` has a
**corrupt `category` column** (truncated at the first underscore, merging `airport`,
`airport_hangar` and `airport_terminal` into one class). Both still hold. The route that
works was neither:

* ⛔ `danielz01/fMoW` (the byte-exact WILDS parquet) is **GATED**: `HTTP/2 401`,
  `x-error-code: GatedRepo`. No HF token exists on `dsisco01` or locally, so a columnar
  range read is not possible anonymously. **Recorded so the next attempt does not
  rediscover it.**
* ✅ `jbourcier/fmow-rgb-baseline` ships **metadata separately from images**.
  `val-metadata.tar.gz` is **52 MB** and holds 63,422 per-image JSON sidecars carrying
  `country_code` and `timestamp`; it streams and parses to a CSV **in 8 seconds**. That is
  the pre-acquisition screen 2(n) asks for, on the dataset 2(n) said could not have one.

🛑 **THE LABEL COMES FROM THE PATH, NEVER FROM A `category` FIELD.** Paths are
`<split>/<class>/<class>_<seq>/<class>_<aoi>/<class>_<seq>_<i>_rgb.json`, so the class is
unambiguous and the 2(n) truncation trap cannot bite. `false_detection` is dropped.

**The slice.** Group = `country_code`, whole countries held out entire, 8 classes (the
most frequent of 62, matching how the camera-trap slices are cut).

train 17,670 / test 3,442 · **10 unseen countries** (CAN, DZA, EGY, IND, IRQ, JPN, MEX,
NLD, PHL, TUR) · imbalance 1.7x · rarest test class 320. The training set is 17,670
against iwildcam's 20,000, so **no compute confound** -- unlike ISIC's 8,381.

**🟢 THE BINDING STRUCTURE IS BETTER THAN IWILDCAM'S.** A K=0 ceiling binds regardless of
sum slack, and here the classes that have them are also the well-sized ones:

| class | test n | countries present | **ZERO ceilings of 10** |
|---|---|---|---|
| **single-unit_residential** | **408** | 3 | **7** |
| **military_facility** | **511** | 6 | **4** |
| recreational_facility | 482 | 8 | 2 |
| parking_lot_or_garage | 336 | 8 | 2 |
| educational_institution | 474 | 8 | 2 |
| ground_transportation_station | 320 | 8 | 2 |
| place_of_worship | 546 | 9 | 1 |
| crop_field | 365 | 10 | 0 |

Capping `{single-unit_residential, military_facility}` gives **11 of 20 per-group ceilings
at zero** on classes of 408 and 511 test items. `iwildcam` gives 7 of 14 on 370 and 456;
`cct` 4 of 10 on 426 and 382; **ISIC only 4 of 20, and on classes of 86 and 135.** On the
property 2(n) selected iwildcam for, fMoW is the best dataset in this document.

**✅ LEAKAGE, all zero and all counted.**

| check | result |
|---|---|
| own train/test filename overlap | **0** of 19,554 |
| own train/test COUNTRY overlap | **0** |
| own train/test SITE overlap (`<class>_<seq>` prefix) | **0** -- a site sits in one country, so holding countries out holds sites out |
| `fmow` x `iwildcam` filenames | **0** (19,554 vs 22,943) |
| `fmow` x `cct` filenames | **0** (19,554 vs 22,985) |
| `fmow` x `isic` filenames | **0** (19,554 vs 12,174) |

**Disk.** `val-images.tar.gz` = **1,652,231,185 B (1.65 GB)**, and the README states the
images are **already cropped to the AOI and resized to 224x224 JPEG** -- our pipeline's
exact format, so no resize step and no surprise. The slice's 19,554 images are **2.94 GB**
of uint8 npy. **This is the cheapest acquisition of the three, by 5x.**
`train-images.tar.gz` (9.57 GB) and `train-metadata.tar.gz` (298 MB) are **not needed**:
the val split alone yields a 17,670-image training side, and we re-split by country
ourselves regardless of the original train/val boundary. 📌 `country_code_mapping.csv` is
absent from every ungated mirror but is reconstructible from the stock ISO-3166 table, so
its absence blocks nothing -- and grouping on `country_code` directly gives 176 levels
against region's 6, which is strictly the better holdout anyway.

⚠️ **THE ONE CAVEAT, stated rather than buried.** fMoW's per-country class mix is partly a
*collection* artifact -- which AOIs were annotated in each country -- not purely what
exists on the ground. That is also true of iwildcam (which cameras someone deployed), and
it is categorically different from the two failure modes this document rejects elsewhere:
there is no rule forcing uniformity (DomainNet's `quickdraw` at exactly 350/class) and no
rule capping classes per group (So2Sat caps 7 of its 17 classes per city, which makes the
per-city composition a construction artifact and is why it stays last). fMoW has neither.

---

## 4c. 🚨 THE FULL ISIC ARCHIVE -- and the LABEL-COVERAGE confound that decides it

The ISIC 2019 challenge file pools only 4 archives (§2). The **full ISIC Archive** carries
**553,019 images with an `attribution` field**, so the institution axis that is vacuous at
4 levels might be live at 11+. Its public v2 API is the only route: there is no bulk
metadata endpoint, and the `query` and `collections` parameters are **ignored** (verified
-- `count` stays 553,019 either way), so it is cursor pagination at 100/page, ~5,500
requests, ~75 minutes. No approved 7.8 MB metadata parquet could be located on
HuggingFace: the only repo matching "ISIC Archive" is `TobanDjan/isic_archive`, whose 55
files are image-bearing shards with no metadata sibling.

🛑 **AND THE FIRST THING THE METADATA SAYS IS THAT THE AXIS IS CONFOUNDED.** The archive
labels a wildly uneven fraction of each contributor's images, so the LABELLED subset is
not a random sample of any institution. Measured on the first 113,776 rows pulled:

| institution | rows | labelled (`diagnosis_2`) | **coverage** | classes |
|---|---|---|---|---|
| ViDIR Group, Med. Univ. Vienna | 5,466 | 5,466 | **100.0%** | 2 |
| Univ. of Athens | 1,801 | 1,801 | **100.0%** | 3 |
| Medical University of Vienna | 328 | 328 | **100.0%** | 2 |
| MILK study team | 11,720 | 11,720 | **100.0%** | 8 |
| Hospital Italiano de Buenos Aires | 1,635 | 1,635 | **100.0%** | 8 |
| Hospital Clínic de Barcelona | 19,593 | 18,063 | **92.2%** | 8 |
| *Anonymous* | 19,640 | 19,556 | 99.6% | 17 |
| **Memorial Sloan Kettering** | 16,287 | 3,582 | **22.0%** | 15 |
| **Sydney Melanoma Diagnostic Center** | 2,437 | 433 | **17.8%** | 5 |
| **Dept. of Dermatology, Hosp. Clínic de Barcelona** | 9,473 | 227 | **2.4%** | **1** |
| **Univ. of Queensland Diamantina Institute** | 10,799 | 182 | **1.7%** | 9 |

**Coverage spans 1.7% to 100%.** If institution A labels 100% of its images and B labels
1.7%, a per-group class distribution computed on the labelled rows is measuring *what each
site chose to annotate*, not what walked through its door. That is the same shape of
defect as octmnist's `synth_group = index % 3` and So2Sat's per-city class cap: **a
per-group distribution fixed by construction.** The Barcelona dermatology department is
the clearest case -- 227 labelled images of **one single class** out of 9,473. Screened
naively it would look like an enormous per-group novelty and would be pure annotation
policy.

🔑 **So the screen must be coverage-matched, and the restriction must be stated, not
performed by a silent `dropna`.** `~/_isic_archive_slice.py` keeps institutions with
**>=90% coverage and >=300 rows**, which is the 6 real institutions above, and prints the
full table above every run so the exclusion is auditable. Two further decisions, made
explicitly:

* **`attribution == "Anonymous"` is EXCLUDED.** It is not an institution -- it is the
  legacy MSK/UDA submissions with the contributor stripped, and treating it as one group
  would merge distinct sources into a fake one.
* **The label is `diagnosis_2`, not `diagnosis_3`.** `diagnosis_3` has 44 values archive-
  wide and 1-39 per institution; a class present at exactly one site would dominate any
  novelty statistic for structural reasons. `diagnosis_2` has 18 archive-wide and a much
  more comparable per-site set. Top 8 are taken, as everywhere else in this document.

⏳ **STATUS: the pull is in flight** (~113k of 553,019 rows at the time of writing) and a
chained job builds the coverage-matched slice and screens it against `iwildcam`, `cct`,
`fmow` and `isic/bcn` in ONE invocation the moment it lands
(`~/_isic_chain.log`). 📌 **Whatever it returns, it does not change the top
recommendation**: §4b's `fmow` is atomic-grouped, 5x cheaper, has better binding structure
and carries no coverage confound at all.

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
| **So2Sat LCZ42** | not screened, and deliberately: *"for each city we reduced the number of samples of each of the nonurban classes A to G to N_m"* -- the per-city composition of **7 of its 17 classes is set by the dataset's own capping rule**. That is octmnist's failure with a different mechanism. A NET over all 17 classes would be uninterpretable; if it is ever screened, restrict to the untouched urban classes 1-10 and say so |
| **fMoW via `danielz01/fMoW`** | ⛔ **GATED** (`HTTP 401`, `x-error-code: GatedRepo`), no token on either machine. The dataset is NOT rejected -- see §4b for the route that works |
| **fMoW via `EVER-Z/fMoW_rgb`** | corrupt `category` column, unchanged from 2(n). §4b takes the label from the PATH instead, where the trap cannot bite |
| **rxrx1** | unchanged from 2(n): every siRNA appears in every experiment by design |
| **Camelyon17** | 2 classes |
| **povertymap / globalwheat** | regression / detection |

---

## 5b. 🔁 RETROSPECTIVE -- the dermmnist institution axis was in the file all along

The coordinator's lead was that DermaMNIST hides HAM10000's acquisition source. **Checked
against our own code, and it is sharper than that: the column is in a CSV this repo
already downloads, and `create_slices.py` simply never reads it.**

`data/dermmnist/download_data.py` fetches
`combined_metadata_corrected-HAM10000_corrected.csv` (748 KB), whose header is
`lesion_id,image_id,dx,dx_type,age,sex,localization,dataset,split`. **`dataset` is the
acquisition source**, and its class mix is far more extreme than the body-site axis this
project did use:

| source | n | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|---|
| rosendahl | 2,259 | **13.1%** | 13.1% | 21.7% | 1.3% | 15.1% | 35.5% | 0.1% |
| vidir_modern | 3,363 | 1.0% | 6.3% | 14.1% | 1.5% | 20.2% | 54.5% | 2.4% |
| vidir_molemax | 3,954 | **0.0%** | 0.1% | 3.1% | 0.8% | 0.6% | **94.1%** | 1.4% |
| vienna_dias | 439 | **0.0%** | 1.1% | 2.3% | 0.9% | 15.3% | 79.7% | 0.7% |

`akiec` is 13.1% of one source and **absent from two**; `nv` runs 35.5% to 94.1%. Against
that, `loc_group` (body site) -- the axis that WAS used, and which scored +65 -- spreads
5.4x. **So the institution axis was not weak, it was unread.** ⚠️ The stock MedMNIST npz
carries images and labels only, which is why `scripts/prep_octmnist.py` had to invent
`synth_group = index % 3` in the first place; the corrected CSV restores what the npz
drops, and this repo has been downloading it since dermmnist was set up.

🛑 **AND IT WOULD NOT HAVE SAVED DERMMNIST. Measured, not argued:**

| slice | NET | z | LOCAL | GLOBAL | NET/LOCAL | unseen | survives §3 |
|---|---|---|---|---|---|---|---|
| `ham/source` (hold out 1 source) | **-86** | **-2.5** | 1427 | 1424 | -6.0% | 1 | 101.8% |
| `ham/source x localization` | +690 | 34.1 | 758 | 200 | 91.0% | 4 | **52.8% (+365)** |

The first row reproduces §2 exactly, on a different dataset: **four institutions is too
few.** Only `rosendahl` and `vidir_modern` carry all seven classes, so a legal holdout is
one institution, and with one test group the per-group and global shifts are the same
object -- NET goes NEGATIVE. The factorial fallback survives at 52.8%, giving **+365
items**: five times the body-site axis's +65, and an order of magnitude below `fmow`
(+2968) and `iwildcam` (+3132).

🔑 **How to read the old dermmnist null after this.** It was not measuring a weak axis
badly; it was measuring the weaker of two axes, and the stronger one tops out around
+365 items anyway. **The null stands. What changes is the reason** -- "the group variable
was wrong" is a better account than "the dataset had nothing", and it is the account that
generalises to the rule in §2: count the LEVELS of a group variable before trusting how
sharply they differ.

---

## 6. RECOMMENDATION

### Acquire, in this order

**1. `fmow` val images -- 1.65 GB, do this one first.**
The best candidate in this document on the two properties that decide: **NET/LOCAL 95.8%
with an ATOMIC group** (survives the control at 100.0%, so none of it is the baseline's
doing), and **11 of 20 per-group ceilings at zero** on capped classes of 408 and 511 test
items -- better than iwildcam's 7 of 14. Training side 17,670 images, so no compute
confound. Images arrive **already cropped and resized to 224x224 JPEG**. It is also the
cheapest of the three by 5x and the only one that leaves both camera traps and
dermatology. Metadata is already on the server (`~/_fmow/val_meta.csv`, 63,422 rows).

**2. `cct` images -- ~8 GB.**
Highest NET/LOCAL measured anywhere (90.6%), atomic groups (survives the control at
99.9%), 5 unseen cameras, 4 of 10 per-group ceilings at zero, on classes of 426 and 382
items. The metadata is already screened, `~/_cct_chunks` holds **2.2 GB of a partial
fetch**, and the wiring is committed at `3ea7d35f`. 🛑 **Deploy the wiring only after the
last `results/iwc2` run** -- `code_version` is a git hash.

**3. `isic` (BCN, site x age) -- 9.8 GB transient, 1.8 GB steady.**
A second independent modality if a third dataset is wanted. Weaker than `fmow` on every
axis (factorial group, thin zero ceilings on rare classes, a smaller training set), so it
is worth acquiring only after `fmow` lands -- not instead of it.

### The independence argument, stated plainly

Section 0 of the FRAMEWORK prices generality in datasets, and the clustered sign-flip
floor is p=1.000 at one dataset and p=0.500 at two. **`cct` moves the count but not the
floor's honesty**: same modality, same schema, same authors, same reader as `iwildcam`, so
a reviewer may collapse the two into one unit and the effective count stays 1. `wcs`,
`idaho`, `wellington` and `serengeti` have the same problem four more times over -- they
are cheap resolution, not generality.

**`fmow` is the candidate a reviewer cannot collapse into anything we hold**: satellite
imagery instead of ground photography, a country instead of a sensor, land-use classes
instead of species, and an entirely different acquisition chain. `isic` adds a third
family (dermoscopy, patient subpopulations) but shares "medical imaging" with nothing we
own, so it is independent too. With `iwildcam` + `fmow` the effective independent count is
**2** for the first time in this project; adding `isic` makes it **3**, and adding `cct`
raises resolution without raising that count.

📌 **And the ordering is not a coin flip.** `fmow` dominates `isic` on NET (+2968 vs
+1424 under the same control), on group kind (atomic vs factorial), on binding structure
(11 of 20 zero ceilings on 408/511-item classes vs 4 of 20 on 86/135-item classes), on
training-set size (17,670 vs 8,381) and on cost (1.65 GB vs 9.8 GB). There is no axis on
which ISIC is the better first buy.

### And the honest caveats a campaign designer must carry

0. **For `fmow`: the per-country class mix is partly a COLLECTION artifact** (which AOIs
   were annotated where), not purely ground truth about each country. So is iwildcam's
   (which cameras were deployed where). What makes it acceptable, and So2Sat not, is that
   no rule forces uniformity or caps a class per group -- see §4b.
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
| `fmow` `val-metadata.tar.gz` (per-image JSON sidecars, NO images) | server `~/_fmow` | 52 MB |
| HAM10000 corrected metadata CSV (the §5b retrospective) | local | 748 KB |
| derived candidate slices (CSV only) | server `~/_cand` | 18 MB |
| ISIC zip central directory (range reads, acquisition check) | server, not kept | 3.4 MB |
| ISIC Archive metadata pull, in flight (553,019 rows via the public v2 API) | server `~/_isic` | ~550 MB transfer, ~60 MB retained |
| **total downloaded** | | **~86 MB excluding the in-flight archive pull. No images, no GPU, no experiment.** |

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
