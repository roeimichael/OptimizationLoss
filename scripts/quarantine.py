"""Make dead results READ as dead, and refuse to score them.

WHY THIS EXISTS AND WHY IT IS NOT `rm -rf`. Disk is not the problem here: the
NFS home is 31% used with 588T free, and `results/` is gitignored, so anything
deleted is unrecoverable. The problem is ANALYTICAL. A campaign that died,
or that ran at a fraction of its dose, or that targets a dataset which no
longer exists, looks from the outside exactly like one that is merely
unfinished -- `status: pending` on every remaining run. This project has been
bitten by that specific confusion more than once ("a dead arm reads as
pending"), and the cost is not wasted disk, it is a number quoted from a
campaign that should never have been scored.

So the rule here is:

  * NEVER delete a `completed` run. Completed runs are receipts. `iwc2` is the
    only evidence that ViTB16 under fp16 without `--constraint-fp32` loses
    25.4% of its dose; `selectrun` is the only evidence that the selection
    head costs 22 items. Deleting either would make a documented claim
    unverifiable, and the corpus already cannot be rebuilt.
  * DO delete a run that can never execute -- a `pending` config whose dataset
    has been removed from disk. It contributes nothing but a false "unfinished"
    reading.
  * DO correct a `running` status with no process behind it. That one is not
    clutter, it is false.
  * MARK the rest, in a file both a human and a tool can read, and make the
    tools refuse.

A quarantined campaign keeps its data and gains a `QUARANTINE.json` at its
root. `is_quarantined()` is what the scorers call; `--check` is what a launch
or a test calls.

    python -m scripts.quarantine --list
    python -m scripts.quarantine --apply            # writes markers, dry by default
    python -m scripts.quarantine --apply --execute  # actually writes/removes
    python -m scripts.quarantine --check <root>     # exit 1 if quarantined
    python -m scripts.quarantine --self-test
"""

import argparse
import glob
import io
import json
import os
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MARKER = "QUARANTINE.json"

# The registry. Each entry names a campaign, why it is dead, and what it is
# still good for -- because "dead" and "worthless" are different, and every
# entry below is still the receipt for something.
REGISTRY = {
    # ------------------------------------------------------------------
    # PARTIAL quarantine: the campaign is SCORABLE, but not for every
    # contrast. `scorable=True` with a non-empty `dead_arms` says exactly
    # that, and it exists because a blanket marker here would delete the
    # evidence behind the project's headline claim in order to describe a
    # defect that touches two arms.
    # ------------------------------------------------------------------
    "dom1": dict(
        reason="the SAME unequal constraint dose that quarantined `vitdual1`: "
               "`alm`, `tralo` and `tralo_uniform` attempt 29.00 steps/run "
               "while `fioretto` and `hounie` attempt 28.00, a 3.4% gap in "
               "the only phase the comparison is about. Every arm landed "
               "100% of what it ATTEMPTED, so no gate was red and only the "
               "DENOMINATORS differ. FRAMEWORK 2(z38), 2(z40).",
        keep_for="EVERYTHING NOT INVOLVING `fioretto` OR `hounie`. `tralo` vs "
                 "`clip` / `focal_clip` / `lp` / `alm` / `tralo_uniform` / its "
                 "own `_null` is at equal dose and is UNAFFECTED -- this "
                 "campaign carries two of the independent units behind the "
                 "headline. It is also the receipt for FRAMEWORK 2944 (the "
                 "four `_null` arms are byte-identical).",
        dead_arms=["fioretto", "hounie"],
        scorable=True),
    "dom1b": dict(
        reason="same unequal constraint dose as `dom1`: `alm`/`tralo`/"
               "`tralo_uniform` at 29.00 attempted steps/run against "
               "`fioretto`/`hounie` at 28.00. FRAMEWORK 2(z40).",
        keep_for="everything not involving `fioretto` or `hounie`; it is the "
                 "RegNetY400MF independent unit and the receipt for 2(z5).",
        dead_arms=["fioretto", "hounie"],
        scorable=True),
    "equaldose1": dict(
        reason="the campaign NAMED for equal dose does not have it: `alm` and "
               "`tralo` attempt 29.00 steps/run while `fioretto`, `hounie` "
               "AND `tralo_lam0` attempt 28.00. `tralo_lam0` is the extra one "
               "-- it is a lambda=0 arm that still gates its backward on a "
               "multiplier, so it loses epoch 0 exactly as the duals do. "
               "FRAMEWORK 2(z40).",
        keep_for="everything not involving `fioretto`, `hounie` or "
                 "`tralo_lam0`. `tralo` vs `clip` and vs `tralo_null` are at "
                 "equal dose, and this is the MobileNetV2 independent unit.",
        dead_arms=["fioretto", "hounie", "tralo_lam0"],
        scorable=True),
    "vitdual1": dict(
        reason="the four-dual head-to-head ran at UNEQUAL CONSTRAINT DOSE: "
               "`alm` and `tralo` attempted 29.00 steps/run, `fioretto` and "
               "`hounie` 28.00. Every arm landed 100% of what it ATTEMPTED, so "
               "no gate was red and only the DENOMINATORS differed -- both "
               "duals initialise their multipliers at exactly 0 and updated "
               "them AFTER the primal step, so epoch 0 formed no constraint "
               "gradient at all. 3.4% of the dose in the only phase the "
               "comparison is about, and UNDER `full_panel`'s 5-point refusal, "
               "so the number would have been quoted. Superseded by `vitdual2` "
               "at 7ce4ee5ac41e, where the dual update precedes the primal "
               "gate and all four arms take 29",
        keep_for="the receipt for the dose gap itself -- `dose_landed` on this "
                 "tree is the measurement that found it. ALSO the receipt for "
                 "the ViTB16 TASK WINDOW [0.80, 0.90] on both classes, "
                 "measured from its two distinct `tralo_null` models: those "
                 "are lambda=0, take no constraint step either way, and are "
                 "therefore UNAFFECTED by the dose fix. Its `L60-90` and "
                 "`L70-90` cells are the receipt that those caps are non_task",
        scorable=False),
    "vittask1": dict(
        reason="BOTH its cells are `non_task`. `configs.task_cells.classify` "
               "on ViTB16 returns non_task for L60-90_G95 and L70-90_G95: "
               "class 2 sits at K/n 0.600 and 0.700, outside the measured "
               "ViTB16 strict band [0.80, 0.90]. The cap does not pose the "
               "question the campaign was launched to answer, so no "
               "arm-vs-arm number from it is about the constraint. It was ALSO "
               "found stalled on 2026-09-04 -- 34 pending, 13 completed, 1 "
               "crashed, and no dispatcher process -- so the pending runs were "
               "dropped rather than resumed",
        keep_for="its four lambda=0 runs, which are valid ViTB16 reference "
                 "models (a null is cap-invariant, so a non-task cap cannot "
                 "spoil it), and the cells themselves as a second, independent "
                 "receipt that L60-90 and L70-90 are non-tasks on ViTB16 -- "
                 "`vitdual1` is the first. The one `crashed` run "
                 "(L60-90_G95/focal_clip/seed_2) is a stale `running` marker "
                 "already reaped by a prior quarantine pass, not a live "
                 "failure",
        scorable=False),
    "uniform1": dict(
        reason="NINE of nine cells fail the task window. Its caps are "
               "L20_G50 / L30_G50 / L50_G30 -- exactly the L20/L30/L50 regime "
               "FRAMEWORK 2(z16) closed -- and `classify` returns non_task on "
               "MobileNetV2 and RegNetY400MF (both classes outside every band) "
               "and no_strict_band on MobileNetV3. Parity, health and dose are "
               "all CLEAN (1044/1044 steps, zero collapse, zero non-finite): "
               "nothing went wrong mechanically. 252 runs measured the absence "
               "of a question, which is why this marker is here and not in the "
               "health section",
        keep_for="the `tralo_uniform` vs `tralo` count-function comparison at "
                 "FULL dose, which is a claim about the LOSS SHAPE and not "
                 "about whether the cap binds -- though note `tralo_uniform` "
                 "was separately REJECTED 0/4 once task cells existed. ALSO "
                 "the receipt that L20/L30/L50 are non-tasks on three "
                 "backbones at once, and 32 of its distinct models are shared "
                 "byte-for-byte with `dom1`, so the two are not independent "
                 "units",
        scorable=False),
    "taskwin1": dict(
        reason="staged WITHOUT --constraint-fp32 (`constraint_fp32: False` in "
               "all 8 tralo configs): `tralo` landed 20/29 constraint steps, "
               "69.0%, on amp=float16. Killed at 3/48 and regenerated as "
               "`taskwin2`, which lands 203/203 and 174/174 on the SAME host, "
               "same backbone and the same two cap tags",
        keep_for="the CLEANEST A/B on --constraint-fp32 in the project: "
                 "taskwin1 69.0% vs taskwin2 100%, with host, backbone, caps "
                 "and arms all held fixed. `iwc2` and `iwc3` compare across "
                 "designs; this pair does not",
        scorable=False),
    "uniform1_VOID_dose3.4pct_2026-08-25": dict(
        reason="ran at 3.4% of its constraint dose; superseded by `uniform1`",
        keep_for="the receipt for FRAMEWORK 2(u): an arm can land 1 of 29 steps "
                 "and still write `status: completed`",
        scorable=False),
    "iwc2": dict(
        reason="ViTB16 under fp16+GradScaler WITHOUT --constraint-fp32: landed "
               "173/232 constraint steps, 74.6%. `check_parity` is GREEN on it "
               "anyway -- only `dose_landed` sees this",
        keep_for="the only evidence that --constraint-fp32 is load-bearing on "
                 "ViTB16; superseded for results by `vitu1` at 100% dose",
        scorable=False),
    "iwc3": dict(
        reason="fp16+GradScaler with `constraint_fp32: False`: landed 716/1044 "
               "constraint steps, 68.6% -- a LOWER landing rate than `iwc2`, "
               "which was quarantined for exactly this. Superseded by `iwc4` "
               "(same design, 1044/1044, fp32 True)",
        keep_for="the receipt that fp16 without --constraint-fp32 silently "
                 "drops a third of the dose on a CNN",
        scorable=False),
    "iwc1": dict(
        reason="fp16+GradScaler with `constraint_fp32: False`: alm landed "
               "51.7%, fioretto 57.1%, tralo 66.8%, hounie 100% -- an "
               "arm-DEPENDENT dose spread, so its cross-arm ordering is a "
               "measurement of the GradScaler. It also carries no per-family "
               "nulls, the only live campaign with a trained arm lacking its "
               "own twin",
        keep_for="the receipt for the representation-channel finding, and for "
                 "how far an fp16 dose can diverge BETWEEN arms",
        scorable=False),
    "xfam1": dict(
        reason="fails `check_parity`: `run_code_version` splits by SEED, "
               "`9b89ce26d6bb` x142 against `9b89ce26d6bb-dirty` x182 (seed 1 "
               "clean, seeds 3-4 dirty). Its commit predates the "
               "`TRAINING_PATHS` fix `ca373f4e` by 53 minutes, so the dirty "
               "flag still diffed the WHOLE tree -- most likely a `scripts/` "
               "deploy, but unprovable after the fact",
        keep_for="the receipt that all four constraint terms are negative, "
                 "and the only TIGHT-cap source for the rival duals",
        scorable=False),
    "mc_sgd": dict(reason="dermmnist: leaked test set AND removed from disk. "
                          "32 pending, 0 completed -- nothing ever ran",
                   keep_for="nothing", scorable=False),
    "vit_diag": dict(reason="dermmnist: leaked and removed from disk; 40 of 49 "
                            "pending and unrunnable",
                     keep_for="8 completed diagnostic runs", scorable=False),
    "vit_ceskip": dict(reason="dermmnist: leaked and removed from disk; 46 of 48 "
                              "pending and unrunnable",
                       keep_for="1 completed run", scorable=False),
    "mnv3bar": dict(reason="dermmnist: leaked and removed from disk; 62 of 80 "
                           "pending and unrunnable",
                    keep_for="17 completed runs", scorable=False),
    "mc29": dict(reason="dermmnist: leaked test set (38.7% of test, 67.3% of "
                        "melanoma). Completed, but on data whose split is invalid",
                 keep_for="the receipt for the mc29 dose finding", scorable=False),
    "dosefix": dict(reason="dermmnist: leaked test set", keep_for="dose receipts",
                    scorable=False),
    "dualbar2": dict(reason="dermmnist: leaked test set", keep_for="the dual-bar "
                     "comparison, on invalid data", scorable=False),
    "selectrun": dict(reason="dermmnist: leaked test set",
                      keep_for="the receipt that `select` costs 22 items vs clip",
                      scorable=False),
}

# A dataset that is gone from disk. A `pending` run against one can never
# execute, so it is the one category safe to remove.
_UNSET = object()   # "not passed" -- distinct from None, which means "unknown"

DEAD_DATASETS = ("dermmnist", "octmnist", "tissuemnist")


def campaign_roots(home=None):
    home = home or os.path.expanduser("~")
    out = []
    for t in sorted(glob.glob(os.path.join(home, "optloss-*"))
                    + [os.path.join(home, "OptimizationLoss")]):
        for c in sorted(glob.glob(os.path.join(t, "results", "*"))):
            if os.path.isdir(c):
                out.append(c)
    return out


def _marker_at(root):
    """The marker on disk, else THE REGISTRY, which is the source of truth.

    🛑 THE REGISTRY FALLBACK IS LOAD-BEARING, NOT A CONVENIENCE. The
    `QUARANTINE.json` file is only the on-disk copy, written by
    `--apply --execute` on ONE host, while scoring happens in fourteen
    worktrees and on a laptop that has no `results/` at all. Checking the file
    alone would miss every entry whose marker has not been written there yet.
    Erring toward refusing is the right direction: a false refusal costs
    `--allow-quarantined`, a false pass costs a number in a paper.

    (Added to `is_quarantined` a second time on 2026-09-04 by someone who did
    not notice it was already here; the mutation test is what found the
    duplicate, because removing one copy changed no behaviour.)
    """
    p = os.path.join(root, MARKER)
    if not os.path.exists(p):
        name = os.path.basename(os.path.normpath(root))
        return REGISTRY.get(name)
    try:
        return json.load(open(p))
    except Exception:
        return dict(reason="unreadable %s -- treat as quarantined" % MARKER,
                    scorable=False)


def is_quarantined(root):
    """What the scorers call. Returns the marker dict, or None.

    🛑 CHECKS EVERY ANCESTOR, NOT JUST `root`. The marker sits at the
    campaign root, but scoring one backbone at a time is a normal thing to do:
    `--campaign results/iwc3/ViTB16` used to walk straight past the marker at
    `results/iwc3` and print a full, plausible panel for a campaign that landed
    68.6% of its dose. Fourteen campaigns carry a marker, so this was fourteen
    ways to score a dead campaign by adding one path component.

    The walk stops at a directory named `results` (or at the filesystem root),
    so it can never reach outside a campaign tree.
    """
    p = os.path.abspath(root)
    while True:
        q = _marker_at(p)
        if q:
            return q
        parent = os.path.dirname(p)
        if parent == p or os.path.basename(p) == "results":
            return None
        p = parent


EXTENDS = "EXTENDS.json"

# Verification is expensive (it globs and parses configs in two trees) and
# `campaign_name` is called once per run path, so the verdict is memoised.
_EXT_CACHE = {}


def _configs_by_cell(root):
    """One config per (model, dataset, cap, arm), with its seed.

    PER CELL, not "the first config in the tree". Sampling one config per
    ROOT compared `vitdual2/alm/seed_1` against `vitseed1/clip/seed_5` --
    different ARMS -- and reported sixteen differing knobs for two campaigns
    that are byte-identical where they overlap. The single-arm fixture in the
    self-test could not see it; the live tree refused on the first try.
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "*", "*", "*",
                                           "seed_*", "config.json"))):
        parts = os.path.normpath(f).replace(os.sep, "/").split("/")
        key = tuple(parts[-6:-2])          # model, dataset, cap, arm
        if key in out:
            continue
        try:
            with io.open(f, encoding="utf-8") as fh:
                out[key] = json.load(fh)
        except Exception:
            continue
    return out


def _seeds_of(root):
    """Every seed present in a campaign tree, read from the PATH.

    From the path rather than the config, because a pending run has nothing
    but its config and this has to work before anything has executed.
    """
    out = set()
    for d in glob.glob(os.path.join(root, "*", "*", "*", "*", "seed_*")):
        tail = os.path.basename(d)
        if tail.startswith("seed_") and tail[5:].isdigit():
            out.add(int(tail[5:]))
    return out


def _extension_is_verified(root, parent_root):
    """(ok, why). Do these two roots hold ONE experiment at disjoint seeds?

    Every (model, dataset, cap, arm) the two roots SHARE must carry configs
    that differ in `seed` and nothing else. An extension usually holds fewer
    arms than its parent, which is fine -- what may not differ is any arm they
    both claim to run.
    """
    if not os.path.isdir(parent_root):
        return False, "the named parent %s is not a directory" % parent_root
    child, par = _configs_by_cell(root), _configs_by_cell(parent_root)
    if not child or not par:
        return False, ("no readable config under %s"
                       % (root if not child else parent_root))
    shared = sorted(set(child) & set(par))
    if not shared:
        return False, ("the two roots share no (model, dataset, cap, arm); "
                       "there is nothing to pool")
    for key in shared:
        c, p = child[key], par[key]
        if c.get("code_version") != p.get("code_version"):
            return False, ("code_version differs at %s, %s vs %s -- two "
                           "different builds, not one experiment"
                           % ("/".join(key), c.get("code_version"),
                              p.get("code_version")))
        ch = c.get("hyperparams") or {}
        ph = p.get("hyperparams") or {}
        diff = sorted(k for k in set(ch) | set(ph) if ch.get(k) != ph.get(k))
        if diff != ["seed"]:
            return False, ("hyperparams differ at %s in %s; an extension may "
                           "differ in seed and in nothing else"
                           % ("/".join(key), diff or "nothing at all"))
    cs, ps = _seeds_of(root), _seeds_of(parent_root)
    if not cs or not ps:
        return False, "one of the roots holds no seed_* directory"
    if cs & ps:
        return False, ("seeds %s appear in BOTH roots. Pooling them would put "
                       "two runs under one (cell, seed, arm) key and average "
                       "a swept axis" % sorted(cs & ps))
    return True, ("seeds %s extend %s across %d shared cell(s)"
                  % (sorted(cs), sorted(ps), len(shared)))


def extension_parent(root):
    """The campaign `root` EXTENDS, VERIFIED, or None if it extends nothing.

    THE DEFECT THIS EXISTS FOR (2026-09-05). `add_seeds` writes seeds 5-8 to
    their OWN root, deliberately, so the parent keeps a green `check_parity`
    while its coverage is ragged. Its docstring then says the two roots pool
    because they share a protocol and a `code_version`. Nothing pooled them.
    `deployed_h2h` and `cell_table` key a cell by campaign NAME, so seeds 5-8
    formed a SEPARATE four-seed cell rather than extending the parent to eight.

    Not cosmetic. `vitseed1` exists to lift `vitdual2`'s RNG floor past
    `MIN_FLOOR_OBS`, the bar `deployed_h2h` refuses under, and as its own cell
    it could never do that: 40 runs, about 30 GPU-hours, aimed at a capability
    that did not exist. `seed58a` has sat beside `dom1b` the same way since it
    landed, and `dom1b` still scores at four seeds.

    DECLARATION IS NOT ENOUGH. A marker naming the wrong parent would pool two
    different experiments silently, which is worse than never pooling. So the
    claim is re-verified on every read: same `code_version`, hyperparams equal
    except `seed`, and seed sets DISJOINT. A marker that fails verification
    RAISES instead of degrading to "no pooling" -- a campaign that claims to
    extend something and does not is a staging error, and quietly scoring it
    as its own cell is exactly how it would go unnoticed.
    """
    key = os.path.abspath(root)
    if key in _EXT_CACHE:
        return _EXT_CACHE[key]
    path = os.path.join(key, EXTENDS)
    if not os.path.exists(path):
        _EXT_CACHE[key] = None
        return None
    try:
        with io.open(path, encoding="utf-8") as fh:
            parent = json.load(fh).get("parent")
    except Exception as exc:
        raise ValueError("%s is unreadable (%s). It claims this campaign "
                         "extends another; a claim that cannot be read cannot "
                         "be verified, and pooling on an unverified claim is "
                         "the failure this guard exists to prevent."
                         % (path, exc))
    if not parent:
        raise ValueError("%s names no parent." % path)
    ok, why = _extension_is_verified(key, os.path.join(os.path.dirname(key),
                                                       parent))
    if not ok:
        raise ValueError(
            "%s claims to extend %s, but it does not: %s. REFUSING to pool. "
            "Either the marker is wrong, or the two campaigns really are "
            "different and must be scored apart." % (path, parent, why))
    _EXT_CACHE[key] = parent
    return parent


def write_extends_marker(root, parent, out=sys.stdout):
    """Declare `root` an extension of `parent`, after verifying it is one."""
    key = os.path.abspath(root)
    ok, why = _extension_is_verified(
        key, os.path.join(os.path.dirname(key), parent))
    if not ok:
        print("REFUSED to mark %s as extending %s: %s" % (root, parent, why),
              file=out)
        return 1
    with io.open(os.path.join(key, EXTENDS), "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"parent": parent, "verified": why}, indent=2))
    _EXT_CACHE.pop(key, None)
    print("%s now pools into %s (%s)" % (root, parent, why), file=out)
    return 0


def campaign_name(root):
    """The CAMPAIGN this path belongs to, not its last path component.

    🛑 `paper_rows` gates by NAME, because it reads a CSV and has no path to
    walk. `cell_table` wrote `os.path.basename(root)` into that CSV, so
    scoring one backbone at a time -- `--campaign results/dom1/MobileNetV2`,
    a normal thing to do -- wrote `MobileNetV2`, and `by_name("MobileNetV2")`
    returned None. The name-keyed gate was defeated by one path component,
    exactly as the path-keyed one had been before `is_quarantined` learned to
    walk ancestors.

    So resolve the name the same way the marker is found: walk up to the
    nearest directory that carries a marker or matches a registry key, and stop
    at `results`. Falls back to the basename when nothing matches, which is the
    right answer for an unregistered campaign.
    """
    root_path = _campaign_root(root)
    # An `add_seeds` extension reports its PARENT name, so its seeds land in
    # the parent cell instead of forming a separate, under-powered one. The
    # claim is verified inside `extension_parent`, which RAISES rather than
    # pool two campaigns that are not one experiment.
    return extension_parent(root_path) or os.path.basename(root_path)


def _campaign_root(root):
    """The directory that IS the campaign for `root`."""
    p = os.path.abspath(root)
    while True:
        base = os.path.basename(p)
        if base in REGISTRY or _marker_at(p):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            return os.path.abspath(root)
        if os.path.basename(parent) == "results":
            # The child of `results` IS the campaign root by convention, so an
            # UNREGISTERED campaign scored one backbone at a time still reports
            # the campaign rather than the backbone.
            return p
        p = parent


def refuses_scoring(root):
    """The HARD refusal: the marker, but only when nothing may be scored.

    🛑 `is_quarantined` is INFORMATIONAL and returns any marker.
    A scorer must branch on this one instead, or a PARTIAL marker
    (`scorable=True` with `dead_arms`) would block a campaign whose other
    contrasts are perfectly good -- which would delete the evidence behind
    the headline in order to describe a defect touching two arms.
    """
    q = is_quarantined(root)
    if q and q.get("scorable") is False:
        return q
    return None


def dead_arms(root):
    """Arms whose CONTRASTS are invalid here, as a set. Empty when none.

    A partial marker names the arms the defect touched. Any contrast with one
    of these on either side is not comparable; every other contrast in the
    same campaign is untouched. Measured case: `dom1` ran `fioretto` and
    `hounie` at 28.00 attempted constraint steps against `tralo`'s 29.00, so
    `tralo` vs `hounie` is invalid there while `tralo` vs `clip` is fine.
    """
    q = is_quarantined(root)
    return set((q or {}).get("dead_arms") or ())


def by_name(name):
    """The registry entry for a campaign NAME, or None.

    For tools whose input is a table rather than a directory -- `paper_rows`
    reads a `cell_table` CSV and never touches the campaign tree, so it has no
    path to walk and would otherwise be ungated. It is the tool that decides
    what may be WRITTEN, which makes it the worst one to leave ungated.
    """
    return REGISTRY.get(name)


# The four statuses that mean "this cell did not pose the question", and they
# are NOT interchangeable -- `configs.task_cells.classify` is emphatic about
# this and collapsing them turns an absence of measurement into a null.
NOT_A_TASK = ("non_task", "no_strict_band", "unmeasured", "no_window",
              "no_data")


def cell_status(root):
    """Task status of every (dataset, model, cap) cell under `root`.

    Returns `{(dataset, model, cap): status}`, or **None** when the question
    cannot be asked here at all.

    None is not the empty dict, and the difference is the same fail-closed
    distinction `live_config_paths` makes. A campaign worktree is PINNED at the
    commit its configs were generated from, so `configs/task_cells.py` and
    `configs/task_windows.yml` may postdate it and simply not exist there.
    Reporting a campaign as non-task because the INSTRUMENT is missing would
    blame a healthy campaign for version skew -- the third outcome
    `run_campaign` already names UNRUNNABLE.

    WHY THIS LIVES IN THE GATE. `uniform1` ran 252 runs and `vittask1` 13, both
    with parity, health and dose entirely CLEAN, and both measured the absence
    of a question: every cell sits outside the measured task window. Nothing
    mechanical went wrong, so no health check could have caught it, and the
    scorers printed full plausible panels. Hand-listing such campaigns in
    REGISTRY catches the two we know about and goes stale on the next one;
    classifying the cells the scorer is actually about to read cannot.
    """
    try:
        import yaml
        from configs.task_cells import classify, load_windows
        P = yaml.safe_load(io.open(os.path.join(_REPO, "configs",
                                                "protocol.yml"),
                                   encoding="utf-8"))
        TW = load_windows()
    except Exception:
        return None

    cells = {}
    for cfg in glob.glob(os.path.join(root, "**", "config.json"),
                         recursive=True):
        parts = os.path.normpath(cfg).split(os.sep)
        if len(parts) < 6:
            continue
        model, dataset, cap = parts[-6], parts[-5], parts[-4]
        key = (dataset, model, cap)
        if key in cells:
            continue
        # classify() narrates every K=0 local budget it meets and iwildcam has
        # 7 of 14. Swallow the narration, never the exception.
        keep, sys.stdout = sys.stdout, io.StringIO()
        try:
            cells[key] = classify(P, TW, dataset, model, cap)["status"]
        except Exception:
            cells[key] = "no_data"
        finally:
            sys.stdout = keep
    return cells


def _announce_cells(campaigns, out):
    """Print the task-window standing of what is about to be scored."""
    for c in campaigns:
        cells = cell_status(c)
        if cells is None:
            print("?? TASK STATUS UNVERIFIABLE in this checkout: %s" % c,
                  file=out)
            print("   `configs.task_cells` or `task_windows.yml` is absent "
                  "here, which is normal in a PINNED campaign worktree. This "
                  "has verified NOTHING -- it is not a pass.", file=out)
            print("", file=out)
            continue
        if not cells:
            continue
        bad = {k: v for k, v in cells.items() if v in NOT_A_TASK}
        if not bad:
            continue
        every = len(bad) == len(cells)
        print("!! %s OF %d CELLS DO NOT POSE THE CAP QUESTION: %s"
              % ("ALL %d" % len(bad) if every else "%d" % len(bad),
                 len(cells), c), file=out)
        for (ds, model, cap), st in sorted(bad.items()):
            print("     %-14s %-14s %-14s %s" % (model, ds, cap, st),
                  file=out)
        print("   `non_task` is a measured statement about the experiment; "
              "`unmeasured`,", file=out)
        print("   `no_strict_band`, `no_window` and `no_data` are absences of "
              "measurement.", file=out)
        print("   They are NOT the same and neither is a null. A contrast on "
              "these cells", file=out)
        print("   is not evidence about the constraint.", file=out)
        print("", file=out)


def gate(campaigns, allow=False, verb="score", out=sys.stdout):
    """THE refusal every scorer calls. Returns (blocked, dead_arm_set).

    ONE implementation, because there were two and five tools had neither.
    Audited 2026-09-04: `full_panel` and `cell_table` carried a copy each,
    while `deployed_h2h`, `paper_rows`, `score_scan`, `paired_noise` and
    `sensitivity_screen` checked nothing at all -- and `paper_rows` is the tool
    whose whole job is saying what may be WRITTEN. A marker only prevents a
    mistake in the tools that read it.

    Two outcomes, and they are not the same:

      * `scorable is False` -> BLOCKED. Nothing here may be scored.
      * a PARTIAL marker (`scorable=True` with `dead_arms`) -> not blocked,
        but the named arms are announced and returned, so the caller can drop
        contrasts that touch them. Announcing is not optional: an unannounced
        partial marker is worse than none, because the table looks complete.

    🛑 IMPORT THIS WITHOUT A FALLBACK. Wrapping it in a bare handler
    that degrades to `lambda: None` turns the refusal off with no message when
    the file is hand-copied into a worktree whose `scripts/` predates it --
    which CLAUDE.md explicitly sanctions mid-flight. A gate that cannot fail is
    decoration. If this import breaks, the scorer must break.
    """
    hard = [(c, q) for c in campaigns for q in [refuses_scoring(c)] if q]
    if hard and not allow:
        for c, q in hard:
            print("REFUSING to %s %s" % (verb, c), file=out)
            print("  reason   : %s" % q.get("reason"), file=out)
            print("  keep for : %s" % q.get("keep_for"), file=out)
        print("", file=out)
        print("Pass --allow-quarantined only if you know why the marker is "
              "there and are", file=out)
        print("reporting the campaign as quarantined anyway.", file=out)
        return True, DeadArms()
    for c, q in hard:
        print("!! %sING A QUARANTINED CAMPAIGN: %s -- %s"
              % (verb.upper(), c, q.get("reason")), file=out)
        print("", file=out)

    _announce_cells(campaigns, out)

    dead = DeadArms()
    for c in campaigns:
        d = dead_arms(c)
        if not d:
            continue
        dead[campaign_name(c) or c] = frozenset(d)
        q = is_quarantined(c) or {}
        print("!! PARTIAL QUARANTINE: %s" % c, file=out)
        print("   %s" % q.get("reason"), file=out)
        print("   DEAD ARMS (any contrast touching one is NOT comparable): %s"
              % ", ".join(sorted(d)), file=out)
        print("   everything else here is untouched: %s"
              % q.get("keep_for"), file=out)
        print("", file=out)
    return False, dead


def arm_of_run(path):
    """The arm a run path belongs to. `<root>/<model>/<ds>/<cap>/<arm>/seed_N`.

    Accepts the seed directory or any file inside it, so a caller holding
    `.../seed_1/config.json` and one holding `.../seed_1` get the same answer.
    """
    q = os.path.normpath(path)
    if os.path.basename(q).startswith("seed_"):
        return os.path.basename(os.path.dirname(q))
    parts = q.split(os.sep)
    for i, seg in enumerate(parts):
        if seg.startswith("seed_") and i:
            return parts[i - 1]
    return None


class DeadArms(dict):
    """{campaign name -> frozenset(dead arms)}. PER CAMPAIGN, never a union.

    🛑 A UNION DELETES ARMS FROM A HEALTHY CAMPAIGN. `gate()` used to fold
    every campaign's dead arms into one set and hand that to the filter, so
    `full_panel --campaign results/dom1 results/taskwin2` dropped `fioretto`
    and `hounie` from `taskwin2` -- which carries no marker at all -- while
    printing "everything else in this campaign is unaffected". A disqualified
    contrast and a missing one read identically in the output, and the second
    is absence of evidence.

    `paper_rows` already keyed by campaign; the six path-based scorers did
    not. This is that shape, made shared, so the two cannot disagree again.
    """

    def for_path(self, path):
        """The dead arms of the campaign `path` belongs to. Empty if none."""
        if not self:
            return frozenset()
        name = campaign_name(path)
        if name in self:
            return self[name]
        # A path under a campaign the caller named by a different string still
        # has to resolve, so fall back to a prefix match on the raw roots.
        q = os.path.normpath(path).replace(os.sep, "/")
        for camp, arms in self.items():
            if ("/%s/" % camp) in q + "/":
                return arms
        return frozenset()

    def union(self):
        out = set()
        for v in self.values():
            out |= set(v)
        return out


def drop_dead_runs(paths, dead, out=sys.stdout, label="run"):
    """Remove runs belonging to a PARTIALLY quarantined campaign's dead arms.

    🛑 ANNOUNCING A PARTIAL MARKER IS NOT ENFORCING IT. `gate()` returns the
    dead arms so callers can drop the contrasts that touch them, and until
    2026-09-04 SIX of the seven scorers bound that return value and never
    looked at it again: `deployed_h2h` would print the PARTIAL banner and then
    rank `fioretto` or `hounie` #1 in a `dom1` cell, which is precisely the
    comparison the marker says is not comparable. Only `paper_rows` filtered.

    So the filtering lives here, next to the marker, and every scorer calls it
    at the point it enumerates runs. Returns the survivors, and PRINTS what it
    removed -- a silent drop turns a disqualified contrast into a missing one,
    which reads as absence of evidence rather than exclusion.
    """
    if not dead:
        return list(paths)
    per_campaign = isinstance(dead, DeadArms)
    keep, dropped, unreadable = [], {}, []
    for p in paths:
        arms = dead.for_path(p) if per_campaign else dead
        arm = arm_of_run(p)
        if arm is None:
            # 🛑 FAIL CLOSED. `arm_of_run` returns None for any layout it does
            # not recognise, and KEEPING those silently made the filter a
            # no-op on `score_scan`'s flat `seed1_<arm>` roots: the PARTIAL
            # banner printed, 0 of N were dropped, and the dead arms scored.
            # Every other fail-direction in this module is fail-closed.
            unreadable.append(p)
            continue
        if arm in arms:
            dropped[arm] = dropped.get(arm, 0) + 1
        else:
            keep.append(p)
    if unreadable:
        print("!! %d %s(s) are in a layout this filter cannot read, so their "
              "ARM is unknown:" % (len(unreadable), label), file=out)
        for u in unreadable[:5]:
            print("     %s" % u, file=out)
        if len(unreadable) > 5:
            print("     ... and %d more" % (len(unreadable) - 5), file=out)
        print("   A dead arm could be among them. They are EXCLUDED rather "
              "than kept: this campaign", file=out)
        print("   carries a partial quarantine, and keeping an unclassifiable "
              "run fails the marker open.", file=out)
        print("", file=out)
    if dropped:
        print("!! DROPPED %d %s(s) belonging to quarantined arms: %s"
              % (sum(dropped.values()), label,
                 ", ".join("%s x%d" % (a, n) for a, n in sorted(dropped.items()))),
              file=out)
        print("   Those arms ran at a different constraint dose, so any "
              "contrast touching them", file=out)
        print("   is not comparable. Everything else in this campaign is "
              "unaffected.", file=out)
        print("", file=out)
    return keep


def live_config_paths():
    """Config paths a LIVE runner currently holds, or None if that is unknown.

    None is NOT the empty set, and the difference is the whole point: an empty
    set licenses removal, None must forbid it. `ps` failing must never read as
    "no runner is using this file". Same fail-closed direction as the
    unreadable-marker gate.
    """
    try:
        txt = subprocess.check_output(["ps", "-eo", "args="],
                                      stderr=subprocess.DEVNULL)
    except Exception:
        return None
    out = set()
    for line in txt.decode("utf-8", "replace").splitlines():
        for tok in line.split():
            if tok.endswith("config.json"):
                out.add(os.path.realpath(tok))
    return out


def scan(root, live=_UNSET):
    """Status counts, dataset set, and the runs that can never execute."""
    counts, datasets, unrunnable, stale, held = {}, set(), [], [], []
    quarantined = is_quarantined(root) is not None
    if live is _UNSET:
        live = live_config_paths()
    for f in glob.glob(os.path.join(root, "*", "*", "*", "*", "seed_*",
                                    "config.json")):
        try:
            d = json.load(open(f))
        except Exception:
            counts["unreadable"] = counts.get("unreadable", 0) + 1
            continue
        st = d.get("status", "?")
        counts[st] = counts.get(st, 0) + 1
        ds = d.get("dataset_mode") or ""
        datasets.add(ds)
        # A pending run is removable when it can never execute (its dataset is
        # gone) OR when it sits inside a quarantined campaign. The second case
        # is the live hazard: `uniform1_VOID` holds 240 pending runs on
        # iwildcam, which EXISTS, so `main.py` would dispatch all 240 of a
        # campaign that is void by name. A marker file does not stop the
        # dispatcher; an absent config does.
        if st == "pending" and (ds in DEAD_DATASETS or quarantined):
            unrunnable.append(f)
        elif st == "running" and quarantined:
            # `main.py` RESETS `running` to `pending` when a dispatcher starts
            # on the root, so a `running` config in a DEAD campaign is exactly
            # as dispatchable as a pending one -- the same hazard the block
            # above exists to close, and this branch was missing until
            # 2026-09-01. `taskwin1` was left holding two of them.
            #
            # The 2-day mtime guard below is for LIVE campaigns, where age is
            # the only evidence a run died. Here the campaign must never
            # execute at ALL, so age is irrelevant and the only question is
            # whether a process holds the file right now. `live is None` means
            # we could not find out, and then we do not touch it.
            if live is not None and os.path.realpath(f) not in live:
                unrunnable.append(f)
            else:
                held.append(f)
        elif st == "running" and (time.time() - os.path.getmtime(f)) > 2 * 86400:
            stale.append(f)
    return counts, datasets, unrunnable, stale, held


def cmd_list(home=None, out=sys.stdout):
    print("%-36s %-9s %-28s %s"
          % ("campaign", "state", "status counts", "why"), file=out)
    print("-" * 118, file=out)
    for root in campaign_roots(home):
        name = os.path.basename(root)
        counts, _, unrunnable, stale, held = scan(root)
        if not counts:
            continue
        q = is_quarantined(root)
        state = "QUARANTINE" if q else "live"
        note = (q or {}).get("reason", "")
        if len(note) > 60:
            note = note[:57] + "..."
        print("%-36s %-9s %-28s %s" % (name[:36], state, str(counts)[:28], note),
              file=out)
        if unrunnable:
            print("%-36s %-9s   %d pending run(s) that must never execute "
                  "(dead dataset, or a quarantined campaign the dispatcher "
                  "would still pick up)" % ("", "", len(unrunnable)), file=out)
        if stale:
            print("%-36s %-9s   %d run(s) claim `running` with no process behind "
                  "them" % ("", "", len(stale)), file=out)
        if held:
            print("%-36s %-9s   %d `running` config(s) NOT removed: a live "
                  "process holds them, or `ps` was unreadable"
                  % ("", "", len(held)), file=out)
    return 0


def cmd_apply(execute=False, home=None, out=sys.stdout):
    """Write markers, drop unrunnable configs, correct stale statuses."""
    wrote = removed = fixed = 0
    for root in campaign_roots(home):
        name = os.path.basename(root)
        counts, _, unrunnable, stale, held = scan(root)
        if not counts:
            continue
        entry = REGISTRY.get(name)

        if entry and not os.path.exists(os.path.join(root, MARKER)):
            payload = dict(entry, campaign=name, quarantined_by="scripts.quarantine")
            print("  marker  %s" % name, file=out)
            if execute:
                with open(os.path.join(root, MARKER), "w") as fh:
                    json.dump(payload, fh, indent=2)
            wrote += 1

        for f in unrunnable:
            if removed < 3:
                print("  remove  %s" % f.replace(os.path.expanduser("~/"), ""),
                      file=out)
            if execute:
                os.remove(f)
            removed += 1

        for f in stale:
            if fixed < 6:
                print("  correct %s  running -> crashed"
                      % f.replace(os.path.expanduser("~/"), ""), file=out)
            if execute:
                d = json.load(open(f))
                d["status"] = "crashed"
                d["status_corrected_by"] = "scripts.quarantine: claimed `running` " \
                                           "with no process behind it"
                json.dump(d, open(f, "w"), indent=2)
            fixed += 1

    print("\n%s: %d marker(s), %d unrunnable config(s) removed, %d stale status(es) "
          "corrected" % ("APPLIED" if execute else "DRY RUN (pass --execute)",
                         wrote, removed, fixed), file=out)
    if not execute:
        print("Nothing was changed.", file=out)
    return 0


def cmd_check(root, out=sys.stdout):
    q = is_quarantined(root)
    if not q:
        print("OK -- %s is not quarantined" % root, file=out)
        return 0
    print("QUARANTINED: %s" % root, file=out)
    print("  reason   : %s" % q.get("reason"), file=out)
    print("  keep for : %s" % q.get("keep_for"), file=out)
    print("  Do not score it. Do not pool it. It is kept as a receipt.", file=out)
    return 1


def self_test(out=sys.stdout):
    """Can the gate say NO, and does it say YES for anything else?

    A quarantine that never fires is decoration. A quarantine that fires on
    everything stops the project. Both directions are checked.
    """
    import shutil
    import tempfile

    ok = True
    tmp = tempfile.mkdtemp()
    try:
        live = os.path.join(tmp, "results", "a_live_campaign")
        dead = os.path.join(tmp, "results", "iwc2")
        for d in (live, dead):
            os.makedirs(d)

        checks = [
            ("a campaign in the registry is refused BY NAME, with no marker file",
             is_quarantined(dead) is not None),
            ("a campaign not in the registry is allowed",
             is_quarantined(live) is None),
            ("--check exits 1 on the dead one", cmd_check(dead, open(os.devnull, "w")) == 1),
            ("--check exits 0 on the live one", cmd_check(live, open(os.devnull, "w")) == 0),
        ]

        # a written marker must override the name, in BOTH directions
        with open(os.path.join(live, MARKER), "w") as fh:
            json.dump(dict(reason="marked by hand", scorable=False), fh)
        checks.append(("a hand-written marker quarantines a campaign not in the "
                       "registry", is_quarantined(live) is not None))

        # an unreadable marker must fail CLOSED, never open
        with open(os.path.join(live, MARKER), "w") as fh:
            fh.write("{ this is not json")
        q = is_quarantined(live)
        checks.append(("an unreadable marker fails CLOSED",
                       q is not None and q.get("scorable") is False))

        # every registry entry must carry a reason and a keep_for
        checks.append(("every registry entry states a reason and what it is kept for",
                       all(e.get("reason") and e.get("keep_for")
                           for e in REGISTRY.values())))
        # THREE STATES, and every entry must be exactly one of them.
        #
        # This check used to read "no registry entry claims to be scorable",
        # because until 2026-09-04 `scorable=True` really was a contradiction:
        # an entry that blocks nothing. The PARTIAL state gave it a meaning --
        # `scorable=True` WITH `dead_arms` says the campaign may be scored for
        # every contrast that does not touch the named arms. What is still a
        # contradiction, and what this now catches, is `scorable=True` with NO
        # dead arms: a registry row that does nothing at all.
        bad_state = [k for k, e in REGISTRY.items()
                     if e.get("scorable") is not False and not e.get("dead_arms")]
        checks.append(("every entry either blocks the campaign or names the "
                       "arms it blocks (offenders: %s)" % (bad_state or "none"),
                       not bad_state))
        # A partial entry must not ALSO claim to be unscorable, or the two
        # states are being read off different fields and one of them loses.
        contradictory = [k for k, e in REGISTRY.items()
                         if e.get("dead_arms") and e.get("scorable") is False]
        checks.append(("no entry is both partial and wholly unscorable "
                       "(offenders: %s)" % (contradictory or "none"),
                       not contradictory))
        # LIVENESS on the partial state itself: at least one entry must use it,
        # or the branch below has never executed against a real row.
        checks.append(("at least one PARTIAL entry exists, so the branch is "
                       "exercised",
                       any(e.get("dead_arms") for e in REGISTRY.values())))

        # A `running` config inside a QUARANTINED root is exactly as
        # dispatchable as a `pending` one, because `main.py` resets it on
        # start. Gate all three directions: it goes; it is SPARED when a live
        # process holds it or when `ps` is unreadable; and a live campaign
        # loses nothing to the branch. `taskwin1` was left holding two of them
        # until 2026-09-01.
        qroot = os.path.join(tmp, "results", "a_dead_campaign")
        for arm, st in (("a", "pending"), ("b", "running"),
                        ("c", "running"), ("d", "completed")):
            d = os.path.join(qroot, "M", "iwildcam", "L80_G95", arm, "seed_1")
            os.makedirs(d)
            json.dump({"status": st, "dataset_mode": "iwildcam"},
                      open(os.path.join(d, "config.json"), "w"))
        json.dump(dict(reason="self-test fixture", scorable=False),
                  open(os.path.join(qroot, MARKER), "w"))
        heldp = os.path.realpath(os.path.join(
            qroot, "M", "iwildcam", "L80_G95", "b", "seed_1", "config.json"))
        _, _, un1, _, hd1 = scan(qroot, live={heldp})
        checks.append(("a `running` run in a DEAD campaign is dropped too -- "
                       "main.py would reset it to pending and run it",
                       len(un1) == 2 and len(hd1) == 1 and heldp not in un1))
        _, _, un2, _, hd2 = scan(qroot, live=None)
        checks.append(("...but NOT when `ps` is unreadable: unknown is not the "
                       "empty set, so it fails CLOSED",
                       len(un2) == 1 and len(hd2) == 2))
        os.remove(os.path.join(qroot, MARKER))
        _, _, un3, _, hd3 = scan(qroot, live=set())
        checks.append(("...and a LIVE campaign loses nothing to that branch",
                       not un3 and not hd3))

        # ---- verified extensions: pooling must be EARNED (2026-09-05) -----
        # Under a `results` dir on purpose: `_campaign_root` stops at the
        # child of `results` by convention, which is how a campaign scored
        # one backbone at a time still resolves to the campaign.
        ext_root = os.path.join(tmp, "results")

        # MULTI-ARM on purpose. The first version of this fixture had ONE arm,
        # and the verifier sampled one config per ROOT rather than per (cap,
        # arm) -- so on the live tree it compared `vitdual2/alm` against
        # `vitseed1/clip` and refused two campaigns that are identical where
        # they overlap. A single-arm fixture cannot see that; the arms here
        # carry DIFFERENT knobs from each other so that a cross-arm comparison
        # is guaranteed to look wrong.
        ARM_KNOB = {"tralo": {"lambda_step": 0.5},
                    "alm": {"alm_eta": 0.1, "alm_mu0": 1.0},
                    "clip": {"constraint_epochs": 0}}

        def _mk(camp, seeds, code="abc123", extra=None, arms=("tralo",)):
            """A campaign tree: the given arms, each with its OWN knobs."""
            for arm in arms:
                for s in seeds:
                    d = os.path.join(ext_root, camp, "ViTB16", "iwildcam",
                                     "L80_G95", arm, "seed_%d" % s)
                    os.makedirs(d, exist_ok=True)
                    hp = {"lr": 0.0001, "warmup_epochs": 1, "seed": s}
                    hp.update(ARM_KNOB.get(arm, {}))
                    hp.update(extra or {})
                    with io.open(os.path.join(d, "config.json"), "w",
                                 encoding="utf-8") as fh:
                        json.dump({"status": "completed",
                                   "code_version": code,
                                   "hyperparams": hp}, fh)
            return os.path.join(ext_root, camp)

        def _mark(root, parent):
            with io.open(os.path.join(root, EXTENDS), "w",
                         encoding="utf-8") as fh:
                json.dump({"parent": parent}, fh)

        def _raises(root):
            _EXT_CACHE.clear()
            try:
                extension_parent(root)
                return False
            except ValueError:
                return True

        par = _mk("parent1", [1, 2, 3, 4], arms=("tralo", "alm", "clip"))
        good = _mk("parent1seed", [5, 6, 7, 8], arms=("tralo", "clip"))
        _EXT_CACHE.clear()
        checks.append(("an ordinary campaign reports its OWN name",
                       campaign_name(par) == "parent1"))
        checks.append(("...and an UNMARKED sibling does too, so nothing pools "
                       "by accident", campaign_name(good) == "parent1seed"))

        _mark(good, "parent1")
        _EXT_CACHE.clear()
        checks.append(("a VERIFIED extension reports its PARENT, so its seeds "
                       "land in the parent cell",
                       campaign_name(good) == "parent1"))
        checks.append(("  and it resolves from a path INSIDE the extension too",
                       campaign_name(os.path.join(good, "ViTB16", "iwildcam"))
                       == "parent1"))

        # NEGATIVE CONTROLS. Each is a marker that CLAIMS an extension it is
        # not; every one must raise rather than pool.
        bad_hp = _mk("badhp", [5, 6], extra={"lr": 0.5})
        _mark(bad_hp, "parent1")
        checks.append(("a marker whose hyperparams differ in more than `seed` "
                       "RAISES", _raises(bad_hp)))

        bad_code = _mk("badcode", [5, 6], code="deadbeef")
        _mark(bad_code, "parent1")
        checks.append(("a marker across a different code_version RAISES",
                       _raises(bad_code)))

        overlap = _mk("overlap", [3, 4, 5])
        _mark(overlap, "parent1")
        checks.append(("OVERLAPPING seeds RAISE -- pooling them would put two "
                       "runs under one (cell, seed, arm) key",
                       _raises(overlap)))

        missing = _mk("missing", [5, 6])
        _mark(missing, "no_such_campaign")
        checks.append(("a marker naming a parent that does not exist RAISES",
                       _raises(missing)))

        # THE LIVE BUG, as a control: an extension holding a SUBSET of the
        # parent arms must VERIFY. Comparing one config per root instead of per
        # (cap, arm) refuses this, which is what happened on `vitseed1`.
        checks.append(("an extension holding a SUBSET of the parent arms is "
                       "verified, comparing per (cap, arm) not per root",
                       campaign_name(good) == "parent1"))

        # Seeds 11-12, DISJOINT from parent1seed's 5-8 on purpose: with
        # overlapping seeds this would refuse for the overlap instead and the
        # no-shared-arm branch would never run. A mutation test caught exactly
        # that -- the control passed while checking nothing.
        disjoint = _mk("disjointarms", [11, 12], arms=("alm",))
        _mark(disjoint, "parent1seed")
        checks.append(("two roots sharing NO arm RAISE -- there is nothing to "
                       "pool", _raises(disjoint)))

        skew = _mk("skewarm", [5, 6], arms=("tralo",),
                   extra={"lambda_step": 0.9})
        _mark(skew, "parent1")
        checks.append(("a SHARED arm whose knobs differ RAISES, even though "
                       "the other arms match", _raises(skew)))

        # `write_extends_marker` must refuse to CREATE a claim it cannot verify.
        refused = _mk("refused", [3, 4])
        buf = io.StringIO()
        rc = write_extends_marker(refused, "parent1", out=buf)
        checks.append(("write_extends_marker REFUSES an unverifiable claim and "
                       "writes no marker",
                       rc == 1
                       and not os.path.exists(os.path.join(refused, EXTENDS))))

        accepted = _mk("accepted", [9, 10])
        rc = write_extends_marker(accepted, "parent1", out=io.StringIO())
        _EXT_CACHE.clear()
        checks.append(("  and ACCEPTS a real one, which then pools",
                       rc == 0 and campaign_name(accepted) == "parent1"))

        print("SELF-TEST\n", file=out)
        for label, good in checks:
            print("  %-4s %s" % ("OK" if good else "FAIL", label), file=out)
            ok = ok and good
        print("\nSELF-TEST %s" % ("PASSED" if ok else "FAILED"), file=out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--execute", action="store_true",
                    help="with --apply, actually write and remove")
    ap.add_argument("--check", metavar="ROOT")
    ap.add_argument("--home", help="scan under this home instead of ~")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--extends", nargs=2, metavar=("ROOT", "PARENT"),
                    help="declare ROOT an add_seeds extension of the "
                         "sibling campaign PARENT, so its seeds pool "
                         "into PARENT cells. Verified before writing.")
    a = ap.parse_args(argv)

    if a.extends:
        return write_extends_marker(a.extends[0], a.extends[1])

    if a.self_test:
        return self_test()
    if a.check:
        return cmd_check(a.check)
    if a.apply:
        return cmd_apply(execute=a.execute, home=a.home)
    if a.list:
        return cmd_list(home=a.home)
    ap.error("give --list, --apply, --check <root> or --self-test")


if __name__ == "__main__":
    sys.exit(main())
