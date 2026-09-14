> ARCHIVED � retired historical-prose checks, not active tests.

# Prose-check retirement, 2026-09-14

These checks required historical tables, headings, numbers or the retired
campaign-state prose parser. The user requested a fresh evidence boundary.
Loss, allocator, metric, data-loading and process-safety tests remain active.
The source snippets below preserve exactly what was retired; they are not
standalone executable modules. Mixed tests retain their behavioral parts.

## tests/test_pipeline.py :: test_the_documented_test_count_is_the_real_one

```python
def test_the_documented_test_count_is_the_real_one(request):
    """CLAUDE.md and FRAMEWORK.md both quote this number. Both were wrong.

    CLAUDE.md said 75, FRAMEWORK.md said 96 in three places, and pytest
    collected 107. A reader uses the number to decide whether their checkout is
    complete, so a stale one says "you are missing tests" to someone who is not.
    """


    # Only meaningful when the whole suite was collected. Running a single node
    # id collects 1, which would fail the guard on every targeted run.
    # `-k` is an OPTION, not a positional arg, so scanning config.args never
    # saw it: a targeted `-k` run collected 1 test and failed this guard on
    # `n > 1` instead of skipping. Read the option itself.
    # A FILE PATH IS ALSO A SUBSET, and it was not detected. `-k` was the
    # first miss (an option, not a positional); `::` was the second; a bare
    # `pytest tests/test_pipeline.py` is the third -- it collects 329 of 606
    # and failed this guard on `n != documented` while testing nothing about
    # the docs. Same defect, one step further out. A whole-suite run names
    # only DIRECTORIES, so that is the discriminator.
    if (request.config.option.keyword
            or any("::" in a for a in request.config.args)
            or any(not os.path.isdir(a.split("::")[0])
                   for a in request.config.args)):
        pytest.skip("subset run: the collected count is not the suite count")
    n = request.session.testscollected or len(request.session.items)
    assert n > 1

    claimed = {}
    for path in ("CLAUDE.md", "docs/FRAMEWORK.md"):
        txt = io.open(path, encoding="utf-8").read()
        for m in re.finditer(r"(\d+)\s+(?:regression\s+)?tests", txt):
            claimed.setdefault(path, set()).add(int(m.group(1)))

    wrong = {path: sorted(v - {n}) for path, v in claimed.items() if v - {n}}
    assert not wrong, (
        "pytest collects %d, but the docs claim %s. Update them, or the count "
        "tells a reader their checkout is incomplete." % (n, wrong))
```

## tests/test_pipeline.py :: test_the_deletion_table_does_not_claim_live_code_was_deleted

```python
def test_the_deletion_table_does_not_claim_live_code_was_deleted():
    """FRAMEWORK is the law, so a false claim in it is a defect, not a typo.

    Section (f) listed `danits_lp`, `focal`, `class_balanced` and
    `logit_adjust` as deleted methodology packages, and `cb_beta` /
    `logit_adjust_tau` as removed keys. All four packages exist, are registered
    in TRAIN_FNS, and are among the nine methodologies the PAPER claims; both
    keys are live in protocol.yml. Anyone trusting the table would conclude
    those arms are gone.

    Rather than fix the prose and hope, the table is checked: anything it says
    was removed must actually be absent.
    """



    text = io.open("docs/FRAMEWORK.md", encoding="utf-8").read()
    start = text.index("### (f) What was DELETED FROM THE CODE")
    section = text[start:text.index(chr(10) + "### ", start + 10)]

    proto = yaml.safe_load(io.open("configs/protocol.yml", encoding="utf-8"))
    live_keys = set(proto.get("core", {})) | set(proto.get("constraint_phase", {}))
    for blk in proto.get("blocks", {}).values():
        live_keys |= set(blk)

    claimed = set()
    for row in section.splitlines():
        if not row.startswith("|") or row.startswith("| removed") or "---" in row:
            continue
        cells = row.split("|")
        # Columns 1 AND 2 -- "removed" and "was". Reading only column 1 made the
        # methodology half of this check VACUOUS: that row says "5 methodology
        # packages" in column 1 and puts the actual names in column 2, so a
        # false claim about a live package sailed through. Caught by running the
        # negative control instead of trusting a green test.
        text2 = " ".join(cells[1:3])
        claimed |= set(re.findall(r"[a-z_][a-z0-9_]{2,}", text2))

    assert claimed, "the deletion table parsed to nothing -- the check is vacuous"

    still_live = sorted(k for k in claimed if k in live_keys)
    assert not still_live, (
        "FRAMEWORK section (f) says these were removed, but they are live keys "
        "in protocol.yml: %s" % still_live)

    registered = sorted(m for m in claimed if m in TRAIN_FNS)
    assert not registered, (
        "FRAMEWORK section (f) says these methodologies were deleted, but they "
        "are registered in TRAIN_FNS: %s" % registered)
```

## tests/test_pipeline.py :: test_the_warmup_1_row_is_flagged_as_the_LR_trap_not_a_result

```python
def test_the_warmup_1_row_is_flagged_as_the_LR_trap_not_a_result():
    """1b records +15.20 pp at warm-up 1 specifically so nobody rediscovers it
    and reads it as section 3's regime effect. It is the shape the LR trap
    makes -- 1b documents an unequal `lr_constraint` fabricating 16.7 pp that
    became 1.7 pp once equalized -- and the corpus cannot separate the two.
    The gate keeps the number honest and keeps the warning attached to it.
    """
    res, _ = _headline_power_table()
    w1 = res[res["warmup_epochs"] == 1]
    assert len(w1) == 10, len(w1)
    assert 14.0 < 100 * w1["mean"].mean() < 16.5, w1["mean"].mean()
    assert (w1["mean"] > 0).all()

    fw = io.open(os.path.join(REPO, "docs", "FRAMEWORK.md"),
                 encoding="utf-8").read()
    i = fw.find("+15.20 pp")
    assert i > 0, "1b no longer quotes the warm-up-1 figure"
    near = fw[i - 400:i + 700]
    assert "LR TRAP" in near.upper(), (
        "the +15.20 pp figure is quoted without the LR-trap warning attached")
    assert "Do not quote it" in near, near[:300]
```

## tests/test_lessons_learned.py :: test_no_campaign_is_discussed_without_a_recorded_state

```python
def test_no_campaign_is_discussed_without_a_recorded_state():
    """2026-09-10: `price1` was launched and no doc recorded that it existed."""
    from scripts import campaign_state

    _, unrecorded = campaign_state.scan()
    orphans = [(n, len(ex)) for n, _cnt, ex, _hits in unrecorded if ex]
    assert not orphans, (
        "these campaigns carry a PAST-EXECUTION verb and no recorded state: "
        "%s\nGive each a row in MISSION 0-RUNNING's campaign-state ledger, or "
        "a marker in quarantine.REGISTRY. Run "
        "`python -m scripts.campaign_state --all` for the lines." % orphans)
```

## tests/test_lessons_learned.py :: test_the_campaign_state_audit_actually_detects_an_orphan

```python
def test_the_campaign_state_audit_actually_detects_an_orphan():
    """2026-09-10: the negative control -- the gate above must be able to fail.

    A gate that has never failed has never been shown to work, and this one
    reads five separate authorities, so "it passes" is weak evidence on its
    own.
    """
    import shutil

    from scripts import campaign_state

    root = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(root, "docs"))
        doc = os.path.join(root, "docs", "FRAMEWORK.md")
        io.open(doc, "w", encoding="utf-8").write(
            "# f\n\n`zzz9` was launched on dsisco01 and landed 48 runs\n"
            "`qqq9` would be generated by `--root results/qqq9`\n")
        _, unrec = campaign_state.scan(("docs/FRAMEWORK.md",), root=root)
        flagged = [n for n, _c, ex, _h in unrec if ex]
        assert flagged == ["zzz9"], (
            "the detector must flag the campaign with an execution verb and "
            "ONLY that one, got %s" % flagged)
        # NEGATIVE CONTROL inside the control: a name with no execution verb
        # is a proposal, not an orphan, and must not be reported as a defect.
        assert "qqq9" in [n for n, _c, _e, _h in unrec]
        assert "qqq9" not in flagged
    finally:
        shutil.rmtree(root, ignore_errors=True)
```

## tests/test_lessons_learned.py :: test_a_documented_launcher_path_either_EXISTS_or_is_in_the_DELETED_registry

```python
def test_a_documented_launcher_path_either_EXISTS_or_is_in_the_DELETED_registry():
    """A doc line naming `docs/launch_*.sh` must resolve to something.

    LESSON, 2026-09-10. NOT ONE `docs/launch_*.sh` exists: the survivors were
    archived to `docs/archive/launchers/` and the rest were deleted outright.
    Seven lines across MISSION and FRAMEWORK still point a reader at the old
    paths -- a link followed once, not found, and silently distrusted. The
    2026-09-06 cleanup found them, could not decide whether annotating the law
    was a janitorial call, and DEFERRED them; four days later nothing had
    changed, because a DEFERRED list is a defect with a comment attached.

    The fix is a registry in MISSION 0-LAUNCH carrying the recovery command per
    file, and this gate, which requires every named launcher to either exist on
    disk or appear there. The registry earns its place by being a TABLE: the
    deferred note offered one blanket recovery command and it is wrong for
    `launch_margin1.sh`, deleted at `2c5f292a` rather than `e7d9e893`.

    Deliberately NOT a check that the file exists -- these are correctly gone.
    It checks that a reader who follows a dead path finds the correction.
    """
    import re

    docs = [os.path.join(REPO, "CLAUDE.md")]
    ddir = os.path.join(REPO, "docs")
    for name in sorted(os.listdir(ddir)):
        if name.endswith(".md"):
            docs.append(os.path.join(ddir, name))

    mission = io.open(os.path.join(REPO, "docs", "MISSION.md"),
                      encoding="utf-8").read()
    assert "DELETED LAUNCHERS -- the registry" in mission, (
        "MISSION 0-LAUNCH lost its deleted-launcher registry; the seven stale "
        "doc lines it corrects are still there")
    registry = mission.split("DELETED LAUNCHERS -- the registry", 1)[1]
    registry = registry.split("\n### ", 1)[0]

    named, unresolved = set(), []
    pat = re.compile(r"docs/launch_[A-Za-z0-9_]+\.sh")
    for path in docs:
        for hit in pat.findall(io.open(path, encoding="utf-8").read()):
            named.add(hit)
    for hit in sorted(named):
        if os.path.exists(os.path.join(REPO, hit)):
            continue
        if os.path.basename(hit) in registry:
            continue
        unresolved.append(hit)

    assert named, "no launcher paths found at all -- the pattern stopped matching"
    assert not unresolved, (
        "documented launcher path(s) that neither exist nor appear in "
        "MISSION's DELETED LAUNCHERS registry: %s" % unresolved)

    # NEGATIVE CONTROL: the gate must FIRE on a launcher that is absent from
    # both disk and registry. A gate that has never failed has never been shown
    # to work.
    fake = "docs/launch_this_never_existed.sh"
    assert not os.path.exists(os.path.join(REPO, fake))
    assert os.path.basename(fake) not in registry
    # ...and the same two conditions the loop applies would mark it unresolved.
    assert not (os.path.exists(os.path.join(REPO, fake))
                or os.path.basename(fake) in registry)

    # POSITIVE CONTROL: the archived launchers DO exist and must never be
    # flagged, so the gate cannot be satisfied by deleting every launcher.
    arch = os.path.join(REPO, "docs", "archive", "launchers")
    assert os.path.isdir(arch), arch
    assert [f for f in os.listdir(arch) if f.endswith(".sh")], (
        "docs/archive/launchers/ holds no .sh -- the survivors are gone too")
```

## tests/test_lessons_learned.py :: _unseen_groups_on_disk

```python
def _unseen_groups_on_disk(slice_dir):
    """Test groups ABSENT from train -- the table's own definition."""
    import csv

    def col(path, name):
        if not os.path.exists(path):
            return None
        with io.open(path, encoding="utf-8", newline="") as fh:
            r = csv.DictReader(fh)
            if name not in (r.fieldnames or []):
                return None
            return [row[name] for row in r]

    tr = col(os.path.join(slice_dir, "train_meta.csv"), "location")
    te = col(os.path.join(slice_dir, "test_meta.csv"), "location")
    if tr is None or te is None:
        return None
    return len(set(te) - set(tr))
```

## tests/test_lessons_learned.py :: _dataset_table_rows

```python
def _dataset_table_rows(txt):
    """(slice name -> declared unseen-group count) from a markdown table."""
    rows = {}
    for line in txt.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().strip("*").strip() for c in line.split("|")]
        if len(cells) <= _UNSEEN_COL:
            continue
        name = cells[_SLICE_NAME_COL]
        if "/" not in name or name.startswith("-"):
            continue
        want = cells[_UNSEEN_COL]
        if not re.match(r"^[-+]?\d+$", want):
            continue
        rows[name] = int(want)
    return rows
```

## tests/test_lessons_learned.py :: test_the_dataset_table_unseen_group_counts_match_the_TRACKED_meta

```python
def test_the_dataset_table_unseen_group_counts_match_the_TRACKED_meta():
    """LESSON, 2026-09-11 (FRAMEWORK 2(z104)). CLAUDE.md's dataset table decides
    which dataset gets run next, and two of its seven rows were not
    measurements: `bcn` was three dashes -- on the slice carrying a COMPLETE
    228-run campaign and licensing unit D1, whose NEGATIVE sign is load-bearing
    -- and `fmow` claimed 10 unseen groups where the slice on disk has 13.

    THE GROUP COUNT IS THE FIELD WORTH GATING. A NET or a z can drift for
    benign reasons (a tool fix, a re-screen); the number of test groups absent
    from train is a property of two CSVs, both TRACKED IN GIT.

    THE yml IS A CONSISTENCY CHECK, NOT A SECOND MEASUREMENT, and the first
    draft of this docstring said otherwise. `configs/task_windows.yml`'s group
    table reads the SAME test labels -- its own header says "Counted from the
    test labels alone, cap-invariant" -- so it cannot corroborate the meta, only
    agree with it. Its WINDOW rows do come from `fmow1`'s nulls; its group
    counts do not, and collapsing those two provenances is what produced the
    wrong claim. What the yml genuinely provides is a CONTEMPORANEOUS RECORD,
    written while `fmow1` was in flight at 114/304 and revised at 304/304, of
    the slice that campaign was using. Gating the two against each other stops
    them drifting apart, which is a real failure mode; it is not independent
    corroboration of either.

    Only slices whose meta is ON DISK are checked. `dermmnist`, `octmnist`,
    `tissuemnist` and `terra` have rows and no files; they are counted and
    named, never silently skipped, because a gate that quietly checks nothing
    is this suite's most expensive recurring defect.
    """
    txt = io.open("CLAUDE.md", encoding="utf-8").read()
    rows = _dataset_table_rows(txt)

    assert len(rows) >= 5, (
        "parsed only %d dataset rows from CLAUDE.md; the table moved or its "
        "column order changed -- fix the parse, do not relax the gate"
        % len(rows))

    checked, absent, bad = [], [], []
    for name, want in sorted(rows.items()):
        d = os.path.join("data", *name.split("/"))
        got = _unseen_groups_on_disk(d)
        if got is None:
            absent.append(name)
            continue
        checked.append(name)
        if got != want:
            bad.append("%s: the table says %d unseen groups, the tracked meta "
                       "has %d" % (name, want, got))

    assert checked, (
        "no dataset row could be checked against a slice on disk, so this gate "
        "verified NOTHING. Rows with no readable meta: %s" % absent)
    assert not bad, (
        "the dataset table disagrees with the tracked meta:\n  "
        + "\n  ".join(bad)
        + "\n(rows with no files on disk, not checked: %s)" % absent)

    # CONSISTENCY, NOT CORROBORATION. `configs/task_windows.yml` quotes the same
    # group counts and reads the same labels to get them, so agreement here
    # proves only that the two documents have not drifted -- which is worth
    # gating, and is not a second measurement. See the docstring.
    yml = io.open(os.path.join("configs", "task_windows.yml"),
                  encoding="utf-8").read()
    seen_in_yml = []
    for name in checked:
        short = name.split("/")[0]
        m = re.search(r"^\s*#\s*" + re.escape(short)
                      + r"\s+\d+\s+of\s+\d+\s+\d+%\s+\d+\s+of\s+(\d+)",
                      yml, re.M)
        if not m:
            continue
        seen_in_yml.append(short)
        assert int(m.group(1)) == rows[name], (
            "%s: task_windows.yml says %s groups, CLAUDE.md says %d. Both read "
            "the same test labels, so one of them was transcribed wrong -- "
            "recount from the meta and fix whichever disagrees with it"
            % (short, m.group(1), rows[name]))
    assert seen_in_yml, (
        "the task_windows.yml group table did not parse for ANY checked slice, "
        "so the consistency half of this gate verified nothing -- fix the parse")

    # NEGATIVE CONTROLS. Each must show the comparison can say NO.
    probe = os.path.join("data", *checked[0].split("/"))
    real = _unseen_groups_on_disk(probe)
    assert real, ("CONTROL: %s reports %r unseen groups, so no mismatch could "
                  "be detected there" % (checked[0], real))

    fake = dict(rows)
    fake[checked[0]] = real + 1
    assert fake[checked[0]] != real, "CONTROL: a wrong count must differ"

    bogus = "| **nope/slice** | g | +1 | 2.0 | 4 | x |"
    assert _dataset_table_rows(bogus) == {"nope/slice": 4}, (
        "CONTROL: the row parser does not read the unseen column at index %d"
        % _UNSEEN_COL)
    assert _dataset_table_rows("| not a table row") == {}, (
        "CONTROL: a short line must parse to nothing, not raise")
    assert _unseen_groups_on_disk(os.path.join("data", "no_such_slice")) is None, (
        "CONTROL: a missing slice must read None so it lands in `absent`, "
        "never 0, which would silently match a `0` row")
```
