"""The archive must stay quarantined, and the live record must stay small.

Two years of documentation accumulated in this repository, and the cost was
concrete: of 45 tracked markdown files only six were reachable from the entry
points, and a proved theorem package sat in an orphaned file long enough that a
later campaign re-derived it on GPUs. Worse, the archived paper sources carry
78-115 references each to datasets that are no longer viable -- dermMNIST leaked
(38.7% of test lesions appear in train), BCN is blocked on cross-split duplicates
with conflicting labels, iwildcam passes 2 of 8 candidate conditions.

So these are not style checks. Reading an archived file as current produces a
wrong answer about which datasets are usable, and that is the specific failure
the user asked to be made impossible.
"""
import os
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(ROOT, "docs", "archive")

BANNER = "ARCHIVED -- NOT CURRENT EVIDENCE"

# The entire live record. A document outside this set is either archived or new,
# and a new one is a decision that should be made deliberately rather than by
# accretion -- which is how the previous sprawl happened.
LIVE_DOCS = {
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "RULESET.md",
    "docs/FRAMEWORK.md",
    "docs/GIT_TRACKING.md",
    "docs/LEDGER.md",
    "docs/MISSION.md",
}

# Datasets that are retired, removed or blocked. A live document may discuss
# them -- MISSION records WHY each was dropped, and that record is load-bearing
# -- but it must say so in the same breath.
RETIRED = ("dermmnist", "octmnist", "pathmnist", "medmnist", "iwildcam", "isic")
RETIREMENT_WORDS = ("retired", "removed", "withdrawn", "blocked", "leak",
                    "archived", "not viable", "quarantin")


def _markdown_under(path):
    for base, _dirs, files in os.walk(path):
        for fn in files:
            if fn.endswith(".md"):
                yield os.path.join(base, fn)


def _rel(p):
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


def test_every_archived_markdown_carries_the_banner():
    """An agent that opens an archived file must see it is archived immediately.

    A directory name is easy to miss in a tool result; a banner in the first
    lines is not. `docs/archive/README.md` is exempt because it IS the
    quarantine notice and states the rule at greater length.
    """
    if not os.path.isdir(ARCHIVE):
        pytest.skip("no archive in this checkout")
    missing = []
    for p in sorted(_markdown_under(ARCHIVE)):
        rel = _rel(p)
        if rel == "docs/archive/README.md":
            continue
        with open(p, encoding="utf-8", errors="replace") as fh:
            head = fh.read(1200)
        if BANNER not in head:
            missing.append(rel)
    assert not missing, (
        "archived markdown without the ARCHIVED banner in its first 1200 bytes "
        "-- an agent reading one of these would treat retired-dataset results as "
        "current:\n  " + "\n  ".join(missing))


def test_the_quarantine_notice_exists_and_names_the_live_record():
    notice = os.path.join(ARCHIVE, "README.md")
    if not os.path.isfile(notice):
        pytest.skip("no archive in this checkout")
    text = open(notice, encoding="utf-8", errors="replace").read()
    for needed in ("RULESET.md", "MISSION.md", "LEDGER.md", "FRAMEWORK.md"):
        assert needed in text, (
            "the quarantine notice does not point at %s, so it does not tell a "
            "resuming agent where the live record actually is" % needed)


def test_the_live_document_set_is_small_and_known():
    """Catch documentation sprawl at the moment it starts, not two years later."""
    # Git-tracked files only. Walking the filesystem sweeps in agent skills,
    # MCP caches and virtualenvs, none of which are this project's documentation.
    out = subprocess.run(["git", "-c", "gc.auto=0", "ls-files", "*.md"],
                         cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    found = {rel for rel in out.stdout.splitlines()
             if rel.strip() and not rel.startswith("docs/archive/")}
    unexpected = sorted(found - LIVE_DOCS)
    assert not unexpected, (
        "new live markdown outside the known record:\n  " + "\n  ".join(unexpected)
        + "\n\nEither fold it into RULESET/MISSION/LEDGER, archive it under "
          "docs/archive/ with the banner, or add it to LIVE_DOCS deliberately.")
    gone = sorted(d for d in LIVE_DOCS - found
                  if os.path.exists(os.path.join(ROOT, d)) is False)
    assert not gone, "a live document disappeared: %s" % gone


def test_no_live_document_mentions_a_retired_dataset_without_saying_it_is_retired():
    """The contamination guard.

    The user's concern is that a retired medical dataset gets quoted as if it
    were current. A live document may name one -- MISSION explains why each was
    dropped, and LEDGER records the leakage -- but the same file must carry the
    retirement language, so the name can never appear as a live result.
    """
    offenders = []
    for rel in sorted(LIVE_DOCS):
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            continue
        low = open(p, encoding="utf-8", errors="replace").read().lower()
        named = [d for d in RETIRED if d in low]
        if named and not any(w in low for w in RETIREMENT_WORDS):
            offenders.append("%s names %s with no retirement language"
                             % (rel, ", ".join(named)))
    assert not offenders, "\n  ".join([""] + offenders)
