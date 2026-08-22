"""The repository's commit, as one implementation.

TWO DIFFERENT FACTS, ONE PRIMITIVE. A run carries two provenance stamps and
they answer different questions:

    code_version       the commit that WROTE config.json (configs/gen_campaign)
    run_code_version   the commit that PRODUCED THE WEIGHTS (src/experiments/runner)

They are equal only when nothing landed between generating a campaign and
finishing it. The generator stamps every config once, at generation time, and
never revisits a config it already wrote -- so run half a campaign, land a
change to a training file, resume the rest, and every config still carries the
ORIGINAL stamp. `full_panel`'s provenance gate then sees one value across the
whole tree and scores both halves as one comparison, which is the exact thing
it was written to refuse; and `model_cache` hands the post-change runs the
pre-change warm-up because the two configs agree on a key that describes
neither run.

The stamp is only as sharp as the working tree is clean: `-dirty` says the
tree had uncommitted changes but not WHICH, so two dirty runs an edit apart
carry the same string. That is the same limitation `code_version` has always
had, and it is why `check_parity` warns on `-dirty` rather than passing it.

Stdlib only, on purpose: `configs/gen_campaign.py` imports this and must keep
running on a machine with no torch.
"""

import os
import subprocess

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def git_version(repo_root=None):
    """Short git SHA + a `-dirty` suffix, or "unknown" off a checkout.

    `cwd` is pinned to the repo rather than left to the process's working
    directory: campaigns are launched from several directories, and a bare
    `git rev-parse` run from outside the tree reports a DIFFERENT repository's
    commit rather than failing.
    """
    root = repo_root or _REPO_ROOT
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=root, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"],
            cwd=root, stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"
