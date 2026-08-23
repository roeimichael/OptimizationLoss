#!/bin/bash
# Remove a research arm completely: processes, runs, worktree, branch, scratch.
#
#   bash paper/scripts/nd_fire.sh <arm> [--dry-run]
#
# Used when a direction has been tested and failed, so its slot can be given to
# a new one. The point is to leave NO residue: no orphan process still holding a
# GPU, no half-written results that a later analysis script would silently pick
# up and average into a comparison, no stale branch.
#
# SAFETY. Every arm worktree contains `data` and `model_cache` SYMLINKS into the
# shared 71G dataset store and the 396-entry warm-up cache. Those are shared with
# every other arm and with the frozen checkout, and must survive. So the symlinks
# are unlinked FIRST, individually, before anything recursive runs. `rm -rf` does
# not follow symlinks, but it does follow a path written with a trailing slash,
# and one such typo would destroy data that takes days to rebuild. Unlinking
# first makes that mistake impossible rather than merely unlikely.
set -u
ARM="${1:-}"
DRY="${2:-}"
[ -z "$ARM" ] && { echo "usage: nd_fire.sh <arm> [--dry-run]"; exit 1; }

REPO="$HOME/OptimizationLoss"
WT="$REPO/newdirections/arm_$ARM"
run() { if [ "$DRY" = "--dry-run" ]; then echo "  WOULD: $*"; else eval "$@"; fi; }

cd "$REPO" || exit 1
echo "=== firing arm: $ARM ==="

# 1. Stop anything still running for this arm. Matched on the arm's own
#    EXPERIMENT_DIR so no other lane is touched.
PIDS=$(ps -eo pid,args | grep -E "EXPERIMENT_DIR=[^ ]*arm_$ARM|newdirections/arm_$ARM" \
       | grep -v grep | awk '{print $1}')
if [ -n "$PIDS" ]; then
  echo "-- stopping processes: $PIDS"
  for p in $PIDS; do run "kill $p 2>/dev/null || true"; done
  [ "$DRY" = "--dry-run" ] || sleep 3
  STILL=$(ps -eo pid,args | grep -E "EXPERIMENT_DIR=[^ ]*arm_$ARM|newdirections/arm_$ARM" \
          | grep -v grep | awk '{print $1}')
  for p in $STILL; do echo "-- force"; run "kill -9 $p 2>/dev/null || true"; done
else
  echo "-- no running processes"
fi

# 2. Unlink EVERY symlink in the worktree before any recursive delete.
#    Not just data/ and model_cache/ at the top level: `data/` is partially
#    tracked in git (52 files), so a worktree checkout creates a REAL data/
#    directory and the shared datasets are linked one level down as
#    data/dermmnist, data/octmnist, data/tissuemnist. Enumerating the links
#    rather than naming two expected paths is what makes this safe against that
#    layout and any future one.
if [ -d "$WT" ]; then
  LINKS=$(find "$WT" -type l 2>/dev/null)
  if [ -n "$LINKS" ]; then
    echo "-- unlinking $(echo "$LINKS" | wc -l) symlinks (targets preserved):"
    echo "$LINKS" | while read -r l; do
      echo "     $(basename "$l") -> $(readlink "$l")"
      run "rm -f '$l'"
    done
  fi
fi

# 3. Report and remove the arm's runs.
if [ -d "$WT/results" ]; then
  N=$(find "$WT/results" -name config.json 2>/dev/null | wc -l)
  SZ=$(du -sh "$WT/results" 2>/dev/null | cut -f1)
  echo "-- removing $N run configs ($SZ)"
  run "rm -rf '$WT/results'"
fi

# 4. Remove the worktree and its branch.
if [ -d "$WT" ]; then
  echo "-- removing worktree $WT"
  run "git worktree remove --force '$WT' 2>/dev/null || rm -rf '$WT'"
fi
run "git worktree prune"
if git show-ref --quiet "refs/heads/nd/$ARM"; then
  echo "-- deleting branch nd/$ARM"
  run "git branch -D 'nd/$ARM' 2>/dev/null || true"
fi

# 5. Scratch directories arms tend to create outside the worktree.
for d in "$HOME/nd_${ARM}_analysis" "$HOME/${ARM}_out" "$REPO/newdirections/briefs/$ARM"; do
  [ -e "$d" ] && { echo "-- removing scratch $d"; run "rm -rf '$d'"; }
done

# 6. Verify. A silent partial failure here is the exact thing this script exists
#    to prevent, so the state is asserted rather than assumed.
echo "=== verification ==="
P=$(ps -eo args | grep -E "arm_$ARM" | grep -v grep | wc -l)
echo "  processes remaining : $P   $([ "$P" -eq 0 ] && echo OK || echo LEFTOVER)"
echo "  worktree exists     : $([ -d "$WT" ] && echo YES-LEFTOVER || echo no)"
echo "  branch exists       : $(git show-ref --quiet "refs/heads/nd/$ARM" && echo YES-LEFTOVER || echo no)"
echo "  shared data intact  : $([ -d "$REPO/data" ] && du -sh "$REPO/data" 2>/dev/null | cut -f1 || echo MISSING)"
echo "  shared cache intact : $(ls "$REPO/model_cache" 2>/dev/null | wc -l) entries"
df -h "$HOME" | tail -1
