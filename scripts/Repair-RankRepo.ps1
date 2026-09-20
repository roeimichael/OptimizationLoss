<#
.SYNOPSIS
  Re-attach ~/optloss-rank to git after its parent worktree repo was removed.

.DESCRIPTION
  optloss-rank was a git WORKTREE whose parent repository lived in
  optloss-probe. Clear-ServerSpace removed optloss-probe -- correctly, by its
  own rules: the tree held zero predictions and was no longer the target of the
  data symlink. What those rules did not model is that a worktree keeps its git
  metadata in the PARENT, so removing the parent leaves the child's .git as a
  dangling 69-byte pointer.

  Nothing was lost. A worktree's working files are ordinary files: all source,
  all configs and all 364 prediction files are intact. Only the pointer is
  broken. This converts optloss-rank into a standalone clone, in place, which
  also removes the fragility that caused this.

  Dry run by default. Nothing changes until you pass -Apply.

.PARAMETER Apply
  Perform the repair.

.EXAMPLE
  .\Repair-RankRepo.ps1
  .\Repair-RankRepo.ps1 -Apply
#>
[CmdletBinding()]
param(
    [switch]$Apply,
    [ValidateSet('dsisco01', 'dsisco02')]
    [string]$ServerHost = 'dsisco02'
)

$applyArg = if ($Apply) { 'apply' } else { 'dryrun' }

$remote = @'
set -u
APPLY="$1"
R="$HOME/optloss-rank"
BRANCH="cleanup/consolidate-pipeline"
ORIGIN="git@github.com:roeimichael/OptimizationLoss.git"

say() { printf '%s\n' "$*"; }
run() { if [ "$APPLY" = "apply" ]; then eval "$@"; else say "    would run: $*"; fi; }

# --- pre-flight: refuse to touch anything unless the tree looks right --------
say "== pre-flight =="
fail=0
[ -d "$R/src" ]     || { say "  MISSING $R/src";     fail=1; }
[ -f "$R/main.py" ] || { say "  MISSING $R/main.py"; fail=1; }
PREDS=$(find "$R/results" -name final_predictions.csv 2>/dev/null | wc -l | tr -d ' ')
say "  predictions present: $PREDS"
[ "$PREDS" -ge 364 ] || { say "  REFUSING: expected at least 364 predictions"; fail=1; }
DATA=$(readlink -f "$R/data/fmow2" 2>/dev/null || true)
say "  data symlink -> ${DATA:-MISSING}"
[ -n "$DATA" ] && [ -d "$DATA" ] || { say "  REFUSING: data symlink is broken"; fail=1; }
if [ "$fail" -ne 0 ]; then say ""; say "PRE-FLIGHT FAILED -- nothing done."; exit 2; fi
say "  ok"
say ""

if [ "$APPLY" = "apply" ]; then say "MODE: REPAIRING"; else say "MODE: DRY RUN"; fi
say ""

# --- 1. back up the source, so a bad checkout is recoverable -----------------
say "1. back up src/ configs/ scripts/ main.py"
run "tar czf \"\$HOME/rank_source_backup_\$(date +%Y%m%d_%H%M%S).tgz\" -C \"$R\" src configs scripts main.py"

# --- 2. park the dangling pointer (moved, never deleted) ---------------------
say "2. park the dangling .git pointer"
if [ -f "$R/.git" ]; then
    say "    current: $(cat "$R/.git")"
    run "mv \"$R/.git\" \"$R/.git.dangling-worktree-pointer\""
else
    say "    no .git file -- already repaired or already a real repo"
fi

# --- 3. make it a standalone repository --------------------------------------
say "3. init a standalone repo and fetch $BRANCH"
run "git -C \"$R\" -c gc.auto=0 init -q"
run "git -C \"$R\" -c gc.auto=0 remote add origin \"$ORIGIN\" 2>/dev/null || git -C \"$R\" -c gc.auto=0 remote set-url origin \"$ORIGIN\""
run "git -C \"$R\" -c gc.auto=0 fetch --depth=50 origin \"$BRANCH\""
run "git -C \"$R\" -c gc.auto=0 checkout -f -B \"$BRANCH\" \"origin/$BRANCH\""

# --- 4. the data symlink is the one thing a checkout could clobber -----------
say "4. re-assert the data symlink"
run "[ -e \"$R/data/fmow2\" ] || ln -s \"$DATA\" \"$R/data/fmow2\""

# --- 5. verify by EXECUTION, not by reading the diff -------------------------
say ""
say "== verification =="
if [ "$APPLY" = "apply" ]; then
    say "  HEAD:        $(git -C "$R" -c gc.auto=0 log --oneline -1 2>&1)"
    say "  branch:      $(git -C "$R" -c gc.auto=0 rev-parse --abbrev-ref HEAD 2>&1)"
    say "  new code:    permute_group_budgets appears $(grep -c permute_group_budgets "$R/src/training/constraints.py" 2>/dev/null || echo 0) times in constraints.py"
    say "  predictions: $(find "$R/results" -name final_predictions.csv 2>/dev/null | wc -l | tr -d ' ') (was $PREDS)"
    say "  data:        $(readlink -f "$R/data/fmow2" 2>/dev/null || echo BROKEN)"
    say "  backups:     $(ls -1 "$HOME"/rank_source_backup_*.tgz 2>/dev/null | wc -l | tr -d ' ')"
else
    say "  (dry run -- nothing verified because nothing changed)"
fi
'@

$payload = $remote -replace "`r`n", "`n"
$b64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($payload))

Write-Host "Connecting to $ServerHost ..." -ForegroundColor Cyan
ssh $ServerHost "echo $b64 | base64 -d | sh -s -- $applyArg"

if (-not $Apply) {
    Write-Host ""
    Write-Host "Dry run only. Re-run with -Apply to repair." -ForegroundColor Yellow
}
