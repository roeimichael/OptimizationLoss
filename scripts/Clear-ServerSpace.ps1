<#
.SYNOPSIS
  Reclaim space in the michaer8 home quota on the dsisco servers.

.DESCRIPTION
  Dry run by default: it prints what it WOULD remove and stops. Nothing is
  deleted until you pass -Apply.

  It never works from a hand-written list of doomed directories. It keeps an
  explicit KEEP set and treats everything else as a candidate, then applies two
  safety rules to each candidate ON THE SERVER:

    1. A tree containing any final_predictions.csv is EVIDENCE and is kept,
       unless you also pass -IncludeEvidence.
    2. A tree containing the target of the live data symlink is never removed,
       whatever else you pass.

  Because the rules are evaluated remotely against the real tree, a directory
  this script has never heard of is still protected.

.PARAMETER Apply
  Actually delete. Without it, nothing is removed.

.PARAMETER IncludeEvidence
  Also remove trees holding prediction files. These are the pre-reset campaign
  results. Deleting them is not recoverable from git.

.PARAMETER Host
  dsisco02 (default) or dsisco01. Both mount the same NFS home, so it makes no
  difference which one runs it.

.EXAMPLE
  .\Clear-ServerSpace.ps1
  .\Clear-ServerSpace.ps1 -Apply
  .\Clear-ServerSpace.ps1 -Apply -IncludeEvidence
#>
[CmdletBinding()]
param(
    [switch]$Apply,
    [switch]$IncludeEvidence,
    [ValidateSet('dsisco01', 'dsisco02')]
    [string]$ServerHost = 'dsisco02'
)

$applyArg    = if ($Apply)           { 'apply' }    else { 'dryrun' }
$evidenceArg = if ($IncludeEvidence) { 'evidence' } else { 'protect' }

# Runs on the server. Keep it POSIX sh: no bashisms beyond what dash allows.
$remote = @'
set -u
APPLY="$1"
EVIDENCE="$2"

# Everything NOT listed here is a candidate for removal.
#   anaconda3        the optloss env, hours to rebuild
#   optloss-audit    the ONLY real dataset bytes on the server
#   optloss-rank     the working tree and the scored Option C results
#   OptimizationLoss / StocksProject  other checkouts, not this pipeline's
#   inv_probe / logs / queue_logs     diagnostics and run logs, ~25M total
# A directory whose name contains stray carriage returns (queue_logs,
# from the CRLF transport bug) does NOT match its clean twin and is removed.
KEEP="anaconda3 optloss-audit optloss-rank OptimizationLoss StocksProject inv_probe logs queue_logs"

# Resolve the live data symlink. Whatever holds it is untouchable.
LIVE=$(readlink -f "$HOME/optloss-rank/data/fmow2" 2>/dev/null || true)
[ -n "$LIVE" ] && echo "live data resolves to: $LIVE"

echo ""
if [ "$APPLY" = "apply" ]; then echo "MODE: DELETING"; else echo "MODE: DRY RUN -- nothing will be removed"; fi
echo ""
printf '%-40s %9s %7s  %s\n' "TREE" "SIZE" "PREDS" "ACTION"
printf '%s\n' "------------------------------------------------------------------------------"

freed=0
kept_evidence=0

remove_tree() {
    # $1 path, $2 human size, $3 bytes, $4 label
    if [ "$APPLY" = "apply" ]; then
        rm -rf -- "$1" && printf '%-40s %9s %7s  REMOVED\n' "$4" "$2" "${5:-0}"
    else
        printf '%-40s %9s %7s  would remove\n' "$4" "$2" "${5:-0}"
    fi
    freed=$((freed + $3))
}

# --- the regenerable warm-up cache, which lives INSIDE a kept tree ----------
MC="$HOME/optloss-rank/model_cache"
if [ -d "$MC" ]; then
    b=$(du -sb "$MC" 2>/dev/null | cut -f1)
    h=$(du -sh "$MC" 2>/dev/null | cut -f1)
    remove_tree "$MC" "$h" "$b" "optloss-rank/model_cache" "0"
fi

# --- every other top-level directory ---------------------------------------
for p in "$HOME"/*; do
    [ -d "$p" ] || continue
    [ -L "$p" ] && continue
    name=$(basename "$p")

    skip=0
    for k in $KEEP; do [ "$name" = "$k" ] && skip=1; done
    [ "$skip" -eq 1 ] && continue

    # Rule 2: never remove a tree that holds the live data.
    case "$LIVE" in
        "$p"|"$p"/*)
            printf '%-40s %9s %7s  KEPT -- holds the live data\n' "$name" "-" "-"
            continue ;;
    esac

    h=$(du -sh "$p" 2>/dev/null | cut -f1)
    b=$(du -sb "$p" 2>/dev/null | cut -f1)
    np=$(find "$p" -name final_predictions.csv 2>/dev/null | wc -l | tr -d ' ')

    # Rule 1: predictions are evidence.
    if [ "$np" -gt 0 ] && [ "$EVIDENCE" != "evidence" ]; then
        printf '%-40s %9s %7s  KEPT -- evidence (pass -IncludeEvidence)\n' "$name" "$h" "$np"
        kept_evidence=$((kept_evidence + b))
        continue
    fi
    remove_tree "$p" "$h" "$b" "$name" "$np"
done

echo ""
echo "reclaimable here: $(echo "$freed" | awk '{printf "%.1f GiB", $1/1073741824}')"
if [ "$kept_evidence" -gt 0 ]; then
    echo "held back as evidence: $(echo "$kept_evidence" | awk '{printf "%.1f GiB", $1/1073741824}') (-IncludeEvidence to release)"
fi
echo ""
echo "quota now:"
df -h "$HOME" | tail -n 1
'@

# Shell transport mangles text, and two separate mechanisms corrupt a piped
# here-string here: PowerShell 5.1 prepends a BOM when writing to a native
# command's stdin, so sh reports "set: command not found" on line 1; and this
# .ps1 has CRLF endings, so every remote line arrives with a trailing CR and
# tail reports "invalid number of lines". Base64 is immune to both -- normalise
# to LF, encode, and let the server decode exactly the bytes intended.
$payload = $remote -replace "`r`n", "`n"
$b64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($payload))

Write-Host "Connecting to $ServerHost ..." -ForegroundColor Cyan
ssh $ServerHost "echo $b64 | base64 -d | sh -s -- $applyArg $evidenceArg"

if (-not $Apply) {
    Write-Host ""
    Write-Host "Dry run only. Re-run with -Apply to delete." -ForegroundColor Yellow
}
