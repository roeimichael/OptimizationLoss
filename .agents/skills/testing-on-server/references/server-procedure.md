# DSI deployment and execution

## Known locations (verify, do not overwrite blindly)

- GitHub: `https://github.com/roeimichael/OptimizationLoss.git`
- Working branch: `codex/tralo-rebuild-20260922` (read current branch each time).
- Local worktree: `C:/Users/roeym/.codex/worktrees/tralo-rebuild-20260922/OptimizationLoss`
- SSH aliases: `dsihead`, `dsisco01`, `dsisco02`, user `michaer8`.
- DSI mirror: `/home/dsi/michaer8/tralo-rebuild.git`
- Immutable releases: `/home/dsi/michaer8/tralo-rebuild/releases/<full-SHA>`
- Receipts: `/home/dsi/michaer8/tralo-rebuild/verification/<SHA>/<host>/<attempt>`
- Runs/data live outside releases under `/home/dsi/michaer8/tralo-rebuild/`.
- Existing interpreter: `/home/dsi/michaer8/anaconda3/envs/optloss/bin/python`.

Prefix local shell commands with `rtk`; `rtk proxy` passes commands through.
Use `git -c gc.auto=0`. The user's Windows shell is PowerShell. For remote Git
pushes explicitly use Windows OpenSSH if Git's bundled SSH fails with the
ProxyJump error `exec is not recognized`:

```text
git -c gc.auto=0 -c core.sshCommand=C:/Windows/System32/OpenSSH/ssh.exe push ssh://dsisco02/home/dsi/michaer8/tralo-rebuild.git HEAD:refs/heads/<branch>
```

Use subprocess argument lists, not interpolated shell command strings. Bound
the entire SSH subprocess, e.g. `subprocess.run(..., timeout=30)`. ConnectTimeout
alone does not bound blocked banner/authentication or remote-command phases.
Read-only peer checks may run in parallel. Check return codes and stderr.

## Checkpoint sequence

1. Inspect branch/status; protect unrelated changes. Run tests appropriate to the
   actual change. Independently calculate a small expected result when changing
   mathematics. Compile/smoke the real CLI, not only isolated helpers.
2. Stage only this checkpoint; commit a message describing the concrete change.
   Push the exact branch to origin, then the DSI bare mirror. No force-push.
3. Create a fresh detached checkout, with a path derived from the verified full
   hexadecimal commit SHA, using the remote bare repository's `git worktree add`.
   If the path exists, validate it rather than resetting it. Never deploy over
   `/home/dsi/michaer8/optloss-*` research trees.
4. Compare HEAD, clean tracked status and SHA-256 of tracked files against local
   COMMITTED blob bytes (`git show SHA:path`), avoiding Windows newline ambiguity.
   Save the manifest and interpreter/package/device information in a receipt.
5. Run CPU regression tests and a CLI example from the new release on both hosts.
   For the current first slice: `python -m unittest discover -s tests -v`, then
   `python -m tralo.inspect_predictions examples/hand_predictions.json <new-output>`.
6. Recheck ownership immediately before GPU use. Query `nvidia-smi` for physical
   indices, UUIDs, names, memory and compute PIDs; join PIDs to `ps` owners. Inspect
   current user jobs' cwd and `EXPERIMENT_DIR`, `CUDA_VISIBLE_DEVICES` only. Never
   dump complete environments, which may contain secrets.
7. Launch using an explicit environment dictionary or correctly quoted shell.
   Do not nest `bash -c` layers that lose environment variables. A bounded small
   test can be foreground; a long run needs an exclusive receipt, detached launch
   and PID, and active progress checks. Never duplicate a run after SSH loss.
8. Read result files and logs. Verify actual predictions, quotas and metrics;
   don't infer success from process exit alone. Save stdout/stderr, return code,
   hashes, timing and scope of inference. Any failure is preserved before repair.

The current CUDA fixture is `python tools/gpu_smoke.py <exclusive-output>` with
`CUDA_VISIBLE_DEVICES=<free-index>`. It checks a two-row analytic CE/Adam example
and logger state neutrality. It is not a dataset/Clipper run.

## Architecture adaptations

Check `torch.cuda.get_device_capability()`, `torch.cuda.is_bf16_supported()`,
Torch/CUDA versions and model memory needs. An architecture-specific failure
requires a tested source change and new release, not a local server patch.
Keep the math and scientific configuration common where intended; record
intentional precision/batch differences as separate conditions.

For FP16 training, record the scaler and whether optimizer steps applied. A
nonfinite loss/gradient or skipped update cannot be hidden by logging a nominal
epoch count. For FP32/BF16 training do not add a scaler merely by convention.
Use a small forward/backward/step check on the chosen GPU before a long run.
