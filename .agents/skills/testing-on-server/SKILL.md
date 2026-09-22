---
name: testing-on-server
description: Commit, publish, deploy and verify OptimizationLoss changes on the DSI GPU servers, with ownership checks, architecture-aware execution and evidence receipts. Use when testing code or running authorized experiments on dsisco01 or dsisco02.
---

# Testing on Server

Read [the server procedure](references/server-procedure.md) before deployment or
GPU use. Use [the checkpoint record](references/checkpoint-record.md) to document
what changed, why, exact versions, commands, tests, results and limitations.

## Working agreement

For every coherent change: run relevant local tests; inspect the actual outputs;
commit locally; push to GitHub and the DSI mirror; create a new immutable release;
verify commit AND file hashes remotely; run server tests; retain pass/fail receipts.
Local commits already update the local repository; a separate local push is not
needed. Do not claim a checkpoint is deployed or validated if any stage failed.

Keep scientific choices revisable. Record the rationale for the dataset, split,
metric, quota, model, optimizer, precision and budget. Fixtures do not prescribe
research hyperparameters. Distinguish infrastructure smoke, pipeline pilot and
scientific comparison. A passing smoke does not validate Clipper mathematics.

## GPU ownership and hardware

- Check BOTH hosts each time. They share NFS paths, not processes or GPUs.
- `dsisco01`: Quadro RTX 6000, Turing architecture (Quadro is the product family).
- `dsisco02`: RTX PRO 6000 Blackwell Server Edition. Prefer a free Blackwell GPU
  when the campaign's precision/source requirements allow it; never displace a
  user to obtain one. Hardware names, capability and free memory must be verified
  at runtime, not assumed from this inventory.
- Map GPU UUID -> compute PID -> process owner and command. Inspect this user's
  relevant `/proc/PID/cwd` and only needed environment keys. Low utilization is
  NOT ownership permission. Do not kill, share or repurpose another user's GPU.
- Use only an unoccupied GPU within the user's authorized allocation; respect
  scheduler/reservation rules if present. Recheck immediately before dispatch.
- Select explicitly with `CUDA_VISIBLE_DEVICES`; record physical UUID and local
  device index. Refuse silent CPU fallback for a GPU test.

## Precision and memory

Use explicit, tested device-aware paths, not edits to a live server checkout.
FP32 is a useful initial reference on both hosts. Turing does not natively support
BF16: use FP32 or tested FP16; FP16 gradient training needs scaling and applied/
skipped/nonfinite-step accounting. Blackwell can use BF16 when supported by the
installed build. Verify capability and kernel support with a small operation.
Inference-only autocast has no optimizer and needs no GradScaler. Do not claim
FP16 training is tested merely because FP16 inference passed. Microbatching,
gradient accumulation, TF32 and effective batch size must be explicit and logged.
Do not combine different precisions/hosts as interchangeable replications.

## Evidence and stopping

- Never edit an active release or its configurations. Fix, test, commit and
  deploy a NEW version; retain the failed attempt and explain the amendment.
- Use exclusive run directories and append-only logs. Do not replace a file
  held open by a process. A timeout means unknown state; inspect before retrying.
- Individual evaluation labels do not enter training or allocation. Any permitted
  aggregate quota information needs explicit provenance. Do not silently derive
  budgets from held-out labels.
- Record planned/attempted/applied/skipped updates separately. Check real source,
  data, gradients, allocation, metrics and artifacts, not only test counts.
- No automatic recurring monitor is authorized by this skill. The previous
  five-minute monitor was cancelled; create one only when the user asks.
- Preserve datasets, predictions, checkpoints and Git objects. Cleanup requires
  an explicit path/symlink/process inventory and recoverable archival.

The user's current task determines research scope. Global-only work is authorized
for the rebuild; do not import the old dual-constraint-only restriction.
