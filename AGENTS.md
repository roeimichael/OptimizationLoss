# Working on the rebuild

- Prefix shell commands with `rtk`; use `rtk proxy` for passthrough.
- This branch is a fresh implementation. Do not import or copy legacy modules.
- Preserve old evidence outside the active tree; never delete data, checkpoints,
  predictions, or Git objects. Do not edit the original checkout or live servers.
- Read DESIGN.md. Keep this first slice small: allocation, metrics, logging.
- Write independent examples before implementation. A legacy result is not an oracle.
- Never let labels enter allocation; labels are used only by the metric function.
- No GPU campaigns until data, training, gradients, and source identity are validated.
- No silent scientific defaults: explicitly name allocation policy and quotas.
- Never claim feasibility implies optimality, or tests prove no bugs remain.
- Ask before changing the scientific question, data access, or compute budget.
- Use `git -c gc.auto=0`; no pruning or destructive cleanup.
