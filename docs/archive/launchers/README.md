> 🛑 **ARCHIVED -- HISTORY, NOT INSTRUCTIONS.** (banner added 2026-09-11)
> These four `.sh` files are EXECUTABLE and each one starts a real
> campaign on a real GPU. Every campaign they launch is quarantined or
> PARTIALLY quarantined, they pin hosts and roots that have since
> moved, and they predate the current recipe. Do not run them. To
> launch anything, generate it with `configs.gen_campaign` and walk it
> through `scripts.run_campaign --step`.
> `docs/FRAMEWORK.md` is the ONLY operational document.

# Archived campaign launchers

One-off launch wrappers for `dom1`, `dom1b`, `equaldose1` and `uniform1`,
moved here 2026-09-04. They are history, not instructions.

Every campaign they launch is either **quarantined** or **partially
quarantined**:

* `uniform1` -- `scorable=False`. 252 runs, mechanically perfect, and all 9
  cells sit outside the measured task window, so it measured the absence of a
  question.
* `dom1`, `dom1b`, `equaldose1` -- **PARTIAL**. `fioretto` and `hounie` (and
  `equaldose1`'s `tralo_lam0`) ran at 28.00 attempted constraint steps against
  29.00, so any contrast touching them is not comparable.

They also hardcode a host, a GPU index and an `EXPERIMENT_DIR`, and the
2026-09-04 incident showed how easily that splits a campaign across two hosts
with different AMP regimes. **Do not re-run them.** Stage a campaign with
`configs.gen_campaign` and check `scripts.rig_status` before and after launch.
