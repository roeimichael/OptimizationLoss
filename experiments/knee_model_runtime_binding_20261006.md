# Model placement and runtime identity in the snapshot driver

Prospective infrastructure change, 2026-10-06. No scientific seed, CUDA entry,
GPU plan or scientific campaign is authorized by this component.

The existing requested/runtime UUID observer is disconnected from the model
accepted by `fit_public`. Connect it to the driver before public payloads and
output creation: every parameter and buffer must occupy the single visible
CUDA index zero, and a GPU model requires an explicit requested physical UUID.
The observer must actually obtain raw runtime UUID bytes; accepting a caller's
JSON observation is not an alternative. The CPU readiness branch keeps no GPU
observation and requires all parameters and buffers to be on CPU. Neither
branch moves or constructs a model, changes RNG or alters the scientific fit.

The execution record carries separate requested and observed UUIDs, hardware
properties and explicit false ownership/permission fields. Label-free artifact
verification checks those stored fields against device/initialization tags.
This is metadata consistency, not an independent replay of runtime identity.
Missing, mixed, wrong-index or incompatible placements must fail before the
runtime query. Mutated or rehashed logs must still fail semantic validation.

Alternative: leave all placement checks to a future launcher. That leaves a
constructed model and the stored execution record unbound at the driver call.
Reconsider this check only with evidence that another authenticated entry point
enforces the same placement and observation contract. The scientific optimizer,
gradient, snapshot, dose, ensemble and approved recipe stay unchanged.

Prospective validation: 24 new standard-library cases with fictitious model and
CUDA providers, plus one NEW CPU Tiny-classifier integration on the existing
40-train/100-development synthetic pack, two epochs, infrastructure seed
9483001. No scientific 7001-7004 claims. The integration checks CPU logs and
rehashed device-log refusal in that same original run. Native collection, if
needed, uses 60 CPU seconds, 16 GiB address space, 120-second alarm wall,
125-second own collector wall and 180-second outer SSH bound on each host.
No actual public image/private/weight/model/quality payload or GPU device is
mounted. Focused fake-provider checks stay within 15 CPU seconds/512 MiB address
space/45-second alarm/90-second SSH; do not relax limits after a failure.

Both scientific CLIs remain refused. Real use can initialize CUDA and is
forbidden until source/private/seed/authentic mapping/fresh both-host exclusive
ownership and certified finite budget gates permit it. Checking a model already
placed by a future launcher does not authenticate that earlier placement or
prove isolation from other GPUs. No native CUDA check is performed here.
