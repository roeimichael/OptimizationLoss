# Requested single-device mount preparation

This source step prepares the approved knee pilot's future GPU namespace.
It does not execute a GPU command, inspect a device, or enable either scientific
CLI. The CPU command, finite collector, scientific recipe and model driver stay
unchanged. No infrastructure/model seed is needed for these stdlib examples.

The builder receives one complete nonzero physical UUID and one explicit
`/dev/nvidiaN` node. It retains the existing read-only public/runtime/source
mounts, fresh `/dev` and `/tmp`, clear environment, isolated Python, closed network
and finite outer collector. It adds only that requested physical node plus the
three declared driver control/UVM interfaces. It changes only the command's
visible-device selection; it does not forward a host selection or `all` setting.

Purpose: make the device exposure explicit before implementing a real launcher.
An environment selection alone does not restrict the available device nodes.
Mounting all of host `/dev` would expose unnecessary interfaces. A requested
node/UUID pair is nevertheless only a declaration. Shared driver control/UVM
interfaces mean this mount list alone does not prove isolation from other GPUs.
Kernel/driver behavior and the authentic same-host UUID-to-node mapping require
actual validation, after source/private/seed, fresh both-host exclusive ownership
and certified finite budget gates permit entry. Missing nodes must refuse rather
than silently broaden mounts. Do not execute this plan to discover whether it
works while those gates remain open.

Prospective validation: fifteen new stdlib fictitious command examples plus one
directly affected CPU mount example, locally and at the immutable release on
each host. Plugin loading is disabled and Torch must remain absent before/after.
Host checks have 15 CPU seconds, 512 MiB address space, 45 alarm-wall seconds and
90 outer SSH seconds. No plan is executed, no real GPU node is read/mounted,
and no model, tensor library, input image, weight or target is opened. A future
need to change the device interfaces or GPU boundary requires its own observed
driver/OS evidence; these command examples cannot certify real CUDA readiness.
