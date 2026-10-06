# Namespace source-cache boundary

The approved snapshot pilot needs authenticated source identity before imports.
The CPU namespace already starts isolated Python and disables bytecode writes,
but `-B` still accepts existing caches. A cache can carry different executable
code while its header matches the authenticated `.py` file. This is a demonstrated
fictitious substitution, not evidence of historical or current server tampering.

The command now sets `-X pycache_prefix=/tmp/python-bytecode` inside the fresh
namespace `/tmp`, retaining `-I -B`. Standard source imports look for caches under
that fresh prefix and cannot write new ones. No cache is deleted or modified.
The existing mounts, finite limits, child collection and scientific recipe are
unchanged. This does not enable either scientific CLI or issue campaign gates.

Seven new stdlib CPU checks independently build timestamp, checked-hash and
unchecked-hash substitutions. Three negative controls show that isolated `-B`
alone accepts them; three corrected-command checks execute authenticated source,
leave the original cache intact and create no prefix cache. One check binds the
prefix to fresh namespace `/tmp`. The directly affected mount-contract check is
also required. Original local four failures are preserved outside source.
The first two-host focused attempt exposed a fixture assumption: its cache path
inherited the parent's new cache prefix, while the negative-control child had
none. All six cache fixtures stopped before their assertions. Those originals
remain preserved; the fixture now places the intended ordinary source-cache
path explicitly. The production command and all assertions remain unchanged.

Source/host receipts and any new fictitious namespace execution belong outside
the immutable release. No real image, weight, model, development target, GPU or
scientific seed is needed for these checks. Passing them does not certify runtime
libraries, native extensions, arbitrary explicit bytecode loaders, source mutation
between authentication and import, private/device isolation or historical cache
contents. The earlier actual CPU trajectory is not rerun at this source. Actual
campaign gates, historical seed freshness and certified finite budget remain open.
