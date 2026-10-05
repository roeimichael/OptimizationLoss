# Authenticate the bytes consumed by the knee image readers

The approved snapshot pilot's training and development readers now hash the
same retained PNG bytes that PIL decodes. Previously they hashed a file and
then reopened its path for decoding. A new fictitious fixture replaces the file
as the first read closes: both old readers decoded the replacement despite the
original hash passing. The two failures are preserved outside the source tree.
This is a demonstrated reader race in a synthetic fixture, not evidence that
any actual input or earlier study was modified.

Each reader now reads an image once, authenticates those bytes, and decodes a
memory buffer. The training reader still accepts only training rows and retains
RGB images for the unchanged per-access transforms. The development reader
still consumes image-only rows in their original order and chunk sizes. No
target, group, cap, recipe, transform, sampler, allocator, seed or backbone is
changed. Existing immutable releases and results remain untouched.

Fourteen new independent CPU cases check known RGB, grayscale and RGBA pixels,
replacement after the authenticated read, refusal before decode for a wrong
hash, malformed image bytes, public chunk order and training-role refusal.
The two mutation cases failed before implementation; all fourteen then passed.
Sixteen directly affected fictitious driver tests also passed, for thirty
focused nodes locally. Target-host source and focused-test receipts belong to
the exact deployed commit; no actual images or private targets are needed for
this validation.

This prepares the actual input boundary. It does not issue a campaign approval,
certify OS/device/FD isolation, historical seed non-use, GPU cost or a finite
balance. The training and scoring CLIs retain their refusal. The completed real
one-epoch CPU gate remains evidence at `778331aca8399345255f510bf6d769b4da5012cb`;
it is not a newly rerun real trajectory at this later reader release. The full
CPU baseline remains `cc21dd1c41f332f4147a7cf298e96762b13cde3e`.

No earlier full suite, decode audit, snapshot core, scoring block or completed
real CPU attempt is repeated. Scientific seeds 7001–7004 remain unclaimed.
The partial planning charge remains 75.158211024 GPUh; certified remaining
budget remains unavailable. No GPU query, context, runtime or cost is added.
