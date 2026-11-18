# 810E direct INT4 K-pack tuning

## Contract

Continue from `711c96b8` on physical ZW810E card 1, native `ppu_10`, SDK
`/usr/local/PPU_SDK`. Preserve the pinned dependencies and original comparison
installation. Deliver a CUTLASS extension selected by the QUactlize public
query/prepare/run path and packaged by the normal wheel builder.

Compare identical INT4 codes, group32 parameters, FP16 activations/output and
FP32 accumulation against installed Humming 0.1.19. Offline repacking is outside
timing. Include required reduction and initialization in each measured operation;
exclude task-owned adapters. Promotion requires candidate latency at most 90%
of Humming under paired ABBA measurements, with independently validated outputs.
Symmetric and integer-zero affine contracts require separate evidence.

## Baseline and corrections

The old 936-row delivery compares Q4_K/FP32 endpoints against affine INT4/FP16
endpoints and is diagnostic only. The initial new `affine-native-sym-r0/r1`
screening accidentally instantiated historical Xplane `DenseKernelTypes`.
Its 456 successful configurations do not measure the current K-pack production
route. Its serial Split-K also retains an FP16 intermediate; it cannot establish
the required FP32-partial comparison. Preserve these artifacts as rejected
baseline selection, not as evidence about current K-pack performance.

The production type authority is `quactlize/runtime/kernel_types.cuh`.
Q4 uses `DenseQ4KPack4KernelTypes`; the prepared ordinary path uses
`PreparedMultiformatLauncher`, including compact FP32 partials and a final
reducer. K-pack describes code storage, independently of Q4_K metadata.
Reuse that reader with native FP16 scales; do not synthesize GGUF metadata.

Source also contains `StreamKMixedInputKernel`, `MarlinMixedInputKernel`,
and classic `MarlinCollectivePPU`. The current runtime recipe enum exposes
ordinary/persistent, not all these experimental schedulers. Their existence
does not prove selection or group32/K-pack compatibility. CTA scheduling,
warp-level K partitioning and operand readers are separate axes.

## Ranked work

1. Correct the binding to K-pack and production FP32 reduction, with compile-time
   layout witnesses and an independent host decoder of the emitted K-pack bytes.
2. Run a bounded baseline grid, then profile a slow small-M and large-M case.
   Check actual symbols, launches, thread geometry, register billing, pipeline
   issue, conversion and memory traffic before naming a bottleneck.
3. Sweep existing tile/warp/stage/delivery/reduction choices on fixed K-pack.
   Evaluate existing Stream-K/Marlin drivers as separate controlled candidates.
4. Implement one evidenced reader/pipeline/scheduler change at a time.
5. Add integer-zero affine arithmetic with the same operation ordering as
   Humming. A floating affine offset is not an integer zero point.
6. Integrate admitted configurations, build an isolated wheel, and repeat
   correctness and paired performance through the public API.

## Risks and counterexamples

- Native layout alone does not establish performance superiority.
- WarpN/TileK delivery restrictions must be derived from the selected reader;
  historical Xplane restrictions must not prune K-pack candidates.
- A larger warp-M amortizes conversion but can cross a register billing bucket.
- Source Stream-K support may be absent from production recipe selection.
- GGUF Scale-first expands metadata, not weight codes. It preserves a floating
  affine offset with explicit FP16 rounding; it does not imply integer-zero GPTQ
  equivalence. Other bitwidths retain their respective code planes and group sizes.
- Query/prepare may initialize resources. Graph run must allocate/JIT/synchronize
  neither host nor device and may not omit algorithm-required reduction/reset.

## Evaluation

Use the commands in `plan.md`. Keep all result directories fresh and retain
raw JSONL, source/binary identities, harness snapshots, candidate lineage and
profiling settings. Do not promote a sampled, failed, unsupported or unrun case.
ACU replay times are diagnostic, not paired event latency.
