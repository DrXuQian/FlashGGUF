# Native affine INT4 on PPU

The affine API consumes ordinary unsigned INT4 codes, FP16 group32 scales and
optional integer zero points. It uses native K-pack code storage and the
CUTLASS/actlize extension. It does not reinterpret these parameters as GGUF
Q4_K metadata.

The arithmetic contract is FP16 activations and output, FP32 MMA accumulation
and FP32 partial reduction. Each weight operand is:

```
FP16((unsigned_code - integer_zero) * FP16_scale)
```

`zero_mode=0` fixes the zero point at8. `zero_mode=1` accepts independent
uint8 zero points in `[0,15]`. Preparation rejects out-of-range values. A
rounded FP16 affine offset is not a substitute for this integer-zero contract.

## Public ABI

The normal PPU builder links the affine objects into `libncp_quactlize.so` and
ships `include/quactlize/affine/api.h`. The existing dense/MoE/FFN exports remain
in that library. `quactlize-runtime/affine.json` records the compiled
configurations, selection policy, metadata layout and binary/source identity.
The private module produced by `tools/build_affine.py` remains a diagnostic
artifact; it is not shipped as the public execution boundary.

```cpp
#include <quactlize/affine/api.h>

qwa_problem_v1 problem{
    1, sizeof(qwa_problem_v1), m, n, k, 32, -1, 1, zero_mode
};
qwa_config_v1 selected{};
qwa_info_v1 info{};
// Check every returned status before using an output.
int status = quactlize_ppu_w4a16_select_v1(&problem, &selected);
status = quactlize_ppu_w4a16_query_v1(&problem, &info);
```

`config=-1` selects the compiled default during query/preparation. An explicit
configuration ID is also accepted when compiled and compatible. Native default
selection currently uses `split=1`; kernels can still partition K internally.
Existing ABI structure sizes are unchanged.

The candidate V5 selector includes exact integer-zero configurations for the
86 shapes whose shipping Humming route has FP32 accumulation and reduction.
Symmetric entries and generic fallback coverage are retained. The selector
receipt still reports `performance_admitted=false`: passing numerical and
graph-replay checks is not a claim that every shape meets the performance
threshold in a public wheel.

N and K must be positive multiples of256, M must be positive, and matrix
extents must fit the checked indexing range. Query is authoritative for each
compiled configuration's shape and workspace requirements. The native path is
currently built for `ppu_10`. Other supported public-library targets retain the
affine exports with no compiled affine kernels and report unsupported queries.

## Buffer preparation and lifetime

Host packing accepts unsigned `uint8[N,K]` codes and writes exactly `N*K/2`
bytes of canonical K-pack data. Copy these packed bytes to the device before
preparation. Device activations/output are compact row-major FP16 arrays of
shape `[M,K]`/`[M,N]`; scales are FP16 `[K/32,N]`. Integer zeros use
`uint8[K/32,N]`, or a null pointer in symmetric mode.

Allocate `info.workspace_bytes` with256-byte alignment, then fill a version1
`qwa_buffers_v1` with16-byte-aligned data pointers, workspace and stream. Call
`quactlize_ppu_w4a16_prepare_v1` once to obtain the opaque plan. Preparation
validates pointers/aliasing, initializes required state, and packs metadata.
It may allocate host plan state and synchronize; keep it outside graph capture
and operator timing.

Keep data allocations and exclusive workspace alive until the plan's queued
work completes. The metadata layout is specific to the prepared configuration:
preparing another plan into the same workspace invalidates the first plan's
metadata. Input contents may change between calls/replays. Changing bound
addresses requires a new preparation.

`quactlize_ppu_w4a16_run_v1(plan, stream)` launches the prepared operation,
including any required reduction/reset. It does not allocate, JIT, launch a
weight-format conversion kernel or synchronize the host. The TensorCore
main loop dequantizes the packed INT4 operands as it consumes them.
Destroy the plan with
`quactlize_ppu_w4a16_destroy_v1` after its work completes. Buffer ownership
remains with the caller.

Python exposes the same ABI through `quactlize.affine.Library`, `problem`,
`Buffers`, `Config` and `Info`. `Library()` resolves the installed verified
public bundle. Passing a path is reserved for explicit bundle or diagnostic
validation.

## Validation boundary

Correctness, native compilation, selector integration and performance admission
are distinct gates. Generic fallback coverage does not inherit the specialized
anchor timings. Builder manifests do not claim device/performance acceptance.
The complete public wheel and paired hot/256MiB weight-rotation acceptance
remain separate from the retained module experiments.

## Recorded R4 checkpoint

The [result receipt](affine-810e-r4-results.json) freezes the installed V5
public-library measurements on physical ZW810E card1, not a standalone DSO.
The public binary SHA256 is
`f5f31ae330cc83e099473432377217d31f1facb8bbd5ca0c08a894f4338b9a6b`.
Its96 compiled configurations retain all89 R3 configurations and70 C exports.
The complete host suite passed1368 tests; a fresh affine contract check passed17.

The hot/graph matrix contains126 QUactlize-correct shapes, including changed
input replays. Only86 Humming routes match the FP32 accumulation/reduction
contract; the other40 are correctness-only exclusions, not performance wins.
Of the86 comparable points,86 win the median,85 win the full sample range,
and44 pass `Q_max <= 0.9 * H_min`. The M2/N1024/K3072 range spike remains in
the receipt. Thus neither all-range nor all-shape10% admission is established.
R4 rotation and R3 whole-product accuracy were still running when this receipt
was frozen; R4 whole-product accuracy had not run.

The measured N/K geometries are8192/8192,16384/3072,512/3072,3072/8192,
1024/3072,2048/3072 and3072/1024, each at
M=1,2,4,6,8,12,16,24,32,48,64,128,256,512,1024,2048,4096,8192.
Six geometries represent Qwen3.5-122B-A10B projections;8192/8192 is a Humming
synthetic anchor. These do not establish all operators of that model, other
models, grouped expert workloads or end-to-end inference performance.

GGUF scale-first readers can share the code-storage mapping, but their
floating scale/min arithmetic and dispatch are not this integer-zero affine
API. No GGUF scale-first speedup is admitted by this checkpoint.

## GitLab integration boundary

The source port is based on GitLab develop `d8f4b6fd`, retaining its existing
ZW810E dense and grouped gate/up endpoints, policies, package receipts and
dependency gitlinks. GitLab actlize `46a05857` has the same source tree as the
original measured dependency `021c6930`; their commit histories differ.
The classic diagnostic mapping ID remains affine-owned, leaving the measured
legacy Marlin format header unchanged.

The archived R4 measurements identify the original public-library build from
source base `711c96b8` plus its recorded source hashes. They are not timings or
whole-product accuracy admission for a newly built combined GitLab wheel.
Build, correctness and performance gates for that combined package remain
separate; no existing GGUF measurement receipt is rewritten by this port.

The [port verification receipt](affine-810e-r4-integration.json) records1372
host tests passed with zero failures/errors/skips,43 focused contract/package
checks and3 CPU-only native ABI checks. All96 production configurations compile
in99 translation units with zero compiler errors on native `ppu_10`.
The successful rebuild follows a retained diagnostic failure caused by an
unmoved classic mapping-ID reference; no failed build is counted as a pass.
These gates validate compilation and host integration, not a combined public
wheel's device accuracy or performance.
