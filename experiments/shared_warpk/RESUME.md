# Shared warp-K: what to build and what the result means

The ordinary FlashGGUF build, llama wiring and automatic winner tables are
unchanged. To build the opt-in **dense and grouped SF/FQ** warp-K in the same public DSO:

```bash
git submodule update --init third_party/actlize third_party/flashquant
bash build.sh --library --with-warpk --arch ppu_10 --output /path/to/fresh/package
```

## FQ: original units, same warp-K mainloop

From this checkout on an idle PPU1.0:

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/shared_warpk/box.sh --fq
```

For correctness **and** a bounded Asys comparison in the same invocation:

```bash
CUDA_VISIBLE_DEVICES=0 WARPK_BENCH=1 TRACE_Q=14 TRACE_OP=moe-down TRACE_M="1 2 4 8" \
  bash experiments/shared_warpk/box.sh --fq
```

Public dense: candidate **kind 5, route 0**, split=1, dequant_config=-1,
native_f16 matching compute. Public MoE: `quactlize_ppu_moe_matmul_prepare_fq_warpk_v1`.
Prepared native16: `quactlize_ppu_warpk_prepare_fq_v2` with **canonical units**
as metadata, zero=NULL, followed by existing run_v2/destroy_v2. Retains
ordinary adapters and MoE GPU routing, but no global SF planes or expansion
kernel. Per K256 tile, one lane per column decodes only its current superblock
into the CTA's stage (including the 2-byte-aligned second Q3/Q6 share).
FP16/BF16 use production FQ field/sign/rounding decoders. GPTQ remains SF only.
Q8's original FP16 d already needs no prepass, so FQ and SF use **the same**
Q8 device body; do not double count it. No default routing changes.

`WARPK_FQ_PASS` requires 10 host unit/rounding tests and 40 PPU test cases
(public dense/grouped M ranges, GGUF byte-oracle witnesses, alias checks,
changed-ID/offset graph replay and exact workspace difference). The `--fq`
benchmark uses all six orders of auto/SF/FQ; on Q8 only auto vs the shared
body. `routes` names the order of medians and trace rotations. The public
shape here is a screen, not a guarantee about a real model. Never reuse an
SF-only package for this flag. The results are in `review.tgz`; Asys stays local.

Local evidence: PPU1.0 SDK 2.1.1 built and verified public code `1b575ed8`
(DSO SHA-256 `c29470502f307fe039aa9c3141f6bd6a24c64548de998ef37e3af2b1865747d7`).
All ten new packed-unit FQ bodies emit the expected F16/BF16 FP32-accumulator
MMA and native A loads (144–238 VREG, zero stack). Ten host raw-unit/rounding
checks passed; the forty new PPU tests have not run locally. The compiler did
not emit byte-identical existing SF bodies after factoring out the common
mainloop: remeasure auto/SF/FQ in this binary; keep the 70a97881 package
separate if investigating old/new SF performance. No timings admitted.

## All canonical SF formats

On one free PPU1.0 (ZW810 / ZW810E) in this checkout, with the SDK already installed:

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/shared_warpk/box.sh --formats
```

This builds locally, checks the native instructions of every new specialization
when build intermediates are available, then runs the public package. No
separate FlashQuant reference DSO or Python C extension is required for this mode.
It prints `WARPK_OUTPUT`, then `WARPK_FORMATS_PASS` only if correctness passed.
`formats.xml` is the test result; `isa.json` is compiler evidence, not a GPU result.
The last line gives a small `review.tgz` with the XML and JSON only.

| Projection | Candidate API | Compute | GGUF weight formats |
|---|---|---|---|
| Dense F32 A/C | dense_prepare_candidate_v1, kind 4, route 1, split 1 | F16 or BF16 | Q2_K through Q6_K, Q8_0 |
| MoE F32 A/C (broadcast or per-slot) | moe_matmul_prepare_warpk_v1, ordinary run_v1 | BF16 | same six |
| Already prepared native-16 A/C, dense or sorted expert rows | warpk_prepare/run_v2 | F16 or BF16 | same six, plus symmetric/asymmetric GPTQ W4/group32 |

One CTA owns its output tile across **all** K. In the new kernel 16 warps are
four N cohorts times four K peers: tile M16/N64/K256, each peer handles K64,
then the peers reduce in FP32 inside the CTA. A uses the FlashQuant AIU/SWZL
load pair; B uses canonical vectorized LOW/(HIGH)/metadata stripes, with each
format's own register decoder and MMA operand map. No full weight expansion
or cross-CTA reducer. Dense FP16 Q4/GPTQ retains the previous M8 donor.
Public calls include their existing SF metadata expansion (Q8 uses FP16 scale
directly), input adapters and, for MoE, device routing/gather/scatter. The
low-level prepared call needs no global workspace; a complete public call does.
Run with new addresses/stream outside capture, or with unchanged pointers and
changed offsets/activations in replay. Sorted offsets must describe valid rows.
The public MoE API expects valid, distinct IDs **within** each token.

The test uses independent GGUF decoding, varying group scales/minima, K
peers and stage boundaries. Covers M=1..8,9,16,17,128, both input addressing
modes, strided outputs, empty/hot/dispersed experts, changed-ID replay,
N/K=1024/2048 and 2048/512, and both GPTQ zero modes. Does not cover Q5_1,
Q2_0, MXFP4, noncanonical tails, other GPTQ widths, or PPU1.5.

For an optional **bounded** same-card Asys A/B, append `WARPK_BENCH=1` to
the environment before `bash`. Defaults: M=1, six formats × dense, MoE-gate,
MoE-down. Example limited to one case, reusing the package from the first run:

```bash
QUACTLIZE_CI_PACKAGE=/path/from/WARPK_OUTPUT/package CUDA_VISIBLE_DEVICES=0 \
  WARPK_BENCH=1 TRACE_Q=14 TRACE_OP=moe-down TRACE_M=8 bash experiments/shared_warpk/box.sh --formats
```

Each JSON records source/package revisions, DSO hash, format/shape, physical
device, incumbent config and six alternating whole-call event samples per arm.
Same-address and disjoint rotating weights are separate (intervening active LOW
bytes ≥2.25× queried L2). Each Asys report contains a warmed full rotation of
the automatic route, then one of warp-K. MoE screen uses E=8/topk=2 to bound
ring memory: it is not a model-sized router benchmark, a throughput claim,
or sufficient by itself to promote a default.

Local evidence for kernel revision `f7238eac`: SDK 2.1.1 (ppu_10) built
and verified the public package; libncp_quactlize.so SHA-256
`c54b3cc1697c601b4c7dc99750b54598848aa0fe4e4bc79ba3c11cfb152f3c15`.
All 14 new native bodies contain the expected MMA and AIU/SWZL A delivery;
the SDK reports 138–238 VREG and zero stack bytes, not measured occupancy.
246 related host tests passed; the 34 device tests were collected only.
No PPU numeric, latency or table-promotion result is recorded yet.

## Earlier Q4/FlashQuant parity suite

To additionally run the earlier Q4 tests and the independent donor comparison:

```bash
CUDA_VISIBLE_DEVICES=0 WARPK_BENCH=1 bash experiments/shared_warpk/box.sh
```

The script prints its new output directory. The default without `WARPK_BENCH`
is correctness only. The SDK must already be installed; neither the checkout
nor the build needs to download an SDK. A matching `QUACTLIZE_CI_PACKAGE` and
original `FLASHQUANT_REFERENCE` with its manifest can be reused. Numerics
require numpy, gguf, torch and pytest; timing also needs Asys.

- `numerics.xml`: public GGUF candidate vs independent GGUF for M=1 through
  9, 16 and 17 with strided F32 A/C and changed-input graph replay. GPTQ
  symmetric/integer-zero vs original FlashQuant #616/#619 at those M.
  Exercises regular and persistent dense/grouped SF, compact reducers
  (rc2/4/8), AP1 at M=1, typed F16/BF16, and legal Split-K recipes.
- `twins.xml`: seven separately built dynamic-M FlashQuant specializations.
  Only the static M restriction is lifted; their static N/K, arithmetic and
  tiling stay intact. The original M=1 IDs and the exact selector stay put.
  AP1/row-pruned kernels that actually read only the first row are not relaxed.
  `CHECK_TWINS=0` skips **only** this slower group; do not count it as passed.
- `paired.json`: dense GGUF compares the whole F32 public operation, including
  SF expansion and adapters, to unchanged automatic FlashGGUF dispatch. GPTQ
  compares FP16 prepared calls vs FlashQuant #619, not vs all historical configs.
  Benchmarks N=25600,K=5120 at M=1,4,8,9, alternating arms, separately with
  repeated and rotating weights (>2.25x L2 intervening LOW bytes). Records raw
  device-event samples, medians, binary hashes and commit. Each `.asysrep`
  contains one primed rotation of the control followed by one of the candidate
  at `TRACE_M` (default 1). JIT, packing and profiler overhead are not counted
  in the event times. These are screens, not production timing receipts.

## Operand contract

All existing **W4/group32 SF** JIT parents can also be built in GPTQ style:

```python
from quactlize.runtime.sf_gptq import pack, parent
from quactlize.runtime.tuning import Request

planes = pack(codes, scales, zeros)  # or zeros=None for symmetric zp=8
record = compiler.build(parent(existing_q4_sf_geometry))
req = Request("sf-dense", 12, n, k, m, weight_style="gptq")
```

Codes are resolved logical uint8 `[N,K]` or `[E,N,K]` in 0..15; scales are
FP16 `[...,K/32,N]`, actual zero points have the same shape and are uint8.
K and N must be positive multiples of 256. For the explicit grouped BF16
metadata ABI use `metadata="bf16"`. Use the returned LOW, metadata, zero and
mapping_id together. The packed LOW bytes agree with the FlashQuant packer.
The upstream caller must resolve any checkpoint qzeros offset or g_idx
permutation; this is not an AutoGPTQ checkpoint loader. Old measured GGUF
records and identically shaped GPTQ requests are never interchangeable.

The original dense FP16 Q4 embedded warp-K shape is TM8/TN64/TK256, WN16/WK64, 2 stages, 4 K peers
per output warp cohort; reduction stays within the CTA. For canonical GGUF
use explicit dense candidate kind 4 (SF, native F16, split 1, dequant_config
-1). For already prepared FP16 planes call `quactlize_ppu_warpk_prepare_v1`
with an address-free shape (set device, actual CU count, mapping_id), then
`run_v1` with the same shape and current A/planes/C/stream; no weight
conversion inside this prepared run. See the device test for exact signatures.
Neither path claims a measured win on an unseen M or support for GPTQ W2/3/8,
group128, or warp-K on PPU1.5.

## Host evidence

Base FlashGGUF a1b14166, FlashQuant reference 0a4c5b08, shared donor
0d1a39ea, actlize 021c6930. PPU SDK 2.1.1 compiled and linked the public
library and both prepared-operand warp-K specializations. The pre-existing
FlashQuant device ISA bytes (not entire ELF metadata) stayed identical for
535, 625, 627, 756, 829, 1208, 1325, 616 and 619 versus the reference.
This is not a device numerics or timing result. No default should change until
box results are available. The experiment scope is in [the plan](docs/plan.md).
