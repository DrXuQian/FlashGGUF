# Shared warp-K: what to build and what the result means

The ordinary FlashGGUF build, llama wiring and automatic winner tables are
unchanged. To build the opt-in dense FP16 warp-K in the **same public DSO**:

```bash
git submodule update --init third_party/actlize third_party/flashquant
bash build.sh --library --with-warpk --arch ppu_10 --output /path/to/fresh/package
```

On one idle PPU1.0 (ZW810 or ZW810E), run from this branch:

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

The embedded warp-K shape is TM8/TN64/TK256, WN16/WK64, 2 stages, 4 K peers
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
