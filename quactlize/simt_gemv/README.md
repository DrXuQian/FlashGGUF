# SIMT GEMV

This is the source directory for CUDA-core weight-only GEMV, not the
selector for all small M. A decode shape may still choose a Tensor Core.

For automatic K-pack GEMV, start at [the small-M selector](../dispatch/smallm_matched.hpp),
then [the SIMT launcher](simt_codegen.py) or the [Q4 S1 reader](q4_decode_kernel.cuh).
The token-CTA device body is kept byte-identical to its measurement (including
its candidate-era comment); [the exact-key bindings](token_cta.inc) determine
where it runs today. A neighboring bucket does not enable that specialization.

| Path | Input / role |
| --- | --- |
| `gemv.cu`, `simt_*`, `q4_*` | K-pack dense and indexed GEMV: general, measured and token-CTA readers; includes Q4 decode specializations. |
| `raw/` | Raw GGUF and block-contiguous vecdot readers; not K-pack buffers. |
| `native16.cuh` | Native FP16/BF16 packed and indexed compatibility GEMV; the Tensor Core main loop stays in execution. |
| `lowbit/` | Legacy packed low/high planes with separately materialized scales; shared by the raw-pointer backend. |
| `fused.cu` | Paired gate/up SIMT body; FFN matching and shared fusion helpers remain in `../fusion/`. |
| `ppu17/` | SIMT device bodies for the separate PPU1.7 backend. Its launcher and WGMMA kernels stay in the backend. |

The execution call ABI, common reader and size validation are shared with
other operations in `../execution/`. The public include names and C symbols
are unchanged. Old measurement receipts keep their original paths and hashes:
the verifier maps those paths to the moved files and still checks their bytes.
