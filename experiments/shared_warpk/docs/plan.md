# Shared group32 INT4 warp-K (PPU1.0)

Base: FlashGGUF a1b14166; independent affine reference: FlashQuant 0a4c5b08.
Reuse the actual AIU/ldmatrix collective, intra-CTA K reduction and store. No
global Split-K, Stream-K, default-table edit, alternate packed codes or llama change.
GGUF keeps its SF expansion and F32 adapters; reconstruct half(half((q-8)*S)+Z).
GPTQ retains FlashQuant's symmetric/uint4-zero modes and FP16 reference.
Use its pinned source, not a second linked DSO. All W4/g32 SF module families
(native/typed dense, grouped ordinary/compact/persistent, and their legal
Split-K recipes) use the same prepared-center semantics. Original FlashQuant
specializations and its measured selector remain available unchanged.

Priority: dense Q4_K, M=1..8, plus 9/16/17 (new tile, full tile, tail),
N/K=256/512 (GGUF), 256/1024 (GPTQ), and 25600/5120 (timing). Exercise nontrivial minima and groups, noncompact
F32 strides and numerical comparisons against independently decoded GGUF bytes.
For GPTQ compare the embedded API to the original DSO: match the tile, shape,
low-plane bytes and decoded weights. Preparation, scale/zero transport and store
specializations can differ. Check one-hot output bits and independent random
dots, then compare timed calls on the same card.
Generate separate dynamic twins only for unpruned, whole-K static-M1 candidates;
do not loosen assertions or relabel unmeasured M as a table hit.

First gate: compile actual ppu_10 device kernels, compare the unmodified GPTQ
specializations' native instructions to reference and run host tests. Then device
correctness and a bounded same-card paired whole-call A/B, hot and rotating cold
weights (>2.25x L2), baseline is the unchanged automatic FlashGGUF winner.
Stop on a numerical failure; do not claim a performance win from compile checks.
No time deadline specified; hand off when physical PPU execution is required.
