# Q4_K SF: W0/W8 … W7/W15 (ZW810E)

Parent a1b14166f3396cc6ced78de3a91a93659b2044dd, actlize 021c69300864e03926108867d318572871a44d35. One dense point: M512 N25600 K5120. Keep canonical K-pack4, expanded SF half scales/zeros, half MMA inputs, FP32 accumulators, and the same output conversion. Only the prepared main kernel is timed, not the public F32 call or the model.

Controls: the selected 64x64x64 / 64x32 persistent tile; a plain 64x256x64 / 64x16 persistent tile; the latter plus a per-pair prologue rendezvous. All use three stages, DN64, AP0 and split 1. The new tile has 16 N warps, no M split. Keep the plain and paired grids identical at their common supported resident capacity. Do not infer a WE ID from a software warp ID.

At TK64 there is only one register delivery per shared stage. All warps still do their own prepare and MMA. After CTA publication, low warp i finishes its own prepare and signals high warp i+8, which can now prepare while low does its MMA. There is no cross-CTA synchronization, no cross-warp register exchange, and no promise of overlap across the next CTA-wide stage barrier.

Gate: independent GGUF numeric sample; all outputs written/finite; bitwise identical for the two 16-warp arms. Inspect the persistent device body (not the ordinary kernel) and check the actual named count-64 wait/release around preparation. Alternate warmed measurements on one idle 810E. Compare plain vs paired (scheduling) and paired vs production geometry (overall). Only consider promotion after reproducible reduction in total kernel cycles and then full-call verification. No route changes. Stop this pass after the one-shape host/build check and Box handoff; no wider sweep.
