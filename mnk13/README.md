# MNK13 measurement protocol

From the root of this Git checkout, run `bash run_mnk13.sh --target zw810e --gpu 0 --confirm-idle` on an idle E card. The original card uses `--target zw810`. `--phase anchors` is a short first run; keep the emitted MNK13_RUN directory and add `--out /absolute/path` on resume. Do not reuse an output directory from another device or checkout.

13 identical frozen shapes on either SKU: 2 Q5_K anchors, 7 calibration shapes (including two Q4_K cases), 4 held-out Q5_K shapes. Never train on the holdout CSV. F32 endpoints, F16 operands, FP32 accumulation, rotating packed weights. Numerical evidence is required for both capture directions. Each complete call includes adapter, expansion, main, reducer and output as applicable. We average actual complete calls, not isolated stage minima. JIT, cuBLAS warmup and the first ring are outside the timed set.

The diagnostic dequant library supplies only FP16 Scale-first metadata for this plan; Full-F16 uses the dense provider. Its ISA receipt checks the FP16 SF kernels and records the other disassemblies as unchecked. It cannot certify BF16 for another suite. The device binaries, receipt hashes, independent scale/zero bit checks and whole-call numerical checks remain in place. After updating from the earlier kit, use a new output directory; the old source and results are left intact.

On ZW810 the same-recipe anchor order and historical duration (15% band) are checked before proceeding. On ZW810E those durations are a different device's measurements. Record which arm wins in each direction on the E card; flag disagreements without treating the historical winner as a pass/fail condition. Keep E measurements separate even if CU count, SDK and L2 match a ZW810. A failed numerical check or an incomplete capture is still an error.

The clock-lock receipt records the 1700 MHz request. The profiler samples real clocks around each capture. If the E board denies the request, the run is explicitly labelled REQUEST_DENIED and cannot be combined with a REQUEST_ACCEPTED review. Upload only review.tgz; traces and SQLite remain on the box. No selector, kernel, model or ABI is modified.
