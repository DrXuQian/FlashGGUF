# FlashGGUF MNK13 · ZW810

On an idle ZW810 with the usual PPU-enabled Python (numpy, torch, gguf), HGCC,
g++, ppu-smi and Asys. No checkout, GGUF model, llama.cpp, submodule update or
network is needed. Extract the archive in a fresh directory; leave it in place
for resume. Start from its flashgguf-mnk13 directory:

```bash
python3 run.py --plan-only
python3 -u run.py --gpu 0 --confirm-idle
```

The default runs two historical anchors, seven calibration shapes, then four
held-out shapes. Pass `--phase anchors` for just the bridge. To continue or
resume use **the same** `--gpu` and add `--out /absolute/path/from/MNK13_RUN`.
Set `--sdk PATH` only if PPU_SDK/the default is wrong; `--jobs 4` is the
per-build concurrency. The runner requests 1700 MHz for the selected GPU;
observed clocks are diagnostic, throttling is not an automatic failure.

F32 endpoints, F16 operands, FP32 accumulation, rotating packed weights.
Precision and separate adapter, expansion, main, reducer and output activities
are checked inside actual calls. JIT, cuBLAS warmup and first ring are not
timed. Forward and reverse Asys captures per point; CSV mean_us weights all
complete calls, not a sum of stage minima. Two old per-route control pairs
bridge historical results. By default an ordering change between the two
directions or >15% absolute drift of a historical control pauses the all run;
inspect anchor-check.json before optionally resuming with --allow-anchor-drift.
This is a diagnostic, not a product performance threshold.

Upload only the printed review.tgz. It contains CSVs, identity, numeric/timing
receipts, clock diagnostics and error context; raw traces/SQLite/cache remain
on the box. Holdout data live in a separate CSV: do not fit using those rows.
A shipping dense-table miss does not assert that the product has no fallback;
no unsupported comparator or historical measurement is assigned to that shape.
This is not a llama.cpp/model/decode/MoE benchmark or a change to any selector.
