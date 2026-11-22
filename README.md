# FlashGGUF Bench

Standalone benchmark branch; do not merge it into production. It neither installs FlashGGUF nor changes its selection table.

## ZW810E / ZW810 · 13 dense prefill shapes

Use an idle PPU with HGCC, Asys, ppu-smi, patch and the PPU Python environment (NumPy, torch, gguf). The frozen product, profiling harness and actlize sources are in this branch. No GGUF model, llama.cpp checkout, nested submodule or extra network fetch is needed.

```bash
git clone --depth 1 --branch bench/zw810e-mnk13 https://github.com/DrXuQian/FlashGGUF.git FlashGGUF-mnk13
cd FlashGGUF-mnk13
bash run_mnk13.sh --target zw810e --gpu 0 --confirm-idle --out /workspace/mnk13-zw810e-gpu0
```

Switch `--gpu` and the output path if card 0 is busy. For an original ZW810, use `--target zw810` and a different output directory. `--plan-only` checks the frozen inventory with no GPU. `--sdk PATH` overrides `PPU_SDK` (default `/usr/local/PPU_SDK`); `PYTHON` may name an alternate PPU Python executable. For two anchors only, add `--phase anchors`. Re-run with the same card, SDK, checkout and output directory to resume.

The default is two anchors, seven calibration shapes and four blind shapes: Q5_K Scale-first vs Full-F16, plus Q4_K FQ vs Scale-first. Three configs per shape, two Asys capture directions, F32 endpoints, F16 operands and FP32 accumulation. The E card reuses the **candidate recipes**, not the old timings or the 810's 72 CUs. Its real CU count and L2 capacity are probed; if the SDK and torch both omit L2, supply an independently verified `--l2-bytes BYTES`. The script requests 1700 MHz and records actual clocks. A denied clock request on E is reported as such and kept separate from locked runs. The two historical anchors are remeasured on E; its old 810 absolute microseconds never block a run. Keep the four blind points out of fitting.

Return only **`/workspace/mnk13-zw810e-gpu0/review.tgz`**. Raw Asys traces stay on the box. Passing is evidence for the listed operators on that card only, not a speed claim for an entire model.

The actual [runner](mnk13/run.py), [frozen cases](mnk13/plan.json) and [measurement protocol](mnk13/README.md) are readable in Git. Only the pinned dependency sources are compressed. The wrapper verifies and unpacks them on first use; the runner checks file identities before measurement. To run the host-side tests after unpacking:

```bash
python3 -B tests/check_mnk13.py mnk13 -v
```
