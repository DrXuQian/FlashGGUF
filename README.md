# FlashGGUF Bench

Independent box benchmarks; this repository neither installs FlashGGUF nor changes its production selection table.

## ZW810 · 13 dense prefill shapes

Use an idle ZW810 (72 CUs, 64 MiB L2), with HGCC, Asys, ppu-smi and the PPU Python environment (NumPy, torch, gguf). The frozen product, profiling harness and actlize sources are in the clone. No GGUF model, llama.cpp checkout, nested submodule or extra network fetch is needed.

```bash
bash run_mnk13.sh --plan-only
bash run_mnk13.sh --gpu 0 --confirm-idle --out /workspace/mnk13-gpu0
```

Switch `--gpu` and the output path if card 0 is busy. `--sdk PATH` overrides `PPU_SDK` (default `/usr/local/PPU_SDK`); `PYTHON` may name an alternate PPU Python executable. For two anchors only, add `--phase anchors`. Re-run the original command with the same card, SDK, checkout and output directory to resume.

The default is two historical controls, seven calibration shapes and four blind shapes: Q5_K Scale-first vs Full-F16, plus Q4_K FQ vs Scale-first. Three configs per shape, two Asys capture directions, F32 endpoints, F16 operands and FP32 accumulation. The script requests 1700 MHz on the selected card. It first checks the old winners on the two anchors; if the bridge changes, inspect `anchor-check.json` before running more points. Keep the four blind points out of fitting.

Return only **`/workspace/mnk13-gpu0/review.tgz`**. Raw Asys traces stay on the box. Passing is evidence for the listed operators only, not a speed claim for an entire model.

The actual [runner](mnk13/run.py), [frozen cases](mnk13/plan.json) and [measurement protocol](mnk13/README.md) are readable in Git. Only the pinned dependency sources are compressed. The wrapper verifies and unpacks them on first use; the runner checks file identities before measurement. To verify the five host-side packaging/plan tests after unpacking:

```bash
python3 -B tests/check_mnk13.py mnk13 -v
```
