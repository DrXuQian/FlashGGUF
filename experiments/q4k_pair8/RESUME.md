Production is unchanged. No device results yet. See docs/plan.md for the single fixed workload and comparison scope.

On an idle ZW810E with the normal PPU SDK, Python, NumPy, torch (host fixture) and gguf-py:

    CUDA_VISIBLE_DEVICES=0 python3 experiments/q4k_pair8/run.py --asys

The process prints PAIR8_OUTPUT (a fresh directory), then PAIR8_DONE on success. Keep report.json and pair8.asysrep. The report identifies the source, module hashes, actual occupancy/grid, numerical gate, per-arm device-event samples, and the launch order for the warmed Asys range. Only the 16-warp arms are expected to be bit-identical. The selected arm is the shipping geometry, not a historical latency or model-level baseline.

Local checks on PPU SDK 2.1.1-a5c56e: both 512-thread persistent kernels compiled to native objects. Each has 16 static F16 MMA instructions per tiled loop; VREG 148/150 (wide/pair8), SREG 232/232, stack 0/0. Only the candidate has a count-64 wait and nonblocking arrival, bracketing the SWZL B load and packed decode. The compiler schedules some final half2 adds among the first MMAs; no claim of perfect overlap. The CuTe host layout check enumerated all 512 lanes; the independent GGUF host check compared 491520 logical weights and 49152 reference output cells. Incorrectly treating the stored offset-binary code as an uncentered operand fails 49149 output cells. No device speed or numerical admission yet.
