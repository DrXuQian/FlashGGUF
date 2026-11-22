#pragma once
#include "../../backends/ppu17/io.cuh"
#include "../../backends/ppu17/launch.cuh"

namespace quactlize::ppu17 {

template <int Q, class T, bool SF, int LanesN, bool Pdl = false>
__global__ void simt(q17_call_v1 c) {
  constexpr int lanes_k = 32 / LanesN, columns = 4 * LanesN;
  int expert = blockIdx.z;
  wait_for_predecessors<Pdl>();
  auto rows = row_range(c, expert);
  int row = blockIdx.y;
  if (row >= rows.size) return;
  int lane_n = threadIdx.x % LanesN, lane_k = (threadIdx.x % 32) / LanesN;
  int col = int(blockIdx.x) * columns + int(threadIdx.x / 32) * LanesN + lane_n;
  WeightReader<Q, T, SF> weight(c, expert);
  float sum = 0;
  if (col < c.problem.n) {
    for (int k = lane_k; k < c.problem.k; k += lanes_k) {
      T a(load_element(c.a, c.problem.input_type, int64_t(rows.begin + row) * c.a_stride + k));
      sum = fmaf(float(a), float(weight(col, k)), sum);
    }
  }
  // All lanes participate, including N-tail lanes. Only matching N columns
  // communicate. Accumulation and the warp reduction both remain FP32.
  release_successors<Pdl>();
  for (int delta = LanesN; delta < 32; delta *= 2)
    sum += __shfl_xor_sync(0xffffffffu, sum, delta);
  if (lane_k == 0 && col < c.problem.n)
    store_element(c.output, c.problem.output_type,
                  int64_t(rows.begin + row) * c.output_stride + col, sum);
}

} // namespace quactlize::ppu17
