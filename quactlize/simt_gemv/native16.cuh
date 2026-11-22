#pragma once
#include "../execution/native16_geometry.hpp"
#include "../execution/native16_rows.cuh"

namespace quactlize::execution::native16 {
using compat::Weight;
// N32/CTA, four adjacent columns per lane. Each group of eight lanes loads
// contiguous 64B code / 128B affine packets; A is shared with width-8 exchanges.
// K/group is a multiple of four, so each warp's K tail is uniformly inactive.
template <class T> __global__ void packed_simt(qpm_problem_v1 p, qpm_buffers_v1 b, Weight w) {
    unsigned tid = threadIdx.x, lane = tid % 32, worker = tid / 8;
    int tiles = (uint64_t(p.n) + 31) / 32;
    int row = blockIdx.x / tiles, first = (blockIdx.x % tiles) * 32;
    int e = expert(p, b, row);
    if (e < 0 || e >= p.experts) {
        if (tid == 0 && e != -1)
            atomicOr(b.device_error, QPM_BAD_ID);
        if (tid < 32 && first + int(tid) < p.n)
            static_cast<T *>(b.output)[int64_t(row) * p.output_stride + first + tid] = T(0.f);
        return;
    }
    int col = first + (tid % 8) * 4;
    auto low = reinterpret_cast<uint16_t const *>(w.low + uint64_t(e) * w.low_stride);
    auto high = reinterpret_cast<uint16_t const *>(w.high + uint64_t(e) * w.high_stride);
    auto units = reinterpret_cast<uint32_t const *>(w.units + uint64_t(e) * w.units_stride);
    auto a = static_cast<T const *>(b.input) + input_offset(p, row);
    float sum[4]{};
    for (unsigned g = worker; g < unsigned(p.k / 32); g += 16) {
        uint4 dm = col < p.n ? *reinterpret_cast<uint4 const *>(units + uint64_t(g) * p.n + col)
                             : make_uint4(0, 0, 0, 0);
        uint32_t metadata[4] = {dm.x, dm.y, dm.z, dm.w};
        float activation[4];
#pragma unroll
        for (int v = 0; v < 4; ++v)
            activation[v] = float(a[g * 32 + (lane % 8) * 4 + v]);
#pragma unroll
        for (int residue = 0; residue < 8; ++residue) {
            uint2 lo = col < p.n
                           ? *reinterpret_cast<uint2 const *>(low + (uint64_t(g) * 8 + residue) * p.n + col)
                           : make_uint2(0, 0);
            uint2 hi =
                col < p.n
                    ? *reinterpret_cast<uint2 const *>(high + (uint64_t(g / 4) * 8 + residue) * p.n + col)
                    : make_uint2(0, 0);
            uint16_t lows[4] = {uint16_t(lo.x), uint16_t(lo.x >> 16), uint16_t(lo.y), uint16_t(lo.y >> 16)};
            uint16_t highs[4] = {uint16_t(hi.x), uint16_t(hi.x >> 16), uint16_t(hi.y), uint16_t(hi.y >> 16)};
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                int offset = slot * 8 + residue;
                float av = __shfl_sync(0xffffffffu, activation[offset % 4], (lane & ~7u) + offset / 4);
#pragma unroll
                for (int v = 0; v < 4; ++v)
                    sum[v] = fmaf(
                        av, float(q5_weight<T>(lows[v], highs[v], slot, (g % 4) * 4 + slot, metadata[v])),
                        sum[v]);
            }
        }
    }
#pragma unroll
    for (int delta = 8; delta <= 16; delta *= 2) {
#pragma unroll
        for (int v = 0; v < 4; ++v)
            sum[v] += __shfl_xor_sync(0xffffffffu, sum[v], delta);
    }
    __shared__ float partial[4][32];
    if (lane < 8) {
#pragma unroll
        for (int v = 0; v < 4; ++v)
            partial[tid / 32][lane * 4 + v] = sum[v];
    }
    __syncthreads();
    if (tid < 32 && first + int(tid) < p.n) {
        float value = ((partial[0][tid] + partial[1][tid]) + partial[2][tid]) + partial[3][tid];
        static_cast<T *>(b.output)[int64_t(row) * p.output_stride + first + tid] = T(value);
    }
}
} // namespace quactlize::execution::native16

namespace quactlize::compat {
// Defined in compat/runtime.cu; the grouped row helpers use the same function.
__device__ int expert(qpm_problem_v1 const &, qpm_buffers_v1 const &, int);
template <int Q, class T, bool Packed>
__global__ void indexed_simt(qpm_problem_v1 p, qpm_buffers_v1 b, Weight w, int rows) {
    int lane = threadIdx.x % 32;
    for (int64_t cell = int64_t(blockIdx.x) * 4 + threadIdx.x / 32; cell < int64_t(rows) * p.n;
         cell += int64_t(gridDim.x) * 4) {
        int row = int(cell / p.n), n = int(cell % p.n), e = expert(p, b, row);
        float sum = 0.f;
        if (e >= 0 && e < p.experts) {
            auto a = static_cast<T const *>(b.input) + int64_t(row / p.topk) * p.input_token_stride +
                     (p.channels == 1 ? 0 : int64_t(row % p.topk) * p.input_row_stride);
            for (int k = lane; k < p.k; k += 32) {
                T w16(value<Q, Packed>(w, e, n, k));
                sum = fmaf(float(a[k]), float(w16), sum);
            }
        } else if (e != -1 && lane == 0)
            atomicOr(b.device_error, QPM_BAD_ID);
        for (int delta = 16; delta; delta /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, delta);
        if (lane == 0)
            static_cast<T *>(b.output)[int64_t(row) * p.output_stride + n] = T(sum);
    }
}
} // namespace quactlize::compat
