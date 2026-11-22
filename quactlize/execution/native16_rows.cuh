#pragma once
#include "../compat/api.h"
#include <hggc_runtime.h>

namespace quactlize::execution::native16 {
__device__ __forceinline__ int expert(qpm_problem_v1 const &p, qpm_buffers_v1 const &b, int row) {
    return p.mode == QPM_DENSE ? 0 : b.ids[int64_t(row / p.topk) * p.ids_stride + row % p.topk];
}
__device__ __forceinline__ int64_t input_offset(qpm_problem_v1 const &p, int row) {
    return int64_t(row / p.topk) * p.input_token_stride +
           (p.channels == 1 ? 0 : int64_t(row % p.topk) * p.input_row_stride);
}
} // namespace quactlize::execution::native16
