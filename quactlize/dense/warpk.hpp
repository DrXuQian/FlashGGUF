#pragma once
#include <climits>
#include <cstdint>

namespace quactlize::dense::warpk {
inline constexpr int TM=8, TN=64, TK=256, WN=16, WK=64, stages=2;
// A is padded by the bulk load; C is predicated by the driver. M is not
// static, including on the final partial M tile. Keep the donor's int index
// domain and the ordinary (M,N) grid's Y limit; no cross-CTA completion.
inline bool supported(int m,int n,int k) {
    return m>0 && n>0 && k>0 && n%256==0 && k%TK==0 && n/TN<=65535 &&
        int64_t(n)*k<=INT_MAX && int64_t(m)*k<=INT_MAX && int64_t(m)*n<=INT_MAX;
}
#ifdef QUACTLIZE_WITH_WARPK
int initialize();
int launch(int m,int n,int k,void const* a,void const* low,void const* scale,
           void const* offset,void* output,void* stream);
#endif
}
