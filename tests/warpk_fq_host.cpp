#include <hggc_runtime.h>
#include "warpk_units.hpp"
#include "gguf_packed_unit.hpp"
using namespace quactlize::dense::warpk;

template<int Q,class T> static void decode(uint8_t const* data,int n,int nsb,int experts,
                                           uint16_t* scale,uint16_t* zero) {
    using F=PackedUnit<Q>;
    constexpr auto qt=gguf_scale::KType(Q-10);
    static_assert(F::correction==gguf_scale::packed_unit::kCanonicalPlacedZMul<qt>);
    static_assert(F::bytes==gguf_scale::packed_unit::Unit<qt>::kSbBytes);
    for(int i=0;i<nsb*experts;++i) for(int col=0;col<n;++col) {
        auto base=i*F::groups*n+col;
        F::template publish<T>(data+F::offset(i,n,col),scale+base,zero+base,n);
    }
}

extern "C" void decode_units(int q,bool bf,uint8_t const* data,int n,int nsb,int experts,
                             uint16_t* scale,uint16_t* zero) {
#define FORMAT(Q) case Q: return bf?decode<Q,cutlass::bfloat16_t>(data,n,nsb,experts,scale,zero): \
                                   decode<Q,cutlass::half_t>(data,n,nsb,experts,scale,zero);
    switch(q) {FORMAT(10) FORMAT(11) FORMAT(12) FORMAT(13) FORMAT(14)}
#undef FORMAT
}
