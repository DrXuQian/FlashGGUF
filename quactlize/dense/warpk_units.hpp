#pragma once
#include "warpk_layout.hpp"
#include "actlize_extensions/cutlass/gguf_bfloat_scale.h"
#include "cute/tensor.hpp"

namespace quactlize::dense::warpk {
// Transport only. The field map, signedness and F16/BF16 rounding are the
// same group_of_words decoders used by the ordinary FQ-TC collective.
template<int Q> struct PackedUnit {
    static_assert(Q>=10 && Q<=14);
    static constexpr auto fmt=cutlass::gguf_packed::Fmt(ppu_formats::for_qtype(Q).packed_format);
    using U=cutlass::gguf_packed::Unit<fmt>;
    static constexpr int groups=U::kGroups, bytes=U::kSbBytes;
    static_assert(groups*Planes<Q>::group==256 && bytes%2==0);
    // Placed code (LOW4 is centred at 8) minus the GGUF code's own centre.
    static constexpr int correction=(Planes<Q>::low==4?8:0)-(Q==11?4:Q==14?32:0);

    CUTLASS_HOST_DEVICE static constexpr int64_t offset(int64_t sb,int n,int col) {
        return (sb/U::kSbPerUnit*n+col)*U::kUnitTotal+(sb%U::kSbPerUnit)*bytes;
    }
    template<class T> CUTLASS_HOST_DEVICE static void publish(uint8_t const* p,uint16_t* scales,
            uint16_t* zeros,int stride) {
        using namespace cutlass::gguf_packed;
        uint32_t words[(bytes+3)/4]{};
        // Q3/Q6's second superblock begins at byte 14/18. Read exactly
        // one superblock even for the very last column, never a padded word.
        if constexpr(bytes%4==0) {
            __builtin_memcpy(words,p,bytes);
        } else {
            CUTLASS_PRAGMA_UNROLL
            for(int i=0;i<bytes/2;++i) {
                uint16_t pair;__builtin_memcpy(&pair,p+2*i,2);
                words[i/2]|=uint32_t(pair)<<(16*(i%2));
            }
        }
        cute::for_each(cute::make_int_sequence<groups>{},[&](auto g) {
            constexpr int G=decltype(g)::value;
            auto sz=[&] {
                if constexpr(std::is_same_v<T,cutlass::bfloat16_t>)
                    return bfloat_group_of_words<G,correction,fmt>(words,bfloat_head_of_words(words));
                else {
                    static_assert(std::is_same_v<T,cutlass::half_t>);
                    return group_of_words<G,U::kScaleBias,U::kHasMin,correction,fmt>(words,head_of_words(words));
                }
            }();
            scales[G*stride]=sz.scale.raw();
            zeros[G*stride]=sz.zero.raw();
        });
    }
};
} // namespace quactlize::dense::warpk
