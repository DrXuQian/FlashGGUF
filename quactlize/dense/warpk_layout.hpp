#pragma once
#include "kquant_kpack_offline.hpp"
#include "ppu_format_config.hpp"
#include "quactlize_ppu_config.h"
#include <cstdint>

namespace quactlize::dense::warpk {
// One CTA owns this entire output tile. Each K peer visits a disjoint
// contiguous 64-element stripe of every K256 block until the dot is complete.
inline constexpr int rows_tile=16, cols_tile=64, block_k=256, peer_k=64;
inline constexpr int n_warps=cols_tile/16, k_peers=block_k/peer_k;
constexpr int group_size(int q) {return q==8 ? 32 : ppu_formats::for_qtype(q).group_size;}
constexpr int low_bits(int q) {return q==8 ? 8 : ppu_formats::for_qtype(q).low_bits;}
constexpr int high_bits(int q) {return q==8 ? 0 : ppu_formats::for_qtype(q).high_bits;}
constexpr bool is_format(int q) {return q==8 || (q>=10 && q<=14);}
constexpr uint64_t gguf_map(int q) {
    return q==8 ? QUACTLIZE_PPU_Q8_KPACK2_MAPPING_ID : q==12 ?
        QUACTLIZE_PPU_Q4_KPACK4_MAPPING_ID : kquant_kpack::kMappingId;
}

template<int Q> struct Planes {
    static_assert(is_format(Q));
    static constexpr int low=low_bits(Q), high=high_bits(Q), group=group_size(Q);
    using L=kquant_kpack::PlaneMap<low,group>;
    // Never instantiate PlaneMap<0,...>.
    using H=kquant_kpack::HighPlaneMap<low,high ? high : 1,group>;
    static constexpr int code(uint16_t const* lo,uint16_t const* hi,int col,int k,int n) {
        int v=(lo[L::word_index(col,k,n)]>>(low*L::word_slot(k)))&((1<<low)-1);
        if constexpr (high) {
            int slot=0;
            if constexpr (Q==13) slot=H::word_slot(col,k); // Q5 high: physical N8/K128 bit exchange
            else slot=H::word_slot(k);
            v|=((hi[H::word_index(col,k,n)]>>(high*slot))&((1<<high)-1))<<low;
        }
        // Match the existing SF converter, whose zero already includes its code bias.
        return v-(Q==8 ? 128 : low==4 ? 8 : 0);
    }
};

struct RowTile {int expert,first,end;};
constexpr int tasks(int rows,int experts,bool grouped) {
    return !grouped ? 1+(rows-1)/rows_tile :
        (rows<=rows/rows_tile+experts ? rows : rows/rows_tile+experts);
}
// All CTAs in one grid use the current sorted-row offsets, never a host copy.
// For tiny loads probe each row; otherwise each expert owns a virtual range
// starting at floor(offset[e]/16)+e. At most one unused tile per expert.
constexpr RowTile row_tile(int task,int rows,int experts,int const* offsets) {
    if(!offsets) return {0,task*rows_tile,rows};
    bool by_row=rows<=rows/rows_tile+experts;
    int lo=0,hi=experts;
    while(lo+1<hi) {
        int mid=lo+(hi-lo)/2,off=offsets[mid];
        if ((by_row ? int64_t(off) : int64_t(off)/rows_tile+mid)<=task) lo=mid; else hi=mid;
    }
    int begin=offsets[lo],end=offsets[lo+1];
    if(begin<0 || end<begin || end>rows) return {-1,0,0};
    int64_t first=by_row ? task : begin+(int64_t(task)-begin/rows_tile-lo)*rows_tile;
    return first<begin || first>=end || (first-begin)%rows_tile ? RowTile{-1,0,0} : RowTile{lo,int(first),end};
}
} // namespace quactlize::dense::warpk
