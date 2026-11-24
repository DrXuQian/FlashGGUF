#include "warpk_layout.hpp"
#include <array>
#include <algorithm>
using namespace quactlize::dense::warpk;

// Copy whole physical word stripes just like the kernel; compare at the logical
// end of the pipeline in Python. A raw-code scalar reader over the full matrix
// would miss bugs at stage/tile boundaries and Q5's folded high plane.
template<int Q> static void decode(uint16_t const* low,uint16_t const* high,int n,int k,int16_t* out) {
    using P=Planes<Q>; constexpr int nl=block_k*P::low/16,nh=block_k*P::high/16;
    std::array<uint16_t,nl*cols_tile> l; std::array<uint16_t,nh*cols_tile> h;
    for(int base=0;base<k;base+=block_k) for(int col=0;col<n;col+=cols_tile) {
        for(int i=0;i<nl;++i) std::copy_n(low+(base*P::low/16+i)*n+col,cols_tile,l.data()+i*cols_tile);
        if constexpr(nh) for(int i=0;i<nh;++i)
            std::copy_n(high+(base*P::high/16+i)*n+col,cols_tile,h.data()+i*cols_tile);
        for(int peer=0;peer<k_peers;++peer) for(int i=0;i<peer_k;++i) for(int j=0;j<cols_tile;++j) {
            int kk=peer*peer_k+i;
            out[(col+j)*k+base+kk]=P::code(l.data(),h.data(),j,kk,cols_tile);
        }
    }
}
extern "C" int codes(int q,uint16_t const* l,uint16_t const* h,int n,int k,int16_t* o) {
    switch(q) {
#define CASE(Q) case Q: decode<Q>(l,h,n,k,o);return 0;
        CASE(8) CASE(10) CASE(11) CASE(12) CASE(13) CASE(14)
#undef CASE
    default:return 1;
    }
}
extern "C" int schedule(int rows,int experts,int const* offsets,int* owners,int* visits) {
    int num=tasks(rows,experts,offsets!=nullptr);
    for(int t=0;t<num;++t) {
        RowTile tile=row_tile(t,rows,experts,offsets);
        if(tile.expert<0) continue;
        for(int r=tile.first;r<std::min(tile.first+rows_tile,tile.end);++r) {
            if(r<0 || r>=rows) return -1;
            owners[r]=tile.expert;++visits[r];
        }
    }
    return num;
}
