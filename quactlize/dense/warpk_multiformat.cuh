#pragma once
#include "warpk_layout.hpp"
#include "actlize_extensions/cutlass/gemm/kernel/kpack_warpk_ppu.hpp"

// Native PPU1.0 Tensor Core mainloop. Keep the donor's paired AIU/SWZL A
// delivery and FP32 K-peer reducer. B is a *plain* shared tile of canonical
// packed words: never feed it to the opaque swizzled Q4 matrix load.
namespace quactlize::dense::warpk::multi {
namespace cc=cutlass::gemm::collective;
namespace load=cc::marlin_ppu_detail;
using Delivery=cc::NativeInt4WarpKCollectivePPU<rows_tile,cols_tile,block_k,16,peer_k,2,true,false,true,true>;
using Reduce=cutlass::gemm::kernel::KPackWarpKKernelPPU<Delivery,true>;
static_assert(Delivery::WarpN==n_warps && Delivery::WarpK==k_peers && Delivery::InstructionM==16);
static_assert(Delivery::MAtoms==1 && Delivery::NAtoms==1 && Reduce::SliceValues==8);
constexpr int threads=Delivery::Threads, tm=rows_tile,tn=cols_tile,tk=block_k;

template<int Q> struct alignas(128) Stage {
    using F=Planes<Q>;
    alignas(128) uint16_t a[tk/64][tm*64]; // four native, opaque M16/K64 cubes
    alignas(16) uint16_t low[tk*tn*F::low/16], high[F::high ? tk*tn*F::high/16 : 1];
    alignas(16) uint16_t scales[tk/F::group*tn], zeros[Q==8 ? 1 : tk/F::group*tn];
};
template<int Q> union Storage { Stage<Q> pipe[2]; typename Reduce::SharedStorage peers; };

template<class Layout> constexpr bool adjacent_k_pairs() {
    for(int lane=0;lane<32;++lane)
        for(int v=0;v<8;v+=2)
            if(Layout{}(lane,v+1)!=Layout{}(lane,v)+16) return false;
    return true;
}

template<int Words>
CUTLASS_DEVICE void copy_stripe(uint16_t* dst,uint16_t const* src,int stride) {
    static_assert(Words%tn==0 && tn%8==0);
    for(int v=threadIdx.x;v<Words/8;v+=threads) {
        int row=v/(tn/8), col=v%(tn/8)*8;
        load::cp_async_16(dst+v*8,src+int64_t(row)*stride+col);
    }
}

template<int Q>
CUTLASS_DEVICE void stage(Stage<Q>& dst,qk_call_v1 const& c,RowTile tile,int col,int start) {
    using F=Planes<Q>;
    int warp=threadIdx.x/32;
    if(warp<tk/64)
        Delivery::copy_native_a(dst.a[warp],static_cast<uint16_t const*>(c.a)+int64_t(tile.first)*c.k,
            tile.end-tile.first,c.k,0,start+warp*64);
    int64_t kbase=int64_t(tile.expert)*c.k+start;
    copy_stripe<tk*tn*F::low/16>(dst.low,static_cast<uint16_t const*>(c.low)+
            (kbase/(16/F::low))*c.n+col,c.n);
    if constexpr(F::high)
        copy_stripe<tk*tn*F::high/16>(dst.high,static_cast<uint16_t const*>(c.high)+
            (kbase/(16/F::high))*c.n+col,c.n);
    auto meta=kbase/F::group*c.n+col;
    copy_stripe<tk/F::group*tn>(dst.scales,static_cast<uint16_t const*>(c.metadata)+meta,c.n);
    if constexpr(Q!=8)
        copy_stripe<tk/F::group*tn>(dst.zeros,static_cast<uint16_t const*>(c.zero)+meta,c.n);
}

template<int Q,class T,bool GPTQ>
CUTLASS_DEVICE T weight(Stage<Q> const& s,int n,int k) {
    using F=Planes<Q>;
    auto idx=k/F::group*tn+n;
    auto code=F::code(s.low,s.high,n,k,tn);
    // Unlike the K-quants, Q8's scales always carry FP16 bits, even with
    // BF16 activations. K-quant metadata matches the 16-bit compute type.
    using Meta=std::conditional_t<Q==8,cutlass::half_t,T>;
    float scale=float(Meta::bitcast(s.scales[idx]));
    if constexpr(Q==8) return T(__fmul_rn(float(code),scale));
    else {
        float offset=float(T::bitcast(s.zeros[idx]));
        if constexpr(GPTQ) return T(__fmul_rn(float(code)+offset,scale)); // center is exactly 8-zp
        return T(__fadd_rn(float(T(__fmul_rn(float(code),scale))),offset));
    }
}

template<int Q,class T,bool GPTQ>
__global__ __launch_bounds__(threads) void kernel(qk_call_v1 c) {
    extern __shared__ __align__(128) uint8_t memory[];
    auto& mem=*reinterpret_cast<Storage<Q>*>(memory);
    RowTile tile=row_tile(blockIdx.x,c.m,c.experts,c.offsets_device);
    if(tile.expert<0) return; // uniform, before any CTA barrier
    int tid=threadIdx.x, lane=tid%32, nw=tid/32%n_warps, peer=tid/(32*n_warps);
    int col=blockIdx.y*tn;
    using Atom=std::conditional_t<std::is_same_v<T,cutlass::half_t>,
        cute::PPU0010_16x16x16_F32F16F16F32_TN,cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;
    using Traits=cute::MMA_Traits<Atom>;
    static_assert(adjacent_k_pairs<typename Traits::BLayout>(), "a packed pair must be (n,k), (n,k+1)");
    float acc[8]{};
    stage(mem.pipe[0],c,tile,col,0);load::cp_async_commit();
    for(int it=0;it<c.k/tk;++it) {
        if(it+1<c.k/tk) {
            stage(mem.pipe[(it+1)%2],c,tile,col,(it+1)*tk);
            load::cp_async_commit();load::cp_async_wait<1>();
        } else {
            load::cp_async_wait<0>(); // drain the last populated stage, including one-block K
        }
        __syncthreads();
        auto const& s=mem.pipe[it%2];
        #pragma unroll
        for(int atom=0;atom<peer_k/16;++atom) {
            typename Delivery::FragmentA a;
            Delivery::load_swizzled_a(a,s.a[peer],0,atom*16*2);
            uint32_t ar[4],b[4];__builtin_memcpy(ar,&a,sizeof(ar));
            #pragma unroll
            for(int reg=0;reg<4;++reg) {
                // The actlize BLayout maps (lane,value) to n + 16*k.
                int coord=typename Traits::BLayout{}(lane,2*reg);
                int n=nw*16+coord%16, k=peer*peer_k+atom*16+coord/16;
                auto x=weight<Q,T,GPTQ>(s,n,k), y=weight<Q,T,GPTQ>(s,n,k+1);
                b[reg]=uint32_t(x.raw())|(uint32_t(y.raw())<<16);
            }
            Atom::fma(acc[0],acc[1],acc[2],acc[3],acc[4],acc[5],acc[6],acc[7],
                ar[0],ar[1],ar[2],ar[3],b[0],b[1],b[2],b[3],
                acc[0],acc[1],acc[2],acc[3],acc[4],acc[5],acc[6],acc[7]);
        }
        __syncthreads(); // retire all readers before reusing a stage or aliasing reduction scratch
    }
    Reduce::reduce_chunks(mem.peers,acc,peer,tid%(32*n_warps));
    if(peer==0) {
        #pragma unroll
        for(int v=0;v<8;++v) {
            int coord=typename Traits::CLayout{}(lane,v);
            int row=tile.first+coord%16;
            if(row<tile.end) static_cast<T*>(c.output)[int64_t(row)*c.n+col+nw*16+coord/16]=T(acc[v]);
        }
    }
}
} // namespace quactlize::dense::warpk::multi
