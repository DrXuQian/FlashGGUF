#pragma once
#include "../fusion/store.cuh"

namespace quactlize::fusion {
struct SimtFinish {
    template<int TileN>
    CUTLASS_DEVICE static void invalid(DeviceCall const& c, int row, int tile, int partition, int split) {
        int tid = int(threadIdx.x);
        float nan = __int_as_float(0x7fc00000);
        if (split == 1) {
            if (tid < TileN/2) output(c,row,tile*TileN/2+tid,nan);
        } else if (tid < TileN) {
            static_cast<float*>(c.workspace)[(int64_t(row)*split+partition)*c.n+tile*TileN+tid] = nan;
        }
    }
    template<int TileN,int Warps>
    CUTLASS_DEVICE static void finish(DeviceCall const& c,int row,int tile,int partition,int split,float const* partial) {
        static_assert(TileN == 16 || TileN == 32, "paired SIMT requires complete G4/U4 pairs");
        int tid = int(threadIdx.x);
        if (split > 1) {
            if (tid < TileN) {
                float sum = 0.f;
                #pragma unroll
                for (int w=0;w<Warps;++w) sum += partial[w*TileN+tid];
                static_cast<float*>(c.workspace)[(int64_t(row)*split+partition)*c.n+tile*TileN+tid] = sum;
            }
        } else if (tid < TileN/2) {
            int ng = PairedN4::gate(tid);
            float gate = 0.f, up = 0.f;
            #pragma unroll
            for (int w=0;w<Warps;++w) {
                gate += partial[w*TileN+ng]; up += partial[w*TileN+ng+4];
            }
            output(c,row,tile*TileN/2+tid,activate(c,gate,up));
        }
    }
};

// Keep the confirmed paired-16 sequential inter-warp reduction order.
struct Paired16Finish : SimtFinish {
    template<int TileN,int Warps>
    CUTLASS_DEVICE static void finish(DeviceCall const& c,int row,int tile,
        int partition,int split,float const* partial) {
        static_assert(TileN==16,"half-width paired tile");
        int tid=int(threadIdx.x);
        if(tid<8) {
            int ng=PairedN4::gate(tid);
            float gate=0.f,up=0.f;
            #pragma unroll
            for(int w=0;w<Warps;++w) {
                gate+=partial[w*16+ng];up+=partial[w*16+ng+4];
            }
            output(c,row,tile*8+tid,activate(c,gate,up));
        }
    }
};
}
