// Host-only CuTe check of the production M16 atom and warp-grid partition.
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_ppu0010.hpp>
#include <bitset>
#include <cstdio>

using namespace cute;
int main() {
    using Mma = TiledMMA<MMA_Atom<PPU0010_16x16x16_F32F16F16F32_TN>,
                         Layout<Shape<_1, _16, _1>>, Tile<_16, _256, _64>>;
    static_assert(size(Mma{}) == 512);
    auto c = make_identity_tensor(Shape<_64,_256>{});
    auto b = make_identity_tensor(Shape<_256,_64>{});
    std::bitset<64*256> output;
    for (int w = 0; w < 16; ++w) {
        std::bitset<16*64> weight;
        for (int lane = 0; lane < 32; ++lane) {
            auto self = Mma{}.get_thread_slice(32*w + lane);
            auto peer = Mma{}.get_thread_slice(32*(w ^ 8) + lane);
            auto l = self.partition_C(c), r = peer.partition_C(c);
            auto u = self.partition_B(b), v = peer.partition_B(b);
            const int delta = w < 8 ? 128 : -128;
            for (int i = 0; i < size(l); ++i) {
                int m = get<0>(l(i)), n = get<1>(l(i));
                if (m != get<0>(r(i)) || n+delta != get<1>(r(i)) ||
                    m < 0 || m >= 64 || n/16 != w || output.test(m*256+n)) return 1;
                output.set(m*256+n);
            }
            for (int i = 0; i < size(u); ++i) {
                int n = get<0>(u(i)), k = get<1>(u(i));
                if (n/16 != w || n+delta != get<0>(v(i)) || k != get<1>(v(i)) ||
                    k < 0 || k >= 64) return 1;
                weight.set(n%16*64+k);
            }
        }
        if (!weight.all()) return 1;
    }
    if (!output.all()) return 1;
    std::puts("PASS: 16 N warps, 8 pairs, all lanes; C disjoint, B columns +128 within each pair");
}
