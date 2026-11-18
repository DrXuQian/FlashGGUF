import ctypes as C
from pathlib import Path
import subprocess

import pytest

from quactlize.affine.native import Buffers, Config, Info, Problem, problem
from quactlize.affine.tuning import CONFIGS, SCHEDULES, COOPERATIVE_K, COHORTS, NATIVE_A, NATIVE_B, CHUNK_A, BOUNDED_REDUCTION, CHUNK_REDUCTION, SPECIALIZATIONS, MIN_BLOCKS, STAGE_CYCLES, GLOBAL_SLICES, FUSED_STORE_WARPS, PERSISTENT_MN, MMA_M8, REGISTER_PREFETCH, PAIRED_SCALES, FULL_REGISTER_PIPELINE, DRIVER_OPTIONS
from quactlize.affine.tuning import GRID_MULTIPLE, VECTOR_OUTPUT, DIRECT_OUTPUT, UNROLL_CYCLES
from quactlize.affine.tuning import COMPILER_FENCE, OPAQUE_COORDINATES, SHARED_B, UNSIGNED_INDICES
from quactlize.affine.tuning import INTERLEAVED_A, STREAM_K_WORKERS, N_FAST_RASTER
from quactlize.affine.tuning import STATIC_M, PRUNE_ROWS, TAIL_M8, CACHE_SCALES, COMPACT_OUTPUT
from quactlize.affine.tuning import SKIP_STAGE2_BARRIER
from quactlize.affine.tuning import REDUCTION_STORAGE
from quactlize.affine.tuning import DIRECT_A
from quactlize.affine.tuning import COHORT_PIPELINE
from quactlize.affine.tuning import WARP_ORDER
from quactlize.affine.tuning import B_PRODUCER_WARP
from quactlize.affine.tuning import PREFETCH_REDUCTION
from quactlize.affine.tuning import GROUP_OUTPUT_STORES
from quactlize.affine.tuning import INTEGER_ZERO
from quactlize.affine.tuning import ZERO_WORD_LOAD
from quactlize.affine.tuning import SINGLE_ZERO_BIAS
from quactlize.affine.tuning import INTERLEAVE_ZERO_GROUPS
from quactlize.affine.tuning import PAIRED_ZERO_BIAS
from quactlize.affine.tuning import ZERO_BIAS_CACHE
from quactlize.affine.tuning import PERMUTE_ZERO_BIAS
from quactlize.affine.tuning import PREPARED_ZERO_BIAS
from quactlize.affine.tuning import BYTE_ZERO_WORDS
from quactlize.affine.tuning import INLINE_REDUCTION
from quactlize.affine.tuning import RASTER_GROUP_M
from quactlize.affine.tuning import STREAM_K_MODE
from quactlize.affine.tuning import GLOBAL_OUTPUT_PADDING
from quactlize.affine.tuning import PERSISTENT_PREFETCH


def test_affine_abi_sizes():
    assert C.sizeof(Problem) == 36
    assert C.sizeof(Buffers) == 72
    assert C.sizeof(Config) == 36
    assert C.sizeof(Info) == 32


def test_generated_affine_selector_dispatches_compiled_entries(tmp_path):
    from quactlize.affine.selection import EXACT, FALLBACKS, PRODUCTION_CONFIGS, render

    assert len({row[:4] for row in EXACT}) == len(EXACT)
    for mode,m,n,k,config in EXACT:
        assert (config in INTEGER_ZERO) == bool(mode)
        static_n,static_k=SPECIALIZATIONS.get(config,(0,0))
        assert static_n in (0,n) and static_k in (0,k)
        assert STATIC_M.get(config,0) in (0,m)
    root=Path(__file__).resolve().parents[1]
    # Compile the actual generator output, including its filtered diagnostic
    # build, rather than copying its selection logic into a Python oracle.
    for label,compiled in (("full",PRODUCTION_CONFIGS),("fallback",(621,))):
        rows=[row for row in EXACT if row[-1] in compiled]
        cases="\n".join(
            f"p.zero_mode={z}; p.m={m}; p.n={n}; p.k={k}; "
            f"assert(default_config(p)=={c});" for z,m,n,k,c in rows)
        fallback_checks="\n".join(
            f"p.zero_mode={z}; p.m={maximum}; p.n=256; p.k=256; "
            f"assert(default_config(p)=={c});"
            for z,maximum,c in FALLBACKS if c in compiled)
        source=tmp_path/f"selector-{label}.cpp"
        source.write_text(
            '#include <cassert>\n#include "quactlize/affine/api.h"\n' +
            render(compiled) + "\nint main() {\nqwa_problem_v1 p{}; p.split=1;\n" +
            cases + "\n" + fallback_checks +
            "\np.split=2; assert(default_config(p)==-1);\n"
            "p.split=1; p.zero_mode=2; assert(default_config(p)==-1);\n}\n")
        executable=tmp_path/f"selector-{label}"
        subprocess.run(["g++","-std=c++17","-Wall","-Wextra","-Werror",
                        f"-I{root}",str(source),"-o",str(executable)],check=True,
                       capture_output=True,text=True)
        subprocess.run([str(executable)],check=True,capture_output=True,text=True)


def test_streamk_schedule_coverage_and_boundary_ownership(tmp_path):
    source=tmp_path/"schedule.cpp"
    source.write_text(r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <set>
#include <vector>
#include "actlize_extensions/cutlass/gemm/kernel/kpack_streamk_schedule.hpp"
using Schedule=cutlass::gemm::kernel::KPackStreamKSchedulePPU;
template<unsigned GroupM>
void check_raster(unsigned mt,unsigned nt) {
  std::vector<unsigned> visits(mt*nt,0);
  for(unsigned tile=0;tile<mt*nt;++tile) {
    auto c=cutlass::gemm::kernel::grouped_mn_tile<GroupM>(tile,mt,nt);
    assert(c.m<mt && c.n<nt);
    assert(++visits[c.m*nt+c.n]==1);
    if constexpr(GroupM==1) assert(c.m==tile/nt && c.n==tile%nt);
  }
  for(auto count:visits) assert(count==1);
}
using Pipeline=cutlass::gemm::kernel::KPackMNPipelinePPU;
void check_pipeline(unsigned tiles,unsigned kt,unsigned workers,unsigned stages) {
  assert(Pipeline::aligned(kt,stages));
  std::vector<unsigned> visits(tiles*kt,0);
  for(unsigned w=0;w<workers;++w) {
    std::vector<unsigned> slots(stages,UINT32_MAX);
    for(unsigned tile=w;tile<tiles;tile+=workers) {
      auto copy=[&](unsigned logical_k) {
        auto target=Pipeline::copy(logical_k,kt,tile+workers<tiles);
        if(target.k>=kt) { assert(!target.next); return; }
        unsigned owner=target.next ? tile+workers : tile;
        assert(owner<tiles && logical_k%stages==target.k%stages);
        unsigned unit=owner*kt+target.k;
        assert(++visits[unit]==1);
        slots[logical_k%stages]=unit;
      };
      if(tile==w)
        for(unsigned k=0;k<Pipeline::prime(stages);++k) copy(k);
      for(unsigned k=0;k<kt;++k) {
        assert(slots[k%stages]==tile*kt+k);
        copy(k+Pipeline::prime(stages));
      }
    }
  }
  for(auto count:visits) assert(count==1);
}
int main() {
  assert(Pipeline::prime(2)==2 && Pipeline::prime(3)==2 && Pipeline::prime(4)==3);
  assert(!Pipeline::aligned(0,2) && !Pipeline::aligned(1,2));
  assert(!Pipeline::aligned(128,3) && Pipeline::aligned(48,3));
  auto next_copy=Pipeline::copy(4,4,true);
  assert(next_copy.next && next_copy.k==0);
  auto last_copy=Pipeline::copy(4,4,false);
  assert(!last_copy.next && last_copy.k==4);
  for(unsigned stages:{2,3,4})
  for(unsigned kt:{2,3,4,6,8,12,24,48,64,128}) {
    if(!Pipeline::aligned(kt,stages)) continue;
    for(unsigned tiles:{1,2,7,39,40,41,83,129})
    for(unsigned workers:{1,2,20,40,80})
      check_pipeline(tiles,kt,workers,stages);
  }
  for(unsigned mt:{1,2,3,4,5,7,8,9,13,17,31,32,33,64,65})
  for(unsigned nt:{1,2,3,7,12,32,64}) {
    check_raster<1>(mt,nt); check_raster<2>(mt,nt);
    check_raster<4>(mt,nt); check_raster<8>(mt,nt); check_raster<16>(mt,nt);
  }
  auto pair=cutlass::gemm::kernel::grouped_mn_tile<2>(1,5,3);
  assert(pair.m==1 && pair.n==0);
  auto tail=cutlass::gemm::kernel::grouped_mn_tile<2>(14,5,3);
  assert(tail.m==4 && tail.n==2);
  assert(!Schedule::make(0,24,40).valid());
  assert(!Schedule::make(64,0,40).valid());
  assert(!Schedule::make(64,24,0).valid());
  assert(!Schedule::make(UINT32_MAX,UINT32_MAX,40).valid());
  assert(!Schedule::make_tail(64,24,0).valid());
  assert(Schedule::make_tail(2048,128,40).tiles==48);
  for(unsigned tiles:{1,7,32,40,41,48,64,65,80,129,768,2048})
  for(unsigned kt:{1,2,7,24,128})
  for(unsigned workers:{1,2,8,20,40}) {
    auto s=Schedule::make_tail(tiles,kt,workers);
    if(!s.valid()) { assert(tiles*kt<workers); continue; }
    unsigned prefix=tiles-s.tiles;
    assert(prefix%workers==0);
    if(tiles>=workers) assert(s.max_contributors()<=2);
    // The ordinary prefix may have an independent launch grid in mode 3.
    for(unsigned prefix_workers:{1u,workers,2*workers}) {
      std::vector<unsigned> visits(tiles*kt,0);
      for(unsigned w=0;w<prefix_workers;++w)
      for(unsigned tile=w;tile<prefix;tile+=prefix_workers)
        for(unsigned k=0;k<kt;++k) ++visits[tile*kt+k];
      for(unsigned w=0;w<workers;++w) {
        unsigned cursor=s.begin(w);
        while(cursor<s.begin(w+1)) {
          auto work=s.next(w,cursor);
          assert(work.k_count && prefix+work.tile<tiles);
          if(work.peers==1) assert(work.k_begin==0 && work.k_count==kt);
          for(unsigned k=work.k_begin;k<work.k_begin+work.k_count;++k)
            ++visits[(prefix+work.tile)*kt+k];
          cursor+=work.k_count;
        }
        assert(cursor==s.begin(w+1));
      }
      for(auto count:visits) assert(count==1);
    }
  }
  assert(Schedule::make(32,24,40).max_contributors()==2);
  assert(Schedule::make(64,24,40).max_contributors()==2);
  assert(Schedule::make(16,6,32).max_contributors()==2);
  assert(Schedule::make(16,6,40).max_contributors()>2);
  for(unsigned tiles:{1,2,7,8,16,24,31,32,40,48,64,65,129})
  for(unsigned kt:{1,2,3,6,7,24,31})
  for(unsigned workers:{1,2,3,8,20,32,40}) {
    auto s=Schedule::make(tiles,kt,workers);
    if(tiles*kt<workers) { assert(!s.valid()); continue; }
    assert(s.valid());
    std::vector<unsigned> visits(tiles*kt,0),owners(tiles*kt,UINT32_MAX);
    std::vector<std::set<unsigned>> observed(tiles);
    // Enumerate the production intervals before examining its peer formula.
    for(unsigned w=0;w<workers;++w)
      for(unsigned u=s.begin(w);u<s.begin(w+1);++u) {
        assert(u<owners.size() && owners[u]==UINT32_MAX);
        owners[u]=w;
        observed[u/kt].insert(w);
      }
    std::map<unsigned,unsigned> scratch_owner;
    unsigned maximum=0;
    for(auto const& peers:observed) maximum=std::max(maximum,unsigned(peers.size()));
    assert(maximum==s.max_contributors());
    for(unsigned w=0;w<workers;++w) {
      unsigned cursor=s.begin(w);
      while(cursor<s.begin(w+1)) {
        auto work=s.next(w,cursor);
        assert(work.k_count && work.tile<tiles && work.k_begin+work.k_count<=kt);
        auto const& peers=observed[work.tile];
        assert(work.peers==peers.size());
        assert(work.peer==unsigned(std::distance(peers.begin(),peers.find(w))));
        assert(work.slot==*peers.begin() && work.slot<workers);
        if(work.peers==1) {
          assert(work.k_begin==0 && work.k_count==kt);
        } else {
          auto inserted=scratch_owner.emplace(work.slot,work.tile);
          assert(inserted.second || inserted.first->second==work.tile);
        }
        for(unsigned k=work.k_begin;k<work.k_begin+work.k_count;++k) {
          unsigned u=work.tile*kt+k;
          assert(u==cursor+k-work.k_begin && owners[u]==w);
          ++visits[u];
        }
        cursor+=work.k_count;
      }
      assert(cursor==s.begin(w+1));
      assert(s.next(w,cursor).k_count==0);
    }
    for(auto count:visits) assert(count==1);
  }
}
""")
    executable=tmp_path/"schedule"
    include=Path(__file__).resolve().parents[1]/"quactlize"/"include"
    subprocess.run(["g++","-std=c++17","-O2","-Wall","-Wextra","-Werror",
                    "-fsanitize=undefined",f"-I{include}",str(source),"-o",str(executable)],
                   check=True,capture_output=True,text=True)
    subprocess.run([str(executable)],check=True,capture_output=True,text=True)


def test_affine_candidate_registry():
    identities = [(config, SCHEDULES.get(index, (0, 0)), COOPERATIVE_K.get(index), COHORTS.get(index),
                   index in NATIVE_A, index in NATIVE_B, index in CHUNK_A,
                   index in BOUNDED_REDUCTION, index in CHUNK_REDUCTION,
                   DRIVER_OPTIONS.get(index,(0,0)),SPECIALIZATIONS.get(index,(0,0)),
                   MIN_BLOCKS.get(index,1),index in STAGE_CYCLES,GLOBAL_SLICES.get(index,1),
                   FUSED_STORE_WARPS.get(index,0),index in PERSISTENT_MN,MMA_M8.get(index,0),
                   index in REGISTER_PREFETCH,index in PAIRED_SCALES,index in FULL_REGISTER_PIPELINE,
                   GRID_MULTIPLE.get(index,40),index in VECTOR_OUTPUT,index in DIRECT_OUTPUT,
                   index in UNROLL_CYCLES,COMPILER_FENCE.get(index,0),index in OPAQUE_COORDINATES,
                   index in SHARED_B,index in UNSIGNED_INDICES,index in INTERLEAVED_A,
                   STREAM_K_WORKERS.get(index,0),index in N_FAST_RASTER,
                   STATIC_M.get(index,0),index in PRUNE_ROWS,index in TAIL_M8,index in CACHE_SCALES,
                   COMPACT_OUTPUT.get(index,0),index in SKIP_STAGE2_BARRIER,
                   REDUCTION_STORAGE.get(index,0),DIRECT_A.get(index,0),
                   COHORT_PIPELINE.get(index,0),WARP_ORDER.get(index,0),
                   B_PRODUCER_WARP.get(index,0),index in PREFETCH_REDUCTION,
                   index in GROUP_OUTPUT_STORES,index in INTEGER_ZERO,index in ZERO_WORD_LOAD,
                   index in SINGLE_ZERO_BIAS,index in INTERLEAVE_ZERO_GROUPS,
                   index in PAIRED_ZERO_BIAS,index in ZERO_BIAS_CACHE,
                   index in PERMUTE_ZERO_BIAS,index in PREPARED_ZERO_BIAS,
                   index in BYTE_ZERO_WORDS,index in INLINE_REDUCTION,RASTER_GROUP_M.get(index,1),
                   STREAM_K_MODE.get(index,0),GLOBAL_OUTPUT_PADDING.get(index,8),
                   index in PERSISTENT_PREFETCH)
                  for index, config in enumerate(CONFIGS)]
    assert len(identities) == len(set(identities))
    assert (64, 128, 64, 64, 32, 3) in CONFIGS
    for index,(tm, tn, tk, wm, wn, stages) in enumerate(CONFIGS):
        assert tm % wm == tn % wn == 0
        assert wn * tk * 4 >= 4096
        assert stages >= 1
        if index in CHUNK_REDUCTION:
            assert index not in INLINE_REDUCTION
        if index in SKIP_STAGE2_BARRIER:
            assert stages==2 and index in FULL_REGISTER_PIPELINE
        if index in PERSISTENT_PREFETCH:
            assert index in PERSISTENT_MN and index in DIRECT_OUTPUT
            assert index in FULL_REGISTER_PIPELINE and COOPERATIVE_K[index]==tk
            assert GLOBAL_SLICES.get(index,1)==1 and not STREAM_K_WORKERS.get(index,0)
            assert SPECIALIZATIONS[index][1]//tk%stages==0
        assert 1<=RASTER_GROUP_M.get(index,1)<=16
        if RASTER_GROUP_M.get(index,1)>1:
            assert index in N_FAST_RASTER
        assert 0<=STREAM_K_MODE.get(index,0)<=3
        if STREAM_K_MODE.get(index,0)>0:
            assert STREAM_K_WORKERS.get(index,0)>0 and index not in PERSISTENT_MN
        assert GLOBAL_OUTPUT_PADDING.get(index,8) in (0,8)
        if GLOBAL_OUTPUT_PADDING.get(index,8)==0:
            assert GLOBAL_SLICES.get(index,1)>1 or STREAM_K_WORKERS.get(index,0)>0
        if index in COOPERATIVE_K:
            assert wm%8==0
            assert wm<16 or wm%16==0 or index in TAIL_M8
        if stages==1:
            assert index in COOPERATIVE_K and index in NATIVE_A and index in NATIVE_B
            assert index not in FULL_REGISTER_PIPELINE


def test_affine_problem_preserves_quantization_contract():
    value = problem(12, 1024, 3072, config=0, split=1)
    assert (value.m, value.n, value.k, value.group_size, value.zero_mode) == (12, 1024, 3072, 32, 0)
    assert value.version == 1 and value.size == C.sizeof(Problem)
    assert problem(12,1024,3072,zero_mode=1).zero_mode==1

def test_compact_partials_cover_live_outputs_without_aliasing(tmp_path):
    source=tmp_path/"compact.cpp"
    source.write_text(r"""
#include <cassert>
#include <vector>
#include "actlize_extensions/cutlass/gemm/kernel/marlin_output_map_ppu.hpp"
namespace map=cutlass::gemm::kernel::marlin_ppu_detail;
template<int IM> void check() {
  for(int rows=1;rows<=IM;++rows)
  for(int wn:{1,2,4,8})
  for(int na:{1,2,4,8}) {
    int n=wn*na*16;
    std::vector<int> slots(rows*n,0),outputs(rows*n,0);
    for(int w=0;w<wn;++w)
    for(int a=0;a<na;++a)
    for(int lane=0;lane<32;++lane)
    for(int v=0;v<IM/2;v+=4) {
      int row=map::output_row<IM>(lane,v);
      if(row>=rows) continue;
      int slot=map::compact_partial_vector_slot<IM>(rows,na,w,lane,a,v);
      assert(slot>=0 && (slot+1)*4<=rows*n);
      for(int j=0;j<4;++j) {
        assert(map::output_row<IM>(lane,v+j)==row);
        int col=(w*na+a)*16+map::output_col_offset<IM>(lane,v+j);
        assert(col>=0 && col<n);
        ++slots[slot*4+j];
        ++outputs[row*n+col];
      }
    }
    for(auto visits:slots) assert(visits==1);
    for(auto visits:outputs) assert(visits==1);
  }
}
int main() { check<8>();check<16>(); }
""")
    root=Path(__file__).resolve().parents[1]
    executable=tmp_path/"compact"
    subprocess.run(["g++","-std=c++17","-O2","-Wall","-Wextra","-Werror",
                    "-fsanitize=undefined",f"-I{root/'quactlize/include'}",
                    f"-I{root/'third_party/actlize/include'}",
                    str(source),"-o",str(executable)],check=True,capture_output=True,text=True)
    subprocess.run([str(executable)],check=True,capture_output=True,text=True)


@pytest.mark.parametrize("kwargs", [
    {"m": 0}, {"n": 257}, {"k": 257}, {"group_size": 128},
    {"split": 0}, {"split": 3}, {"config": -2}, {"config": 2**31},
    {"m": 2**31}, {"zero_mode": 2}, {"zero_mode": -1},
])
def test_affine_problem_rejects_unsupported_contract(kwargs):
    args = dict(m=12, n=1024, k=3072)
    args.update(kwargs)
    with pytest.raises(ValueError):
        problem(**args)
