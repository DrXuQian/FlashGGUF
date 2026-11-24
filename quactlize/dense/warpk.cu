// FlashQuant owns the shared collective and its warp-K reduction driver.
// Compile against its header closure, never link a second provider DSO.
#include "warpk.hpp"
#include "../runtime/dynamic.hpp"
#include "../include/ppu_device.hpp"
#include <new>
#include "quactlize_actlize.hpp"
#include "cutlass/device_kernel.h"
#include "actlize_extensions/cutlass/gemm/kernel/kpack_warpk_ppu.hpp"

namespace quactlize::dense::warpk {
namespace cc = cutlass::gemm::collective;
// Native A, canonical K-pack4 (not classic Marlin), native B, chunked A.
template<cc::Group32Operand Style>
using Mainloop = cc::NativeInt4WarpKCollectivePPU<
    TM,TN,TK,WN,WK,stages,true,Style,true,true,false,0,false,false,false,8>;
using Style=cc::Group32Operand;
using GGUF=Mainloop<Style::PostScaleOffset>;
using GPTQ=Mainloop<Style::PreScaleCenter>;
template<class Loop> using Driver=cutlass::gemm::kernel::KPackWarpKKernelPPU<Loop,true>;
static_assert(GGUF::MappingId==UINT64_C(0x51344b5034540001));
static_assert(GGUF::WarpM==1 && GGUF::WarpN==4 && GGUF::WarpK==4);
static_assert(Driver<GGUF>::MaxThreadsPerBlock==512);

template<class Loop> int initialize_kernel() {
    using D=Driver<Loop>;
    return hggcFuncSetAttribute(cutlass::device_kernel<D>,
        hggcFuncAttributeMaxDynamicSharedMemorySize,D::SharedStorageSize);
}
template<class Loop> int launch_kernel(int m,int n,int k,void const* a,void const* low,void const* scale,
           void const* offset,void* output,void* stream) {
    using Half=cutlass::half_t;
    using D=Driver<Loop>;
    int const mt=(m-1)/TM+1;
    auto operands=Loop::make_params(m,n,k,static_cast<Half const*>(a),
        static_cast<uint16_t const*>(low),static_cast<Half const*>(scale));
    operands.offsets=static_cast<Half const*>(offset);
    typename D::Params params{operands,static_cast<Half*>(output),cutlass::FastDivmod(mt),mt*(n/TN)};
    void* args[]={&params};
    return hggcLaunchKernel(reinterpret_cast<void const*>(cutlass::device_kernel<D>),
        dim3(mt,n/TN),dim3(D::MaxThreadsPerBlock),args,D::SharedStorageSize,
        static_cast<hggcStream_t>(stream));
}
int initialize() { return initialize_kernel<GGUF>(); }
int launch(int m,int n,int k,void const* a,void const* low,void const* scale,
           void const* offset,void* output,void* stream) {
    return launch_kernel<GGUF>(m,n,k,a,low,scale,offset,output,stream);
}
struct Prepared {
    qk_call_v1 shape;
    decltype(&launch) submit;
};
}

extern "C" int quactlize_ppu_warpk_prepare_v1(qk_call_v1 const* c,void** out) {
    using namespace quactlize::dense::warpk;
    if (!out) return QK_INVALID;
    *out=nullptr;
    if (!c || c->version!=1 || c->size!=sizeof(*c) || !quactlize::runtime::address_free(*c))
        return QK_INVALID;
    if (!supported(c->m,c->n,c->k) || c->experts!=1 || c->group_size!=32 ||
        (c->mapping_id!=GGUF::MappingId && c->mapping_id!=QK_MAPPING_GPTQ_W4_G32)) return QK_UNSUPPORTED;
    int dev=-1;hggcDeviceProp prop{};hggcStreamCaptureStatus capture;
    if (hggcGetDevice(&dev)!=hggcSuccess || dev!=c->device ||
        hggcGetDeviceProperties(&prop,dev)!=hggcSuccess ||
        hggcStreamIsCapturing(static_cast<hggcStream_t>(c->stream),&capture)!=hggcSuccess) return QK_RUNTIME_ERROR;
    if (capture!=hggcStreamCaptureStatusNone || prop.warpSize!=32 ||
        c->compute_units!=prop.multiProcessorCount ||
        !quactlize::device::uses_zw810_policy(prop.name,prop.multiProcessorCount)) return QK_UNSUPPORTED;
    bool const gptq=c->mapping_id==QK_MAPPING_GPTQ_W4_G32;
    if ((gptq?initialize_kernel<GPTQ>():initialize())!=0) return QK_INITIALIZE_ERROR;
    *out=new(std::nothrow) Prepared{*c,gptq?launch_kernel<GPTQ>:launch_kernel<GGUF>};
    return *out?QK_OK:QK_INITIALIZE_ERROR;
}
extern "C" int quactlize_ppu_warpk_run_v1(void* handle,qk_call_v1 const* c) {
    using namespace quactlize::dense::warpk;
    auto* p=static_cast<Prepared*>(handle);
    if (!p || !c || !quactlize::runtime::same_dense_shape(*c,p->shape) ||
        c->rows_device || c->offsets_device || c->workspace || c->workspace_bytes ||
        !quactlize::runtime::dense_buffers(*c,0,false,true)) return QK_INVALID;
    // No alias with C: distinct CTAs may still read A or weights after the first store.
    uint64_t const nk=uint64_t(c->n)*c->k, cn=uint64_t(c->m)*c->n*2;
    auto const dst=reinterpret_cast<uintptr_t>(c->output);
    auto disjoint=[&](void const* ptr,uint64_t bytes) {
        auto const src=reinterpret_cast<uintptr_t>(ptr);
        return cn<=UINTPTR_MAX-dst && bytes<=UINTPTR_MAX-src &&
               (dst+cn<=src || src+bytes<=dst);
    };
    if (!disjoint(c->a,uint64_t(c->m)*c->k*2) || !disjoint(c->low,nk/2) ||
        !disjoint(c->metadata,nk/16) || !disjoint(c->zero,nk/16)) return QK_INVALID;
    int dev=-1;
    if (hggcGetDevice(&dev)!=hggcSuccess || dev!=c->device) return QK_INVALID;
    return p->submit(c->m,c->n,c->k,c->a,c->low,c->metadata,c->zero,c->output,c->stream)==0 ?
        QK_OK:QK_RUNTIME_ERROR;
}
extern "C" void quactlize_ppu_warpk_destroy_v1(void* p) {
    delete static_cast<quactlize::dense::warpk::Prepared*>(p);
}
