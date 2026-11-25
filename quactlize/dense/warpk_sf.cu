#include "warpk.hpp"
#include "../packing/api.h"
#include "../runtime/dynamic.hpp"
#include "../include/ppu_device.hpp"
#include "quactlize_actlize.hpp"
#include "warpk_multiformat.cuh"
#include <new>

namespace quactlize::dense::warpk {
int device_units(int device) {
    int current=-1;hggcDeviceProp p{};
    if(hggcGetDevice(&current)!=hggcSuccess || current!=device ||
       hggcGetDeviceProperties(&p,device)!=hggcSuccess || p.warpSize!=32 ||
       !quactlize::device::uses_zw810_policy(p.name,p.multiProcessorCount)) return 0;
    return p.multiProcessorCount;
}
using Submit=int(*)(qk_call_v1 const&);
struct WarpPlan {int q; bool zero; uint64_t metadata_bytes; qk_compute_device_call_v4 shape; Submit run; void* legacy;};

template<int Q,class T,bool GPTQ=false,bool FQ=false> int submit(qk_call_v1 const& c) {
    constexpr int shared=sizeof(multi::Storage<Q>);
    dim3 grid(tasks(c.m,c.experts,c.offsets_device!=nullptr),c.n/cols_tile);
    if constexpr(FQ)
        multi::fq_kernel<Q,T><<<grid,multi::threads,shared,static_cast<hggcStream_t>(c.stream)>>>(c);
    else
        multi::kernel<Q,T,GPTQ><<<grid,multi::threads,shared,static_cast<hggcStream_t>(c.stream)>>>(c);
    return hggcGetLastError()==hggcSuccess ? QK_OK : QK_RUNTIME_ERROR;
}
template<int Q,class T,bool GPTQ=false,bool FQ=false> Submit configure() {
    auto entry=[] {
        if constexpr(FQ) return multi::fq_kernel<Q,T>;
        else return multi::kernel<Q,T,GPTQ>;
    }();
    return hggcFuncSetAttribute(entry,hggcFuncAttributeMaxDynamicSharedMemorySize,
        sizeof(multi::Storage<Q>))==hggcSuccess ? &submit<Q,T,GPTQ,FQ> : nullptr;
}
template<class T,bool FQ> Submit configure(int q,bool gptq) {
    switch(q) {
    case 8: return configure<8,T>(); // original FP16 d, no SF prepass and no second binary
    case 10:return configure<10,T,false,FQ>();
    case 11:return configure<11,T,false,FQ>();
    case 12:return gptq?configure<12,T,true>():configure<12,T,false,FQ>();
    case 13:return configure<13,T,false,FQ>();
    case 14:return configure<14,T,false,FQ>();
    default:return nullptr;
    }
}
bool header(qk_compute_device_call_v4 const& d) {
    auto const& w=d.device_call;auto const& c=w.call;
    return d.version==4 && d.size==sizeof(d) && w.version==2 && w.size==sizeof(w) &&
        !w.reserved && c.version==1 && c.size==sizeof(c);
}
bool disjoint(qk_call_v1 const& c,WarpPlan const& p) {
    uint64_t nk=uint64_t(c.experts)*c.n*c.k, out_size=uint64_t(c.m)*c.n*2;
    auto out=uintptr_t(c.output);
    auto ok=[&](void const* p,uint64_t sz) {
        auto src=uintptr_t(p);
        return !sz || (src && sz<=UINTPTR_MAX-src && (out+out_size<=src || src+sz<=out));
    };
    // Nk is bounded per expert; E<=1024. No overflow in plane byte counts.
    return out && out_size<=UINTPTR_MAX-out && ok(c.a,uint64_t(c.m)*c.k*2) &&
        ok(c.low,nk*low_bits(p.q)/8) && ok(c.high,nk*high_bits(p.q)/8) &&
        ok(c.metadata,p.metadata_bytes) && ok(c.zero,p.zero?p.metadata_bytes:0) &&
        ok(c.offsets_device,c.offsets_device?(uint64_t(c.experts)+1)*4:0);
}
}

template<bool FQ> static int prepare_warpk(int q,qk_compute_device_call_v4 const* d,void** out) {
    using namespace quactlize::dense::warpk;
    if(!out) return QK_INVALID;
    *out=nullptr;
    if(!d || !header(*d)) return QK_INVALID;
    auto const& w=d->device_call;auto const& c=w.call;
    if(!quactlize::runtime::address_free(c)) return QK_INVALID;
    bool gptq=q==12 && c.mapping_id==QK_MAPPING_GPTQ_W4_G32;
    if(!is_format(q) || (FQ && gptq) || (!gptq && c.mapping_id!=gguf_map(q)) || c.group_size!=group_size(q) ||
        !supported(c.m,c.n,c.k) || ((q==11 || q==14) && c.k%512) || c.experts<1 || c.experts>1024 ||
        w.max_rows<0 || (w.max_rows==0 ? c.experts!=1 :
          (w.max_rows>c.m || int64_t(w.max_rows)*c.experts<c.m)) ||
        (d->compute_type!=QK_COMPUTE_F16 && d->compute_type!=QK_COMPUTE_BF16) ||
        d->metadata_type!=(q==8?int(QK_METADATA_F16):d->compute_type)) return QK_UNSUPPORTED;
    uint64_t meta_bytes=uint64_t(c.experts)*c.n*c.k/group_size(q)*2;
    if constexpr(FQ) {
        quactlize_ppu_placed_arrangement_v2 a{};quactlize_ppu_kpack_sizes_v1 sz{};
        if(quactlize_ppu_kpack_canonical_arrangement_v1(q,&a) ||
           quactlize_ppu_kpack_sizes_for_arrangement_v1(c.n,c.k,c.experts,q,&a,&sz)) return QK_UNSUPPORTED;
        meta_bytes=sz.units_bytes;
    }
    hggcStreamCaptureStatus capture;
    if(c.compute_units<=0 || device_units(c.device)!=c.compute_units ||
       hggcStreamIsCapturing(static_cast<hggcStream_t>(c.stream),&capture)!=hggcSuccess ||
       capture!=hggcStreamCaptureStatusNone) return QK_UNSUPPORTED;
    // Keep the previous tuned Q4/F16 dense binary as-is, not a newly named reimplementation.
    void* legacy=nullptr;Submit run=nullptr;
    if(!FQ && q==12 && !w.max_rows && !d->compute_type) {
        int rc=quactlize_ppu_warpk_prepare_v1(&c,&legacy);if(rc) return rc;
    } else {
        run=d->compute_type ? configure<cutlass::bfloat16_t,FQ>(q,gptq) : configure<cutlass::half_t,FQ>(q,gptq);
        if(!run) return QK_INITIALIZE_ERROR;
    }
    *out=new(std::nothrow) WarpPlan{q,!FQ && q!=8,meta_bytes,*d,run,legacy};
    if(!*out) {quactlize_ppu_warpk_destroy_v1(legacy);return QK_INITIALIZE_ERROR;}
    return QK_OK;
}
extern "C" int quactlize_ppu_warpk_prepare_v2(int q,qk_compute_device_call_v4 const* d,void** out) {
    return prepare_warpk<false>(q,d,out);
}
extern "C" int quactlize_ppu_warpk_prepare_fq_v2(int q,qk_compute_device_call_v4 const* d,void** out) {
    return prepare_warpk<true>(q,d,out);
}
extern "C" int quactlize_ppu_warpk_run_v2(void* h,qk_compute_device_call_v4 const* d) {
    using namespace quactlize::dense::warpk;
    if(!h || !d || !header(*d)) return QK_INVALID;
    auto& p=*static_cast<WarpPlan*>(h);auto const& w=d->device_call;auto const& c=w.call;
    auto const& s=p.shape.device_call.call;
    if(c.m!=s.m || c.n!=s.n || c.k!=s.k || c.experts!=s.experts || c.device!=s.device ||
       c.mapping_id!=s.mapping_id || c.group_size!=s.group_size || c.compute_units!=s.compute_units ||
       w.max_rows!=p.shape.device_call.max_rows || d->compute_type!=p.shape.compute_type ||
       d->metadata_type!=p.shape.metadata_type || c.rows_host || c.rows_device || c.workspace || c.workspace_bytes ||
       bool(c.offsets_device)!=bool(w.max_rows) || uintptr_t(c.offsets_device)%4 ||
       !quactlize::runtime::dense_buffers(c,0,high_bits(p.q)!=0,p.zero) || !disjoint(c,p)) return QK_INVALID;
    if(p.legacy) return quactlize_ppu_warpk_run_v1(p.legacy,&c);
    int dev=-1;
    if(hggcGetDevice(&dev)!=hggcSuccess || dev!=c.device) return QK_INVALID;
    return p.run(c);
}
extern "C" void quactlize_ppu_warpk_destroy_v2(void* h) {
    auto* p=static_cast<quactlize::dense::warpk::WarpPlan*>(h);
    if(p) quactlize_ppu_warpk_destroy_v1(p->legacy);
    delete p;
}
