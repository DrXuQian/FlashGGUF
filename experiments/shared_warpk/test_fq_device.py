"""Real PPU FQ public projections and rounded packed-unit witnesses."""
import ctypes as C
from contextlib import ExitStack
import numpy as np
import pytest
from ci.support.fixture import bf16_bits,bf16_float
from ci.support.graph import Replay
from quactlize.runtime.native import Call,checked
from quactlize.runtime.sf_gptq import GPTQ_W4_G32
from quactlize.decode.native_compute import GroupedDeviceCall,GroupedMetadataCall
from tests.ppu.conftest import package,sdk
from tests.ppu.support import Arena,Public,QTYPES,packed,bind,Arrangement,Sizes
from tests.warpk_fq_oracle import fixture,oracle
from experiments.shared_warpk.test_formats_device import public_case,MS

ref=C.byref

@pytest.mark.parametrize('q',QTYPES)
@pytest.mark.parametrize('bf,e,ch',[(False,1,1),(True,1,1),(True,4,1),(True,4,2)],
                         ids=['dense-f16','dense-bf16','moe-gate','moe-down'])
def test_fq_projection(package,sdk,q,bf,e,ch):
    api=Public(package)
    with Arena(sdk) as weights:
        w=packed(weights,api,q,256,512,e,varied_groups=True)
        for m in MS:
            try:
                public_case(sdk,api,q,256,512,m,w,bf=bf,experts=e,channels=ch,candidate='fq')
            except Exception as exc:
                raise AssertionError(f'FQ q={q}, M={m}, E={e}, channels={ch}, BF16={bf}') from exc


@pytest.mark.parametrize('q,n,k,ch',[(13,1024,2048,1),(14,2048,512,8)])
def test_fq_projection_model_dimensions(package,sdk,q,n,k,ch):
    api=Public(package)
    with Arena(sdk) as weights:
        w=packed(weights,api,q,n,k,8,varied_groups=True)
        public_case(sdk,api,q,n,k,17,w,bf=True,experts=8,channels=ch,topk=8,candidate='fq')


@pytest.mark.parametrize('q',(8,10,12,13))
def test_fq_one_stage(package,sdk,q):
    api=Public(package)
    with Arena(sdk) as weights:
        w=packed(weights,api,q,256,256,1,varied_groups=True)
        public_case(sdk,api,q,256,256,9,w,bf=True,experts=1,channels=1,candidate='fq')


@pytest.mark.parametrize('q',range(10,15))
@pytest.mark.parametrize('bf',[False,True],ids=['f16','bf16'])
def test_fq_original_units_replay(package,sdk,q,bf):
    """Only raw packed units provided; changing offsets, two SB pairs, no SF workspace."""
    public=Public(package);ptr=C.c_void_p;dp=C.POINTER(GroupedMetadataCall)
    prep=bind(public.lib,'quactlize_ppu_warpk_prepare_fq_v2',[C.c_int,dp,C.POINTER(ptr)])
    run=bind(public.lib,'quactlize_ppu_warpk_run_v2',[ptr,dp])
    destroy=bind(public.lib,'quactlize_ppu_warpk_destroy_v2',[ptr],None)
    get=bind(sdk.lib,'hggcDeviceGetAttribute',[C.POINTER(C.c_int),C.c_int,C.c_int])
    cu=C.c_int();checked(get(ref(cu),16,0),'device CUs')
    m,n,k,e=33,256,1024,2
    raws=[fixture(q,n,k,81+i) for i in range(e)]
    w=np.stack([oracle(r,q,bf)[2] for r in raws])
    arr,sz=Arrangement(),Sizes()
    checked(public.canonical(q,ref(arr)),'arrangement');checked(public.sizes(n,k,e,q,ref(arr),ref(sz)),'sizes')
    c=Call(1,C.sizeof(Call),m,n,k,e,arr.group_size,0,cu.value,arr.mapping_id)
    d=GroupedMetadataCall(4,C.sizeof(GroupedMetadataCall),GroupedDeviceCall(2,C.sizeof(GroupedDeviceCall),c,m,0),
                          int(bf),int(bf))
    h=ptr()
    bad=GroupedMetadataCall.from_buffer_copy(d);bad.device_call.call.mapping_id=GPTQ_W4_G32
    assert prep(q,ref(bad),ref(h))!=0 and not h
    with Arena(sdk) as a, ExitStack() as st:
        checked(prep(q,ref(d),ref(h)),'prepare FQ');st.callback(destroy,h)
        stream=a.stream();source=a.upload(np.stack(raws))
        lo=a.buffer(sz.low_bytes);hi=a.buffer(sz.high_bytes) if sz.high_bytes else None;units=a.buffer(sz.units_bytes)
        a.sync();checked(public.pack(source.ptr,lo.ptr,hi.ptr if hi else None,units.ptr,n,k,e,q,ref(arr),stream),
                         'pack raw GGUF');a.sync()
        inp=a.buffer(m*k*2);out=a.buffer(m*n*2);off=a.buffer((e+1)*4)
        c=d.device_call.call
        c.a,c.low,c.high,c.metadata,c.output,c.offsets_device,c.stream=(
            inp.ptr,lo.ptr,hi.ptr if hi else None,units.ptr,out.ptr,off.ptr,stream.value)
        for field,value in [('zero',units.ptr),('output',units.ptr+units.size-16),('metadata',None)]:
            bad=GroupedMetadataCall.from_buffer_copy(d);setattr(bad.device_call.call,field,value)
            assert run(h,ref(bad))!=0,field
        launch=lambda:run(h,ref(d));graph=None
        for step,split in enumerate((0,33,13)):
            rows=np.arange(m)
            idx=(np.resize([0,15,31,63,127,255,256,383,511,512,767,768,1023],m)+17*step)%k
            x=np.zeros((m,k),'f4');x[rows,idx]=1
            inp.write(bf16_bits(x) if bf else x.astype('f2'));off.write(np.array([0,split,m],'i4'))
            out.poison();a.sync()
            if step==1:
                graph=Replay(sdk,stream,launch,1);st.callback(graph.close);st.callback(a.sync)
            checked(graph() if graph else launch(),'FQ one-hot');a.sync()
            got=np.frombuffer(out.bytes(),'u2' if bf else 'f2').reshape(m,n)
            np.testing.assert_array_equal(bf16_float(got) if bf else got,w[(rows>=split).astype(int),:,idx])
        units.guard()
