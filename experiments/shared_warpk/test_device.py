"""Opt-in PPU1.0 admission: actual public GGUF, FlashQuant A/B, all W4 SF schedules."""
import ctypes as C
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from ci.support.fixture import bf16_bits, bf16_float
from ci.support.graph import Replay
from quactlize.runtime import sf_gptq
from quactlize.runtime.compiler import Compiler
from quactlize.runtime.native import Call, Module, Recipe, Resources, checked
from quactlize.decode.compiler import DecodeCompiler
from quactlize.decode.grouped_compiler import GroupedComputeCompiler
from quactlize.decode.native_compute import (DenseComputeCall,DenseIO,GroupedDeviceCall,
                                            GroupedMetadataCall,bind_compute)
from tests.ppu.conftest import sdk, package  # use the real package verification/device fixtures
from tests.ppu.support import Arena,Public,Options,Params,Candidate,Info,packed,bind

ROOT=Path(__file__).resolve().parents[2]
MS=(*range(1,10),16,17)
NATIVE_DENSE=('dense','dense-ap1','dense-rc2','dense-rc4','dense-rc8')
SF_MODULES=(*NATIVE_DENSE,'typed-f16','typed-bf16','grouped','grouped-bf16')
ref=C.byref


@pytest.fixture(scope='module')
def modules(tmp_path_factory):
    sdk=Path(os.environ['PPU_SDK'])
    cache=tmp_path_factory.mktemp('sf-gptq-jit')
    native=Compiler(sdk,cache)
    compilers={**dict.fromkeys((*NATIVE_DENSE,'grouped'),native),
               'typed-f16':DecodeCompiler(sdk,cache,compute_type='f16'),
               'typed-bf16':DecodeCompiler(sdk,cache,compute_type='bf16'),
               'grouped-bf16':GroupedComputeCompiler(sdk,cache,compute_type='bf16')}
    out={}
    for name,compiler in compilers.items():
        g='grouped' in name
        ap=int(name=='dense-ap1')
        p=sf_gptq.parent(dict(symbol='box_'+name.replace('-','_'),qtype=12,
            route='sf-grouped' if g else 'sf-dense',tm=8 if ap else 16,tn=64,tk=128,
            wm=8 if ap else 16,wn=16,stages=2,ap=ap,dn=64,persistent=-1))
        record=compiler.build(p)
        out[name]=(C.CDLL(record['path'],mode=C.RTLD_LOCAL) if name.startswith('typed') else Module(record))
    return out


def data(n,k,e,seed,integer=True):
    rng=np.random.default_rng(seed)
    q=rng.integers(0,16,(e,n,k),dtype='u1')
    s=rng.uniform(.017,.099,(e,k//32,n)).astype('f2')
    z=rng.integers(0,16,s.shape,dtype='u1') if integer else None
    return rng,q,s,z


def golden(q,s,z,*,bf=False):
    rnd=(lambda v: bf16_float(bf16_bits(v))) if bf else (lambda v:v.astype('f2').astype('f4'))
    center=z.swapaxes(-1,-2)[...,None] if z is not None else 8
    return rnd((q.astype('f4').reshape(*q.shape[:-1],q.shape[-1]//32,32)-center)
               *rnd(s).swapaxes(-1,-2)[...,None]).reshape(q.shape)


@pytest.mark.parametrize('m',MS)
def test_gguf_public_and_graph(sdk,package,m):
    lib=Public(package);n,k=256,512
    with Arena(sdk) as a:
        arr,sz,pointers,w=packed(a,lib,12,n,k)
        p=Params(1,C.sizeof(Params),12,n,k,m,k+4,n+8,arr,sz)
        opts=Options(1,C.sizeof(Options),0,0);ctx=C.c_void_p();plan=C.c_void_p()
        spec=Candidate(1,C.sizeof(Candidate),4,1,1,0,-1);spec.split=1
        info=Info(1,C.sizeof(Info));lib.check(lib.create(0,ref(opts),ref(ctx)))
        graph=None
        try:
            lib.check(lib.candidate(ctx,ref(p),*pointers,ref(spec),ref(plan),ref(info)))
            assert info.route==1 and info.split==1 and b'warpk' in info.config
            av=a.buffer(m*(k+4)*4);cv=a.buffer(m*(n+8)*4);ws=a.buffer(info.workspace_bytes);stream=a.stream()
            launch=lambda:lib.run(plan,av.ptr,cv.ptr,ws.ptr,ws.size,stream)
            for it in range(3):
                inp=np.zeros((m,k+4),'f4');idx=(np.arange(m)*37+it*129)%k
                inp[np.arange(m),idx]=1;av.write(inp);cv.write(np.full((m,n+8),-1234,'f4'));a.sync()
                if it==1: graph=Replay(sdk,stream,launch,1)
                lib.check(graph() if graph else launch());a.sync()
                actual=np.frombuffer(cv.bytes(),'f4').reshape(m,n+8)
                np.testing.assert_array_equal(actual[:,:n],w[0,:,idx])
                assert np.all(actual[:,n:]==-1234)
            ws.guard()
        finally:
            a.sync()
            if graph: graph.close()
            lib.destroy(ctx)


@pytest.fixture(scope='module')
def baseline():
    sys.path.append(str(ROOT/'third_party/flashquant'))
    from flashquant.affine import Library,Buffers,Info,problem
    return Library(os.environ['FLASHQUANT_REFERENCE']),Buffers,Info,problem


@pytest.mark.parametrize('m',MS)
@pytest.mark.parametrize('integer',(False,True))
def test_gptq_warpk_vs_flashquant(package,sdk,modules,baseline,m,integer):
    old,Buffers,ShapeInfo,problem=baseline
    device=modules['dense'].device_identity()
    public=C.CDLL(str(package/'libncp_quactlize.so'))
    ptr=C.c_void_p
    prep=bind(public,'quactlize_ppu_warpk_prepare_v1',[C.POINTER(Call),C.POINTER(ptr)])
    run=bind(public,'quactlize_ppu_warpk_run_v1',[ptr,C.POINTER(Call)])
    destroy=bind(public,'quactlize_ppu_warpk_destroy_v1',[ptr],None)
    n,k=256,1024
    rng,q,s,z=data(n,k,1,91+m,integer)
    w=golden(q,s,z)[0];planes=sf_gptq.pack(q,s,z)
    p=problem(m,n,k,config=619 if integer else 616,zero_mode=int(integer))
    sizes=ShapeInfo();old.check(old.query(ref(p),ref(sizes)))
    with Arena(sdk) as a:
        stream=a.stream();inv=a.buffer(m*k*2);low=a.upload(planes['low']);scale=a.upload(planes['metadata'])
        center=a.upload(planes['zero']);zp=a.upload(z) if integer else None
        result=a.buffer(m*n*2);control=a.buffer(m*n*2);ws=a.buffer(max(256,sizes.workspace_bytes))
        c=Call(1,C.sizeof(Call),m,n,k,1,32,device['ordinal'],device['compute_units'],sf_gptq.GPTQ_W4_G32)
        b=Buffers(1,C.sizeof(Buffers),inv.ptr,low.ptr,scale.ptr,zp.ptr if zp else None,control.ptr,
                  ws.ptr,sizes.workspace_bytes,stream)
        h,oh=ptr(),ptr();a.sync()
        try:
            checked(prep(ref(c),ref(h)),'prepared warp-K');old.check(old.prepare(ref(p),ref(b),ref(oh)))
            c.a,c.low,c.metadata,c.zero,c.output,c.stream=inv.ptr,low.ptr,scale.ptr,center.ptr,result.ptr,stream.value
            wrong=Call.from_buffer_copy(c);wrong.mapping_id=old.mapping()
            assert run(h,ref(wrong))!=0
            for one_hot in (True,False):
                x=rng.uniform(-.25,.25,(m,k)).astype('f2')
                if one_hot:
                    x.fill(0);x[np.arange(m),(np.arange(m)*193+31)%k]=1
                inv.write(x);result.poison();control.poison();a.sync()
                old.check(old.run(oh,stream));checked(run(h,ref(c)),'prepared GPTQ warp-K');a.sync()
                both=[np.frombuffer(b.bytes(),'f2').astype('f4').reshape(m,n) for b in (result,control)]
                if one_hot:
                    np.testing.assert_array_equal(both[0],both[1])
                    np.testing.assert_array_equal(both[0],x.astype('f4')@w.T)
                else:
                    target=x.astype('f8')@w.astype('f8').T
                    norm=np.abs(x.astype('f8'))@np.abs(w.astype('f8')).T
                    for y in both:
                        assert np.max(np.abs(y-target)/np.maximum(norm,1e-10))<.0006
                    assert np.max(np.abs(both[0]-both[1])/np.maximum(norm,1e-10))<.0012
        finally:
            a.sync();destroy(h);old.destroy(oh)


@pytest.mark.parametrize('kind',SF_MODULES)
def test_sf_schedules(sdk,modules,kind):
    grouped='grouped' in kind;typed=kind.startswith('typed');bf='bf16' in kind
    rows=(3,0,5) if grouped else ((1,) if kind=='dense-ap1' else (8,))
    m=sum(rows);e=len(rows);n,k=256,1024
    _,q,s,z=data(n,k,e,98)
    planes=sf_gptq.pack(q,s,z,metadata='bf16' if grouped and bf else 'f16')
    w=golden(q,s,z,bf=bf)
    # Dense BF16's scale plane is FP16; the arithmetic boundary rounds the
    # product to BF16, not the scale first (activate one K coefficient only).
    if typed and bf:
        w=bf16_float(bf16_bits((q.astype('f4').reshape(e,n,k//32,32)-
            z.swapaxes(-1,-2)[...,None])*s.astype('f4').swapaxes(-1,-2)[...,None])).reshape(e,n,k)
    module=modules[kind];device=modules['dense'].device_identity();entry=module if typed else module.lib
    if typed or (bf and grouped):
        identity,query,prepare,run,destroy=bind_compute(entry,grouped)
        assert identity().contents.parent.contents.mapping_id==sf_gptq.GPTQ_W4_G32
    else:
        query,prepare,run,destroy=module.query,module.prepare,module.run,module.destroy
    with Arena(sdk) as a:
        stream=a.stream();idx=(np.arange(m)*127+17)%k
        x=np.zeros((m,k),'f4');x[np.arange(m),idx]=1
        src=a.upload(x if typed else bf16_bits(x) if bf else x.astype('f2'))
        dst=a.buffer(m*n*(4 if typed else 2));ps=[a.upload(planes[i]) for i in ('low','metadata','zero')]
        c=Call(1,C.sizeof(Call),m,n,k,e,32,device['ordinal'],device['compute_units'],sf_gptq.GPTQ_W4_G32)
        c.a,c.low,c.metadata,c.zero,c.output,c.stream=src.ptr,ps[0].ptr,ps[1].ptr,ps[2].ptr,dst.ptr,stream.value
        counts=(C.c_int*e)(*rows)
        if grouped:
            c.offsets_device=a.upload(np.array((0,*np.cumsum(rows)),'i4')).ptr
            if not bf:
                c.rows_host=C.cast(counts,C.c_void_p);c.rows_device=a.upload(np.array(rows,'i4')).ptr
        def wrap():
            if grouped and bf:
                return GroupedMetadataCall(4,C.sizeof(GroupedMetadataCall),
                    GroupedDeviceCall(2,C.sizeof(GroupedDeviceCall),c,max(rows),0),1,1)
            if typed:
                return DenseComputeCall(2,C.sizeof(DenseComputeCall),DenseIO(1,C.sizeof(DenseIO),c,1,1),int(bf))
            return c
        oracle=w[np.repeat(np.arange(e),rows),:,idx]
        for persistent,split in ((0,1),(0,2),(0,4),(0,8),(1,1),*(((1,2),) if grouped else ())):
            recipe=Recipe(1,C.sizeof(Recipe),persistent,split,persistent)
            info=Resources();h=C.c_void_p();wrapped=wrap()
            checked(query(ref(wrapped),ref(recipe),ref(info)),kind+' query')
            c.workspace=a.buffer(info.workspace_bytes).ptr if info.workspace_bytes else None
            c.workspace_bytes=info.workspace_bytes;wrapped=wrap();dst.poison();a.sync()
            try:
                checked(prepare(ref(wrapped),ref(recipe),ref(h)),kind+' prepare')
                checked(run(h,stream),kind+' run');a.sync()
                y=np.frombuffer(dst.bytes(),dtype='f4' if typed else 'u2' if bf else 'f2').reshape(m,n)
                np.testing.assert_array_equal(bf16_float(y) if bf and not typed else y,oracle)
            finally:
                a.sync();destroy(h)
