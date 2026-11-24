"""Paired device-event screen, plus Asys capture of the same warmed calls.

GGUF includes SF expansion and F32 adapters. GPTQ compares prepared FP16
calls against the original same-geometry FlashQuant reader, not against its
best ever measured latency at another shape. No winning config is installed.
"""
import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.append(str(ROOT/'third_party/flashquant'))
import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize
from ci.support.packing import bind as bind_sdk,device_identity
from flashquant.affine import Library,Buffers,Info as FInfo,problem
from quactlize.runtime.compiler import sha
from quactlize.runtime.native import Call, SDK, checked
from quactlize.runtime.sf_gptq import pack, GPTQ_W4_G32
from tests.ppu.support import Arena,Public,Options,Params,Info,Candidate,Arrangement,Sizes,bind
from contextlib import ExitStack

ref=C.byref

def gguf_plans(n,k,m,count,a,stream,lib,stack):
    rng=np.random.default_rng(810)
    raw=rng.integers(0,256,(n,k//256,144),dtype='u1')
    raw[...,0:2]=np.array([2**-8],dtype='<f2').view('u1')
    raw[...,2:4]=np.array([2**-9],dtype='<f2').view('u1')
    source=a.upload(raw);arr=Arrangement();sz=Sizes()
    lib.check(lib.canonical(12,ref(arr)));lib.check(lib.sizes(n,k,1,12,ref(arr),ref(sz)))
    ctx=C.c_void_p();opts=Options(1,C.sizeof(Options),0,0);lib.check(lib.create(0,ref(opts),ref(ctx)))
    stack.callback(lib.destroy,ctx)
    p=Params(1,C.sizeof(Params),12,n,k,m,k,n,arr,sz)
    cand=Candidate(1,C.sizeof(Candidate),4,1,1,0,-1);cand.split=1
    idx=(np.arange(m)*127+33)%k
    x=np.zeros((m,k),'f4');x[np.arange(m),idx]=1
    oracle=np.stack([dequantize(raw[:,ki//256,:],GGMLQuantizationType(12)).reshape(n,256)[:,ki%256] for ki in idx])
    inp=a.upload(x);out=a.buffer(m*n*4);arms=[[],[]];names=[];a.sync()
    for slot in range(count):
        low,units=a.buffer(sz.low_bytes),a.buffer(sz.units_bytes)
        lib.check(lib.pack(source.ptr,low.ptr,None,units.ptr,n,k,1,12,ref(arr),stream));a.sync()
        for i in range(2):
            h=C.c_void_p();info=Info(1,C.sizeof(Info))
            args=(ctx,ref(p),low.ptr,None,units.ptr)
            lib.check(lib.candidate(*args,ref(cand),ref(h),ref(info)) if i else lib.prepare(*args,ref(h),ref(info)))
            if slot==0: names.append(info.config.decode())
            assert info.config.decode()==names[i]
            ws=a.buffer(info.workspace_bytes) if info.workspace_bytes else None
            arms[i].append(lambda h=h,ws=ws:lib.run(h,inp.ptr,out.ptr,ws.ptr if ws else None,ws.size if ws else 0,stream))
    return names,arms,inp,out,oracle,'f4'


def gptq_plans(n,k,m,count,a,stream,lib,stack,cu):
    rng=np.random.default_rng(811);ref_lib=Library(os.environ['FLASHQUANT_REFERENCE'])
    q=rng.integers(0,16,(n,k),dtype='u1')
    s=rng.uniform(.015,.12,(k//32,n)).astype('f2');z=rng.integers(0,16,s.shape,dtype='u1')
    data=pack(q,s,z);idx=(np.arange(m)*127+33)%k
    x=np.zeros((m,k),'f2');x[np.arange(m),idx]=1
    oracle=((q[:,idx].T.astype('i2')-z[idx//32]).astype('f4')*s[idx//32]).astype('f2')
    inp=a.upload(x);out=a.buffer(m*n*2)
    prep=bind(lib.lib,'quactlize_ppu_warpk_prepare_v1',[C.POINTER(Call),C.POINTER(C.c_void_p)])
    run=bind(lib.lib,'quactlize_ppu_warpk_run_v1',[C.c_void_p,C.POINTER(Call)])
    destroy=bind(lib.lib,'quactlize_ppu_warpk_destroy_v1',[C.c_void_p],None)
    shape=Call(1,C.sizeof(Call),m,n,k,1,32,0,cu,GPTQ_W4_G32)
    h=C.c_void_p();checked(prep(ref(shape),ref(h)),'warp-K prepare');stack.callback(destroy,h)
    # ID619 is the same TM8/TN64/TK256/WK64 integer-zero geometry.
    problem1=problem(m,n,k,config=619,zero_mode=1);info=FInfo();ref_lib.check(ref_lib.query(ref(problem1),ref(info)))
    arms=[[],[]]
    for _ in range(count):
        low,scale,center,zp=(a.upload(v) for v in (data['low'],data['metadata'],data['zero'],z))
        ws=a.buffer(max(256,info.workspace_bytes));a.sync()
        b=Buffers(1,C.sizeof(Buffers),inp.ptr,low.ptr,scale.ptr,zp.ptr,out.ptr,ws.ptr,info.workspace_bytes,stream)
        old=C.c_void_p();ref_lib.check(ref_lib.prepare(ref(problem1),ref(b),ref(old)));stack.callback(ref_lib.destroy,old)
        c=Call.from_buffer_copy(shape)
        c.a,c.low,c.metadata,c.zero,c.output,c.stream=inp.ptr,low.ptr,scale.ptr,center.ptr,out.ptr,stream.value
        arms[0].append(lambda old=old:ref_lib.run(old,stream))
        arms[1].append(lambda c=c:run(h,ref(c)))
    return ['flashquant:619','warpk:prepared-gptq'],arms,inp,out,oracle,'f2'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=25600);p.add_argument('--k',type=int,default=5120)
    p.add_argument('--m',type=int,nargs='+',default=[1,4,8,9]);p.add_argument('--rounds',type=int,default=6)
    p.add_argument('--family',choices=('gguf','gptq','both'),default='both')
    p.add_argument('--profiler',action='store_true');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();n,k=args.n,args.k
    if min(n,k,*args.m)<=0 or n%256 or k%256 or n*k>2**31-1 or args.rounds<2:
        p.error('expected positive legal dimensions and at least two rounds')
    if args.profiler and (len(args.m)!=1 or args.family=='both'):
        p.error('capture exactly one --m and one --family per .asysrep')
    sdk=SDK(Path(os.environ['PPU_SDK']));lib=Public(Path(os.environ['QUACTLIZE_CI_PACKAGE']))
    bind_sdk(sdk,lib.lib);device=device_identity(sdk)
    attr=bind(sdk.lib,'hggcDeviceGetAttribute',[C.POINTER(C.c_int),C.c_int,C.c_int])
    def get(i):
        value=C.c_int();checked(attr(ref(value),i,device['ordinal']),'device attribute')
        if value.value<=0:raise ValueError('device did not report a positive CU count/L2 size')
        return value.value
    l2,cu=get(38),get(16)
    ring=1+math.ceil(2.25*l2/(n*k//2)) # intervening low planes alone exceed 2.25 L2
    rows=[];report=dict(device=device,cu=cu,l2=l2,ring=ring,rounds=args.rounds,
        source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        reference_commit=json.loads(Path(os.environ['FLASHQUANT_REFERENCE']).with_name('manifest.json').read_text())['revision'],
        package_sha256=sha(Path(os.environ['QUACTLIZE_CI_PACKAGE'])/'libncp_quactlize.so'),
        reference_sha256=sha(Path(os.environ['FLASHQUANT_REFERENCE'])),performance_admitted=False,rows=rows)
    for m in args.m:
        for family in (('gguf','gptq') if args.family=='both' else (args.family,)):
            with Arena(sdk) as a, ExitStack() as stack:
                stream=a.stream()
                names,arms,inp,out,want,dt=(gguf_plans(n,k,m,ring,a,stream,lib,stack) if family=='gguf'
                    else gptq_plans(n,k,m,ring,a,stream,lib,stack,cu))
                begin,end=C.c_void_p(),C.c_void_p()
                for ev in (begin,end):
                    checked(sdk.lib.hggcEventCreate(ref(ev)),'event create');stack.callback(sdk.lib.hggcEventDestroy,ev)
                stack.callback(a.sync)
                for arm in arms:
                    for f in arm:
                        out.poison();a.sync();checked(f(),'warm');a.sync()
                        np.testing.assert_array_equal(np.frombuffer(out.bytes(),dt).reshape(m,n),want)
                inp.write(np.random.default_rng(812+m).uniform(-.25,.25,(m,k)).astype(dt));a.sync()
                for label,sequence in [('same-address',[0]*ring),('rotating',range(ring))]:
                    samples=[[],[]]
                    for rnd in range(args.rounds):
                        for i in ((0,1) if rnd%2==0 else (1,0)):
                            # Prime each arm outside its timed interval. Even at
                            # the A/B boundary slot 0 has a full ring of separation.
                            for j in sequence:checked(arms[i][j](),'prime')
                            checked(sdk.lib.hggcEventRecord(begin,stream),'start')
                            for j in sequence:checked(arms[i][j](),'call')
                            checked(sdk.lib.hggcEventRecord(end,stream),'end')
                            checked(sdk.lib.hggcEventSynchronize(end),'wait')
                            ms=C.c_float();checked(sdk.lib.hggcEventElapsedTime(ref(ms),begin,end),'elapsed')
                            if not math.isfinite(ms.value) or ms.value<=0:raise ValueError('invalid elapsed time')
                            samples[i].append(ms.value*1000/ring)
                    us=list(map(statistics.median,samples))
                    row=dict(family=family,m=m,n=n,k=k,cache=label,configs=names,us=us,samples_us=samples,
                             delta_pct=100*(us[1]/us[0]-1))
                    rows.append(row);args.output.write_text(json.dumps(report,indent=2)+'\n')
                    print('WARPK_AB',json.dumps({k:v for k,v in row.items() if k!='samples_us'}),flush=True)
                if args.profiler:
                    for f in arms[0]:checked(f(),'prime trace')
                    a.sync()
                    checked(sdk.lib.hggcProfilerStart(),'Asys begin')
                    # First full rotation = control; second = candidate. Enqueued
                    # on the same stream after setup with no downloads or resets.
                    for arm in arms:
                        for f in arm:checked(f(),'profiled call')
                    a.sync();checked(sdk.lib.hggcProfilerStop(),'Asys end')


if __name__=='__main__':main()
