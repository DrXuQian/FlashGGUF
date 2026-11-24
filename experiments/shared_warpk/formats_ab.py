"""One Asys trace per q/projection/M: two warmed rotations, automatic first.

Separate event samples alternate arms and include the complete public call.
No selector changes, and no machine-independent speed claims.
"""
import argparse
import ctypes as C
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from ci.support.fixture import compare, round_compute
from ci.support.graph import Replay
from ci.support.graph_api import graph_bind
from ci.support.packing import bind as setup, device_identity
from quactlize.runtime.compiler import sha
from quactlize.runtime.native import SDK, checked
from tests.ppu.support import Arena, Public, QTYPES, bind, packed
from experiments.shared_warpk.projection import projection


def run(args):
    sdk = SDK(Path(os.environ['PPU_SDK']))
    package = Path(os.environ['QUACTLIZE_CI_PACKAGE'])
    public = Public(package); setup(sdk,public.lib); graph_bind(sdk)
    device = device_identity(sdk); l2 = C.c_int()
    attr = bind(sdk.lib,'hggcDeviceGetAttribute',[C.POINTER(C.c_int),C.c_int,C.c_int])
    checked(attr(C.byref(l2),38,device['ordinal']), 'L2 bytes')
    if l2.value <= 0: raise ValueError('device does not report L2 size')
    m,q,mode = args.m,args.q,args.mode
    e,topk,ch,n,k = {'dense':(1,1,1,2560,2048), 'moe-gate':(8,2,1,1024,2048),
                     'moe-down':(8,2,2,2048,512)}[mode]
    bf = e>1
    ids = ((3*np.arange(m)[:,None]+np.arange(topk))%e).astype('i4')
    active = np.unique(ids).size
    with Arena(sdk) as a, ExitStack() as st:
        arr,sz,original,w = packed(a,public,q,n,k,e,varied_groups=True)
        touched = int(sz.low_bytes)//e*active
        ring = 1+math.ceil(2.25*l2.value/touched)
        stream = a.stream()
        x = np.random.default_rng(110+q).uniform(-.25,.25,(m,ch,k)).astype('f4')
        inp, route, out = a.upload(x),a.upload(ids),a.buffer(m*topk*n*4)
        dt = 'bf16' if bf else 'f16'
        x16 = round_compute(x,dt).astype('f8')
        gold = np.empty((m,topk,n),'f8'); denom = np.empty_like(gold)
        for t in range(m):
            for s in range(topk):
                row,wt = x16[t,s%ch],w[ids[t,s]].astype('f8')
                gold[t,s] = row@wt.T; denom[t,s] = np.abs(row)@np.abs(wt).T
        prep = st.enter_context(projection(public,a,q,n,k,m,arr,sz,experts=e,topk=topk,channels=ch,bf=bf))
        arms=[[],[]];names=[];workspace=0
        for j in range(ring):
            planes=[]
            for source, size in zip(original,(sz.low_bytes,sz.high_bytes,sz.units_bytes)):
                if j and size:
                    dst=a.buffer(size)
                    checked(sdk.lib.hggcMemcpy(dst.ptr,source,size,3),'duplicate immutable plane D2D')
                    planes.append(dst.ptr)
                else: planes.append(source)
            for i in range(2):
                info,launch = prep(planes,bool(i));arms[i].append(launch)
                if j==0: names.append(info.config.decode())
                assert names[i]==info.config.decode()
                workspace=max(workspace,info.workspace_bytes)
        ws = a.buffer(workspace) if workspace else None
        begin,end = C.c_void_p(),C.c_void_p()
        for ev in (begin,end):
            checked(sdk.lib.hggcEventCreate(C.byref(ev)),'event create')
            st.callback(sdk.lib.hggcEventDestroy,ev)
        a.sync()
        for arm in arms:
            for j in (0,ring-1):
                out.poison();a.sync()
                arm[j](inp.ptr,route.ptr,out.ptr,ws,stream);a.sync()
                compare(np.frombuffer(out.bytes(),'f4').reshape(gold.shape),gold,denom)
        manifest=json.loads((package/'manifest.json').read_text())
        report=dict(q=q,projection=mode,m=m,n=n,k=k,experts=e,topk=topk,active_experts=int(active),
                    source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                    package_commit=manifest['build_info']['source_revision'],
                    package_sha256=sha(package/'libncp_quactlize.so'),sdk_release=manifest['build_info']['sdk_release'],
                    fixture=f'raw_weight:varied_groups:seed=913+17*expert+{q}',device=device,configs=names,
                    ring=ring,l2_bytes=l2.value,intervening_low_bytes=(ring-1)*touched,rows=[],
                    scope='ALTERNATING_PUBLIC_GRAPH_SCREEN_NOT_PRODUCTION_ADMISSION',performance_admitted=False)
        for rotated in (False,True):
            sequence=range(ring) if rotated else [0]*ring
            def launch(i):
                for j in sequence: arms[i][j](inp.ptr,route.ptr,out.ptr,ws,stream)
                return 0
            with ExitStack() as gs:
                graphs=[]
                for i in range(2):
                    g=Replay(sdk,stream,lambda i=i:launch(i),1);graphs.append(g);gs.callback(g.close)
                gs.callback(a.sync)
                samples=[[],[]]
                for r in range(6):
                    for i in ((0,1) if r%2==0 else (1,0)):
                        checked(graphs[i](),'prime complete rotation')
                        checked(sdk.lib.hggcEventRecord(begin,stream),'begin')
                        checked(graphs[i](),'whole-call rotation')
                        checked(sdk.lib.hggcEventRecord(end,stream),'end')
                        checked(sdk.lib.hggcEventSynchronize(end),'wait')
                        ms=C.c_float();checked(sdk.lib.hggcEventElapsedTime(C.byref(ms),begin,end),'elapsed')
                        if not math.isfinite(ms.value) or ms.value<=0:raise ValueError('invalid elapsed time')
                        samples[i].append(ms.value*1000/ring)
                med=list(map(statistics.median,samples))
                report['rows'].append(dict(cache='rotating' if rotated else 'same-address',samples_us=samples,
                    median_us=med,delta_pct=(med[1]/med[0]-1)*100))
                if rotated and args.profiler:
                    checked(graphs[0](),'prime trace');a.sync()
                    checked(sdk.lib.hggcProfilerStart(),'begin Asys')
                    try:
                        for g in graphs: checked(g(),'capture warmed public calls')
                        a.sync()
                    finally: checked(sdk.lib.hggcProfilerStop(),'end Asys')
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print('WARPK_FORMAT_AB',mode,q,m,'median_us(auto,warpk)=',
              [row['median_us'] for row in report['rows']],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--q',type=int,choices=QTYPES,required=True)
    p.add_argument('--mode',choices=('dense','moe-gate','moe-down'),required=True)
    p.add_argument('--m',type=int,choices=(1,2,3,4,5,6,7,8,9,16,17,128),default=1)
    p.add_argument('--profiler',action='store_true');p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
