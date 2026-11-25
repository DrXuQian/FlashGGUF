"""Byte-domain oracle: the input is the original GGUF, never emitted PPU planes."""
import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize
from reference import gguf_kpack as ref
from ci.support.fixture import raw_weight, round_compute


def fixture(q,n,k,seed):
    raw,_=raw_weight(q,n,k,seed)
    spec=ref.SPECS[q]
    ri,bi=np.indices(raw.shape[:2]);tags=(ri+3*bi+seed)%7
    ds=np.array([.013693,-.05673,2**-21,.1261,-.001241,.2137,.06353],'f2')
    raw[...,spec.d_offset:spec.d_offset+2]=ds[tags].view('u1').reshape(n,-1,2)
    if spec.has_min:
        raw[...,spec.dmin_offset:spec.dmin_offset+2]=ds[(tags+3)%7].view('u1').reshape(n,-1,2)
    width=16 if q in (10,14) else 12
    def fields(unit,t,g):
        ref._metadata_put(unit,0,spec,g,(17*g+53*t+5)%(1<<spec.scale_bits),
                          (11*g+7*t+1)%(1<<spec.min_bits))
    tables=[]
    for t in range(7):
        group=bytearray(spec.raw_bytes)
        for g in range(spec.groups):fields(group,t,g)
        tables.append(np.frombuffer(group,'u1')[spec.scale_offset:spec.scale_offset+width])
    raw[...,spec.scale_offset:spec.scale_offset+width]=np.stack(tables)[tags]
    return raw


def oracle(raw,q,bf):
    sp=ref.SPECS[q]; n,nsb,_=raw.shape;dt='bf16' if bf else 'f16'
    rnd=lambda x:round_compute(x,dt)
    def header(off):return raw[...,off:off+2].copy().view('f2').reshape(n,nsb).astype('f4')
    d=header(sp.d_offset);dm=header(sp.dmin_offset) if sp.has_min else None
    correction={10:0,11:-4,12:8,13:8,14:-24}[q]
    scales=np.empty((n,nsb,sp.groups),'f4');zeros=np.empty_like(scales)
    unit=bytearray(sp.raw_bytes)
    for g in range(sp.groups):
        sc,mn=ref._metadata_codes(raw.reshape(-1,sp.raw_bytes).T,0,sp,g)
        sc=sc.view('i1').astype('f4') if q==14 else sc.astype('f4')-(32 if q==11 else 0)
        s=rnd(d*sc.reshape(n,nsb))
        z=rnd(-(dm*mn.astype('f4').reshape(n,nsb))) if sp.has_min else np.zeros_like(s)
        scales[:,:,g]=s;zeros[:,:,g]=rnd(z+rnd(correction*s)) if correction else z
        # Ask gguf-py for the signed weight code alone, retaining all LOW/HIGH bits.
        ref._metadata_put(unit,0,sp,g,33 if q==11 else 1,0)
    codes=raw.copy(); width=16 if q in (10,14) else 12
    codes[...,sp.scale_offset:sp.scale_offset+width]=np.frombuffer(unit,'u1')[sp.scale_offset:sp.scale_offset+width]
    codes[...,sp.d_offset:sp.d_offset+2]=np.array([1],'f2').view('u1')
    if sp.has_min:codes[...,sp.dmin_offset:sp.dmin_offset+2]=0
    code=dequantize(codes.ravel(),GGMLQuantizationType(q)).reshape(n,nsb,sp.groups,-1)-correction
    weights=rnd(rnd(code*scales[...,None])+zeros[...,None]).reshape(n,-1)
    return scales.reshape(n,-1).T,zeros.reshape(n,-1).T,weights
