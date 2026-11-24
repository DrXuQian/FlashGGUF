"""Prepare FlashQuant-style group32 W4 operands for the SF module ABI.

Input is *logical unpacked* uint4 codes and zero points, not serialized
AutoGPTQ qweight/qzeros/g_idx. The caller resolves any checkpoint zero
convention or permutation first. No GGUF superblocks or re-quantization.
"""
from .compiler import materialize_parent
from .tuning import GPTQ_W4_G32


def parent(sf_parent):
    """Use the same tiling/schedule/reducer with a distinct weight contract."""
    base=materialize_parent(sf_parent)
    if base.get("weight_style")=="gptq":
        return base
    return materialize_parent(dict(base, symbol="gptq_"+base["symbol"], weight_style="gptq"))


def pack(codes, scales, zeros=None, *, metadata="f16"):
    """[E,]N,K -> b16 [E,]K/4,N; S and signed center are [E,]K/32,N.

    metadata=f16 for FlashQuant parity, native dense/typed dense and grouped
    F16. Use bf16 only for the explicit grouped BF16-metadata ABI v4. Returned
    b16 planes are unsigned *bit storage*, not integers used in arithmetic.
    """
    import numpy as np
    q,s=np.asarray(codes),np.asarray(scales)
    if q.ndim not in (2,3) or q.dtype!=np.uint8:
        raise ValueError("codes must be uint8 [N,K] or [E,N,K]")
    n,k=q.shape[-2:]; lead=q.shape[:-2]
    if min(q.shape)<=0 or n%256 or k%256 or q.max()>15:
        raise ValueError("expected uint4 codes and positive N/K multiples of 256")
    if s.dtype!=np.float16 or s.shape!=(*lead,k//32,n) or not np.isfinite(s).all():
        raise ValueError("scales must be finite FP16 [E,]K/32,N")
    if zeros is None:
        center=np.zeros(s.shape,dtype=np.float16)  # symmetric zp=8
    else:
        z=np.asarray(zeros)
        if z.dtype!=np.uint8 or z.shape!=s.shape or z.max()>15:
            raise ValueError("zeros must be uint8 [E,]K/32,N in [0,15]")
        center=(8-z.astype(np.int16)).astype(np.float16)
    if metadata not in ("f16","bf16"):
        raise ValueError("metadata must be f16 or bf16")
    def bits(x):
        if metadata=="f16":
            return np.ascontiguousarray(x,dtype="<f2").view("<u2")
        b=x.astype("<f4").view("<u4")
        return np.ascontiguousarray(((b+0x7fff+((b>>16)&1))>>16).astype("<u2"))
    lanes=q.reshape(*lead,n,k//32,4,8)
    w=lanes[...,0,:].astype("<u2")
    for i in (1,2,3):
        w |= lanes[...,i,:].astype("<u2") << (4*i)
    return dict(mapping_id=GPTQ_W4_G32,
                low=np.ascontiguousarray(w.reshape(*lead,n,k//4).swapaxes(-1,-2)),
                metadata=bits(s),zero=bits(center))
