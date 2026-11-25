"""Exercise the real public F32 endpoints; no kernel launches during prepare."""
import ctypes as C
from contextlib import contextmanager
from ci.support import dense, moe
from tests.ppu.support import bind


@contextmanager
def projection(public, arena, q, n, k, m, arr, sizes, *, experts=1, topk=1,
               channels=1, bf=False, padded=False):
    grouped = experts > 1
    in_stride, out_stride = k + 4 * padded, n + 8 * padded
    token_stride = (channels + padded) * in_stride
    ids_stride = topk + padded
    ctx = C.c_void_p()
    if grouped:
        if not bf:
            raise ValueError("the public MoE endpoint uses BF16 compute")
        api = moe.Public(public.lib)
        api.check(api.create(0, C.byref(ctx)))
    else:
        api = public
        options = dense.Options(1, C.sizeof(dense.Options), int(bf), 0)
        api.check(api.create(0, C.byref(options), C.byref(ctx)))
    try:
        if grouped:
            p = moe.Params(1, C.sizeof(moe.Params), q, n, k, experts, m, topk, channels, 0,
                           in_stride, token_stride, ids_stride, out_stride, arr, sizes)
        else:
            p = dense.Params(1, C.sizeof(dense.Params), q, n, k, m, in_stride, out_stride, arr, sizes)

        def prepare(planes, route):
            if route not in ('auto','sf','fq'):
                raise ValueError('unknown projection route: '+str(route))
            kind = moe.Info if grouped else dense.Info
            info, handle = kind(1, C.sizeof(kind)), C.c_void_p()
            args = (ctx, C.byref(p), *planes)
            tail = (C.byref(handle), C.byref(info))
            if route=='auto':
                rc = api.prepare(*args, *tail)
            elif grouped:
                name='prepare_fq_warpk_v1' if route=='fq' else 'prepare_warpk_v1'
                forced=bind(public.lib,'quactlize_ppu_moe_matmul_'+name,api.prepare.argtypes)
                rc = forced(*args, *tail)
            else:
                spec=dense.Candidate(1,C.sizeof(dense.Candidate),5 if route=='fq' else 4,
                                     int(route=='sf'),int(not bf),0,-1);spec.split=1
                rc = api.candidate(*args, C.byref(spec), *tail)
            api.check(rc)
            if route!='auto':
                assert b'warpk' in info.config and info.config.startswith(route.encode()+b':')
                assert info.route == int(route=='sf') and info.split == 1

            def launch(a, ids, c, ws, stream):
                args = (handle, a, ids, c) if grouped else (handle, a, c)
                rc = api.run(*args, ws.ptr if ws else None, ws.size if ws else 0, stream)
                api.check(rc)
                return rc
            return info, launch
        yield prepare
    finally:
        try:
            arena.sync()
        finally:
            api.destroy(ctx)
