"""PPU-only: public GGUF projections and prepared grouped W4/G32 GPTQ."""
import ctypes as C
from contextlib import ExitStack

import numpy as np
import pytest
from ci.support.fixture import bf16_bits, bf16_float, compare, round_compute
from ci.support.graph import Replay
from quactlize.decode.native_compute import GroupedDeviceCall, GroupedMetadataCall
from quactlize.runtime import sf_gptq
from quactlize.runtime.native import Call, checked
from tests.ppu.conftest import package, sdk
from tests.ppu.support import Arena, Public, QTYPES, bind, packed
from experiments.shared_warpk.projection import projection

MS = (*range(1, 10), 16, 17, 128)
ref = C.byref


def activations(rows, k, step, bf):
    x = np.zeros((rows, k), 'f4')
    r = np.arange(rows)
    if step == 0:
        boundaries = [0, 7, 15, 31, 63, 64, 127, 128, 191, 192, 255, 256, k-1]
        x[r, np.resize(boundaries, rows) % k] = 1
    elif step == 1:
        # All four K peers and at least two pipeline stages contribute.
        for peer, sign in enumerate((1, -1, -.5, .5)):
            x[r, ((r * 17 % 64) + 64 * peer + 256 * ((r + peer) % (k // 256)))] = sign
    else:
        x = np.random.default_rng(733 + rows).uniform(-.25, .25, (rows, k)).astype('f4')
        if bf:  # detects accidental use of F16 A
            x[0, k-1] = 262144
    return x


def ids_for(m, topk, experts, step):
    t, s = np.arange(m)[:, None], np.arange(topk)[None, :]
    # Distinct per token, but repeated across tokens. Middle/edge experts can be empty.
    return (s + (experts-1 if step == 0 else 3*t+1 if step == 1 else t//7+2)) % experts


def oracle(y, x, owners, w, bf, exact):
    dt = 'bf16' if bf else 'f16'
    for e in np.unique(owners):
        match = owners == e
        a = round_compute(x[match], dt).astype('f8')
        b = w[e].astype('f8')
        want = a @ b.T
        if exact:  # 1 or 4 dyadic terms: FP32 summation is exact
            np.testing.assert_array_equal(y[match], round_compute(want, dt))
        else:
            compare(y[match], want if bf else round_compute(want, dt), np.abs(a) @ np.abs(b).T)


def public_case(sdk, public, q, n, k, m, weights, *, bf, experts, channels, topk=None):
    arr, sizes, planes, w = weights
    if topk is None:
        topk = 1 if experts == 1 else 2
    # Both group modes see strided A/IDs/C. Padding is poisoned on input and checked on output.
    with Arena(sdk) as a, projection(public, a, q, n, k, m, arr, sizes,
            experts=experts, topk=topk, channels=channels, bf=bf, padded=True) as prepare, ExitStack() as graph_stack:
        info, submit = prepare(planes, True)
        assert info.compute_type == int(bf)
        ws = a.buffer(info.workspace_bytes)
        stream = a.stream()
        inp = a.buffer(m * (channels+1 if experts > 1 else 1) * (k+4) * 4)
        ids = a.buffer(m * (topk+1) * 4)
        out = a.buffer(m * topk * (n+8) * 4)
        launch = lambda: submit(inp.ptr, ids.ptr, out.ptr, ws, stream)
        graph = None
        for step in range(3):
            route = ids_for(m, topk, experts, step).astype('i4')
            ix = np.arange(m)[:, None]*channels + np.arange(topk)[None, :] % channels
            x = activations(m*channels, k, step, bf)
            padded = np.full((m, channels+1 if experts > 1 else 1, k+4), np.nan, 'f4')
            padded[:, :channels, :k] = x.reshape(m, channels, k)
            inp.write(padded)
            ids.write(np.pad(route, ((0,0),(0,1)), constant_values=-113))
            out.write(np.full((m*topk, n+8), -789, 'f4'))
            ws.poison(); a.sync()
            if step == 1:
                graph = Replay(sdk, stream, launch, 1)
                graph_stack.callback(graph.close)
                graph_stack.callback(a.sync)
            checked(graph() if graph else launch(), 'GGUF projection')
            a.sync()
            actual = np.frombuffer(out.bytes(), 'f4').reshape(-1, n+8)
            assert np.all(actual[:, n:] == -789)
            oracle(actual[:, :n], x[ix.ravel()], route.ravel(), w, bf, step < 2)
            ws.guard()


@pytest.mark.parametrize('q', QTYPES)
@pytest.mark.parametrize('bf,experts,channels', [(False,1,1), (True,1,1), (True,4,1), (True,4,2)],
                         ids=['dense-f16','dense-bf16','moe-gate','moe-down'])
def test_gguf_projections(package, sdk, q, bf, experts, channels):
    public = Public(package)
    with Arena(sdk) as weights:
        w = packed(weights, public, q, 256, 512, experts, varied_groups=True)
        for m in MS:
            try:
                public_case(sdk, public, q, 256, 512, m, w, bf=bf, experts=experts, channels=channels)
            except Exception as exc:
                raise AssertionError(f'q={q} M={m} E={experts} channels={channels} bf16={bf}') from exc


@pytest.mark.parametrize('q,n,k,channels', [(13,1024,2048,1), (14,2048,512,8)])
def test_projection_model_dimensions(package, sdk, q, n, k, channels):
    public = Public(package)
    with Arena(sdk) as weights:
        w = packed(weights, public, q, n, k, 8, varied_groups=True)
        public_case(sdk, public, q, n, k, 17, w, bf=True, experts=8, channels=channels, topk=8)


@pytest.mark.parametrize('q',(8,10,12,13))
def test_one_k_block(package, sdk, q):
    public=Public(package)
    with Arena(sdk) as weights:
        w=packed(weights,public,q,256,256,varied_groups=True)
        public_case(sdk,public,q,256,256,9,w,bf=True,experts=1,channels=1)


@pytest.mark.parametrize('bf', [False, True])
@pytest.mark.parametrize('asym', [False, True], ids=['sym','zp'])
def test_grouped_prepared_gptq(package, sdk, bf, asym):
    lib = Public(package).lib
    ptr, dp = C.c_void_p, C.POINTER(GroupedMetadataCall)
    prep = bind(lib, 'quactlize_ppu_warpk_prepare_v2', [C.c_int, dp, C.POINTER(ptr)])
    run = bind(lib, 'quactlize_ppu_warpk_run_v2', [ptr, dp])
    destroy = bind(lib, 'quactlize_ppu_warpk_destroy_v2', [ptr], None)
    cu = C.c_int()
    get = bind(sdk.lib, 'hggcDeviceGetAttribute', [C.POINTER(C.c_int), C.c_int, C.c_int])
    checked(get(ref(cu), 16, 0), 'CU count')
    m, n, k, e = 35, 256, 1024, 4
    dt = 'bf16' if bf else 'f16'
    rng = np.random.default_rng(881)
    q = rng.integers(0,16,(e,n,k),dtype='u1')
    s = rng.uniform(.01,.087,(e,k//32,n)).astype('f2')
    z = rng.integers(0,16,s.shape,dtype='u1') if asym else None
    planes = sf_gptq.pack(q, s, z, metadata=dt)
    center = 8 if z is None else z.swapaxes(1,2)[...,None]
    w = round_compute((q.astype('f4').reshape(e,n,k//32,32)-center) *
                      round_compute(s,dt).swapaxes(1,2)[...,None],dt).reshape(e,n,k)
    c = Call(1,C.sizeof(Call),m,n,k,e,32,0,cu.value,planes['mapping_id'])
    d = GroupedMetadataCall(4,C.sizeof(GroupedMetadataCall),
                            GroupedDeviceCall(2,C.sizeof(GroupedDeviceCall),c,m,0),int(bf),int(bf))
    h = ptr()
    assert prep(13,ref(d),ref(h)) != 0 and not h  # W4 GPTQ is never a Q5_K plane set
    with Arena(sdk) as a, ExitStack() as st:
        checked(prep(12,ref(d),ref(h)), 'prepare grouped GPTQ')
        st.callback(destroy,h)
        stream = a.stream()
        c = d.device_call.call
        inp, out, off = a.buffer(m*k*2), a.buffer(m*n*2), a.buffer((e+1)*4)
        c.low,c.metadata,c.zero = (a.upload(planes[key]).ptr for key in ('low','metadata','zero'))
        c.a,c.output,c.offsets_device,c.stream = inp.ptr,out.ptr,off.ptr,stream.value
        bad = GroupedMetadataCall.from_buffer_copy(d); bad.device_call.call.zero = None
        assert run(h,ref(bad)) != 0
        bad = GroupedMetadataCall.from_buffer_copy(d); bad.device_call.call.output = c.a
        assert run(h,ref(bad)) != 0
        launch = lambda: run(h,ref(d))
        graph = None
        for step, counts in enumerate(((0,2,0,33),(35,0,0,0),(8,0,17,10))):
            x = activations(m,k,step,bf)
            inp.write(bf16_bits(x) if bf else x.astype('f2'))
            off.write(np.array([0,*np.cumsum(counts)],'i4'));out.poison();a.sync()
            if step == 1:
                graph = Replay(sdk,stream,launch,1);st.callback(graph.close);st.callback(a.sync)
            checked(graph() if graph else launch(),'GPTQ projection');a.sync()
            y = np.frombuffer(out.bytes(),'u2' if bf else 'f2').reshape(m,n)
            # Random, non-dyadic GPTQ metadata; only one-hot is exact independent of accumulation order.
            oracle(bf16_float(y) if bf else y,x,np.repeat(np.arange(e),counts),w,bf,step == 0)
