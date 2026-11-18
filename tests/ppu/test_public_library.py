"""Real-device public ABI coverage; run via run_tests.sh --suite accuracy.

No model, llama.cpp checkout, six-library bundle, timing sweep or profiler.
"""
import ctypes as C
import json
import os

import numpy as np
import pytest

from ci.support.graph import Replay
from .support import (Arena, Candidate, DENSE_M, Info, Options, Params, Public,
                      QTYPES, check_output, packed)
from quactlize.dispatch.dense_prefill import EVIDENCE
from quactlize.execution.token_cta_policy import load as token_choices


@pytest.fixture(scope="module")
def affine_library(request):
    from quactlize.affine.native import Library
    # Explicit diagnostic builds are useful for CPU-only native ABI checks.
    # Product accuracy always resolves its verified public package.
    path=os.environ.get("QUACTLIZE_AFFINE_TEST_LIBRARY")
    if path and os.environ.get("QUACTLIZE_CI_SUITE")=="accuracy":
        pytest.fail("product accuracy must use its public package, not a diagnostic affine DSO")
    if not path:
        path=request.getfixturevalue("package")/"libncp_quactlize.so"
    return Library(path)


def test_affine_native_specializations(affine_library):
    from quactlize.affine.native import Info as AffineInfo, problem
    from quactlize.affine import tuning
    library=affine_library
    checked=0
    for config in library.configurations():
        index=config["id"]
        if index not in tuning.SPECIALIZATIONS:
            continue
        sn,sk=tuning.SPECIALIZATIONS[index]
        n,k=sn or 1024,sk or 3072
        workers=tuning.STREAM_K_WORKERS.get(index,0)
        m=tuning.STATIC_M.get(index,0) or (512 if workers else 16)
        mode=config["zero_mode"]
        split=tuning.GLOBAL_SLICES.get(index,1)
        p=problem(m,n,k,config=index,split=split,zero_mode=mode)
        info=AffineInfo()
        library.check(library.query(C.byref(p),C.byref(info)))
        assert info.threads>0 and 0<info.shared_bytes<=262144
        wrong_mode=problem(m,n,k,config=index,split=split,zero_mode=1-mode)
        assert library.query(C.byref(wrong_mode),C.byref(AffineInfo()))!=0
        assert b"quantization mode" in library.error()
        if tuning.STATIC_M.get(index,0):
            rejected=problem(m+1,n,k,config=index,split=split,zero_mode=mode)
            assert library.query(C.byref(rejected),C.byref(AffineInfo()))!=0
            assert b"specialized M" in library.error()
        elif split==1 and not workers:
            # An explicit zero denotes dynamic M, just like an absent entry.
            dynamic=problem(m+1,n,k,config=index,split=split,zero_mode=mode)
            library.check(library.query(C.byref(dynamic),C.byref(AffineInfo())))
        for wrong_n,wrong_k in ([(n+256,k)] if sn else [])+([(n,k+512)] if sk else []):
            rejected=problem(m,wrong_n,wrong_k,config=index,split=split,zero_mode=mode)
            assert library.query(C.byref(rejected),C.byref(AffineInfo()))!=0
            assert b"specialized" in library.error()
        checked+=1
    assert checked>0,"the supplied native build contains no specialized kernels"


def test_affine_native_default_selection(affine_library):
    from quactlize.affine.native import Info as AffineInfo, Problem, problem
    library=affine_library
    available={config["id"]:config for config in library.configurations()}
    for mode in (0,1):
        for m,n,k in ((1,1024,3072),(12,2048,3072),(512,2048,3072),
                       (7,512,256),(24,3072,8192),(128,8192,8192)):
            p=problem(m,n,k,zero_mode=mode)
            selected=library.selected_config(p)
            assert available[selected["id"]]["zero_mode"]==mode
            automatic,explicit=AffineInfo(),AffineInfo()
            library.check(library.query(C.byref(p),C.byref(automatic)))
            forced=Problem.from_buffer_copy(p)
            forced.config=selected["id"]
            library.check(library.query(C.byref(forced),C.byref(explicit)))
            assert bytes(automatic)==bytes(explicit)
            assert p.config==-1


@pytest.mark.parametrize("mode",(0,1),ids=("symmetric","integer-zero"))
@pytest.mark.parametrize("shape",((7,512,256),(24,512,512),(128,512,256),
                                 (12,2048,3072),(512,2048,3072),(16,3072,1024)))
def test_affine_public_output_and_changed_graph(affine_library,sdk,mode,shape,record_property):
    from quactlize.affine import Buffers as AffineBuffers, Info as AffineInfo, problem
    library=affine_library
    m,n,k=shape
    rng=np.random.default_rng(1009+mode+m+n+k)
    codes=rng.integers(0,16,(n,k),dtype=np.uint8)
    scales=rng.uniform(.005,.025,(k//32,n)).astype(np.float16)
    zeros=rng.integers(0,16,(k//32,n),dtype=np.uint8) if mode else None
    center=zeros.T[...,None].astype(np.float32) if mode else 8
    golden=((codes.reshape(n,k//32,32).astype(np.float32)-center)*
            scales.T[...,None].astype(np.float32)).astype(np.float16).reshape(n,k).astype(np.float32)
    packed=np.empty(n*k//2,dtype=np.uint8)
    library.check(library.pack_host(codes.ctypes.data,n,k,packed.ctypes.data,packed.nbytes))
    p=problem(m,n,k,zero_mode=mode)
    selected=library.selected_config(p)
    info=AffineInfo()
    library.check(library.query(C.byref(p),C.byref(info)))
    record_property("affine_selected_config",str(selected["id"]))
    record_property("affine_checked_cells_per_replay",str(m*n))
    with Arena(sdk) as arena:
        weights=arena.upload(packed)
        scale=arena.upload(scales)
        zero=arena.upload(zeros) if mode else None
        inputs=arena.buffer(m*k*2)
        outputs=arena.buffer(m*n*2)
        workspace=arena.buffer(max(256,info.workspace_bytes))
        stream=arena.stream()
        arena.sync()
        buffers=AffineBuffers(1,C.sizeof(AffineBuffers),inputs.ptr,weights.ptr,scale.ptr,
                              zero.ptr if zero else None,outputs.ptr,workspace.ptr,
                              info.workspace_bytes,stream)
        handle=C.c_void_p()
        graph=None
        try:
            library.check(library.prepare(C.byref(p),C.byref(buffers),C.byref(handle)))
            if mode:
                # Integer-zero preparations own both metadata copies.
                scale.poison()
                zero.poison()
            def launch():
                return library.run(handle,stream)
            for repeat in range(3):
                source=rng.uniform(-.125,.125,(m,k)).astype(np.float16)
                inputs.write(source)
                outputs.poison()
                arena.sync()
                if repeat==1:
                    graph=Replay(sdk,stream,launch,1)
                library.check(graph() if graph else launch())
                arena.sync()
                got=np.frombuffer(outputs.bytes(),dtype=np.float16).reshape(m,n).astype(np.float32)
                reference=source.astype(np.float32)@golden.T
                assert np.isfinite(got).all()
                error=got-reference
                assert np.linalg.norm(error)/max(np.linalg.norm(reference),1e-12)<.001
                dot_norm=np.abs(source.astype(np.float32))@np.abs(golden).T
                assert np.max(np.abs(error)/np.maximum(dot_norm,1e-12))<.0005
                for allocation in arena.buffers:
                    allocation.guard()
        finally:
            arena.sync()
            if graph:
                graph.close()
            library.destroy(handle)


def test_affine_native_workspace_guards(affine_library,record_property):
    from quactlize.affine.native import Buffers as AffineBuffers, Info as AffineInfo, problem
    from quactlize.affine import tuning
    library=affine_library
    checked,fused=[],[]
    for config in library.configurations():
        index=config["id"]
        splits=tuning.GLOBAL_SLICES.get(index,1)
        workers=tuning.STREAM_K_WORKERS.get(index,0)
        n,k=tuning.SPECIALIZATIONS.get(index,(1024,3072))
        n,k=n or 1024,k or 3072
        m=tuning.STATIC_M.get(index,0) or (512 if workers else 16)
        mode=config["zero_mode"]
        p=problem(m,n,k,config=index,split=splits,zero_mode=mode)
        info=AffineInfo()
        library.check(library.query(C.byref(p),C.byref(info)))
        if not info.workspace_bytes:
            continue
        assert info.workspace_bytes<=64*1024*1024
        if index in tuning.PAIRED_SCALES:
            assert n*k//16<=info.workspace_bytes
        for field in ("input","weight","scales")+(("zeros",) if mode else ()):
            pointers=dict(input=0x10000000,weight=0x20000000,scales=0x30000000,
                          zeros=0x60000000 if mode else None)
            buffers=AffineBuffers(1,C.sizeof(AffineBuffers),pointers["input"],pointers["weight"],
                                  pointers["scales"],pointers["zeros"],0x40000000,pointers[field],
                                  info.workspace_bytes,None)
            handle=C.c_void_p()
            assert library.prepare(C.byref(p),C.byref(buffers),C.byref(handle))!=0
            assert not handle.value and b"workspace aliases" in library.error()
        buffers=AffineBuffers(1,C.sizeof(AffineBuffers),0x10000000,0x20000000,0x30000000,
                              0x60000000 if mode else None,0x40000000,0x50000000,
                              info.workspace_bytes-1,None)
        handle=C.c_void_p()
        assert library.prepare(C.byref(p),C.byref(buffers),C.byref(handle))!=0
        assert not handle.value and b"insufficient" in library.error()
        buffers.workspace_bytes=info.workspace_bytes
        buffers.zeros=None if mode else 0x60000000
        assert library.prepare(C.byref(p),C.byref(buffers),C.byref(handle))!=0
        assert not handle.value and b"invalid W4A16 pointers" in library.error()
        if workers or splits>1:
            partial_bytes=2*workers*config["tm"]*config["tn"]*4 if workers else splits*m*n*4
            assert partial_bytes<info.workspace_bytes
            if workers:
                larger=problem(1024,n,k,config=index,split=splits,zero_mode=mode)
                larger_info=AffineInfo()
                library.check(library.query(C.byref(larger),C.byref(larger_info)))
                assert larger_info.workspace_bytes==info.workspace_bytes
                if config["tm"]>=128:
                    undersupplied=problem(1,n,k,config=index,split=splits,zero_mode=mode)
                    assert library.query(C.byref(undersupplied),C.byref(AffineInfo()))!=0
                    assert b"contributors" in library.error()
            else:
                oversized=problem(config["tm"]+1,n,k,config=index,split=splits,zero_mode=mode)
                assert library.query(C.byref(oversized),C.byref(AffineInfo()))!=0
                assert b"one M tile" in library.error()
            fused.append(index)
        checked.append(index)
    assert checked,"the supplied native build contains no workspace-backed kernels"
    record_property("affine_workspace_configs",json.dumps(checked))
    record_property("affine_fused_partition_configs",json.dumps(fused))


def helper_followup_points():
    # Immutable measured scope, selected through the public ABI, not a forced
    # experimental launcher. Includes both retained controls as well as winners.
    data=json.loads(EVIDENCE.read_text())
    review='d3eec044e46f3beff68dc1fc5ce6371faaed7232aa73e673d91fb81f7e1f4128'
    historical=[p for p in data.get('superseded_points',data['points'])
                if p.get('admission',{}).get('review_sha256')==review]
    keys={p['key'] for p in historical}
    points=[p for p in data['points'] if p['key'] in keys]
    assert len(points)==15, 'helper followup public coverage changed'
    return [(p['shape'],data['recipes'][p['winner']]) for p in points]


def m_span_boundary_points():
    # Exercise public preparation at each new high-M boundary without another
    # performance sweep or a model. Prefer small matrices for routine CI.
    data=json.loads(EVIDENCE.read_text())
    points=[min((p for p in data['points'] if p['shape'][0]==q and p['shape'][1]==m),
                key=lambda p:(p['shape'][2]*p['shape'][3],p['key']))
            for q,m in ((8,3072),(12,5120),(14,8192))]
    return [(p['shape'],data['recipes'][p['winner']]) for p in points]


def middle_m_points():
    # A bounded public-ABI smoke for both newly admitted TC routes at each
    # transition M. Exhaustive selection replay remains host-only.
    data=json.loads(EVIDENCE.read_text())
    points=[min((p for p in data['points'] if p['shape'][1]==m and
                 data['recipes'][p['winner']]['route']==route),
                key=lambda p:(p['shape'][2]*p['shape'][3],p['key']))
            for m in (16,32,64) for route in (0,1)]
    return [(p['shape'],data['recipes'][p['winner']]) for p in points]


def concat_prefill_points():
    # Exact merged gate/up projections; activation and full FFN are separate
    # contracts. Small-M coverage is already driven by token_choices below.
    data = json.loads(EVIDENCE.read_text())
    points = [p for p in data['points'] if 'model_confirmation' in p]
    assert len(points) == 69
    return [(p['shape'],data['recipes'][p['winner']]) for p in points]


@pytest.mark.parametrize("q", QTYPES)
@pytest.mark.parametrize("pair", [False, True])
def test_device_pack_all_formats_and_experts(package, sdk, q, pair):
    with Arena(sdk) as arena:
        packed(arena, Public(package), q, experts=3, pair=pair, bit_patterns=True)


@pytest.mark.parametrize("q", QTYPES)
@pytest.mark.parametrize("m", DENSE_M)
@pytest.mark.parametrize("compute", [0, 1], ids=["f16", "bf16"])
def test_selected_dense_dynamic_bindings(package, sdk, q, m, compute, record_property):
    run_dense(package, sdk, q, m, compute, record_property)


@pytest.mark.parametrize('row',token_choices()['rows'],ids=lambda r:
    f'q{r["key"][0]}-n{r["key"][2]}-k{r["key"][3]}-m{r["key"][7]}')
def test_token_cta_exact_public_bindings(package,sdk,row,record_property):
    # Verify the final public execution path, including unchanged Q4/TC arms,
    # two streams, moved addresses, padded strides, reducer and Graph replay.
    from ci.support.typed import DenseLibrary
    from quactlize.dispatch.table import Workload
    query=DenseLibrary(package/'quactlize-runtime/libquactlize_ppu_prefill.so')
    policy_device=query.choice(Workload(12,1024,5120,2048),0,allow_miss=True) is not None
    q,mode,n,k,e,top,ch,m,compute=row['key']
    run_dense(package,sdk,q,m,compute,record_property,n=n,k=k,
              expected_decode=row['selected'],policy_device=policy_device)


@pytest.mark.parametrize("q", [10, 11, 12, 13, 14])
def test_full_dequant_provider_ordinary_calls(package, sdk, q, record_property):
    # This is not a cuBLAS graph/timer test. Explicit route admission is logged.
    run_dense(package, sdk, q, 128, 1, record_property, full=True)


@pytest.mark.parametrize('shape,recipe',helper_followup_points()+m_span_boundary_points()+middle_m_points()+concat_prefill_points(),
                         ids=lambda x: str(x) if isinstance(x,list) else '')
def test_admitted_helper_followup_public_calls(package,sdk,shape,recipe,record_property):
    from ci.support.typed import DenseLibrary
    from quactlize.dispatch.table import Workload
    # ZW610/ZW810E reuse the ZW810 exact table, not local measurements. A
    # helper-policy hit checks compatibility, not which device was measured.
    query=DenseLibrary(package/'quactlize-runtime/libquactlize_ppu_prefill.so')
    known=Workload(qtype=12,m=2048,n=1024,k=5120,operation='dense',compute='f16',storage='f32')
    policy_device=query.choice(known,0,allow_miss=True) is not None
    record_property('selection_domain','ZW810_EXACT_TABLE' if policy_device else 'OTHER_DEVICE_FALLBACK')
    q,m,n,k=shape
    run_dense(package,sdk,q,m,0,record_property,n=n,k=k,expected=recipe,policy_device=policy_device)


def run_dense(package, sdk, q, m, compute, record_property, *, full=False,n=256,k=512,expected=None,policy_device=True,expected_decode=None):
    lib = Public(package)
    with Arena(sdk) as arena:
        arr, sizes, weight_ptrs, gold = packed(arena, lib, q,n=n,k=k)
        n, k = gold.shape[1:]
        padding = 8 if m <= 8 else 0
        p = Params(1, C.sizeof(Params), q, n, k, m, k + padding, n + padding, arr, sizes)
        admission = Params.from_buffer_copy(p); admission.m = 0
        assert lib.supported(C.byref(admission)) == 1
        assert lib.supported(C.byref(p)) == 1
        admission.qtype = 9
        assert lib.supported(C.byref(admission)) == 0
        context, handle = C.c_void_p(), C.c_void_p()
        options, info = Options(1, C.sizeof(Options), compute, 0), Info(1, C.sizeof(Info))
        lib.check(lib.create(0, C.byref(options), C.byref(context)))
        graph = None
        try:
            if full:
                candidate = Candidate(1, C.sizeof(Candidate))
                candidate.kind = candidate.route = 3
                candidate.dequant_config = 5
                lib.check(lib.candidate(context, C.byref(p), *weight_ptrs, C.byref(candidate), C.byref(handle), C.byref(info)))
                assert info.route == 3 and info.compute_type == 1
            else:
                lib.check(lib.prepare(context, C.byref(p), *weight_ptrs, C.byref(handle), C.byref(info)))
            record_property("plan", f"route={info.route} compute={info.compute_type} split={info.split} config={info.config.decode()}")
            assert handle.value and info.config and info.split >= 1
            if expected_decode and policy_device:
                f=expected_decode['config'];impl=expected_decode['implementation']
                if f['kind']=='tc':
                    assert info.route==f['route'] and info.split==f['split']
                    assert info.config.decode()==f['symbol']
                    if f.get('legacy_adapters'):
                        assert info.policy==17, 'retained native F16 control lost its adapter contract'
                elif f['kind']=='q4':
                    assert info.route==2 and info.split==1
                    assert info.config.decode()==f'q4:r{f["reader"]}:v{f["variant"]}:w{f["warps"]}:x{f["values"]}:c{f["columns"]}'
                else:
                    assert info.route==2 and info.split==f['split']
                    name=f'simt:v{f["variant"]}:c{f["columns"]}:w{f["warps"]}:x{f["values"]}:s{f["split"]}'
                    if impl['kind']=='token-cta':name+=f':tokens{m}:window{impl["window"]}'
                    assert info.config.decode()==name
            if expected and policy_device:
                assert info.policy==20 and info.compute_type==0, 'expected ZW810 exact F16 recipe'
                route=expected['route']; assert info.route==(3 if route==2 else route)
                if route==2:
                    name=f'full-f16:cublas:typed-dq{expected["dequant"]}:io{expected["io"]}'
                    if expected.get('dequant_first'): name+=':dq-first'
                    assert info.split==1
                else:
                    name=f'{expected["tc"]["symbol"]}:dq{expected["dequant"]}:io{expected["io"]}'
                    assert info.split==expected['tc']['split']
                assert info.config.decode()==name, 'public recipe does not match admitted winner'
            elif expected:
                assert info.policy!=20, 'ZW810 exact policy must not run on an incompatible device'
            if expected:
                record_property('oracle','ALL_OUTPUTS_FINITE; 9x33_SPOTS_FULL_K_FP64_DOT; ALL_PACKED_BYTES; TWO_STREAMS')
            def check(got,source):
                if not expected:
                    return check_output(got,source,gold[0],info.compute_type)
                assert np.isfinite(got).all(), 'unwritten or nonfinite public output'
                rows=np.linspace(0,m-1,min(m,9),dtype=int)
                cols=np.linspace(0,n-1,min(n,33),dtype=int)
                return check_output(got[np.ix_(rows,cols)],source[rows],gold[0,cols],info.compute_type)
            streams = [arena.stream(), arena.stream()]
            inputs = [arena.buffer(m * p.input_stride * 4) for _ in streams]
            outputs = [arena.buffer(m * p.output_stride * 4) for _ in streams]
            scratch = [arena.buffer(info.workspace_bytes) for _ in streams]
            def launch(index):
                return lib.run(handle, inputs[index].ptr, outputs[index].ptr,
                               scratch[index].ptr, info.workspace_bytes, streams[index])
            for repeat in range(3):
                sources = []
                for i in range(2):
                    source = np.random.default_rng(q * 31 + m + repeat * 7 + i).uniform(-.13, .13, (m, p.input_stride)).astype("f4")
                    if compute and m == 1:
                        source *= 1_000_000  # Exercise range beyond F16, without clipping.
                    inputs[i].write(source); outputs[i].poison()
                    sources.append(source[:, :k])
                arena.sync()
                if repeat == 1 and info.route != 3:
                    graph = Replay(sdk, streams[0], lambda: launch(0), 1)
                for i in range(2):
                    lib.check(graph() if i == 0 and graph else launch(i))
                arena.sync()
                for i in range(2):
                    image = outputs[i].bytes()
                    got = np.frombuffer(image, dtype="f4").reshape(m, p.output_stride)
                    check(got[:, :n], sources[i])
                    if padding:
                        assert np.all(got[:, n:].view("u4") == 0xffffffff), "output stride padding overwritten"
                    inputs[i].guard(); scratch[i].guard()
            # An omitted/zero producer must not pass against retained golden output.
            inputs[0].write(np.zeros((m, p.input_stride), "f4")); outputs[0].poison()
            arena.sync(); lib.check(graph() if graph else launch(0)); arena.sync()
            got = np.frombuffer(outputs[0].bytes(), dtype="f4").reshape(m, p.output_stride)[:, :n]
            assert np.all(got == 0) and np.isfinite(got).all()
            with pytest.raises(ValueError):
                check(got, sources[0])
            # Validate the actual workspace contract before an enqueue.
            if info.workspace_bytes:
                assert lib.run(handle, inputs[0].ptr, outputs[0].ptr, scratch[0].ptr,
                               info.workspace_bytes - 1, streams[0]) != 0
        finally:
            arena.sync()
            if graph:
                graph.close()
            lib.destroy(context)
