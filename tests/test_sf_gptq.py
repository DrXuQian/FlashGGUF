import ctypes as C
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from quactlize.runtime import sf_gptq
from quactlize.runtime.compiler import Compiler, materialize_parent
from quactlize.runtime.candidates import MeasuredCandidates
from quactlize.runtime.dispatch import prepare_selected
from quactlize.runtime.tuning import Request, GPTQ_W4_G32, UnsupportedTactic, digest


@pytest.mark.parametrize("mode",("symmetric","integer"))
def test_prepared_planes_vs_original_affine(mode):
    rng=np.random.default_rng(718)
    e,n,k=2,256,512
    q=rng.integers(0,16,(e,n,k),dtype="u1")
    s=rng.uniform(-3,3,(e,k//32,n)).astype("f2")
    z=rng.integers(0,16,s.shape,dtype="u1") if mode=="integer" else None
    data=sf_gptq.pack(q,s,z)
    kk=np.arange(k)
    decoded=((data["low"].swapaxes(-1,-2)[...,(kk//32)*8+kk%8] >>
             ((kk%32//8)*4).astype("u2")) & 15).astype("f2")
    np.testing.assert_array_equal(decoded,q)
    s16,center=(data[x].view("f2").swapaxes(-1,-2)[...,kk//32] for x in ("metadata","zero"))
    zp=z.swapaxes(-1,-2)[...,kk//32] if z is not None else 8
    oracle=((q.astype("i2")-zp).astype("f4")*s16.astype("f4")).astype("f2")
    restored=((decoded-8)+center)*s16
    np.testing.assert_array_equal(restored.view("u2"),oracle.view("u2"))
    if mode=="integer":
        # Subtracting after a rounded multiply silently changes this fixture.
        assert np.any((decoded-8)*s16+center*s16 != oracle)
    assert data["mapping_id"]==GPTQ_W4_G32


def test_low_bytes_vs_flashquant_packer(tmp_path):
    root=Path(__file__).resolve().parents[1]/"third_party/flashquant/flashquant"
    if not (root/"affine/pack.cu").is_file():
        # Ordinary product builds deliberately need no second checkout. Use
        # the existing canonical C++ K-pack4 producer for that host suite.
        root=Path(__file__).resolve().parents[1]/"quactlize/include"
        dllsrc=tmp_path/"pack.cpp"
        dllsrc.write_text('#include "q4_kpack4_offline.hpp"\n#include <vector>\n'
            'extern "C" int pack(uint8_t const*q,int n,int k,uint8_t*out,uint64_t bytes){'
            'std::vector<uint8_t>b(bytes);for(size_t i=0;i<bytes;++i)b[i]=q[i*2]|(q[i*2+1]<<4);'
            'return q4_kpack4::prepare(b.data(),out,n,k);}')
        inc,src,symbol=root,dllsrc,"pack"
    else:
        inc,src,symbol=root/"include",root/"affine/pack.cu","quactlize_ppu_w4a16_pack_host_v1"
    so=tmp_path/"host-pack.so"
    subprocess.run(["c++","-std=c++17","-shared","-fPIC","-O2","-x","c++","-I"+str(inc),
                    str(src),"-o",str(so)],check=True)
    pack=getattr(C.CDLL(str(so)),symbol)
    pack.argtypes=[C.c_void_p,C.c_int32,C.c_int32,C.c_void_p,C.c_uint64]
    q=np.random.default_rng(12).integers(0,16,(256,512),dtype="u1")
    want=np.empty(q.size//2,dtype="u1")
    assert pack(q.ctypes.data,256,512,want.ctypes.data,want.size)==0
    assert sf_gptq.pack(q,np.ones((16,256),"f2"))["low"].tobytes()==want.tobytes()


def test_module_and_cache_contract_is_separate():
    p=dict(qtype=12,route="sf-dense",symbol="sf_q4_rc4",tm=16,tn=64,tk=128,wm=16,
           wn=16,stages=2,dn=64,ap=0,persistent=-1,dense_reducer=4)
    g=sf_gptq.parent(p)
    assert g["symbol"]=="gptq_sf_q4_rc4" and g["dense_reducer"]==4
    assert sf_gptq.parent(g)==g==materialize_parent({k:v for k,v in g.items() if k!="weight_style"})
    assert "#define QK_SF_GPTQ 1" in Compiler.source(None,g,"test")
    assert "QK_SF_GPTQ" not in Compiler.source(None,p,"test")
    a=Request("sf-dense",12,256,512,2); b=Request("sf-dense",12,256,512,2,weight_style="gptq")
    assert (a.mapping!=b.mapping and a.exact_key!=b.exact_key and a.bucket!=b.bucket)
    assert a.exact_key==digest(dict(route=a.route,qtype=12,m=2,n=256,k=512,rows=()))
    for fields in (dict(route="fq-dense"),dict(qtype=13)):
        with pytest.raises(ValueError): sf_gptq.parent(p|fields)
    with pytest.raises(ValueError): materialize_parent(p|dict(weight_style="gptq"))


def test_metadata_rejects_raw_or_unrepresentable_inputs():
    q=np.zeros((256,256),"u1"); s=np.ones((8,256),"f2")
    for value in (np.full(s.shape,16,"u1"),np.ones((256,8),"u1"),np.ones(s.shape,"f2")):
        with pytest.raises(ValueError): sf_gptq.pack(q,s,value)
    q[0,0]=16
    with pytest.raises(ValueError): sf_gptq.pack(q,s)


def test_exact_and_loaded_receipts_do_not_cross_weight_contracts(tmp_path):
    gg=Request("sf-dense",12,256,512,2)
    gp=Request("sf-dense",12,256,512,2,weight_style="gptq")
    geometry=dict(qtype=12,route="sf-dense",symbol="sf_q4",tm=16,tn=64,tk=128,
                  wm=16,wn=16,stages=2,dn=64,ap=0,persistent=-1)
    alg=dict(algorithm="NONPERSISTENT",split=1,grid_mode="ordinary",grid_b=0)
    configs={"gg":geometry|alg,"gp":sf_gptq.parent(geometry)|alg}
    entries=[dict(config_id=label,key=[12,1,256,512,1,2],status="MEASURED_WITHIN_5PCT") for label in configs]
    data=dict(schema="quactlize.kpack-runtime-policy.v1",configurations=configs,entries=entries)
    file=tmp_path/"policy.json";file.write_text(json.dumps(data|{"policy_digest":digest(data)}))
    catalog=MeasuredCandidates(file)
    ctx=dict(sdk="sdk",kernel="kernel")
    backend=SimpleNamespace(modules={k:SimpleNamespace(record=dict(parent=v,identity=ctx)) for k,v in catalog.parents.items()},
        identity=ctx|dict(device="PPU-ZW810",compute_units=72),is_capturing=lambda:False,prepare=lambda r,t:t.parent)
    for label,req,other in (("gg",gg,gp),("gp",gp,gg)):
        name=configs[label]["symbol"]
        assert catalog.exact_entry(req)["config_id"]==label
        assert [t.parent for t in catalog.shortlist(req)]==[name]
        saved=asdict(req)
        if label=="gg": del saved["weight_style"]  # original persisted shape/key
        selection=dict(status="MEASURED_RECENT",request=saved,config=configs[label],module_contract=ctx)
        assert prepare_selected(backend,req,selection)==name
        with pytest.raises(UnsupportedTactic): prepare_selected(backend,other,selection)
