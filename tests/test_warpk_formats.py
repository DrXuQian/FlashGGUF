"""Canonical physical stripes and device-offset scheduler, no GPU or emulation."""
import ctypes as C
from pathlib import Path
import subprocess
import numpy as np
import pytest
from ci.support.fixture import planes,raw_weight

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def host(tmp_path_factory):
    so=tmp_path_factory.mktemp("warpk-host")/"check.so"
    subprocess.run(["c++","-std=c++17","-O2","-shared","-fPIC","-I"+str(ROOT/"quactlize/dense"),
                    "-I"+str(ROOT/"quactlize/include"),str(ROOT/"tests/warpk_host.cpp"),"-o",str(so)],check=True)
    lib=C.CDLL(str(so));p=C.c_void_p;i=C.c_int
    lib.codes.argtypes=[i,p,p,i,i,p];lib.schedule.argtypes=[i,i,p,p,p]
    return lib

@pytest.mark.parametrize("q",(8,10,11,12,13,14))
def test_packed_tile_against_independent_gguf(host,q):
    n,k=256,1024
    for seed in (19,837):
        raw,oracle=raw_weight(q,n,k,seed,varied_groups=True)
        p=planes(raw,q);codes=np.full((n,k),-999,dtype="i2")
        assert host.codes(q,p["low"].ctypes.data,p["high"].ctypes.data,n,k,codes.ctypes.data)==0
        groups=p["scale"].shape[0]
        w=codes.reshape(n,groups,-1).astype("f2")*p["scale"].T[...,None]
        if q!=8: w+=p["zero"].T[...,None]
        np.testing.assert_array_equal(w.reshape(n,k),oracle)

def test_expert_tiles_exactly_once(host):
    rng=np.random.default_rng(25)
    for e in (1,2,8,256,512,1024):
        for rows in (1,2,7,8,9,15,16,17,33,65,128,2048,5120):
            for ids in (rng.integers(e,size=rows),np.full(rows,e-1)):
                counts=np.bincount(ids,minlength=e); off=np.array([0,*np.cumsum(counts)],"i4")
                owner=np.full(rows,-1,"i4");visits=np.zeros(rows,"i4")
                nt=host.schedule(rows,e,off.ctypes.data,owner.ctypes.data,visits.ctypes.data)
                assert nt==min(rows,rows//16+e)
                np.testing.assert_array_equal(visits,np.ones_like(visits))
                np.testing.assert_array_equal(owner,np.sort(ids))
