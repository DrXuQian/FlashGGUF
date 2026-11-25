import ctypes as C
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from ci.support.fixture import planes,bf16_bits
from tests.warpk_fq_oracle import fixture,oracle

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def decoder(tmp_path_factory):
    path=tmp_path_factory.mktemp('fq-warpk')/'decode.so'
    sdk=Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK'))
    includes=[ROOT/'quactlize/dense',ROOT/'quactlize/include',ROOT/'third_party/actlize/include',
              sdk/'include',sdk/'targets/x86_64-linux/include']
    subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-fno-strict-aliasing','-ffp-contract=off',
                    *['-I'+str(d) for d in includes],str(ROOT/'tests/warpk_fq_host.cpp'),'-o',str(path)],check=True)
    fn=C.CDLL(str(path)).decode_units
    fn.argtypes=[C.c_int,C.c_bool,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_void_p,C.c_void_p]
    fn.restype=None
    return fn

@pytest.mark.parametrize('q',range(10,15))
@pytest.mark.parametrize('bf',[False,True],ids=['f16','bf16'])
def test_fq_warpk_units(decoder,q,bf):
    n,k,e=256,1024,2
    raw=[fixture(q,n,k,719+3*i) for i in range(e)]
    units=np.stack([planes(r,q)['units'] for r in raw])
    want=[oracle(r,q,bf) for r in raw]
    s=np.zeros((e,*want[0][0].shape),'u2');z=np.zeros_like(s)
    decoder(q,bf,units.ctypes.data,n,k//256,e,s.ctypes.data,z.ctypes.data)
    for i,actual in enumerate((s,z)):
        expected=np.stack([p[i] for p in want])
        bits=bf16_bits(expected) if bf else expected.astype('f2').view('u2')
        np.testing.assert_array_equal(actual,bits)
