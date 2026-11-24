#!/usr/bin/env python3
"""One-shape 810E SF experiment. No catalog/selection changes or online tuning."""
import argparse
import faulthandler
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quactlize.runtime.compiler import Compiler
from quactlize.runtime.tuning import Request, Tactic

M, N, K = 512, 25600, 5120
ARMS = ("selected", "wide", "pair8")


def parent(arm):
    tn, wn = (64, 32) if arm == "selected" else (256, 16)
    return dict(symbol="q4k_sf_m512_" + arm, route="sf-dense", qtype=12, tm=64,
                tn=tn, tk=64, wm=64, wn=wn, stages=3, ap=0, dn=64, persistent=-1)


class ExperimentCompiler(Compiler):
    def source(self, p, key):
        src = super().source(p, key)
        return ("#define QZ_EXPERIMENT_SF_PAIR8 1\n" if p == parent("pair8") else "") + src


def persistent_isa(text):
    matches = [s for s in re.split(r"(?m)^Disassembly of section ", text)
               if s.startswith(".text.kernel.") and "26PersistentMixedInputKernel" in s.splitlines()[0]]
    if len(matches) != 1:
        raise ValueError("could not identify the executed persistent kernel")
    body = matches[0]
    # User IDs 0..7 become physical IDs 6..13. Exclude ordinary CTA/epilogue barriers.
    hits = list(re.finditer(r"\bs\.blksyn\.(defer|nb)\s+(?:sreg\d+|0x[6-9a-d]),\s*(?:0x40|64)\b", body))
    windows = [body[a.end():b.start()] for a,b in zip(hits,hits[1:])
               if a[1] == "defer" and b[1] == "nb"]
    return dict(mmas=body.count("v.mma.f32.f16.m16n16k16"), pair_ops=[m[1] for m in hits],
                decode_between=any("tsm.ld.swzl" in w and "v.fma.f16x2" in w for w in windows))


def isa_gate(wide, pair):
    if (wide["pair_ops"] or not wide["mmas"] or wide["mmas"] != pair["mmas"] or
            not pair["decode_between"]):
        raise ValueError(f"unverified native rendezvous in persistent mainloop: {wide}, {pair}")


def revision():
    # The source-only Box tar contains the tested commit but deliberately no .git.
    stamp = ROOT/"PAIR8_REVISION"
    rev = (stamp.read_text() if stamp.is_file() else subprocess.check_output(
        ["git","rev-parse","HEAD"], cwd=ROOT, text=True)).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", rev):
        raise ValueError("missing or invalid FlashGGUF source identity")
    return rev


def fixture():
    import numpy as np
    from ci.support.weights import prepare_expert
    from ci.support.fixture import matmul_reference
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize
    from reference.gguf_kpack import SPECS
    rng = np.random.default_rng(810)
    s = SPECS[12]
    q = rng.integers(0, 256, (N, K//256, s.raw_bytes), dtype="u1")
    d = np.ldexp(np.ones(q.shape[:2], dtype="<f2"), -rng.integers(6,10,q.shape[:2])).astype("<f2")
    for offset, value in ((s.d_offset, d), (s.dmin_offset, d*np.float16(.5))):
        q[...,offset:offset+2] = value.view("u1").reshape(*q.shape[:2],2)
    # Sample every warp's columns, both edges, and different persistent work items.
    cols = np.unique(np.concatenate([np.arange(16)*16 + edge + start
                                     for start in (0,N//2,N-256) for edge in (0,15)]))
    oracle = dequantize(q[cols].reshape(-1), GGMLQuantizationType(12)).reshape(len(cols),K)
    a = (rng.integers(-64,65,(M,K)).astype("<f2")/np.float16(64))
    gold, denom = matmul_reference(a, oracle, "f16", output_compute="f16")
    planes = prepare_expert(q,12,N,K)
    k = np.arange(K)
    code = ((planes["low"][:,cols][(k//32)*8+k%8] >> (4*((k//8)%4))[:,None]) & 15).astype("<f2")
    # Offset-binary Q4: SF zero already includes +8*scale. Do not apply it twice.
    restored = (code - np.float16(8))*planes["scale"][:,cols][k//32] + planes["zero"][:,cols][k//32]
    np.testing.assert_array_equal(restored.T, oracle)
    return dict(a=a, low=planes["low"], metadata=planes["scale"], zero=planes["zero"]), cols, gold, denom


def worker(args):
    import numpy as np
    from ci.support.fixture import compare, digest
    from quactlize.runtime.native import NativeBackend, SDK, checked
    print("PAIR8_FIXTURE", flush=True)
    operands, cols, gold, denom = fixture()  # check the independent oracle before compiling
    compiler = ExperimentCompiler(args.sdk, args.output/"cache", arch="ppu_10")
    modules = {}
    report = dict(shape=[M,N,K], source=revision(), runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"), scope="PREPARED_SF_MAIN_NOT_PUBLIC_CALL",
                  performance_admitted=False)
    def save():
        (args.output/"report.json").write_text(json.dumps(report, indent=2, default=str)+"\n")
    for arm in ARMS:
        print("PAIR8_BUILD",arm,flush=True)
        modules[arm] = compiler.build(parent(arm))
    report["modules"] = {a: {k:r[k] for k in ("key","sha256","parent","identity")} for a,r in modules.items()}
    report["isa"] = {a:persistent_isa(subprocess.check_output([args.sdk/"bin/hgobjdump","--dump-isa",
                          modules[a]["path"]], text=True)) for a in ("wide","pair8")}
    save()
    isa_gate(report["isa"]["wide"],report["isa"]["pair8"])
    sdk = SDK(args.sdk)
    pointers, handles = {}, {}
    outputs, us = {}, {a:[] for a in ARMS}
    backend = None
    # An unmatched device barrier can hang a foreign call (also under Asys).
    faulthandler.dump_traceback_later(180, exit=True)
    try:
        for k,v in operands.items():
            pointers[k] = sdk.upload(v.tobytes())
        pointers["output"] = sdk.allocate(M*N*2)
        backend = NativeBackend(args.sdk, list(modules.values()), pointers, correctness=None)
        report["device"] = backend.identity
        if backend.identity["device"].upper().removeprefix("PPU-") != "ZW810E":
            raise ValueError("expected one selected ZW810E, got "+str(backend.identity))
        request = Request("sf-dense",12,N,K,M)
        def tactic(a,b):
            return Tactic(parent(a)["symbol"],"PERSISTENT",1,"capacity",b)
        resource = {a:backend.arguments(request,tactic(a,1))[3] for a in ARMS}
        occ = {a:v.occupancy for a,v in resource.items()}
        report["shared_bytes"] = {a:v.shared_bytes for a,v in resource.items()}
        shared_cap = min(8,occ["wide"],occ["pair8"])
        report["occupancy"], report["grids"], report["numerics"] = occ, {}, {}
        for a in ARMS:
            t = tactic(a,min(8,occ[a]) if a=="selected" else shared_cap)
            report["grids"][a] = backend.arguments(request,t)[2].grid
            handles[a] = backend.prepare(request,t)
            sdk.fill(pointers["output"],255,M*N*2)
            backend.run(handles[a]); backend.synchronize()
            y = np.frombuffer(sdk.download(pointers["output"],M*N*2),"<f2").reshape(M,N)
            if not np.isfinite(y).all():
                raise ValueError(a+": nonfinite or unwritten output")
            report["numerics"][a] = compare(y[:,cols],gold,denom)
            outputs[a] = digest(y)
        report["output_hashes"] = outputs
        save()
        if outputs["wide"] != outputs["pair8"] or report["grids"]["wide"] != report["grids"]["pair8"]:
            raise ValueError("same-geometry arms must have bit-identical outputs and identical grids")
        for h in handles.values():
            for _ in range(4): backend.run(h)
        backend.synchronize()
        orders = list(itertools.permutations(ARMS))
        for seq in orders:
            for a in seq: us[a].append(backend.measure(handles[a],12))
        report["samples_us"] = us
        report["median_us"] = {a:statistics.median(v) for a,v in us.items()}
        save()
        if args.asys:
            # One capture, exactly one prepared main launch per entry. Do not include events, warmup,
            # oracle or packing. Keep explicit order: two modules can share a demangled kernel name.
            trace_order = [a for seq in orders for a in seq]
            checked(sdk.lib.hggcProfilerStart(), "profiler start")
            for a in trace_order: backend.run(handles[a])
            backend.synchronize()
            checked(sdk.lib.hggcProfilerStop(), "profiler stop")
            report["trace_order"] = trace_order
        save()
    finally:
        if backend:
            for h in handles.values(): backend.close(h)
            backend.release()
        for p in pointers.values(): sdk.free(p)
        faulthandler.cancel_dump_traceback_later()
    report["status"] = "PASS"
    save()
    print("PAIR8_DONE median_us=",report["median_us"],"report=",args.output/"report.json",flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdk",type=Path,default=Path(os.getenv("PPU_SDK") or "/usr/local/PPU_SDK"))
    p.add_argument("--asys",action="store_true")
    p.add_argument("--output",type=Path)
    p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=p.parse_args()
    args.sdk=args.sdk.resolve()
    if args.worker: return worker(args)
    if args.output is None: args.output=Path(tempfile.mkdtemp(prefix="q4k-sf-pair8-"))
    else: args.output=args.output.resolve(); args.output.mkdir(parents=True,exist_ok=False)
    cmd=[sys.executable,"-u",str(Path(__file__).resolve()),"--worker","--sdk",str(args.sdk),"--output",str(args.output)]
    if args.asys:
        cmd=[str(args.sdk/"asight/bin/asys"),"profile","-t","hggc","--sample","none","--python-sampling","false",
             "--capture-range","hggcProfilerApi","--capture-range-end","merge","--kill","none","--wait","primary",
             "-f","false","-o",str(args.output/"pair8"),*cmd,"--asys"]
    print("PAIR8_OUTPUT",args.output,flush=True)
    subprocess.run(cmd,check=True,timeout=900)
    if json.loads((args.output/"report.json").read_text()).get("status") != "PASS":
        raise RuntimeError("worker did not finish; inspect "+str(args.output))
    if args.asys and not (args.output/"pair8.asysrep").is_file():
        raise RuntimeError("Asys did not write pair8.asysrep in "+str(args.output))


if __name__=="__main__":
    main()
