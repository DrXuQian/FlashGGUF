"""Check each emitted specialization, not just that a DSO somewhere has an MMA."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.build_kpack_dequant import resource_usage


def audit(objdump, image, *, require_fq=True):
    text=subprocess.check_output([str(objdump),'--dump-isa',str(image)],text=True)
    resources=resource_usage(subprocess.check_output([str(objdump),'--dump-resource-usage=all',str(image)],text=True))
    sections=re.split(r'Disassembly of section \.text\.kernel\.([^:\n]+):',text)
    result={}
    for symbol, body in zip(sections[1::2],sections[2::2]):
        m=re.search(r'warpk5multi6kernelILi(\d+)EN7cutlass(6half_t|10bfloat16_t)ELb([01])E',symbol)
        if m:
            style='gptq' if m[3]=='1' else 'gguf'
        else:
            m=re.search(r'warpk5multi9fq_kernelILi(\d+)EN7cutlass(6half_t|10bfloat16_t)E',symbol)
            if not m:continue
            style='fq'
        q,dt=int(m[1]),'f16' if m[2]=='6half_t' else 'bf16'
        key=f'{style}:q{q}:{dt}'
        ops=[f'v.mma.f32.{dt}.m16n16k16','vmem.aiu.ld','tsm.ld.swzl']
        counts={op:body.count('\t'+op) for op in ops}
        if not all(counts.values()) or key in result:
            raise ValueError(f'{key}: missing/duplicate bulk-A, SWZL load or native MMA: {counts}')
        result[key]={'instructions':counts,'resources':resources[symbol]}
    expected={f'gguf:q{q}:{dt}' for q in (8,10,11,12,13,14) for dt in ('f16','bf16')}
    expected.update(('gptq:q12:f16','gptq:q12:bf16'))
    if require_fq or any(k.startswith('fq:') for k in result):
        expected.update(f'fq:q{q}:{dt}' for q in range(10,15) for dt in ('f16','bf16'))
    if set(result)!=expected: raise ValueError(f'missing/unexpected specializations: {set(result)^expected}')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('objdump',type=Path);p.add_argument('image',type=Path)
    p.add_argument('--allow-legacy-sf-only',action='store_true')
    args=p.parse_args()
    print(json.dumps(audit(args.objdump,args.image,require_fq=not args.allow_legacy_sf_only),sort_keys=True))
