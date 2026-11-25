#!/usr/bin/env bash
# One idle PPU1.0, already configured SDK. Only the explicit run dir is written.
set -euo pipefail
mode=all
case "$#:${1:-}" in
    0:) ;;
    1:--formats) mode=formats ;;
    1:--fq) mode=fq ;;
    *) echo 'Usage: bash experiments/shared_warpk/box.sh [--formats|--fq]' >&2; exit 2 ;;
esac
root=$(git rev-parse --show-toplevel)
cd "${root:?}"
test -f quactlize/dense/warpk.cu
sdk=${PPU_SDK:-/usr/local/PPU_SDK}
test -x "$sdk/bin/hgcc"
test -f "$sdk/lib/libhggc_wrapper.so"
py=${PYTHON:-python3}
export PPU_SDK="$sdk" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="$sdk/bin:$PATH" LD_LIBRARY_PATH="$sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
run=$(mktemp -d "${TMPDIR:-/tmp}/flashgguf-warpk.XXXXXXXX")
test -d "${run:?}"
echo "WARPK_OUTPUT=$run"
export OPENBLAS_NUM_THREADS="${WARPK_CPU_THREADS:-2}" OMP_NUM_THREADS="${WARPK_CPU_THREADS:-2}"
"$py" -c 'import torch, gguf, numpy, pytest'
git submodule update --init third_party/actlize third_party/flashquant
if [[ -z ${QUACTLIZE_CI_PACKAGE:-} ]]; then
    export QUACTLIZE_CI_PACKAGE="$run/package"
    bash build.sh --library --with-warpk --arch ppu_10 --sdk "$sdk" --output "$QUACTLIZE_CI_PACKAGE" --jobs "${JOBS:-8}"
fi
"$py" -c 'import json,os,sys;from pathlib import Path
w=json.loads((Path(os.environ["QUACTLIZE_CI_PACKAGE"])/"manifest.json").read_text()).get("warpk",{})
if not w.get("multiformat") or set(w["qtypes"])!={8,10,11,12,13,14}:
    raise SystemExit("build this branch with --with-warpk; the package is missing the multi-format candidate")
if sys.argv[1] in ("fq","all") and "gguf-fq" not in w.get("styles",[]):
    raise SystemExit("FQ not in package: unset QUACTLIZE_CI_PACKAGE and rebuild this branch")' "$mode"
obj="${QUACTLIZE_CI_PACKAGE}.build/warpk_sf.o"
if [[ -f "$obj" ]]; then
    isa_args=()
    if [[ $mode == formats ]]; then isa_args+=(--allow-legacy-sf-only); fi
    "$py" experiments/shared_warpk/isa.py "$sdk/bin/hgobjdump" "$obj" "${isa_args[@]}" >"$run/isa.json"
fi
if [[ $mode != fq ]]; then
    QUACTLIZE_CI_SUITE=accuracy "$py" -m pytest -qx --confcutdir=experiments/shared_warpk \
        tests/test_warpk_formats.py experiments/shared_warpk/test_formats_device.py --junitxml="$run/formats.xml"
    echo "WARPK_FORMATS_PASS q=8,10,11,12,13,14 modes=dense,grouped output=$run performance_admitted=false"
fi
bench_args=()
if [[ $mode != formats ]]; then
    QUACTLIZE_CI_SUITE=accuracy "$py" -m pytest -qx --confcutdir=experiments/shared_warpk \
        tests/test_warpk_fq.py experiments/shared_warpk/test_fq_device.py --junitxml="$run/fq.xml"
    echo "WARPK_FQ_PASS q=8,10,11,12,13,14 modes=dense,grouped output=$run performance_admitted=false"
    bench_args+=(--with-fq)
fi
if [[ $mode != all ]]; then
    if [[ ${WARPK_BENCH:-0} == 1 ]]; then
        for q in ${TRACE_Q:-8 10 11 12 13 14}; do
            for op in ${TRACE_OP:-dense moe-gate moe-down}; do
                for m in ${TRACE_M:-1}; do
                    prefix="$run/$q-$op-m$m"
                    "$sdk/asight/bin/asys" profile -t hggc --sample none --python-sampling false \
                        --capture-range hggcProfilerApi --capture-range-end merge --kill none --wait primary \
                        -f false -o "$prefix" "$py" experiments/shared_warpk/formats_ab.py \
                        --q "$q" --mode "$op" --m "$m" --profiler "${bench_args[@]}" --output "$prefix.json"
                    test -s "$prefix.asysrep"
                done
            done
        done
        echo "WARPK_FORMATS_AB_DONE output=$run"
    fi
    # XML + compact JSON only. The original Asys reports stay on the box.
    ( cd "$run"; shopt -s nullglob; tar -czf review.tgz *.xml *.json )
    echo "review=$run/review.tgz"
    exit 0
fi
if [[ -z ${FLASHQUANT_REFERENCE:-} ]]; then
    # Independent historical source, not the new shared collective compiled twice.
    git clone --quiet --no-hardlinks --no-checkout third_party/flashquant "$run/reference"
    git -C "$run/reference" checkout --quiet --detach 0a4c5b0845bd366ad1d2a106bd04db1f39181a69
    git -C "$run/reference" -c protocol.file.allow=always \
        -c "submodule.third_party/actlize.url=$root/third_party/actlize" submodule update --init third_party/actlize
    "$py" "$run/reference/tools/build_affine.py" --sdk "$sdk" --output "$run/affine" --configs 616,619 --jobs "${JOBS:-8}"
    export FLASHQUANT_REFERENCE="$run/affine/libflashquant.so"
fi
"$py" -c 'import json,os;from pathlib import Path;from quactlize.runtime.compiler import sha
p=Path(os.environ["FLASHQUANT_REFERENCE"]);m=json.loads((p.parent/"manifest.json").read_text())
if m["revision"]!="0a4c5b0845bd366ad1d2a106bd04db1f39181a69" or m["library_sha256"]!=sha(p):
    raise ValueError("FLASHQUANT_REFERENCE must be the original FlashQuant build and match its manifest")'
QUACTLIZE_CI_SUITE=accuracy "$py" -m pytest --confcutdir=experiments/shared_warpk -q tests/test_sf_gptq.py experiments/shared_warpk/test_device.py \
    --junitxml="$run/numerics.xml"
if [[ ${CHECK_TWINS:-1} == 1 ]]; then
    # Reuse the same pinned actlize checkout locally, including on an offline box.
    git -C third_party/flashquant -c protocol.file.allow=always \
        -c "submodule.third_party/actlize.url=$root/third_party/actlize" submodule update --init third_party/actlize
    "$py" third_party/flashquant/tools/build_affine.py --sdk "$sdk" --output "$run/twins" \
        --configs 535,625,627,756,829,1208,1325 --relax-m1 --jobs "${JOBS:-8}"
    ( cd third_party/flashquant
      FLASHQUANT_LIBRARY="$run/twins/libflashquant.so" "$py" -m pytest -q tests/ppu/test_affine.py \
          -k dynamic_m_twins --junitxml="$run/twins.xml" )
fi
echo "WARPK_NUMERICS_PASS output=$run performance_admitted=false"
if [[ ${WARPK_BENCH:-0} == 1 ]]; then
    "$py" experiments/shared_warpk/bench.py --output "$run/paired.json"
    for family in gguf gptq; do
        "$sdk/asight/bin/asys" profile -t hggc --sample none --python-sampling false \
            --capture-range hggcProfilerApi --capture-range-end merge --kill none --wait primary \
            -f false -o "$run/$family" "$py" experiments/shared_warpk/bench.py \
            --family "$family" --m "${TRACE_M:-1}" --rounds 2 --profiler --output "$run/$family.json"
        test -s "$run/$family.asysrep"
    done
    echo "WARPK_AB_DONE report=$run/paired.json traces=$run/{gguf,gptq}.asysrep"
fi
