#!/usr/bin/env bash
# ZW810-family dense prefill; unpack the benchmark overlays alongside older runs.
set -euo pipefail
suite="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/mnk13" && pwd -P)"
test -n "${suite}" && test -f "${suite}/identity.json" && test -f "${suite}/run.py"
(
    cd -- "${suite}"
    flock -x 9
    cache=.source-f16-v3
    if [[ -L "$cache" || ( -e "$cache" && ! -d "$cache" ) ]]; then
        echo "mnk13/$cache is not an ordinary directory; use a fresh clone" >&2
        exit 1
    fi
    if [[ ! -d "$cache" ]]; then
        sha256sum --check source.sha256
        temp_src="$(mktemp -d -- "${suite}/.source-XXXXXXXX")"
        [[ -n "${temp_src}" && "${temp_src}" == "${suite}"/.source-* && -d "${temp_src}" && ! -L "${temp_src}" ]]
        tar -xzf source.tar.gz --no-same-owner -C "${temp_src:?}"
        test -f "${temp_src}/tools/run_prefill_model_asys.py"
        patch --quiet --batch --fuzz=0 --forward --no-backup-if-mismatch -p1 -d "${temp_src:?}" < measurement-domain.patch
        patch --quiet --batch --fuzz=0 --forward --no-backup-if-mismatch -p1 -d "${temp_src:?}" < sf-f16-audit.patch
        mv -T -- "${temp_src:?}" "${suite}/${cache}"
    fi
) 9>"${suite}/.unpack.lock"
exec "${PYTHON:-python3}" -B -u "${suite}/run.py" "$@"
