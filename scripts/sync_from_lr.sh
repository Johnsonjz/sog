#!/usr/bin/env bash
# Copy the vendored SOG sources from deepmd-kit-LR into this repository.
#
# LR is the source of truth (see lmp/SYNC_NOTE.md).  This script only READS
# from the LR tree; it never modifies it.  After running, review `git diff`
# and rebuild/test (docs/08-validation.md).
set -euo pipefail

LR_ROOT="${LR_ROOT:-/data/home/public/jiangzhen/dp/deepmd-kit-LR}"
PIN="a266642d2ae797ef818575d50dbc4c43b8e3d7ca"

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${LR_ROOT}/source"

if [[ ! -d "${SRC}" ]]; then
  echo "error: LR source tree not found at ${SRC} (set LR_ROOT)" >&2
  exit 1
fi

current="$(git -C "${LR_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
if [[ "${current}" != "${PIN}" ]]; then
  echo "WARNING: ${LR_ROOT} is at ${current}, pinned ${PIN}" >&2
  if [[ -t 0 ]]; then
    echo "         sync anyway? press Ctrl-C to abort, Enter to continue" >&2
    read -r _
  else
    echo "         non-interactive: continuing" >&2
  fi
fi

copy_pair() {
  local from="$1" to="$2"
  if [[ ! -f "${from}" ]]; then
    echo "error: missing ${from}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${DEST}/${to}")"
  cp "${from}" "${DEST}/${to}"
  echo "  ${to}"
}

echo "syncing vendored files from ${LR_ROOT} @ ${current}"
copy_pair "${SRC}/lmp/sog_gpu.cu"                          "lmp/sog_gpu.cu"
copy_pair "${SRC}/lmp/sog_gpu.cuh"                         "lmp/sog_gpu.cuh"
copy_pair "${SRC}/lmp/sog_spline.h"                        "lmp/sog_spline.h"
copy_pair "${SRC}/lmp/sog_charge_source.h"                 "lmp/sog_charge_source.h"
copy_pair "${SRC}/op/pt/sog_kspace/sog_kspace_op.cc"       "op/sog_kspace/sog_kspace_op.cc"
copy_pair "${SRC}/op/pt/sog_kspace/sog_green.h"            "op/sog_kspace/sog_green.h"

echo
echo "Patched files (NOT copied; re-apply lmp/patches/sog-model-agnostic.diff):"
echo "  lmp/sog.cpp lmp/sog.h lmp/fix_sog_response.cpp lmp/fix_sog_response.h"
echo
git -C "${DEST}" diff --stat || true
