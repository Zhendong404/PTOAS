#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

candidates=()
add_candidate() {
  local candidate="$1"
  local resolved
  [[ -n "${candidate}" ]] || return 0
  resolved="$(command -v "${candidate}" 2>/dev/null || true)"
  [[ -n "${resolved}" && -x "${resolved}" ]] || return 0
  resolved="$(readlink -f "${resolved}")"
  [[ " ${candidates[*]} " == *" ${resolved} "* ]] || candidates+=("${resolved}")
}

add_candidate "${PTO_DSL_ST_PYTHON_BIN:-}"
add_candidate python3.11
shopt -s nullglob
for candidate in \
  /home/*/miniconda3/bin/python3.11 /home/*/anaconda3/bin/python3.11 \
  /home/*/miniconda3/envs/*/bin/python /home/*/anaconda3/envs/*/bin/python \
  /opt/conda/envs/*/bin/python
do
  add_candidate "${candidate}"
done

for candidate in "${candidates[@]}"; do
  if TORCH_DEVICE_BACKEND_AUTOLOAD=0 "${candidate}" - <<'PY' >/dev/null 2>&1
import sys
import torch
import torch_npu
raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)
PY
  then
    printf '%s\n' "${candidate}"
    exit 0
  fi
done

echo "ERROR: no Python 3.11 runtime can import both torch and torch_npu" >&2
echo "ERROR: set PTO_DSL_ST_PYTHON_BIN to a compatible preinstalled interpreter" >&2
exit 1
