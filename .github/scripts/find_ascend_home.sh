#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail
shopt -s nullglob
for candidate in \
  "${ASCEND_HOME_PATH:-}" \
  /usr/local/Ascend/cann \
  /usr/local/Ascend/cann-* \
  /usr/local/Ascend/ascend-toolkit/latest
do
  if [[ -n "${candidate}" && -d "${candidate}" && -f "${candidate}/bin/setenv.bash" ]]; then
    readlink -f "${candidate}"
    exit 0
  fi
done
echo "ERROR: no CANN installation with bin/setenv.bash was found" >&2
exit 1
