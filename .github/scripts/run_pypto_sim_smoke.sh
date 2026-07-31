#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail
: "${PYPTO_WORKSPACE:?}"
: "${PYPTO_RUN_WORKSPACE:?}"
: "${CONSUMER_PYTHON:?}"
mkdir -p "${PYPTO_RUN_WORKSPACE}"
cd "${PYPTO_WORKSPACE}"

run_pypto_pytest() {
  local platform="$1" suite_name="$2"
  shift 2
  "${CONSUMER_PYTHON}" -m pytest "$@" -v --platform="${platform}" --save-kernels \
    --kernels-dir="${PYPTO_RUN_WORKSPACE}/${suite_name}_${platform}" \
    2>&1 | tee "${PYPTO_RUN_WORKSPACE}/${suite_name}_${platform}.log"
}

run_core_smoke() {
  local platform="$1"
  local tests=(
    tests/st/examples/00_hello_world/test_hello_world.py
    tests/st/examples/02_intermediate/test_softmax.py::TestTileSoftmax::test_tile_softmax
    tests/st/examples/02_intermediate/test_rms_norm.py::TestRMSNormCore::test_rms_norm_core
    tests/st/runtime/ops/test_elementwise.py
    tests/st/runtime/ops/test_assemble.py
    tests/st/runtime/framework_and_models/test_jit.py::TestJITExecution::test_cache_hit_reuses_compiled_program
    tests/st/runtime/framework_and_models/test_jit.py::TestJITDynamicBatch::test_one_artifact_serves_multiple_batches
    tests/st/runtime/framework_and_models/test_compiled_program.py::TestManualWorkerExtraction::test_block_dim_override_runs
    "tests/st/runtime/cross_core/test_cross_core.py::TestCrossCore::test_tpush_tpop_v2c_updown[${platform}]"
  )
  if [[ "${platform}" != "a5sim" ]]; then
    tests+=("tests/st/runtime/control_flow/test_dyn_orch_shape.py::TestDynOrchShapeOperations::test_dyn_orch_valid_shape_add[shape0-valid_shape0-${platform}]")
  fi
  run_pypto_pytest "${platform}" core_ptoas "${tests[@]}"
}

run_fa_smoke() {
  local platform="$1"
  run_pypto_pytest "${platform}" fa_ptoas \
    tests/st/runtime/ops/test_cast.py::TestCast::test_tile_cast_col_major_narrow \
    'tests/st/runtime/framework_and_models/test_paged_attention.py::TestPagedAttentionKernels::test_qk_matmul_ptoas[16-128-128]' \
    'tests/st/runtime/framework_and_models/test_paged_attention.py::TestPagedAttentionKernels::test_softmax_prepare_ptoas[16-128]' \
    'tests/st/runtime/framework_and_models/test_paged_attention.py::TestPagedAttentionKernels::test_softmax_prepare_unaligned_ptoas[16-128-100]' \
    'tests/st/runtime/framework_and_models/test_paged_attention.py::TestPagedAttentionKernels::test_pv_matmul_ptoas[16-128-128]' \
    'tests/st/runtime/framework_and_models/test_paged_attention.py::TestPagedAttentionKernels::test_online_update_ptoas[16-128-0-1]'
}

run_core_smoke a5sim
run_fa_smoke a5sim
run_core_smoke a2a3sim
run_fa_smoke a2a3sim
run_pypto_pytest a2a3sim int8_ptoas_codegen \
  tests/st/codegen/dsl/test_batch_matmul_pipeline.py::test_no_mat_to_mat_tmov
