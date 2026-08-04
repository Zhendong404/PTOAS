#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tdiv.
#   tload(a) + tload(b) + pto.tile.div(a, b) -> c + tstore(c)
#
# All 17 legacy cases are preserved (dtype / shape / valid_shape / eps /
# test_pattern / high_precision).  High-precision cases pass
# precision=pto.Precision.HighPrecision, mirroring the legacy
# {precisionType = #pto<div_precision high_precision>} attribute.  The RNG is
# seeded with crc32(legacy_name) exactly like st_common.setup_case_rng, and the
# input generators / numpy golden are copied from the legacy gen_data.py so
# data and expected values match the original suite.

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto

NP_TO_PTO = {
    np.float32: pto.f32,
    np.float16: pto.f16,
}

# (legacy case name, numpy dtype, shape, valid_shape, eps, test_pattern,
#  precision_type)
# test_pattern selects the same data generator as the legacy gen_data.py;
# precision_type "high_precision" selects the high-precision div algorithm.
CASE_SPECS = [
    ("f32_16x64", np.float32, (16, 64), (16, 64), 1e-6, "normal", None),
    ("f32_32x32", np.float32, (32, 32), (32, 32), 1e-6, "normal", None),
    ("f32_64x64", np.float32, (64, 64), (64, 64), 1e-6, "normal", None),
    ("f16_16x256", np.float16, (16, 256), (16, 256), 1e-3, "normal", None),
    ("f32_16x64_hp_precision", np.float32, (16, 64), (16, 64), 1e-6, "precision_sensitive", "high_precision"),
    ("f16_16x64_hp_precision", np.float16, (16, 64), (16, 64), 1e-3, "precision_sensitive", "high_precision"),
    ("f32_16x64_hp_subnormal", np.float32, (16, 64), (16, 64), 1e-6, "subnormal", "high_precision"),
    ("f16_16x64_hp_subnormal", np.float16, (16, 64), (16, 64), 1e-3, "subnormal", "high_precision"),
    ("f32_16x64_hp_overflow", np.float32, (16, 64), (16, 64), 1e-6, "overflow", "high_precision"),
    ("f16_16x64_hp_overflow", np.float16, (16, 64), (16, 64), 1e-3, "overflow", "high_precision"),
    ("f32_32x32_hp", np.float32, (32, 32), (32, 32), 1e-5, "precision_sensitive", "high_precision"),
    ("f32_64x64_hp", np.float32, (64, 64), (64, 64), 1e-5, "precision_sensitive", "high_precision"),
    ("f16_16x256_hp", np.float16, (16, 256), (16, 256), 1e-3, "precision_sensitive", "high_precision"),
    ("f32_16x64_hp_partial", np.float32, (16, 64), (16, 31), 1e-5, "precision_sensitive", "high_precision"),
    ("f16_16x64_hp_partial", np.float16, (16, 64), (16, 63), 1e-3, "precision_sensitive", "high_precision"),
    ("f32_2x16_hp", np.float32, (2, 16), (2, 16), 1e-6, "precision_sensitive", "high_precision"),
    ("f16_2x32_hp", np.float16, (2, 32), (2, 32), 1e-3, "precision_sensitive", "high_precision"),
]


def _make_kernel(name, np_dtype, shape, valid_shape, high_precision):
    rows, cols = shape
    valid_rows, valid_cols = valid_shape
    pto_dtype = NP_TO_PTO[np_dtype]

    @pto.jit(name="tdiv_" + name, target="a5")
    def _kernel(
        a_ptr: pto.ptr(pto_dtype, "gm"),
        b_ptr: pto.ptr(pto_dtype, "gm"),
        c_ptr: pto.ptr(pto_dtype, "gm"),
    ):
        a_view = pto.make_tensor_view(a_ptr, shape=[rows, cols], strides=[cols, 1])
        b_view = pto.make_tensor_view(b_ptr, shape=[rows, cols], strides=[cols, 1])
        c_view = pto.make_tensor_view(c_ptr, shape=[rows, cols], strides=[cols, 1])

        a_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=pto_dtype, valid_shape=[valid_rows, valid_cols]
        )
        b_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=pto_dtype, valid_shape=[valid_rows, valid_cols]
        )
        c_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=pto_dtype, valid_shape=[valid_rows, valid_cols]
        )

        pto.tile.load(a_view, a_tile)
        pto.tile.load(b_view, b_tile)
        if high_precision:
            pto.tile.div(a_tile, b_tile, c_tile, precision=pto.Precision.HighPrecision)
        else:
            pto.tile.div(a_tile, b_tile, c_tile)
        pto.tile.store(c_tile, c_view)

    return _kernel


_kernels = {
    name: _make_kernel(name, np_dtype, shape, valid_shape, precision_type == "high_precision")
    for name, np_dtype, shape, valid_shape, _, _, precision_type in CASE_SPECS
}


# ---------------------------------------------------------------------------
# Data generators, copied from the legacy gen_data.py to preserve RNG parity.
# Each case seeds np.random with crc32(legacy case name) as st_common.setup_case_rng did.
# ---------------------------------------------------------------------------

def _generate_normal_data(shape, dtype):
    input1 = np.random.uniform(0.1, 100.0, size=shape).astype(dtype)
    input2 = np.random.uniform(0.1, 100.0, size=shape).astype(dtype)
    return input1, input2


def _generate_precision_sensitive_data(shape, dtype):
    rows, cols = shape
    input1 = np.zeros(shape, dtype=dtype)
    input2 = np.ones(shape, dtype=dtype)

    ratios = [(1, 3), (1, 7), (7, 3), (1, 11), (5, 3), (10, 3)]

    section_size = rows // len(ratios)
    for i, (a, b) in enumerate(ratios):
        start_row = i * section_size
        end_row = min((i + 1) * section_size, rows)
        input1[start_row:end_row, :] = dtype(a)
        input2[start_row:end_row, :] = dtype(b)

    remaining_rows = rows - len(ratios) * section_size
    if remaining_rows > 0:
        input1[-remaining_rows:, :] = np.random.choice([-1, 1], size=(remaining_rows, cols)).astype(dtype)
        input2[-remaining_rows:, :] = dtype(3)

    return input1, input2


def _generate_subnormal_test_data(shape, dtype):
    rows, cols = shape
    input1 = np.zeros(shape, dtype=dtype)
    input2 = np.ones(shape, dtype=dtype)

    if dtype == np.float32:
        tiny = np.finfo(np.float32).tiny
        subnormal_max = np.frombuffer(np.array([0x007FFFFF], dtype=np.uint32), dtype=np.float32)[0]
        subnormal_min = np.float32(1e-45)
        normal_min = tiny * np.float32(2.0)
    else:  # float16
        tiny = np.finfo(np.float16).tiny
        subnormal_max = np.frombuffer(np.array([0x03FF], dtype=np.uint16), dtype=np.float16)[0]
        subnormal_min = np.float16(1e-8)
        normal_min = tiny * np.float16(2.0)

    quarter = rows // 4

    input1[:quarter, :] = subnormal_max
    input2[:quarter, :] = np.random.uniform(normal_min, 100.0, size=(quarter, cols)).astype(dtype)

    input1[quarter:2 * quarter, :] = subnormal_max
    input2[quarter:2 * quarter, :] = np.random.uniform(subnormal_max * 0.1, subnormal_max,
                                                      size=(quarter, cols)).astype(dtype)

    input1[2 * quarter:3 * quarter, :] = subnormal_max
    input2[2 * quarter:3 * quarter, :] = np.random.uniform(subnormal_min, subnormal_max * 0.1,
                                                          size=(quarter, cols)).astype(dtype)

    input1[3 * quarter:, :] = np.random.uniform(0.1, 100.0, size=(rows - 3 * quarter, cols)).astype(dtype)
    input2[3 * quarter:, :] = np.random.uniform(0.1, 100.0, size=(rows - 3 * quarter, cols)).astype(dtype)

    return input1, input2


def _generate_overflow_test_data(shape, dtype):
    rows, cols = shape
    input1 = np.zeros(shape, dtype=dtype)
    input2 = np.ones(shape, dtype=dtype)

    if dtype == np.float32:
        large_val = np.float32(1e30)
        tiny_val = np.float32(1e-30)
        overflow_trigger = np.float32(1e38)
        underflow_trigger = np.float32(1e-45)
        max_normal = np.float32(3.4e38)
    else:  # float16
        large_val = np.float16(60000)
        tiny_val = np.float16(0.0001)
        overflow_trigger = np.float16(65000)
        underflow_trigger = np.float16(1e-7)
        max_normal = np.float16(65504)

    quarter = rows // 4
    input1[:quarter, :cols // 2] = overflow_trigger
    input2[:quarter, :cols // 2] = tiny_val

    input1[:quarter, cols // 2:] = large_val
    input2[:quarter, cols // 2:] = np.random.uniform(1e-35 if dtype == np.float32 else 1e-7,
                                                     tiny_val,
                                                     size=(quarter, cols // 2)).astype(dtype)

    input1[quarter:2 * quarter, :cols // 2] = underflow_trigger
    input2[quarter:2 * quarter, :cols // 2] = large_val

    input1[quarter:2 * quarter, cols // 2:] = tiny_val
    input2[quarter:2 * quarter, cols // 2:] = np.random.uniform(large_val, max_normal,
                                                               size=(quarter, cols // 2)).astype(dtype)

    input1[2 * quarter:3 * quarter, :] = np.random.uniform(large_val / 10, max_normal,
                                                          size=(quarter, cols)).astype(dtype)
    input2[2 * quarter:3 * quarter, :] = np.random.uniform(tiny_val / 10, tiny_val,
                                                          size=(quarter, cols)).astype(dtype)

    input1[3 * quarter:, :] = np.random.uniform(0.1, 100.0,
                                                size=(rows - 3 * quarter, cols)).astype(dtype)
    input2[3 * quarter:, :] = np.random.uniform(0.1, 100.0,
                                                size=(rows - 3 * quarter, cols)).astype(dtype)

    return input1, input2


_DATA_GENERATORS = {
    "normal": _generate_normal_data,
    "precision_sensitive": _generate_precision_sensitive_data,
    "subnormal": _generate_subnormal_test_data,
    "overflow": _generate_overflow_test_data,
}


def _make_inputs(name, np_dtype, shape, test_pattern):
    # Same per-case deterministic seed as st_common.setup_case_rng.
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    input1, input2 = _DATA_GENERATORS[test_pattern](shape, np_dtype)
    return [input1, input2]


def _make_expected(input1, input2, valid_shape):
    vr, vc = valid_shape
    golden = np.zeros(input1.shape, dtype=input1.dtype)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        golden[:vr, :vc] = (input1[:vr, :vc] / input2[:vr, :vc]).astype(input1.dtype, copy=False)
    return golden


CASES = [
    golden_output_case(
        "tdiv_" + name,
        _kernels[name],
        inputs=lambda name=name, np_dtype=np_dtype, shape=shape, test_pattern=test_pattern: _make_inputs(name, np_dtype, shape, test_pattern),
        expected=lambda input1, input2, valid_shape=valid_shape: _make_expected(input1, input2, valid_shape),
        rtol=eps,
        atol=eps,
    )
    for name, np_dtype, shape, valid_shape, eps, test_pattern, _precision_type in CASE_SPECS
]


auto_main(globals())
