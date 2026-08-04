#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tands.
#   tload(src) + tands(src, 3) -> dst + tstore(dst)

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
    np.uint8: pto.ui8,
    np.int8: pto.i8,
    np.int32: pto.i32,
    np.int16: pto.i16,
    np.uint16: pto.ui16,
    np.uint32: pto.ui32,
}

SCALAR = 3

# (case suffix, numpy dtype, shape, valid_shape, eps)
CASE_SPECS = [
    ("i32_32x64", np.int32, (32, 64), (32, 64), 0.0, None),
    ("i16_63x64", np.int16, (63, 64), (63, 64), 0.0, None),
    ("i32_31x128", np.int32, (31, 128), (31, 128), 0.0, None),
    ("i16_15x192", np.int16, (15, 192), (15, 192), 0.0, None),
]


def _make_kernel(name: str, np_dtype, shape, valid_shape):
    rows, cols = shape
    valid_rows, valid_cols = valid_shape
    pto_dtype = NP_TO_PTO[np_dtype]
    scalar_val = int(SCALAR) if np.issubdtype(np_dtype, np.integer) else float(SCALAR)

    @pto.jit(name="tands_" + name, target="a5")
    def _kernel(
        src_ptr: pto.ptr(pto_dtype, "gm"),
        dst_ptr: pto.ptr(pto_dtype, "gm"),
    ):
        src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[rows, cols], strides=[cols, 1])

        src_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=pto_dtype, valid_shape=[valid_rows, valid_cols]
        )
        dst_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=pto_dtype, valid_shape=[valid_rows, valid_cols]
        )

        pto.tile.load(src_view, src_tile)
        pto.tile.bit_ands(src_tile, scalar_val, dst_tile)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


_kernels = {
    name: _make_kernel(name, np_dtype, shape, valid_shape)
    for name, np_dtype, shape, valid_shape, _, _ in CASE_SPECS
}


def _make_inputs(name: str, np_dtype, shape, valid_shape):
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    if np.issubdtype(np_dtype, np.integer):
        if np_dtype == np.int32:
            src = np.random.randint(-100, 100, size=shape).astype(np_dtype)
        else:
            src = np.random.randint(0, 100, size=shape).astype(np_dtype)
    else:
        src = np.random.uniform(-10, 10, size=shape).astype(np_dtype)
    return [src]


def _make_expected(src, valid_shape, direction):
    valid_rows, valid_cols = valid_shape
    golden = np.zeros(src.shape, dtype=src.dtype)
    if direction == "scalar_src":
        golden[:valid_rows, :valid_cols] = (src.dtype.type(SCALAR) / src)[:valid_rows, :valid_cols]
    else:
        golden[:valid_rows, :valid_cols] = (src & src.dtype.type(3))[:valid_rows, :valid_cols]
    return golden


CASES = [
    golden_output_case(
        "tands_" + name,
        _kernels[name],
        inputs=lambda name=name, np_dtype=np_dtype, shape=shape, valid_shape=valid_shape: _make_inputs(name, np_dtype, shape, valid_shape),
        expected=lambda src, valid_shape=valid_shape, direction=direction: _make_expected(src, valid_shape, direction),
        rtol=eps,
        atol=eps,
    )
    for name, np_dtype, shape, valid_shape, eps, direction in CASE_SPECS
]


auto_main(globals())
