#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/trowmin.
#   tload(src) + trowmin(src) -> dst(R, 1) + tstore(dst)
#
# Row reduction: dst[i, 0] = np.min over src[i, valid_cols].  The dst is a
# (R, 1) column vector authored as a ColMajor tile; the source may be
# partially valid (valid_shape may be smaller than shape).

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
    np.int32: pto.i32,
    np.int16: pto.i16,
}

# (case suffix, numpy dtype, shape, valid_shape, eps)
CASE_SPECS = [
    ("f32_127x64_valid127x63", np.float32, (127, 64), (127, 63), 1e-05),
    ("f32_63x64", np.float32, (63, 64), (63, 64), 1e-05),
    ("f32_31x128_valid31x127", np.float32, (31, 128), (31, 127), 1e-05),
    ("f32_15x192", np.float32, (15, 192), (15, 192), 1e-05),
    ("f32_7x448_valid7x447", np.float32, (7, 448), (7, 447), 1e-05),
    ("f16_256x16_valid256x15", np.float16, (256, 16), (256, 15), 0.01),
    ("f32_30x216", np.float32, (30, 216), (30, 216), 1e-05),
    ("f32_30x216_valid30x24", np.float32, (30, 216), (30, 24), 1e-05),
    ("f32_30x216_valid11x216", np.float32, (30, 216), (11, 216), 1e-05),
    ("f32_30x216_valid11x24", np.float32, (30, 216), (11, 24), 1e-05),
    ("f32_238x40", np.float32, (238, 40), (238, 40), 1e-05),
    ("f32_238x40_valid238x16", np.float32, (238, 40), (238, 16), 1e-05),
    ("f32_238x40_valid121x40", np.float32, (238, 40), (121, 40), 1e-05),
    ("f32_238x40_valid121x16", np.float32, (238, 40), (121, 16), 1e-05),
    ("f32_64x128", np.float32, (64, 128), (64, 128), 1e-05),
    ("f32_32x256", np.float32, (32, 256), (32, 256), 1e-05),
    ("f32_16x512", np.float32, (16, 512), (16, 512), 1e-05),
    ("f32_8x1024", np.float32, (8, 1024), (8, 1024), 1e-05),
    ("i32_127x64_valid127x63", np.int32, (127, 64), (127, 63), 0.0),
    ("i32_63x64", np.int32, (63, 64), (63, 64), 0.0),
    ("i32_31x128_valid31x127", np.int32, (31, 128), (31, 127), 0.0),
    ("i32_15x192", np.int32, (15, 192), (15, 192), 0.0),
    ("i32_7x448_valid7x447", np.int32, (7, 448), (7, 447), 0.0),
    ("i16_128x64", np.int16, (128, 64), (128, 64), 0.0),
    ("i16_64x64", np.int16, (64, 64), (64, 64), 0.0),
    ("i16_32x128", np.int16, (32, 128), (32, 128), 0.0),
    ("i16_16x192", np.int16, (16, 192), (16, 192), 0.0),
    ("i16_8x448", np.int16, (8, 448), (8, 448), 0.0),
]


def _make_kernel(name: str, np_dtype, shape, valid_shape):
    rows, cols = shape
    valid_rows, valid_cols = valid_shape
    pto_dtype = NP_TO_PTO[np_dtype]
    # ColMajor tile columns must be 32-byte aligned: align physical rows.
    elem_bytes = np.dtype(np_dtype).itemsize
    aligned_rows = ((rows * elem_bytes + 31) // 32) * 32 // elem_bytes

    @pto.jit(name="trowmin_" + name, target="a5")
    def _kernel(
        src_ptr: pto.ptr(pto_dtype, "gm"),
        dst_ptr: pto.ptr(pto_dtype, "gm"),
    ):
        src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[rows, 1], strides=[1, 1])

        src_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=pto_dtype, valid_shape=[valid_rows, valid_cols]
        )
        dst_tile = pto.alloc_tile(
            shape=[aligned_rows, 1], dtype=pto_dtype, valid_shape=[valid_rows, 1],
            blayout="ColMajor",
        )

        pto.tile.load(src_view, src_tile)
        pto.tile.rowmin(src_tile, dst_tile)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


_kernels = {
    name: _make_kernel(name, np_dtype, shape, valid_shape)
    for name, np_dtype, shape, valid_shape, _ in CASE_SPECS
}


def _make_inputs(name: str, np_dtype, shape, valid_shape):
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    if np.issubdtype(np_dtype, np.integer):
        if np_dtype == np.int32:
            src = np.random.randint(-100, 100, size=shape).astype(np_dtype)
        else:
            src = np.random.randint(-50, 50, size=shape).astype(np_dtype)
    else:
        src = np.random.uniform(-16, 16, size=shape).astype(np_dtype)
    return [src]


def _make_expected(src, valid_shape):
    valid_rows, valid_cols = valid_shape
    golden = np.zeros((valid_rows, 1), dtype=src.dtype)
    golden[:, 0] = np.min(src[:valid_rows, :valid_cols], axis=1).astype(src.dtype)
    return golden


CASES = [
    golden_output_case(
        "trowmin_" + name,
        _kernels[name],
        inputs=lambda name=name, np_dtype=np_dtype, shape=shape, valid_shape=valid_shape: _make_inputs(name, np_dtype, shape, valid_shape),
        expected=lambda src, valid_shape=valid_shape: _make_expected(src, valid_shape),
        output_shape=(valid_shape[0], 1),
        rtol=eps,
        atol=eps,
    )
    for name, np_dtype, shape, valid_shape, eps in CASE_SPECS
]


auto_main(globals())
