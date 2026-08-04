#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tcolargmin.
#   tload(src) + tcolargmin(src, tmp) -> dst(1, C, i32) + tstore(dst)
#
# Column argmin: dst[r=0, c] = index of min over src[valid_rows, c].  The
# legacy cases use partially-valid tiles (valid_shape may be smaller than
# shape and dst_valid_shape == (1, valid_cols)); the kernel writes only the
# valid dst columns and the host golden keeps the invalid tail zeroed, so the
# full-array comparison stays valid.  dst is always int32 indices, and the
# tmp workspace tile mirrors src (same shape / valid_shape / dtype).

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


# (case suffix, pto dtype, numpy dtype, shape, valid_shape, dst_shape, eps)
CASE_SPECS = [
    ("f32_1x256",   pto.f32,  np.float32, (1, 256),  (1, 255),  (1, 256),  0),
    ("f32_16x128",  pto.f32,  np.float32, (16, 128), (16, 127), (1, 128),  0),
    ("f32_16x256",  pto.f32,  np.float32, (16, 256), (15, 255), (1, 256),  0),
    ("f16_1x256",   pto.f16,  np.float16, (1, 256),  (1, 255),  (1, 256),  0),
    ("f16_16x128",  pto.f16,  np.float16, (16, 128), (16, 127), (1, 128),  0),
    ("f16_16x256",  pto.f16,  np.float16, (16, 256), (15, 255), (1, 256),  0),
    ("ui32_1x256",  pto.ui32, np.uint32,  (1, 256),  (1, 255),  (1, 256),  0),
    ("ui32_16x128", pto.ui32, np.uint32,  (16, 128), (16, 127), (1, 128),  0),
    ("ui32_16x256", pto.ui32, np.uint32,  (16, 256), (15, 255), (1, 256),  0),
    ("ui16_1x256",  pto.ui16, np.uint16,  (1, 256),  (1, 255),  (1, 256),  0),
    ("ui16_16x128", pto.ui16, np.uint16,  (16, 128), (16, 127), (1, 128),  0),
    ("ui16_16x256", pto.ui16, np.uint16,  (16, 256), (15, 255), (1, 256),  0),
    ("ui8_1x256",   pto.ui8,  np.uint8,   (1, 256),  (1, 255),  (1, 256),  0),
    ("ui8_16x128",  pto.ui8,  np.uint8,   (16, 128), (16, 127), (1, 128),  0),
    ("ui8_16x256",  pto.ui8,  np.uint8,   (16, 256), (15, 255), (1, 256),  0),
    ("i8_1x256",    pto.i8,   np.int8,    (1, 256),  (1, 255),  (1, 256),  0),
    ("i8_16x128",   pto.i8,   np.int8,    (16, 128), (16, 127), (1, 128),  0),
    ("i8_16x256",   pto.i8,   np.int8,    (16, 256), (15, 255), (1, 256),  0),
]


def _make_kernel(name: str, pto_dtype, shape, valid_shape, dst_shape):
    rows, cols = shape
    valid_rows, valid_cols = valid_shape
    dst_rows, dst_cols = dst_shape

    @pto.jit(name="tcolargmin_" + name, target="a5")
    def _kernel(
        src_ptr: pto.ptr(pto_dtype, "gm"),
        dst_ptr: pto.ptr(pto.i32, "gm"),
    ):
        src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[dst_rows, dst_cols], strides=[dst_cols, 1])

        src_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=pto_dtype, valid_shape=[valid_rows, valid_cols]
        )
        tmp_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=pto_dtype, valid_shape=[valid_rows, valid_cols]
        )
        dst_tile = pto.alloc_tile(
            shape=[dst_rows, dst_cols], dtype=pto.i32, valid_shape=[1, valid_cols]
        )

        pto.tile.load(src_view, src_tile)
        pto.tile.colargmin(src_tile, tmp_tile, dst_tile)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


_kernels = {name: _make_kernel(name, pto_dtype, shape, valid_shape, dst_shape)
            for name, pto_dtype, _, shape, valid_shape, dst_shape, _ in CASE_SPECS}


def _make_inputs(name: str, shape, np_dtype):
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    src = np.random.randint(1, 10, size=shape).astype(np_dtype)
    return [src]


def _make_expected(src, valid_shape, dst_shape):
    valid_rows, valid_cols = valid_shape
    golden = np.zeros(dst_shape, dtype=np.int32)
    golden[:1, :valid_cols] = np.argmin(src[:valid_rows, :valid_cols], axis=0, keepdims=True)
    return golden


CASES = [
    golden_output_case(
        "tcolargmin_" + name,
        _kernels[name],
        inputs=lambda name=name, shape=shape, np_dtype=np_dtype: _make_inputs(name, shape, np_dtype),
        expected=lambda src, valid_shape=valid_shape, dst_shape=dst_shape: _make_expected(src, valid_shape, dst_shape),
        rtol=eps,
        atol=eps,
    )
    for name, _, np_dtype, shape, valid_shape, dst_shape, eps in CASE_SPECS
]


auto_main(globals())
