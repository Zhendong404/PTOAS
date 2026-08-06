#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of the legacy tinsert Vec->Vec ND cases.

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


CASE_SPECS = [
    ("vec2vec_nd_f16_16x16_into_32x32_idx00", np.float16, pto.f16, 0, 0, 1e-2),
    ("vec2vec_nd_f16_16x16_into_32x32_idx816", np.float16, pto.f16, 8, 16, 1e-2),
    ("vec2vec_nd_f32_16x16_into_32x32_idx00", np.float32, pto.f32, 0, 0, 1e-6),
]


def _make_kernel(name, pto_dtype, index_row, index_col):
    @pto.jit(name="tinsert_" + name, target="a5")
    def _kernel(src_ptr: pto.ptr(pto_dtype, "gm"), dst_ptr: pto.ptr(pto_dtype, "gm"), out_ptr: pto.ptr(pto_dtype, "gm")):
        src_view = pto.make_tensor_view(src_ptr, shape=[1, 1, 1, 16, 16], strides=[256, 256, 256, 16, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[1, 1, 1, 32, 32], strides=[1024, 1024, 1024, 32, 1])
        out_view = pto.make_tensor_view(out_ptr, shape=[1, 1, 1, 32, 32], strides=[1024, 1024, 1024, 32, 1])
        src_tile = pto.alloc_tile(shape=[16, 16], dtype=pto_dtype, valid_shape=[16, 16])
        dst_tile = pto.alloc_tile(shape=[32, 32], dtype=pto_dtype, valid_shape=[32, 32])
        pto.tile.load(src_view, src_tile, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, 16, 16])
        pto.tile.load(dst_view, dst_tile, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, 32, 32])
        pto.tile.insert(src_tile, dst_tile, index_row, index_col)
        pto.tile.store(dst_tile, out_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, 32, 32])
    return _kernel


_kernels = {name: _make_kernel(name, pto_dtype, row, col) for name, _, pto_dtype, row, col, _ in CASE_SPECS}


def _inputs(name, np_dtype):
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    return [
        np.random.uniform(-1.0, 1.0, (16, 16)).astype(np_dtype),
        np.random.uniform(-1.0, 1.0, (32, 32)).astype(np_dtype),
    ]


def _expected(src, dst, row, col):
    result = dst.copy()
    result[row:row + 16, col:col + 16] = src
    return result


CASES = [
    golden_output_case(
        "tinsert_" + name,
        _kernels[name],
        inputs=lambda name=name, np_dtype=np_dtype: _inputs(name, np_dtype),
        expected=lambda src, dst, row=row, col=col: _expected(src, dst, row, col),
        rtol=eps,
        atol=eps,
    )
    for name, np_dtype, _, row, col, eps in CASE_SPECS
]


auto_main(globals())
