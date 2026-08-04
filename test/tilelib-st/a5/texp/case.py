#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/texp.
#   tload(src) + texp(src) -> dst + tstore(dst)

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
}

# (case suffix, numpy dtype, shape, valid_shape, eps)
CASE_SPECS = [
    ("f32_16x64", np.float32, (16, 64), (16, 64), 1e-06),
    ("f32_32x32", np.float32, (32, 32), (32, 32), 1e-06),
    ("f16_16x64", np.float16, (16, 64), (16, 64), 0.001),
    ("f16_32x32", np.float16, (32, 32), (32, 32), 0.001),
    ("f32_64x64_hp1", np.float32, (64, 64), (64, 64), 1e-07),
    ("f16_64x64_hp2", np.float16, (64, 64), (64, 64), 1e-07),
]


def _make_kernel(name: str, np_dtype, shape, valid_shape):
    rows, cols = shape
    valid_rows, valid_cols = valid_shape
    pto_dtype = NP_TO_PTO[np_dtype]

    @pto.jit(name="texp_" + name, target="a5")
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
        pto.tile.exp(src_tile, dst_tile)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


_kernels = {
    name: _make_kernel(name, np_dtype, shape, valid_shape)
    for name, np_dtype, shape, valid_shape, _ in CASE_SPECS
}


def _make_inputs(name: str, np_dtype, shape, valid_shape):
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    src = np.random.uniform(-8.0, 0.0, size=shape).astype(np_dtype)
    return [src]


def _make_expected(src, valid_shape):
    valid_rows, valid_cols = valid_shape
    golden = np.zeros(src.shape, dtype=src.dtype)
    golden[:valid_rows, :valid_cols] = np.exp(src)[:valid_rows, :valid_cols]
    return golden


CASES = [
    golden_output_case(
        "texp_" + name,
        _kernels[name],
        inputs=lambda name=name, np_dtype=np_dtype, shape=shape, valid_shape=valid_shape: _make_inputs(name, np_dtype, shape, valid_shape),
        expected=lambda src, valid_shape=valid_shape: _make_expected(src, valid_shape),
        rtol=eps,
        atol=eps,
    )
    for name, np_dtype, shape, valid_shape, eps in CASE_SPECS
]


auto_main(globals())
