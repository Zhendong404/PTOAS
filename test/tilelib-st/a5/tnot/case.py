#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tnot.
#   tload(src) + tnot(src) -> dst + tstore(dst)
# Unary bitwise NOT over signed/unsigned integer dtypes; the uint variants use
# partially-valid tiles (valid 60x60 in a 64x64 physical tile), mirroring the
# legacy cases.

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto

PTO_TO_NP = {
    pto.i8: np.int8,
    pto.ui8: np.uint8,
    pto.i16: np.int16,
    pto.ui16: np.uint16,
    pto.i32: np.int32,
    pto.ui32: np.uint32,
}

ROWS = 64
COLS = 64

# (case suffix, pto dtype, valid_shape)
CASE_SPECS = [
    ("int8_64x64",   pto.i8,  (64, 64)),
    ("uint8_60x60",  pto.ui8, (60, 60)),
    ("int16_64x64",  pto.i16, (64, 64)),
    ("uint16_60x60", pto.ui16, (60, 60)),
    ("int32_64x64",  pto.i32, (64, 64)),
    ("uint32_60x60", pto.ui32, (60, 60)),
]


def _make_kernel(name: str, pto_dtype, valid_shape):
    valid_rows, valid_cols = valid_shape

    @pto.jit(name="tnot_" + name, target="a5")
    def _kernel(
        src_ptr: pto.ptr(pto_dtype, "gm"),
        dst_ptr: pto.ptr(pto_dtype, "gm"),
    ):
        src_view = pto.make_tensor_view(src_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[ROWS, COLS], strides=[COLS, 1])

        src_tile = pto.alloc_tile(
            shape=[ROWS, COLS], dtype=pto_dtype, valid_shape=[valid_rows, valid_cols]
        )
        dst_tile = pto.alloc_tile(
            shape=[ROWS, COLS], dtype=pto_dtype, valid_shape=[valid_rows, valid_cols]
        )

        pto.tile.load(src_view, src_tile)
        pto.tile.bit_not(src_tile, dst_tile)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


_kernels = {name: _make_kernel(name, pto_dtype, valid_shape)
            for name, pto_dtype, valid_shape in CASE_SPECS}


def _make_inputs(name: str, np_dtype, valid_shape):
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    info = np.iinfo(np_dtype)
    src = np.random.randint(info.min, info.max, size=(ROWS, COLS), dtype=np_dtype)
    return [src]


def _make_expected(src, valid_shape):
    valid_rows, valid_cols = valid_shape
    golden = np.zeros(src.shape, dtype=src.dtype)
    golden[:valid_rows, :valid_cols] = np.bitwise_not(src[:valid_rows, :valid_cols])
    return golden


CASES = [
    golden_output_case(
        "tnot_" + name,
        _kernels[name],
        inputs=lambda name=name, np_dtype=PTO_TO_NP[pto_dtype], valid_shape=valid_shape: _make_inputs(name, np_dtype, valid_shape),
        expected=lambda src, valid_shape=valid_shape: _make_expected(src, valid_shape),
        rtol=0,
        atol=0,
    )
    for name, pto_dtype, valid_shape in CASE_SPECS
]


auto_main(globals())
