#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.  You may not use this file except in compliance with the License.

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


PTO_TO_NP_DTYPE = {
    pto.f32: np.float32,
    pto.f16: np.float16,
    pto.i32: np.int32,
    pto.i16: np.int16,
    pto.i8: np.int8,
}


# The i64/ui64 legacy variants are intentionally omitted: the public A5
# tcolexpand template currently supports i8/i16/i32/f16/bf16/f32 only.
CASE_SPECS = [
    ("f16_1x16x512", pto.f16, 1, 16, 512, 512),
    ("i8_2x32x256_valid255", pto.i8, 2, 32, 256, 255),
    ("f32_1x8x128_valid63", pto.f32, 1, 8, 128, 63),
    ("f16_1x33x512", pto.f16, 1, 33, 512, 512),
    ("i8_2x17x256_valid44", pto.i8, 2, 17, 256, 44),
    ("f32_1x54x64_valid63", pto.f32, 1, 54, 64, 63),
]


def _make_kernel(name, dtype, src_rows, dst_rows, cols, valid_cols):
    @pto.jit(name="tcolexpand_" + name, target="a5")
    def _kernel(src_ptr: pto.ptr(dtype, "gm"), dst_ptr: pto.ptr(dtype, "gm")):
        src_view = pto.make_tensor_view(src_ptr, shape=[src_rows, cols], strides=[cols, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[dst_rows, cols], strides=[cols, 1])
        src_tile = pto.alloc_tile(
            shape=[src_rows, cols], dtype=dtype, valid_shape=[src_rows, valid_cols]
        )
        dst_tile = pto.alloc_tile(
            shape=[dst_rows, cols], dtype=dtype, valid_shape=[dst_rows, valid_cols]
        )
        pto.tile.load(src_view, src_tile)
        pto.tile.colexpand(src_tile, dst_tile)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


_KERNELS = {
    name: _make_kernel(name, dtype, src_rows, dst_rows, cols, valid_cols)
    for name, dtype, src_rows, dst_rows, cols, valid_cols in CASE_SPECS
}


def _make_inputs(name, dtype, src_rows, cols):
    import zlib

    np.random.seed(zlib.crc32(("TCOLEXPANDTest.case_" + name).encode()) & 0xFFFFFFFF)
    return [
        (np.random.rand(src_rows, cols) * 10).astype(PTO_TO_NP_DTYPE[dtype])
    ]


def _make_expected(src, dtype, dst_rows, valid_cols):
    result = np.zeros((dst_rows, src.shape[1]), dtype=PTO_TO_NP_DTYPE[dtype])
    result[:, :valid_cols] = src[0, :valid_cols]
    return result


CASES = [
    golden_output_case(
        "tcolexpand_" + name,
        _KERNELS[name],
        inputs=lambda _name=name, _dtype=dtype, _rows=src_rows, _cols=cols: _make_inputs(
            _name, _dtype, _rows, _cols
        ),
        expected=lambda src, _dtype=dtype, _dst_rows=dst_rows, _valid_cols=valid_cols: _make_expected(
            src, _dtype, _dst_rows, _valid_cols
        ),
        output_shape=(dst_rows, cols),
        output_dtype=PTO_TO_NP_DTYPE[dtype],
        rtol=1e-3 if dtype is pto.f16 else 1e-6,
        atol=1e-3 if dtype is pto.f16 else 1e-6,
    )
    for name, dtype, src_rows, dst_rows, cols, valid_cols in CASE_SPECS
]


auto_main(globals())
