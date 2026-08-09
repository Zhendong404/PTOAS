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
    pto.i16: np.int16,
    pto.i8: np.int8,
}


# These are the PTO-ISA cases without the optional isBinary/tmp form.
CASE_SPECS = [
    ("f32_1x256_valid255", pto.f32, 1, 1, 256, 255),
    ("f32_16x128_valid127", pto.f32, 16, 16, 128, 127),
    ("f32_16x256_valid15x255", pto.f32, 16, 15, 256, 255),
    ("i16_1x256_valid255", pto.i16, 1, 1, 256, 255),
    ("i16_16x128_valid127", pto.i16, 16, 16, 128, 127),
    ("i16_16x256_valid15x255", pto.i16, 16, 15, 256, 255),
    ("i8_1x256_valid255", pto.i8, 1, 1, 256, 255),
    ("i8_16x128_valid127", pto.i8, 16, 16, 128, 127),
    ("i8_16x256_valid15x255", pto.i8, 16, 15, 256, 255),
]

BINARY_CASE_SPECS = [
    ("f32_binary_64x128_valid63x127", pto.f32, 64, 63, 128, 127),
    ("f32_binary_64x128_valid64x128", pto.f32, 64, 64, 128, 128),
]


def _make_kernel(name, dtype, rows, valid_rows, cols, valid_cols):
    @pto.jit(name="tcolsum_" + name, target="a5")
    def _kernel(src_ptr: pto.ptr(dtype, "gm"), dst_ptr: pto.ptr(dtype, "gm")):
        src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[1, cols], strides=[cols, 1])
        src_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=dtype, valid_shape=[valid_rows, valid_cols]
        )
        dst_tile = pto.alloc_tile(
            shape=[1, cols], dtype=dtype, valid_shape=[1, valid_cols]
        )
        pto.tile.load(src_view, src_tile)
        pto.tile.colsum(src_tile, dst_tile)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


def _make_binary_kernel(name, dtype, rows, valid_rows, cols, valid_cols):
    @pto.jit(name="tcolsum_" + name, target="a5")
    def _kernel(src_ptr: pto.ptr(dtype, "gm"), dst_ptr: pto.ptr(dtype, "gm")):
        src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[1, cols], strides=[cols, 1])
        src_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=dtype, valid_shape=[valid_rows, valid_cols]
        )
        tmp_tile = pto.alloc_tile(
            shape=[(rows + 1) // 2, cols],
            dtype=dtype,
            valid_shape=[(valid_rows + 1) // 2, valid_cols],
        )
        dst_tile = pto.alloc_tile(
            shape=[1, cols], dtype=dtype, valid_shape=[1, valid_cols]
        )
        pto.tile.load(src_view, src_tile)
        pto.tile.colsum(src_tile, dst_tile, tmp=tmp_tile, is_binary=True)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


_KERNELS = {
    name: _make_kernel(name, dtype, rows, valid_rows, cols, valid_cols)
    for name, dtype, rows, valid_rows, cols, valid_cols in CASE_SPECS
}
_KERNELS.update(
    {
        name: _make_binary_kernel(name, dtype, rows, valid_rows, cols, valid_cols)
        for name, dtype, rows, valid_rows, cols, valid_cols in BINARY_CASE_SPECS
    }
)


def _make_inputs(name, dtype, rows, cols):
    import zlib

    np_dtype = PTO_TO_NP_DTYPE[dtype]
    np.random.seed(zlib.crc32(("TCOLSUMTest." + name).encode()) & 0xFFFFFFFF)
    if dtype is pto.i8:
        values = np.random.uniform(-5, 5, size=(rows, cols))
    else:
        values = np.random.uniform(-1, 1, size=(rows, cols))
    return [values.astype(np_dtype)]


def _make_expected(src, dtype, valid_rows, valid_cols):
    np_dtype = PTO_TO_NP_DTYPE[dtype]
    result = np.zeros((1, src.shape[1]), dtype=np_dtype)
    result[0, :valid_cols] = np.sum(src[:valid_rows, :valid_cols], axis=0).astype(np_dtype)
    return result


CASES = [
    golden_output_case(
        "tcolsum_" + name,
        _KERNELS[name],
        inputs=lambda _name=name, _dtype=dtype, _rows=rows, _cols=cols: _make_inputs(
            _name, _dtype, _rows, _cols
        ),
        expected=lambda src, _dtype=dtype, _vr=valid_rows, _vc=valid_cols: _make_expected(
            src, _dtype, _vr, _vc
        ),
        output_shape=(1, cols),
        output_dtype=PTO_TO_NP_DTYPE[dtype],
        rtol=1e-3,
        atol=1e-3,
    )
    for name, dtype, rows, valid_rows, cols, valid_cols in CASE_SPECS
]
CASES.extend(
    golden_output_case(
        "tcolsum_" + name,
        _KERNELS[name],
        inputs=lambda _name=name, _dtype=dtype, _rows=rows, _cols=cols: _make_inputs(
            _name, _dtype, _rows, _cols
        ),
        expected=lambda src, _dtype=dtype, _vr=valid_rows, _vc=valid_cols: _make_expected(
            src, _dtype, _vr, _vc
        ),
        output_shape=(1, cols),
        output_dtype=PTO_TO_NP_DTYPE[dtype],
        rtol=1e-3,
        atol=1e-3,
    )
    for name, dtype, rows, valid_rows, cols, valid_cols in BINARY_CASE_SPECS
)


auto_main(globals())
