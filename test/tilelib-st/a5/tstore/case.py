#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.

"""Direct public load/store coverage for the PTO-ISA A5 ``tstore`` shapes."""

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


CASE_SPECS = [
    # format, logical 5D shape, physical 5D shape, dtype
    ("nd_f32", "ND", (2, 1, 1, 39, 47), (3, 2, 1, 43, 61), pto.f32, np.float32),
    ("dn_f32", "DN", (1, 1, 1, 4, 21), (1, 1, 1, 8, 32), pto.f32, np.float32),
    ("nz_f32", "NZ", (1, 1, 1, 16, 8), (1, 1, 2, 16, 8), pto.f32, np.float32),
    ("nd_i16", "ND", (1, 2, 1, 23, 121), (3, 2, 2, 35, 125), pto.i16, np.int16),
    ("dn_i8", "DN", (2, 3, 7, 47, 13), (2, 3, 7, 55, 29), pto.i8, np.int8),
]


def _strides(fmt, whole):
    w0, w1, w2, w3, w4 = whole
    if fmt == "DN":
        return [w1 * w2 * w3 * w4, w2 * w3 * w4, w3 * w4, 1, w3]
    return [w1 * w2 * w3 * w4, w2 * w3 * w4, w3 * w4, w4, 1]


def _host_shape(fmt, whole):
    return (*whole[:3], whole[4], whole[3]) if fmt == "DN" else whole


def _tile_shape(fmt, logical, dtype):
    g0, g1, g2, g3, g4 = logical
    if fmt == "ND":
        elem_bytes = np.dtype({pto.f32: np.float32, pto.i16: np.int16, pto.i8: np.int8}[dtype]).itemsize
        cols = ((g4 * elem_bytes + 31) // 32) * 32 // elem_bytes
        return [g0 * g1 * g2 * g3, cols], [g0 * g1 * g2 * g3, g4], {}
    if fmt == "DN":
        elem_bytes = np.dtype({pto.f32: np.float32, pto.i16: np.int16, pto.i8: np.int8}[dtype]).itemsize
        rows = ((g3 * elem_bytes + 31) // 32) * 32 // elem_bytes
        return [rows, g0 * g1 * g2 * g4], [g3, g0 * g1 * g2 * g4], {"blayout": "ColMajor"}
    return [g2 * g3, g0 * g1 * g4], [g2 * g3, g0 * g1 * g4], {
        "blayout": "ColMajor", "slayout": "RowMajor"
    }


def _make_kernel(name, fmt, logical, whole, dtype):
    strides = _strides(fmt, whole)
    tile_shape, valid_shape, tile_kwargs = _tile_shape(fmt, logical, dtype)

    @pto.jit(name="tstore_" + name, target="a5")
    def _kernel(src_ptr: pto.ptr(dtype, "gm"), dst_ptr: pto.ptr(dtype, "gm")):
        src_view = pto.make_tensor_view(src_ptr, shape=list(logical), strides=strides)
        dst_view = pto.make_tensor_view(dst_ptr, shape=list(logical), strides=strides)
        tile = pto.alloc_tile(
            shape=tile_shape, dtype=dtype, valid_shape=valid_shape, **tile_kwargs
        )
        offsets = [0, 0, 0, 0, 0]
        sizes = list(logical)
        pto.tile.load(src_view, tile, offsets=offsets, sizes=sizes)
        pto.tile.store(tile, dst_view, offsets=offsets, sizes=sizes)

    return _kernel


_KERNELS = {
    name: _make_kernel(name, fmt, logical, whole, dtype)
    for name, fmt, logical, whole, dtype, _ in CASE_SPECS
}


def _make_input(name, np_dtype, fmt, whole):
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    return np.random.randint(-5, 5, size=_host_shape(fmt, whole)).astype(np_dtype)


def _expected(src, fmt, logical, whole):
    out = np.zeros_like(src)
    g0, g1, g2, g3, g4 = logical
    if fmt == "DN":
        out[:g0, :g1, :g2, :g4, :g3] = src[:g0, :g1, :g2, :g4, :g3]
    elif fmt == "NZ":
        out[:g0, :g1, :g2, :g3, :g4] = src[:g0, :g1, :g2, :g3, :g4]
    else:
        out[:g0, :g1, :g2, :g3, :g4] = src[:g0, :g1, :g2, :g3, :g4]
    return out


CASES = [
    golden_output_case(
        "tstore_" + name,
        _KERNELS[name],
        inputs=lambda _n=name, _dt=np_dtype, _fmt=fmt, _whole=whole: [
            _make_input(_n, _dt, _fmt, _whole)
        ],
        expected=lambda src, _fmt=fmt, _logical=logical, _whole=whole: _expected(
            src, _fmt, _logical, _whole
        ),
        rtol=0.0,
        atol=0.0,
    )
    for name, fmt, logical, whole, _, np_dtype in CASE_SPECS
]


auto_main(globals())
