#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib ST case for ``pto.ttrans`` (A5 2D tile transpose).

Port of test/tilelang_st/npu/a5/src/st/testcase/ttrans from the pto-isa repo.
Each case loads a source tile, transposes it via ``pto.tile.transpose``, and
stores the result; the golden is a numpy ``.T``.
"""

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


# (name, dtype, physical source rows, physical source cols, valid source rows,
# valid source cols, physical destination rows, physical destination cols).
# The physical major dimension is 32-byte aligned, as required by the current
# A5 templates. The valid dimensions preserve legacy partial-tile cases.
CASE_SHAPES = [
    ("f32_8x64", pto.f32, 8, 64, 8, 64, 64, 8),
    ("f32_64x8", pto.f32, 64, 8, 64, 8, 8, 64),
    ("f32_32x32", pto.f32, 32, 32, 32, 32, 32, 32),
    ("i32_8x64", pto.i32, 8, 64, 8, 64, 64, 8),
    ("f16_64x16", pto.f16, 64, 16, 64, 16, 16, 64),
    ("f16_16x32_valid15x31", pto.f16, 16, 32, 15, 31, 32, 16),
    ("f32_16x64_valid15x63", pto.f32, 16, 64, 15, 63, 64, 16),
    ("f32_2x8", pto.f32, 2, 8, 2, 8, 8, 8),
    ("f16_16x16", pto.f16, 16, 16, 16, 16, 16, 16),
    ("f32_4x8", pto.f32, 4, 8, 4, 8, 8, 8),
    ("f32_8x8", pto.f32, 8, 8, 8, 8, 8, 8),
    ("f32_1x8", pto.f32, 1, 8, 1, 8, 8, 8),
    ("i8_32x64", pto.i8, 32, 64, 32, 64, 64, 32),
    ("ui8_64x32", pto.ui8, 64, 32, 64, 32, 32, 64),
    ("hif8_32x32", pto.hif8, 32, 32, 32, 32, 32, 32),
    ("hif8_32x64", pto.hif8, 32, 64, 32, 64, 64, 32),
    ("hif8_64x64_valid22x63", pto.hif8, 64, 64, 22, 63, 64, 64),
    ("f8e4m3_32x32", pto.f8e4m3, 32, 32, 32, 32, 32, 32),
    ("f8e4m3_32x64", pto.f8e4m3, 32, 64, 32, 64, 64, 32),
    ("f8e4m3_64x64_valid22x63", pto.f8e4m3, 64, 64, 22, 63, 64, 64),
    ("f8e5m2_32x32", pto.f8e5m2, 32, 32, 32, 32, 32, 32),
    ("f8e5m2_32x64", pto.f8e5m2, 32, 64, 32, 64, 64, 32),
    ("f8e5m2_64x64_valid22x63", pto.f8e5m2, 64, 64, 22, 63, 64, 64),
]


def _ttrans_body(src_ptr, dst_ptr, *, rows, cols, valid_rows, valid_cols,
                 dst_rows, dst_cols, dtype):
    """Shared kernel body: load src, transpose into dst (with tmp), store."""
    src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[dst_rows, dst_cols], strides=[dst_cols, 1])

    src_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype,
                              valid_shape=[valid_rows, valid_cols])
    tmp_tile = pto.alloc_tile(shape=[dst_rows, dst_cols], dtype=dtype)
    dst_tile = pto.alloc_tile(shape=[dst_rows, dst_cols], dtype=dtype,
                              valid_shape=[valid_cols, valid_rows])

    pto.tile.load(src_view, src_tile)
    pto.tile.transpose(src_tile, tmp_tile, dst_tile)
    pto.tile.store(dst_tile, dst_view)


_ttrans_kernels = {}
for _name, _dtype, _rows, _cols, _vr, _vc, _dr, _dc in CASE_SHAPES:
    _r, _c, _valid_r, _valid_c, _dst_r, _dst_c = _rows, _cols, _vr, _vc, _dr, _dc

    def _make(r=_r, c=_c, valid_r=_valid_r, valid_c=_valid_c,
              dst_r=_dst_r, dst_c=_dst_c, dtype=_dtype,
              kernel_name=f"ttrans_{_name}"):
        @pto.jit(name=kernel_name, target="a5")
        def _kernel(
            src_ptr: pto.ptr(dtype, "gm"),
            dst_ptr: pto.ptr(dtype, "gm"),
        ):
            _ttrans_body(src_ptr, dst_ptr, rows=r, cols=c,
                          valid_rows=valid_r, valid_cols=valid_c,
                          dst_rows=dst_r, dst_cols=dst_c, dtype=dtype)

        return _kernel

    _ttrans_kernels[_name] = _make()


# numpy dtype mapping for input generation + golden
_PTO_TO_NP = {
    pto.f32: np.float32, pto.f16: np.float16,
    pto.i32: np.int32,
    pto.i8: np.int8, pto.ui8: np.uint8,
    pto.hif8: np.uint8, pto.f8e4m3: np.uint8, pto.f8e5m2: np.uint8,
}


def _make_inputs(name, dtype, rows, cols):
    import zlib
    np_dtype = _PTO_TO_NP[dtype]
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    # use small ints in float range to avoid f16 overflow / bf16 precision loss
    a = np.random.randint(1, 10, size=(rows, cols)).astype(np_dtype)
    return [a]


def _make_expected(a, valid_rows, valid_cols, dst_rows, dst_cols):
    result = np.zeros((dst_rows, dst_cols), dtype=a.dtype)
    result[:valid_cols, :valid_rows] = a[:valid_rows, :valid_cols].T
    return result


CASES = []
for _name, _dtype, _rows, _cols, _vr, _vc, _dr, _dc in CASE_SHAPES:
    CASES.append(
        golden_output_case(
            "ttrans_" + _name,
            _ttrans_kernels[_name],
            inputs=lambda _n=_name, _d=_dtype, _r=_rows, _c=_cols: _make_inputs(_n, _d, _r, _c),
            expected=lambda src, _vr=_vr, _vc=_vc, _dr=_dr, _dc=_dc: _make_expected(
                src, _vr, _vc, _dr, _dc
            ),
            output_shape=(_dr, _dc),
            output_dtype=_PTO_TO_NP[_dtype],
            rtol=1e-3,  # f16/bf16 need looser tol
            atol=1e-3,
        )
    )


auto_main(globals())
