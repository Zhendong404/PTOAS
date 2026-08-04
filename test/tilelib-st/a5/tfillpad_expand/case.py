#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tfillpad_expand
# (legacy .pto TFILLPAD_EXPAND_u16_260x32_src_259x7 / TFILLPAD_EXPAND_s8_260x64_src_259x7).
#
#   tload(src) + tfillpad_expand(src) -> dst + tstore(dst)
#
# TFILLPAD_EXPAND semantics (auto mode, pto.tile.fillpad_expand):
#   1. copy src.valid_shape data into dst
#   2. fill cols from src.valid_cols to dst.valid_cols with the dst pad value
#   3. fill rows from src.rows to dst.rows with the dst pad value
#
# The src tile is allocated at the dst physical size with valid_shape=(src rows,
# src cols), mirroring the legacy .pto tile_buf (e.g. tile_buf<vec, 260x32xui16,
# valid=259x7, pad=3>); the dst tile carries the fill pad value (pad=2/Max in the
# legacy encoding 0=Null 1=Zero 2=Max 3=Min).  Legacy dtype / shape / valid_shape
# / layout / pad params / eps and the randint(1, 10) draw order are preserved.

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto

PTO_TO_NP = {
    pto.ui16: np.uint16,
    pto.i8: np.int8,
}

# PadValue enum names matching the legacy cases.py / gen_data.py.
PADVAL_MAX = "Max"     # FLT_MAX for float, max for integers
PADVAL_MIN = "Min"     # -FLT_MAX for float, min for integers
PADVAL_NEG1 = "Neg1"   # -1.0f / -1 (unused here, kept for parity)
PADVAL_ZERO = "Zero"   # 0

# (legacy case name, pto dtype, src shape, src valid_shape, dst shape,
#  dst valid_shape, load_padval, fill_padval, eps).
# shape/valid_shape follow the legacy cases.py table: shape is the src GM input
# (259x7), dst_shape/dst_valid_shape the output (260x32 / 260x64).
CASE_SPECS = [
    ("u16_260x32_src_259x7", pto.ui16, (259, 7), (259, 7), (260, 32), (260, 32), PADVAL_MIN, PADVAL_MAX, 0),
    ("s8_260x64_src_259x7",   pto.i8,   (259, 7), (259, 7), (260, 64), (260, 64), PADVAL_MIN, PADVAL_MAX, 0),
]


def _make_kernel(name, pto_dtype, src_rows, src_cols, dst_rows, dst_cols,
                 dst_valid_rows, dst_valid_cols, load_pad, fill_pad):
    @pto.jit(name="tfillpad_expand_" + name, target="a5")
    def _kernel(
        src_ptr: pto.ptr(pto_dtype, "gm"),
        dst_ptr: pto.ptr(pto_dtype, "gm"),
    ):
        # Src view spans the GM input (src physical rows x cols); the src tile is
        # allocated at the dst physical size with valid_shape=(src rows, src cols)
        # and the TLOAD pad value, mirroring the legacy .pto alloc_tile.
        src_view = pto.make_tensor_view(src_ptr, shape=[src_rows, src_cols], strides=[src_cols, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[dst_rows, dst_cols], strides=[dst_cols, 1])

        src_tile = pto.alloc_tile(
            shape=[dst_rows, dst_cols], dtype=pto_dtype,
            valid_shape=[src_rows, src_cols],
            pad=load_pad,
        )
        dst_tile = pto.alloc_tile(
            shape=[dst_rows, dst_cols], dtype=pto_dtype,
            valid_shape=[dst_valid_rows, dst_valid_cols],
            pad=fill_pad,
        )

        pto.tile.load(src_view, src_tile)
        pto.tile.fillpad_expand(src_tile, dst_tile)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


_kernels = {
    name: _make_kernel(name, pto_dtype, src_rows, src_cols, dst_rows, dst_cols,
                       dst_valid_rows, dst_valid_cols, load_pad, fill_pad)
    for name, pto_dtype, src_shape, _, dst_shape, dst_valid_shape, load_pad, fill_pad, _ in CASE_SPECS
    for src_rows, src_cols in [src_shape]
    for dst_rows, dst_cols in [dst_shape]
    for dst_valid_rows, dst_valid_cols in [dst_valid_shape]
}


def get_pad_value(dtype, padval_name):
    """Actual pad scalar for a dtype + PadValue name (mirrors legacy gen_data.py)."""
    if padval_name == PADVAL_MAX:
        if np.issubdtype(dtype, np.floating):
            return np.float32(np.finfo(np.float32).max)
        return np.iinfo(dtype).max
    if padval_name == PADVAL_MIN:
        if np.issubdtype(dtype, np.floating):
            return np.float32(-np.finfo(np.float32).max)
        return np.iinfo(dtype).min
    if padval_name == PADVAL_NEG1:
        if np.issubdtype(dtype, np.floating):
            return np.float32(-1.0)
        return dtype(-1)
    return dtype(0)  # PADVAL_ZERO / PADVAL_NULL


def _make_inputs(name, np_dtype, src_shape):
    # Mirrors legacy gen_data.py: deterministic per-case seed (crc32 on the
    # legacy case name) and randint(1, 10) over the full src shape.
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    src = np.random.randint(1, 10, size=src_shape).astype(np_dtype)
    return [src]


def _make_expected(src, src_valid_shape, dst_valid_shape, fill_padval):
    # Mirrors legacy gen_data.py golden: copy src valid data, then fill the
    # column and row expansion regions with FillPadVal.
    src_vr, src_vc = src_valid_shape
    dst_vr, dst_vc = dst_valid_shape
    golden = np.zeros(dst_valid_shape, dtype=src.dtype)

    copy_vr = min(src_vr, dst_vr)
    copy_vc = min(src_vc, dst_vc)
    golden[:copy_vr, :copy_vc] = src[:copy_vr, :copy_vc]

    if dst_vc > src_vc:
        fill_val = get_pad_value(src.dtype, fill_padval)
        golden[:dst_vr, src_vc:dst_vc] = fill_val

    if dst_vr > src_vr:
        fill_val = get_pad_value(src.dtype, fill_padval)
        golden[src_vr:dst_vr, :dst_vc] = fill_val

    return golden


CASES = [
    golden_output_case(
        "tfillpad_expand_" + name,
        _kernels[name],
        inputs=lambda name=name, np_dtype=PTO_TO_NP[pto_dtype], src_shape=src_shape: _make_inputs(name, np_dtype, src_shape),
        expected=lambda src, src_valid_shape=src_valid_shape, dst_valid_shape=dst_valid_shape, fill_padval=fill_padval: _make_expected(
            src, src_valid_shape, dst_valid_shape, fill_padval
        ),
        rtol=eps,
        atol=eps,
    )
    for name, pto_dtype, src_shape, src_valid_shape, dst_shape, dst_valid_shape, _, fill_padval, eps in CASE_SPECS
]


auto_main(globals())
