#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tfillpad_inplace
# (legacy .pto TFILLPAD_INPLACE_f32_260x16_noexpand).
#
#   tload(src) + tfillpad_inplace(tile, tile) -> tile + tstore(dst)
#
# TFILLPAD_INPLACE semantics (auto mode, pto.tile.fillpad_inplace):
#   skip the copy phase; only fill the expansion regions of an already
#   materialized tile in place, using the dst tile's pad value:
#   1. for rows [0, src.valid_rows), fill cols [src.valid_cols, dst.valid_cols)
#   2. for rows [src.valid_rows, dst.valid_rows), fill cols [0, dst.valid_cols)
#
# The legacy kernel uses a single tile buffer: src and dst share the same SSA
# tile (ins == outs).  This case has src_valid == dst_valid == full tile
# (260x16), so both fill regions are empty and the loaded data is stored back
# unchanged (golden == input).
#
# The tile is allocated at the dst physical size with valid_shape=(260, 16)
# and pad=Max, mirroring the legacy .pto tile_buf<vec, 260x16xf32, pad=2>
# (PadValue encoding 0=Null 1=Zero 2=Max 3=Min).  Legacy dtype / shape /
# valid_shape / layout / pad params / eps and the uniform(1, 10) draw order
# are preserved.

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto

PTO_TO_NP = {
    pto.f32: np.float32,
}

# PadValue enum names matching the legacy cases.py / gen_data.py.
PADVAL_MAX = "Max"     # FLT_MAX for float, max for integers
PADVAL_MIN = "Min"     # -FLT_MAX for float, min for integers
PADVAL_ZERO = "Zero"   # 0

# (legacy case name, pto dtype, src shape, src valid_shape, dst shape,
#  dst valid_shape, fill_padval, eps).
# shape/valid_shape follow the legacy cases.py table: src_shape is the GM
# input (260x16), dst_shape/dst_valid_shape the output (260x16).
CASE_SPECS = [
    ("f32_260x16_noexpand", pto.f32, (260, 16), (260, 16), (260, 16), (260, 16), PADVAL_MAX, 1e-6),
]


def _make_kernel(name, pto_dtype, src_shape, dst_shape, dst_valid_shape, fill_pad):
    @pto.jit(name="tfillpad_inplace_" + name, target="a5")
    def _kernel(
        src_ptr: pto.ptr(pto_dtype, "gm"),
        dst_ptr: pto.ptr(pto_dtype, "gm"),
    ):
        src_rows, src_cols = src_shape
        dst_rows, dst_cols = dst_shape
        src_view = pto.make_tensor_view(src_ptr, shape=[src_rows, src_cols], strides=[src_cols, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[dst_rows, dst_cols], strides=[dst_cols, 1])

        # Single shared tile, allocated at the dst physical size with
        # valid_shape=(dst valid) and the fill pad value, mirroring the legacy
        # .pto alloc_tile !pto.tile_buf<vec, 260x16xf32, pad=2>.  src == dst
        # for tfillpad_inplace (in-place on the same SSA tile).
        tile = pto.alloc_tile(
            shape=[dst_rows, dst_cols], dtype=pto_dtype,
            valid_shape=[dst_valid_shape[0], dst_valid_shape[1]],
            pad=fill_pad,
        )

        pto.tile.load(src_view, tile)
        pto.tile.fillpad_inplace(tile, tile)
        pto.tile.store(tile, dst_view)

    return _kernel


_kernels = {
    name: _make_kernel(name, pto_dtype, src_shape, dst_shape, dst_valid_shape, fill_pad)
    for name, pto_dtype, src_shape, _, dst_shape, dst_valid_shape, fill_pad, _ in CASE_SPECS
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
    return dtype(0)  # PADVAL_ZERO / PADVAL_NULL


def _make_inputs(name, np_dtype, src_shape):
    # Mirrors legacy gen_data.py: deterministic per-case seed (crc32 on the
    # legacy case name, replacing the non-deterministic builtin hash) and
    # np.random.uniform(1.0, 10.0) over the full src shape.
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    src = np.random.uniform(1.0, 10.0, size=src_shape).astype(np_dtype)
    return [src]


def _make_expected(src, src_valid_shape, dst_shape, dst_valid_shape, fill_padval):
    # Mirrors legacy gen_data.py golden: copy src valid data, then fill the
    # column and row expansion regions with FillPadVal.
    src_vr, src_vc = src_valid_shape
    dst_vr, dst_vc = dst_valid_shape
    golden = np.zeros(dst_shape, dtype=src.dtype)

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
        "tfillpad_inplace_" + name,
        _kernels[name],
        inputs=lambda name=name, np_dtype=PTO_TO_NP[pto_dtype], src_shape=src_shape: _make_inputs(name, np_dtype, src_shape),
        expected=lambda src, src_valid_shape=src_valid_shape, dst_shape=dst_shape, dst_valid_shape=dst_valid_shape, fill_padval=fill_padval: _make_expected(
            src, src_valid_shape, dst_shape, dst_valid_shape, fill_padval
        ),
        rtol=eps,
        atol=eps,
    )
    for name, pto_dtype, src_shape, src_valid_shape, dst_shape, dst_valid_shape, fill_padval, eps in CASE_SPECS
]


auto_main(globals())
