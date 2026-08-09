# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib templates for pto.tci."""

from ptodsl import pto, scalar
import ptodsl.tilelib as tilelib


def _require_tci_tmp_shape(tmp_shape, tmp_valid_shape, tmp_dtype, **_):
    return (
        tuple(tmp_shape) == (1, 512)
        and tuple(tmp_valid_shape) == (1, 512)
        and tmp_dtype == "f32"
    )


@tilelib.tile_template(
    op="pto.tci",
    target="a5",
    name="template_tci",
    dtypes=[("i16", "i16"), ("ui16", "ui16"), ("i32", "i32"), ("ui32", "ui32")],
    iteration_axis="none",
    op_engine="other",
    op_class="other",
    constraints=[
        tilelib.check_memory_space("ub"),
        tilelib.check_layout("row_major"),
        tilelib.check_s_layout("none_box"),
        tilelib.require_valid_rows("dst", 1),
    ],
    id=0,
    loop_depth=1,
    is_post_update=False,
)
def template_tci(start, dst: pto.Tile):
    descending = pto.get_op_attr("descending", "false") == "true"
    dtype = dst.dtype
    cast_dtype = pto.i16 if str(dtype) in ("ui16",) else pto.i32
    valid_rows, valid_cols = dst.valid_shape
    ptr = dst.as_ptr()
    if descending:
        for col in range(0, valid_cols, 1):
            scalar.store(scalar.index_cast(cast_dtype, start - col), ptr, col)
    else:
        for col in range(0, valid_cols, 1):
            scalar.store(scalar.index_cast(cast_dtype, start + col), ptr, col)


@tilelib.tile_template(
    op="pto.tci",
    target="a5",
    name="template_tci_tmp",
    # A5's vector TCI form keeps a float32 1x512 scratch tile in its
    # callable contract, while the generated sequence itself is written to
    # dst.  Keep this form distinct from the scalar-only candidate so the
    # mode-1 TileLang ST cases select the same three-operand ABI.
    dtypes=[
        ("i16", "f32", "i16"),
        ("ui16", "f32", "ui16"),
        ("i32", "f32", "i32"),
        ("ui32", "f32", "ui32"),
    ],
    iteration_axis="none",
    op_engine="other",
    op_class="other",
    constraints=[
        tilelib.check_memory_space("ub"),
        tilelib.check_layout("row_major"),
        tilelib.check_s_layout("none_box"),
        tilelib.require_valid_rows("dst", 1),
        tilelib.require_valid_rows("tmp", 1),
        _require_tci_tmp_shape,
    ],
    id=1,
    loop_depth=1,
    is_post_update=False,
)
def template_tci_tmp(start, tmp: pto.Tile, dst: pto.Tile):
    # A5 accepts ``tmp`` for the three-operand TCI ABI, but the A5
    # implementation deliberately does not touch it.  The vector overload
    # emits one VCI/VSTS pair per 256-byte chunk; keep the operand in the
    # signature without manufacturing a scratch-buffer dependency.
    descending = pto.get_op_attr("descending", "false") == "true"
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    del tmp

    lanes = pto.elements_per_vreg(dtype)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            # Keep the scalar's declared integer type.  ``scalar.index_cast``
            # produces an MLIR index, which is not a legal VCI element type;
            # VCI itself accepts the matching signed or unsigned integer
            # scalar and preserves the destination bit pattern.
            col_value = scalar.index_cast(dtype, col)
            if descending:
                # A5's C++ vector overload seeds DESC VCI at
                # ``S - lanes + 1 - col``.  DESC VCI enumerates the lanes in
                # reverse order, so this produces the logical sequence
                # ``S-col, ..., S-col-lanes+1`` in the destination chunk.
                lane_last = scalar.index_cast(dtype, pto.const(lanes - 1))
                base = start - lane_last - col_value
            else:
                base = start + col_value
            index = pto.vci(base, "DESC" if descending else "ASC")
            dist = "NORM_B16" if str(dtype) in ("i16", "ui16") else "NORM_B32"
            pto.vsts(index, dst[row, col:], mask, dist=dist)
