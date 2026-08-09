# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for ``pto.trowexpanddiv``."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._expand_binary import (
    FLOAT_SIGNATURES,
    _row_expand_layout,
    _valid_row_expand_binary,
    register_row_expand_binary,
)
from .div_hp import _div_ieee754_f16_impl, _div_ieee754_f32_impl


HIGH_PRECISION_SIGNATURES = [
    ("f16", "f16", "f16", "f16"),
    ("f32", "f32", "f32", "f32"),
]


template_trowexpanddiv = register_row_expand_binary(
    op="pto.trowexpanddiv",
    name="template_trowexpanddiv",
    vector_op=pto.vdiv,
    dtypes=FLOAT_SIGNATURES,
)


def _is_high_precision(precisionType="default", **_):
    return precisionType == "high_precision"


@tilelib.tile_template(
    op="pto.trowexpanddiv",
    target="a5",
    name="template_trowexpanddiv_high_precision",
    dtypes=HIGH_PRECISION_SIGNATURES,
    iteration_axis="row",
    op_engine="vector",
    op_class="broadcast",
    constraints=[
        _row_expand_layout,
        _valid_row_expand_binary,
        _is_high_precision,
    ],
    id=1,
    loop_depth=2,
    is_post_update=False,
    tags=("row_expand", "binary", "high_precision"),
)
def template_trowexpanddiv_high_precision(
    src0: pto.Tile, src1: pto.Tile, tmp: pto.Tile, dst: pto.Tile
):
    # A5 requires the legacy tmp operand for high-precision TROWEXPANDDIV,
    # although its A5 implementation does not consume the tile.
    _ = tmp
    dtype = dst.dtype
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            scalar_vec = pto.vlds(src1[row, :])
            rhs = pto.vdup(scalar_vec, mask)
            if str(dtype) == "f32":
                result = _div_ieee754_f32_impl(lhs, rhs, mask)
            else:
                result = _div_ieee754_f16_impl(lhs, rhs, mask)
            pto.vsts(result, dst[row, col:], mask)
