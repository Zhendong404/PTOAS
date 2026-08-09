# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared PTODSL implementation for straightforward A5 column reductions."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


def _has_single_output_row(dst_valid_shape=(), **_):
    return len(dst_valid_shape) == 2 and dst_valid_shape[0] == 1


def _ub_or_vec_row_major(operand_memory_spaces, operand_b_layouts, operand_s_layouts, **_):
    return (
        all(space in {"ub", "vec"} for space in operand_memory_spaces)
        and all(layout == "row_major" for layout in operand_b_layouts)
        and all(layout == "none_box" for layout in operand_s_layouts)
    )


def _binary_column_reduction_tmp_contract(
    src_shape=(),
    src_valid_shape=(),
    tmp_shape=(),
    tmp_valid_shape=(),
    **_,
):
    """Check the A5 pairwise-column-reduction scratch-tile contract.

    A5 first reduces adjacent source rows into ``tmp``.  The physical scratch
    tile therefore needs room for the rounded-up half of the source tile, and
    each physical row must cover the source's valid columns.  The ISA verifier
    currently checks the latter; keeping the former here prevents a rendered
    template from indexing beyond the scratch tile.
    """
    if len(src_shape) != 2 or len(src_valid_shape) != 2 or len(tmp_shape) != 2:
        return False
    if len(tmp_valid_shape) != 2:
        return False
    if not all(isinstance(dim, int) for dim in src_shape + tmp_shape):
        return False
    if not all(isinstance(dim, int) for dim in src_valid_shape):
        return False
    return (
        tmp_shape[0] >= (src_shape[0] + 1) // 2
        and tmp_shape[1] >= src_valid_shape[1]
    )


def _binary_add_vector(src0, row0, src1, row1, dst, dst_row, col, mask):
    lhs = pto.vlds(src0[row0, col:])
    rhs = pto.vlds(src1[row1, col:])
    pto.vsts(pto.vadd(lhs, rhs, mask), dst[dst_row, col:], mask)


def _binary_copy_vector(src, src_row, dst, dst_row, col, mask):
    value = pto.vlds(src[src_row, col:])
    pto.vsts(value, dst[dst_row, col:], mask)


def register_column_reduction(*, op, name, vector_op, dtypes):
    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=dtypes,
        iteration_axis="column",
        op_engine="vector",
        op_class="reduction",
        constraints=[
            _ub_or_vec_row_major,
            _has_single_output_row,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("reduction", "column"),
    )
    def template(src: pto.Tile, dst: pto.Tile):
        dtype = dst.dtype
        valid_rows, valid_cols = src.valid_shape
        lanes = pto.elements_per_vreg(dtype)
        remained = valid_cols

        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            accumulator = pto.vlds(src[0, col:])

            for row in range(1, valid_rows, 1):
                value = pto.vlds(src[row, col:])
                accumulator = vector_op(accumulator, value, mask)

            pto.vsts(accumulator, dst[0, col:], mask)

    return template


def register_binary_column_reduction(*, op, name, dtypes):
    """Register the A5 ``isBinary=true`` column-reduction implementation."""

    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=dtypes,
        iteration_axis="column",
        op_engine="vector",
        op_class="reduction",
        constraints=[
            _ub_or_vec_row_major,
            _has_single_output_row,
            _binary_column_reduction_tmp_contract,
        ],
        id=1,
        loop_depth=3,
        is_post_update=False,
        tags=("reduction", "column", "binary"),
    )
    def template_binary(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
        dtype = dst.dtype
        valid_rows, valid_cols = src.valid_shape
        lanes = pto.elements_per_vreg(dtype)
        tmp_rows = tmp.shape[0]

        # Initial adjacent-row reduction.  The odd last row is folded into
        # the previous pair, exactly as TColSum_Binary does on A5.
        pair_count = valid_rows // 2
        for pair in range(0, tmp_rows, 1):
            if pair < pair_count:
                remained = valid_cols
                for col in range(0, valid_cols, lanes):
                    mask, remained = pto.make_mask(dtype, remained)
                    _binary_add_vector(src, pair * 2, src, pair * 2 + 1, tmp, pair, col, mask)

        if valid_rows % 2 == 1:
            if pair_count > 0:
                remained = valid_cols
                for col in range(0, valid_cols, lanes):
                    mask, remained = pto.make_mask(dtype, remained)
                    _binary_add_vector(
                        tmp, pair_count - 1, src, pair_count * 2,
                        tmp, pair_count - 1, col, mask
                    )
            else:
                remained = valid_cols
                for col in range(0, valid_cols, lanes):
                    mask, remained = pto.make_mask(dtype, remained)
                    _binary_copy_vector(src, 0, tmp, 0, col, mask)

        active_rows = pair_count
        if active_rows == 0:
            active_rows = 1

        # Each stage halves the active scratch rows.  Use a physical bound so
        # the stage count is static while the valid-row tail remains dynamic.
        stage_count = 0
        stage_capacity = tmp_rows
        while stage_capacity > 1:
            stage_count += 1
            stage_capacity = (stage_capacity + 1) // 2

        for _stage in range(0, stage_count, 1):
            pto.mem_bar(pto.BarrierType.VST_VLD)
            pair_count = active_rows // 2
            for pair in range(0, tmp_rows // 2, 1):
                if active_rows > 1:
                    if pair < pair_count:
                        remained = valid_cols
                        for col in range(0, valid_cols, lanes):
                            mask, remained = pto.make_mask(dtype, remained)
                            _binary_add_vector(
                                tmp, pair * 2, tmp, pair * 2 + 1,
                                tmp, pair, col, mask
                            )
            if active_rows % 2 == 1:
                if pair_count > 0:
                    remained = valid_cols
                    for col in range(0, valid_cols, lanes):
                        mask, remained = pto.make_mask(dtype, remained)
                        _binary_add_vector(
                            tmp, pair_count - 1, tmp, pair_count * 2,
                            tmp, pair_count - 1, col, mask
                        )
            # The A5 implementation folds an odd leftover row into the last
            # pair instead of retaining a separate output row.  Therefore an
            # odd active count shrinks with floor division (except for the
            # single-row terminal state).
            active_rows = active_rows // 2
            if active_rows == 0:
                active_rows = 1

        pto.mem_bar(pto.BarrierType.VST_VLD)
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            _binary_copy_vector(tmp, 0, dst, 0, col, mask)

    return template_binary


__all__ = ["register_binary_column_reduction", "register_column_reduction"]
