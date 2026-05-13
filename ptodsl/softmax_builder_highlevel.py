# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
High-level builder for the online softmax kernel.

Reconstructs the same IR as softmax_builder_lowlevel.py using the
thin wrappers in ptodsl_utils.  Compare the two files side by side to see
which boilerplate the utils eliminate.
"""

from mlir.ir import F32Type, InsertionPoint

from ptodsl_utils import (
    # context / types
    pto_context, flat_pto_module, pto_aicore_func,
    i32_type, i64_type, idx_type, ptr_type,
    tensor_view_type, part_tensor_view_type, tile_buf_type,
    vreg_type,
    # constants
    c_idx, c_i32, c_i64,
    # arithmetic
    muli, addi, subi, index_cast, cmpi_sgt, select_val,
    # hardware
    get_block_idx, barrier_all,
    # tile domain
    tile_view, part_view, alloc_tile, tload, tstore, tile_ptr,
    # sync (pto.set_flag / pto.wait_flag come from pto module directly)
    # vector / pointer
    castptr, addptr, vlds, vsts,
    plt_b32, pset_b32, vbrc_load, vsts_1pt,
    # vector math
    vcmax, vdup_lowest, vmax, vexpdif, vmul, vcadd, vadd, vdiv,
    # control flow
    vecscope, for_range, for_range_iter, yield_vals,
    if_ctx, if_op_returning,
)
from mlir.dialects import pto


def build():
    with pto_context():
        # ── Types used throughout the kernel ──────────────────────────────
        f32      = F32Type.get()
        i32      = i32_type()
        i64      = i64_type()
        idx      = idx_type()
        ptr_gm   = ptr_type(f32, "gm")    # !pto.ptr<f32, gm>
        ptr_ub   = ptr_type(f32, "ub")    # !pto.ptr<f32, ub>
        tv5d     = tensor_view_type(5, f32)                          # !pto.tensor_view<?x?x?x?x?xf32>
        ptv5d    = part_tensor_view_type(5, f32)                     # !pto.partition_tensor_view<?x?x?x?x?xf32>
        tile_col = tile_buf_type([8,  1], f32, [-1,  1], blayout="ColMajor")  # valid=?x1, col_major
        tile_w   = tile_buf_type([8, 128], f32, [-1, -1])            # valid=?x?
        vf32     = vreg_type(64, f32)     # !pto.vreg<64xf32>

        with flat_pto_module("a5") as mod:
            with pto_aicore_func(
                "online_softmax_update_kernel_2d",
                [ptr_gm] * 7 + [i32, i32],
            ) as (a0, a1, a2, a3, a4, a5, a6, arg7, arg8):

                # ── Index constants ────────────────────────────────────
                c0, c1, c8, c64, c128 = (c_idx(v) for v in (0, 1, 8, 64, 128))

                # ── i64 constants ─────────────────────────────────────
                # Declared in the same order as the reference IR so that
                # the round-tripped MLIR text compares equal.
                c0_i64    = c_i64(0)
                _c1_i64   = c_i64(1)        # present in reference, unused here
                _c8_i64   = c_i64(8)
                _c16_i64  = c_i64(16)
                _c32_i64  = c_i64(32)
                _c64_i64  = c_i64(64)
                c128_i64  = c_i64(128)
                c256_i64  = c_i64(256)
                _c512_i64 = c_i64(512)
                c8448_i64  = c_i64(8448)
                c16640_i64 = c_i64(16640)
                c16768_i64 = c_i64(16768)
                c16896_i64 = c_i64(16896)

                # ── i32 constants ──────────────────────────────────────
                c1_i32 = c_i32(1);  c8_i32 = c_i32(8)
                c64_i32 = c_i32(64); c0_i32 = c_i32(0)

                # ── Block-level row assignment ─────────────────────────
                block_i64     = get_block_idx()
                block_idx     = index_cast(idx, block_i64)
                row_base      = muli(block_idx, c8)
                _             = index_cast(i32, c8)        # block_rows_i32
                row_base_i32  = index_cast(i32, row_base)
                remaining_rows= subi(arg8, row_base_i32)
                has_rows      = cmpi_sgt(remaining_rows, c0_i32)
                too_many_rows = cmpi_sgt(remaining_rows, c8_i32)
                row_count_i32 = select_val(too_many_rows, c8_i32, remaining_rows)
                row_count     = index_cast(idx, row_count_i32)
                seq           = index_cast(idx, arg7)
                rows          = index_cast(idx, arg8)
                rows_x_128    = muli(rows, c128)

                with if_ctx(has_rows):
                    # ── Tensor views ───────────────────────────────────
                    s1   = [rows, rows, rows, c1, rows]
                    s128 = [rows_x_128, rows_x_128, rows_x_128, c128, c1]
                    sh1  = [c1, c1, c1, rows, c1]
                    sh128= [c1, c1, c1, rows, c128]

                    oldmax_view = tile_view(tv5d, a0, sh1,   s1)
                    oldsum_view = tile_view(tv5d, a1, sh1,   s1)
                    qk_view     = tile_view(tv5d, a2, sh128, s128)
                    newmax_view = tile_view(tv5d, a3, sh1,   s1)
                    newsum_view = tile_view(tv5d, a4, sh1,   s1)
                    expmax_view = tile_view(tv5d, a5, sh1,   s1)
                    out_view    = tile_view(tv5d, a6, sh128, s128)

                    # ── Partition views ────────────────────────────────
                    off = [c0, c0, c0, row_base, c0]
                    z1  = [c1, c1, c1, row_count, c1]
                    zs  = [c1, c1, c1, row_count, seq]

                    oldmax_part = part_view(ptv5d, oldmax_view, off, z1)
                    oldsum_part = part_view(ptv5d, oldsum_view, off, z1)
                    qk_part     = part_view(ptv5d, qk_view,     off, zs)
                    newmax_part = part_view(ptv5d, newmax_view, off, z1)
                    newsum_part = part_view(ptv5d, newsum_view, off, z1)
                    expmax_part = part_view(ptv5d, expmax_view, off, z1)
                    out_part    = part_view(ptv5d, out_view,    off, zs)

                    # ── UB tile allocation ─────────────────────────────
                    oldmax_tile = alloc_tile(tile_col, addr=c0_i64,     valid_row=row_count)
                    oldsum_tile = alloc_tile(tile_col, addr=c128_i64,   valid_row=row_count)
                    qk_tile     = alloc_tile(tile_w,   addr=c256_i64,   valid_row=row_count, valid_col=seq)
                    out_tile    = alloc_tile(tile_w,   addr=c8448_i64,  valid_row=row_count, valid_col=seq)
                    newmax_tile = alloc_tile(tile_col, addr=c16640_i64, valid_row=row_count)
                    newsum_tile = alloc_tile(tile_col, addr=c16768_i64, valid_row=row_count)
                    expmax_tile = alloc_tile(tile_col, addr=c16896_i64, valid_row=row_count)

                    # ── Tile loads from GM ─────────────────────────────
                    tload(oldmax_part, oldmax_tile)
                    tload(oldsum_part, oldsum_tile)
                    tload(qk_part,     qk_tile)

                    pto.set_flag("PIPE_MTE2", "PIPE_V", pto.EVENT_ID0)
                    pto.wait_flag("PIPE_MTE2", "PIPE_V", pto.EVENT_ID0)

                    with vecscope():
                        # Materialise typed UB pointers from tile handles
                        ub_om = tile_ptr(oldmax_tile, ptr_ub)
                        ub_os = tile_ptr(oldsum_tile, ptr_ub)
                        ub_qk = tile_ptr(qk_tile,     ptr_ub)
                        ub_out= tile_ptr(out_tile,     ptr_ub)
                        ub_nm = tile_ptr(newmax_tile,  ptr_ub)
                        ub_ns = tile_ptr(newsum_tile,  ptr_ub)
                        ub_em = tile_ptr(expmax_tile,  ptr_ub)

                        active   = pset_b32("PAT_ALL")
                        one_mask, _ = plt_b32(c1_i32)

                        with for_range(c0, row_count, c1) as row:
                            row_qk    = muli(row, c128)
                            oldmax_bc = vbrc_load(ub_om, row, vf32)
                            oldsum_bc = vbrc_load(ub_os, row, vf32)

                            # ── Chunk loop: compute running max & sum ──
                            with for_range_iter(c0, c128, c64,
                                                [oldmax_bc, oldsum_bc]) as cf:
                                chunk       = cf.induction_variable
                                running_max, running_sum = cf.inner_iter_args

                                rem_cols  = subi(arg7, index_cast(i32, chunk))
                                has_chunk = cmpi_sgt(rem_cols, c0_i32)

                                br = if_op_returning(has_chunk, [vf32, vf32])
                                with InsertionPoint(br.then_block):
                                    cmask, _ = plt_b32(rem_cols)
                                    cbase    = addi(row_qk, chunk)
                                    vec      = vlds(ub_qk, cbase, vf32)
                                    cmax     = vcmax(vec, cmask)
                                    cmax_bc  = vdup_lowest(cmax, active)
                                    mmax     = vmax(running_max, cmax_bc, active)
                                    sc_run   = vexpdif(running_max, mmax, active)
                                    rs_sc    = vmul(sc_run, running_sum, active)
                                    c_exp    = vexpdif(vec, mmax, cmask)
                                    c_sum    = vcadd(c_exp, cmask)
                                    c_sum_bc = vdup_lowest(c_sum, active)
                                    m_sum    = vadd(rs_sc, c_sum_bc, active)
                                    yield_vals(mmax, m_sum)
                                with InsertionPoint(br.else_block):
                                    yield_vals(running_max, running_sum)

                                yield_vals(*br.results)

                            final_max, final_sum = cf.results

                            # ── Compute expmax scalar for this row ─────
                            raw_em  = vexpdif(oldmax_bc, final_max, active)
                            sc_os   = vmul(raw_em, oldsum_bc, active)
                            expmax  = vdiv(sc_os, final_sum, active)

                            vsts_1pt(final_max, ub_nm, row, one_mask)
                            vsts_1pt(final_sum, ub_ns, row, one_mask)
                            vsts_1pt(expmax,    ub_em, row, one_mask)

                            # ── Output normalisation loop ──────────────
                            with for_range(c0, c128, c64) as chunk2:
                                rem2      = subi(arg7, index_cast(i32, chunk2))
                                has_c2    = cmpi_sgt(rem2, c0_i32)
                                with if_ctx(has_c2):
                                    cmask2, _ = plt_b32(rem2)
                                    cbase2    = addi(row_qk, chunk2)
                                    vec2      = vlds(ub_qk, cbase2, vf32)
                                    exp2      = vexpdif(vec2, final_max, cmask2)
                                    out2      = vdiv(exp2, final_sum, cmask2)
                                    vsts(out2, ub_out, cbase2, cmask2)

                    pto.set_flag("PIPE_V", "PIPE_MTE3", pto.EVENT_ID0)
                    pto.wait_flag("PIPE_V", "PIPE_MTE3", pto.EVENT_ID0)

                    # ── Tile stores to GM ──────────────────────────────
                    tstore(newmax_tile, newmax_part)
                    tstore(newsum_tile, newsum_part)
                    tstore(expmax_tile, expmax_part)
                    tstore(out_tile,    out_part)

                barrier_all()

        return mod


if __name__ == "__main__":
    print(build())
