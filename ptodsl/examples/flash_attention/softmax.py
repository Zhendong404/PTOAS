# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
PTODSL translation of ``ptodsl/examples/fa_dn_softmax.cpp``.

This file intentionally keeps the C++ helper structure:

- ``softmax_opt_fa_dn_init_impl``: first S1 tile, initializes streaming state
- ``softmax_opt_fa_dn_not_init_impl``: later S1 tiles, updates streaming state
- ``pto_macro_fa_softmax_dn``: causal-mask gate and dispatch wrapper

The goal is to preserve the flash-attention streaming softmax semantics as
closely as PTODSL currently allows. A few C++ micro-details are still only
approximated or stubbed:

- ``triu`` is accepted for signature parity but the actual triangular-mask data
  path is not consumed yet.
- ``tile_id``, ``sync_iter``, and ``last_tile`` are preserved for parity with
  the C++ helper but are not yet used by the current PTODSL translation.
- The exact NZ layout contract behind the C++ pointer choreography is expressed
  with explicit ``vmulscvt`` + ``vpack`` + ``vsstb`` lowering, but the precise
  address schedule may still need a follow-up once the surrounding FA kernel is
  wired end-to-end.
"""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "ptodsl" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            break
    else:
        raise RuntimeError(
            "Unable to locate the PTODSL Python package root from fa_dn_softmax.py"
        )

from ptodsl import pto, scalar


def _inv_sqrt(head_size: int) -> float:
    if head_size <= 0:
        raise ValueError("head_size must be positive")
    return float(head_size) ** -0.5


@pto.simd
def softmax_opt_fa_dn_init_impl(
    tile_id: pto.i32,
    sync_iter: pto.i32,
    x_exp: pto.Tile,
    input_x: pto.Tile,
    local_max: pto.Tile,
    local_sum: pto.Tile,
    new_global_max: pto.Tile,
    new_global_sum: pto.Tile,
    exp_max: pto.Tile,
    triu: pto.Tile,
    nz_conv_buffer: pto.Tile,
    s0_index: pto.i32,
    s1_index: pto.i32,
    last_tile: pto.i1,
    *,
    head_size: pto.constexpr,
):

    scale = _inv_sqrt(head_size)
    ubN, ubM = x_exp.shape

    nz_buffer_ptr1 = nz_conv_buffer.as_ptr()
    nz_buffer_ptr2 = pto.addptr(nz_buffer_ptr1, 16)
    nz_buffer_ptr3 = pto.addptr(nz_buffer_ptr1, ubN // 2 * 16)
    nz_buffer_ptr4 = pto.addptr(nz_buffer_ptr1, ubN // 2 * 16 + 16)

    block_stride = ubN + 1
    repeat_stride = 2

    src0_ub = input_x.as_ptr()
    x_exp_1 = pto.addptr(x_exp.as_ptr(), (ubN // 2) * 8)
    src0_ub1 = pto.addptr(src0_ub, 128)
    src0_ub2 = pto.addptr(src0_ub, 256)
    src0_ub3 = pto.addptr(src0_ub, 384)

    src0_ub_unroll = pto.addptr(src0_ub, 64)
    src0_ub1_unroll = pto.addptr(src0_ub1, 64)
    src0_ub2_unroll = pto.addptr(src0_ub2, 64)
    src0_ub3_unroll = pto.addptr(src0_ub3, 64)

    preg_134 = pto.make_mask(pto.i8, pto.MaskPattern.ALL)
    preg_108 = pto.make_mask(pto.f16, pto.MaskPattern.ALL)
    preg_low_half = pto.make_mask(pto.f16, pto.MaskPattern.VL64)

    p0 = pto.addptr(src0_ub, 4 * 64)
    p1 = pto.addptr(src0_ub, 5 * 64)
    p2 = pto.addptr(src0_ub, 6 * 64)
    p3 = pto.addptr(src0_ub, 7 * 64)

    max_0a = pto.vlds(src0_ub, 0 * 64, dist="NORM")
    max_1a = pto.vlds(src0_ub, 1 * 64, dist="NORM")
    max_2a = pto.vlds(src0_ub, 2 * 64, dist="NORM")
    max_3a = pto.vlds(src0_ub, 3 * 64, dist="NORM")

    row_loop = pto.for_(4, ubN, step=4).carry(
        p0=p0,
        p1=p1,
        p2=p2,
        p3=p3,
        max_0a=max_0a,
        max_1a=max_1a,
        max_2a=max_2a,
        max_3a=max_3a,
    )
    with row_loop:
        p0 = row_loop.p0
        p1 = row_loop.p1
        p2 = row_loop.p2
        p3 = row_loop.p3
        max_0a = row_loop.max_0a
        max_1a = row_loop.max_1a
        max_2a = row_loop.max_2a
        max_3a = row_loop.max_3a

        v_row, p0 = pto.vlds(p0, 4 * 64, dist="NORM", post_update="ON")
        max_0a = pto.vmax(max_0a, v_row, preg_108)

        v_row, p1 = pto.vlds(p1, 4 * 64, dist="NORM", post_update="ON")
        max_1a = pto.vmax(max_1a, v_row, preg_108)

        v_row, p2 = pto.vlds(p2, 4 * 64, dist="NORM", post_update="ON")
        max_2a = pto.vmax(max_2a, v_row, preg_108)

        v_row, p3 = pto.vlds(p3, 4 * 64, dist="NORM", post_update="ON")
        max_3a = pto.vmax(max_3a, v_row, preg_108)
        row_loop.update(
            p0=p0,
            p1=p1,
            p2=p2,
            p3=p3,
            max_0a=max_0a,
            max_1a=max_1a,
            max_2a=max_2a,
            max_3a=max_3a,
        )

    max_0a = row_loop.final("max_0a")
    max_1a = row_loop.final("max_1a")
    max_2a = row_loop.final("max_2a")
    max_3a = row_loop.final("max_3a")

    max_0a = pto.vmax(max_0a, max_1a, preg_108)
    max_2a = pto.vmax(max_2a, max_3a, preg_108)
    max_0a = pto.vmax(max_0a, max_2a, preg_108)

    pto.vsts(max_0a, new_global_max.as_ptr(), 0, preg_108, dist="NORM_B16")
    max_0a = pto.vmuls(max_0a, scale, preg_108)

    vreg_x_sum_even = pto.vdup(pto.f32(0), pto.pbitcast(preg_134, pto.mask_b32))
    vreg_x_sum_odd = pto.vdup(pto.f32(0), pto.pbitcast(preg_134, pto.mask_b32))
    vreg_x_sum_1_even = pto.vdup(pto.f32(0), pto.pbitcast(preg_134, pto.mask_b32))
    vreg_x_sum_1_odd = pto.vdup(pto.f32(0), pto.pbitcast(preg_134, pto.mask_b32))

    preg_100 = preg_108
    preg_101 = preg_108
    preg_135 = pto.make_mask(pto.f32, pto.MaskPattern.ALL)
    sreg_92: pto.i32 = 128
    preg_136, sreg_92 = pto.make_mask(pto.f16, sreg_92)

    input_x_ptr = input_x.as_ptr()
    input_x_half_ptr = pto.addptr(input_x_ptr, ubN * ubM // 2)

    sum_loop = pto.for_(0, ubN // 4, step=1).carry(
        nz_buffer_ptr1=nz_buffer_ptr1,
        nz_buffer_ptr2=nz_buffer_ptr2,
        nz_buffer_ptr3=nz_buffer_ptr3,
        nz_buffer_ptr4=nz_buffer_ptr4,
        vreg_x_sum_even=vreg_x_sum_even,
        vreg_x_sum_odd=vreg_x_sum_odd,
        vreg_x_sum_1_even=vreg_x_sum_1_even,
        vreg_x_sum_1_odd=vreg_x_sum_1_odd,
    )
    with sum_loop:
        i0 = sum_loop.iv
        nz_buffer_ptr1 = sum_loop.nz_buffer_ptr1
        nz_buffer_ptr2 = sum_loop.nz_buffer_ptr2
        nz_buffer_ptr3 = sum_loop.nz_buffer_ptr3
        nz_buffer_ptr4 = sum_loop.nz_buffer_ptr4
        vreg_x_sum_even = sum_loop.vreg_x_sum_even
        vreg_x_sum_odd = sum_loop.vreg_x_sum_odd
        vreg_x_sum_1_even = sum_loop.vreg_x_sum_1_even
        vreg_x_sum_1_odd = sum_loop.vreg_x_sum_1_odd

        vreg_x_f32_a = pto.vlds(input_x_ptr, i0 * 128, dist="NORM")
        vreg_x_f32_b = pto.vlds(pto.addptr(input_x_ptr, 64), i0 * 128, dist="NORM")
        vreg_x_f32_1_a = pto.vlds(input_x_half_ptr, i0 * 128, dist="NORM")
        vreg_x_f32_1_b = pto.vlds(pto.addptr(input_x_half_ptr, 64), i0 * 128, dist="NORM")

        vreg_x_f32_a = pto.vmuls(vreg_x_f32_a, scale, preg_108)
        vreg_x_f32_b = pto.vmuls(vreg_x_f32_b, scale, preg_108)
        vreg_x_exp_even = pto.vexpdif(vreg_x_f32_a, max_0a, pto.pbitcast(preg_134, pto.mask_b32), part="ODD")
        vreg_x_exp_odd = pto.vexpdif(vreg_x_f32_b, max_0a, pto.pbitcast(preg_134, pto.mask_b32), part="ODD")

        vreg_x_f32_1_a = pto.vmuls(vreg_x_f32_1_a, scale, preg_108)
        vreg_x_f32_1_b = pto.vmuls(vreg_x_f32_1_b, scale, preg_108)
        vreg_x_exp_even_1 = pto.vexpdif(vreg_x_f32_1_a, max_0a, pto.pbitcast(preg_134, pto.mask_b32), part="ODD")
        vreg_x_exp_odd_1 = pto.vexpdif(vreg_x_f32_1_b, max_0a, pto.pbitcast(preg_134, pto.mask_b32), part="ODD")

        vreg_x_exp_even_f16 = pto.vmulscvt(vreg_x_exp_even, 1.0, pto.pbitcast(preg_100, pto.mask_b32), rnd="A", part="EVEN")
        vreg_x_exp_odd_f16 = pto.vmulscvt(vreg_x_exp_odd, 1.0, pto.pbitcast(preg_101, pto.mask_b32), rnd="A", part="EVEN")
        vreg_x_exp_even_f16_1 = pto.vmulscvt(vreg_x_exp_even_1, 1.0, pto.pbitcast(preg_135, pto.mask_b32), rnd="A", part="EVEN")
        vreg_x_exp_odd_f16_1 = pto.vmulscvt(vreg_x_exp_odd_1, 1.0, pto.pbitcast(preg_136, pto.mask_b32), rnd="A", part="EVEN")

        vreg_x_exp_even_u16 = pto.vpack(pto.vbitcast(vreg_x_exp_even_f16, pto.ui32), "LOWER")
        vreg_x_exp_odd_u16 = pto.vpack(pto.vbitcast(vreg_x_exp_odd_f16, pto.ui32), "LOWER")
        vreg_x_exp_even_u16_1 = pto.vpack(pto.vbitcast(vreg_x_exp_even_f16_1, pto.ui32), "LOWER")
        vreg_x_exp_odd_u16_1 = pto.vpack(pto.vbitcast(vreg_x_exp_odd_f16_1, pto.ui32), "LOWER")

        nz_buffer_ptr1 = pto.vsstb(vreg_x_exp_even_u16, nz_buffer_ptr1, block_stride, repeat_stride, preg_low_half, post_update="ON")
        nz_buffer_ptr2 = pto.vsstb(vreg_x_exp_odd_u16, nz_buffer_ptr2, block_stride, repeat_stride, preg_low_half, post_update="ON")
        nz_buffer_ptr3 = pto.vsstb(vreg_x_exp_even_u16_1, nz_buffer_ptr3, block_stride, repeat_stride, preg_low_half, post_update="ON")
        nz_buffer_ptr4 = pto.vsstb(vreg_x_exp_odd_u16_1, nz_buffer_ptr4, block_stride, repeat_stride, preg_low_half, post_update="ON")

        vreg_x_sum_even = pto.vadd(vreg_x_sum_even, vreg_x_exp_even, preg_134)
        vreg_x_sum_odd = pto.vadd(vreg_x_sum_odd, vreg_x_exp_odd, preg_134)
        vreg_x_sum_1_even = pto.vadd(vreg_x_sum_1_even, vreg_x_exp_even_1, preg_134)
        vreg_x_sum_1_odd = pto.vadd(vreg_x_sum_1_odd, vreg_x_exp_odd_1, preg_134)

        sum_loop.update(
            nz_buffer_ptr1=nz_buffer_ptr1,
            nz_buffer_ptr2=nz_buffer_ptr2,
            nz_buffer_ptr3=nz_buffer_ptr3,
            nz_buffer_ptr4=nz_buffer_ptr4,
            vreg_x_sum_even=vreg_x_sum_even,
            vreg_x_sum_odd=vreg_x_sum_odd,
            vreg_x_sum_1_even=vreg_x_sum_1_even,
            vreg_x_sum_1_odd=vreg_x_sum_1_odd,
        )

    vreg_x_sum_even = sum_loop.final("vreg_x_sum_even")
    vreg_x_sum_odd = sum_loop.final("vreg_x_sum_odd")
    vreg_x_sum_1_even = sum_loop.final("vreg_x_sum_1_even")
    vreg_x_sum_1_odd = sum_loop.final("vreg_x_sum_1_odd")

    vreg_x_sum0 = pto.vadd(vreg_x_sum_odd, vreg_x_sum_even, preg_134)
    vreg_x_sum1 = pto.vadd(vreg_x_sum_1_odd, vreg_x_sum_1_even, preg_134)
    vreg_x_sum0 = pto.vadd(vreg_x_sum0, vreg_x_sum1, preg_134)
    pto.vsts(vreg_x_sum0, local_sum.as_ptr(), 0, preg_134, dist="NORM_B32")
    pto.vsts(vreg_x_sum0, new_global_sum.as_ptr(), 0, preg_134, dist="NORM_B32")


@pto.simd
def softmax_opt_fa_dn_not_init_impl(
    tile_id: pto.i32,
    sync_iter: pto.i32,
    x_exp: pto.Tile,
    input_x: pto.Tile,
    local_max: pto.Tile,
    local_sum: pto.Tile,
    new_global_max: pto.Tile,
    new_global_sum: pto.Tile,
    exp_max: pto.Tile,
    triu: pto.Tile,
    nz_conv_buffer: pto.Tile,
    s0_index: pto.i32,
    s1_index: pto.i32,
    last_tile: pto.i1,
    *,
    head_size: pto.constexpr,
):
    scale = _inv_sqrt(head_size)
    ubN, ubM = x_exp.shape

    nz_buffer_ptr1 = nz_conv_buffer.as_ptr()
    nz_buffer_ptr2 = pto.addptr(nz_buffer_ptr1, 16)
    nz_buffer_ptr3 = pto.addptr(nz_buffer_ptr1, ubN // 2 * 16)
    nz_buffer_ptr4 = pto.addptr(nz_buffer_ptr1, ubN // 2 * 16 + 16)

    block_stride = ubN + 1
    repeat_stride = 2

    src0_ub = input_x.as_ptr()
    x_exp_1 = pto.addptr(x_exp.as_ptr(), (ubN // 2) * 8)
    src0_ub1 = pto.addptr(src0_ub, 128)
    src0_ub2 = pto.addptr(src0_ub, 256)
    src0_ub3 = pto.addptr(src0_ub, 384)

    src0_ub_unroll = pto.addptr(src0_ub, 64)
    src0_ub1_unroll = pto.addptr(src0_ub1, 64)
    src0_ub2_unroll = pto.addptr(src0_ub2, 64)
    src0_ub3_unroll = pto.addptr(src0_ub3, 64)

    preg_134 = pto.make_mask(pto.i8, pto.MaskPattern.ALL)
    preg_108 = pto.make_mask(pto.f16, pto.MaskPattern.ALL)
    preg_low_half = pto.make_mask(pto.f16, pto.MaskPattern.VL64)
    vreg_x_max_f32_b = pto.vlds(new_global_max.as_ptr(), 0, dist="NORM")

    p0 = pto.addptr(src0_ub, 4 * 64)
    p1 = pto.addptr(src0_ub, 5 * 64)
    p2 = pto.addptr(src0_ub, 6 * 64)
    p3 = pto.addptr(src0_ub, 7 * 64)

    max_0a = pto.vlds(src0_ub, 0 * 64, dist="NORM")
    max_1a = pto.vlds(src0_ub, 1 * 64, dist="NORM")
    max_2a = pto.vlds(src0_ub, 2 * 64, dist="NORM")
    max_3a = pto.vlds(src0_ub, 3 * 64, dist="NORM")
    
    row_loop = pto.for_(4, ubN, step=4).carry(
        p0=p0,
        p1=p1,
        p2=p2,
        p3=p3,
        max_0a=max_0a,
        max_1a=max_1a,
        max_2a=max_2a,
        max_3a=max_3a,
    )
    with row_loop:
        p0 = row_loop.p0
        p1 = row_loop.p1
        p2 = row_loop.p2
        p3 = row_loop.p3
        max_0a = row_loop.max_0a
        max_1a = row_loop.max_1a
        max_2a = row_loop.max_2a
        max_3a = row_loop.max_3a

        v_row, p0 = pto.vlds(p0, 4 * 64, dist="NORM", post_update="ON")
        max_0a = pto.vmax(max_0a, v_row, preg_108)

        v_row, p1 = pto.vlds(p1, 4 * 64, dist="NORM", post_update="ON")
        max_1a = pto.vmax(max_1a, v_row, preg_108)

        v_row, p2 = pto.vlds(p2, 4 * 64, dist="NORM", post_update="ON")
        max_2a = pto.vmax(max_2a, v_row, preg_108)

        v_row, p3 = pto.vlds(p3, 4 * 64, dist="NORM", post_update="ON")
        max_3a = pto.vmax(max_3a, v_row, preg_108)
        row_loop.update(
            p0=p0,
            p1=p1,
            p2=p2,
            p3=p3,
            max_0a=max_0a,
            max_1a=max_1a,
            max_2a=max_2a,
            max_3a=max_3a,
        )

    max_0a = row_loop.final("max_0a")
    max_1a = row_loop.final("max_1a")
    max_2a = row_loop.final("max_2a")
    max_3a = row_loop.final("max_3a")

    max_0a = pto.vmax(max_0a, max_1a, preg_108)
    max_2a = pto.vmax(max_2a, max_3a, preg_108)
    max_0a = pto.vmax(max_0a, max_2a, preg_108)

    max_0a = pto.vmax(max_0a, vreg_x_max_f32_b, preg_108)

    pto.vsts(max_0a, new_global_max.as_ptr(), 0, preg_108, dist="NORM_B16")
    max_0a = pto.vmuls(max_0a, scale, preg_108)
    vreg_x_max_f32_b = pto.vexpdif(vreg_x_max_f32_b, max_0a, pto.pbitcast(preg_134, pto.mask_b32), part="ODD")

    pto.vsts(vreg_x_max_f32_b, exp_max.as_ptr(), 0, preg_108, dist="NORM_B16")

    vreg_x_sum_even = pto.vdup(pto.f32(0), pto.pbitcast(preg_134, pto.mask_b32))
    vreg_x_sum_odd = pto.vdup(pto.f32(0), pto.pbitcast(preg_134, pto.mask_b32))
    vreg_x_sum_1_even = pto.vdup(pto.f32(0), pto.pbitcast(preg_134, pto.mask_b32))
    vreg_x_sum_1_odd = pto.vdup(pto.f32(0), pto.pbitcast(preg_134, pto.mask_b32))

    preg_100 = preg_108
    preg_101 = preg_108
    preg_135 = pto.make_mask(pto.f32, pto.MaskPattern.ALL)
    sreg_92: pto.i32 = 128
    preg_136, sreg_92 = pto.make_mask(pto.f16, sreg_92)

    input_x_ptr = input_x.as_ptr()
    input_x_half_ptr = pto.addptr(input_x_ptr, ubN * ubM // 2)

    sum_loop = pto.for_(0, ubN // 4, step=1).carry(
        nz_buffer_ptr1=nz_buffer_ptr1,
        nz_buffer_ptr2=nz_buffer_ptr2,
        nz_buffer_ptr3=nz_buffer_ptr3,
        nz_buffer_ptr4=nz_buffer_ptr4,
        vreg_x_sum_even=vreg_x_sum_even,
        vreg_x_sum_odd=vreg_x_sum_odd,
        vreg_x_sum_1_even=vreg_x_sum_1_even,
        vreg_x_sum_1_odd=vreg_x_sum_1_odd,
    )
    with sum_loop:
        i0 = sum_loop.iv
        nz_buffer_ptr1 = sum_loop.nz_buffer_ptr1
        nz_buffer_ptr2 = sum_loop.nz_buffer_ptr2
        nz_buffer_ptr3 = sum_loop.nz_buffer_ptr3
        nz_buffer_ptr4 = sum_loop.nz_buffer_ptr4
        vreg_x_sum_even = sum_loop.vreg_x_sum_even
        vreg_x_sum_odd = sum_loop.vreg_x_sum_odd
        vreg_x_sum_1_even = sum_loop.vreg_x_sum_1_even
        vreg_x_sum_1_odd = sum_loop.vreg_x_sum_1_odd

        vreg_x_f32_a = pto.vlds(input_x_ptr, i0 * 128, dist="NORM")
        vreg_x_f32_b = pto.vlds(pto.addptr(input_x_ptr, 64), i0 * 128, dist="NORM")
        vreg_x_f32_1_a = pto.vlds(input_x_half_ptr, i0 * 128, dist="NORM")
        vreg_x_f32_1_b = pto.vlds(pto.addptr(input_x_half_ptr, 64), i0 * 128, dist="NORM")

        vreg_x_f32_a = pto.vmuls(vreg_x_f32_a, scale, preg_108)
        vreg_x_f32_b = pto.vmuls(vreg_x_f32_b, scale, preg_108)
        vreg_x_exp_even = pto.vexpdif(vreg_x_f32_a, max_0a, pto.pbitcast(preg_134, pto.mask_b32), part="ODD")
        vreg_x_exp_odd = pto.vexpdif(vreg_x_f32_b, max_0a, pto.pbitcast(preg_134, pto.mask_b32), part="ODD")

        vreg_x_f32_1_a = pto.vmuls(vreg_x_f32_1_a, scale, preg_108)
        vreg_x_f32_1_b = pto.vmuls(vreg_x_f32_1_b, scale, preg_108)
        vreg_x_exp_even_1 = pto.vexpdif(vreg_x_f32_1_a, max_0a, pto.pbitcast(preg_134, pto.mask_b32), part="ODD")
        vreg_x_exp_odd_1 = pto.vexpdif(vreg_x_f32_1_b, max_0a, pto.pbitcast(preg_134, pto.mask_b32), part="ODD")

        vreg_x_exp_even_f16 = pto.vmulscvt(vreg_x_exp_even, 1.0, pto.pbitcast(preg_100, pto.mask_b32), rnd="A", part="EVEN")  # preg_100 = ?
        vreg_x_exp_odd_f16 = pto.vmulscvt(vreg_x_exp_odd, 1.0, pto.pbitcast(preg_101, pto.mask_b32), rnd="A", part="EVEN")  # preg_101 = ?
        vreg_x_exp_even_f16_1 = pto.vmulscvt(vreg_x_exp_even_1, 1.0, pto.pbitcast(preg_135, pto.mask_b32), rnd="A", part="EVEN")
        vreg_x_exp_odd_f16_1 = pto.vmulscvt(vreg_x_exp_odd_1, 1.0, pto.pbitcast(preg_136, pto.mask_b32), rnd="A", part="EVEN")

        vreg_x_exp_even_u16 = pto.vpack(pto.vbitcast(vreg_x_exp_even_f16, pto.ui32), "LOWER")
        vreg_x_exp_odd_u16 = pto.vpack(pto.vbitcast(vreg_x_exp_odd_f16, pto.ui32), "LOWER")
        vreg_x_exp_even_u16_1 = pto.vpack(pto.vbitcast(vreg_x_exp_even_f16_1, pto.ui32), "LOWER")
        vreg_x_exp_odd_u16_1 = pto.vpack(pto.vbitcast(vreg_x_exp_odd_f16_1, pto.ui32), "LOWER")

        nz_buffer_ptr1 = pto.vsstb(vreg_x_exp_even_u16, nz_buffer_ptr1, block_stride, repeat_stride, preg_low_half, post_update="ON")
        nz_buffer_ptr2 = pto.vsstb(vreg_x_exp_odd_u16, nz_buffer_ptr2, block_stride, repeat_stride, preg_low_half, post_update="ON")
        nz_buffer_ptr3 = pto.vsstb(vreg_x_exp_even_u16_1, nz_buffer_ptr3, block_stride, repeat_stride, preg_low_half, post_update="ON")
        nz_buffer_ptr4 = pto.vsstb(vreg_x_exp_odd_u16_1, nz_buffer_ptr4, block_stride, repeat_stride, preg_low_half, post_update="ON")

        vreg_x_sum_even = pto.vadd(vreg_x_sum_even, vreg_x_exp_even, preg_134)
        vreg_x_sum_odd = pto.vadd(vreg_x_sum_odd, vreg_x_exp_odd, preg_134)
        vreg_x_sum_1_even = pto.vadd(vreg_x_sum_1_even, vreg_x_exp_even_1, preg_134)
        vreg_x_sum_1_odd = pto.vadd(vreg_x_sum_1_odd, vreg_x_exp_odd_1, preg_134)

        sum_loop.update(
            nz_buffer_ptr1=nz_buffer_ptr1,
            nz_buffer_ptr2=nz_buffer_ptr2,
            nz_buffer_ptr3=nz_buffer_ptr3,
            nz_buffer_ptr4=nz_buffer_ptr4,
            vreg_x_sum_even=vreg_x_sum_even,
            vreg_x_sum_odd=vreg_x_sum_odd,
            vreg_x_sum_1_even=vreg_x_sum_1_even,
            vreg_x_sum_1_odd=vreg_x_sum_1_odd,
        )

    vreg_x_sum_even = sum_loop.final("vreg_x_sum_even")
    vreg_x_sum_odd = sum_loop.final("vreg_x_sum_odd")
    vreg_x_sum_1_even = sum_loop.final("vreg_x_sum_1_even")
    vreg_x_sum_1_odd = sum_loop.final("vreg_x_sum_1_odd")

    vreg_x_sum0 = pto.vadd(vreg_x_sum_odd, vreg_x_sum_even, preg_134)
    vreg_x_sum1 = pto.vadd(vreg_x_sum_1_odd, vreg_x_sum_1_even, preg_134)
    vreg_x_sum0 = pto.vadd(vreg_x_sum0, vreg_x_sum1, preg_134)
    pto.vsts(vreg_x_sum0, local_sum.as_ptr(), 0, preg_134, dist="NORM_B32")
    pto.vsts(vreg_x_sum0, new_global_sum.as_ptr(), 0, preg_134, dist="NORM_B32")



def pto_macro_fa_softmax_dn(
    x_exp: pto.Tile,
    input_x: pto.Tile,
    local_max: pto.Tile,
    local_sum: pto.Tile,
    new_global_max: pto.Tile,
    new_global_sum: pto.Tile,
    exp_max: pto.Tile,
    triu: pto.Tile,
    nz_conv_buffer: pto.Tile,
    s0_index: pto.i32,
    s1_index: pto.i32,
    tile_id: pto.i32,
    sync_iter: pto.i32,
    last_tile: pto.i1,
    row_start: pto.i32,
    row_stop: pto.i32,
    valid_cols: pto.i32,
    *,
    init: pto.constexpr = False,
    head_size: pto.constexpr = 64,
    causal_mask: pto.constexpr = False,
):
    """
    PTODSL surface wrapper corresponding to ``pto_macro_fa_softmax_dn`` in C++.
    """
    _ = row_start
    _ = row_stop
    _ = valid_cols

    if not causal_mask:
        if init:
            softmax_opt_fa_dn_init_impl(
                tile_id,
                sync_iter,
                x_exp,
                input_x,
                local_max,
                local_sum,
                new_global_max,
                new_global_sum,
                exp_max,
                triu,
                nz_conv_buffer,
                s0_index,
                s1_index,
                last_tile,
                head_size=head_size,
            )
        else:
            softmax_opt_fa_dn_not_init_impl(
                tile_id,
                sync_iter,
                x_exp,
                input_x,
                local_max,
                local_sum,
                new_global_max,
                new_global_sum,
                exp_max,
                triu,
                nz_conv_buffer,
                s0_index,
                s1_index,
                last_tile,
                head_size=head_size,
            )
        return

    with pto.if_(s1_index <= s0_index) as br:
        with br.then_:
            if init:
                softmax_opt_fa_dn_init_impl(
                    tile_id,
                    sync_iter,
                    x_exp,
                    input_x,
                    local_max,
                    local_sum,
                    new_global_max,
                    new_global_sum,
                    exp_max,
                    triu,
                    nz_conv_buffer,
                    s0_index,
                    s1_index,
                    last_tile,
                    head_size=head_size,
                )
            else:
                softmax_opt_fa_dn_not_init_impl(
                    tile_id,
                    sync_iter,
                    x_exp,
                    input_x,
                    local_max,
                    local_sum,
                    new_global_max,
                    new_global_sum,
                    exp_max,
                    triu,
                    nz_conv_buffer,
                    s0_index,
                    s1_index,
                    last_tile,
                    head_size=head_size,
                )
        with br.else_:
            pto.tile.expands(0.0, x_exp)
            pto.tile.expands(0.0, exp_max)
            pto.tile.adds(exp_max, 1.0, exp_max)


@pto.jit(target="a5", mode="explicit")
def fa_dn_softmax_wrapper(
    *,
    BR: pto.constexpr = 8,
    BC: pto.constexpr = 64,
    INIT: pto.constexpr = False,
    HEAD_SIZE: pto.constexpr = 64,
    CAUSAL_MASK: pto.constexpr = False,
):
    row_start = pto.const(0, dtype=pto.i32)
    row_stop = pto.const(BR, dtype=pto.i32)
    valid_cols = pto.const(BC, dtype=pto.i32)
    s0_index = pto.const(0, dtype=pto.i32)
    s1_index = pto.const(1, dtype=pto.i32)
    tile_id = pto.const(0, dtype=pto.i32)
    sync_iter = pto.const(0, dtype=pto.i32)
    last_tile = pto.const(0, dtype=pto.i1)

    input_x = pto.alloc_tile(shape=[BR, BC], dtype=pto.f32, valid_shape=[BR, BC])
    x_exp = pto.alloc_tile(shape=[BR, BC], dtype=pto.f16, valid_shape=[BR, BC])
    local_max = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")
    local_sum = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")
    new_global_max = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")
    new_global_sum = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")
    exp_max = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")
    triu = pto.alloc_tile(shape=[1, 16], dtype=pto.f16, valid_shape=[1, 1])
    nz_conv_buffer = pto.alloc_tile(shape=[BR, BC], dtype=pto.ui16, valid_shape=[BR, BC])

    pto_macro_fa_softmax_dn(
        x_exp,
        input_x,
        local_max,
        local_sum,
        new_global_max,
        new_global_sum,
        exp_max,
        triu,
        nz_conv_buffer,
        s0_index,
        s1_index,
        tile_id,
        sync_iter,
        last_tile,
        row_start,
        row_stop,
        valid_cols,
        init=INIT,
        head_size=HEAD_SIZE,
        causal_mask=CAUSAL_MASK,
    )


__all__ = [
    "softmax_opt_fa_dn_init_impl",
    "softmax_opt_fa_dn_not_init_impl",
    "pto_macro_fa_softmax_dn",
    "fa_dn_softmax_wrapper",
]


def main() -> None:
    compiled = fa_dn_softmax_wrapper.compile(
        BR=8,
        BC=64,
        INIT=False,
        HEAD_SIZE=64,
        CAUSAL_MASK=False,
    )
    print(compiled.mlir_text())


if __name__ == "__main__":
    main()
