# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
PTODSL softmax helper for Flash Attention vector kernels.

This file now keeps two layers:

- ``fa_softmax_init_vpto_kernel`` / ``fa_softmax_update_vpto_kernel``:
  ptr-ABI VPTO child modules intended to become separate backend objects
- ``fa_softmax_init_vpto`` / ``fa_softmax_update_vpto``:
  Tile-ABI ``@pto.simd`` adapters that materialize ``as_ptr()`` internally
- ``fa_softmax_vpto_probe``: minimal entry wrapper for compile-only inspection

The intended structure is:

- auto-mode callers only see Tile arguments
- the ``@pto.simd`` adapter bridges Tile -> ptr
- the explicit VPTO kernel module owns the micro-instruction body
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

from ptodsl import pto


def _inv_sqrt(head_size: int) -> float:
    if head_size <= 0:
        raise ValueError("head_size must be positive")
    return float(head_size) ** -0.5


@pto.jit(
    name="fa_softmax_init_vpto_kernel",
    target="a5",
    entry=False,
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def fa_softmax_init_vpto_kernel(
    qk_ptr: pto.ptr(pto.f32, "ub"),
    p_nz_ptr: pto.ptr(pto.ui16, "ub"),
    running_max_ptr: pto.ptr(pto.f32, "ub"),
    running_sum_ptr: pto.ptr(pto.f32, "ub"),
    rows: pto.i32,
    cols: pto.i32,
    scale: pto.f32,
):
    ubN = rows
    ubM = cols

    nz_buffer_ptr1 = p_nz_ptr
    nz_buffer_ptr2 = pto.addptr(nz_buffer_ptr1, 16)
    nz_buffer_ptr3 = pto.addptr(nz_buffer_ptr1, ubN // 2 * 16)
    nz_buffer_ptr4 = pto.addptr(nz_buffer_ptr1, ubN // 2 * 16 + 16)

    block_stride = ubN + 1
    repeat_stride = 2

    src0_ub = qk_ptr

    preg_134 = pto.make_mask(pto.i8, pto.MaskPattern.ALL)
    preg_f32_all = pto.make_mask(pto.f32, pto.MaskPattern.ALL)
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
        max_0a = pto.vmax(max_0a, v_row, preg_f32_all)

        v_row, p1 = pto.vlds(p1, 4 * 64, dist="NORM", post_update="ON")
        max_1a = pto.vmax(max_1a, v_row, preg_f32_all)

        v_row, p2 = pto.vlds(p2, 4 * 64, dist="NORM", post_update="ON")
        max_2a = pto.vmax(max_2a, v_row, preg_f32_all)

        v_row, p3 = pto.vlds(p3, 4 * 64, dist="NORM", post_update="ON")
        max_3a = pto.vmax(max_3a, v_row, preg_f32_all)
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

    max_0a = pto.vmax(max_0a, max_1a, preg_f32_all)
    max_2a = pto.vmax(max_2a, max_3a, preg_f32_all)
    max_0a = pto.vmax(max_0a, max_2a, preg_f32_all)

    pto.vsts(max_0a, running_max_ptr, 0, preg_f32_all, dist="NORM_B16")
    max_0a = pto.vmuls(max_0a, scale, preg_f32_all)

    vreg_x_sum_even = pto.vdup(pto.f32(0), pto.pbitcast(preg_134, pto.mask_b32))
    vreg_x_sum_odd = pto.vdup(pto.f32(0), pto.pbitcast(preg_134, pto.mask_b32))
    vreg_x_sum_1_even = pto.vdup(pto.f32(0), pto.pbitcast(preg_134, pto.mask_b32))
    vreg_x_sum_1_odd = pto.vdup(pto.f32(0), pto.pbitcast(preg_134, pto.mask_b32))

    preg_100 = preg_108
    preg_101 = preg_108
    preg_135 = preg_f32_all
    sreg_92: pto.i32 = 128
    preg_136, sreg_92 = pto.make_mask(pto.f16, sreg_92)

    input_x_half_ptr = pto.addptr(qk_ptr, ubN * ubM // 2)

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

        vreg_x_f32_a = pto.vlds(qk_ptr, i0 * 128, dist="NORM")
        vreg_x_f32_b = pto.vlds(pto.addptr(qk_ptr, 64), i0 * 128, dist="NORM")
        vreg_x_f32_1_a = pto.vlds(input_x_half_ptr, i0 * 128, dist="NORM")
        vreg_x_f32_1_b = pto.vlds(pto.addptr(input_x_half_ptr, 64), i0 * 128, dist="NORM")

        vreg_x_f32_a = pto.vmuls(vreg_x_f32_a, scale, preg_f32_all)
        vreg_x_f32_b = pto.vmuls(vreg_x_f32_b, scale, preg_f32_all)
        vreg_x_exp_even = pto.vexpdif(vreg_x_f32_a, max_0a, pto.pbitcast(preg_134, pto.mask_b32), part="ODD")
        vreg_x_exp_odd = pto.vexpdif(vreg_x_f32_b, max_0a, pto.pbitcast(preg_134, pto.mask_b32), part="ODD")

        vreg_x_f32_1_a = pto.vmuls(vreg_x_f32_1_a, scale, preg_f32_all)
        vreg_x_f32_1_b = pto.vmuls(vreg_x_f32_1_b, scale, preg_f32_all)
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

        vreg_x_sum_even = pto.vadd(vreg_x_sum_even, vreg_x_exp_even, preg_f32_all)
        vreg_x_sum_odd = pto.vadd(vreg_x_sum_odd, vreg_x_exp_odd, preg_f32_all)
        vreg_x_sum_1_even = pto.vadd(vreg_x_sum_1_even, vreg_x_exp_even_1, preg_f32_all)
        vreg_x_sum_1_odd = pto.vadd(vreg_x_sum_1_odd, vreg_x_exp_odd_1, preg_f32_all)

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

    vreg_x_sum0 = pto.vadd(vreg_x_sum_odd, vreg_x_sum_even, preg_f32_all)
    vreg_x_sum1 = pto.vadd(vreg_x_sum_1_odd, vreg_x_sum_1_even, preg_f32_all)
    vreg_x_sum0 = pto.vadd(vreg_x_sum0, vreg_x_sum1, preg_f32_all)
    pto.vsts(vreg_x_sum0, running_sum_ptr, 0, preg_f32_all, dist="NORM_B32")


@pto.jit(
    name="fa_softmax_update_vpto_kernel",
    target="a5",
    entry=False,
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def fa_softmax_update_vpto_kernel(
    qk_ptr: pto.ptr(pto.f32, "ub"),
    p_nz_ptr: pto.ptr(pto.ui16, "ub"),
    running_max_ptr: pto.ptr(pto.f32, "ub"),
    running_sum_ptr: pto.ptr(pto.f32, "ub"),
    exp_scale_ptr: pto.ptr(pto.f32, "ub"),
    rows: pto.i32,
    cols: pto.i32,
    scale: pto.f32,
):
    ubN = rows
    ubM = cols

    nz_buffer_ptr1 = p_nz_ptr
    nz_buffer_ptr2 = pto.addptr(nz_buffer_ptr1, 16)
    nz_buffer_ptr3 = pto.addptr(nz_buffer_ptr1, ubN // 2 * 16)
    nz_buffer_ptr4 = pto.addptr(nz_buffer_ptr1, ubN // 2 * 16 + 16)

    block_stride = ubN + 1
    repeat_stride = 2

    src0_ub = qk_ptr

    preg_134 = pto.make_mask(pto.i8, pto.MaskPattern.ALL)
    preg_f32_all = pto.make_mask(pto.f32, pto.MaskPattern.ALL)
    preg_108 = pto.make_mask(pto.f16, pto.MaskPattern.ALL)
    preg_low_half = pto.make_mask(pto.f16, pto.MaskPattern.VL64)
    vreg_x_max_f32_b = pto.vlds(running_max_ptr, 0, dist="NORM")

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
        max_0a = pto.vmax(max_0a, v_row, preg_f32_all)

        v_row, p1 = pto.vlds(p1, 4 * 64, dist="NORM", post_update="ON")
        max_1a = pto.vmax(max_1a, v_row, preg_f32_all)

        v_row, p2 = pto.vlds(p2, 4 * 64, dist="NORM", post_update="ON")
        max_2a = pto.vmax(max_2a, v_row, preg_f32_all)

        v_row, p3 = pto.vlds(p3, 4 * 64, dist="NORM", post_update="ON")
        max_3a = pto.vmax(max_3a, v_row, preg_f32_all)
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

    max_0a = pto.vmax(max_0a, max_1a, preg_f32_all)
    max_2a = pto.vmax(max_2a, max_3a, preg_f32_all)
    max_0a = pto.vmax(max_0a, max_2a, preg_f32_all)

    max_0a = pto.vmax(max_0a, vreg_x_max_f32_b, preg_f32_all)

    pto.vsts(max_0a, running_max_ptr, 0, preg_f32_all, dist="NORM_B16")
    max_0a = pto.vmuls(max_0a, scale, preg_f32_all)
    vreg_x_max_f32_b = pto.vexpdif(vreg_x_max_f32_b, max_0a, pto.pbitcast(preg_134, pto.mask_b32), part="ODD")

    pto.vsts(vreg_x_max_f32_b, exp_scale_ptr, 0, preg_f32_all, dist="NORM_B16")

    vreg_x_sum_even = pto.vdup(pto.f32(0), pto.pbitcast(preg_134, pto.mask_b32))
    vreg_x_sum_odd = pto.vdup(pto.f32(0), pto.pbitcast(preg_134, pto.mask_b32))
    vreg_x_sum_1_even = pto.vdup(pto.f32(0), pto.pbitcast(preg_134, pto.mask_b32))
    vreg_x_sum_1_odd = pto.vdup(pto.f32(0), pto.pbitcast(preg_134, pto.mask_b32))

    preg_100 = preg_108
    preg_101 = preg_108
    preg_135 = preg_f32_all
    sreg_92: pto.i32 = 128
    preg_136, sreg_92 = pto.make_mask(pto.f16, sreg_92)

    input_x_half_ptr = pto.addptr(qk_ptr, ubN * ubM // 2)

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

        vreg_x_f32_a = pto.vlds(qk_ptr, i0 * 128, dist="NORM")
        vreg_x_f32_b = pto.vlds(pto.addptr(qk_ptr, 64), i0 * 128, dist="NORM")
        vreg_x_f32_1_a = pto.vlds(input_x_half_ptr, i0 * 128, dist="NORM")
        vreg_x_f32_1_b = pto.vlds(pto.addptr(input_x_half_ptr, 64), i0 * 128, dist="NORM")

        vreg_x_f32_a = pto.vmuls(vreg_x_f32_a, scale, preg_f32_all)
        vreg_x_f32_b = pto.vmuls(vreg_x_f32_b, scale, preg_f32_all)
        vreg_x_exp_even = pto.vexpdif(vreg_x_f32_a, max_0a, pto.pbitcast(preg_134, pto.mask_b32), part="ODD")
        vreg_x_exp_odd = pto.vexpdif(vreg_x_f32_b, max_0a, pto.pbitcast(preg_134, pto.mask_b32), part="ODD")

        vreg_x_f32_1_a = pto.vmuls(vreg_x_f32_1_a, scale, preg_f32_all)
        vreg_x_f32_1_b = pto.vmuls(vreg_x_f32_1_b, scale, preg_f32_all)
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

        vreg_x_sum_even = pto.vadd(vreg_x_sum_even, vreg_x_exp_even, preg_f32_all)
        vreg_x_sum_odd = pto.vadd(vreg_x_sum_odd, vreg_x_exp_odd, preg_f32_all)
        vreg_x_sum_1_even = pto.vadd(vreg_x_sum_1_even, vreg_x_exp_even_1, preg_f32_all)
        vreg_x_sum_1_odd = pto.vadd(vreg_x_sum_1_odd, vreg_x_exp_odd_1, preg_f32_all)

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

    vreg_x_sum0 = pto.vadd(vreg_x_sum_odd, vreg_x_sum_even, preg_f32_all)
    vreg_x_sum1 = pto.vadd(vreg_x_sum_1_odd, vreg_x_sum_1_even, preg_f32_all)
    vreg_x_sum0 = pto.vadd(vreg_x_sum0, vreg_x_sum1, preg_f32_all)
    pto.vsts(vreg_x_sum0, running_sum_ptr, 0, preg_f32_all, dist="NORM_B32")


@pto.simd
def fa_softmax_init_vpto(
    qk: pto.Tile,
    p_nz: pto.Tile,
    running_max: pto.Tile,
    running_sum: pto.Tile,
    scale: pto.f32,
):
    rows, cols = qk.shape
    fa_softmax_init_vpto_kernel(
        qk.as_ptr(),
        p_nz.as_ptr(),
        running_max.as_ptr(),
        running_sum.as_ptr(),
        rows,
        cols,
        scale,
    )


@pto.simd
def fa_softmax_update_vpto(
    qk: pto.Tile,
    p_nz: pto.Tile,
    running_max: pto.Tile,
    running_sum: pto.Tile,
    exp_scale: pto.Tile,
    scale: pto.f32,
):
    rows, cols = qk.shape
    fa_softmax_update_vpto_kernel(
        qk.as_ptr(),
        p_nz.as_ptr(),
        running_max.as_ptr(),
        running_sum.as_ptr(),
        exp_scale.as_ptr(),
        rows,
        cols,
        scale,
    )


@pto.jit(target="a5", mode="explicit")
def fa_softmax_vpto_probe(
    *,
    BR: pto.constexpr = 8,
    BC: pto.constexpr = 64,
    INIT: pto.constexpr = False,
    HEAD_SIZE: pto.constexpr = 64,
):
    cBR = pto.const(BR, dtype=pto.i32)
    cBC = pto.const(BC, dtype=pto.i32)
    scale_const = pto.const(_inv_sqrt(HEAD_SIZE), dtype=pto.f32)

    qk = pto.alloc_tile(shape=[BR, BC], dtype=pto.f32, valid_shape=[BR, BC])
    p_nz = pto.alloc_tile(shape=[BR, BC], dtype=pto.ui16, valid_shape=[BR, BC])
    running_max = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")
    running_sum = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")

    if INIT:
        fa_softmax_init_vpto(
            qk,
            p_nz,
            running_max,
            running_sum,
            scale_const,
        )
    else:
        exp_scale = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")
        fa_softmax_update_vpto(
            qk,
            p_nz,
            running_max,
            running_sum,
            exp_scale,
            scale_const,
        )

__all__ = [
    "fa_softmax_init_vpto_kernel",
    "fa_softmax_update_vpto_kernel",
    "fa_softmax_init_vpto",
    "fa_softmax_update_vpto",
    "fa_softmax_vpto_probe",
]


def main() -> None:
    compiled = fa_softmax_vpto_probe.compile(
        BR=8,
        BC=64,
        INIT=False,
        HEAD_SIZE=64,
    )
    print(compiled.mlir_text())


if __name__ == "__main__":
    main()
