#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tpartmax.
#   tload(src0) + tload(src1) + pto.tile.partmax(src0, src1) -> dst + tstore(dst)
#
# tpartmax semantics (matches the legacy gen_data.py golden and the PTODSL
# template_tpartmax lowering): dst is first padded with the dtype minimum,
# then src0's valid region is copied into dst, then dst = max(dst, src1) over
# src1's valid region.  The TileLib partmax template accepts arbitrary src0/
# src1 valid regions nested inside the dst valid region (the "complex" cases
# where neither operand covers dst), unlike partadd which needs one full
# operand.
#
# Auto mode: tile addresses, load/store partitions and sync are left to PTOAS.

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto

NP_TO_PTO = {
    np.float32: pto.f32,
    np.float16: pto.f16,
    np.int16: pto.i16,
    np.int32: pto.i32,
    np.uint16: pto.ui16,
    np.uint32: pto.ui32,
    np.int8: pto.i8,
    np.uint8: pto.ui8,
}

# (legacy case name, numpy dtype, shape, src0 valid, src1 valid, dst valid, eps)
# Every legacy case has dst valid == full shape; src0/src1 valid regions anchor
# at the tile origin and participate in the partial max.
CASE_SPECS = [
    ("f32_64x64_full", np.float32, (64, 64), (64, 64), (64, 64), (64, 64), 1e-6),
    ("f32_2x24_src1_col_less", np.float32, (2, 24), (2, 24), (2, 8), (2, 24), 1e-6),
    ("f32_128x64_src1_row_less", np.float32, (128, 64), (128, 64), (96, 64), (128, 64), 1e-6),
    ("f32_95x95_full", np.float32, (95, 95), (95, 95), (95, 95), (95, 95), 1e-6),
    ("f32_122x123_complex", np.float32, (122, 123), (104, 123), (122, 110), (122, 123), 1e-6),
    ("f16_122x123_complex", np.float16, (122, 123), (104, 123), (122, 110), (122, 123), 1e-3),
    ("i16_122x123_complex", np.int16, (122, 123), (104, 123), (122, 110), (122, 123), 0),
    ("i32_122x123_complex", np.int32, (122, 123), (104, 123), (122, 110), (122, 123), 0),
    ("u16_122x123_complex", np.uint16, (122, 123), (104, 123), (122, 110), (122, 123), 0),
    ("u32_122x123_complex", np.uint32, (122, 123), (104, 123), (122, 110), (122, 123), 0),
    ("i8_122x123_complex", np.int8, (122, 123), (104, 123), (122, 110), (122, 123), 0),
    ("u8_122x123_complex", np.uint8, (122, 123), (104, 123), (122, 110), (122, 123), 0),
]


def _make_kernel(name, np_dtype, shape, src0_valid, src1_valid, dst_valid):
    """One @pto.jit kernel per case, binding dtype/shape/valid regions statically
    (mirroring the per-case funcs in tpartmax.pto)."""
    rows, cols = shape
    src0_vr, src0_vc = src0_valid
    src1_vr, src1_vc = src1_valid
    dst_vr, dst_vc = dst_valid
    pto_dtype = NP_TO_PTO[np_dtype]

    @pto.jit(name="tpartmax_" + name, target="a5")
    def _kernel(
        a_ptr: pto.ptr(pto_dtype, "gm"),
        b_ptr: pto.ptr(pto_dtype, "gm"),
        c_ptr: pto.ptr(pto_dtype, "gm"),
    ):
        a_view = pto.make_tensor_view(a_ptr, shape=[rows, cols], strides=[cols, 1])
        b_view = pto.make_tensor_view(b_ptr, shape=[rows, cols], strides=[cols, 1])
        c_view = pto.make_tensor_view(c_ptr, shape=[rows, cols], strides=[cols, 1])

        a_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=pto_dtype, valid_shape=[src0_vr, src0_vc]
        )
        b_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=pto_dtype, valid_shape=[src1_vr, src1_vc]
        )
        c_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=pto_dtype, valid_shape=[dst_vr, dst_vc]
        )

        pto.tile.load(a_view, a_tile)
        pto.tile.load(b_view, b_tile)
        pto.tile.partmax(a_tile, b_tile, c_tile)
        pto.tile.store(c_tile, c_view)

    return _kernel


_kernels = {
    name: _make_kernel(name, np_dtype, shape, src0_valid, src1_valid, dst_valid)
    for name, np_dtype, shape, src0_valid, src1_valid, dst_valid, _ in CASE_SPECS
}


def _make_inputs(name, np_dtype, shape):
    # Deterministic per-case seed (crc32 of the legacy case name), mirroring
    # st_common.setup_case_rng; value range randint(1, 10) as in gen_data.py.
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    a = np.random.randint(1, 10, size=shape).astype(np_dtype)
    b = np.random.randint(1, 10, size=shape).astype(np_dtype)
    return [a, b]


def _min_val(np_dtype):
    """Dtype minimum used as the tpartmax padding value (template_tpartmax's
    pad_min / the legacy gen_data.py corner fill)."""
    if np_dtype == np.float32:
        return np.finfo(np.float32).min
    if np_dtype == np.float16:
        return np.finfo(np.float16).min
    if np_dtype in {np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32}:
        return np.iinfo(np_dtype).min
    return np.iinfo(np_dtype).min


def _make_expected(a, b, src0_valid, src1_valid, dst_valid):
    """Reproduce gen_data.py tpartmax golden semantics exactly:

    dst = Min, dst[0:src0_vr, 0:src0_vc] = src0, then
    dst[0:src1_vr, 0:src1_vc] = max(dst, src1).  The corner not covered by
    either valid region keeps the Min padding (dtype minimum).
    """
    src0_vr, src0_vc = src0_valid
    src1_vr, src1_vc = src1_valid
    dst_vr, dst_vc = dst_valid
    dtype = a.dtype
    golden = np.zeros(a.shape, dtype=dtype)

    src0_eq_dst = (src0_vr == dst_vr and src0_vc == dst_vc)
    src1_eq_dst = (src1_vr == dst_vr and src1_vc == dst_vc)

    if src0_eq_dst and src1_eq_dst:
        # Full max: both src0 and src1 cover the entire dst
        golden[:dst_vr, :dst_vc] = np.maximum(
            a[:dst_vr, :dst_vc], b[:dst_vr, :dst_vc]
        ).astype(dtype, copy=False)
    elif src0_eq_dst:
        # src0 covers dst, src1 is partial: max in src1 region, src0 elsewhere
        golden[:src1_vr, :src1_vc] = np.maximum(
            a[:src1_vr, :src1_vc], b[:src1_vr, :src1_vc]
        ).astype(dtype, copy=False)
        if src1_vc < dst_vc:
            golden[:src1_vr, src1_vc:dst_vc] = a[:src1_vr, src1_vc:dst_vc].copy()
        if src1_vr < dst_vr:
            golden[src1_vr:dst_vr, :dst_vc] = a[src1_vr:dst_vr, :dst_vc].copy()
    elif src1_eq_dst:
        # src1 covers dst, src0 is partial: max in src0 region, src1 elsewhere
        golden[:src0_vr, :src0_vc] = np.maximum(
            a[:src0_vr, :src0_vc], b[:src0_vr, :src0_vc]
        ).astype(dtype, copy=False)
        if src0_vc < dst_vc:
            golden[:src0_vr, src0_vc:dst_vc] = b[:src0_vr, src0_vc:dst_vc].copy()
        if src0_vr < dst_vr:
            golden[src0_vr:dst_vr, :dst_vc] = b[src0_vr:dst_vr, :dst_vc].copy()
    else:
        min_vr = min(src0_vr, src1_vr)
        min_vc = min(src0_vc, src1_vc)

        # Overlapping region: max of both operands
        golden[:min_vr, :min_vc] = np.maximum(
            a[:min_vr, :min_vc], b[:min_vr, :min_vc]
        ).astype(dtype, copy=False)

        # src0-only region (right of the overlap)
        if src0_vc > min_vc:
            golden[:src0_vr, min_vc:src0_vc] = a[:src0_vr, min_vc:src0_vc].copy()

        # src1-only region (below the overlap)
        if src1_vr > min_vr:
            golden[min_vr:src1_vr, :min_vc] = b[min_vr:src1_vr, :min_vc].copy()

        # src1-only corner
        if src1_vr > min_vr and src1_vc > min_vc:
            golden[min_vr:src1_vr, min_vc:src1_vc] = b[min_vr:src1_vr, min_vc:src1_vc].copy()

        # Corner covered by neither operand keeps the Min padding
        if src1_vr > src0_vr and src0_vc > src1_vc:
            golden[src0_vr:src1_vr, src1_vc:src0_vc] = _min_val(dtype)

    return golden


CASES = [
    golden_output_case(
        "tpartmax_" + name,
        _kernels[name],
        inputs=lambda name=name, np_dtype=np_dtype, shape=shape: _make_inputs(name, np_dtype, shape),
        expected=lambda a, b, src0_valid=src0_valid, src1_valid=src1_valid, dst_valid=dst_valid: _make_expected(
            a, b, src0_valid, src1_valid, dst_valid
        ),
        rtol=eps,
        atol=eps,
    )
    for name, np_dtype, shape, src0_valid, src1_valid, dst_valid, eps in CASE_SPECS
]


auto_main(globals())
