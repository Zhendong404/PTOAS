#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tpartmul.
#   tload(src0) + tload(src1) + pto.tile.partmul(src0, src1) -> dst + tstore(dst)
#
# tpartmul semantics (matches the legacy gen_data.py golden and the PTODSL
# template_tpartmul lowering in lib/TileOps/a5/_part.py): one operand is "full"
# (its valid_shape equals the dst valid region) and provides the base values
# over the whole dst; the other operand may be partial (row_less or col_less).
# dst = full * partial over the partial valid region, and dst = full elsewhere.
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
}

# (legacy case name, numpy dtype, shape, src0 valid, src1 valid, dst valid, eps)
# src0/src1 valid regions anchor at the tile origin; dst valid == full shape in
# every legacy case.  Both operands keep the full allocated shape in GM, and the
# partial operand's valid region is what participates in the mul.
CASE_SPECS = [
    ("f32_64x64_full", np.float32, (64, 64), (64, 64), (64, 64), (64, 64), 1e-6),
    ("f32_64x64_src0_row_less", np.float32, (64, 64), (8, 64), (64, 64), (64, 64), 1e-6),
    ("f32_64x64_src0_col_less", np.float32, (64, 64), (64, 8), (64, 64), (64, 64), 1e-6),
    ("f32_64x64_src1_row_less", np.float32, (64, 64), (64, 64), (8, 64), (64, 64), 1e-6),
    ("f32_64x64_src1_col_less", np.float32, (64, 64), (64, 64), (64, 8), (64, 64), 1e-6),
    ("f16_8x48_src0_col_less", np.float16, (8, 48), (8, 16), (8, 48), (8, 48), 1e-3),
    ("f16_8x768_src0_col_less", np.float16, (8, 768), (8, 512), (8, 768), (8, 768), 1e-3),
    ("i16_8x48_src1_col_less", np.int16, (8, 48), (8, 48), (8, 16), (8, 48), 0),
    ("i32_64x64_src0_row_less", np.int32, (64, 64), (8, 64), (64, 64), (64, 64), 0),
]


def _make_kernel(name, np_dtype, shape, src0_valid, src1_valid, dst_valid):
    """One @pto.jit kernel per case, binding dtype/shape/valid regions statically
    (mirroring the per-case funcs in tpartmul.pto)."""
    rows, cols = shape
    src0_vr, src0_vc = src0_valid
    src1_vr, src1_vc = src1_valid
    dst_vr, dst_vc = dst_valid
    pto_dtype = NP_TO_PTO[np_dtype]

    @pto.jit(name="tpartmul_" + name, target="a5")
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
        pto.tile.partmul(a_tile, b_tile, c_tile)
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


def _make_expected(a, b, src0_valid, src1_valid, dst_valid):
    """Reproduce gen_data.py tpartmul golden semantics exactly.

    If src0_valid == dst_valid, src0 is the full operand (base values come
    from src0, src1 is the possibly-partial multiplicand).  Otherwise src1 is
    the full operand and the roles swap.
    """
    src0_vr, src0_vc = src0_valid
    src1_vr, src1_vc = src1_valid
    dst_vr, dst_vc = dst_valid
    dtype = a.dtype
    golden = np.zeros(a.shape, dtype=dtype)

    src0_eq_dst = (src0_vr == dst_vr and src0_vc == dst_vc)
    src1_eq_dst = (src1_vr == dst_vr and src1_vc == dst_vc)

    if src0_eq_dst:
        # src0 is the full operand matching dst
        if src1_eq_dst:
            golden[:dst_vr, :dst_vc] = (a[:dst_vr, :dst_vc] * b[:dst_vr, :dst_vc]).astype(dtype, copy=False)
        elif src1_vc < dst_vc:
            # src1 col_less: copy src0 full, then mul in the overlapping region
            golden[:dst_vr, :dst_vc] = a[:dst_vr, :dst_vc].copy()
            golden[:src1_vr, :src1_vc] = (a[:src1_vr, :src1_vc] * b[:src1_vr, :src1_vc]).astype(dtype, copy=False)
        else:
            # src1 row_less: mul in the src1 region, copy src0 for remaining rows
            golden[:src1_vr, :src1_vc] = (a[:src1_vr, :src1_vc] * b[:src1_vr, :src1_vc]).astype(dtype, copy=False)
            golden[src1_vr:dst_vr, :dst_vc] = a[src1_vr:dst_vr, :dst_vc].copy()
    elif src1_eq_dst:
        # src1 is the full operand matching dst (swap roles vs the branch above)
        if src0_vc < dst_vc:
            # src0 col_less: copy src1 full, then mul in the overlapping region
            golden[:dst_vr, :dst_vc] = b[:dst_vr, :dst_vc].copy()
            golden[:src0_vr, :src0_vc] = (a[:src0_vr, :src0_vc] * b[:src0_vr, :src0_vc]).astype(dtype, copy=False)
        else:
            # src0 row_less: mul in the src0 region, copy src1 for remaining rows
            golden[:src0_vr, :src0_vc] = (a[:src0_vr, :src0_vc] * b[:src0_vr, :src0_vc]).astype(dtype, copy=False)
            golden[src0_vr:dst_vr, :dst_vc] = b[src0_vr:dst_vr, :dst_vc].copy()

    return golden


CASES = [
    golden_output_case(
        "tpartmul_" + name,
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
