#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/texpands.
#   pto.texpands: dst = broadcast(scalar) over the tile's valid region
#
# The legacy kernel takes only a dst pointer: the scalar is baked into each
# kernel as an arith.constant (one func per case in texpands.pto), and the
# tstore partition sizes equal the tile valid_shape.  This is reproduced here
# with one auto-mode pto.tile.expands kernel per case (scalar as a Python
# constant, dst tile with the legacy shape/valid_shape), and no host inputs —
# golden_output_case then passes only the zero-initialized output buffer to
# the kernel.  Golden follows gen_data.py: zeros(shape) with the scalar in the
# valid region; only the valid region is written by tstore.

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto

NP_TO_PTO = {
    np.float32: pto.f32,
    np.float16: pto.f16,
    np.int32: pto.i32,
    np.int16: pto.i16,
}

# (legacy case suffix, numpy dtype, shape, valid_shape, scalar, eps)
CASE_SPECS = [
    # ========== float32 cases ==========
    ("f32_16x64_scalar5", np.float32, (16, 64), (16, 64), 5.0, 1e-06),
    ("f32_32x32_scalar3", np.float32, (32, 32), (32, 32), 3.0, 1e-06),
    ("f32_64x64_scalar2", np.float32, (64, 64), (64, 64), 2.0, 1e-06),
    ("f32_16x64_partial", np.float32, (16, 64), (12, 48), 7.0, 1e-06),
    ("f32_64x64_valid_60x60", np.float32, (64, 64), (60, 60), 42.0, 1e-06),
    # ========== int32 cases ==========
    ("i32_64x64_scalar100", np.int32, (64, 64), (64, 64), 100, 0.0),
    ("i32_64x64_valid_60x60", np.int32, (64, 64), (60, 60), 99, 0.0),
    # ========== half (fp16) cases ==========
    ("f16_64x64_scalar1_5", np.float16, (64, 64), (64, 64), 1.5, 1e-03),
    ("f16_2x4096_valid_1x3600", np.float16, (2, 4096), (1, 3600), 2.5, 1e-03),
    # ========== int16 cases ==========
    ("i16_64x64_scalar50", np.int16, (64, 64), (64, 64), 50, 0.0),
    ("i16_20x512_valid_16x200", np.int16, (20, 512), (16, 200), 25, 0.0),
]


def _make_kernel(name: str, np_dtype, shape, valid_shape, scalar):
    rows, cols = shape
    valid_rows, valid_cols = valid_shape
    pto_dtype = NP_TO_PTO[np_dtype]
    scalar_val = int(scalar) if np.issubdtype(np_dtype, np.integer) else float(scalar)

    @pto.jit(name="texpands_" + name, target="a5")
    def _kernel(dst_ptr: pto.ptr(pto_dtype, "gm")):
        dst_view = pto.make_tensor_view(dst_ptr, shape=[rows, cols], strides=[cols, 1])

        dst_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=pto_dtype, valid_shape=[valid_rows, valid_cols]
        )

        pto.tile.expands(scalar_val, dst_tile)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


_kernels = {
    name: _make_kernel(name, np_dtype, shape, valid_shape, scalar)
    for name, np_dtype, shape, valid_shape, scalar, _ in CASE_SPECS
}


def _make_expected(shape, valid_shape, np_dtype, scalar):
    valid_rows, valid_cols = valid_shape
    golden = np.zeros(shape, dtype=np_dtype)
    golden[:valid_rows, :valid_cols] = np_dtype(scalar)
    return golden


CASES = [
    golden_output_case(
        "texpands_" + name,
        _kernels[name],
        inputs=[],
        expected=lambda shape=shape, valid_shape=valid_shape, np_dtype=np_dtype, scalar=scalar: _make_expected(
            shape, valid_shape, np_dtype, scalar
        ),
        rtol=eps,
        atol=eps,
    )
    for name, np_dtype, shape, valid_shape, scalar, eps in CASE_SPECS
]


auto_main(globals())
