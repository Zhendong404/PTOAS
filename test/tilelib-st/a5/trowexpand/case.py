#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/trowexpand.
#   tload(src) + trowexpand(src) -> dst + tstore(dst)
#
# Row expansion: dst[i, :] = src[i, 0] broadcast across columns.  The source
# is a per-row scalar stored in a physically aligned tile whose valid region is
# (rows, 1); the destination valid region may be a partial row tail (e.g.
# 16x127 in a 16x128 physical tile), mirroring the legacy cases.

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
    pto.f16: np.float16,
    pto.i8: np.int8,
}

# (case suffix, pto dtype, src0_shape, src0_valid_shape, dst_shape, dst_valid_shape, eps)
CASE_SPECS = [
    ("f32_16x128", pto.f32, (16, 8),   (16, 1),   (16, 128), (16, 128), 1e-6),
    ("f32_16x127", pto.f32, (16, 8),   (16, 1),   (16, 128), (16, 127), 1e-6),
    ("f16_16x512", pto.f16, (16, 16),  (16, 1),   (16, 512), (16, 512), 1e-3),
    ("f16_16x511", pto.f16, (16, 16),  (16, 1),   (16, 512), (16, 511), 1e-3),
    ("i8_16x256",  pto.i8,  (16, 32),  (16, 1),   (16, 256), (16, 256), 0),
    ("i8_16x255",  pto.i8,  (16, 32),  (16, 1),   (16, 256), (16, 255), 0),
]


def _make_kernel(name: str, pto_dtype, src0_shape, src0_valid_shape, dst_shape, dst_valid_shape):
    src_rows, src_cols = src0_shape
    src_valid_rows, src_valid_cols = src0_valid_shape
    dst_rows, dst_cols = dst_shape
    dst_valid_rows, dst_valid_cols = dst_valid_shape

    @pto.jit(name="trowexpand_" + name, target="a5")
    def _kernel(
        src_ptr: pto.ptr(pto_dtype, "gm"),
        dst_ptr: pto.ptr(pto_dtype, "gm"),
    ):
        src_view = pto.make_tensor_view(src_ptr, shape=[src_rows, src_cols], strides=[src_cols, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[dst_rows, dst_cols], strides=[dst_cols, 1])

        src_tile = pto.alloc_tile(
            shape=[src_rows, src_cols], dtype=pto_dtype,
            valid_shape=[src_valid_rows, src_valid_cols],
        )
        dst_tile = pto.alloc_tile(
            shape=[dst_rows, dst_cols], dtype=pto_dtype,
            valid_shape=[dst_valid_rows, dst_valid_cols],
        )

        pto.tile.load(src_view, src_tile)
        pto.tile.rowexpand(src_tile, dst_tile)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


_kernels = {
    name: _make_kernel(name, pto_dtype, src0_shape, src0_valid_shape, dst_shape, dst_valid_shape)
    for name, pto_dtype, src0_shape, src0_valid_shape, dst_shape, dst_valid_shape, _ in CASE_SPECS
}


def _make_inputs(name: str, np_dtype, src0_shape, src0_valid_shape):
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    src = np.zeros(src0_shape, dtype=np_dtype)
    src_vr = src0_valid_shape[0]
    src[:src_vr, 0] = np.random.randint(1, 10, size=src_vr).astype(np_dtype)
    return [src]


def _make_expected(src, src0_valid_shape, dst_shape, dst_valid_shape):
    dst_vr, dst_vc = dst_valid_shape
    src_vr = src0_valid_shape[0]
    golden = np.zeros(dst_shape, dtype=src.dtype)
    golden[:dst_vr, :dst_vc] = np.broadcast_to(src[:src_vr, 0:1], (dst_vr, dst_vc))
    return golden


CASES = [
    golden_output_case(
        "trowexpand_" + name,
        _kernels[name],
        inputs=lambda name=name, np_dtype=PTO_TO_NP[pto_dtype], src0_shape=src0_shape, src0_valid_shape=src0_valid_shape: _make_inputs(name, np_dtype, src0_shape, src0_valid_shape),
        expected=lambda src, src0_valid_shape=src0_valid_shape, dst_shape=dst_shape, dst_valid_shape=dst_valid_shape: _make_expected(src, src0_valid_shape, dst_shape, dst_valid_shape),
        rtol=eps,
        atol=eps,
    )
    for name, pto_dtype, src0_shape, src0_valid_shape, dst_shape, dst_valid_shape, eps in CASE_SPECS
]


auto_main(globals())
