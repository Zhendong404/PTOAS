#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/trowexpandmul.
#
# pto.trowexpandmul: dst[i,j] = src0[i,j] * src1[i,0] — src1 is a per-row
# scalar broadcast across all dst columns (legacy src1Col=1).  src1 is stored
# in a physically aligned tile (32/sizeof(dtype) columns, e.g. 8 for f32)
# whose valid region is (rows, 1), encoding the legacy src1Col=1; the TileLib
# template vdup's the LOWEST lane of src1[row, :] so only the first column
# takes part.  src0eqdst is a legacy aliasing hint that auto mode handles
# implicitly.  All 7 legacy cases (f32/f16/i32/i16, fully valid src0/dst) are
# preserved in PTODSL auto mode; tile addresses, partitions and sync are left
# to PTOAS.

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
    np.int32: pto.i32,
    np.int16: pto.i16,
}

# (legacy case name, numpy dtype, src0_shape, src0_valid_shape, src1_shape,
#  src1_valid_shape, dst_shape, dst_valid_shape, eps)
# src1 physical cols = 32/sizeof(dtype) for NPU alignment (f32 -> 8, f16/i16
# -> 16, i32 -> 8); src1_valid_shape (rows, 1) encodes legacy src1Col=1.
# src0/dst are fully valid in every legacy case; src0eqdst=true/false comes
# from the legacy cases.py/.pto (f16_32x64_noeq is src0eqdst=false).
CASE_SPECS = [
    ("f32_16x32",     np.float32, (16, 32),   (16, 32),   (16, 8),   (16, 1),   (16, 32),   (16, 32),   1e-6),
    ("f32_56x128",    np.float32, (56, 128),  (56, 128),  (56, 8),   (56, 1),   (56, 128),  (56, 128),  1e-6),
    ("f16_48x64",     np.float16, (48, 64),   (48, 64),   (48, 16),  (48, 1),   (48, 64),   (48, 64),   1e-3),
    ("f16_16x128",    np.float16, (16, 128),  (16, 128),  (16, 16),  (16, 1),   (16, 128),  (16, 128),  1e-3),
    ("f16_32x64_noeq", np.float16, (32, 64),  (32, 64),   (32, 16),  (32, 1),   (32, 64),   (32, 64),   1e-3),
    ("i32_16x32",     np.int32,   (16, 32),   (16, 32),   (16, 8),   (16, 1),   (16, 32),   (16, 32),   0),
    ("i16_16x64",     np.int16,   (16, 64),   (16, 64),   (16, 16),  (16, 1),   (16, 64),   (16, 64),   0),
]


def _trowexpandmul_body(src0_ptr, src1_ptr, dst_ptr, *, rows, cols, src1_cols, dtype):
    src0_view = pto.make_tensor_view(src0_ptr, shape=[rows, cols], strides=[cols, 1])
    src1_view = pto.make_tensor_view(src1_ptr, shape=[rows, src1_cols],
                                     strides=[src1_cols, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[rows, cols], strides=[cols, 1])

    src0_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)
    src1_tile = pto.alloc_tile(shape=[rows, src1_cols], dtype=dtype,
                               valid_shape=[rows, 1])
    dst_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)

    pto.tile.load(src0_view, src0_tile)
    pto.tile.load(src1_view, src1_tile)
    pto.tile.rowexpandmul(src0_tile, src1_tile, dst_tile)
    pto.tile.store(dst_tile, dst_view)


_kernels = {}
for _name, _np_dtype, _src0_shape, _src0_valid, _src1_shape, _src1_valid, \
        _dst_shape, _dst_valid, _eps in CASE_SPECS:
    _r, _c = _src0_shape
    _s1r, _s1c = _src1_shape
    _pto_dtype = NP_TO_PTO[_np_dtype]

    def _make(r=_r, c=_c, src1_cols=_s1c, dtype=_pto_dtype,
              kernel_name=f"trowexpandmul_{_name}"):
        @pto.jit(name=kernel_name, target="a5")
        def _kernel(
            src0_ptr: pto.ptr(dtype, "gm"),
            src1_ptr: pto.ptr(dtype, "gm"),
            dst_ptr: pto.ptr(dtype, "gm"),
        ):
            _trowexpandmul_body(src0_ptr, src1_ptr, dst_ptr, rows=r, cols=c,
                                src1_cols=src1_cols, dtype=dtype)

        return _kernel

    _kernels[_name] = _make()


def _make_inputs(name, np_dtype, src0_shape, src1_shape):
    # Mirrors st_common.setup_case_rng (per-case crc32 seed) and the legacy
    # gen_data.py draw order: input1 (src0) then input2 (src1).
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    input1 = np.random.randint(1, 10, size=src0_shape).astype(np_dtype)
    input2 = np.random.randint(1, 10, size=src1_shape).astype(np_dtype)
    return [input1, input2]


def _make_expected(input1, input2, dst_shape, dst_valid_shape,
                   src0_valid_shape, src1_valid_shape):
    # Mirrors the legacy gen_data.py golden: broadcast src1 column 0 across
    # dst columns (src1Col=1), then store into a dtype buffer.
    dst_vr, dst_vc = dst_valid_shape
    src0_vr, src0_vc = src0_valid_shape
    src1_vr = src1_valid_shape[0]
    golden = np.zeros(dst_shape, dtype=input1.dtype)
    golden[:dst_vr, :dst_vc] = (
        input1[:src0_vr, :src0_vc] * input2[:src1_vr, 0:1]
    ).astype(input1.dtype, copy=False)
    return golden


CASES = [
    golden_output_case(
        "trowexpandmul_" + name,
        _kernels[name],
        inputs=lambda name=name, np_dtype=np_dtype, src0_shape=src0_shape,
                       src1_shape=src1_shape: _make_inputs(
                           name, np_dtype, src0_shape, src1_shape),
        expected=lambda input1, input2, dst_shape=dst_shape,
                       dst_valid_shape=dst_valid_shape,
                       src0_valid_shape=src0_valid_shape,
                       src1_valid_shape=src1_valid_shape: _make_expected(
                           input1, input2, dst_shape, dst_valid_shape,
                           src0_valid_shape, src1_valid_shape),
        rtol=eps,
        atol=eps,
    )
    for name, np_dtype, src0_shape, src0_valid_shape, src1_shape, \
        src1_valid_shape, dst_shape, dst_valid_shape, eps in CASE_SPECS
]


auto_main(globals())
