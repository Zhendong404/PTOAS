#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tsel.
#
# TSEL selects per element: dst = src0 if mask bit is 1 else src1.
# The mask is a packed bitmask: each byte covers 8 consecutive columns, so the
# mask tile has (cols + 7) // 8 bytes per row.  This mirrors the legacy .pto
# kernel (tload(mask) + tload(src0) + tload(src1) + tsel + tstore) and the
# legacy gen_data.py packed-mask golden logic.

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


ROWS = 2

# (case suffix, pto dtype, numpy dtype, cols, tolerance)
CASE_SPECS = [
    ("f32_2x128", pto.f32, np.float32, 128, 1e-6),
    ("f32_2x32",  pto.f32, np.float32, 32,  1e-6),
    ("f32_2x160", pto.f32, np.float32, 160, 1e-6),
    ("f32_2x512", pto.f32, np.float32, 512, 1e-6),
    ("f16_2x128", pto.f16, np.float16, 128, 1e-3),
    ("f16_2x32",  pto.f16, np.float16, 32,  1e-3),
    ("f16_2x160", pto.f16, np.float16, 160, 1e-3),
    ("i8_2x128",  pto.i8,  np.int8,    128, 0),
    ("i8_2x32",   pto.i8,  np.int8,    32,  0),
    ("i8_2x160",  pto.i8,  np.int8,    160, 0),
]


def _mask_cols(cols: int) -> int:
    return (cols + 7) // 8


def _make_kernel(name: str, pto_dtype, cols: int, rows: int = ROWS):
    mask_cols = _mask_cols(cols)
    # Row-major i8 tiles must be 32-byte aligned; the physical mask tile keeps
    # an aligned row count and the logical tail is expressed via valid_shape.
    mask_phys_cols = ((mask_cols + 31) // 32) * 32

    @pto.jit(name="tsel_" + name, target="a5")
    def _kernel(
        mask_ptr: pto.ptr(pto.i8, "gm"),
        src0_ptr: pto.ptr(pto_dtype, "gm"),
        src1_ptr: pto.ptr(pto_dtype, "gm"),
        dst_ptr: pto.ptr(pto_dtype, "gm"),
    ):
        mask_view = pto.make_tensor_view(
            mask_ptr, shape=[rows, mask_cols], strides=[mask_cols, 1]
        )
        src0_view = pto.make_tensor_view(
            src0_ptr, shape=[rows, cols], strides=[cols, 1]
        )
        src1_view = pto.make_tensor_view(
            src1_ptr, shape=[rows, cols], strides=[cols, 1]
        )
        dst_view = pto.make_tensor_view(
            dst_ptr, shape=[rows, cols], strides=[cols, 1]
        )

        mask_tile = pto.alloc_tile(
            shape=[rows, mask_phys_cols], dtype=pto.i8, valid_shape=[rows, mask_cols]
        )
        src0_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype)
        src1_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype)
        dst_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype)

        pto.tile.load(mask_view, mask_tile)
        pto.tile.load(src0_view, src0_tile)
        pto.tile.load(src1_view, src1_tile)
        pto.tile.sel(mask_tile, src0_tile, src1_tile, dst_tile)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


_kernels = {name: _make_kernel(name, pto_dtype, cols)
            for name, pto_dtype, _, cols, _ in CASE_SPECS}


def _make_inputs(name: str, cols: int, np_dtype, rows: int = ROWS):
    # Deterministic per-case seed, mirroring st_common.setup_case_rng which uses
    # crc32(name).  Original value range was randint(1, 10); mask is a packed
    # uint8 bitmask with (cols + 7) // 8 bytes per row.
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    mask_cols = _mask_cols(cols)
    src0 = np.random.randint(1, 10, size=(rows, cols)).astype(np_dtype)
    src1 = np.random.randint(1, 10, size=(rows, cols)).astype(np_dtype)
    mask = np.random.randint(0, 256, size=(rows, mask_cols)).astype(np.uint8)
    return [mask, src0, src1]


def _make_expected(mask, src0, src1):
    rows, cols = src0.shape
    mask_cols = _mask_cols(cols)
    golden = np.zeros(src0.shape, dtype=src0.dtype)
    for row in range(rows):
        for packed_col in range(mask_cols):
            byte = int(mask[row, packed_col])
            for bit in range(8):
                col = packed_col * 8 + bit
                if col >= cols:
                    break
                golden[row, col] = src0[row, col] if ((byte >> bit) & 1) else src1[row, col]
    return golden


CASES = [
    golden_output_case(
        "tsel_" + name,
        _kernels[name],
        inputs=lambda name=name, cols=cols, np_dtype=np_dtype: _make_inputs(name, cols, np_dtype),
        expected=_make_expected,
        rtol=eps,
        atol=eps,
    )
    for name, _, np_dtype, cols, eps in CASE_SPECS
]


auto_main(globals())
