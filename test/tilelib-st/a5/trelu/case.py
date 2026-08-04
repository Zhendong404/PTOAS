#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/trelu.
#
# pto.trelu: dst = max(src, 0).  Fully-valid and partially-valid (valid_shape
# 60x60 inside a 64x64 tile) f32/f16/i32 vec tiles in PTODSL auto mode; tile
# addresses, partitions and sync are left to PTOAS.  Every legacy case's
# shape / tile_shape / valid_shape and per-case crc32 RNG seed are preserved.

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


# (legacy case name, pto dtype, numpy dtype, shape, tile_shape, eps).
# shape is the global data dimension (also the logical/valid region); the two
# 60x60 cases allocate a 64x64 tile with valid_shape 60x60, mirroring the
# legacy .pto tile_buf<vec, 64x64xf16, valid=60x60>.
CASE_SPECS = [
    ("int32_64x64",           pto.i32, np.int32,   (64, 64), (64, 64), 1e-6),
    ("f16_64x64_valid_60x60", pto.f16, np.float16, (60, 60), (64, 64), 1e-3),
    ("f32_64x64_valid_60x60", pto.f32, np.float32, (60, 60), (64, 64), 1e-6),
]


def _trelu_body(src_ptr, dst_ptr, *, rows, cols, tile_rows, tile_cols, dtype):
    src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[rows, cols], strides=[cols, 1])

    if (rows, cols) == (tile_rows, tile_cols):
        src_tile = pto.alloc_tile(shape=[tile_rows, tile_cols], dtype=dtype)
        dst_tile = pto.alloc_tile(shape=[tile_rows, tile_cols], dtype=dtype)
    else:
        src_tile = pto.alloc_tile(
            shape=[tile_rows, tile_cols], dtype=dtype, valid_shape=[rows, cols]
        )
        dst_tile = pto.alloc_tile(
            shape=[tile_rows, tile_cols], dtype=dtype, valid_shape=[rows, cols]
        )

    pto.tile.load(src_view, src_tile)
    pto.tile.relu(src_tile, dst_tile)
    pto.tile.store(dst_tile, dst_view)


_trelu_kernels = {}
for _name, _dtype, _np_dtype, _shape, _tile_shape, _eps in CASE_SPECS:
    _r, _c = _shape
    _tr, _tc = _tile_shape

    def _make(r=_r, c=_c, tr=_tr, tc=_tc, dtype=_dtype, kernel_name=f"trelu_{_name}"):
        @pto.jit(name=kernel_name, target="a5")
        def _kernel(src_ptr: pto.ptr(dtype, "gm"), dst_ptr: pto.ptr(dtype, "gm")):
            _trelu_body(
                src_ptr, dst_ptr,
                rows=r, cols=c, tile_rows=tr, tile_cols=tc, dtype=dtype,
            )

        return _kernel

    _trelu_kernels[_name] = _make()


def _make_inputs(name, np_dtype, shape):
    # Mirrors st_common.setup_case_rng (per-case deterministic crc32 seed on
    # the legacy case name) and the legacy gen_data.py draw.
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    if np_dtype == np.int32:
        src = np.random.randint(-3_000_000, 3_000_000, size=shape).astype(np_dtype)
    else:
        src = np.random.uniform(-10, 10, size=shape).astype(np_dtype)
    return [src]


def _make_expected(src):
    return np.maximum(src, 0).astype(src.dtype)


CASES = []
for _name, _dtype, _np_dtype, _shape, _tile_shape, _eps in CASE_SPECS:
    CASES.append(
        golden_output_case(
            "trelu_" + _name,
            _trelu_kernels[_name],
            inputs=lambda _n=_name, _d=_np_dtype, _s=_shape: _make_inputs(_n, _d, _s),
            expected=_make_expected,
            rtol=_eps,
            atol=_eps,
        )
    )


auto_main(globals())
