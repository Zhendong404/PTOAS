#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/trowexpandexpdif.
#
# pto.trowexpandexpdif: dst = exp(src0 - broadcast(src1)) across columns.
# src1 is a per-row scalar kept in a physically aligned tile (32/sizeof(dtype)
# columns, e.g. 8 for f32); the TileLib template loads src1[row, 0] and vdup's
# the LOWEST lane, so only the first column of src1 takes part
# (dst = exp(src0 - src1[:, 0:1])), exactly matching the legacy golden.  src1
# is fully valid at its physical width (the legacy src1Col=1 / src1_valid_shape
# (rows, 1) was a launcher template parameter, not a tile valid_shape, mirroring
# the legacy .pto tile_buf<vec, rows x 32/sizeof(dtype)> with no valid tail).
# Fully valid src0/dst tiles in PTODSL auto mode; tile addresses, partitions
# and sync are left to PTOAS.

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


PTO_TO_NP_DTYPE = {
    pto.f32: np.float32,
    pto.f16: np.float16,
}

# (legacy case suffix, pto dtype, src0/dst shape (rows, cols),
#  src1 physical shape (rows, 32/sizeof(dtype)), eps).
# Every legacy case is fully valid (valid_shape == shape == dst_shape); the
# legacy src0eqdst=false f32_16x128_noeq case maps to the same PTODSL op (the
# API carries no src0eqdst parameter).
CASE_SPECS = [
    ("f32_32x64",       pto.f32, (32, 64),   (32, 8),  1e-5),
    ("f32_16x32",       pto.f32, (16, 32),   (16, 8),  1e-5),
    ("f16_16x32",       pto.f16, (16, 32),   (16, 16), 1e-3),
    ("f16_48x64",       pto.f16, (48, 64),   (48, 16), 1e-3),
    ("f32_16x128_noeq", pto.f32, (16, 128),  (16, 8),  1e-5),
]


def _trowexpandexpdif_body(src0_ptr, src1_ptr, dst_ptr, *, rows, cols, src1_cols, dtype):
    src0_view = pto.make_tensor_view(src0_ptr, shape=[rows, cols], strides=[cols, 1])
    src1_view = pto.make_tensor_view(src1_ptr, shape=[rows, src1_cols], strides=[src1_cols, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[rows, cols], strides=[cols, 1])

    src0_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)
    src1_tile = pto.alloc_tile(shape=[rows, src1_cols], dtype=dtype)
    dst_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)

    pto.tile.load(src0_view, src0_tile)
    pto.tile.load(src1_view, src1_tile)
    pto.tile.rowexpandexpdif(src0_tile, src1_tile, dst_tile)
    pto.tile.store(dst_tile, dst_view)


# One decorated kernel per case, each binding a static shape at definition time
# (mirroring the per-case funcs in trowexpandexpdif.pto).
_trowexpandexpdif_kernels = {}
for _name, _dtype, _shape, _src1_shape, _eps in CASE_SPECS:
    _r, _c = _shape
    _src1_c = _src1_shape[1]

    def _make(r=_r, c=_c, src1_c=_src1_c, dtype=_dtype, kernel_name=f"trowexpandexpdif_{_name}"):
        @pto.jit(name=kernel_name, target="a5")
        def _kernel(
            src0_ptr: pto.ptr(dtype, "gm"),
            src1_ptr: pto.ptr(dtype, "gm"),
            dst_ptr: pto.ptr(dtype, "gm"),
        ):
            _trowexpandexpdif_body(
                src0_ptr, src1_ptr, dst_ptr, rows=r, cols=c, src1_cols=src1_c, dtype=dtype,
            )

        return _kernel

    _trowexpandexpdif_kernels[_name] = _make()


def _make_inputs(name, np_dtype, dst_shape, src1_shape):
    # Mirrors st_common.setup_case_rng (crc32 seed on the legacy case name) and
    # gen_data.py: input1/src0 and input2/src1 both drawn from randint(1, 5),
    # src1 with its physical (aligned) width.
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    src0 = np.random.randint(1, 5, size=dst_shape).astype(np_dtype)
    src1 = np.random.randint(1, 5, size=src1_shape).astype(np_dtype)
    return [src0, src1]


def _make_expected(src0, src1):
    # Mirrors gen_data.py: dst = exp(src0 - broadcast(src1[:, 0])), computed in
    # the input dtype (src1Col=1; only the first column of src1 is used).
    diff = src0 - src1[:, 0:1]
    return np.exp(diff).astype(src0.dtype)


CASES = []
for _name, _dtype, _shape, _src1_shape, _eps in CASE_SPECS:
    CASES.append(
        golden_output_case(
            "trowexpandexpdif_" + _name,
            _trowexpandexpdif_kernels[_name],
            inputs=lambda _n=_name, _d=PTO_TO_NP_DTYPE[_dtype], _s=_shape, _s1=_src1_shape: _make_inputs(
                _n, _d, _s, _s1
            ),
            expected=lambda src0, src1: _make_expected(src0, src1),
            rtol=_eps,
            atol=_eps,
        )
    )


auto_main(globals())
