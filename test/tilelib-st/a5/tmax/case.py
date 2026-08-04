#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tmax.
#
# pto.tmax: dst = max(a, b) element-wise over vec tiles, mirroring the legacy
# tload(a) + tload(b) + tmax(a, b) -> c + tstore(c) kernel.  Both legacy cases
# are fully valid (valid_shape == shape) f32 tiles; PTODSL auto mode leaves
# tile addresses, load/store partitions and sync to PTOAS.  Legacy dtype /
# shape / valid_shape / eps and the per-case crc32 RNG seed with the
# randint(1, 10) draw order are preserved.

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


# (legacy case name, pto dtype, numpy dtype, shape, valid_shape, eps).
# shape is the allocated/padded tile dimension and valid_shape the effective
# computation region; every tmax case is fully valid.
CASE_SPECS = [
    ("f32_16x64", pto.f32, np.float32, (16, 64), (16, 64), 1e-6),
    ("f32_32x32", pto.f32, np.float32, (32, 32), (32, 32), 1e-6),
]


def _tmax_body(a_ptr, b_ptr, c_ptr, *, rows, cols, valid_rows, valid_cols, dtype):
    """Shared kernel body: tload(a) + tload(b) + tmax(a, b) -> c + tstore(c)."""

    # The view spans the valid region using the full buffer's row stride, so a
    # padded 64-column buffer exposes its top-left 60x60 block as a 60x60 view.
    a_view = pto.make_tensor_view(a_ptr, shape=[valid_rows, valid_cols], strides=[cols, 1])
    b_view = pto.make_tensor_view(b_ptr, shape=[valid_rows, valid_cols], strides=[cols, 1])
    c_view = pto.make_tensor_view(c_ptr, shape=[valid_rows, valid_cols], strides=[cols, 1])

    if (rows, cols) == (valid_rows, valid_cols):
        a_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)
        b_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)
        c_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)
    else:
        a_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=dtype, valid_shape=[valid_rows, valid_cols]
        )
        b_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=dtype, valid_shape=[valid_rows, valid_cols]
        )
        c_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=dtype, valid_shape=[valid_rows, valid_cols]
        )

    pto.tile.load(a_view, a_tile)
    pto.tile.load(b_view, b_tile)
    pto.tile.max(a_tile, b_tile, c_tile)
    pto.tile.store(c_tile, c_view)


# One decorated kernel per case, each binding a static shape at definition time
# (mirroring the per-case funcs in tmax.pto).
_tmax_kernels = {}
for _name, _dtype, _np_dtype, _shape, _valid_shape, _eps in CASE_SPECS:
    _r, _c = _shape
    _vr, _vc = _valid_shape

    def _make(r=_r, c=_c, vr=_vr, vc=_vc, dtype=_dtype, kernel_name=f"tmax_{_name}"):
        @pto.jit(name=kernel_name, target="a5")
        def _kernel(
            a_ptr: pto.ptr(dtype, "gm"),
            b_ptr: pto.ptr(dtype, "gm"),
            c_ptr: pto.ptr(dtype, "gm"),
        ):
            _tmax_body(
                a_ptr, b_ptr, c_ptr,
                rows=r, cols=c, valid_rows=vr, valid_cols=vc, dtype=dtype,
            )

        return _kernel

    _tmax_kernels[_name] = _make()


def _make_inputs(name, np_dtype, shape):
    # Mirrors st_common.setup_case_rng (per-case deterministic crc32 seed on
    # the legacy case name) and the legacy gen_data.py draw order: input1
    # first, then input2, both randint(1, 10) over the full shape.
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    input1 = np.random.randint(1, 10, size=shape).astype(np_dtype)
    input2 = np.random.randint(1, 10, size=shape).astype(np_dtype)
    return [input1, input2]


def _make_expected(input1, input2, shape, valid_shape):
    # Legacy golden: zero-initialized full shape with only the valid region
    # computed; the kernel writes only the valid region into the zeroed output
    # buffer, so the full-array comparison stays valid.
    golden = np.zeros(shape, dtype=input1.dtype)
    vr, vc = valid_shape
    golden[:vr, :vc] = np.maximum(input1[:vr, :vc], input2[:vr, :vc]).astype(
        input1.dtype, copy=False
    )
    return golden


CASES = [
    golden_output_case(
        "tmax_" + name,
        _tmax_kernels[name],
        inputs=lambda name=name, np_dtype=np_dtype, shape=shape: _make_inputs(name, np_dtype, shape),
        expected=lambda input1, input2, shape=shape, valid_shape=valid_shape: _make_expected(
            input1, input2, shape, valid_shape
        ),
        rtol=eps,
        atol=eps,
    )
    for name, _, np_dtype, shape, valid_shape, eps in CASE_SPECS
]


auto_main(globals())
