#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tsub.
#
# pto.tsub: dst = a - b.  Fully-valid f32 tiles in PTODSL auto mode; tile
# addresses, partitions and sync are left to PTOAS.

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


# (legacy case name, pto dtype, shape, eps).  Every legacy case is fully
# valid (valid_shape == shape).  The legacy case name seeds the RNG so the
# migrated data matches the original gen_data.py.
CASE_SPECS = [
    ("f32_16x64", pto.f32, (16, 64), 1e-6),
    ("f32_32x32", pto.f32, (32, 32), 1e-6),
]


def _tsub_body(a_ptr, b_ptr, c_ptr, *, rows, cols):
    a_view = pto.make_tensor_view(a_ptr, shape=[rows, cols], strides=[cols, 1])
    b_view = pto.make_tensor_view(b_ptr, shape=[rows, cols], strides=[cols, 1])
    c_view = pto.make_tensor_view(c_ptr, shape=[rows, cols], strides=[cols, 1])

    a_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto.f32)
    b_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto.f32)
    c_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto.f32)

    pto.tile.load(a_view, a_tile)
    pto.tile.load(b_view, b_tile)
    pto.tile.sub(a_tile, b_tile, c_tile)
    pto.tile.store(c_tile, c_view)


_tsub_kernels = {}
for _name, _dtype, _shape, _eps in CASE_SPECS:
    _r, _c = _shape

    def _make(r=_r, c=_c, kernel_name=f"tsub_{_name}"):
        @pto.jit(name=kernel_name, target="a5")
        def _kernel(a_ptr: pto.ptr(pto.f32, "gm"), b_ptr: pto.ptr(pto.f32, "gm"), c_ptr: pto.ptr(pto.f32, "gm")):
            _tsub_body(a_ptr, b_ptr, c_ptr, rows=r, cols=c)

        return _kernel

    _tsub_kernels[_name] = _make()


def _make_inputs(name, shape):
    # Mirrors st_common.setup_case_rng: per-case deterministic crc32 seed.
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    a = np.random.randint(1, 10, size=shape).astype(np.float32)
    b = np.random.randint(1, 10, size=shape).astype(np.float32)
    return [a, b]


def _make_expected(a, b):
    return (a - b).astype(np.float32)


CASES = []
for _name, _dtype, _shape, _eps in CASE_SPECS:
    CASES.append(
        golden_output_case(
            "tsub_" + _name,
            _tsub_kernels[_name],
            inputs=lambda _n=_name, _s=_shape: _make_inputs(_n, _s),
            expected=_make_expected,
            rtol=_eps,
            atol=_eps,
        )
    )


auto_main(globals())
