#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tsubs.
#
# pto.tsubs: dst = src - scalar.  Fully-valid f32/f16/i32/i16 tiles in PTODSL
# auto mode; tile addresses, partitions and sync are left to PTOAS.

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
    pto.i32: np.int32,
    pto.i16: np.int16,
}

# (legacy case name, pto dtype, shape, eps).  Every legacy case is fully
# valid (valid_shape == shape).  The legacy case name seeds the RNG so the
# migrated data matches the original gen_data.py.
CASE_SPECS = [
    ("f32_32x64", pto.f32, (32, 64), 1e-6),
    ("f16_63x64", pto.f16, (63, 64), 1e-3),
    ("i32_31x128", pto.i32, (31, 128), 0),
    ("i16_15x192", pto.i16, (15, 192), 0),
    ("f32_7x448", pto.f32, (7, 448), 1e-6),
    ("f32_256x16", pto.f32, (256, 16), 1e-6),
]

# Scalar subtracted from every element.  Matches gen_data.py SCALAR and
# launch.cpp (the f16 launch scalar 0x4200 is exactly 3.0 in half precision).
# Integer tiles receive the integer literal 3 so the constant materializes
# exactly.
SCALAR = 3.0


def _tsubs_body(src_ptr, dst_ptr, *, rows, cols, dtype, scalar):
    src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[rows, cols], strides=[cols, 1])

    src_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)
    dst_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)

    pto.tile.load(src_view, src_tile)
    pto.tile.subs(src_tile, scalar, dst_tile)
    pto.tile.store(dst_tile, dst_view)


_tsubs_kernels = {}
for _name, _dtype, _shape, _eps in CASE_SPECS:
    _r, _c = _shape
    _scalar = 3 if _dtype in (pto.i32, pto.i16) else SCALAR

    def _make(r=_r, c=_c, dtype=_dtype, scalar=_scalar, kernel_name=f"tsubs_{_name}"):
        @pto.jit(name=kernel_name, target="a5")
        def _kernel(src_ptr: pto.ptr(dtype, "gm"), dst_ptr: pto.ptr(dtype, "gm")):
            _tsubs_body(src_ptr, dst_ptr, rows=r, cols=c, dtype=dtype, scalar=scalar)

        return _kernel

    _tsubs_kernels[_name] = _make()


def _make_inputs(name, dtype, shape):
    # Mirrors st_common.setup_case_rng: per-case deterministic crc32 seed.
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    src = np.random.randint(1, 10, size=shape).astype(PTO_TO_NP_DTYPE[dtype])
    return [src]


def _make_expected(src, dtype):
    np_dtype = PTO_TO_NP_DTYPE[dtype]
    return (src - np_dtype(SCALAR)).astype(np_dtype)


CASES = []
for _name, _dtype, _shape, _eps in CASE_SPECS:
    CASES.append(
        golden_output_case(
            "tsubs_" + _name,
            _tsubs_kernels[_name],
            inputs=lambda _n=_name, _d=_dtype, _s=_shape: _make_inputs(_n, _d, _s),
            expected=lambda src, _d=_dtype: _make_expected(src, _d),
            rtol=_eps,
            atol=_eps,
        )
    )


auto_main(globals())
