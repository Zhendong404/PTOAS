#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tcolexpanddiv.
#
# pto.tcolexpanddiv: dst[i,j] = src0[i,j] / src1[0,j], where src1 is a 1xcols
# tile broadcast down to src0 rows.  Fully-valid f32/f16/i32/i16 tiles in PTODSL
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

# (legacy case suffix, pto dtype, src0/dst shape (rows, cols), src1 shape,
#  eps).  Every legacy case is fully valid (valid_shape == shape == dst_shape).
# The legacy case name seeds the RNG so the migrated data matches gen_data.py.
CASE_SPECS = [
    ("fp32_32_64_1_64",   pto.f32, (32, 64),  (1, 64),  1e-6),
    ("fp32_8_32_1_32",    pto.f32, (8, 32),   (1, 32),  1e-6),
    ("fp16_16_64_1_64",   pto.f16, (16, 64),  (1, 64),  1e-3),
    ("fp16_4_128_1_128",  pto.f16, (4, 128),  (1, 128), 1e-3),
    ("int32_16_32_1_32",  pto.i32, (16, 32),  (1, 32),  0),
    ("int16_16_64_1_64",  pto.i16, (16, 64),  (1, 64),  0),
    ("fp32_40_32_1_32",   pto.f32, (40, 32),  (1, 32),  1e-6),
    ("fp16_16_128_1_128", pto.f16, (16, 128), (1, 128), 1e-3),
    ("fp32_20_64_1_64",   pto.f32, (20, 64),  (1, 64),  1e-6),
]


def _tcolexpanddiv_body(src0_ptr, src1_ptr, dst_ptr, *, rows, cols, dtype):
    src0_view = pto.make_tensor_view(src0_ptr, shape=[rows, cols], strides=[cols, 1])
    src1_view = pto.make_tensor_view(src1_ptr, shape=[1, cols], strides=[cols, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[rows, cols], strides=[cols, 1])

    src0_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)
    src1_tile = pto.alloc_tile(shape=[1, cols], dtype=dtype)
    dst_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)

    pto.tile.load(src0_view, src0_tile)
    pto.tile.load(src1_view, src1_tile)
    pto.tile.colexpanddiv(src0_tile, src1_tile, dst_tile)
    pto.tile.store(dst_tile, dst_view)


_tcolexpanddiv_kernels = {}
for _name, _dtype, _shape, _src1_shape, _eps in CASE_SPECS:
    _r, _c = _shape

    def _make(r=_r, c=_c, dtype=_dtype, kernel_name=f"tcolexpanddiv_{_name}"):
        @pto.jit(name=kernel_name, target="a5")
        def _kernel(
            src0_ptr: pto.ptr(dtype, "gm"),
            src1_ptr: pto.ptr(dtype, "gm"),
            dst_ptr: pto.ptr(dtype, "gm"),
        ):
            _tcolexpanddiv_body(src0_ptr, src1_ptr, dst_ptr, rows=r, cols=c, dtype=dtype)

        return _kernel

    _tcolexpanddiv_kernels[_name] = _make()


def _make_inputs(name, dtype, shape, src1_shape):
    # Mirrors st_common.setup_case_rng: per-case deterministic crc32 seed,
    # then gen_data.py's uniform(1.0, 10.0) draw order (src0 then src1).
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    np_dtype = PTO_TO_NP_DTYPE[dtype]
    src0 = np.random.uniform(1.0, 10.0, size=shape).astype(np_dtype)
    src1 = np.random.uniform(1.0, 10.0, size=src1_shape).astype(np_dtype)
    return [src0, src1]


def _make_expected(src0, src1, *, rows, cols):
    # Mirrors gen_data.py: broadcast src1 down to dst rows, divide, then store
    # into a dtype buffer (int dtypes truncate, matching the legacy golden).
    dtype = src0.dtype
    golden = np.zeros((rows, cols), dtype=dtype)
    reps = rows // src1.shape[0]
    expanded_src1 = np.tile(src1, (reps, 1))[:, :cols]
    golden[:rows, :cols] = src0[:rows, :cols] / expanded_src1
    return golden


CASES = []
for _name, _dtype, _shape, _src1_shape, _eps in CASE_SPECS:
    _r, _c = _shape
    CASES.append(
        golden_output_case(
            "tcolexpanddiv_" + _name,
            _tcolexpanddiv_kernels[_name],
            inputs=lambda _n=_name, _d=_dtype, _s=_shape, _s1=_src1_shape: _make_inputs(
                _n, _d, _s, _s1
            ),
            expected=lambda src0, src1, _r=_r, _c=_c: _make_expected(
                src0, src1, rows=_r, cols=_c
            ),
            rtol=_eps,
            atol=_eps,
        )
    )


auto_main(globals())
