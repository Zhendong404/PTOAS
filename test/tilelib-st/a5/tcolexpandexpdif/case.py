#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tcolexpandexpdif.
#
# pto.tcolexpandexpdif: dst = exp(src0 - expand_cols(src1)), where src1 is a
# 1xcols tile broadcast down to src0 rows (f-only: f32/f16).  Fully-valid tiles
# in PTODSL auto mode; tile addresses, partitions and sync are left to PTOAS.
#
# NOTE on legacy semantics: the legacy .pto comment ("exp(src0) - exp(tiled
# src1)") is misleading; gen_data.py's golden is np.exp(src0 - expanded_src1),
# which matches the PTODSL API doc for pto.tile.colexpandexpdif, so the golden
# below follows gen_data.py exactly (f64 compute, then cast to dtype).

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

# (legacy case suffix, pto dtype, src0/dst shape (rows, cols), src1 shape,
#  eps).  Every legacy case is fully valid (valid_shape == shape == dst_shape).
# The legacy case name seeds the RNG so the migrated data matches gen_data.py.
CASE_SPECS = [
    ("fp32_32_16_1_16",    pto.f32, (32, 16),  (1, 16),  1e-5),
    ("fp32_16_32_1_32",    pto.f32, (16, 32),  (1, 32),  1e-5),
    ("fp16_32_32_1_32",    pto.f16, (32, 32),  (1, 32),  1e-2),
    ("fp16_16_128_1_128",  pto.f16, (16, 128), (1, 128), 1e-2),
]


def _tcolexpandexpdif_body(src0_ptr, src1_ptr, dst_ptr, *, rows, cols, dtype):
    src0_view = pto.make_tensor_view(src0_ptr, shape=[rows, cols], strides=[cols, 1])
    src1_view = pto.make_tensor_view(src1_ptr, shape=[1, cols], strides=[cols, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[rows, cols], strides=[cols, 1])

    src0_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)
    src1_tile = pto.alloc_tile(shape=[1, cols], dtype=dtype)
    dst_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)

    pto.tile.load(src0_view, src0_tile)
    pto.tile.load(src1_view, src1_tile)
    pto.tile.colexpandexpdif(src0_tile, src1_tile, dst_tile)
    pto.tile.store(dst_tile, dst_view)


_tcolexpandexpdif_kernels = {}
for _name, _dtype, _shape, _src1_shape, _eps in CASE_SPECS:
    _r, _c = _shape

    def _make(r=_r, c=_c, dtype=_dtype, kernel_name=f"tcolexpandexpdif_{_name}"):
        @pto.jit(name=kernel_name, target="a5")
        def _kernel(
            src0_ptr: pto.ptr(dtype, "gm"),
            src1_ptr: pto.ptr(dtype, "gm"),
            dst_ptr: pto.ptr(dtype, "gm"),
        ):
            _tcolexpandexpdif_body(src0_ptr, src1_ptr, dst_ptr, rows=r, cols=c, dtype=dtype)

        return _kernel

    _tcolexpandexpdif_kernels[_name] = _make()


def _make_inputs(name, dtype, shape, src1_shape):
    # Mirrors st_common.setup_case_rng: per-case deterministic crc32 seed,
    # then gen_data.py's uniform draw order (src0 then src1).
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    np_dtype = PTO_TO_NP_DTYPE[dtype]
    src0 = np.random.uniform(-255, 255, size=shape).astype(np_dtype)
    src1 = np.random.uniform(1, 255, size=src1_shape).astype(np_dtype)
    return [src0, src1]


def _make_expected(src0, src1, *, rows, cols):
    # Mirrors gen_data.py: broadcast src1 down to dst rows, subtract, exp in
    # f64 to reduce precision loss, then cast back to the input dtype.
    dtype = src0.dtype
    src1_row = src1.shape[0]
    reps = (rows + src1_row - 1) // src1_row
    expanded_src1 = np.tile(src1, (reps, 1))[:rows, :cols]
    golden = np.exp(src0.astype(np.float64) - expanded_src1.astype(np.float64))
    return golden.astype(dtype)


CASES = []
for _name, _dtype, _shape, _src1_shape, _eps in CASE_SPECS:
    _r, _c = _shape
    CASES.append(
        golden_output_case(
            "tcolexpandexpdif_" + _name,
            _tcolexpandexpdif_kernels[_name],
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
