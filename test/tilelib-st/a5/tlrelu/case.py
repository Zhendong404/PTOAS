#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tlrelu.
#
# pto.tlrelu: dst = src if src > 0 else src * slope.  Vec tiles in PTODSL auto
# mode; tile addresses, partitions and sync are left to PTOAS.  The dst GM view
# keeps the legacy layout: view shape = valid_shape (rows x cols) while the
# physical dst buffer is the padded dst_shape (rows x dst_cols), so only the
# valid region is written and the rest of the golden stays zero.  The slope is
# the per-case deterministic float32 scalar from gen_data.py, passed to the
# kernel as a runtime scalar argument (same pattern as a5/tci).

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


# (legacy case name, pto dtype, numpy dtype, shape, dst_shape, eps).
# shape is the src/valid region (rows x cols); dst_shape is the padded
# physical dst buffer (rows x dst_cols) that the valid region is stored into.
CASE_SPECS = [
    ("f32_32x64_dst128",   pto.f32, np.float32, (32, 64),   (32, 128), 1e-3),
    ("f16_63x64_dst128",   pto.f16, np.float16, (63, 64),   (63, 128), 1e-3),
    ("f32_7x448_dst512",   pto.f32, np.float32, (7, 448),   (7, 512),  1e-3),
    ("f32_256x16_dst32",   pto.f32, np.float32, (256, 16),  (256, 32), 1e-3),
]


def _tlrelu_body(src_ptr, dst_ptr, slope, *, rows, cols, dst_cols, dtype):
    src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[rows, cols], strides=[dst_cols, 1])

    src_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)
    dst_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)

    pto.tile.load(src_view, src_tile)
    pto.tile.lrelu(src_tile, slope, dst_tile)
    pto.tile.store(dst_tile, dst_view)


_tlrelu_kernels = {}
for _name, _dtype, _np_dtype, _shape, _dst_shape, _eps in CASE_SPECS:
    _r, _c = _shape
    _dr, _dc = _dst_shape

    def _make(r=_r, c=_c, dc=_dc, dtype=_dtype, kernel_name=f"tlrelu_{_name}"):
        @pto.jit(name=kernel_name, target="a5")
        def _kernel(
            src_ptr: pto.ptr(dtype, "gm"),
            dst_ptr: pto.ptr(dtype, "gm"),
            slope: pto.f32,
        ):
            _tlrelu_body(
                src_ptr, dst_ptr, slope,
                rows=r, cols=c, dst_cols=dc, dtype=dtype,
            )

        return _kernel

    _tlrelu_kernels[_name] = _make()


def _make_inputs(name, np_dtype, shape):
    # Mirrors st_common.setup_case_rng (per-case deterministic crc32 seed on
    # the legacy case name) and the legacy gen_data.py draw order: the input
    # array first, then the 1x1 slope.  The slope comes back as a 0-d numpy
    # scalar, matching the a5/tci scalar-input convention.
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    src = np.random.uniform(low=-8, high=8, size=shape).astype(np_dtype)
    slope = np.random.uniform(low=-8, high=8, size=(1, 1)).astype(np.float32)
    return [src, slope[0, 0]]


def _make_expected(src, slope, np_dtype, dst_shape):
    vr, vc = src.shape
    slope_val = slope.item() if hasattr(slope, "item") else float(slope)
    golden = np.zeros(dst_shape, dtype=np_dtype)
    leaky = (src * slope_val).astype(np_dtype)
    golden[:vr, :vc] = np.where(src > 0, src, leaky)
    return golden


CASES = []
for _name, _dtype, _np_dtype, _shape, _dst_shape, _eps in CASE_SPECS:
    CASES.append(
        golden_output_case(
            "tlrelu_" + _name,
            _tlrelu_kernels[_name],
            inputs=lambda _n=_name, _d=_np_dtype, _s=_shape: _make_inputs(_n, _d, _s),
            expected=lambda src, slope, _d=_np_dtype, _ds=_dst_shape: _make_expected(
                src, slope, _d, _ds
            ),
            rtol=_eps,
            atol=_eps,
        )
    )


auto_main(globals())
