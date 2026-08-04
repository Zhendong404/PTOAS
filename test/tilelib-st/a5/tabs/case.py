#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tabs.
#
# pto.tabs: dst = |src|.  Fully-valid f32/f16 vec tiles in PTODSL auto mode;
# tile addresses, partitions and sync are left to PTOAS.  Every legacy case is
# fully valid (valid_shape == shape), so each case allocates a full 2D vec
# tile matching the legacy .pto tload(tabs)->tstore sequence.

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

# (legacy case name, pto dtype, shape, eps).  Every legacy case is fully valid
# (valid_shape == shape).  The legacy case name seeds the RNG so the migrated
# data matches the original gen_data.py.
CASE_SPECS = [
    ("f32_16x64", pto.f32, (16, 64), 1e-6),
    ("f32_32x32", pto.f32, (32, 32), 1e-6),
    ("f16_16x64", pto.f16, (16, 64), 1e-3),
    ("f16_32x32", pto.f16, (32, 32), 1e-3),
]


def _tabs_body(src_ptr, dst_ptr, *, rows, cols, dtype):
    src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
    dst_view = pto.make_tensor_view(dst_ptr, shape=[rows, cols], strides=[cols, 1])

    src_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)
    dst_tile = pto.alloc_tile(shape=[rows, cols], dtype=dtype)

    pto.tile.load(src_view, src_tile)
    pto.tile.abs(src_tile, dst_tile)
    pto.tile.store(dst_tile, dst_view)


_tabs_kernels = {}
for _name, _dtype, _shape, _eps in CASE_SPECS:
    _r, _c = _shape

    def _make(r=_r, c=_c, dtype=_dtype, kernel_name=f"tabs_{_name}"):
        @pto.jit(name=kernel_name, target="a5")
        def _kernel(src_ptr: pto.ptr(dtype, "gm"), dst_ptr: pto.ptr(dtype, "gm")):
            _tabs_body(src_ptr, dst_ptr, rows=r, cols=c, dtype=dtype)

        return _kernel

    _tabs_kernels[_name] = _make()


def _make_inputs(name, dtype, shape):
    # Mirrors st_common.setup_case_rng (per-case deterministic crc32 seed) and
    # the legacy gen_data.py draw (np.random.randn on the legacy name seed).
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    src = np.random.randn(*shape).astype(PTO_TO_NP_DTYPE[dtype])
    return [src]


def _make_expected(src, dtype):
    np_dtype = PTO_TO_NP_DTYPE[dtype]
    return np.abs(src).astype(np_dtype)


CASES = []
for _name, _dtype, _shape, _eps in CASE_SPECS:
    CASES.append(
        golden_output_case(
            "tabs_" + _name,
            _tabs_kernels[_name],
            inputs=lambda _n=_name, _d=_dtype, _s=_shape: _make_inputs(_n, _d, _s),
            expected=lambda src, _d=_dtype: _make_expected(src, _d),
            rtol=_eps,
            atol=_eps,
        )
    )


auto_main(globals())
