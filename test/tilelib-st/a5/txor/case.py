#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/txor.
#   tload(a) + tload(b) + txor(a, b) -> c + tstore(c)

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


# (case suffix, rows, cols)
CASE_SHAPES = [
    ("i32_16x64", 16, 64),
    ("i32_32x32", 32, 32),
]


def _make_kernel(name: str, rows: int, cols: int):
    @pto.jit(name="txor_" + name, target="a5")
    def _kernel(
        a_ptr: pto.ptr(pto.i32, "gm"),
        b_ptr: pto.ptr(pto.i32, "gm"),
        c_ptr: pto.ptr(pto.i32, "gm"),
    ):
        a_view = pto.make_tensor_view(a_ptr, shape=[rows, cols], strides=[cols, 1])
        b_view = pto.make_tensor_view(b_ptr, shape=[rows, cols], strides=[cols, 1])
        c_view = pto.make_tensor_view(c_ptr, shape=[rows, cols], strides=[cols, 1])

        a_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto.i32)
        b_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto.i32)
        c_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto.i32)

        pto.tile.load(a_view, a_tile)
        pto.tile.load(b_view, b_tile)
        # txor needs a scratch tmp; mirror the legacy .pto which reused dst as tmp.
        pto.tile.bit_xor(a_tile, b_tile, c_tile, c_tile)
        pto.tile.store(c_tile, c_view)

    return _kernel


_kernels = {name: _make_kernel(name, rows, cols)
            for name, rows, cols in CASE_SHAPES}


def _make_inputs(name: str, rows: int, cols: int):
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    a = np.random.randint(0, 100, size=(rows, cols)).astype(np.int32)
    b = np.random.randint(0, 100, size=(rows, cols)).astype(np.int32)
    return [a, b]


def _make_expected(a, b):
    return (a ^ b).astype(np.int32)


CASES = [
    golden_output_case(
        "txor_" + name,
        _kernels[name],
        inputs=lambda name=name, rows=rows, cols=cols: _make_inputs(name, rows, cols),
        expected=_make_expected,
        rtol=0,
        atol=0,
    )
    for name, rows, cols in CASE_SHAPES
]


auto_main(globals())
