#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL port of the legacy A5 MAT-to-RIGHT TMOV case.

GM->MAT operand staging and the ACC->GM numerical endpoint use the public
``pto.tile.load`` / ``pto.tile.store`` surface (ND2NZ MAT loads and the
nz2nd ACC store, mirroring tload_mat).  The MAT->RIGHT ``pto.tile.mov``
remains the behavior under test.
"""

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto

M = K = N = 16
L1_A = 0
L1_B = 512


@pto.jit(name="tmov2right_f16_16x16x16", kernel_kind="cube", target="a5", mode="explicit", insert_sync=False)
def _kernel(a_ptr: pto.ptr(pto.f16, "gm"), b_ptr: pto.ptr(pto.f16, "gm"), c_ptr: pto.ptr(pto.f32, "gm")):
    a_mat = pto.alloc_tile(shape=[M, K], dtype=pto.f16, memory_space=pto.MemorySpace.MAT,
                           addr=L1_A, valid_shape=[M, K], blayout="ColMajor", slayout="RowMajor")
    b_mat = pto.alloc_tile(shape=[K, N], dtype=pto.f16, memory_space=pto.MemorySpace.MAT,
                           addr=L1_B, valid_shape=[K, N], blayout="ColMajor", slayout="RowMajor")
    a_left = pto.alloc_tile(shape=[M, K], dtype=pto.f16, memory_space=pto.MemorySpace.LEFT,
                            addr=0, valid_shape=[M, K], blayout="ColMajor", slayout="RowMajor")
    b_right = pto.alloc_tile(shape=[K, N], dtype=pto.f16, memory_space=pto.MemorySpace.RIGHT,
                             addr=0, valid_shape=[K, N], blayout="RowMajor", slayout="ColMajor")
    c_acc = pto.alloc_tile(shape=[M, N], dtype=pto.f32, memory_space=pto.MemorySpace.ACC,
                           addr=0, valid_shape=[M, N], blayout="ColMajor", slayout="RowMajor", fractal_size=1024)

    a_view = pto.make_tensor_view(a_ptr, shape=[1, 1, 1, M, K], strides=[M * K, M * K, M * K, K, 1])
    b_view = pto.make_tensor_view(b_ptr, shape=[1, 1, 1, K, N], strides=[K * N, K * N, K * N, N, 1])
    c_view = pto.make_tensor_view(c_ptr, shape=[1, 1, 1, M, N], strides=[M * N, M * N, M * N, N, 1])
    a_shape = [1, 1, 1, M, K]
    b_shape = [1, 1, 1, K, N]
    c_shape = [1, 1, 1, M, N]

    pto.tile.load(a_view, a_mat, offsets=[0, 0, 0, 0, 0], sizes=a_shape)
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.tile.mov(a_mat, a_left)

    pto.tile.load(b_view, b_mat, offsets=[0, 0, 0, 0, 0], sizes=b_shape)
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
    pto.tile.mov(b_mat, b_right)

    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.tile.matmul(a_left, b_right, c_acc)
    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.tile.store(c_acc, c_view, offsets=[0, 0, 0, 0, 0], sizes=c_shape)
    pto.pipe_barrier(pto.Pipe.ALL)


def _inputs():
    np.random.seed(zlib.crc32(b"f16_16x16x16") & 0xFFFFFFFF)
    return [np.random.uniform(-1.0, 1.0, (M, K)).astype(np.float16),
            np.random.uniform(-1.0, 1.0, (K, N)).astype(np.float16)]


def _expected(a, b):
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float32)


CASES = [golden_output_case("tmov2right_f16_16x16x16", _kernel, inputs=_inputs,
                            expected=_expected, rtol=1e-2, atol=1e-2)]

auto_main(globals())
