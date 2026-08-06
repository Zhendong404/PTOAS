#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL port of the legacy A5 MAT-to-BIAS TMOV case."""

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


M = K = N = 16


@pto.jit(
    name="tmov2bias_f16_16x16x16",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel(
    a_ptr: pto.ptr(pto.f16, "gm"),
    b_ptr: pto.ptr(pto.f16, "gm"),
    bias_ptr: pto.ptr(pto.f32, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    a_mat = pto.alloc_tile(
        shape=[M, K], dtype=pto.f16, memory_space=pto.MemorySpace.MAT,
        addr=0, valid_shape=[M, K], blayout="ColMajor", slayout="RowMajor",
    )
    b_mat = pto.alloc_tile(
        shape=[K, N], dtype=pto.f16, memory_space=pto.MemorySpace.MAT,
        addr=512, valid_shape=[K, N], blayout="ColMajor", slayout="RowMajor",
    )
    # Keep the source as a MAT tile so the operation under test is the
    # MAT->BIAS public tile.mov surface.  Its 1x16 shape is also the exact
    # legacy bias-table shape.
    bias_mat = pto.alloc_tile(
        shape=[1, N], dtype=pto.f32, memory_space=pto.MemorySpace.MAT,
        addr=1024, valid_shape=[1, N], blayout="ColMajor", slayout="RowMajor",
    )
    a_left = pto.alloc_tile(
        shape=[M, K], dtype=pto.f16, memory_space=pto.MemorySpace.LEFT,
        addr=0, valid_shape=[M, K], blayout="ColMajor", slayout="RowMajor",
    )
    b_right = pto.alloc_tile(
        shape=[K, N], dtype=pto.f16, memory_space=pto.MemorySpace.RIGHT,
        addr=0, valid_shape=[K, N], blayout="RowMajor", slayout="ColMajor",
    )
    bias_tile = pto.alloc_tile(
        shape=[1, N], dtype=pto.f32, memory_space=pto.MemorySpace.BIAS,
        addr=0, valid_shape=[1, N], blayout="RowMajor", slayout="NoneBox",
    )
    c_acc = pto.alloc_tile(
        shape=[M, N], dtype=pto.f32, memory_space=pto.MemorySpace.ACC,
        addr=0, valid_shape=[M, N], blayout="ColMajor", slayout="RowMajor",
        fractal_size=1024,
    )

    a_view = pto.make_tensor_view(a_ptr, shape=[1, 1, 1, M, K], strides=[M * K, M * K, M * K, K, 1])
    b_view = pto.make_tensor_view(b_ptr, shape=[1, 1, 1, K, N], strides=[K * N, K * N, K * N, N, 1])
    bias_view = pto.make_tensor_view(bias_ptr, shape=[1, 1, 1, 1, N], strides=[N, N, N, N, 1])
    c_view = pto.make_tensor_view(c_ptr, shape=[1, 1, 1, M, N], strides=[M * N, M * N, M * N, N, 1])

    pto.tile.load(a_view, a_mat, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, M, K])
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.tile.mov(a_mat, a_left)

    pto.tile.load(b_view, b_mat, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, K, N])
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
    pto.tile.mov(b_mat, b_right)

    pto.tile.load(bias_view, bias_mat, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, 1, N])
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=2)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=2)
    pto.tile.mov(bias_mat, bias_tile)

    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.mad_bias(a_left.as_ptr(), b_right.as_ptr(), c_acc.as_ptr(), bias_tile.as_ptr(), M, N, K, disable_gemv=True)
    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.tile.store(c_acc, c_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, M, N])
    pto.pipe_barrier(pto.Pipe.ALL)


def _inputs():
    np.random.seed(zlib.crc32(b"f16_16x16x16") & 0xFFFFFFFF)
    return [
        np.random.uniform(-1.0, 1.0, (M, K)).astype(np.float16),
        np.random.uniform(-1.0, 1.0, (K, N)).astype(np.float16),
        np.random.uniform(-0.5, 0.5, (1, N)).astype(np.float32),
    ]


def _expected(a, b, bias):
    return (a.astype(np.float32) @ b.astype(np.float32) + bias).astype(np.float32)


CASES = [golden_output_case(
    "tmov2bias_f16_16x16x16", _kernel, inputs=_inputs, expected=_expected,
    rtol=1e-3, atol=1e-3,
)]


auto_main(globals())
