#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Minimal PTODSL cube/tmatmul pilot for A5.
# Goal: validate plain cube tile.matmul lowering/runtime first, without mixing
# MX-specific scale/bias handling or reusable helper boundaries.

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


M = 16
K = 32
N = 64
ELEM_BYTES = 4

L1_A_ADDR = 0
L1_B_ADDR = 4096
L0A_ADDR = 0
L0B_ADDR = 0
L0C_ADDR = 0

# This case keeps explicit L1/L0 addresses because the cube staging templates
# consume explicit MAT/LEFT/RIGHT tile buffers. GM↔MAT and ACC→GM boundaries
# use the public tile.load/store surface below.


@pto.jit(
    name="tmatmul_f32_16x32x64",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _tmatmul_kernel(
    a_ptr: pto.ptr(pto.f32, "gm"),
    b_ptr: pto.ptr(pto.f32, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    a_mat = pto.alloc_tile(
        shape=[M, K],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.MAT,
        addr=L1_A_ADDR,
        valid_shape=[M, K],
        blayout="ColMajor",
        slayout="RowMajor",
    )
    b_mat = pto.alloc_tile(
        shape=[K, N],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.MAT,
        addr=L1_B_ADDR,
        valid_shape=[K, N],
        blayout="ColMajor",
        slayout="RowMajor",
    )
    a_l0a = pto.alloc_tile(
        shape=[M, K],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.LEFT,
        addr=L0A_ADDR,
        valid_shape=[M, K],
        blayout="ColMajor",
        slayout="RowMajor",
    )
    b_l0b = pto.alloc_tile(
        shape=[K, N],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.RIGHT,
        addr=L0B_ADDR,
        valid_shape=[K, N],
        blayout="RowMajor",
        slayout="ColMajor",
    )
    c_acc = pto.alloc_tile(
        shape=[M, N],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.ACC,
        addr=L0C_ADDR,
        valid_shape=[M, N],
        blayout="ColMajor",
        slayout="RowMajor",
        fractal_size=1024,
    )

    a_view = pto.make_tensor_view(
        a_ptr, shape=[1, 1, 1, M, K], strides=[M * K, M * K, M * K, K, 1]
    )
    b_view = pto.make_tensor_view(
        b_ptr, shape=[1, 1, 1, K, N], strides=[K * N, K * N, K * N, N, 1]
    )
    c_view = pto.make_tensor_view(
        c_ptr, shape=[1, 1, 1, M, N], strides=[M * N, M * N, M * N, N, 1]
    )
    a_shape = [1, 1, 1, M, K]
    b_shape = [1, 1, 1, K, N]
    c_shape = [1, 1, 1, M, N]

    pto.tile.load(a_view, a_mat, offsets=[0, 0, 0, 0, 0], sizes=a_shape)
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.mte_l1_l0a(a_mat.as_ptr(), a_l0a.as_ptr(), M, K)

    pto.tile.load(b_view, b_mat, offsets=[0, 0, 0, 0, 0], sizes=b_shape)
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
    pto.mte_l1_l0b(b_mat.as_ptr(), b_l0b.as_ptr(), K, N, transpose=True)

    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.tile.matmul(a_l0a, b_l0b, c_acc)

    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.tile.store(c_acc, c_view, offsets=[0, 0, 0, 0, 0], sizes=c_shape)
    pto.pipe_barrier(pto.Pipe.ALL)


def _make_inputs():
    rng = np.random.default_rng(0x7A7A7A71)
    a = rng.uniform(-2.0, 2.0, size=(M, K)).astype(np.float32)
    b = rng.uniform(-2.0, 2.0, size=(K, N)).astype(np.float32)
    return [a, b]


def _make_expected(a, b):
    return (a @ b).astype(np.float32)


CASES = [
    golden_output_case(
        "tmatmul_f32_16x32x64",
        _tmatmul_kernel,
        inputs=_make_inputs,
        expected=_make_expected,
        rtol=1e-4,
        atol=1e-4,
    ),
]


auto_main(globals())
