#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of the legacy tinsert ACC->Vec cases.  The numerical endpoint
# is a public Vec tile store; the legacy mte_ub_gm micro-instruction is not
# used in this TileLib ST case.

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


M = K = N = 16
CASE_SPECS = [
    ("acc2vec_nd_f16_16x16", pto.f16, np.float16, "nd", 1e-2),
    ("acc2vec_nd_f32_16x16", pto.f32, np.float32, "nd", 1e-2),
    ("acc2vec_nz_f32_16x16", pto.f32, np.float32, "nz", 1e-2),
]


def _make_kernel(name, dst_dtype, kind):
    if kind == "nd":
        out_shape = [1, 1, 1, M, N]
        out_strides = [M * N, M * N, M * N, N, 1]
        b_layout, s_layout = "RowMajor", "NoneBox"
    else:
        out_shape = [2, 1, 16, 1, 8]
        out_strides = [128, 128, 8, 8, 1]
        b_layout, s_layout = "ColMajor", "RowMajor"

    # The legacy case is a mixed cube/vector kernel: ACC->VEC insert is a
    # cube-side operation, while the public VEC->GM store must be emitted in
    # the vector section.  Leaving the whole function as kernel_kind="cube"
    # makes Bisheng compile the VEC store with the cube backend.
    @pto.jit(name="tinsert_" + name, target="a5", mode="explicit", insert_sync=False)
    def _kernel(a_ptr: pto.ptr(pto.f16, "gm"), b_ptr: pto.ptr(pto.f16, "gm"), out_ptr: pto.ptr(dst_dtype, "gm")):
        a_mat = pto.alloc_tile(shape=[M, K], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, addr=0, valid_shape=[M, K], blayout="ColMajor", slayout="RowMajor")
        b_mat = pto.alloc_tile(shape=[K, N], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, addr=512, valid_shape=[K, N], blayout="ColMajor", slayout="RowMajor")
        left = pto.alloc_tile(shape=[M, K], dtype=pto.f16, memory_space=pto.MemorySpace.LEFT, addr=0, valid_shape=[M, K], blayout="ColMajor", slayout="RowMajor")
        right = pto.alloc_tile(shape=[K, N], dtype=pto.f16, memory_space=pto.MemorySpace.RIGHT, addr=0, valid_shape=[K, N], blayout="RowMajor", slayout="ColMajor")
        acc = pto.alloc_tile(shape=[M, N], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, addr=0, valid_shape=[M, N], blayout="ColMajor", slayout="RowMajor", fractal_size=1024)
        vec = pto.alloc_tile(shape=[M, N], dtype=dst_dtype, memory_space=pto.MemorySpace.VEC, addr=0, valid_shape=[M, N], blayout=b_layout, slayout=s_layout, fractal_size=512 if dst_dtype is pto.f16 else 1024)

        a_view = pto.make_tensor_view(a_ptr, shape=[1, 1, 1, M, K], strides=[256, 256, 256, K, 1])
        b_view = pto.make_tensor_view(b_ptr, shape=[1, 1, 1, K, N], strides=[256, 256, 256, N, 1])
        out_view = pto.make_tensor_view(out_ptr, shape=out_shape, strides=out_strides)
        with pto.section("cube"):
            pto.tile.load(a_view, a_mat, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, M, K])
            pto.tile.load(b_view, b_mat, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, K, N])
            pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
            pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
            pto.tile.mov(a_mat, left)
            pto.tile.mov(b_mat, right)
            pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
            pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
            pto.tile.matmul(left, right, acc)
            pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
            pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
            pto.tile.insert(acc, vec, 0, 0)
            # Cross-section synchronization is required by the mixed physical
            # placement; data movement itself remains on the TileOp surface.
            pto.set_cross_flag(pto.Pipe.FIX, 1)

        with pto.section("vector"):
            subblock = pto.get_subblock_idx()
            if subblock == 0:
                pto.wait_intra_flag(pto.Pipe.MTE3, 1)
                pto.tile.store(vec, out_view, offsets=[0] * 5, sizes=out_shape)
        pto.pipe_barrier(pto.Pipe.ALL)
    return _kernel


_kernels = {name: _make_kernel(name, dtype, kind) for name, dtype, _, kind, _ in CASE_SPECS}


def _inputs(name):
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    return [np.random.uniform(-1.0, 1.0, (M, K)).astype(np.float16), np.random.uniform(-1.0, 1.0, (K, N)).astype(np.float16)]


def _expected(a, b, kind, np_dtype):
    result = (a.astype(np.float32) @ b.astype(np.float32)).astype(np_dtype)
    if kind == "nz":
        return result.reshape(M, 2, 8).reshape(-1)
    return result


CASES = [
    golden_output_case(
        "tinsert_" + name,
        _kernels[name],
        inputs=lambda name=name: _inputs(name),
        expected=lambda a, b, kind=kind, np_dtype=np_dtype: _expected(a, b, kind, np_dtype),
        rtol=eps,
        atol=eps,
    )
    for name, _, np_dtype, kind, eps in CASE_SPECS
]


auto_main(globals())
