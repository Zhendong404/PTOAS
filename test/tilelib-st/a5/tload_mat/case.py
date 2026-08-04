#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL port of the legacy A5 TLOAD.MAT suite.

The public ``pto.tile.load`` surface selects the existing MAT load templates
from the rank-5 view metadata: ND views select ND2NZ and DN views select
DN2NZ.  The loaded MAT tiles are multiplied and written back through the
existing ACC ``pto.tile.store`` path.
"""

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


M, N, K = 16, 32, 16

CASE_SPECS = [
    ("f16_nd2nz", pto.f16, np.float16, "nd2nz"),
    ("bf16_nd2nz", pto.bf16, np.uint16, "nd2nz"),
    ("f32_nd2nz", pto.f32, np.float32, "nd2nz"),
    ("f16_dn2nz", pto.f16, np.float16, "dn2nz"),
    ("bf16_dn2nz", pto.bf16, np.uint16, "dn2nz"),
    ("f32_dn2nz", pto.f32, np.float32, "dn2nz"),
]


def _view_metadata(layout, rows, cols):
    if layout == "nd2nz":
        return [1, 1, 1, rows, cols], [rows * cols, rows * cols, rows * cols, cols, 1]
    # DN is stored as a transposed [cols, rows] tensor in column-major order.
    return [1, 1, 1, cols, rows], [rows * cols, rows * cols, rows * cols, 1, cols]


def _make_kernel(name, dtype, layout):
    a_shape, a_strides = _view_metadata(layout, M, K)
    b_shape, b_strides = _view_metadata(layout, K, N)
    out_shape = [1, 1, 1, M, N]
    out_strides = [M * N, M * N, M * N, N, 1]
    # Cube kernels are emitted at level 3 by the simulator runner, so every
    # tile buffer needs an explicit address.  Addresses are per memory space.
    mat_a_addr = 0
    mat_b_addr = 4096
    left_addr = 0
    right_addr = 0
    acc_addr = 0

    @pto.jit(
        name="tload_mat_" + name,
        kernel_kind="cube",
        target="a5",
        mode="explicit",
        insert_sync=False,
    )
    def _kernel(
        x1_ptr: pto.ptr(dtype, "gm"),
        x2_ptr: pto.ptr(dtype, "gm"),
        dst_ptr: pto.ptr(pto.f32, "gm"),
    ):
        # Keep the legacy TLOAD.MAT rank-5 view shape/strides. DN sources are
        # transposed [cols, rows] views with column-major physical strides.
        x1_view = pto.make_tensor_view(
            x1_ptr, shape=a_shape, strides=a_strides,
            layout="DN" if layout == "dn2nz" else None,
        )
        x2_view = pto.make_tensor_view(
            x2_ptr, shape=b_shape, strides=b_strides,
            layout="DN" if layout == "dn2nz" else None,
        )
        dst_view = pto.make_tensor_view(dst_ptr, shape=out_shape, strides=out_strides)

        a_tile = pto.alloc_tile(
            shape=[M, K], dtype=dtype, memory_space=pto.MemorySpace.MAT,
            addr=mat_a_addr,
            valid_shape=[M, K], blayout="ColMajor", slayout="RowMajor",
        )
        b_tile = pto.alloc_tile(
            shape=[K, N], dtype=dtype, memory_space=pto.MemorySpace.MAT,
            addr=mat_b_addr,
            valid_shape=[K, N], blayout="ColMajor", slayout="RowMajor",
        )
        a_left = pto.alloc_tile(
            shape=[M, K], dtype=dtype, memory_space=pto.MemorySpace.LEFT,
            addr=left_addr,
            valid_shape=[M, K], blayout="ColMajor", slayout="RowMajor",
        )
        b_right = pto.alloc_tile(
            shape=[K, N], dtype=dtype, memory_space=pto.MemorySpace.RIGHT,
            addr=right_addr,
            valid_shape=[K, N], blayout="RowMajor", slayout="ColMajor",
        )
        c_tile = pto.alloc_tile(
            shape=[M, N], dtype=pto.f32, memory_space=pto.MemorySpace.ACC,
            addr=acc_addr,
            valid_shape=[M, N], blayout="ColMajor", slayout="RowMajor",
            fractal_size=1024,
        )

        # The MAT movement views are rank-5, while the MAT tiles are
        # logically rank-2.  Supply the full rank-5 transfer metadata so the
        # public load surface can select the ND2NZ/DN2NZ template.
        pto.tile.load(
            x1_view,
            a_tile,
            offsets=[0, 0, 0, 0, 0],
            sizes=a_shape,
        )
        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
        pto.tile.mov(a_tile, a_left)
        pto.tile.load(
            x2_view,
            b_tile,
            offsets=[0, 0, 0, 0, 0],
            sizes=b_shape,
        )
        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
        pto.tile.mov(b_tile, b_right)
        pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
        pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
        pto.tile.matmul(a_left, b_right, c_tile)
        pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
        pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
        pto.tile.store(
            c_tile,
            dst_view,
            offsets=[0, 0, 0, 0, 0],
            sizes=out_shape,
        )

    return _kernel


_KERNELS = {
    name: _make_kernel(name, dtype, layout)
    for name, dtype, _, layout in CASE_SPECS
}


def _f32_to_bf16_f32(array):
    bits = np.asarray(array, dtype=np.float32).view(np.uint32)
    return (bits & np.uint32(0xFFFF0000)).view(np.float32)


def _make_inputs(name, dtype, np_dtype, layout):
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    x1 = np.random.uniform(-1.0, 1.0, size=(M, K)).astype(np.float32)
    x2 = np.random.uniform(-1.0, 1.0, size=(K, N)).astype(np.float32)
    if dtype is pto.bf16:
        x1 = _f32_to_bf16_f32(x1).view(np.uint32).astype(np.uint16)
        x2 = _f32_to_bf16_f32(x2).view(np.uint32).astype(np.uint16)
    elif np_dtype is np.float16:
        x1 = x1.astype(np.float16)
        x2 = x2.astype(np.float16)
    if layout == "dn2nz":
        # Match the legacy host writer: x.T is written in C order, so the
        # physical flat buffer contains the transposed logical matrix.
        x1 = np.ascontiguousarray(x1.T)
        x2 = np.ascontiguousarray(x2.T)
    return [x1, x2]


def _decode_input(array, dtype):
    if dtype is pto.bf16:
        bits = np.asarray(array, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
        return bits.view(np.float32)
    return np.asarray(array)


def _make_expected(x1, x2, dtype, layout):
    x1 = _decode_input(x1, dtype)
    x2 = _decode_input(x2, dtype)
    if layout == "dn2nz":
        x1 = x1.T
        x2 = x2.T
    return (x1.astype(np.float32) @ x2.astype(np.float32)).astype(np.float32)


CASES = [
    golden_output_case(
        "tload_mat_" + name,
        _KERNELS[name],
        inputs=lambda name=name, dtype=dtype, np_dtype=np_dtype, layout=layout:
            _make_inputs(name, dtype, np_dtype, layout),
        expected=lambda x1, x2, dtype=dtype, layout=layout:
            _make_expected(x1, x2, dtype, layout),
        rtol=1e-3,
        atol=1e-3,
    )
    for name, dtype, np_dtype, layout in CASE_SPECS
]


auto_main(globals())
