#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL coverage for the public portion of ``tstore_acc2gm``.

The two f32-output NZ2ND cases use public ``pto.tile.load`` for GM->MAT,
``pto.tile.mov`` for MAT->LEFT/RIGHT, and ``pto.tile.store`` for ACC->GM.
Scalar/vector quantized ``tstore_fp`` variants remain out of scope for this
case module; the current cases cover ordinary ACC writeback, destination
conversion, NZ2DN, and NZ2NZ.
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


def _bf16_to_bits(values):
    return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _bits_to_f32(values):
    return (values.astype(np.uint32) << 16).view(np.float32)


def _view_metadata(layout, rows, cols, dtype=None):
    if layout == "nz2dn":
        # The legacy case uses the DN logical shape [N, M] and contiguous
        # physical storage.  Keep the canonical rank-5 spelling so the
        # fixpipe template can consume the real gShape4/gStride4 fields.
        return [1, 1, 1, cols, rows], [rows * cols] * 3 + [1, cols]
    if layout == "nz2nz":
        # Keep the current A5 rank-5 NZ inference contract: shape[2] is the
        # 16-row fractal axis and shape[2] * shape[3] * elem_bytes is 512.
        # The view has the same 512-element physical extent as the host
        # buffer; the template maps the logical ACC M/N separately.
        elem_bytes = 4 if dtype == pto.f32 else 2
        fractal_cols = 512 // (16 * elem_bytes)
        tail = (rows * cols) // (16 * fractal_cols)
        return [1, 1, 16, fractal_cols, tail], [
            16 * fractal_cols * tail,
            16 * fractal_cols * tail,
            fractal_cols * tail,
            tail,
            1,
        ]
    return [1, 1, 1, rows, cols], [rows * cols, rows * cols, rows * cols, cols, 1]


def _make_kernel(name, src_dtype, dst_dtype, layout):
    a_shape, a_strides = _view_metadata("nz2nd", M, K)
    b_shape, b_strides = _view_metadata("nz2nd", K, N)
    if layout == "nz2dn":
        out_shape, out_strides, out_layout = _view_metadata("nz2dn", M, N, dst_dtype) + ("DN",)
    elif layout == "nz2nz":
        out_shape, out_strides, out_layout = _view_metadata("nz2nz", M, N, dst_dtype) + ("NZ",)
    else:
        out_shape, out_strides, out_layout = _view_metadata("nz2nd", M, N) + (None,)

    @pto.jit(
        name="tstore_acc2gm_" + name,
        kernel_kind="cube",
        target="a5",
        mode="explicit",
        insert_sync=False,
    )
    def _kernel(
        a_ptr: pto.ptr(src_dtype, "gm"),
        b_ptr: pto.ptr(src_dtype, "gm"),
        dst_ptr: pto.ptr(dst_dtype, "gm"),
    ):
        a_mat = pto.alloc_tile(
            shape=[M, K], dtype=src_dtype, memory_space=pto.MemorySpace.MAT,
            addr=0, valid_shape=[M, K], blayout="ColMajor", slayout="RowMajor",
        )
        b_mat = pto.alloc_tile(
            shape=[K, N], dtype=src_dtype, memory_space=pto.MemorySpace.MAT,
            addr=512, valid_shape=[K, N], blayout="ColMajor", slayout="RowMajor",
        )
        a_left = pto.alloc_tile(
            shape=[M, K], dtype=src_dtype, memory_space=pto.MemorySpace.LEFT,
            addr=0, valid_shape=[M, K], blayout="ColMajor", slayout="RowMajor",
        )
        b_right = pto.alloc_tile(
            shape=[K, N], dtype=src_dtype, memory_space=pto.MemorySpace.RIGHT,
            addr=0, valid_shape=[K, N], blayout="RowMajor", slayout="ColMajor",
        )
        acc = pto.alloc_tile(
            shape=[M, N], dtype=pto.f32, memory_space=pto.MemorySpace.ACC,
            addr=0, valid_shape=[M, N], blayout="ColMajor", slayout="RowMajor",
            fractal_size=1024,
        )

        a_view = pto.make_tensor_view(a_ptr, shape=a_shape, strides=a_strides)
        b_view = pto.make_tensor_view(b_ptr, shape=b_shape, strides=b_strides)
        dst_view = pto.make_tensor_view(
            dst_ptr, shape=out_shape, strides=out_strides, layout=out_layout,
        )

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
        pto.tile.matmul(a_left, b_right, acc)
        pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
        pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
        pto.tile.store(
            acc,
            dst_view,
            offsets=[0] * len(out_shape),
            sizes=out_shape,
        )
        pto.pipe_barrier(pto.Pipe.ALL)

    return _kernel


_KERNELS = {}
for _name, _src, _dst, _layout in (
    ("f16_f32_f32_nz2nd", pto.f16, pto.f32, "nz2nd"),
    ("bf16_f32_f32_nz2nd", pto.bf16, pto.f32, "nz2nd"),
    ("f16_f32_f16_nz2nd", pto.f16, pto.f16, "nz2nd"),
    ("bf16_f32_bf16_nz2nd", pto.bf16, pto.bf16, "nz2nd"),
    ("f16_f32_f16_nz2dn", pto.f16, pto.f16, "nz2dn"),
    ("f16_f32_f32_nz2nz", pto.f16, pto.f32, "nz2nz"),
):
    _KERNELS[_name] = _make_kernel(_name, _src, _dst, _layout)


def _make_inputs(name, src_dtype):
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    a = np.random.uniform(-1.0, 1.0, (M, K)).astype(np.float32)
    b = np.random.uniform(-1.0, 1.0, (K, N)).astype(np.float32)
    if src_dtype == pto.bf16:
        return [_bf16_to_bits(a), _bf16_to_bits(b)]
    return [a.astype(np.float16), b.astype(np.float16)]


def _expected(a, b, *, src_dtype, dst_dtype, layout):
    if src_dtype == pto.bf16:
        value = _bits_to_f32(a) @ _bits_to_f32(b)
    else:
        value = a.astype(np.float32) @ b.astype(np.float32)
    if dst_dtype == pto.bf16:
        value = _bf16_to_bits(value)
    elif dst_dtype == pto.f16:
        value = value.astype(np.float16)
    else:
        value = value.astype(np.float32)
    if layout == "nz2dn":
        return value.T.copy()
    if layout == "nz2nz":
        # A5 NZ writeback stores [N/C0, M/16, 16, C0] blocks.  The PTODSL
        # view uses a rank-5 spelling with the same physical element count,
        # but the host golden remains in the legacy NZ block shape.
        c0 = 8 if dst_dtype == pto.f32 else 16
        return value.reshape(M // 16, 16, N // c0, c0).transpose(2, 0, 1, 3).copy()
    return value


_SPECS = [
    ("f16_f32_f32_nz2nd", pto.f16, pto.f32, "nz2nd", np.float32, 1e-3),
    ("bf16_f32_f32_nz2nd", pto.bf16, pto.f32, "nz2nd", np.float32, 1e-3),
    ("f16_f32_f16_nz2nd", pto.f16, pto.f16, "nz2nd", np.float16, 1e-3),
    ("bf16_f32_bf16_nz2nd", pto.bf16, pto.bf16, "nz2nd", np.uint16, 1e-3),
    ("f16_f32_f16_nz2dn", pto.f16, pto.f16, "nz2dn", np.float16, 1e-3),
    ("f16_f32_f32_nz2nz", pto.f16, pto.f32, "nz2nz", np.float32, 1e-3),
]


CASES = [
    golden_output_case(
        "tstore_acc2gm_" + name,
        _KERNELS[name],
        inputs=lambda _n=name, _s=src: _make_inputs(_n, _s),
        expected=lambda a, b, _s=src, _d=dst, _l=layout: _expected(
            a, b, src_dtype=_s, dst_dtype=_d, layout=_l,
        ),
        rtol=eps,
        atol=eps,
    )
    for name, src, dst, layout, _out_dtype, eps in _SPECS
]


auto_main(globals())
