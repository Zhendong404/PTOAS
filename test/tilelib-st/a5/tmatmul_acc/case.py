#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL port of the legacy tmatmul_acc ST suite (Split-K cube matmul).

Each kernel computes C[M, N] = A[M, K] x B[K, N] with K split into two chunks
of BASEK.  The first chunk uses ``pto.tile.matmul`` (zero-init L0C), the second
accumulates with ``pto.tile.matmul_acc``.  This is a cube op with explicit
movement (MAT ``pto.tile.load`` -> L1, ``mte_l1_l0a/l0b`` -> L0A/L0B,
ACC ``pto.tile.store`` -> GM), so the kernels use PTODSL explicit mode with the
exact tile addresses, layouts and set_flag/wait_flag event ids from the legacy
``tmatmul_acc.pto``.

Fidelity notes (mirrored 1:1 from the legacy suite):

- dtype (f16 in, f32 accumulate), shapes, ``M_aligned``/``N_aligned`` padding,
  and ``eps`` are taken verbatim from the legacy ``cases.py`` table.
- Legacy L1 tile addresses are preserved exactly: case ``f16_16x32x16`` uses
  three L1 tiles (a1@0, b1@512, a2@1024, pass 1 reuses b1 for b2); the two
  128-wide cases use two L1 tiles (a@0, b@16384) reused by both passes.
- Event ids are preserved per case: the 16x32x16 kernel uses ``EVENT_ID2`` for
  the pass-1 MTE2->MTE1 sync, the 128-wide kernels use ``EVENT_ID1``.
- Host inputs are the K-chunks of the legacy padded ``a``/``b`` buffers as
  separate GM buffers (the TileLib framework materializes inputs contiguously,
  so each chunk carries its own row stride: ``BASEK * 2`` bytes for A chunks
  and ``N_aligned * 2`` bytes for B chunks).  This is equivalent to the legacy
  launcher passing chunk pointers into one padded buffer with the full-K
  stride; the kernel-visible movement, math and golden are unchanged.
- Data is regenerated deterministically per case with the legacy per-case seed
  ``zlib.crc32(case_name) & 0xFFFFFFFF`` (see ``st_common.setup_case_rng``).
- ``rtol``/``atol`` equal the legacy ``eps`` (legacy ``result_cmp`` used
  ``np.allclose(..., atol=eps, rtol=eps)``).
"""

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


# ---------------------------------------------------------------------------
# Legacy cases.py table (single source of truth, kept byte-for-byte)
# ---------------------------------------------------------------------------

CASE_SPECS = [
    {
        "name": "f16_16x32x16",
        "dtype": np.float16,
        "M": 16,
        "K": 32,
        "N": 16,
        "BASEK": 16,
        "M_aligned": 16,
        "N_aligned": 16,
        "shape_c": (16, 16),
        "eps": 1e-2,
    },
    {
        "name": "f16_128x128x64",
        "dtype": np.float16,
        "M": 128,
        "K": 128,
        "N": 64,
        "BASEK": 64,
        "M_aligned": 128,
        "N_aligned": 64,
        "shape_c": (128, 64),
        "eps": 1e-2,
    },
    {
        "name": "f16_127x128x61",
        "dtype": np.float16,
        "M": 127,
        "K": 128,
        "N": 61,
        "BASEK": 64,
        "M_aligned": 128,
        "N_aligned": 64,
        "shape_c": (127, 61),
        "eps": 1e-2,
    },
]

# ---------------------------------------------------------------------------
# Host data (mirrors legacy gen_data.py + st_common.setup_case_rng)
# ---------------------------------------------------------------------------


def _make_inputs(name, m, k, n, m_aligned, n_aligned, base_k):
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    a = np.random.uniform(-1.0, 1.0, size=(m, k)).astype(np.float16)
    b = np.random.uniform(-1.0, 1.0, size=(k, n)).astype(np.float16)

    # Pad A/B to aligned dimensions exactly like the legacy gen_data.py so the
    # kernel can load aligned tiles without out-of-bounds reads.
    a_padded = np.zeros((m_aligned, k), dtype=np.float16)
    a_padded[:m, :] = a
    b_padded = np.zeros((k, n_aligned), dtype=np.float16)
    b_padded[:, :n] = b

    # K-chunks as separate contiguous GM buffers, in kernel argument order
    # (a1, b1, a2, b2).  K == 2 * BASEK for every case.
    return [
        a_padded[:, :base_k].copy(),
        b_padded[:base_k, :].copy(),
        a_padded[:, base_k:].copy(),
        b_padded[base_k:, :].copy(),
    ]


def _make_expected(a1, b1, a2, b2):
    # Full padded A @ B in fp32 == legacy golden (padded region is zero in
    # both the padded inputs and the kernel output).
    a = np.concatenate([a1, a2], axis=1).astype(np.float32)
    b = np.concatenate([b1, b2], axis=0).astype(np.float32)
    return (a @ b).astype(np.float32)


# ---------------------------------------------------------------------------
# Kernels (cube / explicit mode, movement mirrored from legacy tmatmul_acc.pto)
# ---------------------------------------------------------------------------

# f16_16x32x16: M=16, K=32, N=16, BASEK=16, iter=2
# L1 tiles: a1@0, b1@512, a2@1024 (pass 1 reuses b1 for b2); L0A/L0B/L0C @0.
@pto.jit(
    name="tmatmul_acc_f16_16x32x16",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel_f16_16x32x16(
    a1_ptr: pto.ptr(pto.f16, "gm"),
    b1_ptr: pto.ptr(pto.f16, "gm"),
    a2_ptr: pto.ptr(pto.f16, "gm"),
    b2_ptr: pto.ptr(pto.f16, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    l1_a1 = pto.alloc_tile(
        shape=[16, 16], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, addr=0,
        valid_shape=[16, 16], blayout="ColMajor", slayout="RowMajor",
    )
    l1_b1 = pto.alloc_tile(
        shape=[16, 16], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, addr=512,
        valid_shape=[16, 16], blayout="ColMajor", slayout="RowMajor",
    )
    l1_a2 = pto.alloc_tile(
        shape=[16, 16], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, addr=1024,
        valid_shape=[16, 16], blayout="ColMajor", slayout="RowMajor",
    )
    a_l0a = pto.alloc_tile(
        shape=[16, 16], dtype=pto.f16, memory_space=pto.MemorySpace.LEFT, addr=0,
        valid_shape=[16, 16], blayout="ColMajor", slayout="RowMajor",
    )
    b_l0b = pto.alloc_tile(
        shape=[16, 16], dtype=pto.f16, memory_space=pto.MemorySpace.RIGHT, addr=0,
        valid_shape=[16, 16], blayout="RowMajor", slayout="ColMajor",
    )
    c_acc = pto.alloc_tile(
        shape=[16, 16], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, addr=0,
        valid_shape=[16, 16], blayout="ColMajor", slayout="RowMajor", fractal_size=1024,
    )

    a1_view = pto.make_tensor_view(a1_ptr, shape=[1, 1, 1, 16, 16], strides=[256, 256, 256, 16, 1])
    b1_view = pto.make_tensor_view(b1_ptr, shape=[1, 1, 1, 16, 16], strides=[256, 256, 256, 16, 1])
    a2_view = pto.make_tensor_view(a2_ptr, shape=[1, 1, 1, 16, 16], strides=[256, 256, 256, 16, 1])
    b2_view = pto.make_tensor_view(b2_ptr, shape=[1, 1, 1, 16, 16], strides=[256, 256, 256, 16, 1])
    c_view = pto.make_tensor_view(c_ptr, shape=[1, 1, 1, 16, 16], strides=[256, 256, 256, 16, 1])
    shape_16 = [1, 1, 1, 16, 16]

    # ---- Pass 0: A[:,0:16] * B[0:16,:] (zero-init) ----
    pto.tile.load(a1_view, l1_a1, offsets=[0, 0, 0, 0, 0], sizes=shape_16)
    pto.tile.load(b1_view, l1_b1, offsets=[0, 0, 0, 0, 0], sizes=shape_16)
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.mte_l1_l0a(l1_a1.as_ptr(), a_l0a.as_ptr(), 16, 16)
    pto.mte_l1_l0b(l1_b1.as_ptr(), b_l0b.as_ptr(), 16, 16, transpose=True)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.tile.matmul(a_l0a, b_l0b, c_acc)

    pto.set_flag(pto.Pipe.M, pto.Pipe.MTE2, event_id=0)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.MTE2, event_id=0)

    # ---- Pass 1: A[:,16:32] * B[16:32,:] (accumulate) ----
    pto.tile.load(a2_view, l1_a2, offsets=[0, 0, 0, 0, 0], sizes=shape_16)
    pto.tile.load(b2_view, l1_b1, offsets=[0, 0, 0, 0, 0], sizes=shape_16)
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=2)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=2)
    pto.mte_l1_l0a(l1_a2.as_ptr(), a_l0a.as_ptr(), 16, 16)
    pto.mte_l1_l0b(l1_b1.as_ptr(), b_l0b.as_ptr(), 16, 16, transpose=True)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)
    pto.tile.matmul_acc(c_acc, a_l0a, b_l0b, c_acc)

    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.tile.store(c_acc, c_view, offsets=[0, 0, 0, 0, 0], sizes=shape_16)
    pto.pipe_barrier(pto.Pipe.ALL)


# f16_128x128x64: M=128, K=128, N=64, BASEK=64, iter=2
# L1 tiles: a@0 (128x64 f16), b@16384 (64x64 f16); both reused by pass 1.
@pto.jit(
    name="tmatmul_acc_f16_128x128x64",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel_f16_128x128x64(
    a1_ptr: pto.ptr(pto.f16, "gm"),
    b1_ptr: pto.ptr(pto.f16, "gm"),
    a2_ptr: pto.ptr(pto.f16, "gm"),
    b2_ptr: pto.ptr(pto.f16, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    l1_a1 = pto.alloc_tile(
        shape=[128, 64], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, addr=0,
        valid_shape=[128, 64], blayout="ColMajor", slayout="RowMajor",
    )
    l1_b1 = pto.alloc_tile(
        shape=[64, 64], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, addr=16384,
        valid_shape=[64, 64], blayout="ColMajor", slayout="RowMajor",
    )
    a_l0a = pto.alloc_tile(
        shape=[128, 64], dtype=pto.f16, memory_space=pto.MemorySpace.LEFT, addr=0,
        valid_shape=[128, 64], blayout="ColMajor", slayout="RowMajor",
    )
    b_l0b = pto.alloc_tile(
        shape=[64, 64], dtype=pto.f16, memory_space=pto.MemorySpace.RIGHT, addr=0,
        valid_shape=[64, 64], blayout="RowMajor", slayout="ColMajor",
    )
    c_acc = pto.alloc_tile(
        shape=[128, 64], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, addr=0,
        valid_shape=[128, 64], blayout="ColMajor", slayout="RowMajor", fractal_size=1024,
    )

    a1_view = pto.make_tensor_view(a1_ptr, shape=[1, 1, 1, 128, 64], strides=[8192, 8192, 8192, 64, 1])
    b1_view = pto.make_tensor_view(b1_ptr, shape=[1, 1, 1, 64, 64], strides=[4096, 4096, 4096, 64, 1])
    a2_view = pto.make_tensor_view(a2_ptr, shape=[1, 1, 1, 128, 64], strides=[8192, 8192, 8192, 64, 1])
    b2_view = pto.make_tensor_view(b2_ptr, shape=[1, 1, 1, 64, 64], strides=[4096, 4096, 4096, 64, 1])
    c_view = pto.make_tensor_view(c_ptr, shape=[1, 1, 1, 128, 64], strides=[8192, 8192, 8192, 64, 1])
    a_shape = [1, 1, 1, 128, 64]
    b_shape = [1, 1, 1, 64, 64]

    # ---- Pass 0: A[:,0:64] * B[0:64,:] (zero-init) ----
    pto.tile.load(a1_view, l1_a1, offsets=[0, 0, 0, 0, 0], sizes=a_shape)
    pto.tile.load(b1_view, l1_b1, offsets=[0, 0, 0, 0, 0], sizes=b_shape)
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.mte_l1_l0a(l1_a1.as_ptr(), a_l0a.as_ptr(), 128, 64)
    pto.mte_l1_l0b(l1_b1.as_ptr(), b_l0b.as_ptr(), 64, 64, transpose=True)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.tile.matmul(a_l0a, b_l0b, c_acc)

    pto.set_flag(pto.Pipe.M, pto.Pipe.MTE2, event_id=0)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.MTE2, event_id=0)

    # ---- Pass 1: A[:,64:128] * B[64:128,:] (accumulate) ----
    pto.tile.load(a2_view, l1_a1, offsets=[0, 0, 0, 0, 0], sizes=a_shape)
    pto.tile.load(b2_view, l1_b1, offsets=[0, 0, 0, 0, 0], sizes=b_shape)
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
    pto.mte_l1_l0a(l1_a1.as_ptr(), a_l0a.as_ptr(), 128, 64)
    pto.mte_l1_l0b(l1_b1.as_ptr(), b_l0b.as_ptr(), 64, 64, transpose=True)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)
    pto.tile.matmul_acc(c_acc, a_l0a, b_l0b, c_acc)

    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.tile.store(c_acc, c_view, offsets=[0, 0, 0, 0, 0], sizes=a_shape)
    pto.pipe_barrier(pto.Pipe.ALL)


# f16_127x128x61: M=127 (pad->128), K=128, N=61 (pad->64), BASEK=64, iter=2
# Same L1 layout / sync structure as f16_128x128x64.
@pto.jit(
    name="tmatmul_acc_f16_127x128x61",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel_f16_127x128x61(
    a1_ptr: pto.ptr(pto.f16, "gm"),
    b1_ptr: pto.ptr(pto.f16, "gm"),
    a2_ptr: pto.ptr(pto.f16, "gm"),
    b2_ptr: pto.ptr(pto.f16, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    l1_a1 = pto.alloc_tile(
        shape=[128, 64], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, addr=0,
        valid_shape=[128, 64], blayout="ColMajor", slayout="RowMajor",
    )
    l1_b1 = pto.alloc_tile(
        shape=[64, 64], dtype=pto.f16, memory_space=pto.MemorySpace.MAT, addr=16384,
        valid_shape=[64, 64], blayout="ColMajor", slayout="RowMajor",
    )
    a_l0a = pto.alloc_tile(
        shape=[128, 64], dtype=pto.f16, memory_space=pto.MemorySpace.LEFT, addr=0,
        valid_shape=[128, 64], blayout="ColMajor", slayout="RowMajor",
    )
    b_l0b = pto.alloc_tile(
        shape=[64, 64], dtype=pto.f16, memory_space=pto.MemorySpace.RIGHT, addr=0,
        valid_shape=[64, 64], blayout="RowMajor", slayout="ColMajor",
    )
    c_acc = pto.alloc_tile(
        shape=[128, 64], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, addr=0,
        valid_shape=[128, 64], blayout="ColMajor", slayout="RowMajor", fractal_size=1024,
    )

    a1_view = pto.make_tensor_view(a1_ptr, shape=[1, 1, 1, 128, 64], strides=[8192, 8192, 8192, 64, 1])
    b1_view = pto.make_tensor_view(b1_ptr, shape=[1, 1, 1, 64, 64], strides=[4096, 4096, 4096, 64, 1])
    a2_view = pto.make_tensor_view(a2_ptr, shape=[1, 1, 1, 128, 64], strides=[8192, 8192, 8192, 64, 1])
    b2_view = pto.make_tensor_view(b2_ptr, shape=[1, 1, 1, 64, 64], strides=[4096, 4096, 4096, 64, 1])
    c_view = pto.make_tensor_view(c_ptr, shape=[1, 1, 1, 128, 64], strides=[8192, 8192, 8192, 64, 1])
    a_shape = [1, 1, 1, 128, 64]
    b_shape = [1, 1, 1, 64, 64]

    # ---- Pass 0: A[:,0:64] * B[0:64,:] (zero-init) ----
    pto.tile.load(a1_view, l1_a1, offsets=[0, 0, 0, 0, 0], sizes=a_shape)
    pto.tile.load(b1_view, l1_b1, offsets=[0, 0, 0, 0, 0], sizes=b_shape)
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.mte_l1_l0a(l1_a1.as_ptr(), a_l0a.as_ptr(), 128, 64)
    pto.mte_l1_l0b(l1_b1.as_ptr(), b_l0b.as_ptr(), 64, 64, transpose=True)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.tile.matmul(a_l0a, b_l0b, c_acc)

    pto.set_flag(pto.Pipe.M, pto.Pipe.MTE2, event_id=0)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.MTE2, event_id=0)

    # ---- Pass 1: A[:,64:128] * B[64:128,:] (accumulate) ----
    pto.tile.load(a2_view, l1_a1, offsets=[0, 0, 0, 0, 0], sizes=a_shape)
    pto.tile.load(b2_view, l1_b1, offsets=[0, 0, 0, 0, 0], sizes=b_shape)
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
    pto.mte_l1_l0a(l1_a1.as_ptr(), a_l0a.as_ptr(), 128, 64)
    pto.mte_l1_l0b(l1_b1.as_ptr(), b_l0b.as_ptr(), 64, 64, transpose=True)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)
    pto.tile.matmul_acc(c_acc, a_l0a, b_l0b, c_acc)

    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.tile.store(c_acc, c_view, offsets=[0, 0, 0, 0, 0], sizes=a_shape)
    pto.pipe_barrier(pto.Pipe.ALL)


# ---------------------------------------------------------------------------
# CASES
# ---------------------------------------------------------------------------

CASES = [
    golden_output_case(
        "tmatmul_acc_" + spec["name"],
        globals()["_kernel_" + spec["name"]],
        inputs=lambda spec=spec: _make_inputs(
            spec["name"], spec["M"], spec["K"], spec["N"],
            spec["M_aligned"], spec["N_aligned"], spec["BASEK"],
        ),
        expected=_make_expected,
        rtol=spec["eps"],
        atol=spec["eps"],
    )
    for spec in CASE_SPECS
]


auto_main(globals())
