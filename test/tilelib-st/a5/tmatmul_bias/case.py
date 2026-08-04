#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL port of the legacy tmatmul_bias ST suite (cube matmul with bias).

Each kernel computes C[M, N] = A[M, K] x B[K, N] + bias[N] with the bias added
inside the cube op (a bias tensor staged through the ``bt`` space).  Cases
0-2 and 5 are a single ``pto.mad_bias`` pass; cases 3-4 split K into two
BASEK chunks (``pto.mad_bias`` then ``pto.mad_acc``), exactly like the legacy
``tmatmul_bias.pto``.  This is a cube op with explicit movement
(``mte_gm_l1_frac`` nd2nz -> L1, ``mte_gm_l1``/``mte_l1_bt`` for the bias,
``mte_l1_l0a/l0b`` -> L0A/L0B, ``mte_l0c_gm`` nz2nd -> GM), so the kernels use
PTODSL explicit mode with the exact tile addresses, layouts and
set_flag/wait_flag event ids from the legacy ``tmatmul_bias.pto``.

Movement surface: the f16/bf16/f32 operand GM->MAT loads (including the
split-K strided sub-tile ND2NZ loads) and the ACC->GM store go through the
public ``pto.tile.load``/``pto.tile.store`` (which select the ND2NZ MAT load
and nz2nd ACC store templates with the legacy strides/addresses).  The i8
operand loads (no MAT load template for i8), the bias ``mte_gm_l1`` +
``mte_l1_bt`` path and all ``mte_l1_l0a/l0b`` moves stay explicit.

Fidelity notes (mirrored 1:1 from the legacy suite):

- The legacy ``cases.py`` table (dtype, M/K/N, M_aligned/K_use/N_aligned
  padding, split_k/base_k, eps) is kept verbatim below in ``CASE_SPECS``.
- Legacy L1 tile addresses are preserved exactly: a@0, b@512, bias@1024
  (16x16 cases), a@0, b@512, bias@1536 (i8 case), a@0, b@14336, bias@24576
  (f16 112x127x80), a@0, b@10240, bias@18432 (bf16 80x112x63), a@0, b@65536,
  bias@98304 (f32 127x128x63).
- ``mte_gm_l1_frac`` src_layout byte strides, ``mte_gm_l1`` bias burst sizes,
  ``mte_l1_bt`` element counts, and ``mte_l0c_gm`` strides match the legacy
  kernel constants (e.g. A stride 256 bytes / B stride 160 bytes for the
  112x127x80 split case).
- Event ids are preserved per case: single-pass kernels use MTE2->MTE1=0,
  MTE1->M=0, M->FIX=1; split kernels use MTE2->MTE1=1 and MTE1->M=1 for the
  pass-1 (accumulate) chunk plus an M->MTE2=0 handoff between the passes.
- Host inputs are the legacy padded A/B/bias buffers.  For the split cases the
  K-chunks are passed as separate GM buffers that keep the legacy row stride
  (A chunks are (M_aligned, K_use) with data in the first BASEK columns so the
  kernel-visible stride is K_use elements; B chunks are (BASEK, N_aligned)
  contiguous).  This is equivalent to the legacy launcher passing chunk
  pointers into one padded buffer; the kernel-visible movement, math and golden
  are unchanged.
- The golden is the full aligned-shape result ``A_padded @ B_padded +
  bias_padded`` broadcast over rows (the cube writes bias into the zero-padded
  M rows for valid columns, which the legacy compare simply never looked at).
- Data is regenerated deterministically per case with the legacy per-case seed
  ``zlib.crc32(case_name) & 0xFFFFFFFF`` (see ``st_common.setup_case_rng``).
- bf16 host buffers are passed as raw uint16 storage (identical to the legacy
  uint16_t launcher buffers; the runtime torch cannot materialize
  ml_dtypes.bfloat16 tensors).
- ``rtol``/``atol`` equal the legacy ``eps`` (legacy ``result_cmp`` used
  ``np.allclose(..., atol=eps, rtol=eps)``).
"""

from pathlib import Path
import sys
import zlib

import numpy as np
import ml_dtypes

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto

bfloat16 = ml_dtypes.bfloat16

# ---------------------------------------------------------------------------
# Legacy cases.py table (single source of truth, kept verbatim)
# ---------------------------------------------------------------------------

CASE_SPECS = [
    # ---- case 0: f16 x f16 -> f32, bias=f32, 16x16x16 ----
    {
        "name": "f16_16x16x16",
        "a_dtype": np.float16,
        "b_dtype": np.float16,
        "bias_dtype": np.float32,
        "c_dtype": np.float32,
        "M": 16, "K": 16, "N": 16,
        "M_aligned": 16, "K_use": 16, "N_aligned": 16,
        "eps": 1e-2,
    },
    # ---- case 1: i8 x i8 -> i32, bias=i32, 8x7x6 ----
    {
        "name": "i8_bias_i32_8x7x6",
        "a_dtype": np.int8,
        "b_dtype": np.int8,
        "bias_dtype": np.int32,
        "c_dtype": np.int32,
        "M": 8, "K": 7, "N": 6,
        "M_aligned": 16, "K_use": 32, "N_aligned": 32,
        "eps": 1e-6,
    },
    # ---- case 2: f16 x f16 -> f32, bias=f16, 16x15x16 ----
    {
        "name": "f16_bias_f16_16x15x16",
        "a_dtype": np.float16,
        "b_dtype": np.float16,
        "bias_dtype": np.float32,
        "c_dtype": np.float32,
        "M": 16, "K": 15, "N": 16,
        "M_aligned": 16, "K_use": 16, "N_aligned": 16,
        "eps": 1e-2,
    },
    # ---- case 3: f16 x f16 -> f32, bias=bf16, 112x127x80 (SPLIT_K) ----
    {
        "name": "f16_bias_bf16_112x127x80",
        "a_dtype": np.float16,
        "b_dtype": np.float16,
        "bias_dtype": np.float32,
        "c_dtype": np.float32,
        "M": 112, "K": 127, "N": 80,
        "M_aligned": 112, "K_use": 128, "N_aligned": 80,
        "eps": 1e-2,
        "split_k": True, "base_k": 64,
    },
    # ---- case 4: bf16 x bf16 -> f32, bias=bf16, 80x112x63 (SPLIT_K) ----
    {
        "name": "bf16_bias_bf16_80x112x63",
        "a_dtype": bfloat16,
        "b_dtype": bfloat16,
        "bias_dtype": np.float32,
        "c_dtype": np.float32,
        "M": 80, "K": 112, "N": 63,
        "M_aligned": 80, "K_use": 128, "N_aligned": 64,
        "eps": 1e-2,
        "split_k": True, "base_k": 64,
    },
    # ---- case 5: f32 x f32 -> f32, bias=f32, 127x128x63 (SPLIT_K) ----
    {
        "name": "f32_bias_f32_127x128x63",
        "a_dtype": np.float32,
        "b_dtype": np.float32,
        "bias_dtype": np.float32,
        "c_dtype": np.float32,
        "M": 127, "K": 128, "N": 63,
        "M_aligned": 128, "K_use": 128, "N_aligned": 64,
        "eps": 1e-5,
    },
]

# L1 tile addresses from the legacy tmatmul_bias.pto (a@0, b@<addr>, bias@<addr>).
_L1_B_ADDR = {
    "f16_16x16x16": 512,
    "i8_bias_i32_8x7x6": 512,
    "f16_bias_f16_16x15x16": 512,
    "f16_bias_bf16_112x127x80": 14336,
    "bf16_bias_bf16_80x112x63": 10240,
    "f32_bias_f32_127x128x63": 65536,
}
_L1_BIAS_ADDR = {
    "f16_16x16x16": 1024,
    "i8_bias_i32_8x7x6": 1536,
    "f16_bias_f16_16x15x16": 1024,
    "f16_bias_bf16_112x127x80": 24576,
    "bf16_bias_bf16_80x112x63": 18432,
    "f32_bias_f32_127x128x63": 98304,
}

# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------


def _pto_dtype(dtype):
    if dtype == np.float16:
        return pto.f16
    if dtype == np.float32:
        return pto.f32
    if dtype == np.int8:
        return pto.i8
    if dtype == np.int32:
        return pto.i32
    if dtype == bfloat16:
        return pto.bf16
    raise TypeError(f"unsupported dtype {dtype}")


def _element_bytes(dtype):
    return np.dtype(dtype).itemsize


def _acc_fractal(c_dtype):
    # Legacy acc tiles use fractal=1024 for f32; the i8/i32 acc tile has no
    # fractal attr, which is the MLIR default of 512.
    return 1024 if c_dtype == np.float32 else 512


# ---------------------------------------------------------------------------
# Host data (mirrors legacy gen_data.py + st_common.setup_case_rng)
# ---------------------------------------------------------------------------


def _make_inputs(spec):
    np.random.seed(zlib.crc32(spec["name"].encode("utf-8")) & 0xFFFFFFFF)

    m, k, n = spec["M"], spec["K"], spec["N"]
    m_aligned, k_aligned, n_aligned = spec["M_aligned"], spec["K_use"], spec["N_aligned"]
    a_dtype, b_dtype, bias_dtype = spec["a_dtype"], spec["b_dtype"], spec["bias_dtype"]

    x1 = np.random.randint(-10, 10, size=(m, k)).astype(a_dtype)
    x2 = np.random.randint(-10, 10, size=(k, n)).astype(b_dtype)
    bias = np.random.randint(1, 10, size=(n,)).astype(bias_dtype)

    # Pad A, B, bias to aligned dimensions so the kernel can load aligned
    # tiles without reading out-of-bounds memory (legacy gen_data.py).
    a_padded = np.zeros((m_aligned, k_aligned), dtype=a_dtype)
    a_padded[:m, :k] = x1
    b_padded = np.zeros((k_aligned, n_aligned), dtype=b_dtype)
    b_padded[:k, :n] = x2
    bias_padded = np.zeros((n_aligned,), dtype=bias_dtype)
    bias_padded[:n] = bias

    def host_buffer(array):
        # Runtime torch cannot materialize ml_dtypes.bfloat16; pass the raw
        # 16-bit storage (identical to the legacy uint16_t host buffers).
        if array.dtype == bfloat16:
            return np.asarray(array).view(np.uint16)
        return array

    if spec.get("split_k"):
        base_k = spec["base_k"]
        # K-chunks as separate contiguous GM buffers keeping the legacy row
        # stride: A chunks are (M_aligned, K_use) with data in the first
        # BASEK columns, B chunks are (BASEK, N_aligned) contiguous.
        a1 = np.zeros((m_aligned, k_aligned), dtype=a_dtype)
        a1[:, :base_k] = a_padded[:, :base_k]
        a2 = np.zeros((m_aligned, k_aligned), dtype=a_dtype)
        a2[:, :base_k] = a_padded[:, base_k:]
        b1 = b_padded[:base_k, :]
        b2 = b_padded[base_k:, :]
        return [host_buffer(a1), host_buffer(b1), host_buffer(a2), host_buffer(b2), bias_padded]
    return [host_buffer(a_padded), host_buffer(b_padded), bias_padded]


def _make_expected(spec):
    c_dtype = spec["c_dtype"]

    def expected(*inputs):
        def restore(array):
            arr = np.asarray(array)
            if arr.dtype == np.uint16 and (
                spec["a_dtype"] == bfloat16 or spec["b_dtype"] == bfloat16
            ):
                return arr.view(bfloat16)
            return arr

        if spec.get("split_k"):
            a1, b1, a2, b2, bias = inputs
            base_k = spec["base_k"]
            a = np.concatenate([restore(a1)[:, :base_k], restore(a2)[:, :base_k]], axis=1)
            b = np.concatenate([restore(b1), restore(b2)], axis=0)
        else:
            a, b, bias = inputs
            a = restore(a)
            b = restore(b)
        bias = np.asarray(bias)
        return (
            np.matmul(a.astype(c_dtype), b.astype(c_dtype)).astype(c_dtype)
            + bias.astype(c_dtype)
        ).astype(c_dtype)

    return expected


# ---------------------------------------------------------------------------
# Kernels (cube / explicit mode, movement mirrored from legacy tmatmul_bias.pto)
# ---------------------------------------------------------------------------


def _build_single_pass_kernel(spec):
    """Single-pass pto.mad_bias kernels (cases 0, 1, 2, 5)."""
    m = spec["M_aligned"]
    k = spec["K_use"]
    n = spec["N_aligned"]
    a_pto = _pto_dtype(spec["a_dtype"])
    b_pto = _pto_dtype(spec["b_dtype"])
    bias_pto = _pto_dtype(spec["bias_dtype"])
    c_pto = _pto_dtype(spec["c_dtype"])
    l1_b_addr = _L1_B_ADDR[spec["name"]]
    l1_bias_addr = _L1_BIAS_ADDR[spec["name"]]
    a_stride = k * _element_bytes(spec["a_dtype"])
    b_stride = n * _element_bytes(spec["b_dtype"])
    bias_burst = n * _element_bytes(spec["bias_dtype"])
    acc_fractal = _acc_fractal(spec["c_dtype"])
    kernel_name = "tmatmul_bias_" + spec["name"]
    # f16/bf16/f32 operand GM->MAT goes through the public tile.load ND2NZ
    # surface (exactly the legacy mte_gm_l1_frac with row stride = K_use);
    # the i8 case has no MAT load template, so it keeps the explicit fractal
    # movement.
    use_tile_load = spec["a_dtype"] in (np.float16, np.float32, bfloat16)

    @pto.jit(
        name=kernel_name,
        kernel_kind="cube",
        target="a5",
        mode="explicit",
        insert_sync=False,
    )
    def kernel(
        a_ptr: pto.ptr(a_pto, "gm"),
        b_ptr: pto.ptr(b_pto, "gm"),
        bias_ptr: pto.ptr(bias_pto, "gm"),
        c_ptr: pto.ptr(c_pto, "gm"),
    ):
        if use_tile_load:
            l1_a = pto.alloc_tile(
                shape=[m, k], dtype=a_pto, memory_space=pto.MemorySpace.MAT, addr=0,
                valid_shape=[m, k], blayout="ColMajor", slayout="RowMajor",
            )
            l1_b = pto.alloc_tile(
                shape=[k, n], dtype=b_pto, memory_space=pto.MemorySpace.MAT, addr=l1_b_addr,
                valid_shape=[k, n], blayout="ColMajor", slayout="RowMajor",
            )
        else:
            l1_a = pto.castptr(pto.ui64(0), pto.ptr(a_pto, "mat"))
            l1_b = pto.castptr(pto.ui64(l1_b_addr), pto.ptr(b_pto, "mat"))
        l1_bias = pto.castptr(pto.ui64(l1_bias_addr), pto.ptr(bias_pto, "mat"))

        a_l0a = pto.alloc_tile(
            shape=[m, k], dtype=a_pto, memory_space=pto.MemorySpace.LEFT, addr=0,
            blayout="ColMajor", slayout="RowMajor",
        )
        b_l0b = pto.alloc_tile(
            shape=[k, n], dtype=b_pto, memory_space=pto.MemorySpace.RIGHT, addr=0,
            blayout="RowMajor", slayout="ColMajor",
        )
        c_acc = pto.alloc_tile(
            shape=[m, n], dtype=c_pto, memory_space=pto.MemorySpace.ACC, addr=0,
            blayout="ColMajor", slayout="RowMajor", fractal_size=acc_fractal,
        )
        bias_tile = pto.alloc_tile(
            shape=[1, n], dtype=bias_pto, memory_space=pto.MemorySpace.BIAS, addr=0,
            blayout="RowMajor", slayout="NoneBox",
        )

        if use_tile_load:
            a_view = pto.make_tensor_view(
                a_ptr, shape=[1, 1, 1, m, k], strides=[m * k, m * k, m * k, k, 1]
            )
            b_view = pto.make_tensor_view(
                b_ptr, shape=[1, 1, 1, k, n], strides=[k * n, k * n, k * n, n, 1]
            )
            pto.tile.load(a_view, l1_a, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, m, k])
            pto.tile.load(b_view, l1_b, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, k, n])
        else:
            pto.mte_gm_l1_frac(
                a_ptr, l1_a, pto.FractalMode.ND2NZ,
                shape=(m, k), src_layout=(a_stride,),
                dst_group=(1, 1, m, 0), ctrl=(0, False),
            )
            pto.mte_gm_l1_frac(
                b_ptr, l1_b, pto.FractalMode.ND2NZ,
                shape=(k, n), src_layout=(b_stride,),
                dst_group=(1, 1, k, 0), ctrl=(0, False),
            )
        pto.mte_gm_l1(bias_ptr, l1_bias, bias_burst, nburst=(1, 0, 0))

        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
        pto.mte_l1_l0a(l1_a.as_ptr() if use_tile_load else l1_a, a_l0a.as_ptr(), m, k)
        pto.mte_l1_l0b(l1_b.as_ptr() if use_tile_load else l1_b, b_l0b.as_ptr(), k, n, transpose=True)
        pto.mte_l1_bt(l1_bias, bias_tile.as_ptr(), n, nburst=(1, 0, 0))

        pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
        pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
        pto.mad_bias(
            a_l0a.as_ptr(), b_l0b.as_ptr(), c_acc.as_ptr(), bias_tile.as_ptr(),
            m, n, k, disable_gemv=True,
        )

        pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
        pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
        c_view = pto.make_tensor_view(
            c_ptr, shape=[1, 1, 1, m, n], strides=[m * n, m * n, m * n, n, 1]
        )
        pto.tile.store(c_acc, c_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, m, n])
        pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


def _build_split_kernel(spec):
    """Split-K pto.mad_bias + pto.mad_acc kernels (cases 3, 4)."""
    m = spec["M_aligned"]
    k = spec["K_use"]
    n = spec["N_aligned"]
    base_k = spec["base_k"]
    a_pto = _pto_dtype(spec["a_dtype"])
    b_pto = _pto_dtype(spec["b_dtype"])
    bias_pto = _pto_dtype(spec["bias_dtype"])
    c_pto = _pto_dtype(spec["c_dtype"])
    l1_b_addr = _L1_B_ADDR[spec["name"]]
    l1_bias_addr = _L1_BIAS_ADDR[spec["name"]]
    bias_burst = n * _element_bytes(spec["bias_dtype"])
    acc_fractal = _acc_fractal(spec["c_dtype"])
    kernel_name = "tmatmul_bias_" + spec["name"]

    @pto.jit(
        name=kernel_name,
        kernel_kind="cube",
        target="a5",
        mode="explicit",
        insert_sync=False,
    )
    def kernel(
        a1_ptr: pto.ptr(a_pto, "gm"),
        b1_ptr: pto.ptr(b_pto, "gm"),
        a2_ptr: pto.ptr(a_pto, "gm"),
        b2_ptr: pto.ptr(b_pto, "gm"),
        bias_ptr: pto.ptr(bias_pto, "gm"),
        c_ptr: pto.ptr(c_pto, "gm"),
    ):
        l1_a = pto.alloc_tile(
            shape=[m, base_k], dtype=a_pto, memory_space=pto.MemorySpace.MAT, addr=0,
            valid_shape=[m, base_k], blayout="ColMajor", slayout="RowMajor",
        )
        l1_b = pto.alloc_tile(
            shape=[base_k, n], dtype=b_pto, memory_space=pto.MemorySpace.MAT, addr=l1_b_addr,
            valid_shape=[base_k, n], blayout="ColMajor", slayout="RowMajor",
        )
        l1_bias = pto.castptr(pto.ui64(l1_bias_addr), pto.ptr(bias_pto, "mat"))

        a_l0a = pto.alloc_tile(
            shape=[m, base_k], dtype=a_pto, memory_space=pto.MemorySpace.LEFT, addr=0,
            blayout="ColMajor", slayout="RowMajor",
        )
        b_l0b = pto.alloc_tile(
            shape=[base_k, n], dtype=b_pto, memory_space=pto.MemorySpace.RIGHT, addr=0,
            blayout="RowMajor", slayout="ColMajor",
        )
        c_acc = pto.alloc_tile(
            shape=[m, n], dtype=c_pto, memory_space=pto.MemorySpace.ACC, addr=0,
            blayout="ColMajor", slayout="RowMajor", fractal_size=acc_fractal,
        )
        bias_tile = pto.alloc_tile(
            shape=[1, n], dtype=bias_pto, memory_space=pto.MemorySpace.BIAS, addr=0,
            blayout="RowMajor", slayout="NoneBox",
        )

        a1_view = pto.make_tensor_view(
            a1_ptr, shape=[1, 1, 1, m, k], strides=[m * k, m * k, m * k, k, 1]
        )
        b1_view = pto.make_tensor_view(
            b1_ptr, shape=[1, 1, 1, base_k, n], strides=[base_k * n, base_k * n, base_k * n, n, 1]
        )
        a2_view = pto.make_tensor_view(
            a2_ptr, shape=[1, 1, 1, m, k], strides=[m * k, m * k, m * k, k, 1]
        )
        b2_view = pto.make_tensor_view(
            b2_ptr, shape=[1, 1, 1, base_k, n], strides=[base_k * n, base_k * n, base_k * n, n, 1]
        )

        # ---- Pass 0: A[:, 0:BASEK] * B[0:BASEK, :] + bias ----
        # Split-K operands are f16/bf16/f32, so the public MAT load surface
        # applies; the strided sub-tile (BASEK columns of a K_use-wide GM row)
        # is expressed via the rank-5 view strides and load sizes below.
        pto.tile.load(a1_view, l1_a, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, m, base_k])
        pto.tile.load(b1_view, l1_b, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, base_k, n])
        pto.mte_gm_l1(bias_ptr, l1_bias, bias_burst, nburst=(1, 0, 0))

        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
        pto.mte_l1_l0a(l1_a.as_ptr(), a_l0a.as_ptr(), m, base_k)
        pto.mte_l1_l0b(l1_b.as_ptr(), b_l0b.as_ptr(), base_k, n, transpose=True)
        pto.mte_l1_bt(l1_bias, bias_tile.as_ptr(), n, nburst=(1, 0, 0))

        pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
        pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
        pto.mad_bias(
            a_l0a.as_ptr(), b_l0b.as_ptr(), c_acc.as_ptr(), bias_tile.as_ptr(),
            m, n, base_k, disable_gemv=True,
        )

        pto.set_flag(pto.Pipe.M, pto.Pipe.MTE2, event_id=0)
        pto.wait_flag(pto.Pipe.M, pto.Pipe.MTE2, event_id=0)

        # ---- Pass 1: A[:, BASEK:K_use] * B[BASEK:K_use, :] (accumulate) ----
        pto.tile.load(a2_view, l1_a, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, m, base_k])
        pto.tile.load(b2_view, l1_b, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, base_k, n])

        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
        pto.mte_l1_l0a(l1_a.as_ptr(), a_l0a.as_ptr(), m, base_k)
        pto.mte_l1_l0b(l1_b.as_ptr(), b_l0b.as_ptr(), base_k, n, transpose=True)

        pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)
        pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)
        pto.mad_acc(
            a_l0a.as_ptr(), b_l0b.as_ptr(), c_acc.as_ptr(),
            m, n, base_k, disable_gemv=True,
        )

        pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
        pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
        c_view = pto.make_tensor_view(
            c_ptr, shape=[1, 1, 1, m, n], strides=[m * n, m * n, m * n, n, 1]
        )
        pto.tile.store(c_acc, c_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, m, n])
        pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


def _build_kernel(spec):
    if spec.get("split_k"):
        return _build_split_kernel(spec)
    return _build_single_pass_kernel(spec)


# ---------------------------------------------------------------------------
# CASES
# ---------------------------------------------------------------------------

KERNELS = {spec["name"]: _build_kernel(spec) for spec in CASE_SPECS}

CASES = [
    golden_output_case(
        "tmatmul_bias_" + spec["name"],
        KERNELS[spec["name"]],
        inputs=lambda spec=spec: _make_inputs(spec),
        expected=_make_expected(spec),
        rtol=spec["eps"],
        atol=spec["eps"],
    )
    for spec in CASE_SPECS
]


auto_main(globals())
