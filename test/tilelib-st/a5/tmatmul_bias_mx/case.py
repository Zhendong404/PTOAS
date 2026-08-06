#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL port of the legacy tmatmul_bias_mx ST suite (MXFP matmul + bias).

Each kernel computes C[M, N] = A[M, K] x B[K, N] + bias[N] with MX block
scales on both operands and the bias added inside the cube op (a bias tile
staged from L1 through the ``bt`` pointer space, consumed by
``pto.tile.matmul_mx_bias``).  This is a cube op with explicit movement
(``mte_gm_l1``/``mte_gm_l1_frac`` -> L1, ``mte_l1_l0a/l0b`` + ``_mx`` scale
moves -> L0A/L0B, ``mte_l1_bt`` bias -> BT, ``mte_l0c_gm`` nz2nd -> GM), so
the kernels use PTODSL explicit mode with the exact tile addresses, layouts,
valid shapes, burst/loop counts and set_flag/wait_flag event ids from the
legacy ``tmatmul_bias_mx.pto``.

Fidelity notes (mirrored 1:1 from the legacy suite):

- The legacy ``cases.py`` table (dtype pair, M/K/N, M_padded/N_padded,
  is_fp4, split_m_physical_rows, eps) is kept verbatim below in
  ``CASE_SPECS``.
- Legacy L1 tile addresses are preserved exactly per case (a_data@0 with
  a_scale/b_data/b_scale/bias offsets from ``tmatmul_bias_mx.pto``).
- ``mte_gm_l1`` burst lengths (bytes), ``mte_gm_l1_frac`` nd2nz shapes/src
  strides/dst groups, ``mte_l1_l0a/l0b`` valid shapes, ``mte_l1_bt`` element
  counts, and ``mte_l0c_gm`` strides match the legacy kernel constants.
- The split-M case (``bias_fp8_e4m3_200x192x95``) keeps the legacy two-chunk
  kernel: chunk 0 covers physical rows [0,128) (valid 128), chunk 1 covers
  physical rows [128,208) (valid 72), with independently packed A chunks and
  A-scale chunks matching the legacy launch ABI.
- Event ids are preserved per chunk: MTE2->MTE1=0, MTE1->M=0, M->FIX=1, with
  a PIPE_ALL barrier between the two M chunks.
- Host inputs are the legacy packed/zero-padded buffers: MX-fractal packed A/B
  (raw uint8 storage), converted scale buffers, and the bias padded to
  ``ub_bias_cols(n_padded) = ceil_align(n_padded, 64)`` floats (exactly the
  legacy ``gen_data.py`` ``ub_bias_cols`` sizing).
- The golden is the legacy ``gen_golden`` result: dequantized float64
  matmul + bias, cast to f32 and zero-padded to ``[m_padded, n_padded]``
  (the legacy compare only inspected the ``[:m, :n]`` region).
- Data is regenerated deterministically per case with the legacy per-case
  seed ``zlib.crc32(case_name) & 0xFFFFFFFF`` (``st_common.setup_case_rng``).
- ``rtol``/``atol`` equal the legacy ``eps`` (legacy ``result_cmp`` used
  ``np.allclose(..., atol=eps, rtol=eps)``).
"""

from pathlib import Path
import sys
import zlib

import numpy as np
import ml_dtypes
import en_dtypes

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto

fp8_e4m3fn = ml_dtypes.float8_e4m3fn
fp8_e5m2 = ml_dtypes.float8_e5m2
fp4_e1m2x2 = en_dtypes.float4_e1m2
fp4_e2m1x2 = en_dtypes.float4_e2m1

# ---------------------------------------------------------------------------
# Legacy cases.py table (single source of truth, kept verbatim) + per-case
# L1 tile addresses / burst-loop counts from tmatmul_bias_mx.pto.
# ---------------------------------------------------------------------------

CASE_SPECS = [
    {
        "name": "bias_fp8_e5m2_e4m3_115x64x30",
        "atype": fp8_e5m2, "btype": fp8_e4m3fn,
        "a_pto": pto.f8e5m2, "b_pto": pto.f8e4m3,
        "m": 115, "k": 64, "n": 30,
        "m_padded": 128, "n_padded": 32,
        "is_fp4": False, "eps": 1e-3,
        "l1_a": 0, "l1_a_scale": 8192, "l1_b": 8448, "l1_b_scale": 10496, "l1_bias": 10624,
        "a_loops": [(8, 1024, 1024)],
        "a_scale_loops": [(4, 64, 64)],
        "b_loops": [(2, 1024, 1024)],
        "b_scale_loops": None,
        "bias_loops": [(2, 64, 64)],
    },
    {
        "name": "bias_fp8_e4m3_200x192x95",
        "atype": fp8_e4m3fn, "btype": fp8_e4m3fn,
        "a_pto": pto.f8e4m3, "b_pto": pto.f8e4m3,
        "m": 200, "k": 192, "n": 95,
        "m_padded": 208, "n_padded": 128,
        "is_fp4": False, "split_m_physical_rows": [128, 80], "eps": 1e-3,
        "l1_a": 0, "l1_a_scale": 38912, "l1_b": 40192, "l1_b_scale": 58624, "l1_bias": 59392,
        "a_loops": [(24, 1024, 1024)],
        "a_scale_loops": [(24, 32, 32)],
        "b_loops": [(18, 1024, 1024)],
        "b_scale_loops": [(12, 64, 64)],
        "bias_loops": [(6, 64, 64)],
        "a1_loops": [(15, 1024, 1024)],
        "a1_scale_loops": [(15, 32, 32)],
    },
    {
        "name": "bias_fp4_e2m1_e1m2_35x128x56",
        "atype": fp4_e2m1x2, "btype": fp4_e1m2x2,
        "a_pto": pto.f4e2m1x2, "b_pto": pto.f4e1m2x2,
        "m": 35, "k": 128, "n": 56,
        "m_padded": 48, "n_padded": 64,
        "is_fp4": True, "eps": 1e-3,
        "l1_a": 0, "l1_a_scale": 3072, "l1_b": 3264, "l1_b_scale": 7360, "l1_bias": 7616,
        "a_frac": {"shape": (48, 64), "src_layout": (64,), "dst_group": (1, 1, 48, 0)},
        "a_scale_loops": [(3, 64, 64)],
        "b_frac": {"shape": (128, 32), "src_layout": (32,), "dst_group": (1, 1, 128, 0)},
        "b_scale_loops": [(4, 64, 64)],
        "bias_loops": [(4, 64, 64)],
    },
    {
        "name": "bias_fp4_e1m2_47x128x62",
        "atype": fp4_e1m2x2, "btype": fp4_e1m2x2,
        "a_pto": pto.f4e1m2x2, "b_pto": pto.f4e1m2x2,
        "m": 47, "k": 128, "n": 62,
        "m_padded": 48, "n_padded": 64,
        "is_fp4": True, "eps": 1e-3,
        "l1_a": 0, "l1_a_scale": 3072, "l1_b": 3264, "l1_b_scale": 7360, "l1_bias": 7616,
        "a_frac": {"shape": (48, 64), "src_layout": (64,), "dst_group": (1, 1, 48, 0)},
        "a_scale_loops": [(3, 64, 64)],
        "b_frac": {"shape": (128, 32), "src_layout": (32,), "dst_group": (1, 1, 128, 0)},
        "b_scale_loops": [(4, 64, 64)],
        "bias_loops": [(4, 64, 64)],
    },
    {
        "name": "bias_fp8_e4m3_e5m2_64x192x64",
        "atype": fp8_e4m3fn, "btype": fp8_e5m2,
        "a_pto": pto.f8e4m3, "b_pto": pto.f8e5m2,
        "m": 64, "k": 192, "n": 64,
        "m_padded": 64, "n_padded": 64,
        "is_fp4": False, "eps": 1e-3,
        "l1_a": 0, "l1_a_scale": 12288, "l1_b": 12672, "l1_b_scale": 24960, "l1_bias": 25344,
        "a_loops": [(12, 1024, 1024)],
        "a_scale_loops": [(6, 64, 64)],
        "b_loops": [(12, 1024, 1024)],
        "b_scale_loops": [(6, 64, 64)],
        "bias_loops": [(4, 64, 64)],
    },
]

# ---------------------------------------------------------------------------
# Legacy gen_data.py helpers (reproduced unchanged)
# ---------------------------------------------------------------------------


def pack_two_fp4(matrix):
    row, col = matrix.shape
    flat = matrix.flatten()
    high = flat[::2].view(np.uint8)
    low = flat[1::2].view(np.uint8)
    low_bits = (low & 0x0F) << 4
    high_bits = high & 0x0F
    combined = low_bits | high_bits
    return combined.reshape(row, col // 2)


def ceil_align(num, align):
    return (num + align - 1) // align * align


def ceil_div(num, div):
    return (num + div - 1) // div


def ub_bias_cols(n_padded):
    return ceil_align(n_padded, 64)


def pack_mx_lhs_fp8_fractal(matrix):
    m, k = matrix.shape
    if m >= 32 and m % 32 == 0:
        packed = matrix.reshape(m // 32, 32, k // 32, 32).transpose(2, 0, 1, 3)
    else:
        packed = matrix.reshape(m, k // 32, 32).transpose(1, 0, 2)
    return np.ascontiguousarray(packed)


def pack_mx_lhs_fp8_fractal_chunks(matrix, chunk_physical_rows):
    packed_chunks = []
    row = 0
    for rows in chunk_physical_rows:
        chunk = matrix[row:row + rows, :]
        if chunk.shape[0] != rows:
            raise ValueError(f"invalid split_m_physical_rows {chunk_physical_rows} for M={matrix.shape[0]}")
        packed_chunks.append(pack_mx_lhs_fp8_fractal(chunk).reshape(-1))
        row += rows
    if row != matrix.shape[0]:
        raise ValueError(f"invalid split_m_physical_rows {chunk_physical_rows} for M={matrix.shape[0]}")
    return np.ascontiguousarray(np.concatenate(packed_chunks))


def pack_mx_rhs_fp8_fractal(matrix):
    k, n = matrix.shape
    packed = matrix.reshape(k // 16, 16, n // 32, 32).transpose(2, 0, 1, 3)
    return np.ascontiguousarray(packed)


def convert_scale_a_format(scale, block_size=16, c0_size_mx=2):
    m, k = scale.shape
    pad_m = (block_size - m % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_m > 0 or pad_k > 0:
        padded = np.pad(scale, ((0, pad_m), (0, pad_k)), mode='constant', constant_values=0)
    else:
        padded = scale
    m_padded = m + pad_m
    k_padded = k + pad_k
    result = padded.reshape((int(m_padded / block_size), block_size, int(k_padded / c0_size_mx), c0_size_mx))
    result = result.transpose(0, 2, 1, 3)
    result = result.reshape(result.shape[0] * result.shape[2], result.shape[1] * result.shape[3])
    return result


def convert_scale_b_format(scale, block_size=16, c0_size_mx=2, n_pad_to=None):
    k, n = scale.shape
    # RHS MX scale is packed in 16-column groups even when logical N is not
    # 16-aligned, so pad the physical column extent before reshaping.
    target_n = n if n_pad_to is None else max(n, n_pad_to)
    target_n = ceil_align(target_n, block_size)
    pad_n = target_n - n
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_n > 0 or pad_k > 0:
        padded = np.pad(scale, ((0, pad_k), (0, pad_n)), mode='constant', constant_values=0)
    else:
        padded = scale
    k_padded, n_padded = padded.shape
    result = padded.reshape((int(k_padded / c0_size_mx), c0_size_mx, int(n_padded / 16), 16)).transpose(2, 0, 3, 1)
    result = result.reshape(result.shape[1] * result.shape[3], result.shape[0] * result.shape[2])
    return result


def gen_golden(case):
    """Legacy ``gen_data.gen_golden``; returns packed bins + converted scales +
    bias (padded to ub_bias_cols(n_padded)) + the [m_padded, n_padded] golden."""
    atype = case["atype"]
    btype = case["btype"]
    m, k, n = case["m"], case["k"], case["n"]
    m_padded = case["m_padded"]
    n_padded = case["n_padded"]
    is_bias = case.get("is_bias", True)
    is_fp4 = case["is_fp4"]

    k_aligned = ceil_align(k, 64)

    if atype == fp4_e2m1x2:
        x1 = np.random.randint(-6, 6, [m, k]).astype(atype)
    elif atype == fp4_e1m2x2:
        x1 = np.random.randint(-1, 2, [m, k]).astype(atype)
    else:
        x1 = np.random.randint(-10, 10, [m, k]).astype(atype)

    if btype == fp4_e2m1x2:
        x2 = np.random.randint(-6, 6, [k, n]).astype(btype)
    elif btype == fp4_e1m2x2:
        x2 = np.random.randint(-1, 2, [k, n]).astype(btype)
    else:
        x2 = np.random.randint(-10, 10, [k, n]).astype(btype)

    x1_padded = np.zeros([m_padded, k_aligned], dtype=atype)
    x1_padded[:m, :k] = x1
    x2_padded = np.zeros([k_aligned, n_padded], dtype=btype)
    x2_padded[:k, :n] = x2

    if is_fp4:
        x1_bin = pack_two_fp4(x1_padded)
        x2_bin = pack_two_fp4(x2_padded)
    else:
        if case.get("split_m_physical_rows") is not None:
            x1_bin = pack_mx_lhs_fp8_fractal_chunks(x1_padded, case["split_m_physical_rows"])
        else:
            x1_bin = pack_mx_lhs_fp8_fractal(x1_padded)
        x2_bin = pack_mx_rhs_fp8_fractal(x2_padded)

    x1_scale = np.random.randint(127, 130, [m, ceil_div(k_aligned, 32)]).astype(np.uint8)
    x2_scale = np.random.randint(127, 130, [ceil_div(k_aligned, 32), n]).astype(np.uint8)

    x1_mx = 2 ** (x1_scale.astype(np.float64) - 127)
    x2_mx = 2 ** (x2_scale.astype(np.float64) - 127)

    x1_full = np.zeros([m, k_aligned], dtype=np.float64)
    x2_full = np.zeros([k_aligned, n], dtype=np.float64)

    for i in range(k):
        x1_full[:, i] = x1[:, i] * x1_mx[:, i // 32]
        x2_full[i, :] = x2[i, :] * x2_mx[i // 32, :]

    x1_float = x1_full[:, :k]
    x2_float = x2_full[:k, :]

    # For the split launcher, scale_a1 is a byte offset into the legacy
    # scale stream.  The source stream is still generated by the legacy
    # formatter; flattening is deferred to the split ABI slicing below.
    x1_scale_gm = convert_scale_a_format(x1_scale, 16, 2)
    x2_scale_gm = convert_scale_b_format(x2_scale, 16, 2, n_pad_to=n_padded)

    if is_bias:
        bias = np.random.randint(1, 10, [n]).astype(np.float32)
        golden_valid = np.matmul(x1_float, x2_float).astype(np.float32) + bias
        golden = np.zeros([m_padded, n_padded], dtype=np.float32)
        golden[:m, :n] = golden_valid
        bias_padded = np.zeros([ub_bias_cols(n_padded)], dtype=np.float32)
        bias_padded[:n] = bias
        bias = bias_padded
    else:
        golden_valid = np.matmul(x1_float, x2_float).astype(np.float32)
        golden = np.zeros([m_padded, n_padded], dtype=np.float32)
        golden[:m, :n] = golden_valid

    return x1_bin, x2_bin, x1_scale_gm, x2_scale_gm, bias if is_bias else None, golden


# ---------------------------------------------------------------------------
# Host data (mirrors legacy gen_data.py + st_common.setup_case_rng)
# ---------------------------------------------------------------------------


def _make_data(spec):
    """Deterministically regenerate the legacy packed inputs and golden."""
    np.random.seed(zlib.crc32(spec["name"].encode("utf-8")) & 0xFFFFFFFF)
    x1_bin, x2_bin, x1_scale_gm, x2_scale_gm, bias, golden = gen_golden(spec)
    # The legacy launcher ABI is uint8_t* for the packed FP8/scale buffers;
    # materialize the same raw bytes so torch can transfer them to the device.
    if not spec["is_fp4"]:
        x1_bin = x1_bin.view(np.uint8)
        x2_bin = x2_bin.view(np.uint8)
    inputs = [x1_bin, x2_bin, x1_scale_gm, x2_scale_gm, bias]
    return inputs, golden


def _make_split_m_data(spec):
    """Split-M variant with the legacy independent-chunk ABI."""
    np.random.seed(zlib.crc32(spec["name"].encode("utf-8")) & 0xFFFFFFFF)
    x1_bin, x2_bin, x1_scale_gm, x2_scale_gm, bias, golden = gen_golden(spec)
    a0_bytes = 128 * 192
    a1_bytes = 80 * 192
    x1_bytes = np.ascontiguousarray(x1_bin).view(np.uint8).reshape(-1)
    x1_scale_bytes = np.ascontiguousarray(x1_scale_gm).view(np.uint8).reshape(-1)
    return [
        x1_bytes[:a0_bytes], x2_bin.view(np.uint8),
        x1_scale_bytes[:768], x2_scale_gm, bias,
        x1_bytes[a0_bytes:a0_bytes + a1_bytes], x1_scale_bytes[768:],
    ], golden


# ---------------------------------------------------------------------------
# Kernels (cube / explicit mode, movement mirrored from legacy tmatmul_bias_mx.pto)
# ---------------------------------------------------------------------------


def _build_single_m_kernel(spec):
    """Single-M-chunk pto.tile.matmul_mx_bias kernels (cases 1, 3, 4, 5)."""
    m = spec["m"]
    n = spec["n"]
    m_padded = spec["m_padded"]
    n_padded = spec["n_padded"]
    k_aligned = ceil_align(spec["k"], 64)
    a_pto = spec["a_pto"]
    b_pto = spec["b_pto"]
    scale_groups = k_aligned // 32

    l1_a = spec["l1_a"]
    l1_a_scale = spec["l1_a_scale"]
    l1_b = spec["l1_b"]
    l1_b_scale = spec["l1_b_scale"]
    l1_bias = spec["l1_bias"]
    a_frac = spec.get("a_frac")
    b_frac = spec.get("b_frac")
    a_loops = spec.get("a_loops")
    b_loops = spec.get("b_loops")
    kernel_name = spec["name"]

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
        a_scale_ptr: pto.ptr(a_pto, "gm"),
        b_scale_ptr: pto.ptr(b_pto, "gm"),
        bias_ptr: pto.ptr(pto.f32, "gm"),
        c_ptr: pto.ptr(pto.f32, "gm"),
    ):
        l1_a_data = pto.castptr(pto.ui64(l1_a), pto.ptr(a_pto, "mat"))
        l1_a_scale_buf = pto.castptr(pto.ui64(l1_a_scale), pto.ptr(a_pto, "mat"))
        l1_b_data = pto.castptr(pto.ui64(l1_b), pto.ptr(b_pto, "mat"))
        l1_b_scale_buf = pto.castptr(pto.ui64(l1_b_scale), pto.ptr(b_pto, "mat"))
        l1_bias_buf = pto.castptr(pto.ui64(l1_bias), pto.ptr(pto.f32, "mat"))

        lhs = pto.alloc_tile(
            shape=[m_padded, k_aligned], dtype=a_pto, memory_space=pto.MemorySpace.LEFT,
            addr=0, valid_shape=[m, k_aligned], blayout="ColMajor", slayout="RowMajor", fractal_size=512,
        )
        lhs_scale = pto.alloc_tile(
            shape=[m_padded, scale_groups], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
            addr=0, valid_shape=[m, scale_groups], blayout="RowMajor", slayout="RowMajor", fractal_size=32,
        )
        rhs = pto.alloc_tile(
            shape=[k_aligned, n_padded], dtype=b_pto, memory_space=pto.MemorySpace.RIGHT,
            addr=0, valid_shape=[k_aligned, n], blayout="RowMajor", slayout="ColMajor", fractal_size=512,
        )
        rhs_scale = pto.alloc_tile(
            shape=[scale_groups, n_padded], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
            addr=0, valid_shape=[scale_groups, n], blayout="ColMajor", slayout="ColMajor", fractal_size=32,
        )
        bias_tile = pto.alloc_tile(
            shape=[1, n_padded], dtype=pto.f32, memory_space=pto.MemorySpace.BIAS,
            addr=0, valid_shape=[1, n], blayout="RowMajor", slayout="NoneBox", fractal_size=512,
        )
        dst = pto.alloc_tile(
            shape=[m_padded, n_padded], dtype=pto.f32, memory_space=pto.MemorySpace.ACC,
            addr=0, valid_shape=[m, n], blayout="ColMajor", slayout="RowMajor", fractal_size=512,
        )

        if a_frac is not None:
            pto.mte_gm_l1_frac(
                a_ptr, l1_a_data, pto.FractalMode.ND2NZ,
                shape=a_frac["shape"], src_layout=a_frac["src_layout"],
                dst_group=a_frac["dst_group"], ctrl=(0, False),
            )
        else:
            pto.mte_gm_l1(a_ptr, l1_a_data, 1024, nburst=(1, 0, 0), loops=a_loops)
        pto.mte_gm_l1(a_scale_ptr, l1_a_scale_buf, 64, nburst=(1, 0, 0), loops=spec["a_scale_loops"])
        if b_frac is not None:
            pto.mte_gm_l1_frac(
                b_ptr, l1_b_data, pto.FractalMode.ND2NZ,
                shape=b_frac["shape"], src_layout=b_frac["src_layout"],
                dst_group=b_frac["dst_group"], ctrl=(0, False),
            )
        else:
            pto.mte_gm_l1(b_ptr, l1_b_data, 1024, nburst=(1, 0, 0), loops=b_loops)
        if spec["b_scale_loops"] is not None:
            pto.mte_gm_l1(b_scale_ptr, l1_b_scale_buf, 64, nburst=(1, 0, 0), loops=spec["b_scale_loops"])
        else:
            pto.mte_gm_l1(b_scale_ptr, l1_b_scale_buf, 64, nburst=(1, 0, 0))
        pto.mte_gm_l1(bias_ptr, l1_bias_buf, 64, nburst=(1, 0, 0), loops=spec["bias_loops"])

        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)

        pto.mte_l1_l0a(l1_a_data, lhs.as_ptr(), m, k_aligned)
        pto.mte_l1_l0b(l1_b_data, rhs.as_ptr(), k_aligned, n, transpose=True)
        pto.mte_l1_l0a_mx(l1_a_scale_buf, lhs.as_ptr(), m, k_aligned)
        pto.mte_l1_l0b_mx(l1_b_scale_buf, rhs.as_ptr(), k_aligned, n)
        pto.mte_l1_bt(l1_bias_buf, bias_tile.as_ptr(), n, nburst=(1, 0, 0))

        pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
        pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)

        pto.tile.matmul_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias_tile, dst)

        pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
        pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)

        pto.mte_l0c_gm(dst.as_ptr(), c_ptr, m, n, m_padded, n_padded, 0, 0, layout="nz2nd")
        pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


def _build_split_m_kernel(spec):
    """Split-M pto.tile.matmul_mx_bias kernel (case 2): two M chunks writing
    into two separate GM output buffers exactly like the legacy launcher."""
    a_pto = spec["a_pto"]
    b_pto = spec["b_pto"]
    kernel_name = spec["name"]

    @pto.jit(
        name=kernel_name,
        kernel_kind="cube",
        target="a5",
        mode="explicit",
        insert_sync=False,
    )
    def kernel(
        a0_ptr: pto.ptr(a_pto, "gm"),
        b_ptr: pto.ptr(b_pto, "gm"),
        a0_scale_ptr: pto.ptr(a_pto, "gm"),
        b_scale_ptr: pto.ptr(b_pto, "gm"),
        bias_ptr: pto.ptr(pto.f32, "gm"),
        c0_ptr: pto.ptr(pto.f32, "gm"),
        a1_ptr: pto.ptr(a_pto, "gm"),
        a1_scale_ptr: pto.ptr(a_pto, "gm"),
        c1_ptr: pto.ptr(pto.f32, "gm"),
    ):
        l1_a_data = pto.castptr(pto.ui64(spec["l1_a"]), pto.ptr(a_pto, "mat"))
        l1_a_scale_buf = pto.castptr(pto.ui64(spec["l1_a_scale"]), pto.ptr(a_pto, "mat"))
        l1_b_data = pto.castptr(pto.ui64(spec["l1_b"]), pto.ptr(b_pto, "mat"))
        l1_b_scale_buf = pto.castptr(pto.ui64(spec["l1_b_scale"]), pto.ptr(b_pto, "mat"))
        l1_bias_buf = pto.castptr(pto.ui64(spec["l1_bias"]), pto.ptr(pto.f32, "mat"))

        # ---- Chunk 0: physical rows [0, 128) ----
        lhs = pto.alloc_tile(
            shape=[128, 192], dtype=a_pto, memory_space=pto.MemorySpace.LEFT,
            addr=0, valid_shape=[128, 192], blayout="ColMajor", slayout="RowMajor", fractal_size=512,
        )
        lhs_scale = pto.alloc_tile(
            shape=[128, 6], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
            addr=0, valid_shape=[128, 6], blayout="RowMajor", slayout="RowMajor", fractal_size=32,
        )
        rhs = pto.alloc_tile(
            shape=[192, 128], dtype=b_pto, memory_space=pto.MemorySpace.RIGHT,
            addr=0, valid_shape=[192, 95], blayout="RowMajor", slayout="ColMajor", fractal_size=512,
        )
        rhs_scale = pto.alloc_tile(
            shape=[6, 128], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
            addr=0, valid_shape=[6, 95], blayout="ColMajor", slayout="ColMajor", fractal_size=32,
        )
        bias_tile = pto.alloc_tile(
            shape=[1, 128], dtype=pto.f32, memory_space=pto.MemorySpace.BIAS,
            addr=0, valid_shape=[1, 95], blayout="RowMajor", slayout="NoneBox", fractal_size=512,
        )
        dst = pto.alloc_tile(
            shape=[128, 128], dtype=pto.f32, memory_space=pto.MemorySpace.ACC,
            addr=0, valid_shape=[128, 95], blayout="ColMajor", slayout="RowMajor", fractal_size=512,
        )
        # ---- Chunk 1: physical rows [128, 208), valid 72 ----
        lhs_tail = pto.alloc_tile(
            shape=[80, 192], dtype=a_pto, memory_space=pto.MemorySpace.LEFT,
            addr=0, valid_shape=[72, 192], blayout="ColMajor", slayout="RowMajor", fractal_size=512,
        )
        lhs_scale_tail = pto.alloc_tile(
            shape=[80, 6], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
            addr=0, valid_shape=[72, 6], blayout="RowMajor", slayout="RowMajor", fractal_size=32,
        )
        dst_tail = pto.alloc_tile(
            shape=[80, 128], dtype=pto.f32, memory_space=pto.MemorySpace.ACC,
            addr=0, valid_shape=[72, 95], blayout="ColMajor", slayout="RowMajor", fractal_size=512,
        )

        # ---- Chunk 0 ----
        pto.mte_gm_l1(a0_ptr, l1_a_data, 1024, nburst=(1, 0, 0), loops=spec["a_loops"])
        pto.mte_gm_l1(a0_scale_ptr, l1_a_scale_buf, 32, nburst=(1, 0, 0), loops=spec["a_scale_loops"])
        pto.mte_gm_l1(b_ptr, l1_b_data, 1024, nburst=(1, 0, 0), loops=spec["b_loops"])
        pto.mte_gm_l1(b_scale_ptr, l1_b_scale_buf, 64, nburst=(1, 0, 0), loops=spec["b_scale_loops"])
        pto.mte_gm_l1(bias_ptr, l1_bias_buf, 64, nburst=(1, 0, 0), loops=spec["bias_loops"])

        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)

        pto.mte_l1_l0a(l1_a_data, lhs.as_ptr(), 128, 192)
        pto.mte_l1_l0b(l1_b_data, rhs.as_ptr(), 192, 95, transpose=True)
        pto.mte_l1_l0a_mx(l1_a_scale_buf, lhs.as_ptr(), 128, 192)
        pto.mte_l1_l0b_mx(l1_b_scale_buf, rhs.as_ptr(), 192, 95)
        pto.mte_l1_bt(l1_bias_buf, bias_tile.as_ptr(), 96, nburst=(1, 0, 0))

        pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
        pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)

        pto.tile.matmul_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias_tile, dst)

        pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
        pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)

        pto.mte_l0c_gm(dst.as_ptr(), c0_ptr, 128, 95, 128, 128, 0, 0, layout="nz2nd")
        pto.pipe_barrier(pto.Pipe.ALL)

        # ---- Chunk 1 (reuses L1 B/scale/bias staging from chunk 0) ----
        pto.mte_gm_l1(a1_ptr, l1_a_data, 1024, nburst=(1, 0, 0), loops=spec["a1_loops"])
        pto.mte_gm_l1(a1_scale_ptr, l1_a_scale_buf, 32, nburst=(1, 0, 0), loops=spec["a1_scale_loops"])

        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)

        pto.mte_l1_l0a(l1_a_data, lhs_tail.as_ptr(), 72, 192)
        pto.mte_l1_l0b(l1_b_data, rhs.as_ptr(), 192, 95, transpose=True)
        pto.mte_l1_l0a_mx(l1_a_scale_buf, lhs_tail.as_ptr(), 72, 192)
        pto.mte_l1_l0b_mx(l1_b_scale_buf, rhs.as_ptr(), 192, 95)
        pto.mte_l1_bt(l1_bias_buf, bias_tile.as_ptr(), 96, nburst=(1, 0, 0))

        pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
        pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)

        pto.tile.matmul_mx_bias(lhs_tail, lhs_scale_tail, rhs, rhs_scale, bias_tile, dst_tail)

        pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
        pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)

        pto.mte_l0c_gm(dst_tail.as_ptr(), c1_ptr, 72, 95, 80, 128, 0, 0, layout="nz2nd")
        pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


# ---------------------------------------------------------------------------
# CASES
# ---------------------------------------------------------------------------


def _split_m_case(spec):
    """Case builder for the split-M kernel's independent chunk ABI."""
    kernel = _build_split_m_kernel(spec)

    def make_case():
        inputs, golden = _make_split_m_data(spec)
        c0 = np.zeros((128, 128), dtype=np.float32)
        c1 = np.zeros((80, 128), dtype=np.float32)
        # Keep the host argument order identical to the split kernel ABI:
        # a0, b, a0_scale, b_scale, bias, c0, a1, a1_scale, c1.
        a0, b, a0_scale, b_scale, bias, a1, a1_scale = inputs
        return [a0, b, a0_scale, b_scale, bias, c0, a1, a1_scale, c1], golden

    def check(device_inputs, golden):
        c0 = np.asarray(device_inputs[5].cpu().numpy())
        c1 = np.asarray(device_inputs[8].cpu().numpy())
        actual = np.concatenate([c0, c1], axis=0)
        np.testing.assert_allclose(actual, golden, rtol=spec["eps"], atol=spec["eps"])

    return {"name": spec["name"], "kernel": kernel, "make_case": make_case, "check": check}


CASES = []
for spec in CASE_SPECS:
    if spec.get("split_m_physical_rows") is not None:
        CASES.append(_split_m_case(spec))
        continue
    kernel = _build_single_m_kernel(spec)
    CASES.append(
        golden_output_case(
            spec["name"],
            kernel,
            inputs=lambda spec=spec: _make_data(spec)[0],
            expected=lambda *_, spec=spec: _make_data(spec)[1],
            rtol=spec["eps"],
            atol=spec["eps"],
        )
    )


auto_main(globals())
