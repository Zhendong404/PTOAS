#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL port of the legacy tgemv_mx ST suite.

Each kernel computes C[1,N] = A[1,K] * scale_A x B[K,N] * scale_B, with an
optional bias: C[1,N] += bias[N].  The GEMV mode is selected by the semantic
``pto.tgemv.mx`` family, so these are cube kernels with explicit movement
(GM->L1, L1->L0A/L0B, MX scale staging, L0C->GM) exactly as the legacy
``tgemv_mx.pto`` was authored.

Fidelity notes (mirrored 1:1 from the legacy suite):

- dtype / shapes / valid shapes / scale layouts / eps are taken verbatim from
  the legacy ``cases.py`` table.
- Host inputs are the packed kernels' bins produced by the legacy
  ``gen_data.py`` (``pack_two_fp4`` / ``convert_scale_*`` helpers reproduced
  unchanged), and the golden is the legacy ``gen_golden`` result.  Data is
  regenerated deterministically per case with the legacy per-case seed
  ``zlib.crc32(case_name) & 0xFFFFFFFF`` (see ``st_common.setup_case_rng``).
- The bias host input is padded to ``n_padded`` with zeros, matching the
  effective device buffer the legacy C++ launcher built (a zeroed
  ``biasBytes = n_padded * 4`` buffer filled from a 62-element bin).
- ``pad=0`` on every legacy tile maps to the PTODSL default ``pad="Null"``.
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
# Legacy cases.py table (single source of truth, kept byte-for-byte)
# ---------------------------------------------------------------------------

CASE_SPECS = [
    {"name": "gemv_mx_fp4_e1m2_1x128x62", "atype": fp4_e1m2x2, "btype": fp4_e1m2x2, "m": 1, "k": 128, "n": 62, "m_padded": 16, "n_storage": 64, "n_padded": 64, "is_bias": False, "is_fp4": True, "is_split_k": False, "eps": 1e-3},
    {"name": "gemv_mx_fp8_e4m3_e5m2_1x256x20", "atype": fp8_e4m3fn, "btype": fp8_e5m2, "m": 1, "k": 256, "n": 20, "m_padded": 16, "n_storage": 32, "n_padded": 32, "is_bias": False, "is_fp4": False, "is_split_k": False, "eps": 1e-3},
    {"name": "gemv_mx_bias_fp4_e1m2_1x64x62", "atype": fp4_e1m2x2, "btype": fp4_e1m2x2, "m": 1, "k": 64, "n": 62, "m_padded": 16, "n_storage": 64, "n_padded": 64, "is_bias": True, "is_fp4": True, "is_split_k": False, "eps": 1e-3},
    {"name": "gemv_mx_bias_fp4_e1m2_1x2048x64", "atype": fp4_e1m2x2, "btype": fp4_e1m2x2, "m": 1, "k": 2048, "n": 64, "m_padded": 16, "n_storage": 64, "n_padded": 64, "is_bias": True, "is_fp4": True, "is_split_k": True, "eps": 1e-3},
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


def convert_scale_b_format(scale, block_size=16, c0_size_mx=2):
    k, n = scale.shape
    pad_n = (block_size - n % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_n > 0 or pad_k > 0:
        padded = np.pad(scale, ((0, pad_k), (0, pad_n)), mode='constant', constant_values=0)
    else:
        padded = scale
    k_padded, n_padded = padded.shape
    result = padded.reshape((int(k_padded / c0_size_mx), c0_size_mx, int(n_padded / 16), 16)).transpose(2, 0, 3, 1)
    result = result.reshape(result.shape[1] * result.shape[3], result.shape[0] * result.shape[2])
    return result


def convert_scale_a_row_major_padded(scale, block_size=16, c0_size_mx=2):
    m, k = scale.shape
    pad_m = (block_size - m % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_m > 0 or pad_k > 0:
        return np.pad(scale, ((0, pad_m), (0, pad_k)), mode='constant', constant_values=0)
    return scale.copy()


def convert_scale_b_nd_padded(scale, block_size=16, c0_size_mx=2):
    k, n = scale.shape
    pad_n = (block_size - n % block_size) % block_size
    pad_k = (c0_size_mx - k % c0_size_mx) % c0_size_mx
    if pad_n > 0 or pad_k > 0:
        scale = np.pad(scale, ((0, pad_k), (0, pad_n)), mode='constant', constant_values=0)
    return scale.reshape((scale.shape[0] // c0_size_mx, c0_size_mx, scale.shape[1])).transpose(0, 2, 1).copy()


def gen_golden(case):
    """Legacy ``gen_data.gen_golden``; returns packed bins + converted scales +
    bias (padded to n_padded) + the [m_padded, n_padded] golden."""
    atype = case["atype"]
    btype = case["btype"]
    m, k, n = case["m"], case["k"], case["n"]
    m_padded = case["m_padded"]
    n_storage = case["n_storage"]
    n_padded = case["n_padded"]
    is_bias = case["is_bias"]
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

    if is_fp4:
        x1_padded = np.zeros([m_padded, k_aligned], dtype=atype)
        x1_padded[:m, :k] = x1
        x2_padded = np.zeros([k_aligned, n_storage], dtype=btype)
        x2_padded[:k, :n] = x2
        x1_bin = pack_two_fp4(x1_padded)
        x2_bin = pack_two_fp4(x2_padded)
    else:
        x1_padded = np.zeros([m_padded, k_aligned], dtype=atype)
        x1_padded[:m, :k] = x1
        x2_padded = np.zeros([k_aligned, n_storage], dtype=btype)
        x2_padded[:k, :n] = x2
        x1_bin = x1_padded
        x2_bin = x2_padded

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

    if case["name"] in ("gemv_mx_fp8_e4m3_e5m2_1x256x20", "gemv_mx_bias_fp4_e1m2_1x2048x64"):
        x1_scale_gm = convert_scale_a_row_major_padded(x1_scale, 16, 2)
    else:
        x1_scale_gm = convert_scale_a_format(x1_scale, 16, 2)
    if case["name"] == "gemv_mx_fp4_e1m2_1x128x62":
        x2_scale_gm = convert_scale_b_nd_padded(x2_scale)
    else:
        x2_scale_gm = convert_scale_b_format(x2_scale, 16, 2)

    if is_bias:
        bias = np.random.randint(1, 10, [n]).astype(np.float32)
        # The legacy C++ launcher zero-padded the device bias buffer to
        # n_padded floats (biasBytes = n_padded * 4), so reproduce that here.
        bias_padded = np.zeros([n_padded], dtype=np.float32)
        bias_padded[:n] = bias
        golden_valid = np.matmul(x1_float, x2_float).astype(np.float32) + bias
        golden = np.zeros([m_padded, n_padded], dtype=np.float32)
        golden[:m, :n] = golden_valid
    else:
        golden_valid = np.matmul(x1_float, x2_float).astype(np.float32)
        golden = np.zeros([m_padded, n_padded], dtype=np.float32)
        golden[:m, :n] = golden_valid

    return x1_bin, x2_bin, x1_scale_gm, x2_scale_gm, bias_padded if is_bias else None, golden


def _gen_case_data(spec):
    """Deterministic per-case data, seeded like ``st_common.setup_case_rng``."""
    np.random.seed(zlib.crc32(spec["name"].encode("utf-8")) & 0xFFFFFFFF)
    return gen_golden(spec)


def _case_inputs(spec):
    x1, x2, scale1, scale2, bias, _ = _gen_case_data(spec)
    if spec["is_split_k"]:
        # Split-K kernel consumes two K=1024 halves; the legacy launcher
        # passed a+512, b+32768, scale_a+32, scale_b+512 byte offsets.
        return [
            x1,
            x2,
            scale1,
            scale2,
            x1[:, 512:],
            x2[1024:, :],
            scale1[:, 32:],
            scale2[8:, :],
            bias,
        ]
    if spec["is_bias"]:
        return [x1, x2, scale1, scale2, bias]
    return [x1, x2, scale1, scale2]


# ---------------------------------------------------------------------------
# Kernels (cube / explicit mode, movement mirrored from legacy tgemv_mx.pto)
# ---------------------------------------------------------------------------

# gemv_mx_fp4_e1m2_1x128x62: K=128 as two 64-wide passes (tgemv.mx + acc)
@pto.jit(
    name="gemv_mx_fp4_e1m2_1x128x62",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel_gemv_mx_fp4_e1m2_1x128x62(
    a_ptr: pto.ptr(pto.f4e1m2x2, "gm"),
    b_ptr: pto.ptr(pto.f4e1m2x2, "gm"),
    a_scale_ptr: pto.ptr(pto.f8e4m3, "gm"),
    b_scale_ptr: pto.ptr(pto.f8e4m3, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    a_l1 = pto.castptr(pto.ui64(0), pto.ptr(pto.f4e1m2x2, "mat"))
    a_l1_hi = pto.castptr(pto.ui64(32), pto.ptr(pto.f4e1m2x2, "mat"))
    a_scale_l1 = pto.castptr(pto.ui64(1024), pto.ptr(pto.f8e4m3, "mat"))
    a_scale_l1_hi = pto.castptr(pto.ui64(1056), pto.ptr(pto.f8e4m3, "mat"))
    b_l1 = pto.castptr(pto.ui64(1088), pto.ptr(pto.f4e1m2x2, "mat"))
    b_l1_hi = pto.castptr(pto.ui64(3136), pto.ptr(pto.f4e1m2x2, "mat"))
    b_scale_l1 = pto.castptr(pto.ui64(5184), pto.ptr(pto.f8e4m3, "mat"))
    b_scale_l1_hi = pto.castptr(pto.ui64(5312), pto.ptr(pto.f8e4m3, "mat"))

    lhs_tile = pto.alloc_tile(
        shape=[1, 64], dtype=pto.f4e1m2x2, memory_space=pto.MemorySpace.LEFT, addr=0,
        valid_shape=[1, 64], blayout="ColMajor", slayout="RowMajor", fractal_size=512,
    )
    a_scale_tile = pto.alloc_tile(
        shape=[1, 2], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING, addr=0,
        valid_shape=[1, 2], blayout="RowMajor", slayout="RowMajor", fractal_size=32,
    )
    rhs_tile = pto.alloc_tile(
        shape=[64, 64], dtype=pto.f4e1m2x2, memory_space=pto.MemorySpace.RIGHT, addr=0,
        valid_shape=[64, 62], blayout="RowMajor", slayout="ColMajor", fractal_size=512,
    )
    b_scale_tile = pto.alloc_tile(
        shape=[2, 64], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING, addr=0,
        valid_shape=[2, 62], blayout="ColMajor", slayout="ColMajor", fractal_size=32,
    )
    dst_tile = pto.alloc_tile(
        shape=[1, 64], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, addr=0,
        valid_shape=[1, 62], blayout="ColMajor", slayout="RowMajor", fractal_size=1024,
    )

    pto.mte_gm_l1(a_ptr, a_l1, 1024, nburst=(1, 0, 0))
    pto.mte_gm_l1(a_scale_ptr, a_scale_l1, 64, nburst=(1, 0, 0))
    pto.mte_gm_l1_frac(
        b_ptr, b_l1, pto.FractalMode.ND2NZ,
        shape=(128, 32), src_layout=(32,),
        dst_group=(1, 1, 128, 0), ctrl=(0, False),
    )
    pto.mte_gm_l1(b_scale_ptr, b_scale_l1, 256, nburst=(1, 0, 0))

    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)

    pto.mte_l1_l0a(a_l1, lhs_tile.as_ptr(), 1, 64)
    pto.mte_l1_l0b(b_l1, rhs_tile.as_ptr(), 64, 64, transpose=True)
    pto.mte_l1_l0a_mx(a_scale_l1, lhs_tile.as_ptr(), 1, 64)
    pto.mte_l1_l0b_mx(b_scale_l1, rhs_tile.as_ptr(), 64, 64)

    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)

    pto.tile.gemv_mx(lhs_tile, a_scale_tile, rhs_tile, b_scale_tile, dst_tile)

    pto.set_flag(pto.Pipe.M, pto.Pipe.MTE1, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.MTE1, event_id=1)

    pto.mte_l1_l0a(a_l1_hi, lhs_tile.as_ptr(), 1, 64)
    pto.mte_l1_l0b(b_l1_hi, rhs_tile.as_ptr(), 64, 64, transpose=True)
    pto.mte_l1_l0a_mx(a_scale_l1_hi, lhs_tile.as_ptr(), 1, 64)
    pto.mte_l1_l0b_mx(b_scale_l1_hi, rhs_tile.as_ptr(), 64, 64)

    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)

    pto.tile.gemv_mx_acc(dst_tile, lhs_tile, a_scale_tile, rhs_tile, b_scale_tile, dst_tile)

    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)

    pto.mte_l0c_gm(dst_tile.as_ptr(), c_ptr, 1, 64, 16, 64, 0, 0, layout="nz2nd")
    pto.pipe_barrier(pto.Pipe.ALL)


# gemv_mx_fp8_e4m3_e5m2_1x256x20: single K=256 pass (tgemv.mx)
@pto.jit(
    name="gemv_mx_fp8_e4m3_e5m2_1x256x20",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel_gemv_mx_fp8_e4m3_e5m2_1x256x20(
    a_ptr: pto.ptr(pto.f8e4m3, "gm"),
    b_ptr: pto.ptr(pto.f8e5m2, "gm"),
    a_scale_ptr: pto.ptr(pto.f8e4m3, "gm"),
    b_scale_ptr: pto.ptr(pto.f8e5m2, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    a_l1 = pto.castptr(pto.ui64(0), pto.ptr(pto.f8e4m3, "mat"))
    a_scale_l1 = pto.castptr(pto.ui64(1024), pto.ptr(pto.f8e4m3, "mat"))
    b_l1 = pto.castptr(pto.ui64(1152), pto.ptr(pto.f8e5m2, "mat"))
    b_scale_l1 = pto.castptr(pto.ui64(9344), pto.ptr(pto.f8e5m2, "mat"))

    lhs_tile = pto.alloc_tile(
        shape=[1, 256], dtype=pto.f8e4m3, memory_space=pto.MemorySpace.LEFT, addr=0,
        valid_shape=[1, 256], blayout="ColMajor", slayout="RowMajor", fractal_size=512,
    )
    a_scale_tile = pto.alloc_tile(
        shape=[1, 8], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING, addr=0,
        valid_shape=[1, 8], blayout="RowMajor", slayout="RowMajor", fractal_size=32,
    )
    rhs_tile = pto.alloc_tile(
        shape=[256, 32], dtype=pto.f8e5m2, memory_space=pto.MemorySpace.RIGHT, addr=0,
        valid_shape=[256, 20], blayout="RowMajor", slayout="ColMajor", fractal_size=512,
    )
    b_scale_tile = pto.alloc_tile(
        shape=[8, 32], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING, addr=0,
        valid_shape=[8, 20], blayout="ColMajor", slayout="ColMajor", fractal_size=32,
    )
    dst_tile = pto.alloc_tile(
        shape=[1, 32], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, addr=0,
        valid_shape=[1, 20], blayout="ColMajor", slayout="RowMajor", fractal_size=1024,
    )

    pto.mte_gm_l1(a_ptr, a_l1, 1024, nburst=(1, 0, 0))
    pto.mte_gm_l1(a_scale_ptr, a_scale_l1, 64, nburst=(2, 0, 0))
    pto.mte_gm_l1_frac(
        b_ptr, b_l1, pto.FractalMode.ND2NZ,
        shape=(256, 20), src_layout=(32,),
        dst_group=(1, 1, 256, 0), ctrl=(0, False),
    )
    pto.mte_gm_l1(b_scale_ptr, b_scale_l1, 256, nburst=(1, 0, 0))

    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)

    pto.mte_l1_l0a(a_l1, lhs_tile.as_ptr(), 1, 256)
    pto.mte_l1_l0b(b_l1, rhs_tile.as_ptr(), 256, 20, transpose=True)
    pto.mte_l1_l0a_mx(a_scale_l1, lhs_tile.as_ptr(), 1, 256)
    pto.mte_l1_l0b_mx(b_scale_l1, rhs_tile.as_ptr(), 256, 20)

    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)

    pto.tile.gemv_mx(lhs_tile, a_scale_tile, rhs_tile, b_scale_tile, dst_tile)

    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)

    pto.mte_l0c_gm(dst_tile.as_ptr(), c_ptr, 1, 32, 16, 32, 0, 0, layout="nz2nd")
    pto.pipe_barrier(pto.Pipe.ALL)


# gemv_mx_bias_fp4_e1m2_1x64x62: single K=64 pass with bias (tgemv.mx.bias)
@pto.jit(
    name="gemv_mx_bias_fp4_e1m2_1x64x62",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel_gemv_mx_bias_fp4_e1m2_1x64x62(
    a_ptr: pto.ptr(pto.f4e1m2x2, "gm"),
    b_ptr: pto.ptr(pto.f4e1m2x2, "gm"),
    a_scale_ptr: pto.ptr(pto.f4e1m2x2, "gm"),
    b_scale_ptr: pto.ptr(pto.f4e1m2x2, "gm"),
    bias_ptr: pto.ptr(pto.f32, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    a_l1 = pto.castptr(pto.ui64(0), pto.ptr(pto.f4e1m2x2, "mat"))
    a_scale_l1 = pto.castptr(pto.ui64(1024), pto.ptr(pto.f4e1m2x2, "mat"))
    b_l1 = pto.castptr(pto.ui64(1088), pto.ptr(pto.f4e1m2x2, "mat"))
    b_scale_l1 = pto.castptr(pto.ui64(3136), pto.ptr(pto.f4e1m2x2, "mat"))
    bias_l1 = pto.castptr(pto.ui64(3264), pto.ptr(pto.f32, "mat"))

    lhs_tile = pto.alloc_tile(
        shape=[1, 64], dtype=pto.f4e1m2x2, memory_space=pto.MemorySpace.LEFT, addr=0,
        valid_shape=[1, 64], blayout="ColMajor", slayout="RowMajor", fractal_size=512,
    )
    a_scale_tile = pto.alloc_tile(
        shape=[1, 2], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING, addr=0,
        valid_shape=[1, 2], blayout="RowMajor", slayout="RowMajor", fractal_size=32,
    )
    rhs_tile = pto.alloc_tile(
        shape=[64, 64], dtype=pto.f4e1m2x2, memory_space=pto.MemorySpace.RIGHT, addr=0,
        valid_shape=[64, 62], blayout="RowMajor", slayout="ColMajor", fractal_size=512,
    )
    b_scale_tile = pto.alloc_tile(
        shape=[2, 64], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING, addr=0,
        valid_shape=[2, 62], blayout="ColMajor", slayout="ColMajor", fractal_size=32,
    )
    bias_tile = pto.alloc_tile(
        shape=[1, 64], dtype=pto.f32, memory_space=pto.MemorySpace.BIAS, addr=0,
        valid_shape=[1, 62], blayout="RowMajor", slayout="NoneBox", fractal_size=512,
    )
    dst_tile = pto.alloc_tile(
        shape=[1, 64], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, addr=0,
        valid_shape=[1, 62], blayout="ColMajor", slayout="RowMajor", fractal_size=1024,
    )

    pto.mte_gm_l1(a_ptr, a_l1, 1024, nburst=(1, 0, 0))
    pto.mte_gm_l1(a_scale_ptr, a_scale_l1, 32, nburst=(1, 0, 0))
    pto.mte_gm_l1(b_ptr, b_l1, 1024, nburst=(1, 0, 0), loops=[(2, 1024, 1024)])
    pto.mte_gm_l1(b_scale_ptr, b_scale_l1, 64, nburst=(1, 0, 0), loops=[(2, 64, 64)])
    pto.mte_gm_l1(bias_ptr, bias_l1, 64, nburst=(1, 0, 0), loops=[(4, 64, 64)])

    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)

    pto.mte_l1_l0a(a_l1, lhs_tile.as_ptr(), 1, 64)
    pto.mte_l1_l0b(b_l1, rhs_tile.as_ptr(), 64, 64, transpose=True)
    pto.mte_l1_l0a_mx(a_scale_l1, lhs_tile.as_ptr(), 1, 64)
    pto.mte_l1_l0b_mx(b_scale_l1, rhs_tile.as_ptr(), 64, 64)
    pto.mte_l1_bt(bias_l1, bias_tile.as_ptr(), 62, nburst=(1, 0, 0))

    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)

    pto.tile.gemv_mx_bias(lhs_tile, a_scale_tile, rhs_tile, b_scale_tile, bias_tile, dst_tile)

    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)

    pto.mte_l0c_gm(dst_tile.as_ptr(), c_ptr, 1, 64, 16, 64, 0, 0, layout="nz2nd")
    pto.pipe_barrier(pto.Pipe.ALL)


# gemv_mx_bias_fp4_e1m2_1x2048x64: split-K into two K=1024 passes
# (tgemv.mx.bias then tgemv.mx.acc)
@pto.jit(
    name="gemv_mx_bias_fp4_e1m2_1x2048x64",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel_gemv_mx_bias_fp4_e1m2_1x2048x64(
    a_ptr: pto.ptr(pto.f4e1m2x2, "gm"),
    b_ptr: pto.ptr(pto.f4e1m2x2, "gm"),
    a_scale_ptr: pto.ptr(pto.f4e1m2x2, "gm"),
    b_scale_ptr: pto.ptr(pto.f4e1m2x2, "gm"),
    a1_ptr: pto.ptr(pto.f4e1m2x2, "gm"),
    b1_ptr: pto.ptr(pto.f4e1m2x2, "gm"),
    a1_scale_ptr: pto.ptr(pto.f4e1m2x2, "gm"),
    b1_scale_ptr: pto.ptr(pto.f4e1m2x2, "gm"),
    bias_ptr: pto.ptr(pto.f32, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    a_l1 = pto.castptr(pto.ui64(0), pto.ptr(pto.f4e1m2x2, "mat"))
    a_scale_l1 = pto.castptr(pto.ui64(1024), pto.ptr(pto.f4e1m2x2, "mat"))
    b_l1 = pto.castptr(pto.ui64(2048), pto.ptr(pto.f4e1m2x2, "mat"))
    b_scale_l1 = pto.castptr(pto.ui64(34816), pto.ptr(pto.f4e1m2x2, "mat"))
    bias_l1 = pto.castptr(pto.ui64(36864), pto.ptr(pto.f32, "mat"))

    lhs_tile = pto.alloc_tile(
        shape=[1, 1024], dtype=pto.f4e1m2x2, memory_space=pto.MemorySpace.LEFT, addr=0,
        valid_shape=[1, 1024], blayout="ColMajor", slayout="RowMajor", fractal_size=512,
    )
    a_scale_tile = pto.alloc_tile(
        shape=[1, 32], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING, addr=0,
        valid_shape=[1, 32], blayout="RowMajor", slayout="RowMajor", fractal_size=32,
    )
    rhs_tile = pto.alloc_tile(
        shape=[1024, 64], dtype=pto.f4e1m2x2, memory_space=pto.MemorySpace.RIGHT, addr=0,
        valid_shape=[1024, 64], blayout="RowMajor", slayout="ColMajor", fractal_size=512,
    )
    b_scale_tile = pto.alloc_tile(
        shape=[32, 64], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING, addr=0,
        valid_shape=[32, 64], blayout="ColMajor", slayout="ColMajor", fractal_size=32,
    )
    bias_tile = pto.alloc_tile(
        shape=[1, 64], dtype=pto.f32, memory_space=pto.MemorySpace.BIAS, addr=0,
        valid_shape=[1, 64], blayout="RowMajor", slayout="NoneBox", fractal_size=512,
    )
    dst_tile = pto.alloc_tile(
        shape=[1, 64], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, addr=0,
        valid_shape=[1, 64], blayout="ColMajor", slayout="RowMajor", fractal_size=1024,
    )

    # ---- Pass 0: K[0:1024] ----
    pto.mte_gm_l1(a_ptr, a_l1, 1024, nburst=(1, 0, 0))
    pto.mte_gm_l1(a_scale_ptr, a_scale_l1, 32, nburst=(1, 0, 0))
    pto.mte_gm_l1_frac(
        b_ptr, b_l1, pto.FractalMode.ND2NZ,
        shape=(1024, 32), src_layout=(32,),
        dst_group=(1, 1, 1024, 0), ctrl=(0, False),
    )
    pto.mte_gm_l1(b_scale_ptr, b_scale_l1, 512, nburst=(1, 0, 0), loops=[(4, 1024, 512)])
    pto.mte_gm_l1(bias_ptr, bias_l1, 256, nburst=(1, 0, 0))

    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)

    pto.mte_l1_l0a(a_l1, lhs_tile.as_ptr(), 1, 1024)
    pto.mte_l1_l0b(b_l1, rhs_tile.as_ptr(), 1024, 64, transpose=True)
    pto.mte_l1_l0a_mx(a_scale_l1, lhs_tile.as_ptr(), 1, 1024)
    pto.mte_l1_l0b_mx(b_scale_l1, rhs_tile.as_ptr(), 1024, 64)
    pto.mte_l1_bt(bias_l1, bias_tile.as_ptr(), 64, nburst=(1, 0, 0))

    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)

    pto.tile.gemv_mx_bias(lhs_tile, a_scale_tile, rhs_tile, b_scale_tile, bias_tile, dst_tile)

    pto.set_flag(pto.Pipe.M, pto.Pipe.MTE2, event_id=0)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.MTE2, event_id=0)

    # ---- Pass 1: K[1024:2048] (accumulate) ----
    pto.mte_gm_l1(a1_ptr, a_l1, 1024, nburst=(1, 0, 0))
    pto.mte_gm_l1(a1_scale_ptr, a_scale_l1, 32, nburst=(1, 0, 0))
    pto.mte_gm_l1_frac(
        b1_ptr, b_l1, pto.FractalMode.ND2NZ,
        shape=(1024, 32), src_layout=(32,),
        dst_group=(1, 1, 1024, 0), ctrl=(0, False),
    )
    pto.mte_gm_l1(b1_scale_ptr, b_scale_l1, 512, nburst=(1, 0, 0), loops=[(4, 1024, 512)])

    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)

    pto.mte_l1_l0a(a_l1, lhs_tile.as_ptr(), 1, 1024)
    pto.mte_l1_l0b(b_l1, rhs_tile.as_ptr(), 1024, 64, transpose=True)
    pto.mte_l1_l0a_mx(a_scale_l1, lhs_tile.as_ptr(), 1, 1024)
    pto.mte_l1_l0b_mx(b_scale_l1, rhs_tile.as_ptr(), 1024, 64)

    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)

    pto.tile.gemv_mx_acc(dst_tile, lhs_tile, a_scale_tile, rhs_tile, b_scale_tile, dst_tile)

    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)

    pto.mte_l0c_gm(dst_tile.as_ptr(), c_ptr, 1, 64, 16, 64, 0, 0, layout="nz2nd")
    pto.pipe_barrier(pto.Pipe.ALL)


# ---------------------------------------------------------------------------
# CASES
# ---------------------------------------------------------------------------

CASES = [
    golden_output_case(
        spec["name"],
        globals()["_kernel_" + spec["name"]],
        inputs=lambda spec=spec: _case_inputs(spec),
        expected=lambda *_, spec=spec: _gen_case_data(spec)[-1],
        rtol=spec["eps"],
        atol=spec["eps"],
    )
    for spec in CASE_SPECS
]


auto_main(globals())
