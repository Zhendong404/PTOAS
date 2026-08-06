#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").

from pathlib import Path
import sys
import zlib

import ml_dtypes
import en_dtypes
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto

M, K, N = 128, 64, 64
SCALE_K = 2
L1_A_ADDR = 0
L1_A_SCALE_ADDR = 8192
L1_B_ADDR = 8448
L1_B_SCALE_ADDR = 12544

fp8_e5m2 = ml_dtypes.float8_e5m2


@pto.jit(
    name="tmatmul_mx_fp8_e5m2_128x64x64",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel(
    a_ptr: pto.ptr(pto.f8e5m2, "gm"),
    b_ptr: pto.ptr(pto.f8e5m2, "gm"),
    a_scale_ptr: pto.ptr(pto.f8e5m2, "gm"),
    b_scale_ptr: pto.ptr(pto.f8e5m2, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    a_l1 = pto.castptr(pto.ui64(L1_A_ADDR), pto.ptr(pto.f8e5m2, "mat"))
    a_scale_l1 = pto.castptr(pto.ui64(L1_A_SCALE_ADDR), pto.ptr(pto.f8e5m2, "mat"))
    b_l1 = pto.castptr(pto.ui64(L1_B_ADDR), pto.ptr(pto.f8e5m2, "mat"))
    b_scale_l1 = pto.castptr(pto.ui64(L1_B_SCALE_ADDR), pto.ptr(pto.f8e5m2, "mat"))

    lhs = pto.alloc_tile(
        shape=[M, K], dtype=pto.f8e5m2, memory_space=pto.MemorySpace.LEFT,
        addr=0, valid_shape=[M, K], blayout="ColMajor", slayout="RowMajor", fractal_size=512,
    )
    lhs_scale = pto.alloc_tile(
        shape=[M, SCALE_K], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
        addr=0, valid_shape=[M, SCALE_K], blayout="RowMajor", slayout="RowMajor", fractal_size=32,
    )
    rhs = pto.alloc_tile(
        shape=[K, N], dtype=pto.f8e5m2, memory_space=pto.MemorySpace.RIGHT,
        addr=0, valid_shape=[K, N], blayout="RowMajor", slayout="ColMajor", fractal_size=512,
    )
    rhs_scale = pto.alloc_tile(
        shape=[SCALE_K, N], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
        addr=0, valid_shape=[SCALE_K, N], blayout="ColMajor", slayout="ColMajor", fractal_size=32,
    )
    dst = pto.alloc_tile(
        shape=[M, N], dtype=pto.f32, memory_space=pto.MemorySpace.ACC,
        addr=0, valid_shape=[M, N], blayout="ColMajor", slayout="RowMajor", fractal_size=512,
    )

    pto.mte_gm_l1(a_ptr, a_l1, 8192, nburst=(1, 0, 0))
    pto.mte_gm_l1(a_scale_ptr, a_scale_l1, 256, nburst=(1, 0, 0))
    pto.mte_gm_l1(b_ptr, b_l1, 4096, nburst=(1, 0, 0))
    pto.mte_gm_l1(b_scale_ptr, b_scale_l1, 128, nburst=(1, 0, 0))

    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.mte_l1_l0a(a_l1, lhs.as_ptr(), M, K)
    pto.mte_l1_l0b(b_l1, rhs.as_ptr(), K, N, transpose=True)
    pto.mte_l1_l0a_mx(a_scale_l1, lhs.as_ptr(), M, K)
    pto.mte_l1_l0b_mx(b_scale_l1, rhs.as_ptr(), K, N)

    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, dst)

    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.mte_l0c_gm(dst.as_ptr(), c_ptr, M, N, M, N, 0, 0, layout="nz2nd")
    pto.pipe_barrier(pto.Pipe.ALL)


def _pack_two_fp4(matrix):
    flat = matrix.reshape(-1)
    return (((flat[1::2].view(np.uint8) & 0x0F) << 4) | (flat[::2].view(np.uint8) & 0x0F)).reshape(matrix.shape[0], -1)


def _pack_lhs_fp8(matrix):
    return np.ascontiguousarray(matrix.reshape(M // 32, 32, K // 32, 32).transpose(2, 0, 1, 3))


def _pack_rhs_fp8(matrix):
    return np.ascontiguousarray(matrix.reshape(K // 16, 16, N // 32, 32).transpose(2, 0, 1, 3))


def _pack_lhs_fp8_shape(matrix):
    rows, cols = matrix.shape
    if rows >= 32 and rows % 32 == 0:
        packed = matrix.reshape(rows // 32, 32, cols // 32, 32).transpose(2, 0, 1, 3)
    else:
        packed = matrix.reshape(rows, cols // 32, 32).transpose(1, 0, 2)
    return np.ascontiguousarray(packed)


def _pack_rhs_fp8_shape(matrix):
    rows, cols = matrix.shape
    packed = matrix.reshape(rows // 16, 16, cols // 32, 32).transpose(2, 0, 1, 3)
    return np.ascontiguousarray(packed)


def _scale_a_format(scale):
    rows, cols = scale.shape
    rows_padded = (rows + 15) // 16 * 16
    cols_padded = (cols + 1) // 2 * 2
    padded = np.zeros((rows_padded, cols_padded), dtype=scale.dtype)
    padded[:rows, :cols] = scale
    return np.ascontiguousarray(
        padded.reshape(rows_padded // 16, 16, cols_padded // 2, 2)
        .transpose(0, 2, 1, 3)
        .reshape(-1, 32)
    )


def _scale_b_format(scale, n_padded=None):
    rows, cols = scale.shape
    rows_padded = (rows + 1) // 2 * 2
    target_cols = cols if n_padded is None else max(cols, n_padded)
    cols_padded = (target_cols + 15) // 16 * 16
    padded = np.zeros((rows_padded, cols_padded), dtype=scale.dtype)
    padded[:rows, :cols] = scale
    return np.ascontiguousarray(
        padded.reshape(rows_padded // 2, 2, cols_padded // 16, 16)
        .transpose(2, 0, 3, 1)
        .reshape(-1, cols_padded)
    )


def _make_data():
    np.random.seed(zlib.crc32(b"fp8_e5m2_128x64x64") & 0xFFFFFFFF)
    a = np.random.randint(-10, 10, (M, K)).astype(fp8_e5m2)
    b = np.random.randint(-10, 10, (K, N)).astype(fp8_e5m2)
    a_scale = np.random.randint(127, 130, (M, SCALE_K)).astype(np.uint8)
    b_scale = np.random.randint(127, 130, (SCALE_K, N)).astype(np.uint8)
    a_float = np.zeros((M, K), dtype=np.float64)
    b_float = np.zeros((K, N), dtype=np.float64)
    for i in range(K):
        a_float[:, i] = a[:, i].astype(np.float64) * 2 ** (a_scale[:, i // 32] - 127)
        b_float[i, :] = b[i, :].astype(np.float64) * 2 ** (b_scale[i // 32, :] - 127)
    # The legacy launcher ABI is uint8_t* for packed FP8/scale buffers;
    # materialize the same raw bytes so torch can transfer them to the device.
    return [
        _pack_lhs_fp8(a).view(np.uint8),
        _pack_rhs_fp8(b).view(np.uint8),
        _scale_a_format(a_scale).view(np.uint8),
        _scale_b_format(b_scale).view(np.uint8),
    ], (a_float @ b_float).astype(np.float32)


def _inputs():
    return _make_data()[0]


def _expected(*_inputs):
    return _make_data()[1]


@pto.jit(
    name="tmatmul_mx_fp8_e4m3_16x32x16",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel_small(
    a_ptr: pto.ptr(pto.f8e4m3, "gm"),
    b_ptr: pto.ptr(pto.f8e4m3, "gm"),
    a_scale_ptr: pto.ptr(pto.f8e4m3, "gm"),
    b_scale_ptr: pto.ptr(pto.f8e4m3, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    a_l1 = pto.castptr(pto.ui64(0), pto.ptr(pto.f8e4m3, "mat"))
    a_scale_l1 = pto.castptr(pto.ui64(1024), pto.ptr(pto.f8e4m3, "mat"))
    b_l1 = pto.castptr(pto.ui64(1088), pto.ptr(pto.f8e4m3, "mat"))
    b_scale_l1 = pto.castptr(pto.ui64(2112), pto.ptr(pto.f8e4m3, "mat"))
    lhs = pto.alloc_tile(shape=[16, 64], dtype=pto.f8e4m3, memory_space=pto.MemorySpace.LEFT,
                         addr=0, valid_shape=[16, 64], blayout="ColMajor", slayout="RowMajor", fractal_size=512)
    lhs_scale = pto.alloc_tile(shape=[16, 2], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
                               addr=0, valid_shape=[16, 2], blayout="RowMajor", slayout="RowMajor", fractal_size=32)
    rhs = pto.alloc_tile(shape=[64, 16], dtype=pto.f8e4m3, memory_space=pto.MemorySpace.RIGHT,
                         addr=0, valid_shape=[64, 16], blayout="RowMajor", slayout="ColMajor", fractal_size=512)
    rhs_scale = pto.alloc_tile(shape=[2, 16], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
                               addr=0, valid_shape=[2, 16], blayout="ColMajor", slayout="ColMajor", fractal_size=32)
    dst = pto.alloc_tile(shape=[16, 16], dtype=pto.f32, memory_space=pto.MemorySpace.ACC,
                         addr=0, valid_shape=[16, 16], blayout="ColMajor", slayout="RowMajor", fractal_size=512)

    pto.mte_gm_l1_frac(a_ptr, a_l1, pto.FractalMode.ND2NZ, shape=(16, 64), src_layout=(64,),
                       dst_group=(1, 1, 16, 0), ctrl=(0, False))
    pto.mte_gm_l1(a_scale_ptr, a_scale_l1, 32, nburst=(1, 0, 0))
    pto.mte_gm_l1_frac(b_ptr, b_l1, pto.FractalMode.ND2NZ, shape=(64, 16), src_layout=(16,),
                       dst_group=(1, 1, 64, 0), ctrl=(0, False))
    pto.mte_gm_l1(b_scale_ptr, b_scale_l1, 32, nburst=(1, 0, 0))
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.mte_l1_l0a(a_l1, lhs.as_ptr(), 16, 64)
    pto.mte_l1_l0b(b_l1, rhs.as_ptr(), 64, 16, transpose=True)
    pto.mte_l1_l0a_mx(a_scale_l1, lhs.as_ptr(), 16, 64)
    pto.mte_l1_l0b_mx(b_scale_l1, rhs.as_ptr(), 64, 16)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, dst)
    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.mte_l0c_gm(dst.as_ptr(), c_ptr, 16, 16, 16, 16, 0, 0, layout="nz2nd")
    pto.pipe_barrier(pto.Pipe.ALL)


def _make_small_data():
    np.random.seed(zlib.crc32(b"fp8_e4m3_16x32x16") & 0xFFFFFFFF)
    logical_a = np.random.randint(-10, 10, (16, 32)).astype(ml_dtypes.float8_e4m3fn)
    logical_b = np.random.randint(-10, 10, (32, 16)).astype(ml_dtypes.float8_e4m3fn)
    a = np.zeros((16, 64), dtype=ml_dtypes.float8_e4m3fn)
    b = np.zeros((64, 16), dtype=ml_dtypes.float8_e4m3fn)
    a[:, :32] = logical_a
    b[:32, :] = logical_b
    a_scale = np.random.randint(127, 130, (16, 2)).astype(np.uint8)
    b_scale = np.random.randint(127, 130, (2, 16)).astype(np.uint8)
    a_float = np.zeros((16, 64), dtype=np.float64)
    b_float = np.zeros((64, 16), dtype=np.float64)
    for i in range(32):
        a_float[:, i] = a[:, i].astype(np.float64) * 2 ** (a_scale[:, i // 32] - 127)
        b_float[i, :] = b[i, :].astype(np.float64) * 2 ** (b_scale[i // 32, :] - 127)
    return [a.view(np.uint8), b.view(np.uint8), _scale_a_format(a_scale).view(np.uint8),
            _scale_b_format(b_scale, n_padded=16).view(np.uint8)], (a_float @ b_float).astype(np.float32)


@pto.jit(
    name="tmatmul_mx_fp8_e4m3_127x72x64",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel_fp8_127(
    a_ptr: pto.ptr(pto.f8e4m3, "gm"),
    b_ptr: pto.ptr(pto.f8e4m3, "gm"),
    a_scale_ptr: pto.ptr(pto.f8e4m3, "gm"),
    b_scale_ptr: pto.ptr(pto.f8e4m3, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    a_l1 = pto.castptr(pto.ui64(0), pto.ptr(pto.f8e4m3, "mat"))
    a_scale_l1 = pto.castptr(pto.ui64(16384), pto.ptr(pto.f8e4m3, "mat"))
    b_l1 = pto.castptr(pto.ui64(16896), pto.ptr(pto.f8e4m3, "mat"))
    b_scale_l1 = pto.castptr(pto.ui64(25088), pto.ptr(pto.f8e4m3, "mat"))
    lhs = pto.alloc_tile(shape=[128, 128], dtype=pto.f8e4m3, memory_space=pto.MemorySpace.LEFT,
                         addr=0, valid_shape=[127, 128], blayout="ColMajor", slayout="RowMajor", fractal_size=512)
    lhs_scale = pto.alloc_tile(shape=[128, 4], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
                               addr=0, valid_shape=[127, 4], blayout="RowMajor", slayout="RowMajor", fractal_size=32)
    rhs = pto.alloc_tile(shape=[128, 64], dtype=pto.f8e4m3, memory_space=pto.MemorySpace.RIGHT,
                         addr=0, valid_shape=[128, 64], blayout="RowMajor", slayout="ColMajor", fractal_size=512)
    rhs_scale = pto.alloc_tile(shape=[4, 64], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
                               addr=0, valid_shape=[4, 64], blayout="ColMajor", slayout="ColMajor", fractal_size=32)
    dst = pto.alloc_tile(shape=[128, 64], dtype=pto.f32, memory_space=pto.MemorySpace.ACC,
                         addr=0, valid_shape=[127, 64], blayout="ColMajor", slayout="RowMajor", fractal_size=512)
    pto.mte_gm_l1(a_ptr, a_l1, 1024, nburst=(1, 0, 0), loops=[(16, 1024, 1024)])
    pto.mte_gm_l1(a_scale_ptr, a_scale_l1, 64, nburst=(1, 0, 0), loops=[(8, 64, 64)])
    pto.mte_gm_l1(b_ptr, b_l1, 1024, nburst=(1, 0, 0), loops=[(8, 1024, 1024)])
    pto.mte_gm_l1(b_scale_ptr, b_scale_l1, 64, nburst=(1, 0, 0), loops=[(4, 64, 64)])
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.mte_l1_l0a(a_l1, lhs.as_ptr(), 127, 128)
    pto.mte_l1_l0b(b_l1, rhs.as_ptr(), 128, 64, transpose=True)
    pto.mte_l1_l0a_mx(a_scale_l1, lhs.as_ptr(), 127, 128)
    pto.mte_l1_l0b_mx(b_scale_l1, rhs.as_ptr(), 128, 64)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, dst)
    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.mte_l0c_gm(dst.as_ptr(), c_ptr, 128, 64, 128, 64, 0, 0, layout="nz2nd")
    pto.pipe_barrier(pto.Pipe.ALL)


def _make_127_data():
    m, k, n = 127, 72, 64
    k_aligned = 128
    np.random.seed(zlib.crc32(b"fp8_e4m3_127x72x64") & 0xFFFFFFFF)
    logical_a = np.random.randint(-10, 10, (m, k)).astype(ml_dtypes.float8_e4m3fn)
    logical_b = np.random.randint(-10, 10, (k, n)).astype(ml_dtypes.float8_e4m3fn)
    a = np.zeros((128, k_aligned), dtype=ml_dtypes.float8_e4m3fn)
    b = np.zeros((k_aligned, n), dtype=ml_dtypes.float8_e4m3fn)
    a[:m, :k] = logical_a
    b[:k, :n] = logical_b
    a_scale = np.random.randint(127, 130, (m, 4)).astype(np.uint8)
    b_scale = np.random.randint(127, 130, (4, n)).astype(np.uint8)
    a_float = np.zeros((m, k), dtype=np.float64)
    b_float = np.zeros((k, n), dtype=np.float64)
    for i in range(k):
        a_float[:, i] = a[:m, i].astype(np.float64) * 2 ** (a_scale[:, i // 32] - 127)
        b_float[i, :] = b[i, :].astype(np.float64) * 2 ** (b_scale[i // 32, :] - 127)
    expected = np.zeros((128, n), dtype=np.float32)
    expected[:m, :n] = (a_float @ b_float).astype(np.float32)
    return [
        _pack_lhs_fp8_shape(a).view(np.uint8), _pack_rhs_fp8_shape(b).view(np.uint8),
        _scale_a_format(a_scale).view(np.uint8), _scale_b_format(b_scale, n_padded=n).view(np.uint8),
    ], expected


@pto.jit(
    name="tmatmul_mx_fp8_e4m3_e5m2_128x110x63",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel_fp8_mixed(
    a_ptr: pto.ptr(pto.f8e4m3, "gm"),
    b_ptr: pto.ptr(pto.f8e5m2, "gm"),
    a_scale_ptr: pto.ptr(pto.f8e4m3, "gm"),
    b_scale_ptr: pto.ptr(pto.f8e5m2, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    a_l1 = pto.castptr(pto.ui64(0), pto.ptr(pto.f8e4m3, "mat"))
    a_scale_l1 = pto.castptr(pto.ui64(16384), pto.ptr(pto.f8e4m3, "mat"))
    b_l1 = pto.castptr(pto.ui64(16896), pto.ptr(pto.f8e5m2, "mat"))
    b_scale_l1 = pto.castptr(pto.ui64(25088), pto.ptr(pto.f8e5m2, "mat"))
    lhs = pto.alloc_tile(shape=[128, 128], dtype=pto.f8e4m3, memory_space=pto.MemorySpace.LEFT,
                         addr=0, valid_shape=[128, 128], blayout="ColMajor", slayout="RowMajor", fractal_size=512)
    lhs_scale = pto.alloc_tile(shape=[128, 4], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
                               addr=0, valid_shape=[128, 4], blayout="RowMajor", slayout="RowMajor", fractal_size=32)
    rhs = pto.alloc_tile(shape=[128, 64], dtype=pto.f8e5m2, memory_space=pto.MemorySpace.RIGHT,
                         addr=0, valid_shape=[128, 63], blayout="RowMajor", slayout="ColMajor", fractal_size=512)
    rhs_scale = pto.alloc_tile(shape=[4, 64], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
                               addr=0, valid_shape=[4, 63], blayout="ColMajor", slayout="ColMajor", fractal_size=32)
    dst = pto.alloc_tile(shape=[128, 64], dtype=pto.f32, memory_space=pto.MemorySpace.ACC,
                         addr=0, valid_shape=[128, 63], blayout="ColMajor", slayout="RowMajor", fractal_size=512)
    pto.mte_gm_l1(a_ptr, a_l1, 1024, nburst=(1, 0, 0), loops=[(16, 1024, 1024)])
    pto.mte_gm_l1(a_scale_ptr, a_scale_l1, 64, nburst=(1, 0, 0), loops=[(8, 64, 64)])
    pto.mte_gm_l1(b_ptr, b_l1, 1024, nburst=(1, 0, 0), loops=[(8, 1024, 1024)])
    pto.mte_gm_l1(b_scale_ptr, b_scale_l1, 64, nburst=(1, 0, 0), loops=[(4, 64, 64)])
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.mte_l1_l0a(a_l1, lhs.as_ptr(), 128, 128)
    pto.mte_l1_l0b(b_l1, rhs.as_ptr(), 128, 63, transpose=True)
    pto.mte_l1_l0a_mx(a_scale_l1, lhs.as_ptr(), 128, 128)
    pto.mte_l1_l0b_mx(b_scale_l1, rhs.as_ptr(), 128, 63)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, dst)
    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.mte_l0c_gm(dst.as_ptr(), c_ptr, 128, 64, 128, 64, 0, 0, layout="nz2nd")
    pto.pipe_barrier(pto.Pipe.ALL)


def _make_mixed_data():
    m, k, n = 128, 110, 63
    k_aligned, n_padded = 128, 64
    np.random.seed(zlib.crc32(b"fp8_e4m3_e5m2_128x110x63") & 0xFFFFFFFF)
    logical_a = np.random.randint(-10, 10, (m, k)).astype(ml_dtypes.float8_e4m3fn)
    logical_b = np.random.randint(-10, 10, (k, n)).astype(ml_dtypes.float8_e5m2)
    a = np.zeros((m, k_aligned), dtype=ml_dtypes.float8_e4m3fn)
    b = np.zeros((k_aligned, n_padded), dtype=ml_dtypes.float8_e5m2)
    a[:, :k] = logical_a
    b[:k, :n] = logical_b
    a_scale = np.random.randint(127, 130, (m, 4)).astype(np.uint8)
    b_scale = np.random.randint(127, 130, (4, n)).astype(np.uint8)
    a_float = np.zeros((m, k), dtype=np.float64)
    b_float = np.zeros((k, n), dtype=np.float64)
    for i in range(k):
        a_float[:, i] = a[:, i].astype(np.float64) * 2 ** (a_scale[:, i // 32] - 127)
        b_float[i, :] = b[i, :n].astype(np.float64) * 2 ** (b_scale[i // 32, :] - 127)
    expected = np.zeros((m, n_padded), dtype=np.float32)
    expected[:, :n] = (a_float @ b_float).astype(np.float32)
    return [
        _pack_lhs_fp8_shape(a).view(np.uint8), _pack_rhs_fp8_shape(b).view(np.uint8),
        _scale_a_format(a_scale).view(np.uint8), _scale_b_format(b_scale, n_padded=n_padded).view(np.uint8),
    ], expected


@pto.jit(
    name="tmatmul_mx_fp4_e2m1_128x64x64",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel_fp4_128(
    a_ptr: pto.ptr(pto.f4e2m1x2, "gm"),
    b_ptr: pto.ptr(pto.f4e2m1x2, "gm"),
    a_scale_ptr: pto.ptr(pto.f4e2m1x2, "gm"),
    b_scale_ptr: pto.ptr(pto.f4e2m1x2, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    a_l1 = pto.castptr(pto.ui64(0), pto.ptr(pto.f4e2m1x2, "mat"))
    a_scale_l1 = pto.castptr(pto.ui64(4096), pto.ptr(pto.f4e2m1x2, "mat"))
    b_l1 = pto.castptr(pto.ui64(4352), pto.ptr(pto.f4e2m1x2, "mat"))
    b_scale_l1 = pto.castptr(pto.ui64(6400), pto.ptr(pto.f4e2m1x2, "mat"))
    lhs = pto.alloc_tile(shape=[128, 64], dtype=pto.f4e2m1x2, memory_space=pto.MemorySpace.LEFT,
                         addr=0, valid_shape=[128, 64], blayout="ColMajor", slayout="RowMajor", fractal_size=512)
    lhs_scale = pto.alloc_tile(shape=[128, 2], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
                               addr=0, valid_shape=[128, 2], blayout="RowMajor", slayout="RowMajor", fractal_size=32)
    rhs = pto.alloc_tile(shape=[64, 64], dtype=pto.f4e2m1x2, memory_space=pto.MemorySpace.RIGHT,
                         addr=0, valid_shape=[64, 64], blayout="RowMajor", slayout="ColMajor", fractal_size=512)
    rhs_scale = pto.alloc_tile(shape=[2, 64], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
                               addr=0, valid_shape=[2, 64], blayout="ColMajor", slayout="ColMajor", fractal_size=32)
    dst = pto.alloc_tile(shape=[128, 64], dtype=pto.f32, memory_space=pto.MemorySpace.ACC,
                         addr=0, valid_shape=[128, 64], blayout="ColMajor", slayout="RowMajor", fractal_size=512)
    pto.mte_gm_l1(a_ptr, a_l1, 1024, nburst=(1, 0, 0), loops=[(4, 1024, 1024)])
    pto.mte_gm_l1(a_scale_ptr, a_scale_l1, 64, nburst=(1, 0, 0), loops=[(4, 64, 64)])
    pto.mte_gm_l1(b_ptr, b_l1, 1024, nburst=(1, 0, 0), loops=[(2, 1024, 1024)])
    pto.mte_gm_l1(b_scale_ptr, b_scale_l1, 64, nburst=(1, 0, 0), loops=[(2, 64, 64)])
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.mte_l1_l0a(a_l1, lhs.as_ptr(), 128, 64)
    pto.mte_l1_l0b(b_l1, rhs.as_ptr(), 64, 64, transpose=True)
    pto.mte_l1_l0a_mx(a_scale_l1, lhs.as_ptr(), 128, 64)
    pto.mte_l1_l0b_mx(b_scale_l1, rhs.as_ptr(), 64, 64)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, dst)
    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.mte_l0c_gm(dst.as_ptr(), c_ptr, 128, 64, 128, 64, 0, 0, layout="nz2nd")
    pto.pipe_barrier(pto.Pipe.ALL)


def _make_fp4_128_data():
    m = k = n = 128
    np.random.seed(zlib.crc32(b"fp4_e2m1_128x64x64") & 0xFFFFFFFF)
    logical_a = np.random.randint(-6, 6, (m, 64)).astype(en_dtypes.float4_e2m1)
    logical_b = np.random.randint(-6, 6, (64, 64)).astype(en_dtypes.float4_e2m1)
    a = np.zeros((m, 64), dtype=en_dtypes.float4_e2m1)
    b = np.zeros((64, 64), dtype=en_dtypes.float4_e2m1)
    a[:, :64] = logical_a
    b[:64, :] = logical_b
    a_scale = np.random.randint(127, 130, (m, 2)).astype(np.uint8)
    b_scale = np.random.randint(127, 130, (2, 64)).astype(np.uint8)
    a_float = np.zeros((m, 64), dtype=np.float64)
    b_float = np.zeros((64, 64), dtype=np.float64)
    for i in range(64):
        a_float[:, i] = a[:, i].astype(np.float64) * 2 ** (a_scale[:, i // 32] - 127)
        b_float[i, :] = b[i, :].astype(np.float64) * 2 ** (b_scale[i // 32, :] - 127)
    expected = (a_float @ b_float).astype(np.float32)
    return [
        _pack_two_fp4(a), _pack_two_fp4(b),
        _scale_a_format(a_scale).view(np.uint8), _scale_b_format(b_scale, n_padded=64).view(np.uint8),
    ], expected


@pto.jit(
    name="tmatmul_mx_fp4_e1m2_e2m1_117x64x60",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel_fp4_117(
    a_ptr: pto.ptr(pto.f4e1m2x2, "gm"),
    b_ptr: pto.ptr(pto.f4e2m1x2, "gm"),
    a_scale_ptr: pto.ptr(pto.f4e1m2x2, "gm"),
    b_scale_ptr: pto.ptr(pto.f4e2m1x2, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    a_l1 = pto.castptr(pto.ui64(0), pto.ptr(pto.f4e1m2x2, "mat"))
    a_scale_l1 = pto.castptr(pto.ui64(4096), pto.ptr(pto.f4e1m2x2, "mat"))
    b_l1 = pto.castptr(pto.ui64(4352), pto.ptr(pto.f4e2m1x2, "mat"))
    b_scale_l1 = pto.castptr(pto.ui64(6400), pto.ptr(pto.f4e2m1x2, "mat"))
    lhs = pto.alloc_tile(shape=[128, 64], dtype=pto.f4e1m2x2, memory_space=pto.MemorySpace.LEFT,
                         addr=0, valid_shape=[117, 64], blayout="ColMajor", slayout="RowMajor", fractal_size=512)
    lhs_scale = pto.alloc_tile(shape=[128, 2], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
                               addr=0, valid_shape=[117, 2], blayout="RowMajor", slayout="RowMajor", fractal_size=32)
    rhs = pto.alloc_tile(shape=[64, 64], dtype=pto.f4e2m1x2, memory_space=pto.MemorySpace.RIGHT,
                         addr=0, valid_shape=[64, 60], blayout="RowMajor", slayout="ColMajor", fractal_size=512)
    rhs_scale = pto.alloc_tile(shape=[2, 64], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
                               addr=0, valid_shape=[2, 60], blayout="ColMajor", slayout="ColMajor", fractal_size=32)
    dst = pto.alloc_tile(shape=[128, 64], dtype=pto.f32, memory_space=pto.MemorySpace.ACC,
                         addr=0, valid_shape=[117, 60], blayout="ColMajor", slayout="RowMajor", fractal_size=512)
    pto.mte_gm_l1(a_ptr, a_l1, 1024, nburst=(1, 0, 0), loops=[(4, 1024, 1024)])
    pto.mte_gm_l1(a_scale_ptr, a_scale_l1, 64, nburst=(1, 0, 0), loops=[(4, 64, 64)])
    pto.mte_gm_l1(b_ptr, b_l1, 1024, nburst=(1, 0, 0), loops=[(2, 1024, 1024)])
    pto.mte_gm_l1(b_scale_ptr, b_scale_l1, 64, nburst=(1, 0, 0), loops=[(2, 64, 64)])
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.mte_l1_l0a(a_l1, lhs.as_ptr(), 117, 64)
    pto.mte_l1_l0b(b_l1, rhs.as_ptr(), 64, 60, transpose=True)
    pto.mte_l1_l0a_mx(a_scale_l1, lhs.as_ptr(), 117, 64)
    pto.mte_l1_l0b_mx(b_scale_l1, rhs.as_ptr(), 64, 60)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, dst)
    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.mte_l0c_gm(dst.as_ptr(), c_ptr, 128, 64, 128, 64, 0, 0, layout="nz2nd")
    pto.pipe_barrier(pto.Pipe.ALL)


def _make_fp4_117_data():
    m, k, n = 117, 64, 60
    np.random.seed(zlib.crc32(b"fp4_e1m2_e2m1_117x64x60") & 0xFFFFFFFF)
    logical_a = np.random.randint(-1, 2, (m, k)).astype(en_dtypes.float4_e1m2)
    logical_b = np.random.randint(-6, 6, (k, n)).astype(en_dtypes.float4_e2m1)
    a = np.zeros((128, 64), dtype=en_dtypes.float4_e1m2)
    b = np.zeros((64, 64), dtype=en_dtypes.float4_e2m1)
    a[:m, :k] = logical_a
    b[:k, :n] = logical_b
    a_scale = np.random.randint(127, 130, (m, 2)).astype(np.uint8)
    b_scale = np.random.randint(127, 130, (2, n)).astype(np.uint8)
    a_float = np.zeros((m, k), dtype=np.float64)
    b_float = np.zeros((k, n), dtype=np.float64)
    for i in range(k):
        a_float[:, i] = a[:m, i].astype(np.float64) * 2 ** (a_scale[:, i // 32] - 127)
        b_float[i, :] = b[i, :n].astype(np.float64) * 2 ** (b_scale[i // 32, :] - 127)
    expected = np.zeros((128, 64), dtype=np.float32)
    expected[:m, :n] = (a_float @ b_float).astype(np.float32)
    return [
        _pack_two_fp4(a), _pack_two_fp4(b),
        _scale_a_format(a_scale).view(np.uint8), _scale_b_format(b_scale, n_padded=64).view(np.uint8),
    ], expected


@pto.jit(
    name="tmatmul_mx_fp8_e4m3_e5m2_10x50x54",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel_fp8_small(
    a_ptr: pto.ptr(pto.f8e4m3, "gm"),
    b_ptr: pto.ptr(pto.f8e5m2, "gm"),
    a_scale_ptr: pto.ptr(pto.f8e4m3, "gm"),
    b_scale_ptr: pto.ptr(pto.f8e5m2, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    a_l1 = pto.castptr(pto.ui64(0), pto.ptr(pto.f8e4m3, "mat"))
    a_scale_l1 = pto.castptr(pto.ui64(1024), pto.ptr(pto.f8e4m3, "mat"))
    b_l1 = pto.castptr(pto.ui64(1088), pto.ptr(pto.f8e5m2, "mat"))
    b_scale_l1 = pto.castptr(pto.ui64(5184), pto.ptr(pto.f8e5m2, "mat"))
    lhs = pto.alloc_tile(shape=[16, 64], dtype=pto.f8e4m3, memory_space=pto.MemorySpace.LEFT,
                         addr=0, valid_shape=[10, 64], blayout="ColMajor", slayout="RowMajor", fractal_size=512)
    lhs_scale = pto.alloc_tile(shape=[16, 2], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
                               addr=0, valid_shape=[10, 2], blayout="RowMajor", slayout="RowMajor", fractal_size=32)
    rhs = pto.alloc_tile(shape=[64, 64], dtype=pto.f8e5m2, memory_space=pto.MemorySpace.RIGHT,
                         addr=0, valid_shape=[64, 54], blayout="RowMajor", slayout="ColMajor", fractal_size=512)
    rhs_scale = pto.alloc_tile(shape=[2, 64], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
                               addr=0, valid_shape=[2, 54], blayout="ColMajor", slayout="ColMajor", fractal_size=32)
    dst = pto.alloc_tile(shape=[16, 64], dtype=pto.f32, memory_space=pto.MemorySpace.ACC,
                         addr=0, valid_shape=[10, 54], blayout="ColMajor", slayout="RowMajor", fractal_size=512)
    pto.mte_gm_l1(a_ptr, a_l1, 1024, nburst=(1, 0, 0))
    pto.mte_gm_l1(a_scale_ptr, a_scale_l1, 32, nburst=(1, 0, 0))
    pto.mte_gm_l1(b_ptr, b_l1, 1024, nburst=(1, 0, 0), loops=[(4, 1024, 1024)])
    pto.mte_gm_l1(b_scale_ptr, b_scale_l1, 64, nburst=(1, 0, 0), loops=[(2, 64, 64)])
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.mte_l1_l0a(a_l1, lhs.as_ptr(), 10, 64)
    pto.mte_l1_l0b(b_l1, rhs.as_ptr(), 64, 54, transpose=True)
    pto.mte_l1_l0a_mx(a_scale_l1, lhs.as_ptr(), 10, 64)
    pto.mte_l1_l0b_mx(b_scale_l1, rhs.as_ptr(), 64, 54)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, dst)
    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.mte_l0c_gm(dst.as_ptr(), c_ptr, 16, 64, 16, 64, 0, 0, layout="nz2nd")
    pto.pipe_barrier(pto.Pipe.ALL)


def _make_fp8_small_data():
    m, k, n = 10, 50, 54
    np.random.seed(zlib.crc32(b"fp8_e4m3_e5m2_10x50x54") & 0xFFFFFFFF)
    logical_a = np.random.randint(-10, 10, (m, k)).astype(ml_dtypes.float8_e4m3fn)
    logical_b = np.random.randint(-10, 10, (k, n)).astype(ml_dtypes.float8_e5m2)
    a = np.zeros((16, 64), dtype=ml_dtypes.float8_e4m3fn)
    b = np.zeros((64, 64), dtype=ml_dtypes.float8_e5m2)
    a[:m, :k] = logical_a
    b[:k, :n] = logical_b
    a_scale = np.random.randint(127, 130, (m, 2)).astype(np.uint8)
    b_scale = np.random.randint(127, 130, (2, n)).astype(np.uint8)
    a_float = np.zeros((m, k), dtype=np.float64)
    b_float = np.zeros((k, n), dtype=np.float64)
    for i in range(k):
        a_float[:, i] = a[:m, i].astype(np.float64) * 2 ** (a_scale[:, i // 32] - 127)
        b_float[i, :] = b[i, :n].astype(np.float64) * 2 ** (b_scale[i // 32, :] - 127)
    expected = np.zeros((16, 64), dtype=np.float32)
    expected[:m, :n] = (a_float @ b_float).astype(np.float32)
    return [
        _pack_lhs_fp8_shape(a).view(np.uint8), _pack_rhs_fp8_shape(b).view(np.uint8),
        _scale_a_format(a_scale).view(np.uint8), _scale_b_format(b_scale, n_padded=64).view(np.uint8),
    ], expected


@pto.jit(
    name="tmatmul_mx_fp4_e2m1_e1m2_128x118x64",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel_fp4_mixed_128(
    a_ptr: pto.ptr(pto.f4e2m1x2, "gm"),
    b_ptr: pto.ptr(pto.f4e1m2x2, "gm"),
    a_scale_ptr: pto.ptr(pto.f4e2m1x2, "gm"),
    b_scale_ptr: pto.ptr(pto.f4e1m2x2, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    a_l1 = pto.castptr(pto.ui64(0), pto.ptr(pto.f4e2m1x2, "mat"))
    a_scale_l1 = pto.castptr(pto.ui64(8192), pto.ptr(pto.f4e2m1x2, "mat"))
    b_l1 = pto.castptr(pto.ui64(8704), pto.ptr(pto.f4e1m2x2, "mat"))
    b_scale_l1 = pto.castptr(pto.ui64(12800), pto.ptr(pto.f4e1m2x2, "mat"))
    lhs = pto.alloc_tile(shape=[128, 128], dtype=pto.f4e2m1x2, memory_space=pto.MemorySpace.LEFT,
                         addr=0, valid_shape=[128, 128], blayout="ColMajor", slayout="RowMajor", fractal_size=512)
    lhs_scale = pto.alloc_tile(shape=[128, 4], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
                               addr=0, valid_shape=[128, 4], blayout="RowMajor", slayout="RowMajor", fractal_size=32)
    rhs = pto.alloc_tile(shape=[128, 64], dtype=pto.f4e1m2x2, memory_space=pto.MemorySpace.RIGHT,
                         addr=0, valid_shape=[128, 64], blayout="RowMajor", slayout="ColMajor", fractal_size=512)
    rhs_scale = pto.alloc_tile(shape=[4, 64], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
                               addr=0, valid_shape=[4, 64], blayout="ColMajor", slayout="ColMajor", fractal_size=32)
    dst = pto.alloc_tile(shape=[128, 64], dtype=pto.f32, memory_space=pto.MemorySpace.ACC,
                         addr=0, valid_shape=[128, 64], blayout="ColMajor", slayout="RowMajor", fractal_size=512)
    pto.mte_gm_l1_frac(a_ptr, a_l1, pto.FractalMode.ND2NZ, shape=(128, 64), src_layout=(64,),
                       dst_group=(1, 1, 128, 0), ctrl=(0, False))
    pto.mte_gm_l1(a_scale_ptr, a_scale_l1, 64, nburst=(1, 0, 0), loops=[(8, 64, 64)])
    pto.mte_gm_l1_frac(b_ptr, b_l1, pto.FractalMode.ND2NZ, shape=(128, 32), src_layout=(32,),
                       dst_group=(1, 1, 128, 0), ctrl=(0, False))
    pto.mte_gm_l1(b_scale_ptr, b_scale_l1, 64, nburst=(1, 0, 0), loops=[(4, 64, 64)])
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.mte_l1_l0a(a_l1, lhs.as_ptr(), 128, 128)
    pto.mte_l1_l0b(b_l1, rhs.as_ptr(), 128, 64, transpose=True)
    pto.mte_l1_l0a_mx(a_scale_l1, lhs.as_ptr(), 128, 128)
    pto.mte_l1_l0b_mx(b_scale_l1, rhs.as_ptr(), 128, 64)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, dst)
    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.mte_l0c_gm(dst.as_ptr(), c_ptr, 128, 64, 128, 64, 0, 0, layout="nz2nd")
    pto.pipe_barrier(pto.Pipe.ALL)


def _make_fp4_mixed_128_data():
    m, k, n = 128, 118, 64
    np.random.seed(zlib.crc32(b"fp4_e2m1_e1m2_128x118x64") & 0xFFFFFFFF)
    logical_a = np.random.randint(-6, 6, (m, k)).astype(en_dtypes.float4_e2m1)
    logical_b = np.random.randint(-1, 2, (k, n)).astype(en_dtypes.float4_e1m2)
    a = np.zeros((m, 128), dtype=en_dtypes.float4_e2m1)
    b = np.zeros((128, n), dtype=en_dtypes.float4_e1m2)
    a[:, :k] = logical_a
    b[:k, :] = logical_b
    a_scale = np.random.randint(127, 130, (m, 4)).astype(np.uint8)
    b_scale = np.random.randint(127, 130, (4, n)).astype(np.uint8)
    a_float = np.zeros((m, k), dtype=np.float64)
    b_float = np.zeros((k, n), dtype=np.float64)
    for i in range(k):
        a_float[:, i] = a[:, i].astype(np.float64) * 2 ** (a_scale[:, i // 32] - 127)
        b_float[i, :] = b[i, :].astype(np.float64) * 2 ** (b_scale[i // 32, :] - 127)
    return [_pack_two_fp4(a), _pack_two_fp4(b), _scale_a_format(a_scale).view(np.uint8),
            _scale_b_format(b_scale, n_padded=n).view(np.uint8)], (a_float @ b_float).astype(np.float32)


@pto.jit(
    name="tmatmul_mx_fp4_e2m1_e1m2_115x64x30",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel_fp4_mixed_115(
    a_ptr: pto.ptr(pto.f4e2m1x2, "gm"),
    b_ptr: pto.ptr(pto.f4e1m2x2, "gm"),
    a_scale_ptr: pto.ptr(pto.f4e2m1x2, "gm"),
    b_scale_ptr: pto.ptr(pto.f4e1m2x2, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    a_l1 = pto.castptr(pto.ui64(0), pto.ptr(pto.f4e2m1x2, "mat"))
    # Keep the four L1 regions disjoint: A is 4 * 1024 bytes, followed by
    # its scale, B, and B scale, as in the legacy kernel.
    a_scale_l1 = pto.castptr(pto.ui64(4096), pto.ptr(pto.f4e2m1x2, "mat"))
    b_l1 = pto.castptr(pto.ui64(4352), pto.ptr(pto.f4e1m2x2, "mat"))
    b_scale_l1 = pto.castptr(pto.ui64(6400), pto.ptr(pto.f4e1m2x2, "mat"))
    lhs = pto.alloc_tile(shape=[128, 64], dtype=pto.f4e2m1x2, memory_space=pto.MemorySpace.LEFT,
                         addr=0, valid_shape=[115, 64], blayout="ColMajor", slayout="RowMajor", fractal_size=512)
    lhs_scale = pto.alloc_tile(shape=[128, 2], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
                               addr=0, valid_shape=[115, 2], blayout="RowMajor", slayout="RowMajor", fractal_size=32)
    rhs = pto.alloc_tile(shape=[64, 64], dtype=pto.f4e1m2x2, memory_space=pto.MemorySpace.RIGHT,
                         addr=0, valid_shape=[64, 30], blayout="RowMajor", slayout="ColMajor", fractal_size=512)
    rhs_scale = pto.alloc_tile(shape=[2, 64], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
                               addr=0, valid_shape=[2, 30], blayout="ColMajor", slayout="ColMajor", fractal_size=32)
    dst = pto.alloc_tile(shape=[128, 64], dtype=pto.f32, memory_space=pto.MemorySpace.ACC,
                         addr=0, valid_shape=[115, 30], blayout="ColMajor", slayout="RowMajor", fractal_size=512)
    pto.mte_gm_l1(a_ptr, a_l1, 1024, nburst=(1, 0, 0), loops=[(4, 1024, 1024)])
    pto.mte_gm_l1(a_scale_ptr, a_scale_l1, 64, nburst=(1, 0, 0), loops=[(4, 64, 64)])
    pto.mte_gm_l1(b_ptr, b_l1, 1024, nburst=(1, 0, 0), loops=[(2, 1024, 1024)])
    pto.mte_gm_l1(b_scale_ptr, b_scale_l1, 64, nburst=(1, 0, 0), loops=[(2, 64, 64)])
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.mte_l1_l0a(a_l1, lhs.as_ptr(), 115, 64)
    pto.mte_l1_l0b(b_l1, rhs.as_ptr(), 64, 30, transpose=True)
    pto.mte_l1_l0a_mx(a_scale_l1, lhs.as_ptr(), 115, 64)
    pto.mte_l1_l0b_mx(b_scale_l1, rhs.as_ptr(), 64, 30)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, dst)
    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.mte_l0c_gm(dst.as_ptr(), c_ptr, 128, 64, 128, 64, 0, 0, layout="nz2nd")
    pto.pipe_barrier(pto.Pipe.ALL)


def _make_fp4_mixed_115_data():
    m, k, n = 115, 64, 30
    np.random.seed(zlib.crc32(b"fp4_e2m1_e1m2_115x64x30") & 0xFFFFFFFF)
    logical_a = np.random.randint(-6, 6, (m, k)).astype(en_dtypes.float4_e2m1)
    logical_b = np.random.randint(-1, 2, (k, n)).astype(en_dtypes.float4_e1m2)
    a = np.zeros((128, 64), dtype=en_dtypes.float4_e2m1)
    b = np.zeros((64, 64), dtype=en_dtypes.float4_e1m2)
    a[:m] = logical_a
    b[:, :n] = logical_b
    a_scale = np.random.randint(127, 130, (m, 2)).astype(np.uint8)
    b_scale = np.random.randint(127, 130, (2, n)).astype(np.uint8)
    a_float = np.zeros((m, k), dtype=np.float64)
    b_float = np.zeros((k, n), dtype=np.float64)
    for i in range(k):
        a_float[:, i] = a[:m, i].astype(np.float64) * 2 ** (a_scale[:, i // 32] - 127)
        b_float[i, :] = b[i, :n].astype(np.float64) * 2 ** (b_scale[i // 32, :] - 127)
    expected = np.zeros((128, 64), dtype=np.float32)
    expected[:m, :n] = (a_float @ b_float).astype(np.float32)
    return [_pack_two_fp4(a), _pack_two_fp4(b), _scale_a_format(a_scale).view(np.uint8),
            _scale_b_format(b_scale, n_padded=64).view(np.uint8)], expected


@pto.jit(
    name="tmatmul_mx_fp4_e2m1_4x30x8",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel_fp4_small(
    a_ptr: pto.ptr(pto.f4e2m1x2, "gm"),
    b_ptr: pto.ptr(pto.f4e2m1x2, "gm"),
    a_scale_ptr: pto.ptr(pto.f4e2m1x2, "gm"),
    b_scale_ptr: pto.ptr(pto.f4e2m1x2, "gm"),
    c_ptr: pto.ptr(pto.f32, "gm"),
):
    a_l1 = pto.castptr(pto.ui64(0), pto.ptr(pto.f4e2m1x2, "mat"))
    a_scale_l1 = pto.castptr(pto.ui64(1024), pto.ptr(pto.f4e2m1x2, "mat"))
    b_l1 = pto.castptr(pto.ui64(1088), pto.ptr(pto.f4e2m1x2, "mat"))
    b_scale_l1 = pto.castptr(pto.ui64(3136), pto.ptr(pto.f4e2m1x2, "mat"))
    lhs = pto.alloc_tile(shape=[16, 64], dtype=pto.f4e2m1x2, memory_space=pto.MemorySpace.LEFT,
                         addr=0, valid_shape=[4, 64], blayout="ColMajor", slayout="RowMajor", fractal_size=512)
    lhs_scale = pto.alloc_tile(shape=[16, 2], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
                               addr=0, valid_shape=[4, 2], blayout="RowMajor", slayout="RowMajor", fractal_size=32)
    rhs = pto.alloc_tile(shape=[64, 64], dtype=pto.f4e2m1x2, memory_space=pto.MemorySpace.RIGHT,
                         addr=0, valid_shape=[64, 8], blayout="RowMajor", slayout="ColMajor", fractal_size=512)
    rhs_scale = pto.alloc_tile(shape=[2, 64], dtype=pto.f16, memory_space=pto.MemorySpace.SCALING,
                               addr=0, valid_shape=[2, 8], blayout="ColMajor", slayout="ColMajor", fractal_size=32)
    dst = pto.alloc_tile(shape=[16, 64], dtype=pto.f32, memory_space=pto.MemorySpace.ACC,
                         addr=0, valid_shape=[4, 8], blayout="ColMajor", slayout="RowMajor", fractal_size=512)
    pto.mte_gm_l1(a_ptr, a_l1, 1024, nburst=(1, 0, 0))
    pto.mte_gm_l1(a_scale_ptr, a_scale_l1, 32, nburst=(1, 0, 0))
    pto.mte_gm_l1(b_ptr, b_l1, 1024, nburst=(1, 0, 0), loops=[(2, 1024, 1024)])
    pto.mte_gm_l1(b_scale_ptr, b_scale_l1, 64, nburst=(1, 0, 0), loops=[(2, 64, 64)])
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.mte_l1_l0a(a_l1, lhs.as_ptr(), 4, 64)
    pto.mte_l1_l0b(b_l1, rhs.as_ptr(), 64, 8, transpose=True)
    pto.mte_l1_l0a_mx(a_scale_l1, lhs.as_ptr(), 4, 64)
    pto.mte_l1_l0b_mx(b_scale_l1, rhs.as_ptr(), 64, 8)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
    pto.tile.matmul_mx(lhs, lhs_scale, rhs, rhs_scale, dst)
    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
    pto.mte_l0c_gm(dst.as_ptr(), c_ptr, 16, 64, 16, 64, 0, 0, layout="nz2nd")
    pto.pipe_barrier(pto.Pipe.ALL)


def _make_fp4_small_data():
    m, k, n = 4, 30, 8
    np.random.seed(zlib.crc32(b"fp4_e2m1_4x30x8") & 0xFFFFFFFF)
    logical_a = np.random.randint(-6, 6, (m, k)).astype(en_dtypes.float4_e2m1)
    logical_b = np.random.randint(-6, 6, (k, n)).astype(en_dtypes.float4_e2m1)
    a = np.zeros((16, 64), dtype=en_dtypes.float4_e2m1)
    b = np.zeros((64, 64), dtype=en_dtypes.float4_e2m1)
    a[:m, :k] = logical_a
    b[:k, :n] = logical_b
    a_scale = np.random.randint(127, 130, (m, 2)).astype(np.uint8)
    b_scale = np.random.randint(127, 130, (2, n)).astype(np.uint8)
    a_float = np.zeros((m, k), dtype=np.float64)
    b_float = np.zeros((k, n), dtype=np.float64)
    for i in range(k):
        a_float[:, i] = a[:m, i].astype(np.float64) * 2 ** (a_scale[:, i // 32] - 127)
        b_float[i, :] = b[i, :n].astype(np.float64) * 2 ** (b_scale[i // 32, :] - 127)
    expected = np.zeros((16, 64), dtype=np.float32)
    expected[:m, :n] = (a_float @ b_float).astype(np.float32)
    return [_pack_two_fp4(a), _pack_two_fp4(b), _scale_a_format(a_scale).view(np.uint8),
            _scale_b_format(b_scale, n_padded=64).view(np.uint8)], expected


CASES = [
    golden_output_case("tmatmul_mx_fp8_e5m2_128x64x64", _kernel, inputs=_inputs,
                       expected=_expected, rtol=1e-3, atol=1e-3),
    golden_output_case("tmatmul_mx_fp8_e4m3_16x32x16", _kernel_small,
                       inputs=lambda: _make_small_data()[0],
                       expected=lambda *_: _make_small_data()[1], rtol=1e-3, atol=1e-3),
    golden_output_case("tmatmul_mx_fp8_e4m3_127x72x64", _kernel_fp8_127,
                       inputs=lambda: _make_127_data()[0],
                       expected=lambda *_: _make_127_data()[1], rtol=1e-3, atol=1e-3),
    golden_output_case("tmatmul_mx_fp8_e4m3_e5m2_128x110x63", _kernel_fp8_mixed,
                       inputs=lambda: _make_mixed_data()[0],
                       expected=lambda *_: _make_mixed_data()[1], rtol=1e-3, atol=1e-3),
    golden_output_case("tmatmul_mx_fp4_e2m1_128x64x64", _kernel_fp4_128,
                       inputs=lambda: _make_fp4_128_data()[0],
                       expected=lambda *_: _make_fp4_128_data()[1], rtol=1e-3, atol=1e-3),
    golden_output_case("tmatmul_mx_fp4_e1m2_e2m1_117x64x60", _kernel_fp4_117,
                       inputs=lambda: _make_fp4_117_data()[0],
                       expected=lambda *_: _make_fp4_117_data()[1], rtol=1e-3, atol=1e-3),
    golden_output_case("tmatmul_mx_fp8_e4m3_e5m2_10x50x54", _kernel_fp8_small,
                       inputs=lambda: _make_fp8_small_data()[0],
                       expected=lambda *_: _make_fp8_small_data()[1], rtol=1e-3, atol=1e-3),
    golden_output_case("tmatmul_mx_fp4_e2m1_e1m2_128x118x64", _kernel_fp4_mixed_128,
                       inputs=lambda: _make_fp4_mixed_128_data()[0],
                       expected=lambda *_: _make_fp4_mixed_128_data()[1], rtol=1e-3, atol=1e-3),
    golden_output_case("tmatmul_mx_fp4_e2m1_e1m2_115x64x30", _kernel_fp4_mixed_115,
                       inputs=lambda: _make_fp4_mixed_115_data()[0],
                       expected=lambda *_: _make_fp4_mixed_115_data()[1], rtol=1e-3, atol=1e-3),
    golden_output_case("tmatmul_mx_fp4_e2m1_4x30x8", _kernel_fp4_small,
                       inputs=lambda: _make_fp4_small_data()[0],
                       expected=lambda *_: _make_fp4_small_data()[1], rtol=1e-3, atol=1e-3),
]

auto_main(globals())
