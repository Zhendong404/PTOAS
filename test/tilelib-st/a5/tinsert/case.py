#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tinsert.
# Legacy kernels are cube kernels (pto.kernel_kind<cube>) testing the
# pto.tinsert Acc->Mat path:
#   GM -> L1 MAT (mte_gm_l1_frac)
#   tmatmul(A, B) -> ACC
#   tinsert(acc -> mat dst, index_row=0, index_col=0)
#   readback staged via mte_l1_l0a (dst_mat) / mte_l1_l0b (id)
#   tmatmul(dst_mat, id) -> acc -> GM (tile.store)
# The pto.tile.insert surface (src, dst, index_row, index_col) matches the
# legacy pto.tinsert ins(acc, 0, 0) outs(mat) 1:1; shape/layout come from the
# tile declarations preserved from the legacy .pto.
#
# This case keeps explicit L1/L0 addresses because the cube staging templates
# consume explicit MAT/LEFT/RIGHT tile buffers. GM->MAT and ACC->GM boundaries
# use the public tile.load/store surface (same as a5/tmatmul/case.py), while
# the tinsert Acc->Mat path and the MAT/LEFT/RIGHT readback staging stay
# explicit. The op is authored with mode="explicit", not the auto-mode vector
# form: auto mode cannot express ACC/MAT tiles or the explicit GM/L1/L0
# movement pipeline this op is testing.
#
# The two variants are authored as two straight-line kernels, mirroring the
# legacy .pto which defines one func.func per variant (f16 / bf16); the
# bf16 readback uses dedicated LEFT/RIGHT tiles as in the legacy kernel.
#
# bf16 host input is passed as raw 16-bit storage (uint16 bit patterns),
# identical to the legacy launcher which declared bf16 buffers as uint16_t
# (same convention as a5/tcvt/case.py).

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto

M = 16
K = 16
N = 16
ELEM_BYTES = 2  # f16 / bf16
F16_TILE_BYTES = M * K * ELEM_BYTES  # 512-byte L1 MAT tile

# Legacy L1 MAT addresses (bytes): a at 0, b at 512, id at 1024; dst_mat
# reuses address 0 (a is consumed before tinsert writes dst_mat).
L1_A_ADDR = 0
L1_B_ADDR = F16_TILE_BYTES
L1_ID_ADDR = 2 * F16_TILE_BYTES
L1_DST_ADDR = 0
L0A_ADDR = 0
L0B_ADDR = 0
L0C_ADDR = 0

EPS = 1e-2  # legacy eps for both cases


def _f32_to_bf16_roundtrip(f32_arr):
    """Legacy bf16 truncation simulation (round-to-nearest-even, 16-bit)."""
    as_uint32 = f32_arr.view(np.uint32)
    rounded = ((as_uint32 + np.uint32(0x7FFF)) & np.uint32(0xFFFF0000))
    return rounded.view(np.float32)


def _make_f16_kernel():
    """TINSERT_acc2mat_f16_16x16: f16 dst_mat, readback reuses l0a/l0b."""
    dst_dtype = pto.f16

    @pto.jit(
        name="tinsert_acc2mat_f16_16x16",
        kernel_kind="cube",
        target="a5",
        mode="explicit",
        insert_sync=False,
    )
    def _kernel(
        a_ptr: pto.ptr(pto.f16, "gm"),
        b_ptr: pto.ptr(pto.f16, "gm"),
        id_ptr: pto.ptr(pto.f16, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        l1_a = pto.alloc_tile(
            shape=[M, K],
            dtype=pto.f16,
            memory_space=pto.MemorySpace.MAT,
            addr=L1_A_ADDR,
            valid_shape=[M, K],
            blayout="ColMajor",
            slayout="RowMajor",
        )
        l1_b = pto.alloc_tile(
            shape=[K, N],
            dtype=pto.f16,
            memory_space=pto.MemorySpace.MAT,
            addr=L1_B_ADDR,
            valid_shape=[K, N],
            blayout="ColMajor",
            slayout="RowMajor",
        )
        l1_id = pto.alloc_tile(
            shape=[M, N],
            dtype=dst_dtype,
            memory_space=pto.MemorySpace.MAT,
            addr=L1_ID_ADDR,
            valid_shape=[M, N],
            blayout="ColMajor",
            slayout="RowMajor",
        )
        l0a = pto.alloc_tile(
            shape=[M, K],
            dtype=pto.f16,
            memory_space=pto.MemorySpace.LEFT,
            addr=L0A_ADDR,
            valid_shape=[M, K],
            blayout="ColMajor",
            slayout="RowMajor",
        )
        l0b = pto.alloc_tile(
            shape=[K, N],
            dtype=pto.f16,
            memory_space=pto.MemorySpace.RIGHT,
            addr=L0B_ADDR,
            valid_shape=[K, N],
            blayout="RowMajor",
            slayout="ColMajor",
        )
        l0c = pto.alloc_tile(
            shape=[M, N],
            dtype=pto.f32,
            memory_space=pto.MemorySpace.ACC,
            addr=L0C_ADDR,
            valid_shape=[M, N],
            blayout="ColMajor",
            slayout="RowMajor",
            fractal_size=1024,
        )
        dst_mat = pto.alloc_tile(
            shape=[M, N],
            dtype=dst_dtype,
            memory_space=pto.MemorySpace.MAT,
            addr=L1_DST_ADDR,
            valid_shape=[M, N],
            blayout="ColMajor",
            slayout="RowMajor",
        )

        a_view = pto.make_tensor_view(
            a_ptr,
            shape=[1, 1, 1, M, K],
            strides=[M * K, M * K, M * K, K, 1],
        )
        b_view = pto.make_tensor_view(
            b_ptr,
            shape=[1, 1, 1, K, N],
            strides=[K * N, K * N, K * N, N, 1],
        )
        id_view = pto.make_tensor_view(
            id_ptr,
            shape=[1, 1, 1, M, N],
            strides=[M * N, M * N, M * N, N, 1],
        )
        out_view = pto.make_tensor_view(
            out_ptr,
            shape=[1, 1, 1, M, N],
            strides=[M * N, M * N, M * N, N, 1],
        )
        a_shape = [1, 1, 1, M, K]
        b_shape = [1, 1, 1, K, N]
        id_shape = [1, 1, 1, M, N]
        out_shape = [1, 1, 1, M, N]

        # Stage A -> L1 -> L0A.
        pto.tile.load(a_view, l1_a, offsets=[0, 0, 0, 0, 0], sizes=a_shape)
        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
        pto.mte_l1_l0a(l1_a.as_ptr(), l0a.as_ptr(), M, K)

        # Stage B -> L1 -> L0B (transposed).
        pto.tile.load(b_view, l1_b, offsets=[0, 0, 0, 0, 0], sizes=b_shape)
        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
        pto.mte_l1_l0b(l1_b.as_ptr(), l0b.as_ptr(), K, N, transpose=True)

        pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
        pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
        pto.tile.matmul(l0a, l0b, l0c)

        # Acc -> Mat via tinsert at offset (0, 0).
        pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
        pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
        pto.tile.insert(l0c, dst_mat, 0, 0)

        # Stage id -> L1, then readback dst_mat / id -> L0 -> matmul.
        pto.set_flag(pto.Pipe.FIX, pto.Pipe.MTE1, event_id=0)
        pto.tile.load(id_view, l1_id, offsets=[0, 0, 0, 0, 0], sizes=id_shape)
        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
        pto.wait_flag(pto.Pipe.FIX, pto.Pipe.MTE1, event_id=0)
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)

        pto.mte_l1_l0a(dst_mat.as_ptr(), l0a.as_ptr(), M, N)
        pto.mte_l1_l0b(l1_id.as_ptr(), l0b.as_ptr(), M, N, transpose=True)
        pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
        pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
        pto.tile.matmul(l0a, l0b, l0c)

        pto.tile.store(
            l0c,
            out_view,
            offsets=[0, 0, 0, 0, 0],
            sizes=out_shape,
        )
        pto.pipe_barrier(pto.Pipe.ALL)

    return _kernel


def _make_bf16_kernel():
    """TINSERT_acc2mat_bf16_16x16: bf16 dst_mat, readback uses l0a2/l0b2."""
    dst_dtype = pto.bf16

    @pto.jit(
        name="tinsert_acc2mat_bf16_16x16",
        kernel_kind="cube",
        target="a5",
        mode="explicit",
        insert_sync=False,
    )
    def _kernel(
        a_ptr: pto.ptr(pto.f16, "gm"),
        b_ptr: pto.ptr(pto.f16, "gm"),
        id_ptr: pto.ptr(pto.bf16, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        l1_a = pto.alloc_tile(
            shape=[M, K],
            dtype=pto.f16,
            memory_space=pto.MemorySpace.MAT,
            addr=L1_A_ADDR,
            valid_shape=[M, K],
            blayout="ColMajor",
            slayout="RowMajor",
        )
        l1_b = pto.alloc_tile(
            shape=[K, N],
            dtype=pto.f16,
            memory_space=pto.MemorySpace.MAT,
            addr=L1_B_ADDR,
            valid_shape=[K, N],
            blayout="ColMajor",
            slayout="RowMajor",
        )
        l1_id = pto.alloc_tile(
            shape=[M, N],
            dtype=dst_dtype,
            memory_space=pto.MemorySpace.MAT,
            addr=L1_ID_ADDR,
            valid_shape=[M, N],
            blayout="ColMajor",
            slayout="RowMajor",
        )
        l0a = pto.alloc_tile(
            shape=[M, K],
            dtype=pto.f16,
            memory_space=pto.MemorySpace.LEFT,
            addr=L0A_ADDR,
            valid_shape=[M, K],
            blayout="ColMajor",
            slayout="RowMajor",
        )
        l0b = pto.alloc_tile(
            shape=[K, N],
            dtype=pto.f16,
            memory_space=pto.MemorySpace.RIGHT,
            addr=L0B_ADDR,
            valid_shape=[K, N],
            blayout="RowMajor",
            slayout="ColMajor",
        )
        l0c = pto.alloc_tile(
            shape=[M, N],
            dtype=pto.f32,
            memory_space=pto.MemorySpace.ACC,
            addr=L0C_ADDR,
            valid_shape=[M, N],
            blayout="ColMajor",
            slayout="RowMajor",
            fractal_size=1024,
        )
        dst_mat = pto.alloc_tile(
            shape=[M, N],
            dtype=dst_dtype,
            memory_space=pto.MemorySpace.MAT,
            addr=L1_DST_ADDR,
            valid_shape=[M, N],
            blayout="ColMajor",
            slayout="RowMajor",
        )
        l0a2 = pto.alloc_tile(
            shape=[M, K],
            dtype=dst_dtype,
            memory_space=pto.MemorySpace.LEFT,
            addr=L0A_ADDR,
            valid_shape=[M, K],
            blayout="ColMajor",
            slayout="RowMajor",
        )
        l0b2 = pto.alloc_tile(
            shape=[K, N],
            dtype=dst_dtype,
            memory_space=pto.MemorySpace.RIGHT,
            addr=L0B_ADDR,
            valid_shape=[K, N],
            blayout="RowMajor",
            slayout="ColMajor",
        )

        a_view = pto.make_tensor_view(
            a_ptr,
            shape=[1, 1, 1, M, K],
            strides=[M * K, M * K, M * K, K, 1],
        )
        b_view = pto.make_tensor_view(
            b_ptr,
            shape=[1, 1, 1, K, N],
            strides=[K * N, K * N, K * N, N, 1],
        )
        id_view = pto.make_tensor_view(
            id_ptr,
            shape=[1, 1, 1, M, N],
            strides=[M * N, M * N, M * N, N, 1],
        )
        out_view = pto.make_tensor_view(
            out_ptr,
            shape=[1, 1, 1, M, N],
            strides=[M * N, M * N, M * N, N, 1],
        )
        a_shape = [1, 1, 1, M, K]
        b_shape = [1, 1, 1, K, N]
        id_shape = [1, 1, 1, M, N]
        out_shape = [1, 1, 1, M, N]

        # Stage A -> L1 -> L0A.
        pto.tile.load(a_view, l1_a, offsets=[0, 0, 0, 0, 0], sizes=a_shape)
        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
        pto.mte_l1_l0a(l1_a.as_ptr(), l0a.as_ptr(), M, K)

        # Stage B -> L1 -> L0B (transposed).
        pto.tile.load(b_view, l1_b, offsets=[0, 0, 0, 0, 0], sizes=b_shape)
        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
        pto.mte_l1_l0b(l1_b.as_ptr(), l0b.as_ptr(), K, N, transpose=True)

        pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
        pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
        pto.tile.matmul(l0a, l0b, l0c)

        # Acc -> Mat via tinsert at offset (0, 0).
        pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
        pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
        pto.tile.insert(l0c, dst_mat, 0, 0)

        # Stage id -> L1, then readback dst_mat / id -> L0 -> matmul.
        pto.set_flag(pto.Pipe.FIX, pto.Pipe.MTE1, event_id=0)
        pto.tile.load(id_view, l1_id, offsets=[0, 0, 0, 0, 0], sizes=id_shape)
        pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
        pto.wait_flag(pto.Pipe.FIX, pto.Pipe.MTE1, event_id=0)
        pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)

        pto.mte_l1_l0a(dst_mat.as_ptr(), l0a2.as_ptr(), M, N)
        pto.mte_l1_l0b(l1_id.as_ptr(), l0b2.as_ptr(), M, N, transpose=True)
        pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
        pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
        pto.tile.matmul(l0a2, l0b2, l0c)

        pto.tile.store(
            l0c,
            out_view,
            offsets=[0, 0, 0, 0, 0],
            sizes=out_shape,
        )
        pto.pipe_barrier(pto.Pipe.ALL)

    return _kernel


_kernels = {
    "acc2mat_f16_16x16": _make_f16_kernel(),
    "acc2mat_bf16_16x16": _make_bf16_kernel(),
}


def _make_inputs(name: str):
    """Deterministic per-case seed, mirroring st_common.setup_case_rng.

    The seed uses the *legacy* case name so generated data is identical to the
    legacy gen_data.py output.
    """
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    a = np.random.uniform(-1.0, 1.0, size=(M, K)).astype(np.float16)
    b = np.random.uniform(-1.0, 1.0, size=(K, N)).astype(np.float16)
    id_mat = np.eye(M, dtype=np.float32)
    if name == "acc2mat_bf16_16x16":
        # Legacy id_dtype = uint16: raw bf16 bit patterns of the identity.
        id_as_f32 = _f32_to_bf16_roundtrip(id_mat)
        id_bits = (id_as_f32.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
        return [a, b, id_bits]
    # Legacy id_dtype = f16 identity.
    return [a, b, id_mat.astype(np.float16)]


def _make_expected(name: str):
    def expected(a, b, id_mat_or_bits):
        matmul_f32 = np.matmul(a.astype(np.float32), b.astype(np.float32))
        if name == "acc2mat_bf16_16x16":
            quantized = _f32_to_bf16_roundtrip(matmul_f32)
        else:
            quantized = matmul_f32.astype(np.float16).astype(np.float32)
        # Legacy golden: matmul(quantized, eye(f32)); identity keeps it exact.
        id_f32 = np.eye(M, dtype=np.float32)
        return np.matmul(quantized, id_f32).astype(np.float32)

    return expected


CASES = [
    golden_output_case(
        name,
        _kernels[name],
        inputs=lambda name=name: _make_inputs(name),
        expected=_make_expected(name),
        rtol=EPS,
        atol=EPS,
    )
    for name in ("acc2mat_f16_16x16", "acc2mat_bf16_16x16")
]


auto_main(globals())
