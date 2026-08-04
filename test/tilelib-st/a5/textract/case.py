#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/textract.
# Legacy kernel is a cube kernel (pto.kernel_kind<cube>) testing the
# textract MAT->LEFT (M2L) / MAT->RIGHT (M2R) paths:
#   GM -> L1 MAT (mte_gm_l1_frac)
#   textract(src_mat -> left/right, index_row=0, index_col=0)
#   readback operand staged via mte_l1_l0a (M2R) / mte_l1_l0b (M2L)
#   tmatmul(left, right) -> acc -> GM (tile.store)
# This case keeps explicit L1/L0 addresses because the cube staging templates
# consume explicit MAT/LEFT/RIGHT tile buffers. GM->MAT and ACC->GM boundaries
# use the public tile.load/store surface (same as a5/tmatmul/case.py), while
# the textract MAT->LEFT/RIGHT (M2L/M2R) paths stay explicit. The op is
# authored with mode="explicit", not the auto-mode vector form.

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto

ROWS = 16
COLS = 16
ELEM_BYTES = 2  # f16
F16_TILE_BYTES = ROWS * COLS * ELEM_BYTES  # 512-byte L1 MAT tile

L1_SRC_ADDR = 0
L1_ID_ADDR = F16_TILE_BYTES
L0A_ADDR = 0
L0B_ADDR = 0
L0C_ADDR = 0
EPS = 1e-2


def _make_kernel(direction: str):
    """Build one explicit-mode cube kernel per textract path.

    direction == "mat2left":  textract(src_mat -> left); id staged via mte_l1_l0b
    direction == "mat2right": textract(src_mat -> right); id staged via mte_l1_l0a
    Mirrors the legacy textract.pto functions TEXTRACT_M2L_f16_16x16 /
    TEXTRACT_M2R_f16_16x16 (tile shapes, layouts, addresses, offsets, and sync
    ordering preserved 1:1).
    """
    if direction == "mat2left":
        kernel_name = "textract_m2l_f16_16x16"
        src_addr = L1_SRC_ADDR
        id_addr = L1_ID_ADDR
    else:
        kernel_name = "textract_m2r_f16_16x16"
        id_addr = L1_SRC_ADDR
        src_addr = L1_ID_ADDR

    @pto.jit(
        name=kernel_name,
        kernel_kind="cube",
        target="a5",
        mode="explicit",
        insert_sync=False,
    )
    def _kernel(
        src_ptr: pto.ptr(pto.f16, "gm"),
        id_ptr: pto.ptr(pto.f16, "gm"),
        out_ptr: pto.ptr(pto.f32, "gm"),
    ):
        src_mat = pto.alloc_tile(
            shape=[ROWS, COLS],
            dtype=pto.f16,
            memory_space=pto.MemorySpace.MAT,
            addr=src_addr,
            valid_shape=[ROWS, COLS],
            blayout="ColMajor",
            slayout="RowMajor",
        )
        id_mat = pto.alloc_tile(
            shape=[ROWS, COLS],
            dtype=pto.f16,
            memory_space=pto.MemorySpace.MAT,
            addr=id_addr,
            valid_shape=[ROWS, COLS],
            blayout="ColMajor",
            slayout="RowMajor",
        )
        left_tile = pto.alloc_tile(
            shape=[ROWS, COLS],
            dtype=pto.f16,
            memory_space=pto.MemorySpace.LEFT,
            addr=L0A_ADDR,
            valid_shape=[ROWS, COLS],
            blayout="ColMajor",
            slayout="RowMajor",
        )
        right_tile = pto.alloc_tile(
            shape=[ROWS, COLS],
            dtype=pto.f16,
            memory_space=pto.MemorySpace.RIGHT,
            addr=L0B_ADDR,
            valid_shape=[ROWS, COLS],
            blayout="RowMajor",
            slayout="ColMajor",
        )
        acc_tile = pto.alloc_tile(
            shape=[ROWS, COLS],
            dtype=pto.f32,
            memory_space=pto.MemorySpace.ACC,
            addr=L0C_ADDR,
            valid_shape=[ROWS, COLS],
            blayout="ColMajor",
            slayout="RowMajor",
            fractal_size=1024,
        )

        src_view = pto.make_tensor_view(
            src_ptr,
            shape=[1, 1, 1, ROWS, COLS],
            strides=[ROWS * COLS, ROWS * COLS, ROWS * COLS, COLS, 1],
        )
        id_view = pto.make_tensor_view(
            id_ptr,
            shape=[1, 1, 1, ROWS, COLS],
            strides=[ROWS * COLS, ROWS * COLS, ROWS * COLS, COLS, 1],
        )
        out_view = pto.make_tensor_view(
            out_ptr,
            shape=[1, 1, 1, ROWS, COLS],
            strides=[ROWS * COLS, ROWS * COLS, ROWS * COLS, COLS, 1],
        )
        src_shape = [1, 1, 1, ROWS, COLS]
        id_shape = [1, 1, 1, ROWS, COLS]
        out_shape = [1, 1, 1, ROWS, COLS]

        if direction == "mat2left":
            pto.tile.load(src_view, src_mat, offsets=[0, 0, 0, 0, 0], sizes=src_shape)
            pto.tile.load(id_view, id_mat, offsets=[0, 0, 0, 0, 0], sizes=id_shape)
            pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
            pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)

            pto.tile.extract(src_mat, left_tile, 0, 0)  # MAT -> LEFT, offset (0, 0)

            pto.mte_l1_l0b(id_mat.as_ptr(), right_tile.as_ptr(), ROWS, COLS, transpose=True)
            pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
            pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
        else:
            # The identity is consumed through the L1->L0A path here.  Its
            # physical arrangement is likewise not preserved by the public
            # MAT load template for this layout-sensitive path.
            id_l1 = pto.castptr(pto.ui64(id_addr), pto.ptr(pto.f16, "mat"))
            pto.mte_gm_l1_frac(
                id_ptr,
                id_l1,
                pto.FractalMode.ND2NZ,
                shape=(ROWS, COLS),
                src_layout=(COLS * ELEM_BYTES,),
                dst_group=(1, 1, ROWS, 0),
                ctrl=(0, False),
            )
            pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
            pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)

            pto.mte_l1_l0a(id_mat.as_ptr(), left_tile.as_ptr(), ROWS, COLS)

            # The MAT->RIGHT path is layout-sensitive: the public MAT load
            # template currently produces a different physical arrangement
            # at this nonzero MAT address, so retain the legacy fractal load
            # for this operand until that template is corrected.
            src_l1 = pto.castptr(pto.ui64(src_addr), pto.ptr(pto.f16, "mat"))
            pto.mte_gm_l1_frac(
                src_ptr,
                src_l1,
                pto.FractalMode.ND2NZ,
                shape=(ROWS, COLS),
                src_layout=(COLS * ELEM_BYTES,),
                dst_group=(1, 1, ROWS, 0),
                ctrl=(0, False),
            )
            pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)
            pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=1)

            pto.tile.extract(src_mat, right_tile, 0, 0)  # MAT -> RIGHT, offset (0, 0)

            pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)
            pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=0)

        pto.tile.matmul(left_tile, right_tile, acc_tile)

        pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
        pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=1)
        pto.tile.store(
            acc_tile,
            out_view,
            offsets=[0, 0, 0, 0, 0],
            sizes=out_shape,
        )
        pto.pipe_barrier(pto.Pipe.ALL)

    return _kernel


# (case name, direction); names and eps match legacy cases.py exactly.
CASE_SPECS = [
    ("mat2left_f16_16x16", "mat2left"),
    ("mat2right_f16_16x16", "mat2right"),
]

_kernels = {name: _make_kernel(direction) for name, direction in CASE_SPECS}


def _make_inputs(name: str, direction: str):
    # RNG parity with legacy st_common.setup_case_rng: per-case deterministic seed.
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    src = np.random.uniform(-1.0, 1.0, size=(ROWS, COLS)).astype(np.float16)
    id_mat = np.eye(ROWS, COLS, dtype=np.float16)
    if direction == "mat2left":
        # Kernel arg order for M2L is (src, id); input1=src, input2=id.
        return [src, id_mat]
    # Kernel arg order for M2R is (id, src); input1=id, input2=src.
    return [id_mat, src]


def _make_expected(direction: str):
    if direction == "mat2left":
        def expected(src, id_mat):
            return np.matmul(src.astype(np.float32), id_mat.astype(np.float32)).astype(np.float32)
        return expected

    def expected(id_mat, src):
        return src.astype(np.float32).T.copy()
    return expected


CASES = [
    golden_output_case(
        name,
        _kernels[name],
        inputs=lambda name=name, direction=direction: _make_inputs(name, direction),
        expected=_make_expected(direction),
        rtol=EPS,
        atol=EPS,
    )
    for name, direction in CASE_SPECS
]


auto_main(globals())
