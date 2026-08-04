#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tcmp.
#
# tcmp(a, b) -> packed predicate mask c: 1 bit per compared element, stored as
# an i8 tile of the same physical shape as the inputs.  Only the first
# (valid_cols // 8) bytes of each valid row carry meaningful data; the rest of
# the output tile is padding whose contents are undefined, so the legacy
# compare.py compared only output[:vr, :vc // 8] against the same golden slice
# (see Step 3/4 of the migration skill and the note below _packed_mask_case).

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main
from ptodsl import pto

# (case suffix, numpy dtype, pto dtype, shape, valid_shape, cmp_mode)
# Mirrors testcase/tcmp/cases.py exactly (name, dtype, shape, valid_shape,
# dst_dtype=i8, cmp_mode, eps=0).
CASE_SPECS = [
    ("f16_32x32_eq",   np.float16, pto.f16, (32, 32),    (32, 32),    "eq"),
    ("f32_8x64_gt",    np.float32, pto.f32, (8, 64),     (8, 64),     "gt"),
    ("i32_4x64_ne",    np.int32,   pto.i32, (4, 64),     (4, 64),     "ne"),
    ("i32_128x128_lt", np.int32,   pto.i32, (128, 128),  (64, 64),    "lt"),
    ("i32_64x64_eq",   np.int32,   pto.i32, (64, 64),    (32, 32),    "eq"),
    ("i32_16x32_eq",   np.int32,   pto.i32, (16, 32),    (16, 32),    "eq"),
    ("f32_128x128_le", np.float32, pto.f32, (128, 128),  (64, 64),    "le"),
    ("i32_77x96_eq",   np.int32,   pto.i32, (77, 96),    (32, 32),    "eq"),
    ("i32_32x32_eq",   np.int32,   pto.i32, (32, 32),    (32, 32),    "eq"),
    ("i16_32x32_eq",   np.int16,   pto.i16, (32, 32),    (16, 32),    "eq"),
    ("i16_77x96_le",   np.int16,   pto.i16, (77, 96),    (32, 32),    "le"),
]


def _make_kernel(name, pto_dtype, shape, valid_shape, cmp_mode):
    rows, cols = shape
    vr, vc = valid_shape

    @pto.jit(name="tcmp_" + name, target="a5")
    def _kernel(
        a_ptr: pto.ptr(pto_dtype, "gm"),
        b_ptr: pto.ptr(pto_dtype, "gm"),
        c_ptr: pto.ptr(pto.i8, "gm"),
    ):
        a_view = pto.make_tensor_view(a_ptr, shape=[rows, cols], strides=[cols, 1])
        b_view = pto.make_tensor_view(b_ptr, shape=[rows, cols], strides=[cols, 1])
        c_view = pto.make_tensor_view(c_ptr, shape=[rows, cols], strides=[cols, 1])

        a_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype, valid_shape=[vr, vc])
        b_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype, valid_shape=[vr, vc])
        c_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto.i8, valid_shape=[vr, vc])

        pto.tile.load(a_view, a_tile)
        pto.tile.load(b_view, b_tile)
        pto.tile.cmp(a_tile, b_tile, c_tile, cmp_mode=cmp_mode)
        pto.tile.store(c_tile, c_view)

    return _kernel


_kernels = {
    name: _make_kernel(name, pto_dtype, shape, valid_shape, cmp_mode)
    for name, _, pto_dtype, shape, valid_shape, cmp_mode in CASE_SPECS
}


def _make_inputs(name, np_dtype, shape):
    # Deterministic per-case seed, mirroring st_common.setup_case_rng which
    # uses crc32(case name).  Original value range was randint(1, 10) for both
    # inputs (see tcmp/gen_data.py).
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    input1 = np.random.randint(1, 10, size=shape).astype(np_dtype)
    input2 = np.random.randint(1, 10, size=shape).astype(np_dtype)
    return [input1, input2]


def _make_expected(input1, input2, shape, valid_shape, cmp_mode):
    # Replicates tcmp/gen_data.py: boolean comparison over the valid region,
    # then pack 1 bit per element into the first vc // 8 byte columns of each
    # valid row (dst tile is i8, same physical shape as the inputs).
    vr, vc = valid_shape
    mask_bits = np.zeros(shape, dtype=np.bool_)
    if cmp_mode == "eq":
        mask_bits[:vr, :vc] = input1[:vr, :vc] == input2[:vr, :vc]
    elif cmp_mode == "ne":
        mask_bits[:vr, :vc] = input1[:vr, :vc] != input2[:vr, :vc]
    elif cmp_mode == "lt":
        mask_bits[:vr, :vc] = input1[:vr, :vc] < input2[:vr, :vc]
    elif cmp_mode == "gt":
        mask_bits[:vr, :vc] = input1[:vr, :vc] > input2[:vr, :vc]
    elif cmp_mode == "ge":
        mask_bits[:vr, :vc] = input1[:vr, :vc] >= input2[:vr, :vc]
    elif cmp_mode == "le":
        mask_bits[:vr, :vc] = input1[:vr, :vc] <= input2[:vr, :vc]
    else:
        raise ValueError(f"unsupported cmp_mode {cmp_mode!r}")

    packed_cols = vc // 8
    golden = np.zeros(shape, dtype=np.uint8)
    for row in range(vr):
        for col_byte in range(packed_cols):
            byte_val = 0
            for bit in range(8):
                src_col = col_byte * 8 + bit
                if src_col < vc and mask_bits[row, src_col]:
                    byte_val |= (1 << bit)
            golden[row, col_byte] = byte_val
    return golden.astype(np.int8)


def _packed_mask_case(name, kernel, *, inputs, expected, out_shape,
                      check_region, rtol=0.0, atol=0.0):
    """golden_output_case variant for tcmp's packed-mask output.

    tcmp stores a packed 1-bit-per-element mask into an i8 tile of the same
    physical shape as the inputs; only the first (valid_cols // 8) bytes of
    each valid row carry meaningful data and the rest of the output tile is
    undefined padding.  The legacy compare.py compared only
    output[:vr, :vc // 8] against the matching golden slice, so this case
    builder performs that region comparison instead of the full-buffer compare
    done by golden_output_case.  The kernel itself is standard auto mode and
    uses only the existing pto.tile.cmp API.
    """
    def make_case():
        host_inputs = inputs() if callable(inputs) else inputs
        host_inputs = [np.array(v, copy=True) for v in host_inputs]
        golden = expected(*host_inputs) if callable(expected) else expected
        golden = np.array(golden, copy=True)
        out = np.zeros(out_shape, dtype=golden.dtype)
        return [*host_inputs, out], golden

    def check(device_inputs, golden):
        actual = np.asarray(device_inputs[-1].cpu().numpy())
        actual_region, golden_region = check_region(actual, golden)
        np.testing.assert_allclose(actual_region, golden_region, rtol=rtol, atol=atol)

    return {"name": name, "kernel": kernel, "make_case": make_case, "check": check}


CASES = []
for name, np_dtype, pto_dtype, shape, valid_shape, cmp_mode in CASE_SPECS:
    rows, cols = shape
    vr, vc = valid_shape
    packed_cols = vc // 8
    CASES.append(
        _packed_mask_case(
            "tcmp_" + name,
            _kernels[name],
            inputs=lambda name=name, np_dtype=np_dtype, shape=shape: _make_inputs(name, np_dtype, shape),
            expected=lambda i1, i2, shape=shape, valid_shape=valid_shape, cmp_mode=cmp_mode: _make_expected(
                i1, i2, shape, valid_shape, cmp_mode
            ),
            out_shape=shape,
            check_region=lambda actual, golden, vr=vr, packed_cols=packed_cols: (
                actual[:vr, :packed_cols],
                golden[:vr, :packed_cols],
            ),
            rtol=0.0,
            atol=0.0,
        )
    )


auto_main(globals())
