#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tcmps.
#
# tcmps(src, scalar) -> packed predicate mask (src < scalar), stored as a ui8
# tile.  The comparison result is packed 1 bit per element into a hardware-
# dependent linear bitstream (see gen_data.py); only the first
# total_output_bytes bytes of the output tile carry meaningful data and the
# remaining tile bytes are undefined padding, so the legacy compare.py
# truncated the output to that size before comparing.  _packed_mask_case below
# reproduces that truncation (golden_output_case compares the full buffer and
# is therefore not usable for this op).  The kernels themselves are standard
# auto mode and use only the existing pto.tile.cmps API.

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main
from ptodsl import pto

# Scalar value for the "src < scalar" comparison; matches tcmps/launch.cpp
# (TCMP_SCALAR_F32/F16 = 5.0, TCMP_SCALAR_I32/I16 = 5) and gen_data.py SCALAR.
SCALAR = 5.0

# (case suffix, numpy dtype, pto dtype, shape, valid_shape, dst_shape)
# Mirrors testcase/tcmps/cases.py exactly.  dst_shape matches the legacy .pto
# output tile shape: identical to shape except for f32_256x16 whose legacy dst
# tile was 256x32 (see tcmps.pto, case TCMP_f32_256x16).
CASE_SPECS = [
    ("f32_1x64",             np.float32, pto.f32, (1, 64),     (1, 64),     (1, 64)),
    ("f32_4x64",             np.float32, pto.f32, (4, 64),     (4, 64),     (4, 64)),
    ("f32_8x64",             np.float32, pto.f32, (8, 64),     (8, 64),     (8, 64)),
    ("f32_32x64",            np.float32, pto.f32, (32, 64),    (32, 64),    (32, 64)),
    ("f32_128x128",          np.float32, pto.f32, (128, 128),  (128, 128),  (128, 128)),
    ("i32_16x32",            np.int32,   pto.i32, (16, 32),    (16, 32),    (16, 32)),
    ("i32_32x32",            np.int32,   pto.i32, (32, 32),    (32, 32),    (32, 32)),
    ("i32_32x64_valid32x64", np.int32,   pto.i32, (64, 64),    (32, 64),    (64, 64)),
    ("f32_7x448",            np.float32, pto.f32, (7, 448),    (7, 448),    (7, 448)),
    ("f32_256x16",           np.float32, pto.f32, (256, 16),   (256, 16),   (256, 32)),
    ("i32_31x128",           np.int32,   pto.i32, (31, 128),   (31, 128),   (31, 128)),
    ("f16_32x128",           np.float16, pto.f16, (32, 128),   (32, 128),   (32, 128)),
    ("i16_32x128",           np.int16,   pto.i16, (32, 128),   (32, 128),   (32, 128)),
]


def _make_kernel(name, pto_dtype, scalar, shape, valid_shape, dst_shape):
    rows, cols = shape
    vr, vc = valid_shape
    drows, dcols = dst_shape
    # The legacy .pto dst tiles are fully valid (no valid= attribute), except
    # the partially-valid i32_32x64_valid32x64 case whose dst tile was
    # 64x64 with valid 32x64.
    dst_valid = dst_shape if dst_shape != shape else valid_shape
    dvr, dvc = dst_valid

    @pto.jit(name="tcmps_" + name, target="a5")
    def _kernel(
        src_ptr: pto.ptr(pto_dtype, "gm"),
        dst_ptr: pto.ptr(pto.ui8, "gm"),
    ):
        src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[drows, dcols], strides=[dcols, 1])

        src_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype, valid_shape=[vr, vc])
        dst_tile = pto.alloc_tile(shape=[drows, dcols], dtype=pto.ui8, valid_shape=[dvr, dvc])

        pto.tile.load(src_view, src_tile)
        pto.tile.cmps(src_tile, scalar, dst_tile, cmp_mode="lt")
        pto.tile.store(dst_tile, dst_view)

    return _kernel


_kernels = {
    name: _make_kernel(
        name,
        pto_dtype,
        SCALAR if np.issubdtype(np_dtype, np.floating) else int(SCALAR),
        shape,
        valid_shape,
        dst_shape,
    )
    for name, np_dtype, pto_dtype, shape, valid_shape, dst_shape in CASE_SPECS
}


def _packed_output_bytes(np_dtype, valid_shape):
    """Number of meaningful bytes tcmps writes for this case (gen_data.py /
    main.cpp GetDstElemCount model)."""
    vr, vc = valid_shape
    elem_size = np.dtype(np_dtype).itemsize
    lanes = 256 // elem_size
    if elem_size == 4:  # 32B: 2 vcmps + dintlv_b8, 16 bytes per iteration
        total_elm = vr * vc
        repeat_times = (total_elm + lanes - 1) // lanes + 1
        total_iters = repeat_times // 2
        bytes_per_iter = 16
    else:  # 16B: PK mode, 16 bytes per iteration
        bytes_per_iter = 16
        iters_per_row = (vc + lanes - 1) // lanes
        total_iters = vr * iters_per_row
    return total_iters * bytes_per_iter


def _make_inputs(name, np_dtype, shape):
    # Deterministic per-case seed, mirroring st_common.setup_case_rng which
    # uses crc32(case name).  Value ranges match tcmps/gen_data.py:
    # randint(-5, 5) for floats, randint(1, 10) for ints.
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    if np.issubdtype(np_dtype, np.floating):
        src = np.random.randint(-5, 5, size=shape).astype(np_dtype)
    else:
        src = np.random.randint(1, 10, size=shape).astype(np_dtype)
    return [src]


def _make_expected(src, np_dtype, valid_shape):
    # Replicates tcmps/gen_data.py bit-for-bit: (src < scalar) over the valid
    # region, packed LSB-first into a linear byte stream whose length follows
    # the hardware iteration model (see _packed_output_bytes).
    vr, vc = valid_shape
    elem_size = np.dtype(np_dtype).itemsize
    scalar_val = np_dtype(SCALAR) if np.issubdtype(np_dtype, np.floating) else np_dtype(int(SCALAR))
    cmp_result = (src[:vr, :vc] < scalar_val).astype(np.uint8, copy=False)

    lanes = 256 // elem_size
    if elem_size == 4:  # 32B: linear offset after dintlv_b8 (PK mode)
        bytes_per_iter = 16
        total_elm = vr * vc
        repeat_times = (total_elm + lanes - 1) // lanes + 1
        total_iters = repeat_times // 2
    else:  # 16B: PK mode, sequential per row
        bytes_per_iter = 16
        iters_per_row = (vc + lanes - 1) // lanes
        total_iters = vr * iters_per_row

    total_output_bytes = total_iters * bytes_per_iter
    golden = np.zeros(total_output_bytes, dtype=np.uint8)

    for row in range(vr):
        for col in range(vc):
            if not cmp_result[row, col]:
                continue
            if elem_size == 4:
                # Linear element index maps to a bit position after dintlv_b8.
                linear_idx = row * vc + col
                iter_idx = linear_idx // (2 * lanes)
                pos_in_block = linear_idx % (2 * lanes)
                bit_pos = pos_in_block
                byte_idx = iter_idx * bytes_per_iter + (bit_pos // 8)
                bit_idx = bit_pos % 8
            else:
                col_in_iter = col % lanes
                bit_pos = col_in_iter
                byte_idx = (row * iters_per_row + col // lanes) * bytes_per_iter + (bit_pos // 8)
                bit_idx = bit_pos % 8
            if byte_idx < total_output_bytes:
                golden[byte_idx] |= (1 << bit_idx)

    return golden


def _make_golden_buffer(src, np_dtype, valid_shape, dst_shape):
    """Full dst-shaped golden: packed bytes in the first meaningful positions,
    zero padding afterwards (matching the np.zeros-initialized output buffer)."""
    packed = _make_expected(src, np_dtype, valid_shape)
    buf = np.zeros(dst_shape, dtype=np.uint8)
    buf.ravel()[:packed.size] = packed
    return buf


def _packed_mask_case(name, kernel, *, inputs, expected, out_shape,
                      check_region, rtol=0.0, atol=0.0):
    """golden_output_case variant for tcmps's packed-mask output.

    tcmps writes a hardware-packed linear bitstream into a ui8 tile whose
    physical size is larger than the meaningful output; only the first
    total_output_bytes bytes are compared (the legacy compare.py truncated the
    output to that size and ignored the padding).  golden_output_case compares
    the full buffer, so this case builder performs the truncated comparison
    instead.  The kernel itself is standard auto mode and uses only the
    existing pto.tile.cmps API.
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
for name, np_dtype, pto_dtype, shape, valid_shape, dst_shape in CASE_SPECS:
    total_output_bytes = _packed_output_bytes(np_dtype, valid_shape)
    CASES.append(
        _packed_mask_case(
            "tcmps_" + name,
            _kernels[name],
            inputs=lambda name=name, np_dtype=np_dtype, shape=shape: _make_inputs(name, np_dtype, shape),
            expected=lambda src, np_dtype=np_dtype, valid_shape=valid_shape, dst_shape=dst_shape: _make_golden_buffer(
                src, np_dtype, valid_shape, dst_shape
            ),
            out_shape=dst_shape,
            check_region=lambda actual, golden, n=total_output_bytes: (
                actual.ravel()[:n],
                golden.ravel()[:n],
            ),
            rtol=0.0,
            atol=0.0,
        )
    )


auto_main(globals())
