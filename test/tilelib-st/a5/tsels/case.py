#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tsels.
#
# pto.tile.sels selects per element between the source tile and a scalar
# driven by a packed bitmask: dst[y, x] = mask_bit(y, x) ? src[y, x] : scalar.
# The mask is a packed bitmask (each byte covers 8 consecutive columns) stored
# in a mask tile whose row holds ceil(cols / 8) valid bytes; the mask tile
# element width (i8/i16/i32) only changes the storage layout, not the bit
# semantics (see the pto.tsels TileOps template: mask byte offset for (row,
# col) is row * mask.shape[1] * bytewidth + col // 8).  This mirrors the
# legacy .pto kernels (tload(mask) + tload(src) + tsels(mask, src, tmp,
# scalar) -> dst + tstore(dst)) and the legacy gen_data.py packed-mask golden
# logic, including the per-case deterministic scalar draw and valid_shape
# handling.  The kernels are standard auto mode and use only the existing
# pto.tile.sels API (tmp is auto-synthesized on a5).

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto

# Legacy cases.py used unsigned numpy dtypes for the integer data and mask, but
# the legacy .pto kernels (and the pto.tsels TileOps template) declare the
# tiles with the signed i8/i16/i32 variants -- the values are pure bit patterns
# for this select op, so the signed/unsigned distinction is irrelevant.
NP_TO_PTO = {
    np.uint8: pto.i8,
    np.uint16: pto.i16,
    np.uint32: pto.i32,
    np.float16: pto.f16,
    np.float32: pto.f32,
}

# (legacy case name, numpy dtype, numpy mask dtype, src_shape, mask_shape,
#  dst_shape, valid_shape, dst_valid_shape, eps).
# Mirrors testcase/tsels/cases.py exactly.  The legacy .pto allocated tiles
# with valid= only where the effective computation region differs from the
# tile shape (the "2x31" and "32x666" cases); every other tile is fully valid.
CASE_SPECS = [
    ("uint8_uint8_2x32_2x32_2x32_2x32", np.uint8, np.uint8, (2, 32), (2, 32), (2, 32), (2, 32), (2, 32), 0),
    ("uint8_uint16_2x32_2x16_2x32_2x32", np.uint8, np.uint16, (2, 32), (2, 16), (2, 32), (2, 32), (2, 32), 0),
    ("uint8_uint32_2x32_2x8_2x32_2x32", np.uint8, np.uint32, (2, 32), (2, 8), (2, 32), (2, 32), (2, 32), 0),
    ("uint16_uint8_2x16_2x32_2x16_2x16", np.uint16, np.uint8, (2, 16), (2, 32), (2, 16), (2, 16), (2, 16), 0),
    ("uint16_uint16_2x16_2x16_2x16_2x16", np.uint16, np.uint16, (2, 16), (2, 16), (2, 16), (2, 16), (2, 16), 0),
    ("uint16_uint32_2x16_2x8_2x16_2x16", np.uint16, np.uint32, (2, 16), (2, 8), (2, 16), (2, 16), (2, 16), 0),
    ("uint32_uint8_2x8_2x32_2x8_2x8", np.uint32, np.uint8, (2, 8), (2, 32), (2, 8), (2, 8), (2, 8), 0),
    ("uint32_uint16_2x8_2x16_2x8_2x8", np.uint32, np.uint16, (2, 8), (2, 16), (2, 8), (2, 8), (2, 8), 0),
    ("uint32_uint32_2x8_2x8_2x8_2x8", np.uint32, np.uint32, (2, 8), (2, 8), (2, 8), (2, 8), (2, 8), 0),
    ("f16_uint8_2x16_2x32_2x16_2x16", np.float16, np.uint8, (2, 16), (2, 32), (2, 16), (2, 16), (2, 16), 1e-3),
    ("f16_uint16_2x16_2x16_2x16_2x16", np.float16, np.uint16, (2, 16), (2, 16), (2, 16), (2, 16), (2, 16), 1e-3),
    ("f16_uint32_2x16_2x8_2x16_2x16", np.float16, np.uint32, (2, 16), (2, 8), (2, 16), (2, 16), (2, 16), 1e-3),
    ("f32_uint8_2x8_2x32_2x8_2x8", np.float32, np.uint8, (2, 8), (2, 32), (2, 8), (2, 8), (2, 8), 1e-6),
    ("f32_uint16_2x8_2x16_2x8_2x8", np.float32, np.uint16, (2, 8), (2, 16), (2, 8), (2, 8), (2, 8), 1e-6),
    ("f32_uint32_2x8_2x8_2x8_2x8", np.float32, np.uint32, (2, 8), (2, 8), (2, 8), (2, 8), (2, 8), 1e-6),
    ("uint8_uint8_2x32_2x64_2x128_2x31", np.uint8, np.uint8, (2, 128), (2, 64), (2, 32), (2, 31), (2, 31), 0),
    ("uint16_uint8_2x32_2x64_2x128_2x31", np.uint16, np.uint8, (2, 128), (2, 64), (2, 32), (2, 31), (2, 31), 0),
    ("f32_uint8_2x32_2x64_2x128_2x31", np.float32, np.uint8, (2, 128), (2, 64), (2, 32), (2, 31), (2, 31), 1e-6),
    ("uint8_uint8_32x672_32x96_32x672_32x666", np.uint8, np.uint8, (32, 672), (32, 96), (32, 672), (32, 666), (32, 666), 0),
    ("f16_uint8_32x672_32x96_32x672_32x666", np.float16, np.uint8, (32, 672), (32, 96), (32, 672), (32, 666), (32, 666), 1e-3),
    ("f32_uint8_32x672_32x96_32x672_32x666", np.float32, np.uint8, (32, 672), (32, 96), (32, 672), (32, 666), (32, 666), 1e-6),
    ("f32_uint8_1x8192_1x4096_1x8192_1x8192", np.float32, np.uint8, (1, 8192), (1, 4096), (1, 8192), (1, 8192), (1, 8192), 1e-6),
]

_SPEC_BY_NAME = {spec[0]: spec for spec in CASE_SPECS}


def _gen_case_data(name, np_dtype, np_mask_dtype, src_shape, mask_shape):
    """Reproduce the legacy gen_data.py draws for one case, deterministically.

    Mirrors st_common.setup_case_rng (crc32 seed) and the per-case draw order:
    input1 (src), input2 (scalar), then the packed mask.  Integer data uses
    randint over the full iinfo range; float data uses uniform over finfo.
    """
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    if np.issubdtype(np_dtype, np.integer):
        info = np.iinfo(np_dtype)
        src = np.random.randint(info.min, info.max, size=src_shape).astype(np_dtype)
        scalar = np.random.randint(info.min, info.max, size=[1]).astype(np_dtype)[0]
    else:
        info = np.finfo(np_dtype)
        src = np.random.uniform(info.min, info.max, size=src_shape).astype(np_dtype)
        scalar = np.random.uniform(info.min, info.max, size=[1]).astype(np_dtype)[0]
    mask_info = np.iinfo(np_mask_dtype)
    mask = np.random.randint(mask_info.min, mask_info.max, size=mask_shape).astype(np_mask_dtype)
    return src, scalar, mask


# Per-case data computed once so the kernel closure scalar, the host inputs and
# the golden all see the exact same values.
_CASE_DATA = {
    spec[0]: _gen_case_data(spec[0], spec[1], spec[2], spec[3], spec[4])
    for spec in CASE_SPECS
}


def _make_kernel(name, pto_dtype, pto_mask_dtype, scalar, src_shape, mask_shape,
                 dst_shape, valid_shape, dst_valid_shape):
    src_rows, src_cols = src_shape
    mask_rows, mask_cols = mask_shape
    dst_rows, dst_cols = dst_shape
    # The legacy .pto kept valid= only where the effective region differs from
    # the tile shape; otherwise the tile is fully valid.
    src_valid = list(valid_shape) if valid_shape != src_shape else None
    dst_valid = list(dst_valid_shape) if dst_valid_shape != dst_shape else None
    # pto.tile.sels materializes the scalar as a constant of the src tile
    # element type: an integer literal for integer tiles, a float for f16/f32.
    scalar_const = int(scalar) if pto_dtype in (pto.i8, pto.i16, pto.i32) else float(scalar)

    @pto.jit(name="tsels_" + name, target="a5")
    def _kernel(
        mask_ptr: pto.ptr(pto_mask_dtype, "gm"),
        src_ptr: pto.ptr(pto_dtype, "gm"),
        dst_ptr: pto.ptr(pto_dtype, "gm"),
    ):
        mask_view = pto.make_tensor_view(
            mask_ptr, shape=[mask_rows, mask_cols], strides=[mask_cols, 1]
        )
        src_view = pto.make_tensor_view(
            src_ptr, shape=[src_rows, src_cols], strides=[src_cols, 1]
        )
        dst_view = pto.make_tensor_view(
            dst_ptr, shape=[dst_rows, dst_cols], strides=[dst_cols, 1]
        )

        mask_tile = pto.alloc_tile(shape=[mask_rows, mask_cols], dtype=pto_mask_dtype)
        src_kwargs = {"shape": [src_rows, src_cols], "dtype": pto_dtype}
        if src_valid is not None:
            src_kwargs["valid_shape"] = src_valid
        src_tile = pto.alloc_tile(**src_kwargs)
        dst_kwargs = {"shape": [dst_rows, dst_cols], "dtype": pto_dtype}
        if dst_valid is not None:
            dst_kwargs["valid_shape"] = dst_valid
        dst_tile = pto.alloc_tile(**dst_kwargs)

        pto.tile.load(mask_view, mask_tile)
        pto.tile.load(src_view, src_tile)
        pto.tile.sels(mask_tile, src_tile, scalar_const, dst_tile)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


_kernels = {}
for _spec in CASE_SPECS:
    _name, _np_dtype, _np_mask_dtype, _src_shape, _mask_shape, _dst_shape, \
        _valid_shape, _dst_valid_shape, _eps = _spec
    _src, _scalar, _mask = _CASE_DATA[_name]
    _kernels[_name] = _make_kernel(
        _name,
        NP_TO_PTO[_np_dtype],
        NP_TO_PTO[_np_mask_dtype],
        _scalar,
        _src_shape,
        _mask_shape,
        _dst_shape,
        _valid_shape,
        _dst_valid_shape,
    )


def _make_inputs(name):
    src, _scalar, mask = _CASE_DATA[name]
    return [mask, src]


def _make_expected(mask, src, name):
    # Replicates tsels/gen_data.py bit-for-bit: for every element of the valid
    # region, take src when the corresponding packed-mask bit is set, else the
    # scalar; the rest of the dst buffer stays zero (legacy compare.py only
    # compared the valid region and main.cpp zero-initialized the output).
    _spec = _SPEC_BY_NAME[name]
    dst_shape = _spec[5]
    valid_shape = _spec[6]
    scalar = _CASE_DATA[name][1]
    vr, vc = valid_shape
    mask_u8 = mask.view(np.uint8).reshape(mask.shape[0], -1)
    golden = np.zeros(dst_shape, dtype=src.dtype)
    for y in range(vr):
        row = mask_u8[y]
        for x in range(vc):
            do_select = (1 << (x & 7)) & row[x >> 3]
            golden[y, x] = src[y, x] if do_select != 0 else scalar
    return golden


CASES = [
    golden_output_case(
        "tsels_" + name,
        _kernels[name],
        inputs=lambda name=name: _make_inputs(name),
        expected=lambda mask, src, name=name: _make_expected(mask, src, name),
        rtol=eps,
        atol=eps,
    )
    for name, _np_dtype, _np_mask_dtype, _src_shape, _mask_shape, _dst_shape, \
        _valid_shape, _dst_valid_shape, eps in CASE_SPECS
]


auto_main(globals())
