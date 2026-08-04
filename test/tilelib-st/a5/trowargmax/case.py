#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/trowargmax.
#   tload(src) + trowargmax(src, tmp) -> dst(R, 1) + tstore(dst)
#
# Row argmax: dst[i, 0] = index of max over src[i, valid_cols].  The source
# tiles may be partially valid (valid_shape may be smaller than shape; the
# legacy cases keep valid_rows == rows).  The dst tile keeps the legacy
# physical layout (R x 8, valid R x 1) while the dst GM view is (R, dst_cols)
# with row stride dst_cols; the kernel writes only column 0 (the store
# partition is inferred from the dst tile valid_shape) and the host golden
# zero-pads the remaining columns, so the full-array comparison stays valid.
# dst dtype follows the legacy cases: ui32 (uint32) or i32 (int32) indices.

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto

NP_SRC_TO_PTO = {
    np.float32: pto.f32,
    np.float16: pto.f16,
}

NP_DST_TO_PTO = {
    np.uint32: pto.ui32,
    np.int32: pto.i32,
}

# (legacy case name, src numpy dtype, dst numpy dtype, src shape,
#  src valid_shape, dst_cols, eps)
# dst_cols is the legacy dst GM buffer column count (1 / 8 / 16, see main.cpp);
# the rowargmax dst tile itself is always (R, 8) valid (R, 1) in the legacy .pto.
CASE_SPECS = [
    ("uint32_float_8x1_8x8_8x8",              np.float32, np.uint32, (8, 8),      (8, 8),       1, 0),
    ("uint32_float_1024x1_1024x8_1024x8",     np.float32, np.uint32, (1024, 8),   (1024, 8),    1, 0),
    ("uint32_float_16x1_13x16_13x13",         np.float32, np.uint32, (13, 16),    (13, 13),     1, 0),
    ("uint32_float_1024x1_1023x24_1023x17",   np.float32, np.uint32, (1023, 24),  (1023, 17),   1, 0),
    ("uint32_float_8x1_8x64_8x64",            np.float32, np.uint32, (8, 64),     (8, 64),      1, 0),
    ("uint32_float_264x1_260x64_260x64",      np.float32, np.uint32, (260, 64),   (260, 64),    1, 0),
    ("uint32_float_8x1_1x128_1x128",          np.float32, np.uint32, (1, 128),    (1, 128),     1, 0),
    ("uint32_float_64x1_32x128_32x128",       np.float32, np.uint32, (32, 128),   (32, 128),    1, 0),
    ("uint32_float_8x1_3x4096_3x4095",        np.float32, np.uint32, (3, 4096),   (3, 4095),    1, 0),
    ("uint32_float_8x1_2x16384_2x16381",      np.float32, np.uint32, (2, 16384),  (2, 16381),   1, 0),
    ("uint32_half_16x1_2x16_2x16",            np.float16, np.uint32, (2, 16),     (2, 16),      1, 0),
    ("uint32_half_16x1_13x16_13x13",          np.float16, np.uint32, (13, 16),    (13, 13),     1, 0),
    ("uint32_half_272x1_260x64_260x64",       np.float16, np.uint32, (260, 64),   (260, 64),    1, 0),
    ("uint32_half_16x1_3x8192_3x8191",        np.float16, np.uint32, (3, 8192),   (3, 8191),    1, 0),
    ("uint32_half_16x1_1x16384_1x16381",      np.float16, np.uint32, (1, 16384),  (1, 16381),   1, 0),
    ("uint32_half_16x1_1x32768_1x32761",      np.float16, np.uint32, (1, 32768),  (1, 32761),   1, 0),
    ("int32_float_16x1_13x16_13x13",          np.float32, np.int32,  (13, 16),    (13, 13),     1, 0),
    ("int32_half_16x1_13x16_13x13",           np.float16, np.int32,  (13, 16),    (13, 13),     1, 0),
    ("uint32_float_3x8_3x3480_3x3473",        np.float32, np.uint32, (3, 3480),   (3, 3473),    8, 0),
    ("uint32_float_260x8_260x64_260x64",      np.float32, np.uint32, (260, 64),   (260, 64),    8, 0),
    ("uint32_float_1023x8_1023x24_1023x17",   np.float32, np.uint32, (1023, 24),  (1023, 17),   8, 0),
    ("uint32_half_3x16_3x3488_3x3473",        np.float16, np.uint32, (3, 3488),   (3, 3473),   16, 0),
    ("uint32_half_260x16_260x64_260x64",      np.float16, np.uint32, (260, 64),   (260, 64),   16, 0),
    ("uint32_half_1023x16_1023x32_1023x17",   np.float16, np.uint32, (1023, 32),  (1023, 17),  16, 0),
]


def _make_kernel(name: str, np_src_dtype, np_dst_dtype, shape, valid_shape, dst_cols):
    rows, cols = shape
    valid_rows, valid_cols = valid_shape
    pto_src_dtype = NP_SRC_TO_PTO[np_src_dtype]
    pto_dst_dtype = NP_DST_TO_PTO[np_dst_dtype]

    @pto.jit(name="trowargmax_" + name, target="a5")
    def _kernel(
        src_ptr: pto.ptr(pto_src_dtype, "gm"),
        dst_ptr: pto.ptr(pto_dst_dtype, "gm"),
    ):
        src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
        dst_view = pto.make_tensor_view(
            dst_ptr, shape=[valid_rows, dst_cols], strides=[dst_cols, 1]
        )

        src_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=pto_src_dtype, valid_shape=[valid_rows, valid_cols]
        )
        # Legacy dst tile: physical shape (R, 8), valid (R, 1); tmp is
        # auto-synthesized by pto.tile.rowargmax from src metadata.
        dst_tile = pto.alloc_tile(
            shape=[valid_rows, 8], dtype=pto_dst_dtype, valid_shape=[valid_rows, 1]
        )

        pto.tile.load(src_view, src_tile)
        pto.tile.rowargmax(src_tile, dst_tile)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


_kernels = {
    name: _make_kernel(name, np_src_dtype, np_dst_dtype, shape, valid_shape, dst_cols)
    for name, np_src_dtype, np_dst_dtype, shape, valid_shape, dst_cols, _ in CASE_SPECS
}


def _make_inputs(name: str, shape, np_dtype):
    # Legacy st_common.setup_case_rng + gen_data.py semantics.
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    info = np.finfo(np_dtype)
    src = np.random.uniform(low=info.min, high=info.max, size=shape).astype(np_dtype)
    return [src]


def _make_expected(src, valid_shape, dst_cols, np_dst_dtype):
    valid_rows, valid_cols = valid_shape
    golden = np.zeros((valid_rows, dst_cols), dtype=np_dst_dtype)
    golden[:, 0] = np.argmax(src[:valid_rows, :valid_cols], axis=1).astype(np_dst_dtype)
    return golden


CASES = [
    golden_output_case(
        "trowargmax_" + name,
        _kernels[name],
        inputs=lambda name=name, shape=shape, np_dtype=np_src_dtype: _make_inputs(
            name, shape, np_dtype
        ),
        expected=lambda src, valid_shape=valid_shape, dst_cols=dst_cols, np_dst_dtype=np_dst_dtype: _make_expected(
            src, valid_shape, dst_cols, np_dst_dtype
        ),
        rtol=eps,
        atol=eps,
    )
    for name, np_src_dtype, np_dst_dtype, shape, valid_shape, dst_cols, eps in CASE_SPECS
]


auto_main(globals())
