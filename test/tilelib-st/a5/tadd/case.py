#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tadd.
#
# This case intentionally uses PTODSL auto mode as the vector TileLib pilot:
# tile addresses, load/store partitions, and sync insertion are left to PTOAS.

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


# Each case is (name, dtype, dst_shape, src0_shape, src1_shape, valid_shape).
# The first two entries retain the original TileLib pilot cases.  The
# remaining entries mirror the PTO-ISA A5 tadd variants whose dtypes are
# supported by the public A5 template.  The i64/u64 variants are intentionally
# not represented because template_tadd does not support those dtypes.
CASE_SHAPES = [
    ("f32_16x64", pto.f32, (16, 64), (16, 64), (16, 64), (16, 64)),
    ("f32_32x32", pto.f32, (32, 32), (32, 32), (32, 32), (32, 32)),
    ("f32_64x128", pto.f32, (64, 128), (64, 128), (64, 128), (64, 128)),
    ("f32_64x64", pto.f32, (64, 64), (64, 64), (64, 64), (64, 64)),
    ("i32_64x64", pto.i32, (64, 64), (64, 64), (64, 64), (64, 64)),
    ("i16_64x64", pto.i16, (64, 64), (64, 64), (64, 64), (64, 64)),
    ("f16_16x256", pto.f16, (16, 256), (16, 256), (16, 256), (16, 256)),
    ("f16_16x64_src128", pto.f16, (16, 64), (16, 128), (16, 128), (16, 64)),
    ("f32_16x32_src64", pto.f32, (16, 32), (16, 64), (16, 32), (16, 32)),
    ("i16_32x128_src256", pto.i16, (32, 128), (32, 128), (32, 256), (32, 128)),
    ("i32_16x32_src64", pto.i32, (16, 32), (16, 64), (16, 32), (16, 32)),
    ("f16_16x64_valid63", pto.f16, (16, 64), (16, 128), (16, 128), (16, 63)),
    ("f32_16x32_valid31", pto.f32, (16, 32), (16, 64), (16, 32), (16, 31)),
    ("i16_32x128_valid127", pto.i16, (32, 128), (32, 128), (32, 256), (32, 127)),
    ("i32_16x32_valid31", pto.i32, (16, 32), (16, 64), (16, 32), (16, 31)),
    ("f16_2x128_valid1x106", pto.f16, (2, 128), (2, 128), (2, 128), (1, 106)),
]

PTO_TO_NP_DTYPE = {
    pto.f32: np.float32,
    pto.f16: np.float16,
    pto.i32: np.int32,
    pto.i16: np.int16,
}


def _tadd_body(
    a_ptr,
    b_ptr,
    c_ptr,
    *,
    dtype,
    dst_shape,
    src0_shape,
    src1_shape,
    valid_shape,
):
    """Shared kernel body for the tadd shape/layout variants."""

    dst_rows, dst_cols = dst_shape
    src0_rows, src0_cols = src0_shape
    src1_rows, src1_cols = src1_shape
    valid_rows, valid_cols = valid_shape

    a_view = pto.make_tensor_view(a_ptr, shape=[src0_rows, src0_cols], strides=[src0_cols, 1])
    b_view = pto.make_tensor_view(b_ptr, shape=[src1_rows, src1_cols], strides=[src1_cols, 1])
    c_view = pto.make_tensor_view(c_ptr, shape=[dst_rows, dst_cols], strides=[dst_cols, 1])

    a_tile = pto.alloc_tile(shape=[src0_rows, src0_cols], dtype=dtype, valid_shape=valid_shape)
    b_tile = pto.alloc_tile(shape=[src1_rows, src1_cols], dtype=dtype, valid_shape=valid_shape)
    c_tile = pto.alloc_tile(shape=[dst_rows, dst_cols], dtype=dtype, valid_shape=valid_shape)

    pto.tile.load(a_view, a_tile)
    pto.tile.load(b_view, b_tile)
    pto.tile.add(a_tile, b_tile, c_tile)
    pto.tile.store(c_tile, c_view)


# One decorated kernel per case, each binding a static shape at definition time
# (mirroring the per-case funcs in tadd.pto).
_tadd_kernels = {}
for _name, _dtype, _dst_shape, _src0_shape, _src1_shape, _valid_shape in CASE_SHAPES:

    def _make(
        dtype=_dtype,
        dst_shape=_dst_shape,
        src0_shape=_src0_shape,
        src1_shape=_src1_shape,
        valid_shape=_valid_shape,
        kernel_name=f"tadd_{_name}",
    ):
        @pto.jit(
            name=kernel_name,
            target="a5",
        )
        def _kernel(
            a_ptr: pto.ptr(dtype, "gm"),
            b_ptr: pto.ptr(dtype, "gm"),
            c_ptr: pto.ptr(dtype, "gm"),
        ):
            _tadd_body(
                a_ptr,
                b_ptr,
                c_ptr,
                dtype=dtype,
                dst_shape=dst_shape,
                src0_shape=src0_shape,
                src1_shape=src1_shape,
                valid_shape=valid_shape,
            )

        return _kernel

    _tadd_kernels[_name] = _make()


def _make_inputs(name, dtype, dst_shape, src0_shape, src1_shape, valid_shape):
    # Deterministic per-case seed, mirroring st_common.setup_case_rng which uses
    # crc32(name).  Original value range was randint(1, 10).
    import zlib
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    np_dtype = PTO_TO_NP_DTYPE[dtype]
    a = np.random.randint(1, 10, size=src0_shape).astype(np_dtype)
    b = np.random.randint(1, 10, size=src1_shape).astype(np_dtype)
    return [a, b]


def _make_expected(a, b, dtype, dst_shape, valid_shape):
    valid_rows, valid_cols = valid_shape
    result = np.zeros(dst_shape, dtype=PTO_TO_NP_DTYPE[dtype])
    result[:valid_rows, :valid_cols] = a[:valid_rows, :valid_cols] + b[:valid_rows, :valid_cols]
    return result


CASES = []
for _name, _dtype, _dst_shape, _src0_shape, _src1_shape, _valid_shape in CASE_SHAPES:
    CASES.append(
        golden_output_case(
            "tadd_" + _name,
            _tadd_kernels[_name],
            inputs=lambda _name=_name, _dtype=_dtype, _dst_shape=_dst_shape, _src0_shape=_src0_shape,
                           _src1_shape=_src1_shape, _valid_shape=_valid_shape: _make_inputs(
                _name, _dtype, _dst_shape, _src0_shape, _src1_shape, _valid_shape
            ),
            expected=lambda a, b, _dtype=_dtype, _dst_shape=_dst_shape, _valid_shape=_valid_shape: _make_expected(
                a, b, _dtype, _dst_shape, _valid_shape
            ),
            rtol=1e-6,
            atol=1e-6,
            output_shape=_dst_shape,
            output_dtype=PTO_TO_NP_DTYPE[_dtype],
        )
    )


auto_main(globals())
