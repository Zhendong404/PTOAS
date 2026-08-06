#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/textract_v2v.
# The legacy kernel loads a rank-5 ND view into a Vec tile, extracts the
# (0, 0) window into another Vec tile, and stores it back to a rank-5 view.
# This is the public A5 ``pto.tile.extract`` Vec->Vec ND form.

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


@pto.jit(name="textract_v2v_f32_16x16", target="a5")
def _kernel(
    src_ptr: pto.ptr(pto.f32, "gm"),
    dst_ptr: pto.ptr(pto.f32, "gm"),
):
    src_view = pto.make_tensor_view(
        src_ptr,
        shape=[1, 1, 1, ROWS, COLS],
        strides=[ROWS * COLS, ROWS * COLS, ROWS * COLS, COLS, 1],
    )
    dst_view = pto.make_tensor_view(
        dst_ptr,
        shape=[1, 1, 1, ROWS, COLS],
        strides=[ROWS * COLS, ROWS * COLS, ROWS * COLS, COLS, 1],
    )

    src_tile = pto.alloc_tile(
        shape=[ROWS, COLS],
        dtype=pto.f32,
        valid_shape=[ROWS, COLS],
    )
    dst_tile = pto.alloc_tile(
        shape=[ROWS, COLS],
        dtype=pto.f32,
        valid_shape=[ROWS, COLS],
    )

    pto.tile.load(src_view, src_tile, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, ROWS, COLS])
    pto.tile.extract(src_tile, dst_tile, 0, 0)
    pto.tile.store(dst_tile, dst_view, offsets=[0, 0, 0, 0, 0], sizes=[1, 1, 1, ROWS, COLS])


def _make_inputs():
    # Match legacy setup_case_rng(case) / gen_data.py.
    name = "v2v_f32_16x16"
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    src = np.random.uniform(-1.0, 1.0, size=(ROWS, COLS)).astype(np.float32)
    return [src]


def _make_expected(src):
    return src.copy()


CASES = [
    golden_output_case(
        "textract_v2v_f32_16x16",
        _kernel,
        inputs=_make_inputs,
        expected=_make_expected,
        rtol=1e-6,
        atol=1e-6,
    ),
]


auto_main(globals())
