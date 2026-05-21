# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
TADD kernel – DSL-style builder for the TileLang ST testcase.

Generates the same IR as
  test/tilelang_st/npu/a5/src/st/testcase/tadd/tadd.pto
using the ``@pto.jit`` decorator and the ``pto.*`` namespace.

The Python maps line-for-line to the target MLIR:

  func.func @TADD_f32_16x64(                           # @pto.jit(..., func_attr="pto.aicore")
      %a_ptr: !pto.ptr<f32, gm>, …) attributes {pto.aicore} {
    %c0 = arith.constant 0 : index                      # c0 = pto.const(0)
    …
    %a_view = pto.make_tensor_view %a_ptr, …           # pto.make_tensor_view(a_ptr, shape=…, strides=…)
    %a_part = pto.partition_view %a_view, …            # pto.partition_view(a_view, offsets=…, sizes=…)
    %a = pto.alloc_tile …                               # pto.alloc_tile(shape=[…], dtype=pto.float32)
    pto.tload ins(%a_part) outs(%a)                     # pto.tile.load(a_part, a)
    pto.tadd ins(%a, %b) outs(%c)                       # pto.tile.add(a, b, c)
    pto.tstore ins(%c) outs(%c_part)                    # pto.tile.store(c, c_part)
  }
"""

from ptodsl import pto


def _tadd_tile(a_ptr, b_ptr, c_ptr, rows: int, cols: int) -> None:
    """Shared tile-add body for one static ``rows x cols`` case."""
    c0 = pto.const(0)
    c1 = pto.const(1)
    c_rows = pto.const(rows)
    c_cols = c_rows if rows == cols else pto.const(cols)
    c_elems = pto.const(rows * cols)

    shape = [c1, c1, c1, c_rows, c_cols]
    strides = [c_elems, c_elems, c_elems, c_cols, c1]
    off = [c0, c0, c0, c0, c0]

    a_view = pto.make_tensor_view(a_ptr, shape=shape, strides=strides)
    b_view = pto.make_tensor_view(b_ptr, shape=shape, strides=strides)
    c_view = pto.make_tensor_view(c_ptr, shape=shape, strides=strides)

    a_part = pto.partition_view(a_view, offsets=off, sizes=shape)
    b_part = pto.partition_view(b_view, offsets=off, sizes=shape)
    c_part = pto.partition_view(c_view, offsets=off, sizes=shape)

    a_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto.float32)
    b_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto.float32)
    c_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto.float32)

    pto.tile.load(a_part, a_tile)
    pto.tile.load(b_part, b_tile)
    pto.tile.add(a_tile, b_tile, c_tile)
    pto.tile.store(c_tile, c_part)


@pto.jit(
    name="TADD_f32_16x64",
    kernel_kind="vector",
    target="a5",
    func_attr="pto.aicore",
)
def TADD_f32_16x64(
    a_ptr: pto.ptr(pto.float32, "gm"),
    b_ptr: pto.ptr(pto.float32, "gm"),
    c_ptr: pto.ptr(pto.float32, "gm"),
):
    _tadd_tile(a_ptr, b_ptr, c_ptr, 16, 64)


@pto.jit(
    name="TADD_f32_32x32",
    kernel_kind="vector",
    target="a5",
    func_attr="pto.aicore",
)
def TADD_f32_32x32(
    a_ptr: pto.ptr(pto.float32, "gm"),
    b_ptr: pto.ptr(pto.float32, "gm"),
    c_ptr: pto.ptr(pto.float32, "gm"),
):
    _tadd_tile(a_ptr, b_ptr, c_ptr, 32, 32)


def build():
    return pto.merge_jit_modules(TADD_f32_16x64, TADD_f32_32x32)


if __name__ == "__main__":
    print(build())
