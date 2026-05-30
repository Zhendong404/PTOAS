# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Minimal PTODSL mixed-backend example with one kernel module.

Shape:
  - entry kernel: EmitC backend
  - kernel module: VPTO backend, explicit mode

This is the smallest standalone script for checking that PTODSL now emits an
outer container with child modules carrying different `pto.backend` attrs.
"""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "ptodsl" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            break
    else:
        raise RuntimeError(
            "Unable to locate the PTODSL Python package root from mixed_backend_kernel_module.py"
        )

from ptodsl import pto


@pto.jit(target="a5", entry=False, backend="vpto", mode="explicit", insert_sync=False)
def copy_row_kernel_module(
    src_tile: pto.Tile,
    dst_tile: pto.Tile,
    cols: pto.i32,
):
    vec = pto.elements_per_vreg(pto.f32)
    loop = pto.for_(0, cols, step=vec).carry(remained=cols)
    with loop:
        c = loop.iv
        mask, remained = pto.make_mask(pto.f32, loop.remained)
        src_vec = pto.vlds(src_tile[0, c:])
        pto.vsts(src_vec, dst_tile[0, c:], mask)
        loop.update(remained=remained)


@pto.jit(target="a5", backend="emitc")
def emitc_entry_calls_vpto_module(
    x_ptr: pto.ptr(pto.f32, "gm"),
    o_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
):
    x_view = pto.make_tensor_view(x_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(o_ptr, shape=[rows, cols], strides=[cols, 1])

    x_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)

    with pto.for_(0, rows, step=1) as row:
        x_part = pto.partition_view(x_view, offsets=[row, 0], sizes=[1, cols])
        o_part = pto.partition_view(o_view, offsets=[row, 0], sizes=[1, cols])
        pto.tile.load(x_part, x_tile)
        copy_row_kernel_module(x_tile, o_tile, cols)
        pto.tile.store(o_tile, o_part)


def main() -> None:
    compiled = emitc_entry_calls_vpto_module.compile()
    print(compiled.mlir_text())


if __name__ == "__main__":
    main()
