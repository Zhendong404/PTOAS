# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
TADD kernel – low-level builder for the TileLang ST testcase.

Generates the same IR as
  test/tilelang_st/npu/a5/src/st/testcase/tadd/tadd.pto

Each case is a flat ``pto.aicore`` kernel that performs
``tload(a) + tload(b) + tadd(a,b)->c + tstore(c)`` on a single tile.
"""

from mlir.ir import (
    Attribute,
    Context,
    F32Type,
    IndexType,
    InsertionPoint,
    Location,
    Module,
    StringAttr,
    UnitAttr,
)
from mlir.dialects import arith, func, pto


def _emit_tadd_kernel(entry, rows: int, cols: int) -> None:
    """Emit one ``TADD_f32_{rows}x{cols}`` kernel body into *entry*."""
    idx = IndexType.get()
    f32 = F32Type.get()
    vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC)
    tile_cfg = pto.TileBufConfigAttr.get(
        pto.BLayoutAttr.get(pto.BLayout.RowMajor),
        pto.SLayoutAttr.get(pto.SLayout.NoneBox),
        512,
        pto.PadValueAttr.get(pto.PadValue.Null),
    )

    tv_type = pto.TensorViewType.get([1, 1, 1, rows, cols], f32)
    ptv_type = pto.PartitionTensorViewType.get([1, 1, 1, rows, cols], f32)
    tile_type = pto.TileBufType.get([rows, cols], f32, vec, [rows, cols], tile_cfg)
    elem_count = rows * cols

    a_ptr, b_ptr, c_ptr = entry.arguments

    c0 = arith.ConstantOp(idx, 0).result
    c1 = arith.ConstantOp(idx, 1).result
    c_rows = arith.ConstantOp(idx, rows).result
    c_cols = c_rows if rows == cols else arith.ConstantOp(idx, cols).result
    c_elems = arith.ConstantOp(idx, elem_count).result

    shape = [c1, c1, c1, c_rows, c_cols]
    strides = [c_elems, c_elems, c_elems, c_cols, c1]
    offsets = [c0, c0, c0, c0, c0]

    a_view = pto.MakeTensorViewOp(tv_type, a_ptr, shape, strides).result
    b_view = pto.MakeTensorViewOp(tv_type, b_ptr, shape, strides).result
    c_view = pto.MakeTensorViewOp(tv_type, c_ptr, shape, strides).result

    a_part = pto.PartitionViewOp(ptv_type, a_view, offsets, shape).result
    b_part = pto.PartitionViewOp(ptv_type, b_view, offsets, shape).result
    c_part = pto.PartitionViewOp(ptv_type, c_view, offsets, shape).result

    a_tile = pto.AllocTileOp(tile_type).result
    b_tile = pto.AllocTileOp(tile_type).result
    c_tile = pto.AllocTileOp(tile_type).result

    pto.TLoadOp(None, a_part, a_tile)
    pto.TLoadOp(None, b_part, b_tile)
    pto.TAddOp(a_tile, b_tile, c_tile)
    pto.TStoreOp(None, c_tile, c_part)


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown():
            f32 = F32Type.get()
            ptr_f32_gm = pto.PtrType.get(
                f32,
                memory_space=pto.AddressSpaceAttr.get(pto.AddressSpace.GM),
            )

            module = Module.create()
            module.operation.attributes["pto.target_arch"] = StringAttr.get("a5")
            module.operation.attributes["pto.kernel_kind"] = Attribute.parse(
                "#pto.kernel_kind<vector>"
            )

            fn_ty = func.FunctionType.get([ptr_f32_gm, ptr_f32_gm, ptr_f32_gm], [])
            cases = (
                ("TADD_f32_16x64", 16, 64),
                ("TADD_f32_32x32", 32, 32),
            )

            with InsertionPoint(module.body):
                for fn_name, rows, cols in cases:
                    fn = func.FuncOp(fn_name, fn_ty)
                    fn.attributes["pto.aicore"] = UnitAttr.get()
                    entry = fn.add_entry_block()
                    with InsertionPoint(entry):
                        _emit_tadd_kernel(entry, rows, cols)
                        func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
