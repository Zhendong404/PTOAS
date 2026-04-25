# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.trowargmin"""

import sys
from pathlib import Path
import tilelang_dsl as pto

@pto.vkernel(
    target="a5",
    op="pto.trowargmin",
    advanced=True,
)
def template_trowargmin(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
    src_dtype = src.element_type
    idx_dtype = dst.element_type
    lanes = pto.get_lanes(src_dtype)
    valid_rows, valid_cols = src.valid_shape

    # Initialize with dtype-specific maximum value (aligned with pto-isa Padding<T>::Max)
    if pto.constexpr(src_dtype == pto.f32):
        init_val = pto.f32("0x7F7FFFFF")  # FLT_MAX, IEEE 0x7F7FFFFF
    elif pto.constexpr(src_dtype == pto.f16):
        init_val = pto.f16("0x7BFF")  # F16_MAX, IEEE 0x7BFF
    elif pto.constexpr(src_dtype == pto.si16):
        init_val = pto.si16("0x7FFF")  # INT16_MAX
    elif pto.constexpr(src_dtype == pto.si32):
        init_val = pto.si32("0x7FFFFFFF")  # INT32_MAX

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        
        v_val_acc = pto.vbr(init_val)
        init_zero_idx = idx_dtype(0)
        v_idx_acc = pto.vbr(init_zero_idx)

        # Masks: src for data ops, idx_dtype for index ops and final store
        mask_1, _ = pto.make_mask(src_dtype, 1)
        mask_1_idx, _ = pto.make_mask(idx_dtype, 1)
        
        # Process all column chunks
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(src_dtype, remained)
            v_src = pto.vlds(src[row, col:])
            v_reduced = pto.vcmin(v_src, mask)
            
            v_val, v_idx = pto.vdintlv(v_reduced, pto.vbr(src_dtype(0)))
            v_idx = pto.vbitcast(v_idx, idx_dtype)

            # Add absolute col offset to the chunk's local index
            col_offset = idx_dtype(col)
            v_idx = pto.vadds(v_idx, col_offset, mask_1_idx)
            
            # Compare current chunk min with global min so far
            cmp_mask = pto.vcmp(v_val_acc, v_val, mask_1, "gt")
            
            # Update global min and global argmin
            v_val_acc = pto.vsel(v_val, v_val_acc, cmp_mask)
            # v_idx_acc is ui32, requires b32 mask; cast cmp_mask from src_dtype's mask to b32
            cmp_mask_b32 = pto.pbitcast(cmp_mask, pto.mask_b32)
            v_idx_acc = pto.vsel(v_idx, v_idx_acc, cmp_mask_b32)

        # Store index accumulator to destination tile
        pto.vsts(v_idx_acc, dst[row, 0:], mask_1_idx)
    return
