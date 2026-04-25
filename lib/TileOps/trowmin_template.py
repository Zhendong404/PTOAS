# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.trowmin"""

import sys
from pathlib import Path
import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.trowmin",
    advanced=True,
)
def template_trowmin(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
    dtype = dst.element_type
    lanes = pto.get_lanes(dtype)
    valid_rows, valid_cols = src.valid_shape

    # Initialize with dtype-specific maximum value (aligned with pto-isa Padding<T>::Max)
    if pto.constexpr(dtype == pto.f32):
      init_val = pto.f32("0x7F7FFFFF")  # FLT_MAX, IEEE 0x7F7FFFFF
    elif pto.constexpr(dtype == pto.f16):
      init_val = pto.f16("0x7BFF")  # F16_MAX, IEEE 0x7BFF
    elif pto.constexpr(dtype == pto.i32):
      init_val = pto.i32("0x7FFFFFFF")  # INT32_MAX
    elif pto.constexpr(dtype == pto.i16):
      init_val = pto.i16("0x7FFF")  # INT16_MAX

    mask_1, _ = pto.make_mask(dtype, 1)

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        
        # Initialize the accumulator for ROWMIN
        v_acc = pto.vbr(init_val)
        
        # Process column chunks
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            v_src = pto.vlds(src[row, col:])
            
            # vcmin reduces src_dtype to acc_dtype
            v_reduced = pto.vcmin(v_src, mask)

            # Clear masked lanes to init_val for float types so vmin doesn't see NaN
            if pto.constexpr(dtype == pto.f32):
                v_reduced = pto.vsel(v_reduced, v_acc, mask)
            if pto.constexpr(dtype == pto.f16):
                v_reduced = pto.vsel(v_reduced, v_acc, mask)

            # accumulate using the accumulator's mask logic
            v_acc = pto.vmin(v_acc, v_reduced, mask_1)

        # Write final reduction to dest buffer once
        pto.vsts(v_acc, dst[row, 0:], mask_1)
    return
