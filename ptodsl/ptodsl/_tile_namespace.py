# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from . import _ops


class _TileNamespace:
    load = staticmethod(_ops.tload)
    store = staticmethod(_ops.tstore)
    mov = staticmethod(_ops.tmov)

    add = staticmethod(_ops.tadd)
    sub = staticmethod(_ops.tsub)
    mul = staticmethod(_ops.tmul)
    div = staticmethod(_ops.tdiv)
    max = staticmethod(_ops.tmax)
    min = staticmethod(_ops.tmin)

    adds = staticmethod(_ops.tadds)
    subs = staticmethod(_ops.tsubs)
    muls = staticmethod(_ops.tmuls)
    divs = staticmethod(_ops.tdivs)
    maxs = staticmethod(_ops.tmaxs)
    mins = staticmethod(_ops.tmins)

    exp = staticmethod(_ops.texp)
    log = staticmethod(_ops.tlog)
    sqrt = staticmethod(_ops.tsqrt)
    rsqrt = staticmethod(_ops.trsqrt)
    recip = staticmethod(_ops.trecip)
    abs = staticmethod(_ops.tabs)
    neg = staticmethod(_ops.tneg)

    relu = staticmethod(_ops.trelu)
    lrelu = staticmethod(_ops.tlrelu)

    row_sum = staticmethod(_ops.trowsum)
    row_max = staticmethod(_ops.trowmax)
    row_min = staticmethod(_ops.trowmin)
    row_prod = staticmethod(_ops.trowprod)
    row_argmax = staticmethod(_ops.trowargmax)
    row_argmin = staticmethod(_ops.trowargmin)

    col_sum = staticmethod(_ops.tcolsum)
    col_max = staticmethod(_ops.tcolmax)
    col_min = staticmethod(_ops.tcolmin)
    col_prod = staticmethod(_ops.tcolprod)
    col_argmax = staticmethod(_ops.tcolargmax)
    col_argmin = staticmethod(_ops.tcolargmin)

    cmp = staticmethod(_ops.tcmp)
    cmps = staticmethod(_ops.tcmps)

    expands = staticmethod(_ops.texpands)
    row_expand = staticmethod(_ops.trowexpand)
    col_expand = staticmethod(_ops.tcolexpand)

    row_expand_add = staticmethod(_ops.trowexpandadd)
    row_expand_sub = staticmethod(_ops.trowexpandsub)
    row_expand_mul = staticmethod(_ops.trowexpandmul)
    row_expand_div = staticmethod(_ops.trowexpanddiv)
    row_expand_max = staticmethod(_ops.trowexpandmax)
    row_expand_min = staticmethod(_ops.trowexpandmin)
    row_expand_expdif = staticmethod(_ops.trowexpandexpdif)

    col_expand_add = staticmethod(_ops.tcolexpandadd)
    col_expand_sub = staticmethod(_ops.tcolexpandsub)
    col_expand_mul = staticmethod(_ops.tcolexpandmul)
    col_expand_div = staticmethod(_ops.tcolexpanddiv)
    col_expand_max = staticmethod(_ops.tcolexpandmax)
    col_expand_min = staticmethod(_ops.tcolexpandmin)
    col_expand_expdif = staticmethod(_ops.tcolexpandexpdif)

    sel = staticmethod(_ops.tsel)
    sels = staticmethod(_ops.tsels)
    cvt = staticmethod(_ops.tcvt)

    not_ = staticmethod(_ops.tnot)
    and_ = staticmethod(_ops.tand)
    ands = staticmethod(_ops.tands)
    or_ = staticmethod(_ops.tor)
    ors = staticmethod(_ops.tors)
    xor = staticmethod(_ops.txor)
    xors = staticmethod(_ops.txors)
    shl = staticmethod(_ops.tshl)
    shls = staticmethod(_ops.tshls)
    shr = staticmethod(_ops.tshr)
    shrs = staticmethod(_ops.tshrs)

    part_add = staticmethod(_ops.tpartadd)
    part_mul = staticmethod(_ops.tpartmul)
    part_max = staticmethod(_ops.tpartmax)
    part_min = staticmethod(_ops.tpartmin)

    fill_pad = staticmethod(_ops.tfillpad)
    fill_pad_expand = staticmethod(_ops.tfillpad_expand)
    fill_pad_inplace = staticmethod(_ops.tfillpad_inplace)


tile = _TileNamespace()
