# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for ``pto.tstore`` and ``pto.tstore_fp``."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._load_store import (
    ACC_STORE_DTYPES,
    LOAD_STORE_DTYPES,
    tstore_acc_nz2dn_constraint,
    tstore_acc_nz2nd_constraint,
    tstore_acc_nz2nz_constraint,
    tstore_dn_constraint,
    tstore_fp_constraint,
    tstore_nd_constraint,
    tstore_nz_constraint,
    dma_hw_loop_source_legal,
)


@tilelib.tile_template(
    op="pto.tstore",
    target="a5",
    name="template_tstore_nd",
    dtypes=LOAD_STORE_DTYPES,
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[tstore_nd_constraint],
    id=0,
    loop_depth=3,
    is_post_update=False,
    tags=("store", "ub", "gm", "nd"),
)
def template_tstore_nd(src: pto.Tile, dst: pto.PartitionTensorView):
    elem_bytes = pto.bytewidth(src.dtype)
    if len(dst.shape) == 2:
        valid_rows, valid_cols = src.valid_shape
        _, ub_cols = src.shape
        row_stride, _ = dst.strides
        row_stride = valid_cols if row_stride is None else row_stride
        pto.mte_store(
            src.as_ptr(),
            dst.as_ptr(),
            valid_cols * elem_bytes,
            nburst=(valid_rows, ub_cols * elem_bytes, row_stride * elem_bytes),
        )
        return

    g0, g1, g2, g3, g4 = dst.shape
    s0, s1, s2, s3, s4 = dst.strides
    valid_rows, valid_cols = src.valid_shape
    _, ub_cols = src.shape

    n_burst = valid_rows if g0 == 1 and g1 == 1 and g2 == 1 and g3 is None else g3
    len_burst = valid_cols * elem_bytes
    ub_stride = ub_cols * elem_bytes
    gm_stride = 0 if g3 == 1 or s3 is None else s3 * elem_bytes

    src_stride2 = (valid_rows if g3 is None else g3) * ub_cols
    src_stride1 = g2 * src_stride2
    src_stride0 = g1 * src_stride1

    # Preserve A5's semantic nesting: loop2 is the outer g1 level and loop1
    # is the inner g2 level.  A singleton level must remain present so the
    # other level is not renumbered by VPTO expansion.
    loops = []
    if g1 is not None:
        loops.append((g1, src_stride1 * elem_bytes, s1 * elem_bytes))
    if g2 is not None:
        loops.append((g2, src_stride2 * elem_bytes, s2 * elem_bytes))

    ub_ptr = src.as_ptr()
    gm_ptr = dst.as_ptr()
    use_hw_loops = (
        dma_hw_loop_source_legal(g1, src_stride1 * elem_bytes)
        and dma_hw_loop_source_legal(g2, src_stride2 * elem_bytes)
        and dma_hw_loop_source_legal(n_burst, s3 * elem_bytes)
    )
    if not use_hw_loops:
        for i in range(0, g0, 1):
            ub_offset0 = 0 if s0 is None else i * src_stride0
            for j in range(0, g1, 1):
                for k in range(0, g2, 1):
                    for l in range(0, n_burst, 1):
                        pto.mte_store(
                            pto.addptr(ub_ptr, ub_offset0 + j * src_stride1 + k * src_stride2 + l * ub_cols),
                            pto.addptr(gm_ptr, (0 if s0 is None else i * s0) + j * s1 + k * s2 + l * s3),
                            len_burst,
                            nburst=(1, 0, 0),
                        )
        return
    if g0 == 1 and s0 is None:
        pto.mte_store(
            ub_ptr,
            gm_ptr,
            len_burst,
            nburst=(n_burst, ub_stride, gm_stride),
            loops=loops or None,
        )
    else:
        for i in range(0, g0, 1):
            pto.mte_store(
                pto.addptr(ub_ptr, i * src_stride0),
                pto.addptr(gm_ptr, i * s0),
                len_burst,
                nburst=(n_burst, ub_stride, gm_stride),
                loops=loops or None,
            )


@tilelib.tile_template(
    op="pto.tstore",
    target="a5",
    name="template_tstore_dn",
    dtypes=LOAD_STORE_DTYPES,
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[tstore_dn_constraint],
    id=1,
    loop_depth=3,
    is_post_update=False,
    tags=("store", "ub", "gm", "dn"),
)
def template_tstore_dn(src: pto.Tile, dst: pto.PartitionTensorView):
    elem_bytes = pto.bytewidth(src.dtype)
    g0, g1, g2, g3, g4 = dst.shape
    s0, s1, s2, s3, s4 = dst.strides
    valid_rows, valid_cols = src.valid_shape
    ub_rows, _ = src.shape

    n_burst = valid_cols if g4 is None else g4
    len_burst = valid_rows * elem_bytes
    gm_stride = 0 if g4 == 1 or s4 is None else s4 * elem_bytes
    ub_stride = ub_rows * elem_bytes

    src_stride2 = ub_rows * n_burst
    src_stride1 = g2 * src_stride2
    src_stride0 = g1 * src_stride1

    # The first grouped loop is lowered as hardware loop2 (outer), and the
    # second as loop1 (inner), matching the legacy A5 TSTORE mapping.
    loops = []
    if g1 is not None:
        loops.append((g1, src_stride1 * elem_bytes, s1 * elem_bytes))
    if g2 is not None:
        loops.append((g2, src_stride2 * elem_bytes, s2 * elem_bytes))

    ub_ptr = src.as_ptr()
    gm_ptr = dst.as_ptr()
    use_hw_loops = (
        dma_hw_loop_source_legal(g1, src_stride1 * elem_bytes)
        and dma_hw_loop_source_legal(g2, src_stride2 * elem_bytes)
        and dma_hw_loop_source_legal(n_burst, s4 * elem_bytes)
    )
    if not use_hw_loops:
        for i in range(0, g0, 1):
            ub_offset0 = 0 if s0 is None else i * src_stride0
            for j in range(0, g1, 1):
                for k in range(0, g2, 1):
                    for l in range(0, n_burst, 1):
                        pto.mte_store(
                            pto.addptr(ub_ptr, ub_offset0 + j * src_stride1 + k * src_stride2 + l * ub_rows),
                            pto.addptr(gm_ptr, (0 if s0 is None else i * s0) + j * s1 + k * s2 + l * s4),
                            len_burst,
                            nburst=(1, 0, 0),
                        )
        return
    if g0 == 1 and s0 is None:
        pto.mte_store(
            ub_ptr,
            gm_ptr,
            len_burst,
            nburst=(n_burst, ub_stride, gm_stride),
            loops=loops or None,
        )
    else:
        for i in range(0, g0, 1):
            pto.mte_store(
                pto.addptr(ub_ptr, i * src_stride0),
                pto.addptr(gm_ptr, i * s0),
                len_burst,
                nburst=(n_burst, ub_stride, gm_stride),
                loops=loops or None,
            )


@tilelib.tile_template(
    op="pto.tstore",
    target="a5",
    name="template_tstore_nz",
    dtypes=LOAD_STORE_DTYPES,
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[tstore_nz_constraint],
    id=2,
    loop_depth=1,
    is_post_update=False,
    tags=("store", "ub", "gm", "nz"),
)
def template_tstore_nz(src: pto.Tile, dst: pto.PartitionTensorView):
    elem_bytes = pto.bytewidth(src.dtype)
    g0, g1, g2, g3, g4 = dst.shape
    s0, s1, s2, s3, s4 = dst.strides
    valid_rows, _ = src.valid_shape
    ub_rows, _ = src.shape

    c0_size_bytes = 32
    n_burst = g1
    len_burst = valid_rows * c0_size_bytes
    gm_stride = s1 * elem_bytes
    ub_stride = ub_rows * c0_size_bytes
    tile_stride = g1 * ub_rows * g4

    ub_ptr = src.as_ptr()
    gm_ptr = dst.as_ptr()
    for i in range(0, g0, 1):
        pto.mte_store(
            pto.addptr(ub_ptr, i * tile_stride),
            pto.addptr(gm_ptr, i * s0),
            len_burst,
            nburst=(n_burst, ub_stride, gm_stride),
        )


@tilelib.tile_template(
    op="pto.tstore",
    target="a5",
    name="template_tstore_acc_to_gm_nz2nd",
    dtypes=ACC_STORE_DTYPES,
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[tstore_acc_nz2nd_constraint],
    priority=1,
    id=3,
    loop_depth=0,
    is_post_update=False,
    tags=("store", "acc", "gm", "nz2nd"),
)
def template_tstore_acc_to_gm_nz2nd(src: pto.Tile, dst: pto.PartitionTensorView):
    m, n = src.valid_shape
    strides = dst.strides
    src_stride = src.shape[0]
    dst_stride = n if strides is None or strides[3] is None else strides[3]

    dst_dtype = str(dst.dtype)
    kwargs = {}
    if str(src.dtype) == "f32" and dst_dtype == "f16":
        kwargs["pre_quant"] = (pto.f16(1.0), "f32_f16")
    elif str(src.dtype) == "f32" and dst_dtype == "bf16":
        kwargs["pre_quant"] = (pto.bf16(1.0), "f32_bf16")

    pto.mte_l0c_gm(
        src.as_ptr(),
        dst.as_ptr(),
        m,
        n,
        src_stride,
        dst_stride,
        0,
        0,
        layout="nz2nd",
        **kwargs,
    )


@tilelib.tile_template(
    op="pto.tstore",
    target="a5",
    name="template_tstore_acc_to_gm_nz2dn",
    dtypes=ACC_STORE_DTYPES,
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[tstore_acc_nz2dn_constraint],
    priority=1,
    id=4,
    loop_depth=0,
    is_post_update=False,
    tags=("store", "acc", "gm", "nz2dn"),
)
def template_tstore_acc_to_gm_nz2dn(src: pto.Tile, dst: pto.PartitionTensorView):
    m, n = src.valid_shape
    strides = dst.strides
    src_stride = src.shape[0]
    # NZ2DN writes n contiguous DN bursts of m elements.  The fixpipe
    # destination stride is therefore the logical M, as in the A5
    # TStoreAcc/NZ2DN contract; it is not the rank-5 view's last stride.
    dst_stride = m
    loop0_src_stride = 1

    dst_dtype = str(dst.dtype)
    kwargs = {}
    if str(src.dtype) == "f32" and dst_dtype == "f16":
        kwargs["pre_quant"] = (pto.f16(1.0), "f32_f16")
    elif str(src.dtype) == "f32" and dst_dtype == "bf16":
        kwargs["pre_quant"] = (pto.bf16(1.0), "f32_bf16")

    pto.mte_l0c_gm(
        src.as_ptr(),
        dst.as_ptr(),
        m,
        n,
        src_stride,
        dst_stride,
        0,
        0,
        layout=("nz2dn", loop0_src_stride),
        **kwargs,
    )


@tilelib.tile_template(
    op="pto.tstore",
    target="a5",
    name="template_tstore_acc_to_gm_nz2nz",
    dtypes=ACC_STORE_DTYPES,
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[tstore_acc_nz2nz_constraint],
    priority=1,
    id=5,
    loop_depth=0,
    is_post_update=False,
    tags=("store", "acc", "gm", "nz2nz"),
)
def template_tstore_acc_to_gm_nz2nz(src: pto.Tile, dst: pto.PartitionTensorView):
    m, n = src.valid_shape
    src_stride = src.shape[0]
    # The NZ destination stride is the physical distance between consecutive
    # NZ rows, not the logical column count.  This is the same contract as
    # pto::TStoreAccNZ: one destination row is a 16-row fractal times C0.
    c0_size = 16
    if str(dst.dtype) == "f32" and dst.shape[3] == 8:
        c0_size = 8
    dst_stride = ((m + 15) // 16) * 16 * c0_size

    dst_dtype = str(dst.dtype)
    kwargs = {}
    if str(src.dtype) == "f32" and dst_dtype == "f16":
        kwargs["pre_quant"] = (pto.f16(1.0), "f32_f16")
    elif str(src.dtype) == "f32" and dst_dtype == "bf16":
        kwargs["pre_quant"] = (pto.bf16(1.0), "f32_bf16")

    pto.mte_l0c_gm(
        src.as_ptr(),
        dst.as_ptr(),
        m,
        n,
        src_stride,
        dst_stride,
        0,
        0,
        layout=("nz2nz", 1),
        **kwargs,
    )


@tilelib.tile_template(
    op="pto.tstore_fp",
    target="a5",
    name="template_tstore_fp_acc_to_gm",
    dtypes=(("f32", "f16", "f16"), ("f32", "bf16", "bf16")),
    iteration_axis="none",
    op_engine="other",
    op_class="movement",
    constraints=[tstore_fp_constraint],
    id=0,
    loop_depth=0,
    is_post_update=False,
    tags=("store", "acc", "gm", "fp"),
)
def template_tstore_fp_acc_to_gm(src: pto.Tile, fp: pto.Tile, dst: pto.PartitionTensorView):
    m, n = src.valid_shape
    strides = dst.strides
    quant_mode = "qf322bf16_pre_vec" if str(fp.dtype) == "bf16" else "qf322f16_pre_vec"
    src_stride = src.shape[0]
    dst_stride = n if strides is None or strides[3] is None else strides[3]

    pto.mte_l0c_gm(
        src.as_ptr(),
        dst.as_ptr(),
        m,
        n,
        src_stride,
        dst_stride,
        0,
        0,
        layout="nz2nd",
        pre_quant=(fp.as_ptr(), quant_mode),
    )
