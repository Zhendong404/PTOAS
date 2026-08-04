#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tmrgsort.
#
# tmrgsort sorts interleaved (value, index) 8-byte records inside vec tiles:
#   - Format 1 ("single"): one input list divided into 4 internal blocks, each
#     block sorted by pto.vmrgsort4 then merged -> pto.tile.mrgsort(src, dst,
#     block_len).
#   - Format 2-4 ("multi"): 2..4 already-sorted input lists merged into one
#     sorted output with top-k capacity -> pto.tile.mrgsort([src...], [dst],
#     tmp=..., excuted=..., exhausted=...).
#   - Format 5 ("topk"): iterative Format1 merges followed by a final Format2
#     merge to take the top-k of a fully sorted buffer.
#
# The (value, index) record layout on the device tile (8 bytes per record):
#   - f32: value at even elements, u32 index bits viewed as f32 at odd elements.
#   - f16: [value, pad, index_lo16, index_hi16] per record, index halves viewed
#     as f16.
# Host input buffers below reproduce that interleaved layout, and the golden is
# the same layout after the corresponding sort/merge (compare.py in the legacy
# suite compared values and u32 indices separately; since the hardware moves
# the 8-byte records opaquely, the interleaved buffer is bit-exact and the
# plain golden_output_case full-buffer compare matches the legacy eps-based
# comparison).  Exhausted cases replicate the legacy defensive
# handle_output_data masking (see compare.py) before comparing.

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto

# ---------------------------------------------------------------------------
# Case tables (mirror testcase/tmrgsort/cases.py exactly)
# ---------------------------------------------------------------------------

# (name, np dtype, pto dtype, src_cols, valid_cols, dst_cols, block_len, eps)
SINGLE_SPECS = [
    ("f32_single_1x256_b64", np.float32, pto.f32, 256, 256, 256, 64, 1e-6),
    ("f32_single_1x320_b64", np.float32, pto.f32, 320, 256, 320, 64, 1e-6),
    ("f32_single_1x512_b64", np.float32, pto.f32, 512, 512, 512, 64, 1e-6),
    ("f32_single_1x640_b64", np.float32, pto.f32, 640, 512, 640, 64, 1e-6),
    ("f16_single_1x256_b64", np.float16, pto.f16, 512, 512, 512, 128, 1e-3),
    ("f16_single_1x320_b64", np.float16, pto.f16, 640, 512, 640, 128, 1e-3),
    ("f16_single_1x512_b64", np.float16, pto.f16, 1024, 1024, 1024, 128, 1e-3),
    ("f16_single_1x1024_b256", np.float16, pto.f16, 2048, 2048, 2048, 512, 1e-3),
]

# (name, np dtype, pto dtype, list_num, src_struct_cols, src_view_cols,
#  src_tile_cols, src_tile_valid, dst_view_cols, dst_tile_cols,
#  dst_tile_valid, tmp_cols, topk, exhausted, eps)
MULTI_SPECS = [
    ("f32_2list_b64_basic", np.float32, pto.f32, 2, [128, 128], [256, 256],
     [256, 256], [256, 256], 256, 256, 256, 512, 128, False, 1e-6),
    ("f16_2list_b64_basic", np.float16, pto.f16, 2, [64, 64], [256, 256],
     [256, 256], [256, 256], 256, 256, 256, 512, 64, False, 1e-3),
    ("f32_2list_exhausted", np.float32, pto.f32, 2, [32, 32], [64, 64],
     [256, 256], [64, 64], 128, 256, 128, 512, 64, True, 1e-6),
    ("f32_3list_b64_basic", np.float32, pto.f32, 3, [64, 64, 64],
     [128, 128, 128], [256, 256, 256], [128, 128, 128], 256, 256, 256, 512,
     128, False, 1e-6),
    ("f32_3list_non_uniform", np.float32, pto.f32, 3, [64, 64, 32],
     [128, 128, 64], [256, 256, 256], [128, 128, 64], 128, 256, 128, 512,
     64, False, 1e-6),
    ("f16_3list_exhausted", np.float16, pto.f16, 3, [128, 128, 128],
     [512, 512, 512], [512, 512, 512], [512, 512, 512], 1536, 1536, 1536,
     1536, 384, True, 1e-3),
    ("f32_4list_b32_basic", np.float32, pto.f32, 4, [64, 64, 64, 64],
     [128, 128, 128, 128], [256, 256, 256, 256], [128, 128, 128, 128], 512,
     512, 512, 512, 256, False, 1e-6),
    ("f32_4list_non_uniform", np.float32, pto.f32, 4, [64, 64, 64, 32],
     [128, 128, 128, 64], [256, 256, 256, 256], [128, 128, 128, 64], 448,
     512, 448, 512, 224, False, 1e-6),
    ("f16_4list_b64_basic", np.float16, pto.f16, 4, [64, 64, 64, 64],
     [256, 256, 256, 256], [256, 256, 256, 256], [256, 256, 256, 256],
     1024, 1024, 1024, 1024, 256, False, 1e-3),
    ("f16_4list_basic", np.float16, pto.f16, 4, [64, 64, 64, 64],
     [256, 256, 256, 256], [256, 256, 256, 256], [256, 256, 256, 256],
     1024, 1024, 1024, 1024, 256, False, 1e-3),
]

# (name, np dtype, pto dtype, src_cols, valid_cols, dst_cols, topk, bl1, bl2,
#  block1_offset, block1_size, block1_tile_cols, block1_valid, merge_tmp_cols,
#  merge_dst_cols, merge_dst_valid, dst_tile_cols, dst_tile_valid, eps)
TOPK_SPECS = [
    ("f32_topk_2048_1024", np.float32, pto.f32, 2048, 2048, 1024, 512,
     64, 256, 1024, 1024, 1024, 1024, 2048, 1024, 1024, 1024, 1024, 1e-6),
    ("f32_topk_2048_2048", np.float32, pto.f32, 2048, 2048, 2048, 1024,
     64, 256, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 2048, 1e-6),
    ("f32_topk_1280_512", np.float32, pto.f32, 1280, 1280, 512, 256,
     64, 256, 1024, 256, 1024, 256, 1280, 1280, 512, 1280, 512, 1e-6),
    ("f16_topk_2048_1024", np.float16, pto.f16, 2048, 2048, 1024, 256,
     64, 256, 1024, 1024, 1024, 1024, 2048, 2048, 1024, 2048, 1024, 1e-3),
    ("f16_topk_2048_2048", np.float16, pto.f16, 2048, 2048, 2048, 512,
     64, 256, 1024, 1024, 1024, 1024, 2048, 2048, 2048, 2048, 2048, 1e-3),
    ("f16_topk_1280_512", np.float16, pto.f16, 1280, 1280, 512, 128,
     64, 256, 1024, 256, 512, 256, 1280, 1280, 512, 1280, 512, 1e-3),
]

# ---------------------------------------------------------------------------
# (value, index) record layout helpers
# ---------------------------------------------------------------------------

def _elem_divisor(np_dtype):
    """Elements of the on-tile float dtype per 8-byte (value, index) record."""
    return 4 if np_dtype == np.float16 else 2


def _interleave(values, indices, np_dtype):
    """Interleave (value, u32 index) records into the on-tile float layout."""
    n = values.size
    if np_dtype == np.float32:
        buf = np.zeros(2 * n, dtype=np.float32)
        buf[0::2] = values
        buf[1::2] = indices.astype(np.uint32).view(np.float32)
        return buf
    buf = np.zeros(4 * n, dtype=np.float16)
    buf[0::4] = values
    idx = indices.astype(np.uint32)
    buf[2::4] = (idx & 0xFFFF).astype(np.uint16).view(np.float16)
    buf[3::4] = ((idx >> 16) & 0xFFFF).astype(np.uint16).view(np.float16)
    return buf


def _deinterleave(buf, np_dtype):
    """Recover (values, u32 indices) from an interleaved on-tile buffer."""
    flat = np.ascontiguousarray(buf).reshape(-1)
    if np_dtype == np.float32:
        n = flat.size // 2
        values = flat[0::2].copy()
        indices = np.frombuffer(flat[1::2].tobytes(), dtype=np.uint32).copy()
    else:
        n = flat.size // 4
        values = flat[0::4].copy()
        lo = np.frombuffer(flat[2::4].tobytes(), dtype=np.uint16).astype(np.uint32)
        hi = np.frombuffer(flat[3::4].tobytes(), dtype=np.uint16).astype(np.uint32)
        indices = lo | (hi << np.uint32(16))
    return values[:n], indices[:n]


def _stable_desc_block_sort(values, indices, list_col):
    """Stable descending sort of each list_col-wide block (legacy gen_data)."""
    v2d = values.reshape(-1, list_col)
    i2d = indices.reshape(-1, list_col)
    order = np.argsort(-v2d, kind="stable", axis=1)
    sorted_values = np.take_along_axis(v2d, order, axis=1).reshape(-1)
    sorted_indices = np.take_along_axis(i2d, order, axis=1).reshape(-1)
    return sorted_values, sorted_indices


# ---------------------------------------------------------------------------
# Input / golden generation (port of tmrgsort/gen_data.py)
# ---------------------------------------------------------------------------

def _single_inputs(name, np_dtype, src_cols, valid_cols, block_len):
    # Legacy setup_case_rng: per-case deterministic seed from crc32(case name).
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    divisor = _elem_divisor(np_dtype)
    cols = src_cols // divisor
    valid_structs = valid_cols // divisor
    list_col = block_len // divisor

    values = np.random.uniform(0.0, 1.0, size=(1, valid_structs)).astype(np_dtype)
    indices = np.arange(valid_structs, dtype=np.uint32).reshape(1, valid_structs)
    sorted_values, sorted_indices = _stable_desc_block_sort(values, indices, list_col)

    if cols > valid_structs:
        sorted_values = np.concatenate(
            [sorted_values, np.zeros(cols - valid_structs, dtype=np_dtype)]
        )
        sorted_indices = np.concatenate(
            [sorted_indices, np.zeros(cols - valid_structs, dtype=np.uint32)]
        )
    return [_interleave(sorted_values, sorted_indices, np_dtype).reshape(1, src_cols)]


def _single_expected(input_buf, np_dtype, src_cols, valid_cols, dst_cols, block_len):
    # The device input is already block-sorted (input0.bin); golden merges each
    # group of 4 blocks (block_lens records) with a stable descending sort and
    # pads the file tail with zero records.
    divisor = _elem_divisor(np_dtype)
    cols = src_cols // divisor
    valid_structs = valid_cols // divisor
    block_lens = block_len // divisor * 4

    values, indices = _deinterleave(input_buf, np_dtype)
    golden_values = np.zeros(cols, dtype=np_dtype)
    golden_indices = np.zeros(cols, dtype=np.uint32)

    group_count = valid_structs // block_lens
    if group_count > 0:
        n = group_count * block_lens
        gv = values[:n].reshape(-1, block_lens)
        gi = indices[:n].reshape(-1, block_lens)
        order = np.argsort(-gv, kind="stable", axis=1)
        golden_values[:n] = np.take_along_axis(gv, order, axis=1).reshape(-1)
        golden_indices[:n] = np.take_along_axis(gi, order, axis=1).reshape(-1)
    # valid_structs % block_lens remainder and src_cols > valid_cols padding
    # stay zero, matching gen_data.py.
    return _interleave(golden_values, golden_indices, np_dtype).reshape(1, dst_cols)


def _multi_inputs(name, np_dtype, list_num, src_cols):
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    divisor = _elem_divisor(np_dtype)
    buffers = []
    for i in range(list_num):
        cols_i = src_cols[i]
        values = np.random.uniform(0.0, 1.0, size=(1, cols_i)).astype(np_dtype)
        indices = np.arange(cols_i, dtype=np.uint32).reshape(1, cols_i)
        order = np.argsort(-values, kind="stable", axis=1)
        sorted_values = np.take_along_axis(values, order, axis=1).reshape(-1)
        sorted_indices = np.take_along_axis(indices, order, axis=1).reshape(-1)
        buffers.append(
            _interleave(sorted_values, sorted_indices, np_dtype).reshape(1, cols_i * divisor)
        )
    return buffers


def _find_and_zero(arr, target):
    n = len(arr)
    for i in range(n - 1, -1, -1):
        if arr[i] == target:
            for j in range(i + 1, n):
                arr[j] = 0
            return i
    return -1


def _zero_after_index(arr, i):
    if i < 0 or i >= len(arr):
        return
    for j in range(i + 1, len(arr)):
        arr[j] = 0


def _handle_exhausted_list(input_num, output_global, idx_global, last_data):
    """Port of gen_data.py handle_exhausted_list (pto-isa exhausted semantics)."""
    for i in range(input_num):
        zero_index = _find_and_zero(output_global, last_data[i])
        _zero_after_index(idx_global, zero_index)


def _multi_expected(*inputs, np_dtype, list_num, src_cols, topk, exhausted):
    divisor = _elem_divisor(np_dtype)
    output_values, output_indices, last_data = [], [], []
    for i in range(list_num):
        values, indices = _deinterleave(inputs[i], np_dtype)
        cols_i = src_cols[i]
        output_values.append(values[:cols_i])
        output_indices.append(indices[:cols_i])
        if cols_i > 0:
            last_data.append(values[cols_i - 1])
        else:
            last_data.append(0)

    total = sum(src_cols)
    flat_values = np.concatenate(output_values)
    flat_indices = np.concatenate(output_indices)
    order = np.argsort(-flat_values, kind="stable")
    sorted_values = flat_values[order]
    sorted_indices = flat_indices[order]

    topk_values = sorted_values[:topk]
    topk_indices = sorted_indices[:topk]
    pad_n = total - topk
    if pad_n > 0:
        topk_values = np.concatenate([topk_values, np.zeros(pad_n, dtype=topk_values.dtype)])
        topk_indices = np.concatenate([topk_indices, np.zeros(pad_n, dtype=np.uint32)])

    if exhausted:
        _handle_exhausted_list(list_num, topk_values, topk_indices, last_data)

    golden_values = topk_values[:topk]
    golden_indices = topk_indices[:topk]
    return _interleave(golden_values, golden_indices, np_dtype).reshape(1, topk * divisor)


def _topk_inputs(name, np_dtype, src_cols, valid_cols, block_len):
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    divisor = _elem_divisor(np_dtype)
    cols = valid_cols // divisor
    list_col = block_len // divisor

    values = np.random.uniform(0.0, 1.0, size=(1, cols)).astype(np_dtype)
    indices = np.arange(cols, dtype=np.uint32).reshape(1, cols)
    sorted_values, sorted_indices = _stable_desc_block_sort(values, indices, list_col)
    return [_interleave(sorted_values, sorted_indices, np_dtype).reshape(1, src_cols)]


def _topk_expected(input_buf, np_dtype, src_cols, valid_cols, dst_cols, topk, block_len):
    # Port of gen_data.py gen_golden_topk: iterative Format1 merge
    # (blockLen *= 4) then a final global sort for the tail, take top-k.
    divisor = _elem_divisor(np_dtype)
    cols = valid_cols // divisor
    list_col = block_len // divisor

    values, indices = _deinterleave(input_buf, np_dtype)
    current_values = values[:cols].copy()
    current_indices = indices[:cols].copy()
    current_block_len = list_col

    while current_block_len * 4 <= cols:
        block_lens = current_block_len * 4
        num_groups = cols // block_lens
        for g in range(num_groups):
            start = g * block_lens
            end = start + block_lens
            group = current_values[start:end]
            order = np.argsort(-group, kind="stable")
            current_values[start:end] = group[order]
            current_indices[start:end] = current_indices[start:end][order]
        current_block_len *= 4

    if current_block_len < cols:
        order = np.argsort(-current_values, kind="stable")
        current_values = current_values[order]
        current_indices = current_indices[order]

    golden_values = current_values[:topk]
    golden_indices = current_indices[:topk]
    dst_structures = dst_cols // divisor
    if dst_structures > topk:
        golden_values = np.concatenate(
            [golden_values, np.zeros(dst_structures - topk, dtype=golden_values.dtype)]
        )
        golden_indices = np.concatenate(
            [golden_indices, np.zeros(dst_structures - topk, dtype=np.uint32)]
        )
    return _interleave(golden_values, golden_indices, np_dtype).reshape(1, dst_cols)


# ---------------------------------------------------------------------------
# Kernel builders (auto mode, faithful to tmrgsort.pto)
# ---------------------------------------------------------------------------

def _make_single_kernel(name, pto_dtype, src_cols, valid_cols, dst_cols, block_len):
    @pto.jit(name="tmrgsort_" + name, target="a5")
    def _kernel(
        src_ptr: pto.ptr(pto_dtype, "gm"),
        dst_ptr: pto.ptr(pto_dtype, "gm"),
    ):
        src_view = pto.make_tensor_view(src_ptr, shape=[1, src_cols], strides=[src_cols, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[1, dst_cols], strides=[dst_cols, 1])
        src_tile = pto.alloc_tile(shape=[1, valid_cols], dtype=pto_dtype)
        dst_tile = pto.alloc_tile(shape=[1, valid_cols], dtype=pto_dtype)
        pto.tile.load(src_view, src_tile)
        pto.tile.mrgsort(src_tile, dst_tile, pto.const(block_len, dtype=pto.i32))
        pto.tile.store(dst_tile, dst_view)

    return _kernel


def _valid_kwargs(tile_cols, valid_cols):
    # Match the legacy tile types: `1xNxf32` when valid == shape, otherwise an
    # explicit `valid=1xM` suffix.
    return {"valid_shape": [1, valid_cols]} if valid_cols != tile_cols else {}


def _make_multi2_kernel(name, pto_dtype, src_view_cols, src_tile_cols, src_valid,
                        dst_view_cols, dst_tile_cols, dst_valid, tmp_cols, exhausted):
    (src0_vc, src1_vc) = src_view_cols
    (src0_tc, src1_tc) = src_tile_cols
    (src0_va, src1_va) = src_valid

    @pto.jit(name="tmrgsort_" + name, target="a5")
    def _kernel(
        src0_ptr: pto.ptr(pto_dtype, "gm"),
        src1_ptr: pto.ptr(pto_dtype, "gm"),
        dst_ptr: pto.ptr(pto_dtype, "gm"),
    ):
        src0_view = pto.make_tensor_view(src0_ptr, shape=[1, src0_vc], strides=[src0_vc, 1])
        src1_view = pto.make_tensor_view(src1_ptr, shape=[1, src1_vc], strides=[src1_vc, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[1, dst_view_cols],
                                        strides=[dst_view_cols, 1])
        src0_tile = pto.alloc_tile(shape=[1, src0_tc], dtype=pto_dtype,
                                   **_valid_kwargs(src0_tc, src0_va))
        src1_tile = pto.alloc_tile(shape=[1, src1_tc], dtype=pto_dtype,
                                   **_valid_kwargs(src1_tc, src1_va))
        dst_tile = pto.alloc_tile(shape=[1, dst_tile_cols], dtype=pto_dtype,
                                  **_valid_kwargs(dst_tile_cols, dst_valid))
        tmp_tile = pto.alloc_tile(shape=[1, tmp_cols], dtype=pto_dtype)
        ex_vec = pto.Vec(pto.i16, 4, init=0)

        pto.tile.load(src0_view, src0_tile)
        pto.tile.load(src1_view, src1_tile)
        pto.tile.mrgsort(
            [src0_tile, src1_tile], [dst_tile],
            tmp=tmp_tile, excuted=ex_vec, exhausted=exhausted,
        )
        pto.tile.store(dst_tile, dst_view)

    return _kernel


def _make_multi3_kernel(name, pto_dtype, src_view_cols, src_tile_cols, src_valid,
                        dst_view_cols, dst_tile_cols, dst_valid, tmp_cols, exhausted):
    (src0_vc, src1_vc, src2_vc) = src_view_cols
    (src0_tc, src1_tc, src2_tc) = src_tile_cols
    (src0_va, src1_va, src2_va) = src_valid

    @pto.jit(name="tmrgsort_" + name, target="a5")
    def _kernel(
        src0_ptr: pto.ptr(pto_dtype, "gm"),
        src1_ptr: pto.ptr(pto_dtype, "gm"),
        src2_ptr: pto.ptr(pto_dtype, "gm"),
        dst_ptr: pto.ptr(pto_dtype, "gm"),
    ):
        src0_view = pto.make_tensor_view(src0_ptr, shape=[1, src0_vc], strides=[src0_vc, 1])
        src1_view = pto.make_tensor_view(src1_ptr, shape=[1, src1_vc], strides=[src1_vc, 1])
        src2_view = pto.make_tensor_view(src2_ptr, shape=[1, src2_vc], strides=[src2_vc, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[1, dst_view_cols],
                                        strides=[dst_view_cols, 1])
        src0_tile = pto.alloc_tile(shape=[1, src0_tc], dtype=pto_dtype,
                                   **_valid_kwargs(src0_tc, src0_va))
        src1_tile = pto.alloc_tile(shape=[1, src1_tc], dtype=pto_dtype,
                                   **_valid_kwargs(src1_tc, src1_va))
        src2_tile = pto.alloc_tile(shape=[1, src2_tc], dtype=pto_dtype,
                                   **_valid_kwargs(src2_tc, src2_va))
        dst_tile = pto.alloc_tile(shape=[1, dst_tile_cols], dtype=pto_dtype,
                                  **_valid_kwargs(dst_tile_cols, dst_valid))
        tmp_tile = pto.alloc_tile(shape=[1, tmp_cols], dtype=pto_dtype)
        ex_vec = pto.Vec(pto.i16, 4, init=0)

        pto.tile.load(src0_view, src0_tile)
        pto.tile.load(src1_view, src1_tile)
        pto.tile.load(src2_view, src2_tile)
        pto.tile.mrgsort(
            [src0_tile, src1_tile, src2_tile], [dst_tile],
            tmp=tmp_tile, excuted=ex_vec, exhausted=exhausted,
        )
        pto.tile.store(dst_tile, dst_view)

    return _kernel


def _make_multi4_kernel(name, pto_dtype, src_view_cols, src_tile_cols, src_valid,
                        dst_view_cols, dst_tile_cols, dst_valid, tmp_cols, exhausted):
    (src0_vc, src1_vc, src2_vc, src3_vc) = src_view_cols
    (src0_tc, src1_tc, src2_tc, src3_tc) = src_tile_cols
    (src0_va, src1_va, src2_va, src3_va) = src_valid

    @pto.jit(name="tmrgsort_" + name, target="a5")
    def _kernel(
        src0_ptr: pto.ptr(pto_dtype, "gm"),
        src1_ptr: pto.ptr(pto_dtype, "gm"),
        src2_ptr: pto.ptr(pto_dtype, "gm"),
        src3_ptr: pto.ptr(pto_dtype, "gm"),
        dst_ptr: pto.ptr(pto_dtype, "gm"),
    ):
        src0_view = pto.make_tensor_view(src0_ptr, shape=[1, src0_vc], strides=[src0_vc, 1])
        src1_view = pto.make_tensor_view(src1_ptr, shape=[1, src1_vc], strides=[src1_vc, 1])
        src2_view = pto.make_tensor_view(src2_ptr, shape=[1, src2_vc], strides=[src2_vc, 1])
        src3_view = pto.make_tensor_view(src3_ptr, shape=[1, src3_vc], strides=[src3_vc, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[1, dst_view_cols],
                                        strides=[dst_view_cols, 1])
        src0_tile = pto.alloc_tile(shape=[1, src0_tc], dtype=pto_dtype,
                                   **_valid_kwargs(src0_tc, src0_va))
        src1_tile = pto.alloc_tile(shape=[1, src1_tc], dtype=pto_dtype,
                                   **_valid_kwargs(src1_tc, src1_va))
        src2_tile = pto.alloc_tile(shape=[1, src2_tc], dtype=pto_dtype,
                                   **_valid_kwargs(src2_tc, src2_va))
        src3_tile = pto.alloc_tile(shape=[1, src3_tc], dtype=pto_dtype,
                                   **_valid_kwargs(src3_tc, src3_va))
        dst_tile = pto.alloc_tile(shape=[1, dst_tile_cols], dtype=pto_dtype,
                                  **_valid_kwargs(dst_tile_cols, dst_valid))
        tmp_tile = pto.alloc_tile(shape=[1, tmp_cols], dtype=pto_dtype)
        ex_vec = pto.Vec(pto.i16, 4, init=0)

        pto.tile.load(src0_view, src0_tile)
        pto.tile.load(src1_view, src1_tile)
        pto.tile.load(src2_view, src2_tile)
        pto.tile.load(src3_view, src3_tile)
        pto.tile.mrgsort(
            [src0_tile, src1_tile, src2_tile, src3_tile], [dst_tile],
            tmp=tmp_tile, excuted=ex_vec, exhausted=exhausted,
        )
        pto.tile.store(dst_tile, dst_view)

    return _kernel


def _make_topk_kernel(name, pto_dtype, src_cols, dst_cols, bl1, bl2,
                      block1_offset, block1_size, block1_tile_cols, block1_valid,
                      merge_tmp_cols, merge_dst_cols, merge_dst_valid,
                      dst_tile_cols, dst_tile_valid):
    @pto.jit(name="tmrgsort_" + name, target="a5")
    def _kernel(
        src_ptr: pto.ptr(pto_dtype, "gm"),
        dst_ptr: pto.ptr(pto_dtype, "gm"),
    ):
        src_view = pto.make_tensor_view(src_ptr, shape=[1, src_cols], strides=[src_cols, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[1, dst_cols], strides=[dst_cols, 1])

        src_tile = pto.alloc_tile(shape=[1, src_cols], dtype=pto_dtype)
        tmp_tile = pto.alloc_tile(shape=[1, src_cols], dtype=pto_dtype)
        dst_tile = pto.alloc_tile(shape=[1, dst_tile_cols], dtype=pto_dtype,
                                  **_valid_kwargs(dst_tile_cols, dst_tile_valid))
        block0_tile = pto.alloc_tile(shape=[1, block1_offset], dtype=pto_dtype)
        block1_tile = pto.alloc_tile(shape=[1, block1_tile_cols], dtype=pto_dtype,
                                     **_valid_kwargs(block1_tile_cols, block1_valid))
        merge_tmp_tile = pto.alloc_tile(shape=[1, merge_tmp_cols], dtype=pto_dtype)
        merge_dst_tile = pto.alloc_tile(shape=[1, merge_dst_cols], dtype=pto_dtype,
                                        **_valid_kwargs(merge_dst_cols, merge_dst_valid))
        ex_vec = pto.Vec(pto.i16, 4, init=0)

        # Iteration 1: blockLen=bl1, merge 4 blocks per group.
        pto.tile.load(src_view, src_tile)
        pto.tile.mrgsort(src_tile, tmp_tile, pto.const(bl1, dtype=pto.i32))
        # Copy result back for the next iteration.
        pto.tile.mov(tmp_tile, src_tile)
        # Iteration 2: blockLen=bl2, leaving two sorted blocks.
        pto.tile.mrgsort(src_tile, tmp_tile, pto.const(bl2, dtype=pto.i32))
        # Store back to src memory (reuse as intermediate buffer).
        pto.tile.store(tmp_tile, src_view)

        # Final Format2 merge of the two sorted blocks, top-k into dst.
        block0_view = pto.partition_view(src_view, offsets=[0, 0], sizes=[1, block1_offset])
        block1_view = pto.partition_view(src_view, offsets=[0, block1_offset],
                                         sizes=[1, block1_size])
        pto.tile.load(block0_view, block0_tile)
        pto.tile.load(block1_view, block1_tile)
        pto.tile.mrgsort(
            [block0_tile, block1_tile], [merge_dst_tile],
            tmp=merge_tmp_tile, excuted=ex_vec, exhausted=False,
        )
        pto.tile.mov(merge_dst_tile, dst_tile)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


# ---------------------------------------------------------------------------
# Case assembly
# ---------------------------------------------------------------------------

_kernels = {}

for _spec in SINGLE_SPECS:
    _name, _np_dtype, _pto_dtype, _src_cols, _valid_cols, _dst_cols, _block_len, _eps = _spec
    _kernels[_name] = _make_single_kernel(_name, _pto_dtype, _src_cols, _valid_cols, _dst_cols, _block_len)

for _spec in MULTI_SPECS:
    (_name, _np_dtype, _pto_dtype, _list_num, _src_struct_cols, _src_view_cols,
     _src_tile_cols, _src_valid, _dst_view_cols, _dst_tile_cols, _dst_valid,
     _tmp_cols, _topk, _exhausted, _eps) = _spec
    if _list_num == 2:
        _kernels[_name] = _make_multi2_kernel(
            _name, _pto_dtype, _src_view_cols, _src_tile_cols, _src_valid,
            _dst_view_cols, _dst_tile_cols, _dst_valid, _tmp_cols, _exhausted)
    elif _list_num == 3:
        _kernels[_name] = _make_multi3_kernel(
            _name, _pto_dtype, _src_view_cols, _src_tile_cols, _src_valid,
            _dst_view_cols, _dst_tile_cols, _dst_valid, _tmp_cols, _exhausted)
    else:
        _kernels[_name] = _make_multi4_kernel(
            _name, _pto_dtype, _src_view_cols, _src_tile_cols, _src_valid,
            _dst_view_cols, _dst_tile_cols, _dst_valid, _tmp_cols, _exhausted)

for _spec in TOPK_SPECS:
    (_name, _np_dtype, _pto_dtype, _src_cols, _valid_cols, _dst_cols, _topk,
     _bl1, _bl2, _block1_offset, _block1_size, _block1_tile_cols, _block1_valid,
     _merge_tmp_cols, _merge_dst_cols, _merge_dst_valid, _dst_tile_cols,
     _dst_tile_valid, _eps) = _spec
    _kernels[_name] = _make_topk_kernel(
        _name, _pto_dtype, _src_cols, _dst_cols, _bl1, _bl2,
        _block1_offset, _block1_size, _block1_tile_cols, _block1_valid,
        _merge_tmp_cols, _merge_dst_cols, _merge_dst_valid,
        _dst_tile_cols, _dst_tile_valid)


def _exhausted_case(name, kernel, inputs, expected, out_shape, np_dtype, topk, eps):
    """golden_output_case variant replicating the legacy exhausted compare.

    compare.py applied handle_output_data() to the device output before the
    numeric comparison (zeroing output values/indices where the golden is zero,
    scanning from the end), so the migrated check reproduces that masking and
    compares the top-k values with the legacy eps and the indices exactly.
    """

    def make_case():
        host_inputs = [np.array(v, copy=True) for v in inputs()]
        golden = np.array(expected(*host_inputs), copy=True)
        out = np.zeros(out_shape, dtype=golden.dtype)
        return [*host_inputs, out], golden

    def check(device_inputs, golden):
        actual = np.asarray(device_inputs[-1].cpu().numpy())
        g_values, g_indices = _deinterleave(golden, np_dtype)
        o_values, o_indices = _deinterleave(actual, np_dtype)
        size = len(g_values)
        i = size - 1
        while i > 0:
            if g_values[i] == 0.0:
                o_values[i] = 0.0
                if g_indices[i] == 0:
                    o_indices[i] = 0
                i -= 1
            else:
                break
        np.testing.assert_allclose(g_values[:topk], o_values[:topk], rtol=eps, atol=eps)
        np.testing.assert_array_equal(g_indices[:topk], o_indices[:topk])

    return {"name": name, "kernel": kernel, "make_case": make_case, "check": check}


CASES = []

for _spec in SINGLE_SPECS:
    _name, _np_dtype, _pto_dtype, _src_cols, _valid_cols, _dst_cols, _block_len, _eps = _spec
    CASES.append(
        golden_output_case(
            "tmrgsort_" + _name,
            _kernels[_name],
            inputs=lambda n=_name, d=_np_dtype, sc=_src_cols, vc=_valid_cols, bl=_block_len:
                _single_inputs(n, d, sc, vc, bl),
            expected=lambda src, d=_np_dtype, sc=_src_cols, vc=_valid_cols,
                             dc=_dst_cols, bl=_block_len:
                _single_expected(src, d, sc, vc, dc, bl),
            rtol=_eps,
            atol=_eps,
        )
    )

for _spec in MULTI_SPECS:
    (_name, _np_dtype, _pto_dtype, _list_num, _src_struct_cols, _src_view_cols,
     _src_tile_cols, _src_valid, _dst_view_cols, _dst_tile_cols, _dst_valid,
     _tmp_cols, _topk, _exhausted, _eps) = _spec
    if _exhausted:
        CASES.append(
            _exhausted_case(
                "tmrgsort_" + _name,
                _kernels[_name],
                inputs=lambda n=_name, d=_np_dtype, ln=_list_num, sc=_src_struct_cols:
                    _multi_inputs(n, d, ln, sc),
                expected=lambda *ins, d=_np_dtype, ln=_list_num, sc=_src_struct_cols,
                              tk=_topk, ex=_exhausted:
                    _multi_expected(*ins, np_dtype=d, list_num=ln, src_cols=sc,
                                    topk=tk, exhausted=ex),
                out_shape=(1, _dst_view_cols),
                np_dtype=_np_dtype,
                topk=_topk,
                eps=_eps,
            )
        )
    else:
        CASES.append(
            golden_output_case(
                "tmrgsort_" + _name,
                _kernels[_name],
                inputs=lambda n=_name, d=_np_dtype, ln=_list_num, sc=_src_struct_cols:
                    _multi_inputs(n, d, ln, sc),
                expected=lambda *ins, d=_np_dtype, ln=_list_num, sc=_src_struct_cols,
                              tk=_topk, ex=_exhausted:
                    _multi_expected(*ins, np_dtype=d, list_num=ln, src_cols=sc,
                                    topk=tk, exhausted=ex),
                rtol=_eps,
                atol=_eps,
            )
        )

for _spec in TOPK_SPECS:
    (_name, _np_dtype, _pto_dtype, _src_cols, _valid_cols, _dst_cols, _topk,
     _bl1, _bl2, _block1_offset, _block1_size, _block1_tile_cols, _block1_valid,
     _merge_tmp_cols, _merge_dst_cols, _merge_dst_valid, _dst_tile_cols,
     _dst_tile_valid, _eps) = _spec
    CASES.append(
        golden_output_case(
            "tmrgsort_" + _name,
            _kernels[_name],
            inputs=lambda n=_name, d=_np_dtype, sc=_src_cols, vc=_valid_cols,
                          bl=_bl1:
                _topk_inputs(n, d, sc, vc, bl),
            expected=lambda src, d=_np_dtype, sc=_src_cols, vc=_valid_cols,
                            dc=_dst_cols, tk=_topk, bl=_bl1:
                _topk_expected(src, d, sc, vc, dc, tk, bl),
            rtol=_eps,
            atol=_eps,
        )
    )


auto_main(globals())
