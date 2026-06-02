# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
CV-split PTODSL port of the hw-native Python Flash Attention example.

This variant preserves the legacy ``cube kernel`` + ``vector kernel`` split and
adapts the cross-kernel communication to the current A5 PTODSL local-pipe
surface. Unlike the previous draft, the Cube and Vector pieces are authored as
``@pto.jit(entry=False)`` kernel modules, and a single outer ``@pto.jit``
entry owns the host-visible ABI.
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in {None, ""}:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "ptodsl" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            break
    else:
        raise RuntimeError("Unable to locate the PTODSL Python package root")

from ptodsl import pto, scalar


S0 = 128
HEAD = 128
CUBE_S1 = 128
VEC_CORES = 2
SLOT_NUM = 8
DEFAULT_S1_TILE = 256
DEFAULT_QK_PRELOAD = 3
DEFAULT_Q_ROWS = S0

QK_C2V_PIPE_ID = 0
P_V2C_PIPE_ID = 1
PV_C2V_PIPE_ID = 2

_DEVICE = "npu:0"
NEG_INF_F32 = -3.4028235e38

_ENTRY_SYMBOL = "hw_native_flash_attention_cv_split"
_CUBE_SYMBOL = "hw_native_flash_attention_cv_split_cube"
_VECTOR_SYMBOL = "hw_native_flash_attention_cv_split_vector"


def _validate_specialization(*, head_dim: int, s1_tile: int, qk_preload: int, causal: bool, q_rows: int) -> None:
    if head_dim != HEAD:
        raise ValueError(f"cv-split flash attention currently requires head_dim={HEAD}, got {head_dim}")
    if s1_tile not in (256, 512):
        raise ValueError(f"s1_tile must be 256 or 512, got {s1_tile}")
    if s1_tile % CUBE_S1 != 0:
        raise ValueError(f"s1_tile={s1_tile} must be a multiple of CUBE_S1={CUBE_S1}")
    if qk_preload not in (3, 4):
        raise ValueError(f"qk_preload must be 3 or 4, got {qk_preload}")
    if causal:
        raise ValueError("hw-native flash attention cv-split port is non-causal; causal=True is not supported yet")
    if q_rows % S0 != 0:
        raise ValueError(f"q_rows={q_rows} must be a multiple of S0={S0}")


def _compute_qb_range(total_q_blocks):
    block_num = scalar.index_cast(pto.get_block_num())
    block_idx = scalar.index_cast(pto.get_block_idx())
    floor_div = total_q_blocks // block_num
    extra = total_q_blocks % block_num
    fat_start = block_idx * (floor_div + 1)
    thin_start = extra * (floor_div + 1) + (block_idx - extra) * floor_div
    qb_start = scalar.select(block_idx < extra, fat_start, thin_start)
    per_core = scalar.select(block_idx < extra, floor_div + 1, floor_div)
    return qb_start, qb_start + per_core


def _specialized_symbol(base: str, *, head_dim: int, s1_tile: int, qk_preload: int, q_rows: int) -> str:
    return f"{base}_h{head_dim}_s1t{s1_tile}_qp{qk_preload}_qr{q_rows}"


def _validate_runtime_problem(*, q_rows: int, s1: int, s1_tile: int, qk_preload: int) -> None:
    if q_rows <= 0:
        raise ValueError(f"q_rows must be positive, got {q_rows}")
    if s1 <= 0:
        raise ValueError(f"s1 must be positive, got {s1}")
    if s1 % s1_tile != 0:
        raise ValueError(f"s1={s1} must be a multiple of s1_tile={s1_tile}")
    if s1 // s1_tile < qk_preload:
        raise ValueError(
            f"s1={s1} provides only {s1 // s1_tile} logical S1 tiles, "
            f"but qk_preload={qk_preload} requires at least that many"
        )


def _reference_flash_attention(q: np.ndarray, k_tokens: np.ndarray, v_tokens: np.ndarray) -> np.ndarray:
    q_f32 = q.astype(np.float32, copy=False)
    k_f32 = k_tokens.astype(np.float32, copy=False)
    v_f32 = v_tokens.astype(np.float32, copy=False)
    scale = 1.0 / np.sqrt(q_f32.shape[1])

    scores = q_f32 @ k_f32.T
    scores *= scale
    row_max = np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores - row_max, dtype=np.float32)
    probs /= np.sum(probs, axis=1, keepdims=True, dtype=np.float32)
    return probs @ v_f32


def _init_runtime():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "hw_native_flash_attention_cv_split.py launch requires a Python environment with torch installed"
        ) from exc
    try:
        import torch_npu
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "hw_native_flash_attention_cv_split.py launch requires a Python environment with torch_npu installed"
        ) from exc

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)
    return torch


def _current_stream(torch):
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001


def _build_flash_attention_entry(
    *,
    head_dim: int = HEAD,
    s1_tile: int = DEFAULT_S1_TILE,
    qk_preload: int = DEFAULT_QK_PRELOAD,
    causal: bool = False,
    q_rows: int = DEFAULT_Q_ROWS,
):
    _validate_specialization(
        head_dim=head_dim,
        s1_tile=s1_tile,
        qk_preload=qk_preload,
        causal=causal,
        q_rows=q_rows,
    )

    cube_symbol = _specialized_symbol(
        _CUBE_SYMBOL,
        head_dim=head_dim,
        s1_tile=s1_tile,
        qk_preload=qk_preload,
        q_rows=q_rows,
    )
    vector_symbol = _specialized_symbol(
        _VECTOR_SYMBOL,
        head_dim=head_dim,
        s1_tile=s1_tile,
        qk_preload=qk_preload,
        q_rows=q_rows,
    )
    entry_symbol = _specialized_symbol(
        _ENTRY_SYMBOL,
        head_dim=head_dim,
        s1_tile=s1_tile,
        qk_preload=qk_preload,
        q_rows=q_rows,
    )

    tile_factor = s1_tile // CUBE_S1
    vec_gu_rows = S0 // VEC_CORES
    vec_s0 = vec_gu_rows // tile_factor
    row_slice_count = tile_factor
    total_q_blocks = q_rows // S0
    scale = 1.0 / (head_dim ** 0.5)

    @pto.jit(name=cube_symbol, target="a5", entry=False, kernel_kind="cube", mode="auto", backend="emitc")
    def flash_attention_cube_kernel(
        gm_q: pto.ptr(pto.f16, "gm"),
        gm_k: pto.ptr(pto.f16, "gm"),
        gm_v: pto.ptr(pto.f16, "gm"),
        gm_qk_slots: pto.ptr(pto.f32, "gm"),
        gm_p_slots: pto.ptr(pto.f16, "gm"),
        gm_pv_slots: pto.ptr(pto.f32, "gm"),
        s0: pto.i32,
        s1: pto.i32,
    ):
        s0_index = scalar.index_cast(s0)
        s1_index = scalar.index_cast(s1)
        num_tiles_s1 = s1_index // s1_tile
        steady_tiles = num_tiles_s1 - qk_preload
        qb_start, qb_end = _compute_qb_range(total_q_blocks)

        q_view = pto.make_tensor_view(gm_q, shape=[s0_index, head_dim], strides=[head_dim, 1])
        k_view = pto.make_tensor_view(gm_k, shape=[head_dim, s1_index], strides=[1, head_dim], layout="DN")
        v_view = pto.make_tensor_view(gm_v, shape=[s1_index, head_dim], strides=[head_dim, 1])

        qk_slot_bytes = 32
        p_slot_bytes = 32
        pv_slot_bytes = 32

        qk_slots = pto.make_tensor_view(gm_qk_slots, shape=[SLOT_NUM * S0, s1_tile], strides=[s1_tile, 1])
        p_slots = pto.make_tensor_view(gm_p_slots, shape=[SLOT_NUM * S0, s1_tile], strides=[s1_tile, 1])
        pv_slots = pto.make_tensor_view(gm_pv_slots, shape=[SLOT_NUM * S0, head_dim], strides=[head_dim, 1])

        qk_consumer_buf = pto.import_reserved_buffer("fa_qk_c2v_fifo", peer_func=vector_symbol)
        pv_consumer_buf = pto.import_reserved_buffer("fa_pv_c2v_fifo", peer_func=vector_symbol)
        p_consumer_buf = pto.reserve_buffer("fa_p_v2c_fifo", size=p_slot_bytes * 8, location="mat")

        qk_pipe = pto.pipe.c2v(slot_size=qk_slot_bytes, consumer_buf=qk_consumer_buf, id=QK_C2V_PIPE_ID)
        p_pipe = pto.pipe.v2c(slot_size=p_slot_bytes, consumer_buf=p_consumer_buf, id=P_V2C_PIPE_ID)
        pv_pipe = pto.pipe.c2v(slot_size=pv_slot_bytes, consumer_buf=pv_consumer_buf, id=PV_C2V_PIPE_ID)

        qk_pipe.init_cube()
        p_pipe.init_cube()
        pv_pipe.init_cube()

        q_mat = pto.alloc_tile(shape=[S0, head_dim], dtype=pto.f16, memory_space="mat")
        q_left = pto.alloc_tile(
            shape=[S0, head_dim],
            dtype=pto.f16,
            memory_space="left",
            blayout="ColMajor",
            slayout="RowMajor",
        )
        k_mat = pto.alloc_tile(
            shape=[head_dim, CUBE_S1],
            dtype=pto.f16,
            memory_space="mat",
            blayout="RowMajor",
            slayout="ColMajor",
        )
        k_right = pto.alloc_tile(
            shape=[head_dim, CUBE_S1],
            dtype=pto.f16,
            memory_space="right",
            slayout="ColMajor",
        )
        qk_acc = pto.alloc_tile(
            shape=[S0, CUBE_S1],
            dtype=pto.f32,
            memory_space="acc",
            blayout="ColMajor",
            slayout="RowMajor",
        )
        qk_tile = pto.alloc_tile(shape=[S0, CUBE_S1], dtype=pto.f32)
        qk_token = pto.alloc_tile(shape=[1, 8], dtype=pto.f32)

        p_recv = pto.alloc_tile(shape=[S0, CUBE_S1], dtype=pto.f16, memory_space="mat")
        p_token = pto.alloc_tile(shape=[1, 16], dtype=pto.f16, memory_space="mat")
        p_left = pto.alloc_tile(
            shape=[S0, CUBE_S1],
            dtype=pto.f16,
            memory_space="left",
            blayout="ColMajor",
            slayout="RowMajor",
        )
        v_mat = pto.alloc_tile(shape=[CUBE_S1, head_dim], dtype=pto.f16, memory_space="mat")
        v_right = pto.alloc_tile(
            shape=[CUBE_S1, head_dim],
            dtype=pto.f16,
            memory_space="right",
            slayout="ColMajor",
        )
        pv_acc = pto.alloc_tile(
            shape=[S0, head_dim],
            dtype=pto.f32,
            memory_space="acc",
            blayout="ColMajor",
            slayout="RowMajor",
        )
        pv_tile = pto.alloc_tile(shape=[S0, head_dim], dtype=pto.f32)
        pv_token = pto.alloc_tile(shape=[1, 8], dtype=pto.f32)

        def slot_row_base(tile_id):
            return (tile_id % SLOT_NUM) * S0

        def emit_qk_tile(tile_id):
            tile_base = tile_id * s1_tile
            slot_base = slot_row_base(tile_id)
            for sub in range(tile_factor):
                s1_sub = tile_base + sub * CUBE_S1
                k_part = pto.partition_view(k_view, offsets=[0, s1_sub], sizes=[head_dim, CUBE_S1])
                qk_slot_part = pto.partition_view(
                    qk_slots,
                    offsets=[slot_base, sub * CUBE_S1],
                    sizes=[S0, CUBE_S1],
                )
                pto.tile.load(k_part, k_mat)
                pto.tile.mov(k_mat, k_right)
                pto.tile.matmul(q_left, k_right, qk_acc)
                pto.tile.store(qk_acc, qk_slot_part)
            qk_pipe.push(qk_token, split=0)

        def emit_pv_tile(tile_id):
            tile_base = tile_id * s1_tile
            slot_base = slot_row_base(tile_id)
            _ = p_pipe.pop(split=0, result_type=p_token)
            for sub in range(tile_factor):
                s1_sub = tile_base + sub * CUBE_S1
                v_part = pto.partition_view(v_view, offsets=[s1_sub, 0], sizes=[CUBE_S1, head_dim])
                p_part = pto.partition_view(
                    p_slots,
                    offsets=[slot_base, sub * CUBE_S1],
                    sizes=[S0, CUBE_S1],
                )
                pto.tile.load(p_part, p_recv)
                pto.tile.mov(p_recv, p_left)
                pto.tile.load(v_part, v_mat)
                pto.tile.mov(v_mat, v_right)
                if sub == 0:
                    pto.tile.matmul(p_left, v_right, pv_acc)
                else:
                    pto.tile.matmul_acc(pv_acc, p_left, v_right, pv_acc)
            p_pipe.free(split=0)
            pv_part = pto.partition_view(pv_slots, offsets=[slot_base, 0], sizes=[S0, head_dim])
            pto.tile.store(pv_acc, pv_part)
            pv_pipe.push(pv_token, split=0)

        with pto.for_(qb_start, qb_end, step=1) as qb:
            q_part = pto.partition_view(q_view, offsets=[qb * S0, 0], sizes=[S0, head_dim])
            pto.tile.load(q_part, q_mat)
            pto.tile.mov(q_mat, q_left)

            for kp in range(qk_preload):
                emit_qk_tile(kp)

            with pto.for_(0, steady_tiles, step=1) as tile_id:
                emit_pv_tile(tile_id)
                emit_qk_tile(tile_id + qk_preload)

            for k in range(qk_preload):
                emit_pv_tile(steady_tiles + k)

    @pto.jit(name=vector_symbol, target="a5", entry=False, kernel_kind="vector", mode="auto", backend="emitc")
    def flash_attention_vector_kernel(
        gm_o: pto.ptr(pto.f32, "gm"),
        gm_qk_slots: pto.ptr(pto.f32, "gm"),
        gm_p_slots: pto.ptr(pto.f16, "gm"),
        gm_pv_slots: pto.ptr(pto.f32, "gm"),
        s0: pto.i32,
        s1: pto.i32,
    ):
        s0_index = scalar.index_cast(s0)
        s1_index = scalar.index_cast(s1)
        subblock_idx = scalar.index_cast(pto.get_subblock_idx())
        row_off_sb = subblock_idx * vec_gu_rows
        num_tiles_s1 = s1_index // s1_tile
        steady_tiles = num_tiles_s1 - qk_preload
        qb_start, qb_end = _compute_qb_range(total_q_blocks)

        o_view = pto.make_tensor_view(gm_o, shape=[s0_index, head_dim], strides=[head_dim, 1])

        qk_slot_bytes = 32
        p_slot_bytes = 32
        pv_slot_bytes = 32

        qk_slots = pto.make_tensor_view(gm_qk_slots, shape=[SLOT_NUM * S0, s1_tile], strides=[s1_tile, 1])
        p_slots = pto.make_tensor_view(gm_p_slots, shape=[SLOT_NUM * S0, s1_tile], strides=[s1_tile, 1])
        pv_slots = pto.make_tensor_view(gm_pv_slots, shape=[SLOT_NUM * S0, head_dim], strides=[head_dim, 1])

        qk_consumer_buf = pto.reserve_buffer("fa_qk_c2v_fifo", size=qk_slot_bytes * 8, location="vec")
        pv_consumer_buf = pto.reserve_buffer("fa_pv_c2v_fifo", size=pv_slot_bytes * 8, location="vec")
        p_consumer_buf = pto.import_reserved_buffer("fa_p_v2c_fifo", peer_func=cube_symbol)

        qk_pipe = pto.pipe.c2v(slot_size=qk_slot_bytes, consumer_buf=qk_consumer_buf, id=QK_C2V_PIPE_ID)
        p_pipe = pto.pipe.v2c(slot_size=p_slot_bytes, consumer_buf=p_consumer_buf, id=P_V2C_PIPE_ID)
        pv_pipe = pto.pipe.c2v(slot_size=pv_slot_bytes, consumer_buf=pv_consumer_buf, id=PV_C2V_PIPE_ID)

        qk_pipe.init_simd()
        p_pipe.init_simd()
        pv_pipe.init_simd()

        qk_vec = pto.alloc_tile(shape=[vec_s0, s1_tile], dtype=pto.f32)
        qk_token = pto.alloc_tile(shape=[1, 8], dtype=pto.f32)
        p_fp32 = pto.alloc_tile(shape=[vec_s0, s1_tile], dtype=pto.f32)
        p_fp16 = pto.alloc_tile(shape=[vec_s0, s1_tile], dtype=pto.f16)
        p_token = pto.alloc_tile(shape=[1, 16], dtype=pto.f16)
        pv_vec = [
            pto.alloc_tile(shape=[vec_s0, head_dim], dtype=pto.f32)
            for _ in range(row_slice_count)
        ]
        pv_token = pto.alloc_tile(shape=[1, 8], dtype=pto.f32)
        tmp = pto.alloc_tile(shape=[vec_s0, s1_tile], dtype=pto.f32)

        running_max = [
            pto.alloc_tile(shape=[vec_s0, 8], dtype=pto.f32, valid_shape=[vec_s0, 1])
            for _ in range(row_slice_count)
        ]
        running_sum = [
            pto.alloc_tile(shape=[vec_s0, 8], dtype=pto.f32, valid_shape=[vec_s0, 1])
            for _ in range(row_slice_count)
        ]
        local_max = [
            pto.alloc_tile(shape=[vec_s0, 8], dtype=pto.f32, valid_shape=[vec_s0, 1])
            for _ in range(row_slice_count)
        ]
        local_sum = [
            pto.alloc_tile(shape=[vec_s0, 8], dtype=pto.f32, valid_shape=[vec_s0, 1])
            for _ in range(row_slice_count)
        ]
        exp_max_ring = [
            [
                pto.alloc_tile(shape=[vec_s0, 8], dtype=pto.f32, valid_shape=[vec_s0, 1])
                for _ in range(row_slice_count)
            ]
            for _ in range(qk_preload)
        ]
        o_tile = [
            pto.alloc_tile(shape=[vec_s0, head_dim], dtype=pto.f32)
            for _ in range(row_slice_count)
        ]

        def slot_row_base(tile_id):
            return (tile_id % SLOT_NUM) * S0

        def emit_softmax_tile(tile_id, exp_max_slots, *, is_init):
            _ = qk_pipe.pop(split=0, result_type=qk_token)
            slot_base = slot_row_base(tile_id)
            for row_slice in range(row_slice_count):
                row_off = slot_base + row_off_sb + row_slice * vec_s0
                qk_part = pto.partition_view(
                    qk_slots,
                    offsets=[row_off, 0],
                    sizes=[vec_s0, s1_tile],
                )
                p_part = pto.partition_view(
                    p_slots,
                    offsets=[row_off, 0],
                    sizes=[vec_s0, s1_tile],
                )
                pto.tile.load(qk_part, qk_vec)
                # Row reductions only define the logical [rows, 1] column. Pre-fill
                # the padded lanes so follow-on row-major tile ops do not read junk.
                local_max[row_slice].fill(NEG_INF_F32)
                pto.tile.rowmax(qk_vec, local_max[row_slice], tmp=tmp)
                if is_init:
                    pto.tile.mov(local_max[row_slice], running_max[row_slice])
                    pto.tile.rowexpandsub(qk_vec, running_max[row_slice], p_fp32)
                    pto.tile.muls(p_fp32, scale, p_fp32)
                    pto.tile.exp(p_fp32, p_fp32)
                    pto.tile.rowsum(p_fp32, running_sum[row_slice], tmp=tmp)
                    exp_max_slots[row_slice].fill(1.0)
                else:
                    pto.tile.max(local_max[row_slice], running_max[row_slice], local_max[row_slice])
                    pto.tile.sub(running_max[row_slice], local_max[row_slice], exp_max_slots[row_slice])
                    pto.tile.mov(local_max[row_slice], running_max[row_slice])
                    pto.tile.muls(exp_max_slots[row_slice], scale, exp_max_slots[row_slice])
                    pto.tile.exp(exp_max_slots[row_slice], exp_max_slots[row_slice])
                    pto.tile.rowexpandsub(qk_vec, running_max[row_slice], p_fp32)
                    pto.tile.muls(p_fp32, scale, p_fp32)
                    pto.tile.exp(p_fp32, p_fp32)
                    pto.tile.mul(running_sum[row_slice], exp_max_slots[row_slice], running_sum[row_slice])
                    local_sum[row_slice].fill(0.0)
                    pto.tile.rowsum(p_fp32, local_sum[row_slice], tmp=tmp)
                    pto.tile.add(running_sum[row_slice], local_sum[row_slice], running_sum[row_slice])
                pto.tile.cvt(p_fp32, p_fp16)
                pto.tile.store(p_fp16, p_part)
            p_pipe.push(p_token, split=0)
            qk_pipe.free(split=0)

        def emit_gu_tile(tile_id, exp_max_slots, *, is_init):
            _ = pv_pipe.pop(split=0, result_type=pv_token)
            slot_base = slot_row_base(tile_id)
            for row_slice in range(row_slice_count):
                row_off = slot_base + row_off_sb + row_slice * vec_s0
                pv_part = pto.partition_view(
                    pv_slots,
                    offsets=[row_off, 0],
                    sizes=[vec_s0, head_dim],
                )
                pto.tile.load(pv_part, pv_vec[row_slice])
                if is_init:
                    pto.tile.mov(pv_vec[row_slice], o_tile[row_slice])
                else:
                    pto.tile.rowexpandmul(o_tile[row_slice], exp_max_slots[row_slice], o_tile[row_slice])
                    pto.tile.add(o_tile[row_slice], pv_vec[row_slice], o_tile[row_slice])
            pv_pipe.free(split=0)

        def emit_softmax_dispatch(tile_id):
            mod = tile_id % qk_preload
            for ring in range(qk_preload):
                with pto.if_(mod == ring) as branch:
                    with branch.then_:
                        emit_softmax_tile(tile_id, exp_max_ring[ring], is_init=False)

        def emit_gu_dispatch(tile_id):
            mod = tile_id % qk_preload
            for ring in range(qk_preload):
                with pto.if_(mod == ring) as branch:
                    with branch.then_:
                        emit_gu_tile(tile_id, exp_max_ring[ring], is_init=False)

        with pto.for_(qb_start, qb_end, step=1) as qb:
            for row_slice in range(row_slice_count):
                running_max[row_slice].fill(NEG_INF_F32)
                running_sum[row_slice].fill(0.0)
                o_tile[row_slice].fill(0.0)

            for kp in range(qk_preload):
                emit_softmax_tile(kp, exp_max_ring[kp], is_init=(kp == 0))

            with pto.if_(steady_tiles > 0) as branch:
                with branch.then_:
                    emit_gu_tile(0, exp_max_ring[0], is_init=True)
                    emit_softmax_tile(qk_preload, exp_max_ring[0], is_init=False)
                    with pto.for_(1, steady_tiles, step=1) as tile_id:
                        emit_gu_dispatch(tile_id)
                        emit_softmax_dispatch(tile_id + qk_preload)

            for k in range(qk_preload):
                tile_id = steady_tiles + k
                with pto.if_(tile_id == 0) as init_branch:
                    with init_branch.then_:
                        emit_gu_tile(tile_id, exp_max_ring[0], is_init=True)
                    with init_branch.else_:
                        emit_gu_dispatch(tile_id)

            for row_slice in range(row_slice_count):
                row_off = row_off_sb + row_slice * vec_s0
                pto.tile.rowexpanddiv(o_tile[row_slice], running_sum[row_slice], o_tile[row_slice])
                o_part = pto.partition_view(
                    o_view,
                    offsets=[qb * S0 + row_off, 0],
                    sizes=[vec_s0, head_dim],
                )
                pto.tile.store(o_tile[row_slice], o_part)

    @pto.jit(name=entry_symbol, target="a5", mode="explicit", backend="emitc")
    def flash_attention_entry(
        gm_q: pto.ptr(pto.f16, "gm"),
        gm_k: pto.ptr(pto.f16, "gm"),
        gm_v: pto.ptr(pto.f16, "gm"),
        gm_o: pto.ptr(pto.f32, "gm"),
        gm_qk_slots: pto.ptr(pto.f32, "gm"),
        gm_p_slots: pto.ptr(pto.f16, "gm"),
        gm_pv_slots: pto.ptr(pto.f32, "gm"),
        s0: pto.i32,
        s1: pto.i32,
    ):
        flash_attention_cube_kernel(gm_q, gm_k, gm_v, gm_qk_slots, gm_p_slots, gm_pv_slots, s0, s1)
        flash_attention_vector_kernel(gm_o, gm_qk_slots, gm_p_slots, gm_pv_slots, s0, s1)

    return flash_attention_entry


def emit_flash_attention_mlir(
    *,
    head_dim: int = HEAD,
    s1_tile: int = DEFAULT_S1_TILE,
    qk_preload: int = DEFAULT_QK_PRELOAD,
    causal: bool = False,
    q_rows: int = DEFAULT_Q_ROWS,
) -> str:
    entry_kernel = _build_flash_attention_entry(
        head_dim=head_dim,
        s1_tile=s1_tile,
        qk_preload=qk_preload,
        causal=causal,
        q_rows=q_rows,
    )
    return entry_kernel.compile().mlir_text()


def compile_flash_attention_kernel(
    *,
    head_dim: int = HEAD,
    s1_tile: int = DEFAULT_S1_TILE,
    qk_preload: int = DEFAULT_QK_PRELOAD,
    causal: bool = False,
    q_rows: int = DEFAULT_Q_ROWS,
):
    entry_kernel = _build_flash_attention_entry(
        head_dim=head_dim,
        s1_tile=s1_tile,
        qk_preload=qk_preload,
        causal=causal,
        q_rows=q_rows,
    )
    return entry_kernel.compile()


def run_demo(
    *,
    head_dim: int = HEAD,
    s1_tile: int = DEFAULT_S1_TILE,
    qk_preload: int = DEFAULT_QK_PRELOAD,
    causal: bool = False,
    q_rows: int = DEFAULT_Q_ROWS,
    s1: int = DEFAULT_S1_TILE * DEFAULT_QK_PRELOAD,
    seed: int = 20260601,
) -> None:
    _validate_specialization(
        head_dim=head_dim,
        s1_tile=s1_tile,
        qk_preload=qk_preload,
        causal=causal,
        q_rows=q_rows,
    )
    _validate_runtime_problem(
        q_rows=q_rows,
        s1=s1,
        s1_tile=s1_tile,
        qk_preload=qk_preload,
    )

    torch = _init_runtime()
    rng = np.random.RandomState(seed)

    host_q = rng.randn(q_rows, head_dim).astype(np.float16)
    host_k_tokens = rng.randn(s1, head_dim).astype(np.float16)
    host_v = rng.randn(s1, head_dim).astype(np.float16)
    host_k = host_k_tokens
    host_ref = _reference_flash_attention(host_q, host_k_tokens, host_v)

    q_t = torch.from_numpy(host_q).to(_DEVICE)
    k_t = torch.from_numpy(host_k).to(_DEVICE)
    v_t = torch.from_numpy(host_v).to(_DEVICE)
    o_t = torch.empty((q_rows, head_dim), dtype=torch.float32, device=_DEVICE)
    qk_slots_t = torch.empty((SLOT_NUM * S0, s1_tile), dtype=torch.float32, device=_DEVICE)
    p_slots_t = torch.empty((SLOT_NUM * S0, s1_tile), dtype=torch.float16, device=_DEVICE)
    pv_slots_t = torch.empty((SLOT_NUM * S0, head_dim), dtype=torch.float32, device=_DEVICE)
    stream = _current_stream(torch)

    t0 = time.perf_counter()
    compiled = compile_flash_attention_kernel(
        head_dim=head_dim,
        s1_tile=s1_tile,
        qk_preload=qk_preload,
        causal=causal,
        q_rows=q_rows,
    )
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled[1, stream](
        q_t.data_ptr(),
        k_t.data_ptr(),
        v_t.data_ptr(),
        o_t.data_ptr(),
        qk_slots_t.data_ptr(),
        p_slots_t.data_ptr(),
        pv_slots_t.data_ptr(),
        q_rows,
        s1,
    )
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    host_out = o_t.cpu().numpy()
    np.testing.assert_allclose(host_out, host_ref, rtol=6e-2, atol=6e-2)
    print(
        f"PASS hw-native-fa-cv-split q_rows={q_rows} s1={s1} head={head_dim} "
        f"s1_tile={s1_tile} qk_preload={qk_preload} "
        f"compile={compile_s:.3f}s launch={launch_s:.3f}s"
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Emit or launch the CV-split local-pipe PTODSL hw-native FlashAttention port."
    )
    parser.add_argument(
        "--emit-mlir",
        action="store_true",
        help="print compiled MLIR and exit",
    )
    parser.add_argument("--head-dim", type=int, default=HEAD)
    parser.add_argument("--s1-tile", type=int, default=DEFAULT_S1_TILE)
    parser.add_argument("--qk-preload", type=int, default=DEFAULT_QK_PRELOAD)
    parser.add_argument("--q-rows", type=int, default=DEFAULT_Q_ROWS)
    parser.add_argument(
        "--s1",
        type=int,
        default=DEFAULT_S1_TILE * DEFAULT_QK_PRELOAD,
        help="runtime S1 length; must be a multiple of --s1-tile and provide at least --qk-preload logical tiles",
    )
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("-o", "--output", default="-", help="output MLIR path, or '-' for stdout")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.emit_mlir:
        mlir_text = emit_flash_attention_mlir(
            head_dim=args.head_dim,
            s1_tile=args.s1_tile,
            qk_preload=args.qk_preload,
            causal=args.causal,
            q_rows=args.q_rows,
        )
        if args.output == "-":
            print(mlir_text)
            return 0
        Path(args.output).write_text(mlir_text, encoding="utf-8")
        return 0

    run_demo(
        head_dim=args.head_dim,
        s1_tile=args.s1_tile,
        qk_preload=args.qk_preload,
        causal=args.causal,
        q_rows=args.q_rows,
        s1=args.s1,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
