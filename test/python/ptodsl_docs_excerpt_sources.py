#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from textwrap import dedent


class _NormalizedSnippet(str):
    pass


def _snippet(text: str) -> str:
    return dedent(text).strip("\n")


def _snippet_at(indent: int, text: str) -> _NormalizedSnippet:
    prefix = " " * indent
    return _NormalizedSnippet(
        "\n".join(
            f"{prefix}{line}" if line else line
            for line in _snippet(text).splitlines()
        )
    )


def _group(prefix: str, entries: dict[str, str]) -> dict[str, str]:
    return {
        f"{prefix}.{name}": str(body) if isinstance(body, _NormalizedSnippet) else _snippet(body)
        for name, body in entries.items()
    }


DATA_MOVEMENT = _group(
    "data_movement",
    {
        "tload.basic": """
            a_part = pto.partition_view(a_view, offsets=[offset], sizes=[BLOCK])
            a_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32)
            pto.tload(a_part, a_tile)
        """,
        "tstore.basic": """
            pto.tstore(o_tile, o_part)
        """,
        "mte_gm_ub.contiguous": """
            pto.mte_gm_ub(gm_ptr, ub_ptr, 0, 128,
                          nburst=(32, 128, 128))
            # 32 rows, 128 bytes per row, contiguous in both GM and UB
        """,
        "mte_gm_ub.strided": """
            pto.mte_gm_ub(gm_ptr, ub_ptr, 0, 256,
                          nburst=(64, 1024, 256))
            # 64 rows of 256 bytes each.
            # GM: each row is 1024 bytes apart (full matrix row stride).
            # UB: rows are packed contiguously (256-byte stride).
        """,
        "mte_gm_ub.pad": """
            pto.mte_gm_ub(gm_ptr, ub_ptr, 0, 200,
                          nburst=(64, 200, 256),
                          pad=(0.0, 0, 0))
            # 64 rows, 200 valid bytes per row, 256-byte UB stride.
            # Gap (56 bytes) between len_burst and dst_stride is zero-padded.
        """,
        "mte_gm_ub.loops": """
            pto.mte_gm_ub(gm_ptr, ub_ptr, 0, 256,
                          nburst=(8, 256, 256),
                          loops=[(4, 2048, 2048)])
            # Innermost: 8 rows × 256B (one tile).
            # Outer loop: 4 iterations, each advancing 2048 bytes in both GM and UB.
        """,
        "mte_ub_gm.basic": """
            pto.mte_ub_gm(ub_ptr, gm_ptr, 128,
                          nburst=(32, 128, 128))
        """,
        "mte_ub_gm.strided": """
            pto.mte_ub_gm(ub_ptr, gm_ptr, 256,
                          nburst=(64, 256, 1024))
            # UB: contiguous rows (256-byte stride).
            # GM: rows spaced at 1024-byte intervals (full matrix width).
        """,
        "mte_ub_ub.basic": """
            pto.mte_ub_ub(ub_src, ub_dst, 8,
                          nburst=(16, 0, 4))
            # 16 bursts, each copying 8×32=256 bytes.
            # Source: contiguous (src_gap=0).
            # Destination: 4×32=128-byte gap between bursts.
        """,
        "vlds.tile_syntax": """
            vec = pto.vlds(tile[row, col:])       # load from row, starting at column col
            vec = pto.vlds(tile[start:])          # 1D tile, starting at element start
        """,
        "vldas_vldus": """
            align = pto.vldas(tile[row, col:])
            vec, align, base = pto.vldus(tile[row, col:], align)
        """,
        "vstu_stream": """
            align, base = pto.vstu(align0, base0, vec0, ub_ptr, mode)
            align, base = pto.vstu(align, base, vec1, ub_ptr, mode)
            pto.vsta(align, ub_ptr, flush_offset)
        """,
        "cube.qk_matmul": """
            @pto.cube
            def qk_matmul(q_tile, k_tile, q_l0a, k_l0b, s_acc, s_tile):
                m = q_tile.valid_shape[0]
                k = q_tile.valid_shape[1]
                n = k_tile.valid_shape[0]

                pto.mte_l1_l0a(q_tile.as_ptr(), q_l0a.as_ptr(), m, k)          # UB tile → L0A
                pto.mte_l1_l0b(k_tile.as_ptr(), k_l0b.as_ptr(), k, n, transpose=True)  # UB tile → L0B
                pto.mad(q_l0a.as_ptr(), k_l0b.as_ptr(), s_acc.as_ptr(), m, n, k)        # L0A × L0B → L0C
                pto.mte_l0c_ub(s_acc.as_ptr(), s_tile.as_ptr(), m, n, n, n, 0)          # L0C → UB tile
        """,
        "ukernel_dma_pattern": """
            @pto.ukernel
            def process_block(k_part, v_part, k_tile, v_tile, o_tile, o_part,
                              rows: pto.i32, cols: pto.i32):
                # Stage K and V blocks from GM to UB
                pto.mte_gm_ub(k_part.as_ptr(), k_tile.as_ptr(), 0,
                              cols * pto.bytewidth(pto.f16),
                              nburst=(rows, cols * pto.bytewidth(pto.f16),
                                      cols * pto.bytewidth(pto.f16)))
                pto.mte_gm_ub(v_part.as_ptr(), v_tile.as_ptr(), 0,
                              cols * pto.bytewidth(pto.f16),
                              nburst=(rows, cols * pto.bytewidth(pto.f16),
                                      cols * pto.bytewidth(pto.f16)))
                pto.pipe_barrier(pto.Pipe.ALL)

                # ... compute on tiles ...

                pto.pipe_barrier(pto.Pipe.ALL)
                pto.mte_ub_gm(o_tile.as_ptr(), o_part.as_ptr(),
                              cols * pto.bytewidth(pto.f32),
                              nburst=(rows, cols * pto.bytewidth(pto.f32),
                                      cols * pto.bytewidth(pto.f32)))
        """,
    },
)


FLASH = _group(
    "flash",
    {
        "wrapper": """
            def flash_attention(Q, K, V, *, O=None, causal=False,
                                block_q=128, block_kv=128, stream=None):
                if O is None:
                    O = pto.empty_like(Q)

                batch, seq_q, heads, dim = Q.shape
                _, seq_k, _, _ = K.shape

                compiled = flash_attention_kernel.compile(
                    BLOCK_Q=block_q, BLOCK_KV=block_kv, CAUSAL=causal,
                )
                compiled[batch * heads, stream](Q, K, V, O)
                return O
        """,
        "kernel.signature": """
            @pto.jit(target="a5")
            def flash_attention_kernel(
                Q, K, V, O, *,
                BLOCK_Q: pto.constexpr = 128,
                BLOCK_KV: pto.constexpr = 128,
                CAUSAL: pto.constexpr = False,
                NUM_STAGES: pto.constexpr = 2,
            ):
        """,
        "tensor_views": """
            q_view = pto.make_tensor_view(Q, shape=[batch, seq_q, heads, dim],
                                          strides=Q.strides)
            k_view = pto.make_tensor_view(K, shape=[batch, seq_k, heads, dim],
                                          strides=K.strides)
            v_view = pto.make_tensor_view(V, shape=[batch, seq_k, heads, dim],
                                          strides=V.strides)
            o_view = pto.make_tensor_view(O, shape=[batch, seq_q, heads, dim],
                                          strides=O.strides)
        """,
        "spmd_contract": """
            block_idx = pto.get_block_idx()
            block_num = pto.get_block_num()
            subblock_idx = pto.get_subblock_idx()
            subblock_num = pto.get_subblock_num()

            batch_idx = block_idx // heads
            head_idx = block_idx % heads
        """,
        "head_partitioning": """
            q_head = pto.partition_view(
                q_view,
                offsets=[batch_idx, 0, head_idx, 0],
                sizes=[1, seq_q, 1, dim],
            )
            k_head = pto.partition_view(
                k_view,
                offsets=[batch_idx, 0, head_idx, 0],
                sizes=[1, seq_k, 1, dim],
            )
            v_head = pto.partition_view(
                v_view,
                offsets=[batch_idx, 0, head_idx, 0],
                sizes=[1, seq_k, 1, dim],
            )
            o_head = pto.partition_view(
                o_view,
                offsets=[batch_idx, 0, head_idx, 0],
                sizes=[1, seq_q, 1, dim],
            )
        """,
        "metadata_buffer": """
            meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 3])
            meta_ptr = meta_tile.as_ptr()
        """,
        "tile_alloc.ub": """
            q_tile  = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[full_br, dim])
            k_tile  = pto.alloc_tile(shape=[Bc, D], dtype=pto.f32, valid_shape=[full_bc, dim])
            v_tile  = pto.alloc_tile(shape=[Bc, D], dtype=pto.f32, valid_shape=[full_bc, dim])

            o_prev_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[full_br, dim])
            o_next_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[full_br, dim])
            m_prev_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
            m_next_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
            l_prev_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
            l_next_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")

            s_tile   = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, valid_shape=[full_br, full_bc])
            p_tile   = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, valid_shape=[full_br, full_bc])
            pv_tile  = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[full_br, dim])
            alpha_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
            beta_tile  = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
        """,
        "tile_alloc.cube": """
            q_l0a  = pto.alloc_tile(shape=[Br, D], dtype=pto.f16,
                                    memory_space=pto.MemorySpace.LEFT, valid_shape=[full_br, dim])
            p_l0a  = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f16,
                                    memory_space=pto.MemorySpace.LEFT, valid_shape=[full_br, full_bc])
            rhs_l0b = pto.alloc_tile(shape=[Bc, D], dtype=pto.f16,
                                     memory_space=pto.MemorySpace.RIGHT, valid_shape=[full_bc, dim])
            qk_acc_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32,
                                         memory_space=pto.MemorySpace.ACC, valid_shape=[full_br, full_bc])
            pv_acc_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32,
                                         memory_space=pto.MemorySpace.ACC, valid_shape=[full_br, dim])
        """,
        "outer_q_inner_kv_loop": """
            with pto.for_(0, q_blocks, step=1) as qi:
                q_part = pto.partition_view(q_head, offsets=[0, qi * Br, 0, 0],
                                            sizes=[1, Br, 1, dim])
                o_part = pto.partition_view(o_head, offsets=[0, qi * Br, 0, 0],
                                            sizes=[1, Br, 1, dim])

                pto.tload(q_part, q_tile)

                m_prev_tile.fill(float("-inf"))
                l_prev_tile.fill(0.0)
                o_prev_tile.fill(0.0)

                kv_loop = pto.for_(0, kv_blocks, step=1).carry(
                    m=m_prev_tile, l=l_prev_tile, o=o_prev_tile,
                )
                with kv_loop:
                    kj = kv_loop.iv
                    m_cur = kv_loop.m
                    l_cur = kv_loop.l
                    o_cur = kv_loop.o
                    k_part = pto.partition_view(k_head,
                                offsets=[0, kj * Bc, 0, 0], sizes=[1, Bc, 1, dim])
                    v_part = pto.partition_view(v_head,
                                offsets=[0, kj * Bc, 0, 0], sizes=[1, Bc, 1, dim])

                    kv_block_process(
                        q_tile, k_part, v_part, k_tile, v_tile,
                        o_cur, o_next_tile,
                        m_cur, l_cur, m_next_tile, l_next_tile,
                        s_tile, p_tile, pv_tile,
                        alpha_tile, beta_tile,
                        q_l0a, p_l0a, rhs_l0b,
                        qk_acc_tile, pv_acc_tile,
                        meta_ptr,
                    )

                    kv_loop.update(m=m_next_tile, l=l_next_tile, o=o_next_tile)

                o_final_tile = kv_loop.final("o")
                pto.tstore(o_final_tile, o_part)
        """,
        "ukernel.signature": """
            @pto.ukernel
            def kv_block_process(
                q_tile, k_part, v_part, k_tile, v_tile,
                o_prev_tile, o_next_tile,
                m_prev_tile, l_prev_tile, m_next_tile, l_next_tile,
                s_tile, p_tile, pv_tile,
                alpha_tile, beta_tile,
                q_l0a, p_l0a, rhs_l0b,
                qk_acc_tile, pv_acc_tile,
                meta_ptr,
            ):
        """,
        "ukernel.stage_kv": """
            pto.mte_load(k_part, k_tile)
            pto.mte_load(v_part, v_tile)
            pto.pipe_barrier(pto.Pipe.ALL)
        """,
        "ukernel.materialize_bounds": """
            materialize_tile_bounds(meta_ptr,
                q_tile.valid_shape[0],
                k_tile.valid_shape[0])
            row_start = scalar.load(meta_ptr + 0)
            row_stop  = scalar.load(meta_ptr + 1)
            valid_cols = scalar.load(meta_ptr + 2)
        """,
        "ukernel.qk_dispatch": """
            qk_matmul(q_tile, k_tile, q_l0a, rhs_l0b, qk_acc_tile, s_tile)
            pto.pipe_barrier(pto.Pipe.ALL)
        """,
        "ukernel.softmax_dispatch": """
            online_softmax_rows(
                s_tile, p_tile,
                m_prev_tile, l_prev_tile,
                m_next_tile, l_next_tile,
                alpha_tile, beta_tile,
                row_start, row_stop, valid_cols,
            )
            pto.pipe_barrier(pto.Pipe.ALL)
        """,
        "ukernel.pv_dispatch": """
            pv_matmul(p_tile, v_tile, p_l0a, rhs_l0b, pv_acc_tile, pv_tile)
            pto.pipe_barrier(pto.Pipe.ALL)
        """,
        "ukernel.blend_dispatch": """
            blend_output_rows(
                o_prev_tile, pv_tile, alpha_tile, beta_tile,
                o_next_tile, row_start, row_stop,
                v_tile.valid_shape[1],
            )
            pto.pipe_barrier(pto.Pipe.ALL)
        """,
        "cube.qk_matmul": """
            @pto.cube
            def qk_matmul(q_tile, k_tile, q_l0a, k_l0b, s_acc, s_tile):
                m = q_tile.valid_shape[0]
                k = q_tile.valid_shape[1]
                n = k_tile.valid_shape[0]

                pto.mte_l1_l0a(q_tile.as_ptr(), q_l0a.as_ptr(), m, k)
                pto.mte_l1_l0b(k_tile.as_ptr(), k_l0b.as_ptr(), k, n, transpose=True)
                pto.mad(q_l0a.as_ptr(), k_l0b.as_ptr(), s_acc.as_ptr(), m, n, k)
                pto.mte_l0c_ub(s_acc.as_ptr(), s_tile.as_ptr(), m, n, n, n, 0)
        """,
        "cube.pv_matmul": """
            @pto.cube
            def pv_matmul(p_tile, v_tile, p_l0a, v_l0b, pv_acc, pv_tile):
                m = p_tile.valid_shape[0]
                k = p_tile.valid_shape[1]
                n = v_tile.valid_shape[1]

                pto.mte_l1_l0a(p_tile.as_ptr(), p_l0a.as_ptr(), m, k)
                pto.mte_l1_l0b(v_tile.as_ptr(), v_l0b.as_ptr(), k, n)
                pto.mad(p_l0a.as_ptr(), v_l0b.as_ptr(), pv_acc.as_ptr(), m, n, k)
                pto.mte_l0c_ub(pv_acc.as_ptr(), pv_tile.as_ptr(), m, n, n, n, 0)
        """,
        "simd.signature": """
            @pto.simd
            def online_softmax_rows(
                s_tile, p_tile,
                m_prev_tile, l_prev_tile,
                m_next_tile, l_next_tile,
                alpha_tile, beta_tile,
                row_start, row_stop, valid_cols,
            ):
        """,
        "simd.row_loop": """
            with pto.for_(row_start, row_stop, step=1) as row:
                col_mask = pto.make_mask(pto.f32, valid_cols)

                s_row   = pto.vlds(s_tile[row, 0:])
                m_prev  = scalar.load(m_prev_tile[row, 0])
                l_prev  = scalar.load(l_prev_tile[row, 0])
        """,
        "simd.softmax_math": _snippet_at(
            4,
            """
            row_max   = pto.vcgmax(s_row, col_mask)
            m_next    = scalar.max(m_prev, row_max)

            s_shifted = pto.vsubs(s_row, m_next, col_mask)
            p_row     = pto.vexp(s_shifted, col_mask)

            row_sum   = pto.vcgadd(p_row, col_mask)
            l_scaled  = l_prev * scalar.exp(m_prev - m_next)
            l_next    = l_scaled + row_sum

            alpha = l_scaled / l_next
            beta  = 1.0 / l_next
            """,
        ),
        "simd.store_results": _snippet_at(
            4,
            """
            pto.vsts(p_row, p_tile[row, 0:], col_mask)
            scalar.store(m_next, m_next_tile[row, 0])
            scalar.store(l_next, l_next_tile[row, 0])
            scalar.store(alpha, alpha_tile[row, 0])
            scalar.store(beta, beta_tile[row, 0])
            """,
        ),
        "simt.materialize_bounds": """
            @pto.simt
            def materialize_tile_bounds(meta_ptr, valid_rows, valid_cols):
                scalar.store(0, meta_ptr + 0)
                scalar.store(valid_rows, meta_ptr + 1)
                scalar.store(valid_cols, meta_ptr + 2)
        """,
        "simt.blend_output": """
            @pto.simt
            def blend_output_rows(o_prev_tile, pv_tile, alpha_tile, beta_tile,
                                  o_next_tile, row_start, row_stop, valid_dim):
                with pto.for_(row_start, row_stop, step=1) as row:
                    alpha = scalar.load(alpha_tile[row, 0])
                    beta  = scalar.load(beta_tile[row, 0])

                    with pto.for_(0, valid_dim, step=1) as col:
                        o_prev = scalar.load(o_prev_tile[row, col])
                        pv_val = scalar.load(pv_tile[row, col])
                        o_next = alpha * o_prev + beta * pv_val
                        scalar.store(o_next, o_next_tile[row, col])
        """,
    },
)


ENTRY_POINTS = _group(
    "entry_points",
    {
        "jit.signature": """
            @pto.jit(target="a5")
            def kernel_name(
                tensor_arg_1,           # Python-native tensor (positional)
                tensor_arg_2,           # Python-native tensor (positional)
                ...,
                *,
                CONST_A: pto.constexpr = default,  # compile-time constant (keyword-only)
                CONST_B: pto.constexpr = default,  # compile-time constant (keyword-only)
            ):
        """,
        "jit.compile_and_launch": """
            # Compile (traces the body, lowers through PTOAS, caches the result)
            compiled = kernel_name.compile(CONST_A=128, CONST_B=64)

            # Launch on NPU
            compiled[grid, stream](tensor_1, tensor_2, ...)
        """,
        "jit.typical_body": """
            @pto.jit(target="a5")
            def my_kernel(A, B, O, *, BLOCK: pto.constexpr):
                N = A.shape[0]
                a_view = pto.make_tensor_view(A, shape=[N], strides=A.strides)
                b_view = pto.make_tensor_view(B, shape=[N], strides=B.strides)
                o_view = pto.make_tensor_view(O, shape=[N], strides=O.strides)

                a_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32)
                b_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32)
                o_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32)

                num_blocks = (N + BLOCK - 1) // BLOCK
                with pto.for_(0, num_blocks, step=1) as i:
                    offset = i * BLOCK
                    a_part = pto.partition_view(a_view, offsets=[offset], sizes=[BLOCK])
                    b_part = pto.partition_view(b_view, offsets=[offset], sizes=[BLOCK])
                    o_part = pto.partition_view(o_view, offsets=[offset], sizes=[BLOCK])

                    pto.tload(a_part, a_tile)
                    pto.tload(b_part, b_tile)
                    pto.tadd(a_tile, b_tile, o_tile)
                    pto.tstore(o_tile, o_part)
        """,
        "jit.direct_l3_call": """
            @pto.cube
            def my_matmul(a_tile, b_tile, l0a, l0b, acc, o_tile):
                m = a_tile.valid_shape[0]
                k = a_tile.valid_shape[1]
                n = b_tile.valid_shape[0]
                pto.mte_l1_l0a(a_tile.as_ptr(), l0a.as_ptr(), m, k)
                pto.mte_l1_l0b(b_tile.as_ptr(), l0b.as_ptr(), k, n, transpose=True)
                pto.mad(l0a.as_ptr(), l0b.as_ptr(), acc.as_ptr(), m, n, k)
                pto.mte_l0c_ub(acc.as_ptr(), o_tile.as_ptr(), m, n, n, n, 0)

            @pto.jit(target="a5")
            def my_kernel(A, B, O, *, BLOCK: pto.constexpr):
                N = A.shape[0]
                a_view = pto.make_tensor_view(A, shape=[N], strides=A.strides)
                b_view = pto.make_tensor_view(B, shape=[N], strides=B.strides)
                o_view = pto.make_tensor_view(O, shape=[N], strides=O.strides)

                a_tile = pto.alloc_tile(shape=[BLOCK, BLOCK], dtype=pto.f32)
                b_tile = pto.alloc_tile(shape=[BLOCK, BLOCK], dtype=pto.f32)
                o_tile = pto.alloc_tile(shape=[BLOCK, BLOCK], dtype=pto.f32)
                l0a = pto.alloc_tile(shape=[BLOCK, BLOCK], dtype=pto.f32, memory_space=pto.MemorySpace.LEFT)
                l0b = pto.alloc_tile(shape=[BLOCK, BLOCK], dtype=pto.f32, memory_space=pto.MemorySpace.RIGHT)
                acc = pto.alloc_tile(shape=[BLOCK, BLOCK], dtype=pto.f32, memory_space=pto.MemorySpace.ACC)

                num_blocks = (N + BLOCK - 1) // BLOCK
                with pto.for_(0, num_blocks, step=1) as i:
                    offset = i * BLOCK
                    a_part = pto.partition_view(a_view, offsets=[offset, 0], sizes=[BLOCK, BLOCK])
                    b_part = pto.partition_view(b_view, offsets=[offset, 0], sizes=[BLOCK, BLOCK])
                    o_part = pto.partition_view(o_view, offsets=[offset, 0], sizes=[BLOCK, BLOCK])

                    # Tile Ops stage data from GM to UB (replaces mte_load at L1)
                    pto.tload(a_part, a_tile)
                    pto.tload(b_part, b_tile)

                    # Direct L3 call — PTOAS handles sync between tload and compute
                    my_matmul(a_tile, b_tile, l0a, l0b, acc, o_tile)

                    pto.tstore(o_tile, o_part)
        """,
        "ukernel.signature": """
            @pto.ukernel
            def my_ukernel(
                part: pto.PartitionTensorView,   # GM partition descriptors
                tile: pto.Tile,                  # UB tile buffers
                scratch: pto.Tile,               # cube-local scratch (LEFT, RIGHT, ...)
                ptr: pto.ptr(dtype, space),      # typed UB pointers
                scalar: pto.i32,                 # PTO scalar values
            ):
        """,
        "ukernel.typical_body": """
            @pto.ukernel
            def process_block(k_part, v_part, k_tile, v_tile,
                              s_tile, o_tile, rows: pto.i32, cols: pto.i32):
                # Stage current block from GM to UB
                pto.mte_load(k_part, k_tile)
                pto.mte_load(v_part, v_tile)
                pto.pipe_barrier(pto.Pipe.ALL)

                # Dispatch sub-kernels
                qk_matmul(q_tile, k_tile, s_tile)
                pto.pipe_barrier(pto.Pipe.ALL)

                online_softmax(s_tile, o_tile, rows, cols)
                pto.pipe_barrier(pto.Pipe.ALL)

                # Write result back
                pto.mte_store(o_tile, o_part)
        """,
        "cube.signature": """
            @pto.cube
            def my_cube_kernel(
                input_tile: pto.Tile,            # UB tile (source data)
                output_tile: pto.Tile,           # UB tile (destination)
                left_scratch: pto.Tile,          # LEFT buffer (cube-local)
                right_scratch: pto.Tile,         # RIGHT buffer (cube-local)
                acc_scratch: pto.Tile,           # ACC buffer (cube-local)
            ):
        """,
        "cube.typical_body": """
            @pto.cube
            def qk_matmul(
                q_tile: pto.Tile,
                k_tile: pto.Tile,
                q_l0a: pto.Tile,
                k_l0b: pto.Tile,
                s_acc: pto.Tile,
                s_tile: pto.Tile,
            ):
                m = q_tile.valid_shape[0]
                k = q_tile.valid_shape[1]
                n = k_tile.valid_shape[0]

                pto.mte_l1_l0a(q_tile.as_ptr(), q_l0a.as_ptr(), m, k)
                pto.mte_l1_l0b(k_tile.as_ptr(), k_l0b.as_ptr(), k, n, transpose=True)
                pto.mad(q_l0a.as_ptr(), k_l0b.as_ptr(), s_acc.as_ptr(), m, n, k)
                pto.mte_l0c_ub(s_acc.as_ptr(), s_tile.as_ptr(), m, n, n, n, 0)
        """,
        "simd.signature": """
            @pto.simd
            def my_simd_kernel(
                input_tile: pto.Tile,            # UB tile
                output_tile: pto.Tile,           # UB tile
                rows: pto.i32,                   # PTO scalar
                cols: pto.i32,                   # PTO scalar
            ):
        """,
        "simd.typical_body": """
            @pto.simd
            def add_rows(a_tile: pto.Tile, b_tile: pto.Tile, o_tile: pto.Tile,
                         rows: pto.i32, cols: pto.i32):
                VEC = pto.elements_per_vreg(pto.f32)
                with pto.for_(0, rows, step=1) as r:
                    col_loop = pto.for_(0, cols, step=VEC).carry(remained=cols)
                    with col_loop:
                        c = col_loop.iv
                        remained = col_loop.remained
                        mask, remained = pto.make_mask(pto.f32, remained)
                        a_vec = pto.vlds(a_tile[r, c:])
                        b_vec = pto.vlds(b_tile[r, c:])
                        o_vec = pto.vadd(a_vec, b_vec, mask)
                        pto.vsts(o_vec, o_tile[r, c:], mask)
                        col_loop.update(remained=remained)
        """,
        "simt.signature": """
            @pto.simt
            def my_simt_kernel(
                tile: pto.Tile,                  # UB tile
                ptr: pto.ptr(dtype, space),      # typed UB pointer
                scalar: pto.i32,                 # PTO scalar
            ):
        """,
        "simt.typical_body": """
            @pto.simt
            def blend_output_rows(
                o_prev_tile: pto.Tile, pv_tile: pto.Tile,
                alpha_tile: pto.Tile, beta_tile: pto.Tile,
                o_next_tile: pto.Tile,
                row_start: pto.i32, row_stop: pto.i32, valid_dim: pto.i32,
            ):
                with pto.for_(row_start, row_stop, step=1) as row:
                    alpha = scalar.load(alpha_tile[row, 0])
                    beta = scalar.load(beta_tile[row, 0])
                    with pto.for_(0, valid_dim, step=1) as col:
                        o_prev = scalar.load(o_prev_tile[row, col])
                        pv_val = scalar.load(pv_tile[row, col])
                        o_next = alpha * o_prev + beta * pv_val
                        scalar.store(o_next, o_next_tile[row, col])
        """,
        "context.simd": """
            with pto.simd():
                # Direct L3 instructions — vreg ops, scalar loads/stores
                a_vec = pto.vlds(a_tile[r, c:])
                b_vec = pto.vlds(b_tile[r, c:])
                o_vec = pto.vadd(a_vec, b_vec, mask)
                pto.vsts(o_vec, o_tile[r, c:], mask)
        """,
        "context.simt": """
            with pto.simt():
                alpha = scalar.load(alpha_tile[row, 0])
                beta = scalar.load(beta_tile[row, 0])
                o_next = alpha * o_prev + beta * pv_val
                scalar.store(o_next, o_next_tile[row, col])
        """,
        "context.cube": """
            with pto.cube():
                pto.mte_l1_l0a(q_tile.as_ptr(), q_l0a.as_ptr(), m, k)
                pto.mte_l1_l0b(k_tile.as_ptr(), k_l0b.as_ptr(), k, n, transpose=True)
                pto.mad(q_l0a.as_ptr(), k_l0b.as_ptr(), s_acc.as_ptr(), m, n, k)
                pto.mte_l0c_ub(s_acc.as_ptr(), s_tile.as_ptr(), m, n, n, n, 0)
        """,
        "constexpr.signature": """
            @pto.jit(target="a5")
            def kernel(A, *, BLOCK: pto.constexpr = 128, DTYPE: pto.constexpr = pto.f32):
                ...
        """,
    },
)


QUICK_START = _group(
    "quick_start",
    {
        "tile_copy.entry_point": """
            @pto.jit(target="a5")
            def tile_copy(A, O, *, BLOCK: pto.constexpr = 128):
        """,
        "tile_copy.make_tensor_view": """
            a_view = pto.make_tensor_view(A, shape=A.shape, strides=A.strides)
        """,
        "tile_copy.alloc_tile": """
            a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
        """,
        "tile_copy.partition_view": """
            a_part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
        """,
        "tile_copy.tile_io": """
            pto.tload(a_part, a_tile)   # GM → UB
            pto.tstore(o_tile, o_part)  # UB → GM
        """,
        "tile_copy.tile_io_plain": """
            pto.tload(a_part, a_tile)
            pto.tstore(o_tile, o_part)
        """,
        "compile_and_launch": """
            # Compile once, cache the result.
            compiled = blocked_copy.compile(BLOCK=128)

            # Allocate or obtain input/output tensors (NumPy, torch-npu, ...).
            import numpy as np
            A = np.random.randn(4, 128).astype(np.float32)
            O = np.empty_like(A)

            # Launch on the NPU.
            compiled[1, None](A, O)
        """,
        "spmd_launch": """
            # Process batch * heads slices in parallel.
            compiled[batch * heads, stream](Q, K, V, O)
        """,
        "spmd_builtins": """
            block_idx = pto.get_block_idx()
            block_num = pto.get_block_num()
        """,
    },
)


INTRO = _group(
    "intro",
    {
        "flash_wrapper": """
            def flash_attention(Q, K, V, *, O=None, causal=False):
                if O is None:
                    O = pto.empty_like(Q)
                compiled = flash_attention_kernel.compile(
                    BLOCK_Q=128, BLOCK_KV=128, CAUSAL=causal
                )
                compiled[batch * heads, stream](Q, K, V, O)
                return O
        """,
        "jit_signature": """
            @pto.jit(target="a5")
            def flash_attention_kernel(
                Q: pto.tensor_spec(rank=4, dtype=pto.f32),
                K: pto.tensor_spec(rank=4, dtype=pto.f32),
                V: pto.tensor_spec(rank=4, dtype=pto.f32),
                O: pto.tensor_spec(rank=4, dtype=pto.f32),
                *,
                BLOCK_Q: pto.constexpr = 128,
                BLOCK_KV: pto.constexpr = 128,
                CAUSAL: pto.constexpr = False,
            ):
                ...
        """,
    },
)


CONTROL_FLOW = _group(
    "control_flow",
    {
        "python_for_unroll": """
            @pto.jit(target="a5")
            def unrolled_kernel(A, O, *, N: pto.constexpr):
                a_view = pto.make_tensor_view(A, shape=[N], strides=A.strides)
                o_view = pto.make_tensor_view(O, shape=[N], strides=O.strides)

                # N is constexpr, so range(N) is known at trace time.
                # The loop unrolls: the device gets N copies of the body.
                for i in range(N):
                    a_part = pto.partition_view(a_view, offsets=[i], sizes=[1])
                    o_part = pto.partition_view(o_view, offsets=[i], sizes=[1])
                    a_tile = pto.alloc_tile(shape=[1], dtype=pto.f32)
                    o_tile = pto.alloc_tile(shape=[1], dtype=pto.f32)
                    pto.tload(a_part, a_tile)
                    pto.tadd(a_tile, a_tile, o_tile)
                    pto.tstore(o_tile, o_part)
        """,
        "for_basic_form": """
            with pto.for_(start, stop, step) as iv:
                # iv is the loop index (0-based relative to start)
                ...
        """,
        "for_compare": """
            # Trace-time unrolling — BLOCK must be constexpr
            for i in range(BLOCK):
                ...

            # Device-side loop — num_blocks can be dynamic
            with pto.for_(0, num_blocks, step=1) as i:
                offset = i * BLOCK
                ...
        """,
        "for_nested": """
            with pto.for_(0, rows, step=1) as r:
                with pto.for_(0, cols, step=1) as c:
                    val = scalar.load(tile[r, c])
                    ...
        """,
        "carry_ping_pong": """
            # Allocate ping-pong state tiles
            m_prev = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, blayout="ColMajor")
            m_next = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, blayout="ColMajor")
            l_prev = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, blayout="ColMajor")
            l_next = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, blayout="ColMajor")

            # Initialize prev tiles
            m_prev.fill(float("-inf"))
            l_prev.fill(0.0)

            loop = pto.for_(0, num_blocks, step=1).carry(m=m_prev, l=l_prev)
            with loop:
                m_cur = loop.m
                l_cur = loop.l

                # ... compute new m and l into m_next, l_next ...

                loop.update(m=m_next, l=l_next)
        """,
        "chunked_carry": """
            VEC = pto.elements_per_vreg(pto.f32)
            col_loop = pto.for_(0, cols, step=VEC).carry(remained=cols)
            with col_loop:
                c = col_loop.iv
                remained = col_loop.remained
                mask, remained = pto.make_mask(pto.f32, remained)
                vec = pto.vlds(tile[r, c:])
                # ... operate under mask ...
                pto.vsts(vec, out_tile[r, c:], mask)
                col_loop.update(remained=remained)
        """,
        "if_value_merge": """
            @pto.simt
            def conditional_scale(
                tile: pto.Tile,
                threshold: pto.f32,
                scale: pto.f32,
                rows: pto.i32,
                cols: pto.i32,
            ):
                with pto.for_(0, rows, step=1) as r:
                    with pto.for_(0, cols, step=1) as c:
                        val = scalar.load(tile[r, c])
                        big = scalar.gt(val, threshold)

                        with pto.if_(big):
                            # Branch A: scale the value up
                            val = val * scale
                        with pto.else_():
                            # Branch B: leave it as-is
                            pass

                        # val is usable here — it is the merged result from both branches.
                        # If big was true,  val = original * scale.
                        # If big was false, val = original (passed through unchanged).
                        scalar.store(val, tile[r, c])
        """,
        "if_expression": """
            result = pto.if_(cond, then_value, else_value)
        """,
        "constexpr_tracing": """
            @pto.jit(target="a5")
            def kernel(A, *, BLOCK: pto.constexpr = 128, UNROLL: pto.constexpr = False):
                N = A.shape[0]
                num_blocks = (N + BLOCK - 1) // BLOCK

                if UNROLL:
                    # Trace-time: UNROLL is known, so this branch resolves at compile time.
                    # Each iteration records separately — the loop is fully unrolled.
                    for i in range(num_blocks):
                        ...
                else:
                    # Device-side: a single loop instruction is recorded.
                    with pto.for_(0, num_blocks, step=1) as i:
                        ...
        """,
    },
)


TYPES = _group(
    "types",
    {
        "scalar.constructors": """
            x = pto.i32(1024)
            y = pto.ui16(7)
            z: pto.i32 = 1024
        """,
        "scalar.integer_literals": """
            count = pto.i32(1024)
            delta = pto.i16(-12)
            hi_bit = pto.i32("0x80000000")   # bit-pattern: -2147483648
        """,
        "scalar.float_literals": """
            a = pto.f16(-1.5)
            b = pto.f32("inf")
            c = pto.f32("-inf")
            d = pto.f32("nan")
            # Bit-pattern hex
            f16_neg_inf = pto.f16("0xFC00")
        """,
        "vreg.elements_per_vreg": """
            lanes = pto.elements_per_vreg(pto.f32)  # 64
        """,
        "vreg.vbitcast": """
            fvec = pto.vlds(ptr, offset)            # !pto.vreg<64xf32>
            ivec = pto.vbitcast(fvec, pto.i32)      # !pto.vreg<64xi32>
            f16_vec = pto.vbitcast(fvec, pto.f16)   # !pto.vreg<128xf16>
        """,
        "mask.pbitcast": """
            mask_b16 = pto.pbitcast(mask_b8, pto.mask_b16)
        """,
        "ptr.declare": """
            ptr_gm  = pto.ptr(pto.f32, pto.MemorySpace.GM)
            ptr_ub  = pto.ptr(pto.f16, pto.MemorySpace.UB)
        """,
        "tensor_view.make": """
            @pto.jit(target="a5")
            def kernel(A, *, BLOCK: pto.constexpr):
                tv = pto.make_tensor_view(A, shape=[N], strides=A.strides)
        """,
        "partition_view.basic": """
            part = pto.partition_view(tv, offsets=[row_offset, 0], sizes=[BLOCK, dim])
        """,
        "tile.alloc": """
            # UB tile
            a_tile = pto.alloc_tile(shape=[BLOCK, dim], dtype=pto.f32)

            # Logical column tile
            m_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, blayout="ColMajor")

            # Cube-local scratch with explicit memory space
            q_l0a = pto.alloc_tile(shape=[Br, dim], dtype=pto.f16, memory_space=pto.MemorySpace.LEFT)
            s_acc = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, memory_space=pto.MemorySpace.ACC)
        """,
        "tile.methods": """
            m_prev_tile.fill(float("-inf"))
            l_prev_tile.fill(0.0)

            rows = q_tile.valid_shape[0]
            cols = k_tile.valid_shape[1]

            meta_ptr = meta_tile.as_ptr()
        """,
        "chunk_stride": """
            VEC = pto.elements_per_vreg(pto.f32)
            with pto.for_(0, cols, step=VEC) as c:
                ...
        """,
        "simt_blend_usage": """
            @pto.simt
            def blend_output_rows(
                o_prev_tile: pto.Tile, pv_tile: pto.Tile,
                alpha_tile: pto.Tile, beta_tile: pto.Tile,
                o_next_tile: pto.Tile,
                row_start: pto.i32, row_stop: pto.i32, valid_dim: pto.i32,
            ):
                with pto.for_(row_start, row_stop, step=1) as row:
                    alpha = scalar.load(alpha_tile[row, 0])
                    beta = scalar.load(beta_tile[row, 0])
                    with pto.for_(0, valid_dim, step=1) as col:
                        o_prev = scalar.load(o_prev_tile[row, col])
                        pv_val = scalar.load(pv_tile[row, col])
                        o_next = alpha * o_prev + beta * pv_val
                        scalar.store(o_next, o_next_tile[row, col])
        """,
        "elementwise_scale": """
            @pto.simt
            def elementwise_scale(
                src_tile: pto.Tile,
                dst_tile: pto.Tile,
                scale: pto.f32,
                rows: pto.i32,
                cols: pto.i32,
            ):
                with pto.for_(0, rows, step=1) as r:
                    with pto.for_(0, cols, step=1) as c:
                        val = scalar.load(src_tile[r, c])
                        scaled = val * scale
                        scalar.store(scaled, dst_tile[r, c])
        """,
        "per_row_coeffs": """
            @pto.simt
            def blend_with_per_row_coeffs(
                o_prev_tile: pto.Tile,
                pv_tile: pto.Tile,
                alpha_tile: pto.Tile,    # [rows, 1] — one coefficient per row
                beta_tile: pto.Tile,     # [rows, 1]
                o_next_tile: pto.Tile,
                rows: pto.i32,
                cols: pto.i32,
            ):
                with pto.for_(0, rows, step=1) as r:
                    alpha = scalar.load(alpha_tile[r, 0])   # read once per row
                    beta = scalar.load(beta_tile[r, 0])     # read once per row
                    with pto.for_(0, cols, step=1) as c:
                        o_prev = scalar.load(o_prev_tile[r, c])
                        pv_val = scalar.load(pv_tile[r, c])
                        o_next = alpha * o_prev + beta * pv_val
                        scalar.store(o_next, o_next_tile[r, c])
        """,
    },
)


SCALAR_POINTER = _group(
    "scalar_pointer",
    {
        "mixed_expression": """
            alpha * o_prev + beta * pv_val
            # ^ Python float (trace-time constant, e.g. 1.0 / sqrt(dim))
            #        ^ PTO scalar (loaded from tile at runtime)
            #                  ^ PTO scalar (loaded from tile at runtime)
        """,
        "load.tile_form": """
            val = scalar.load(tile[row, col])
        """,
        "load.pointer_forms": """
            val = scalar.load(ptr, offset)       # explicit offset
            val = scalar.load(ptr + offset)      # pointer arithmetic shorthand
        """,
        "store.tile_form": """
            scalar.store(value, tile[row, col])
        """,
        "store.pointer_form": """
            scalar.store(value, ptr, offset)
        """,
        "arith.example": """
            o_next = alpha * o_prev + beta * pv_val      # multiply-add
            l_scaled = l_prev * scalar.exp(m_prev - m_next)  # subtraction inside exp
            step = (N + BLOCK - 1) // BLOCK               # Python int arithmetic (trace-time)
        """,
        "math.example": """
            m_next = scalar.max(m_prev, row_max)
            l_scaled = l_prev * scalar.exp(m_prev - m_next)
            need_scale = scalar.gt(val, threshold)
        """,
        "alias.example": """
            scalar = pto.scalar

            m_next = scalar.max(m_prev, row_max)
            l_scaled = l_prev * scalar.exp(m_prev - m_next)
        """,
        "as_ptr": """
            gm_ptr = partition.as_ptr()    # GM pointer from a PartitionTensorView
            ub_ptr = tile.as_ptr()         # UB pointer from a Tile
        """,
        "addptr": """
            ptr = pto.addptr(base_ptr, 1024)  # advances by 1024 * sizeof(T) bytes
        """,
        "bytewidth": """
            bw = pto.bytewidth(pto.f32)   # 4
            bw = pto.bytewidth(pto.f16)   # 2
            bw = pto.bytewidth(pto.i8)    # 1
        """,
        "elements_per_vreg": """
            vec = pto.elements_per_vreg(pto.f32)   # 64
            vec = pto.elements_per_vreg(pto.f16)   # 128
            vec = pto.elements_per_vreg(pto.i8)    # 256
        """,
        "chunk_stride": """
            VEC = pto.elements_per_vreg(pto.f32)
            with pto.for_(0, cols, step=VEC) as c:
                ...
        """,
        "simt_blend_usage": """
            @pto.simt
            def blend_output_rows(
                o_prev_tile: pto.Tile, pv_tile: pto.Tile,
                alpha_tile: pto.Tile, beta_tile: pto.Tile,
                o_next_tile: pto.Tile,
                row_start: pto.i32, row_stop: pto.i32, valid_dim: pto.i32,
            ):
                with pto.for_(row_start, row_stop, step=1) as row:
                    alpha = scalar.load(alpha_tile[row, 0])
                    beta = scalar.load(beta_tile[row, 0])
                    with pto.for_(0, valid_dim, step=1) as col:
                        o_prev = scalar.load(o_prev_tile[row, col])
                        pv_val = scalar.load(pv_tile[row, col])
                        o_next = alpha * o_prev + beta * pv_val
                        scalar.store(o_next, o_next_tile[row, col])
        """,
        "elementwise_scale": """
            @pto.simt
            def elementwise_scale(
                src_tile: pto.Tile,
                dst_tile: pto.Tile,
                scale: pto.f32,
                rows: pto.i32,
                cols: pto.i32,
            ):
                with pto.for_(0, rows, step=1) as r:
                    with pto.for_(0, cols, step=1) as c:
                        val = scalar.load(src_tile[r, c])
                        scaled = val * scale
                        scalar.store(scaled, dst_tile[r, c])
        """,
        "per_row_coeffs": """
            @pto.simt
            def blend_with_per_row_coeffs(
                o_prev_tile: pto.Tile,
                pv_tile: pto.Tile,
                alpha_tile: pto.Tile,    # [rows, 1] — one coefficient per row
                beta_tile: pto.Tile,     # [rows, 1]
                o_next_tile: pto.Tile,
                rows: pto.i32,
                cols: pto.i32,
            ):
                with pto.for_(0, rows, step=1) as r:
                    alpha = scalar.load(alpha_tile[r, 0])   # read once per row
                    beta = scalar.load(beta_tile[r, 0])     # read once per row
                    with pto.for_(0, cols, step=1) as c:
                        o_prev = scalar.load(o_prev_tile[r, c])
                        pv_val = scalar.load(pv_tile[r, c])
                        o_next = alpha * o_prev + beta * pv_val
                        scalar.store(o_next, o_next_tile[r, c])
        """,
    },
)


COMPUTE = _group(
    "compute",
    {
        "tile.binary": """
            pto.tadd(a_tile, b_tile, o_tile)
            pto.tmul(scale_tile, data_tile, scaled_tile)
        """,
        "tile.row_expand": """
            # alpha_tile: [rows, 1], beta_tile: [rows, 1], data_tile: [rows, cols]
            pto.trowexpandmul(data_tile, alpha_tile, scaled_tile)
            pto.trowexpandadd(scaled_tile, beta_tile, result_tile)
        """,
        "vector.unary": """
            exp_vec = pto.vexp(s_row, col_mask)
        """,
        "vector.scalar_sub": """
            s_shifted = pto.vsubs(s_row, m_next, col_mask)
        """,
        "vector.group_reduction": """
            row_max = pto.vcgmax(s_row, col_mask)   # per-group max → first lane of each group
            row_sum = pto.vcgadd(p_row, col_mask)   # per-group sum → first lane of each group
        """,
        "cube.matmul_pattern": """
            @pto.cube
            def qk_matmul(q_tile, k_tile, q_l0a, k_l0b, s_acc, s_tile):
                m = q_tile.valid_shape[0]
                k = q_tile.valid_shape[1]
                n = k_tile.valid_shape[0]

                # Stage: UB → L0A / L0B
                pto.mte_l1_l0a(q_tile.as_ptr(), q_l0a.as_ptr(), m, k)
                pto.mte_l1_l0b(k_tile.as_ptr(), k_l0b.as_ptr(), k, n, transpose=True)

                # Compute: L0A × L0B → L0C
                pto.mad(q_l0a.as_ptr(), k_l0b.as_ptr(), s_acc.as_ptr(), m, n, k)

                # Writeback: L0C → UB
                pto.mte_l0c_ub(s_acc.as_ptr(), s_tile.as_ptr(), m, n, n, n, 0)
        """,
    },
)


MASKS = _group(
    "masks",
    {
        "make_mask.chunked_loop": """
            VEC = pto.elements_per_vreg(pto.f32)
            col_loop = pto.for_(0, cols, step=VEC).carry(remained=cols)
            with col_loop:
                c = col_loop.iv
                remained = col_loop.remained
                mask, remained = pto.make_mask(pto.f32, remained)
                vec = pto.vlds(tile[r, c:])
                # ... operate under mask ...
                pto.vsts(vec, out_tile[r, c:], mask)
                col_loop.update(remained=remained)
        """,
        "make_mask.full_pattern": """
            full_mask = pto.make_mask(pto.f32, pto.MaskPattern.ALL)
        """,
        "pbitcast": """
            # Reinterpret a b16 mask as b32
            mask32 = pto.pbitcast(mask16, pto.mask_b32)
        """,
        "vcmps.threshold": """
            big = pto.vcmps(scores, threshold, seed, pto.CmpMode.GT)
            # big[i] = 1 where scores[i] > threshold
        """,
        "tail_safe_vector_pattern": """
            VEC = pto.elements_per_vreg(pto.f32)
            with pto.for_(0, rows, step=1) as r:
                col_loop = pto.for_(0, cols, step=VEC).carry(remained=cols)
                with col_loop:
                    c = col_loop.iv
                    remained = col_loop.remained
                    mask, remained = pto.make_mask(pto.f32, remained)

                    vec = pto.vlds(tile[r, c:])
                    vec = pto.vexp(vec, mask)
                    pto.vsts(vec, out_tile[r, c:], mask)

                    col_loop.update(remained=remained)
        """,
    },
)


SYNC = _group(
    "sync",
    {
        "set_flag": """
            from pto import Pipe, Event

            # MTE2 has finished loading tile data — signal Vector pipeline
            pto.set_flag(Pipe.MTE2, Pipe.V, Event.ID0)
        """,
        "wait_flag": """
            from pto import Pipe, Event

            # Vector pipeline waits for MTE2 to finish loading
            pto.wait_flag(Pipe.MTE2, Pipe.V, Event.ID0)
        """,
        "pipe_barrier": """
            from pto import Pipe

            # Full hardware barrier — all pipelines synchronize
            pto.pipe_barrier(Pipe.ALL)
        """,
        "ukernel_pattern": """
            @pto.ukernel
            def gemm_block(q_tile, k_tile, v_tile, o_tile, ...):
                # DMA: load K and V tiles from GM to UB
                # mte_load derives strides, burst sizes, etc. from k_part / k_tile types
                pto.mte_load(k_part, k_tile)
                pto.mte_load(v_part, v_tile)

                # Signal: DMA done, UB data ready
                pto.set_flag(Pipe.MTE2, Pipe.V, Event.ID0)

                # Wait: vector pipeline stalls until data arrives
                pto.wait_flag(Pipe.MTE2, Pipe.V, Event.ID0)

                # Compute: now safe to use k_tile and v_tile
                qk_matmul(q_tile, k_tile, ...)
                pv_matmul(p_tile, v_tile, ...)

                # Signal: compute done, results ready for store
                pto.set_flag(Pipe.V, Pipe.MTE3, Event.ID1)
                pto.wait_flag(Pipe.V, Pipe.MTE3, Event.ID1)

                # DMA: store results back to GM
                pto.mte_store(o_tile, o_part)
        """,
        "double_buffering": """
            from pto import Pipe

            # Pipeline V acquires buffer 0 for compute
            pto.get_buf(Pipe.V, 0, 0)

            # ... compute into buffer 0 ...

            # Release buffer 0 — DMA can now refill it
            pto.rls_buf(Pipe.V, 0, 0)

            # Pipeline MTE2 acquires buffer 0 for reload
            pto.get_buf(Pipe.MTE2, 0, 0)

            # ... DMA loads next block into buffer 0 ...

            pto.rls_buf(Pipe.MTE2, 0, 0)
        """,
        "mem_bar": """
            from pto import BarrierType

            # Ensure all prior vector stores are visible before any subsequent vector loads
            pto.mem_bar(BarrierType.VST_VLD)
        """,
        "flash_block_barriers": """
            @pto.ukernel
            def flash_attention_block(q_tile, k_tile, v_tile, ...):
                # Phase 1: load K/V
                pto.mte_load(k_part, k_tile)
                pto.mte_load(v_part, v_tile)
                pto.pipe_barrier(Pipe.ALL)

                # Phase 2: S = Q @ K^T
                qk_matmul(q_tile, k_tile, ...)
                pto.pipe_barrier(Pipe.ALL)

                # Phase 3: softmax(S)
                online_softmax(s_tile, ...)
                pto.pipe_barrier(Pipe.ALL)

                # Phase 4: PV = P @ V
                pv_matmul(p_tile, v_tile, ...)
                pto.pipe_barrier(Pipe.ALL)

                # Phase 5: blend output
                blend_output(o_prev_tile, pv_tile, ...)
                pto.pipe_barrier(Pipe.ALL)
        """,
        "set_cross_core": """
            from pto import Event

            # Signal core 0 that our computation is complete
            pto.set_cross_core(0, Event.ID0)
        """,
        "wait_flag_dev": """
            from pto import Event

            # Core 1 waits for core 0 to signal event ID0
            pto.wait_flag_dev(0, Event.ID0)
        """,
        "set_intra_block": """
            from pto import Event

            # Signal event ID0 on block/pipeline 0
            pto.set_intra_block(0, Event.ID0)
        """,
        "wait_intra_core": """
            from pto import Event

            # Pipeline 1 waits for event ID0 from pipeline 0 within the same block
            pto.wait_intra_core(1, Event.ID0)
        """,
        "set_intra_core": """
            pto.set_intra_core(3)
        """,
    },
)


ADDITIONAL_EXAMPLES = _group(
    "additional_examples",
    {
        "mat_add.kernel": """
            @pto.jit(target="a5")
            def mat_add(A, B, O, *, BLOCK_M: pto.constexpr = 64, BLOCK_N: pto.constexpr = 128):
                M, N_ = A.shape

                a_view = pto.make_tensor_view(A, shape=[M, N_], strides=A.strides)
                b_view = pto.make_tensor_view(B, shape=[M, N_], strides=B.strides)
                o_view = pto.make_tensor_view(O, shape=[M, N_], strides=O.strides)

                a_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32)
                b_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32)
                o_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32)

                num_m = (M + BLOCK_M - 1) // BLOCK_M
                num_n = (N_ + BLOCK_N - 1) // BLOCK_N

                with pto.for_(0, num_m, step=1) as mi:
                    m_off = mi * BLOCK_M
                    with pto.for_(0, num_n, step=1) as ni:
                        n_off = ni * BLOCK_N

                        a_part = pto.partition_view(a_view, offsets=[m_off, n_off], sizes=[BLOCK_M, BLOCK_N])
                        b_part = pto.partition_view(b_view, offsets=[m_off, n_off], sizes=[BLOCK_M, BLOCK_N])
                        o_part = pto.partition_view(o_view, offsets=[m_off, n_off], sizes=[BLOCK_M, BLOCK_N])

                        pto.tload(a_part, a_tile)
                        pto.tload(b_part, b_tile)
                        pto.tadd(a_tile, b_tile, o_tile)
                        pto.tstore(o_tile, o_part)
        """,
        "mat_add.wrapper": """
            def mat_add_wrapper(A, B, O=None, stream=None):
                if O is None:
                    O = pto.empty_like(A)
                compiled = mat_add.compile(BLOCK_M=64, BLOCK_N=128)
                m, n = A.shape[1], A.shape[2]  # assuming batch-first: [batch, M, N]
                compiled[A.shape[0], stream](A, B, O)
                return O
        """,
        "tail.simd_kernel": """
            @pto.simd
            def add_rows_with_tail(a_tile: pto.Tile, b_tile: pto.Tile, o_tile: pto.Tile,
                                   rows: pto.i32, cols: pto.i32):
                VEC = pto.elements_per_vreg(pto.f32)          # 64 for f32

                with pto.for_(0, rows, step=1) as r:
                    col_loop = pto.for_(0, cols, step=VEC).carry(remained=cols)
                    with col_loop:
                        c = col_loop.iv
                        remained = col_loop.remained
                        mask, remained = pto.make_mask(pto.f32, remained)

                        a_vec = pto.vlds(a_tile[r, c:])       # load under mask
                        b_vec = pto.vlds(b_tile[r, c:])
                        o_vec = pto.vadd(a_vec, b_vec, mask)  # compute under mask
                        pto.vsts(o_vec, o_tile[r, c:], mask)  # store under mask

                        col_loop.update(remained=remained)
        """,
        "tail.tile_level": """
            @pto.jit(target="a5")
            def vec_add_with_tail(A, B, O, *, BLOCK: pto.constexpr):
                N = A.shape[0]

                a_view = pto.make_tensor_view(A, shape=[N], strides=A.strides)
                b_view = pto.make_tensor_view(B, shape=[N], strides=B.strides)
                o_view = pto.make_tensor_view(O, shape=[N], strides=O.strides)

                a_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32)
                b_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32)
                o_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32)

                num_blocks = (N + BLOCK - 1) // BLOCK

                with pto.for_(0, num_blocks, step=1) as i:
                    offset = i * BLOCK
                    this_block = min(BLOCK, N - offset)

                    a_part = pto.partition_view(a_view, offsets=[offset], sizes=[this_block])
                    b_part = pto.partition_view(b_view, offsets=[offset], sizes=[this_block])
                    o_part = pto.partition_view(o_view, offsets=[offset], sizes=[this_block])

                    pto.tload(a_part, a_tile)
                    pto.tload(b_part, b_tile)

                    a_tile.valid_shape = [this_block]
                    b_tile.valid_shape = [this_block]
                    o_tile.valid_shape = [this_block]

                    pto.tadd(a_tile, b_tile, o_tile)
                    pto.tstore(o_tile, o_part)
        """,
        "gemm.cube_kernel": """
            @pto.cube
            def gemm_tile(a_tile: pto.Tile, b_tile: pto.Tile, o_tile: pto.Tile,
                          a_l0a: pto.Tile, b_l0b: pto.Tile, o_acc: pto.Tile):
                m = a_tile.valid_shape[0]
                k = a_tile.valid_shape[1]
                n = b_tile.valid_shape[0]

                pto.mte_l1_l0a(a_tile.as_ptr(), a_l0a.as_ptr(), m, k)
                pto.mte_l1_l0b(b_tile.as_ptr(), b_l0b.as_ptr(), k, n, transpose=True)
                pto.mad(a_l0a.as_ptr(), b_l0b.as_ptr(), o_acc.as_ptr(), m, n, k)
                pto.mte_l0c_ub(o_acc.as_ptr(), o_tile.as_ptr(), m, n, n, n, 0)
        """,
        "gemm.jit_orchestration": """
            @pto.jit(target="a5")
            def gemm(A, B, O, *, BLOCK_M: pto.constexpr = 64,
                     BLOCK_K: pto.constexpr = 64, BLOCK_N: pto.constexpr = 64):
                M, K_ = A.shape
                _, N_ = B.shape

                a_view = pto.make_tensor_view(A, shape=[M, K_], strides=A.strides)
                b_view = pto.make_tensor_view(B, shape=[K_, N_], strides=B.strides)
                o_view = pto.make_tensor_view(O, shape=[M, N_], strides=O.strides)

                a_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_K], dtype=pto.f32)
                b_tile = pto.alloc_tile(shape=[BLOCK_K, BLOCK_N], dtype=pto.f32)
                o_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32)

                a_l0a = pto.alloc_tile(shape=[BLOCK_M, BLOCK_K], dtype=pto.f32,
                                       memory_space=pto.MemorySpace.LEFT)
                b_l0b = pto.alloc_tile(shape=[BLOCK_K, BLOCK_N], dtype=pto.f32,
                                       memory_space=pto.MemorySpace.RIGHT)
                o_acc = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32,
                                       memory_space=pto.MemorySpace.ACC)

                num_m = (M + BLOCK_M - 1) // BLOCK_M
                num_n = (N_ + BLOCK_N - 1) // BLOCK_N
                num_k = (K_ + BLOCK_K - 1) // BLOCK_K

                with pto.for_(0, num_m, step=1) as mi:
                    m_off = mi * BLOCK_M
                    with pto.for_(0, num_n, step=1) as ni:
                        n_off = ni * BLOCK_N

                        o_tile.fill(0.0)

                        with pto.for_(0, num_k, step=1) as ki:
                            k_off = ki * BLOCK_K

                            a_part = pto.partition_view(a_view, offsets=[m_off, k_off],
                                                        sizes=[BLOCK_M, BLOCK_K])
                            b_part = pto.partition_view(b_view, offsets=[k_off, n_off],
                                                        sizes=[BLOCK_K, BLOCK_N])
                            o_part = pto.partition_view(o_view, offsets=[m_off, n_off],
                                                        sizes=[BLOCK_M, BLOCK_N])

                            pto.tload(a_part, a_tile)
                            pto.tload(b_part, b_tile)

                            gemm_tile(a_tile, b_tile, o_tile, a_l0a, b_l0b, o_acc)

                        pto.tstore(o_tile, o_part)
        """,
        "gemm.wrapper": """
            def gemm_wrapper(A, B, O=None, stream=None):
                if O is None:
                    O = pto.empty([A.shape[0], B.shape[1]], dtype=A.dtype)
                compiled = gemm.compile(BLOCK_M=64, BLOCK_K=64, BLOCK_N=64)
                compiled[1, stream](A, B, O)
                return O
        """,
        "norm.simd_stats": """
            @pto.simd
            def block_mean_var(x_tile: pto.Tile, block_size: pto.i32,
                              mu_prev: pto.f32, n_prev: pto.f32, m2_prev: pto.f32,
                              mu_next_tile: pto.Tile, n_next_tile: pto.Tile,
                              m2_next_tile: pto.Tile):
                VEC = pto.elements_per_vreg(pto.f32)

                # Per-row cross-lane reductions to compute the block sum and sum-of-squares
                row_sum = pto.vdup(0.0, pto.f32)
                row_sum2 = pto.vdup(0.0, pto.f32)

                col_loop = pto.for_(0, block_size, step=VEC).carry(row_sum=row_sum, row_sum2=row_sum2)
                with col_loop:
                    c = col_loop.iv
                    remained = pto.i32(block_size) - c
                    mask, _ = pto.make_mask(pto.f32, remained)

                    x_vec = pto.vlds(x_tile[0, c:])
                    row_sum = pto.vcadd(x_vec, mask)
                    row_sum2 = pto.vcadd(pto.vmul(x_vec, x_vec, mask), mask)
                    col_loop.update(row_sum=row_sum, row_sum2=row_sum2)

                block_n = pto.cvt(block_size, pto.f32)
                block_mean = pto.vdiv(col_loop.final("row_sum"), block_n)
                block_mean_sq = pto.vdiv(col_loop.final("row_sum2"), block_n)

                # Welford update: merge block statistics into running state
                n_next = n_prev + block_n
                delta = block_mean - mu_prev
                mu_next = mu_prev + delta * block_n / n_next
                m2_next = m2_prev + pto.vdiv(row_sum2, block_n) * block_n  # simplified

                scalar.store(n_next, n_next_tile[0, 0])
                scalar.store(mu_next, mu_next_tile[0, 0])
                scalar.store(m2_next, m2_next_tile[0, 0])
        """,
        "norm.ukernel": """
            @pto.ukernel
            def norm_block(x_part: pto.PartitionTensorView, x_tile: pto.Tile,
                           block_size: pto.i32,
                           mu_prev: pto.f32, n_prev: pto.f32, m2_prev: pto.f32,
                           mu_next_tile: pto.Tile, n_next_tile: pto.Tile,
                           m2_next_tile: pto.Tile):
                pto.mte_load(x_part, x_tile)
                pto.pipe_barrier(pto.Pipe.ALL)

                block_mean_var(x_tile, block_size,
                               mu_prev, n_prev, m2_prev,
                               mu_next_tile, n_next_tile, m2_next_tile)
                pto.pipe_barrier(pto.Pipe.ALL)
        """,
        "norm.jit_carry": """
            @pto.jit(target="a5")
            def online_layernorm(X, O, *, BLOCK: pto.constexpr):
                N = X.shape[0]
                x_view = pto.make_tensor_view(X, shape=[N], strides=X.strides)
                o_view = pto.make_tensor_view(O, shape=[N], strides=O.strides)

                x_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32)
                o_tile = pto.alloc_tile(shape=[BLOCK], dtype=pto.f32)

                mu_tile = pto.alloc_tile(shape=[1], dtype=pto.f32)
                n_tile = pto.alloc_tile(shape=[1], dtype=pto.f32)
                m2_tile = pto.alloc_tile(shape=[1], dtype=pto.f32)

                num_blocks = (N + BLOCK - 1) // BLOCK

                # Carry: running statistics across blocks
                block_loop = pto.for_(0, num_blocks, step=1).carry(
                    mu=pto.f32(0.0), n=pto.f32(0.0), m2=pto.f32(0.0)
                )
                with block_loop:
                    i = block_loop.iv
                    offset = i * BLOCK
                    this_block = min(BLOCK, N - offset)

                    x_part = pto.partition_view(x_view, offsets=[offset], sizes=[this_block])

                    mu_prev = block_loop.mu
                    n_prev = block_loop.n
                    m2_prev = block_loop.m2

                    norm_block(x_part, x_tile, pto.i32(this_block),
                               mu_prev, n_prev, m2_prev,
                               mu_tile, n_tile, m2_tile)

                    n_next = scalar.load(n_tile[0, 0])
                    mu_next = scalar.load(mu_tile[0, 0])
                    m2_next = scalar.load(m2_tile[0, 0])

                    block_loop.update(mu=mu_next, n=n_next, m2=m2_next)

                # After all blocks: finalize normalization with the running stats
                global_var = m2_next / n_next

                # Second pass: normalize each block (using same tiling)
                with pto.for_(0, num_blocks, step=1) as i:
                    offset = i * BLOCK
                    this_block = min(BLOCK, N - offset)
                    x_part = pto.partition_view(x_view, offsets=[offset], sizes=[this_block])
                    o_part = pto.partition_view(o_view, offsets=[offset], sizes=[this_block])

                    pto.tload(x_part, x_tile)
                    pto.tnormalize(x_tile, mu_next, global_var, o_tile)
                    pto.tstore(o_tile, o_part)
        """,
    },
)


EXCERPT_SOURCES = {
    **QUICK_START,
    **INTRO,
    **ENTRY_POINTS,
    **CONTROL_FLOW,
    **TYPES,
    **SCALAR_POINTER,
    **COMPUTE,
    **MASKS,
    **SYNC,
    **DATA_MOVEMENT,
    **FLASH,
    **ADDITIONAL_EXAMPLES,
}
