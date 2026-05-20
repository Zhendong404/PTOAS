#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass
from textwrap import dedent


SNIPPET_PLACEHOLDER = "__PTODSL_DOC_SNIPPET__"


@dataclass(frozen=True)
class FragmentFixture:
    template: str


def _fixture(template: str) -> FragmentFixture:
    return FragmentFixture(template=dedent(template).strip("\n"))


def render_fragment_fixture(fixture: FragmentFixture, snippet: str) -> str:
    rendered_lines: list[str] = []
    placeholder_count = 0
    snippet_lines = snippet.rstrip("\n").splitlines()

    for line in fixture.template.splitlines():
        if SNIPPET_PLACEHOLDER not in line:
            rendered_lines.append(line)
            continue

        placeholder_count += 1
        if line.strip() != SNIPPET_PLACEHOLDER:
            raise ValueError(
                f"fixture placeholder must occupy its own line: {line!r}"
            )

        indent = line[: line.index(SNIPPET_PLACEHOLDER)]
        rendered_lines.extend(
            f"{indent}{snippet_line}" if snippet_line else ""
            for snippet_line in snippet_lines
        )

    if placeholder_count != 1:
        raise ValueError(
            f"fixture must contain exactly one placeholder line, found {placeholder_count}"
        )

    return "\n".join(rendered_lines) + "\n"


FRAGMENT_FIXTURES = {
    "type_system.scalar_expr": _fixture(
        f"""
        @pto.jit(target="a5")
        def type_system_scalar_expr_probe():
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "type_system.low_precision_types": _fixture(
        f"""
        @pto.jit(target="a5")
        def type_system_low_precision_types_probe(
            *,
            BLOCK: pto.constexpr = 128,
        ):
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "type_system.tensor_view": _fixture(
        f"""
        @pto.jit(target="a5")
        def type_system_tensor_view_probe(
            A: pto.tensor_spec(rank=2, dtype=pto.f32),
            *,
            BLOCK: pto.constexpr = 128,
        ):
            rows = A.shape[0]
            cols = A.shape[1]
            N = rows
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "type_system.partition_view": _fixture(
        f"""
        @pto.jit(target="a5")
        def type_system_partition_view_probe(
            A: pto.tensor_spec(rank=2, dtype=pto.f32),
            *,
            BLOCK: pto.constexpr = 128,
        ):
            rows = A.shape[0]
            cols = A.shape[1]
            dim = cols
            row_offset = 0
            tv = pto.make_tensor_view(A, shape=A.shape, strides=A.strides)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "type_system.tile_alloc": _fixture(
        f"""
        @pto.jit(target="a5")
        def type_system_tile_alloc_probe(
            *,
            BLOCK: pto.constexpr = 128,
            Br: pto.constexpr = 16,
            Bc: pto.constexpr = 16,
            dim: pto.constexpr = 16,
        ):
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "type_system.tile_methods": _fixture(
        f"""
        @pto.jit(target="a5")
        def type_system_tile_methods_probe(
            *,
            Br: pto.constexpr = 16,
            Bc: pto.constexpr = 16,
            dim: pto.constexpr = 16,
        ):
            m_prev_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, blayout="ColMajor")
            l_prev_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, blayout="ColMajor")
            q_tile = pto.alloc_tile(shape=[Br, dim], dtype=pto.f32)
            k_tile = pto.alloc_tile(shape=[Bc, dim], dtype=pto.f32)
            meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 3])
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "type_system.vreg_bitcast": _fixture(
        f"""
        @pto.jit(target="a5")
        def type_system_vreg_bitcast_probe(
            *,
            BLOCK: pto.constexpr = 128,
        ):
            tile = pto.alloc_tile(shape=[2, BLOCK], dtype=pto.f32)
            row = 0
            fvec = pto.vlds(tile[row, 0:])
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "type_system.vreg_bitcast_ptr": _fixture(
        f"""
        @pto.jit(target="a5")
        def type_system_vreg_bitcast_ptr_probe(
            *,
            BLOCK: pto.constexpr = 128,
        ):
            tile = pto.alloc_tile(shape=[2, BLOCK], dtype=pto.f32)
            ptr = tile.as_ptr()
            offset = pto.const(0)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "type_system.mask_bitcast": _fixture(
        f"""
        @pto.jit(target="a5")
        def type_system_mask_bitcast_probe():
            mask_b8 = pto.pset_b8(pto.MaskPattern.ALL)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "quick_start.make_tensor_view": _fixture(
        f"""
        @pto.jit(target="a5")
        def quick_start_make_tensor_view_probe(
            A: pto.tensor_spec(rank=2, dtype=pto.f32),
        ):
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "quick_start.alloc_tile": _fixture(
        f"""
        @pto.jit(target="a5")
        def quick_start_alloc_tile_probe(
            *,
            BLOCK: pto.constexpr = 128,
        ):
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "quick_start.partition_view": _fixture(
        f"""
        @pto.jit(target="a5")
        def quick_start_partition_view_probe(
            A: pto.tensor_spec(rank=2, dtype=pto.f32),
        ):
            rows = A.shape[0]
            cols = A.shape[1]
            a_view = pto.make_tensor_view(A, shape=A.shape, strides=A.strides)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "quick_start.tile_io": _fixture(
        f"""
        @pto.jit(target="a5")
        def quick_start_tile_io_probe(
            A: pto.tensor_spec(rank=2, dtype=pto.f32),
            O: pto.tensor_spec(rank=2, dtype=pto.f32),
            *,
            BLOCK: pto.constexpr = 128,
        ):
            rows = A.shape[0]
            cols = A.shape[1]

            a_view = pto.make_tensor_view(A, shape=A.shape, strides=A.strides)
            o_view = pto.make_tensor_view(O, shape=O.shape, strides=O.strides)
            a_part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
            o_part = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
            a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "control_flow.basic_for": _fixture(
        f"""
        @pto.jit(target="a5")
        def control_flow_basic_for_probe(
            A: pto.tensor_spec(rank=2, dtype=pto.f32),
            O: pto.tensor_spec(rank=2, dtype=pto.f32),
            *,
            BLOCK: pto.constexpr = 8,
        ):
            start = pto.const(0, dtype=pto.i32)
            stop = pto.const(BLOCK, dtype=pto.i32)
            step = 1
            rows = A.shape[0]
            cols = A.shape[1]
            a_view = pto.make_tensor_view(A, shape=A.shape, strides=A.strides)
            o_view = pto.make_tensor_view(O, shape=O.shape, strides=O.strides)
            tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "control_flow.compare_loops": _fixture(
        f"""
        @pto.jit(target="a5")
        def control_flow_compare_loops_probe(
            A: pto.tensor_spec(rank=2, dtype=pto.f32),
            O: pto.tensor_spec(rank=2, dtype=pto.f32),
            *,
            BLOCK: pto.constexpr = 8,
        ):
            rows = A.shape[0]
            cols = A.shape[1]
            num_blocks = rows
            a_view = pto.make_tensor_view(A, shape=A.shape, strides=A.strides)
            o_view = pto.make_tensor_view(O, shape=O.shape, strides=O.strides)
            tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "control_flow.nested_loops": _fixture(
        f"""
        @pto.jit(target="a5")
        def control_flow_nested_loops_probe(
            A: pto.tensor_spec(rank=2, dtype=pto.f32),
            *,
            BLOCK: pto.constexpr = 8,
        ):
            rows = A.shape[0]
            cols = A.shape[1]
            tile = pto.alloc_tile(shape=[2, BLOCK], dtype=pto.f32, valid_shape=[rows, cols])
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "control_flow.carry_pingpong": _fixture(
        f"""
        @pto.jit(target="a5")
        def control_flow_carry_pingpong_probe(
            *,
            Br: pto.constexpr = 16,
            num_blocks: pto.constexpr = 4,
        ):
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "scalar_ops.tile_access": _fixture(
        f"""
        @pto.jit(target="a5")
        def scalar_ops_tile_access_probe():
            scalar = pto.scalar
            tile = pto.alloc_tile(shape=[1, 8], dtype=pto.f32, valid_shape=[1, 4])
            row = 0
            col = 0
            value = pto.const(1.0, dtype=pto.f32)
            ptr = tile.as_ptr()
            offset = 0
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "scalar_ops.simt_pointer": _fixture(
        f"""
        @pto.jit(target="a5")
        def scalar_ops_simt_pointer_probe():
            scalar = pto.scalar
            meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 4])
            meta_ptr = meta_tile.as_ptr()
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "scalar_ops.math": _fixture(
        f"""
        @pto.jit(target="a5")
        def scalar_ops_math_probe():
            scalar = pto.scalar
            tile = pto.alloc_tile(shape=[1, 8], dtype=pto.f32, valid_shape=[1, 4])
            alpha = pto.scalar.load(tile[0, 0])
            o_prev = pto.scalar.load(tile[0, 1])
            beta = pto.scalar.load(tile[0, 2])
            pv_val = pto.scalar.load(tile[0, 3])
            m_prev = pto.scalar.load(tile[0, 0])
            row_max = pto.scalar.load(tile[0, 1])
            l_prev = pto.scalar.load(tile[0, 2])
            m_next = pto.scalar.load(tile[0, 3])
            val = pto.scalar.load(tile[0, 0])
            threshold = pto.const(0.0, dtype=pto.f32)
            N = pto.const(16, dtype=pto.i32)
            BLOCK = 8
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "scalar_ops.pointer_sources": _fixture(
        f"""
        @pto.jit(target="a5")
        def scalar_ops_pointer_sources_probe(
            A: pto.tensor_spec(rank=2, dtype=pto.f32),
            *,
            BLOCK: pto.constexpr = 8,
        ):
            rows = A.shape[0]
            cols = A.shape[1]
            a_view = pto.make_tensor_view(A, shape=A.shape, strides=A.strides)
            partition = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
            tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "scalar_ops.pointer_manip": _fixture(
        f"""
        @pto.jit(target="a5")
        def scalar_ops_pointer_manip_probe():
            base_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 4])
            base_ptr = base_tile.as_ptr()
            addr = pto.const(0, dtype=pto.i64)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "data_movement.tload": _fixture(
        f"""
        @pto.jit(target="a5")
        def data_movement_tload_probe(
            A: pto.tensor_spec(rank=2, dtype=pto.f32),
            *,
            BLOCK: pto.constexpr = 128,
        ):
            rows = A.shape[0]
            cols = A.shape[1]
            offset = 0
            a_view = pto.make_tensor_view(A, shape=A.shape, strides=A.strides)
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "compute_ops.vector_compute": _fixture(
        f"""
        @pto.simd
        def compute_ops_vector_helper(inp_tile: pto.Tile, out_tile: pto.Tile, row: pto.index):
            col_mask = pto.make_mask(pto.f32, pto.const(16, dtype=pto.i32))
            s_row = pto.vlds(inp_tile[row, 0:])
            p_row = pto.vexp(s_row, col_mask)
            m_next = pto.vcgmax(s_row, col_mask)
            {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def compute_ops_vector_probe(*, BLOCK: pto.constexpr = 128):
            inp_tile = pto.alloc_tile(shape=[2, BLOCK], dtype=pto.f32)
            out_tile = pto.alloc_tile(shape=[2, BLOCK], dtype=pto.f32)
            with pto.for_(0, 1, step=1) as row:
                compute_ops_vector_helper(inp_tile, out_tile, row)
        """
    ),
    "mask_ops.creation": _fixture(
        f"""
        @pto.jit(target="a5")
        def mask_ops_creation_probe():
            remained = pto.const(16, dtype=pto.i32)
            seed = pto.pset_b32("PAT_ALL")
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "sync_ops.basic": _fixture(
        f"""
        @pto.jit(target="a5")
        def sync_ops_basic_probe():
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "flash_attention.l1_tensor_views": _fixture(
        f"""
        @pto.jit(target="a5")
        def flash_attention_l1_tensor_views_probe(
            Q: pto.tensor_spec(rank=4, dtype=pto.f32),
            K: pto.tensor_spec(rank=4, dtype=pto.f32),
            V: pto.tensor_spec(rank=4, dtype=pto.f32),
            O: pto.tensor_spec(rank=4, dtype=pto.f32),
            *,
            BLOCK_Q: pto.constexpr = 128,
            BLOCK_KV: pto.constexpr = 128,
            CAUSAL: pto.constexpr = False,
            NUM_STAGES: pto.constexpr = 2,
        ):
            batch = Q.shape[0]
            seq_q = Q.shape[1]
            seq_k = K.shape[1]
            heads = Q.shape[2]
            dim = Q.shape[3]
            Br = BLOCK_Q
            Bc = BLOCK_KV
            D = dim
            full_br = Br
            full_bc = Bc
            one = 1
            scalar = pto.scalar
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "flash_attention.l1_partitions": _fixture(
        f"""
        @pto.jit(target="a5")
        def flash_attention_l1_partitions_probe(
            Q: pto.tensor_spec(rank=4, dtype=pto.f32),
            K: pto.tensor_spec(rank=4, dtype=pto.f32),
            V: pto.tensor_spec(rank=4, dtype=pto.f32),
            O: pto.tensor_spec(rank=4, dtype=pto.f32),
            *,
            BLOCK_Q: pto.constexpr = 128,
            BLOCK_KV: pto.constexpr = 128,
            CAUSAL: pto.constexpr = False,
            NUM_STAGES: pto.constexpr = 2,
        ):
            batch = Q.shape[0]
            seq_q = Q.shape[1]
            seq_k = K.shape[1]
            heads = Q.shape[2]
            dim = Q.shape[3]
            q_view = pto.make_tensor_view(Q, shape=[batch, seq_q, heads, dim], strides=Q.strides)
            k_view = pto.make_tensor_view(K, shape=[batch, seq_k, heads, dim], strides=K.strides)
            v_view = pto.make_tensor_view(V, shape=[batch, seq_k, heads, dim], strides=V.strides)
            o_view = pto.make_tensor_view(O, shape=[batch, seq_q, heads, dim], strides=O.strides)
            block_idx = pto.get_block_idx()
            batch_idx = block_idx // heads
            head_idx = block_idx % heads
            scalar = pto.scalar
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "flash_attention.l1_tiles": _fixture(
        f"""
        @pto.jit(target="a5")
        def flash_attention_l1_tiles_probe(
            Q: pto.tensor_spec(rank=4, dtype=pto.f32),
            K: pto.tensor_spec(rank=4, dtype=pto.f32),
            V: pto.tensor_spec(rank=4, dtype=pto.f32),
            O: pto.tensor_spec(rank=4, dtype=pto.f32),
            *,
            BLOCK_Q: pto.constexpr = 128,
            BLOCK_KV: pto.constexpr = 128,
        ):
            batch = Q.shape[0]
            seq_q = Q.shape[1]
            seq_k = K.shape[1]
            heads = Q.shape[2]
            dim = Q.shape[3]
            Br = BLOCK_Q
            Bc = BLOCK_KV
            D = dim
            full_br = Br
            full_bc = Bc
            one = 1
            scalar = pto.scalar
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "flash_attention.ukernel_phase": _fixture(
        f"""
        @pto.cube
        def qk_matmul(
            q_tile: pto.Tile,
            k_tile: pto.Tile,
            q_l0a: pto.Tile,
            rhs_l0b: pto.Tile,
            qk_acc_tile: pto.Tile,
            s_tile: pto.Tile,
        ):
            return


        @pto.simd
        def online_softmax_rows(
            s_tile: pto.Tile,
            p_tile: pto.Tile,
            m_prev_tile: pto.Tile,
            l_prev_tile: pto.Tile,
            m_next_tile: pto.Tile,
            l_next_tile: pto.Tile,
            alpha_tile: pto.Tile,
            beta_tile: pto.Tile,
            row_start: pto.i32,
            row_stop: pto.i32,
            valid_cols: pto.i32,
        ):
            return


        @pto.cube
        def pv_matmul(
            p_tile: pto.Tile,
            v_tile: pto.Tile,
            p_l0a: pto.Tile,
            rhs_l0b: pto.Tile,
            pv_acc_tile: pto.Tile,
            pv_tile: pto.Tile,
        ):
            return


        @pto.simt
        def blend_output_rows(
            o_prev_tile: pto.Tile,
            pv_tile: pto.Tile,
            alpha_tile: pto.Tile,
            beta_tile: pto.Tile,
            o_next_tile: pto.Tile,
            row_start: pto.i32,
            row_stop: pto.i32,
            valid_dim: pto.i32,
        ):
            return


        @pto.simt
        def materialize_tile_bounds(meta_ptr, valid_rows: pto.i32, valid_cols: pto.i32):
            scalar = pto.scalar
            scalar.store(0, meta_ptr + 0)
            scalar.store(valid_rows, meta_ptr + 1)
            scalar.store(valid_cols, meta_ptr + 2)


        @pto.ukernel
        def flash_attention_ukernel_phase(
            q_tile: pto.Tile,
            k_part: pto.PartitionTensorView,
            v_part: pto.PartitionTensorView,
            k_tile: pto.Tile,
            v_tile: pto.Tile,
            o_prev_tile: pto.Tile,
            o_next_tile: pto.Tile,
            m_prev_tile: pto.Tile,
            l_prev_tile: pto.Tile,
            m_next_tile: pto.Tile,
            l_next_tile: pto.Tile,
            s_tile: pto.Tile,
            p_tile: pto.Tile,
            pv_tile: pto.Tile,
            alpha_tile: pto.Tile,
            beta_tile: pto.Tile,
            q_l0a: pto.Tile,
            p_l0a: pto.Tile,
            rhs_l0b: pto.Tile,
            qk_acc_tile: pto.Tile,
            pv_acc_tile: pto.Tile,
            meta_ptr,
        ):
            scalar = pto.scalar
            row_start = pto.const(0, dtype=pto.i32)
            row_stop = pto.const(16, dtype=pto.i32)
            valid_cols = pto.const(16, dtype=pto.i32)
            {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def flash_attention_ukernel_phase_probe(
            K: pto.tensor_spec(rank=4, dtype=pto.f32),
            V: pto.tensor_spec(rank=4, dtype=pto.f32),
            *,
            BLOCK_Q: pto.constexpr = 16,
            BLOCK_KV: pto.constexpr = 16,
        ):
            Br = BLOCK_Q
            Bc = BLOCK_KV
            D = 16
            one = 1
            k_view = pto.make_tensor_view(K, shape=K.shape, strides=K.strides)
            v_view = pto.make_tensor_view(V, shape=V.shape, strides=V.strides)
            k_part = pto.partition_view(k_view, offsets=[0, 0, 0, 0], sizes=[1, Bc, 1, D])
            v_part = pto.partition_view(v_view, offsets=[0, 0, 0, 0], sizes=[1, Bc, 1, D])
            q_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[Br, D])
            k_tile = pto.alloc_tile(shape=[Bc, D], dtype=pto.f32, valid_shape=[Bc, D])
            v_tile = pto.alloc_tile(shape=[Bc, D], dtype=pto.f32, valid_shape=[Bc, D])
            o_prev_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[Br, D])
            o_next_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[Br, D])
            m_prev_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[Br, one], blayout="ColMajor")
            l_prev_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[Br, one], blayout="ColMajor")
            m_next_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[Br, one], blayout="ColMajor")
            l_next_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[Br, one], blayout="ColMajor")
            alpha_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[Br, one], blayout="ColMajor")
            beta_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[Br, one], blayout="ColMajor")
            s_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, valid_shape=[Br, Bc])
            p_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, valid_shape=[Br, Bc])
            pv_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[Br, D])
            q_l0a = pto.alloc_tile(shape=[Br, D], dtype=pto.f16, memory_space=pto.MemorySpace.LEFT, valid_shape=[Br, D])
            p_l0a = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f16, memory_space=pto.MemorySpace.LEFT, valid_shape=[Br, Bc])
            rhs_l0b = pto.alloc_tile(shape=[Bc, D], dtype=pto.f16, memory_space=pto.MemorySpace.RIGHT, valid_shape=[Bc, D])
            qk_acc_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, valid_shape=[Br, Bc])
            pv_acc_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, valid_shape=[Br, D])
            meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 3])
            meta_ptr = meta_tile.as_ptr()
            flash_attention_ukernel_phase(
                q_tile, k_part, v_part, k_tile, v_tile,
                o_prev_tile, o_next_tile,
                m_prev_tile, l_prev_tile, m_next_tile, l_next_tile,
                s_tile, p_tile, pv_tile,
                alpha_tile, beta_tile,
                q_l0a, p_l0a, rhs_l0b,
                qk_acc_tile, pv_acc_tile,
                meta_ptr,
            )
        """
    ),
    "flash_attention.online_softmax_loop": _fixture(
        f"""
        @pto.simd
        def flash_attention_online_softmax_loop_helper(
            s_tile: pto.Tile,
            p_tile: pto.Tile,
            m_prev_tile: pto.Tile,
            l_prev_tile: pto.Tile,
            m_next_tile: pto.Tile,
            l_next_tile: pto.Tile,
            alpha_tile: pto.Tile,
            beta_tile: pto.Tile,
            row_start: pto.i32,
            row_stop: pto.i32,
            valid_cols: pto.i32,
        ):
            scalar = pto.scalar
            {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def flash_attention_online_softmax_loop_probe(*, BLOCK: pto.constexpr = 16):
            one = 1
            s_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            p_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            m_prev_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            l_prev_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            m_next_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            l_next_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            alpha_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            beta_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            flash_attention_online_softmax_loop_helper(
                s_tile, p_tile,
                m_prev_tile, l_prev_tile,
                m_next_tile, l_next_tile,
                alpha_tile, beta_tile,
                0, 2, BLOCK,
            )
        """
    ),
    "flash_attention.online_softmax_compute": _fixture(
        f"""
        @pto.simd
        def flash_attention_online_softmax_compute_helper(
            s_tile: pto.Tile,
            p_tile: pto.Tile,
            m_prev_tile: pto.Tile,
            l_prev_tile: pto.Tile,
            m_next_tile: pto.Tile,
            l_next_tile: pto.Tile,
            alpha_tile: pto.Tile,
            beta_tile: pto.Tile,
            row_start: pto.i32,
            row_stop: pto.i32,
            valid_cols: pto.i32,
        ):
            scalar = pto.scalar
            with pto.for_(row_start, row_stop, step=1) as row:
                col_mask = pto.make_mask(pto.f32, valid_cols)
                s_row = pto.vlds(s_tile[row, 0:])
                m_prev = scalar.load(m_prev_tile[row, 0])
                l_prev = scalar.load(l_prev_tile[row, 0])
            {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def flash_attention_online_softmax_compute_probe(*, BLOCK: pto.constexpr = 16):
            one = 1
            s_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            p_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            m_prev_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            l_prev_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            m_next_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            l_next_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            alpha_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            beta_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            flash_attention_online_softmax_compute_helper(
                s_tile, p_tile,
                m_prev_tile, l_prev_tile,
                m_next_tile, l_next_tile,
                alpha_tile, beta_tile,
                0, 2, BLOCK,
            )
        """
    ),
    "flash_attention.online_softmax_store": _fixture(
        f"""
        @pto.simd
        def flash_attention_online_softmax_store_helper(
            s_tile: pto.Tile,
            p_tile: pto.Tile,
            m_prev_tile: pto.Tile,
            l_prev_tile: pto.Tile,
            m_next_tile: pto.Tile,
            l_next_tile: pto.Tile,
            alpha_tile: pto.Tile,
            beta_tile: pto.Tile,
            row_start: pto.i32,
            row_stop: pto.i32,
            valid_cols: pto.i32,
        ):
            scalar = pto.scalar
            with pto.for_(row_start, row_stop, step=1) as row:
                col_mask = pto.make_mask(pto.f32, valid_cols)
                p_row = pto.vexp(pto.vlds(s_tile[row, 0:]), col_mask)
                m_next = pto.scalar.load(m_prev_tile[row, 0])
                l_next = pto.scalar.load(l_prev_tile[row, 0])
                alpha = pto.scalar.load(alpha_tile[row, 0])
                beta = pto.scalar.load(beta_tile[row, 0])
            {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def flash_attention_online_softmax_store_probe(*, BLOCK: pto.constexpr = 16):
            one = 1
            s_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            p_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            m_prev_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            l_prev_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            m_next_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            l_next_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            alpha_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            beta_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            flash_attention_online_softmax_store_helper(
                s_tile, p_tile,
                m_prev_tile, l_prev_tile,
                m_next_tile, l_next_tile,
                alpha_tile, beta_tile,
                0, 2, BLOCK,
            )
        """
    ),
    "flash_attention.simt_materialize": _fixture(
        f"""
        scalar = pto.scalar
        {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def flash_attention_simt_materialize_probe():
            meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 3])
            meta_ptr = meta_tile.as_ptr()
            valid_rows = pto.const(1, dtype=pto.i32)
            valid_cols = pto.const(2, dtype=pto.i32)
            materialize_tile_bounds(meta_ptr, valid_rows, valid_cols)
        """
    ),
    "flash_attention.simt_blend": _fixture(
        f"""
        scalar = pto.scalar
        {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def flash_attention_simt_blend_probe(*, BLOCK: pto.constexpr = 8):
            one = 1
            o_prev_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            pv_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            alpha_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            beta_tile = pto.alloc_tile(shape=[8, 1], dtype=pto.f32, valid_shape=[2, one], blayout="ColMajor")
            o_next_tile = pto.alloc_tile(shape=[8, BLOCK], dtype=pto.f32, valid_shape=[2, BLOCK])
            blend_output_rows(
                o_prev_tile,
                pv_tile,
                alpha_tile,
                beta_tile,
                o_next_tile,
                pto.const(0, dtype=pto.i32),
                pto.const(2, dtype=pto.i32),
                pto.const(BLOCK, dtype=pto.i32),
            )
        """
    ),
    "gemm.cube_helper": _fixture(
        f"""
        @pto.cube
        def gemm_tile(
            a_tile: pto.Tile,
            b_tile: pto.Tile,
            o_tile: pto.Tile,
            a_l0a: pto.Tile,
            b_l0b: pto.Tile,
            o_acc: pto.Tile,
        ):
            {SNIPPET_PLACEHOLDER}


        @pto.jit(target="a5")
        def gemm_tile_probe(*, BLOCK_M: pto.constexpr = 64, BLOCK_K: pto.constexpr = 64, BLOCK_N: pto.constexpr = 64):
            a_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_K], dtype=pto.f32, valid_shape=[BLOCK_M, BLOCK_K])
            b_tile = pto.alloc_tile(shape=[BLOCK_K, BLOCK_N], dtype=pto.f32, valid_shape=[BLOCK_K, BLOCK_N])
            o_tile = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32, valid_shape=[BLOCK_M, BLOCK_N])
            a_l0a = pto.alloc_tile(shape=[BLOCK_M, BLOCK_K], dtype=pto.f32, memory_space=pto.MemorySpace.LEFT, valid_shape=[BLOCK_M, BLOCK_K])
            b_l0b = pto.alloc_tile(shape=[BLOCK_K, BLOCK_N], dtype=pto.f32, memory_space=pto.MemorySpace.RIGHT, valid_shape=[BLOCK_K, BLOCK_N])
            o_acc = pto.alloc_tile(shape=[BLOCK_M, BLOCK_N], dtype=pto.f32, memory_space=pto.MemorySpace.ACC, valid_shape=[BLOCK_M, BLOCK_N])
            gemm_tile(a_tile, b_tile, o_tile, a_l0a, b_l0b, o_acc)
        """
    ),
    "gemm.jit_kernel": _fixture(
        f"""
        @pto.cube
        def gemm_tile(
            a_tile: pto.Tile,
            b_tile: pto.Tile,
            o_tile: pto.Tile,
            a_l0a: pto.Tile,
            b_l0b: pto.Tile,
            o_acc: pto.Tile,
        ):
            m = a_tile.valid_shape[0]
            k = a_tile.valid_shape[1]
            n = b_tile.valid_shape[0]
            pto.mte_l1_l0a(a_tile.as_ptr(), a_l0a.as_ptr(), m, k)
            pto.mte_l1_l0b(b_tile.as_ptr(), b_l0b.as_ptr(), k, n, transpose=True)
            pto.mad(a_l0a.as_ptr(), b_l0b.as_ptr(), o_acc.as_ptr(), m, n, k)
            pto.mte_l0c_ub(o_acc.as_ptr(), o_tile.as_ptr(), m, n, n, n, 0)


        {SNIPPET_PLACEHOLDER}
        """
    ),
}
