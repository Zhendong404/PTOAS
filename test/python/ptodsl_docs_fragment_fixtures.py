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
}
