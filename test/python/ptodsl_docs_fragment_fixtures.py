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
    "quick_start.make_tensor_view": _fixture(
        f"""
        from ptodsl import pto


        @pto.jit(target="a5")
        def quick_start_make_tensor_view_probe(
            A: pto.tensor_spec(rank=2, dtype=pto.f32),
        ):
            {SNIPPET_PLACEHOLDER}
        """
    ),
    "quick_start.alloc_tile": _fixture(
        f"""
        from ptodsl import pto


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
        from ptodsl import pto


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
        from ptodsl import pto


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
}
