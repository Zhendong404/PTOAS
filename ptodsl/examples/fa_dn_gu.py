# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
PTODSL translation of ``ptodsl/examples/fa_dn_gu.cpp``.

This file keeps the C++ helper split:

- ``pto_macro_fa_gu``: running update ``O = O * exp_max + est_sv_tile``
- ``pto_macro_fa_gu_last``: running update followed by row-wise normalization
- ``pto_macro_fa_gu_single_and_last_tile``: final normalization only

The current PTODSL version is intentionally close to the C++ structure while
staying on the public DSL surface. A couple of low-level details are still
approximated:

- The C++ ``TRESHAPE(... ColMajor ...)`` step is represented by assuming the
  reduced tiles are already authored as row-reduced ``[BR, 1]`` PTODSL tiles.
- The C++ pointer choreography that relies on post-update addressing is lowered
  here with explicit row/column offsets, which preserves the math and emitted
  vector ops without depending on the exact micro-address contract.
"""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "ptodsl" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            break
    else:
        raise RuntimeError(
            "Unable to locate the PTODSL Python package root from fa_dn_gu.py"
        )

from ptodsl import pto

DIST = {
    1: "NORM_B8",
    2: "NORM_B16",
    4: "NORM_B32",
}


def _repeat_plan(tile: pto.Tile) -> tuple[int, int, int]:
    rows, cols = tile.shape
    lanes = pto.elements_per_vreg(tile.dtype)
    repeats = (cols + lanes - 1) // lanes
    return rows, cols, lanes * 4 if tile.dtype == pto.f32 else lanes, repeats


def _reshape_reduced_tile_to_col(tile: pto.Tile) -> pto.Tile:
    rows, cols = tile.shape
    if cols == 1:
        return pto.tile.reshape(tile, shape=[rows, 1], blayout="ColMajor")
    return pto.tile.reshape(tile, shape=[cols, 1], blayout="ColMajor")

def pto_macro_fa_gu(
    prev_sv_tile: pto.Tile,
    est_sv_tile: pto.Tile,
    exp_max: pto.Tile,
):
    exp_max_col = _reshape_reduced_tile_to_col(exp_max)

    ubM, ubN = prev_sv_tile.shape
    lanes = pto.elements_per_vreg(prev_sv_tile.dtype)
    _, stride = exp_max_col.shape

    prev_ptr = prev_sv_tile.as_ptr()
    est_ptr = est_sv_tile.as_ptr()
    exp_max_ptr = exp_max_col.as_ptr()

    with pto.for_(0, ubM, step=1) as row:
        exp_row = pto.vlds(exp_max_ptr, row * stride, dist="BRC_B32")
        col_loop = pto.for_(0, ubN, step=lanes).carry(
            prev_ptr=prev_ptr,
            est_ptr=est_ptr,
            remained=ubN,
        )
        with col_loop:
            prev_ptr = col_loop.prev_ptr
            est_ptr = col_loop.est_ptr
            remained = col_loop.remained
            mask, remained = pto.make_mask(prev_sv_tile.dtype, remained)
            prev_vec, prev_ptr = pto.vlds(prev_ptr, 0, dist="NORM", post_update="ON")
            est_vec, est_ptr = pto.vlds(est_ptr, lanes, dist="NORM", post_update="ON")
            out_vec = pto.vadd(pto.vmul(prev_vec, exp_row, mask), est_vec, mask)
            prev_ptr = pto.vsts(out_vec, prev_ptr, lanes, mask, dist=DIST[pto.bytewidth(prev_sv_tile.dtype)], post_update="ON")
            col_loop.update(
                prev_ptr=prev_ptr,
                est_ptr=est_ptr,
                remained=remained,
            )


@pto.simd
def pto_macro_fa_gu_last(
    prev_sv_tile: pto.Tile,
    est_sv_tile: pto.Tile,
    exp_max: pto.Tile,
    new_global_sum: pto.Tile,
):
    exp_max_col = _reshape_reduced_tile_to_col(exp_max)
    new_global_sum_col = _reshape_reduced_tile_to_col(new_global_sum)

    ubM, ubN = prev_sv_tile.shape
    lanes = pto.elements_per_vreg(prev_sv_tile.dtype)
    _, stride = exp_max_col.shape

    prev_ptr = prev_sv_tile.as_ptr()
    est_ptr = est_sv_tile.as_ptr()
    exp_max_ptr = exp_max_col.as_ptr()
    new_global_sum_ptr = new_global_sum_col.as_ptr()

    with pto.for_(0, ubM, step=1) as row:
        exp_row = pto.vlds(exp_max_ptr, row * stride, dist="BRC_B32")
        sum_row = pto.vlds(new_global_sum_ptr, row * stride, dist="BRC_B32")
        col_loop = pto.for_(0, ubN, step=lanes).carry(
            prev_ptr=prev_ptr,
            est_ptr=est_ptr,
            remained=ubN,
        )
        with col_loop:
            prev_ptr = col_loop.prev_ptr
            est_ptr = col_loop.est_ptr
            remained = col_loop.remained
            mask, remained = pto.make_mask(prev_sv_tile.dtype, remained)
            prev_vec, prev_ptr = pto.vlds(prev_ptr, 0, dist="NORM", post_update="ON")
            est_vec, est_ptr = pto.vlds(est_ptr, lanes, dist="NORM", post_update="ON")
            out_vec = pto.vadd(pto.vmul(prev_vec, exp_row, mask), est_vec, mask)
            out_vec = pto.vdiv(out_vec, sum_row, mask)
            prev_ptr = pto.vsts(out_vec, prev_ptr, lanes, mask, dist=DIST[pto.bytewidth(prev_sv_tile.dtype)], post_update="ON")
            col_loop.update(
                prev_ptr=prev_ptr,
                est_ptr=est_ptr,
                remained=remained,
            )


@pto.simd
def pto_macro_fa_gu_single_and_last_tile(
    sv_tile: pto.Tile,
    new_global_sum: pto.Tile,
):
    new_global_sum_col = _reshape_reduced_tile_to_col(new_global_sum)
    pto.tile.rowexpanddiv(sv_tile, new_global_sum_col, sv_tile)


def pto_macro_fa_gu_dispatch(
    prev_sv_tile: pto.Tile,
    est_sv_tile: pto.Tile,
    exp_max: pto.Tile,
    new_global_sum: pto.Tile,
    *,
    last_tile: pto.constexpr = False,
    single_last_tile: pto.constexpr = False,
):
    if single_last_tile:
        pto_macro_fa_gu_single_and_last_tile(prev_sv_tile, new_global_sum)
    elif last_tile:
        pto_macro_fa_gu_last(prev_sv_tile, est_sv_tile, exp_max, new_global_sum)
    else:
        pto_macro_fa_gu(prev_sv_tile, est_sv_tile, exp_max)


@pto.jit(target="a5", mode="explicit")
def fa_dn_gu_wrapper(
    *,
    BR: pto.constexpr = 8,
    BC: pto.constexpr = 64,
    LAST_TILE: pto.constexpr = False,
    SINGLE_LAST_TILE: pto.constexpr = False,
):
    prev_sv_tile = pto.alloc_tile(shape=[BR, BC], dtype=pto.f32, valid_shape=[BR, BC])
    est_sv_tile = pto.alloc_tile(shape=[BR, BC], dtype=pto.f32, valid_shape=[BR, BC])
    exp_max = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")
    new_global_sum = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")

    pto_macro_fa_gu_dispatch(
        prev_sv_tile,
        est_sv_tile,
        exp_max,
        new_global_sum,
        last_tile=LAST_TILE,
        single_last_tile=SINGLE_LAST_TILE,
    )


__all__ = [
    "pto_macro_fa_gu",
    "pto_macro_fa_gu_last",
    "pto_macro_fa_gu_single_and_last_tile",
    "pto_macro_fa_gu_dispatch",
    "fa_dn_gu_wrapper",
]


def main() -> None:
    compiled = fa_dn_gu_wrapper.compile(
        BR=8,
        BC=64,
        LAST_TILE=False,
        SINGLE_LAST_TILE=False,
    )
    print(compiled.mlir_text())


if __name__ == "__main__":
    main()
