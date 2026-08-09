#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tload/tload.pto:
# GM -> tile -> GM round trips for ND / DN / NZ vector tiles. Each kernel only
# performs pto.tile.load + pto.tile.store, so the case validates the DMA layout
# path directly (including padded valid_shape tiles and f8e4m3/hif8 payloads).
#
# Padded-tile metadata from the legacy .pto is preserved: alloc_tile(pad=...)
# maps the legacy tile_buf pad=1/2/3 encoding (Zero/Max/Min) onto the PTODSL
# PadValue surface. Per the TileLib ISA, tstore only writes the tile's valid
# region, so the pad attribute never reaches GM; the harness zero-initializes
# the output buffer, so pad-region golden stays zero (the legacy golden_fill
# 0.0/FLT_MAX/FLT_MIN placeholders were only meaningful in the legacy
# uninitialized-buffer compare, which sliced them away).

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


# One static kernel per legacy case.  View shapes/strides reproduce the legacy
# 5D make_tensor_view metadata reduced to 2D (ND/DN) or kept 5D (NZ fractal).
# ``transfer_sizes`` is only needed for the 5D NZ views where the partition
# rank cannot be inferred from the rank-2 tile.
CASE_SPECS = [
    {
        "case_name": "tload_nd_f32_128x128",
        "legacy_name": "case_float_GT_128_128_VT_128_128_BLK1",
        "kind": "nd",
        "np_dtype": np.float32,
        "pto_dtype": pto.f32,
        "shape": (128, 128),
        "valid_shape": None,
        "view_shape": [128, 128],
        "view_strides": [128, 1],
        "tile_kwargs": {},
        "eps": 1e-6,
    },
    {
        "case_name": "tload_nd_f32_128x127_pad_max",
        "legacy_name": "case_float_GT_128_127_VT_128_128_BLK1_PADMAX",
        "kind": "nd_pad",
        "np_dtype": np.float32,
        "pto_dtype": pto.f32,
        "shape": (128, 128),
        "valid_shape": (128, 127),
        "view_shape": [128, 128],
        "view_strides": [128, 1],
        "tile_kwargs": {"pad": "Max"},
        "eps": 1e-6,
    },
    {
        "case_name": "tload_nd_i16_128x127_pad_max",
        "legacy_name": "case_s16_GT_128_127_VT_128_128_BLK1_PADMAX",
        "kind": "nd_pad",
        "np_dtype": np.int16,
        "pto_dtype": pto.i16,
        "shape": (128, 128),
        "valid_shape": (128, 127),
        "view_shape": [128, 128],
        "view_strides": [128, 1],
        "tile_kwargs": {"pad": "Max"},
        "eps": 0,
    },
    {
        "case_name": "tload_nd_u8_128x127_pad_min",
        "legacy_name": "case_u8_GT_128_127_VT_128_128_BLK1_PADMIN",
        "kind": "nd_pad",
        "np_dtype": np.uint8,
        "pto_dtype": pto.ui8,
        "shape": (128, 128),
        "valid_shape": (128, 127),
        "view_shape": [128, 128],
        "view_strides": [128, 1],
        "tile_kwargs": {"pad": "Min"},
        "eps": 0,
    },
    {
        "case_name": "tload_nd_f32_16x64",
        "legacy_name": "nd_f32_16x64",
        "kind": "nd",
        "np_dtype": np.float32,
        "pto_dtype": pto.f32,
        "shape": (16, 64),
        "valid_shape": None,
        "view_shape": [16, 64],
        "view_strides": [64, 1],
        "tile_kwargs": {},
        "eps": 1e-6,
    },
    {
        "case_name": "tload_dn_f32_16x64",
        "legacy_name": "dn_f32_16x64",
        "kind": "dn",
        "np_dtype": np.float32,
        "pto_dtype": pto.f32,
        "shape": (16, 64),
        "valid_shape": None,
        "view_shape": [16, 64],
        "view_strides": [1, 16],
        "tile_kwargs": {"blayout": "ColMajor"},
        "eps": 1e-6,
    },
    {
        "case_name": "tload_nz_f32_128x128",
        "legacy_name": "nz_f32_128x128",
        "kind": "nz",
        "np_dtype": np.float32,
        "pto_dtype": pto.f32,
        "shape": (128, 128),
        "valid_shape": None,
        "view_shape": [16, 1, 128, 1, 8],
        "view_strides": [1024, 1024, 8, 8, 1],
        "tile_kwargs": {"blayout": "ColMajor", "slayout": "RowMajor"},
        "transfer_sizes": [16, 1, 128, 1, 8],
        "eps": 1e-6,
    },
    {
        "case_name": "tload_nd_f8e4m3_16x64",
        "legacy_name": "nd_f8e4m3_16x64",
        "kind": "nd",
        "np_dtype": np.uint8,
        "pto_dtype": pto.f8e4m3,
        "shape": (16, 64),
        "valid_shape": None,
        "view_shape": [16, 64],
        "view_strides": [64, 1],
        "tile_kwargs": {},
        "eps": 0,
    },
    {
        "case_name": "tload_nd_hif8_16x64",
        "legacy_name": "nd_hif8_16x64",
        "kind": "nd",
        "np_dtype": np.uint8,
        "pto_dtype": pto.hif8,
        "shape": (16, 64),
        "valid_shape": None,
        "view_shape": [16, 64],
        "view_strides": [64, 1],
        "tile_kwargs": {},
        "eps": 0,
    },
    {
        "case_name": "tload_nd_pad_zero_f32_16x64",
        "legacy_name": "nd_pad_zero_f32_16x64",
        "kind": "nd_pad",
        "np_dtype": np.float32,
        "pto_dtype": pto.f32,
        "shape": (16, 64),
        "valid_shape": (16, 63),
        "view_shape": [16, 63],
        "view_strides": [64, 1],
        "tile_kwargs": {"pad": "Zero"},
        "eps": 1e-6,
    },
    {
        "case_name": "tload_dn_pad_max_f32_16x64",
        "legacy_name": "dn_pad_max_f32_16x64",
        "kind": "dn_pad",
        "np_dtype": np.float32,
        "pto_dtype": pto.f32,
        "shape": (16, 64),
        "valid_shape": (15, 64),
        "view_shape": [15, 64],
        "view_strides": [1, 16],
        "tile_kwargs": {"blayout": "ColMajor", "pad": "Max"},
        "eps": 1e-6,
    },
    {
        "case_name": "tload_nz_pad_min_f32_128x128",
        "legacy_name": "nz_pad_min_f32_128x128",
        "kind": "nz_pad",
        "np_dtype": np.float32,
        "pto_dtype": pto.f32,
        "shape": (128, 128),
        "valid_shape": (64, 128),
        "view_shape": [16, 1, 64, 1, 8],
        "view_strides": [1024, 1024, 8, 8, 1],
        "tile_kwargs": {"blayout": "ColMajor", "slayout": "RowMajor", "pad": "Min"},
        "transfer_sizes": [16, 1, 64, 1, 8],
        "eps": 1e-6,
    },
]


# ---------------------------------------------------------------------------
# Kernels: one @pto.jit auto-mode kernel per static shape/layout variant.
# ---------------------------------------------------------------------------

_kernels = {}
for _spec in CASE_SPECS:
    _shape = _spec["shape"]
    _valid_shape = _spec["valid_shape"]
    _view_shape = _spec["view_shape"]
    _view_strides = _spec["view_strides"]
    _tile_kwargs = _spec["tile_kwargs"]
    _transfer_sizes = _spec.get("transfer_sizes")
    _pto_dtype = _spec["pto_dtype"]
    _kernel_name = _spec["case_name"]

    def _make(
        shape=_shape,
        valid_shape=_valid_shape,
        view_shape=_view_shape,
        view_strides=_view_strides,
        tile_kwargs=_tile_kwargs,
        transfer_sizes=_transfer_sizes,
        dtype=_pto_dtype,
        kernel_name=_kernel_name,
    ):
        @pto.jit(name=kernel_name, target="a5")
        def _kernel(
            src_ptr: pto.ptr(dtype, "gm"),
            dst_ptr: pto.ptr(dtype, "gm"),
        ):
            src_view = pto.make_tensor_view(src_ptr, shape=view_shape, strides=view_strides)
            dst_view = pto.make_tensor_view(dst_ptr, shape=view_shape, strides=view_strides)
            if valid_shape is None:
                tile = pto.alloc_tile(shape=shape, dtype=dtype, **tile_kwargs)
            else:
                tile = pto.alloc_tile(
                    shape=shape,
                    dtype=dtype,
                    valid_shape=valid_shape,
                    **tile_kwargs,
                )
            if transfer_sizes is None:
                pto.tile.load(src_view, tile)
                pto.tile.store(tile, dst_view)
            else:
                offsets = [0] * len(transfer_sizes)
                pto.tile.load(src_view, tile, offsets=offsets, sizes=transfer_sizes)
                pto.tile.store(tile, dst_view, offsets=offsets, sizes=transfer_sizes)

        return _kernel

    _kernels[_spec["case_name"]] = _make()


# ---------------------------------------------------------------------------
# Host data: reproduce the legacy per-case RNG (st_common.setup_case_rng) and
# gen_data.py input construction.
# ---------------------------------------------------------------------------

def _make_input(case_name, legacy_name, np_dtype, shape):
    # Legacy seeded with the unprefixed case name, so data is bit-identical.
    np.random.seed(zlib.crc32(legacy_name.encode("utf-8")) & 0xFFFFFFFF)
    return np.random.randint(1, 17, size=shape).astype(np_dtype)


def _make_expected(src, kind, rows, cols, vr, vc):
    if kind == "nd_pad":
        # ND valid region round-trips; the padded column is never stored and
        # the device output buffer is zero-initialized by the harness.
        golden = np.zeros((rows, cols), dtype=src.dtype)
        golden[:vr, :vc] = src[:vr, :vc]
        return golden
    if kind == "dn_pad":
        # DN physical layout: each of the vc columns occupies `rows` contiguous
        # flat elements; only the first vr rows are stored by tstore.
        golden = np.zeros((rows, cols), dtype=src.dtype)
        flat_golden = golden.reshape(-1)
        flat_in = np.asarray(src, dtype=src.dtype).reshape(-1)
        for col in range(vc):
            start = rows * col
            flat_golden[start : start + vr] = flat_in[start : start + vr]
        return golden
    if kind == "nz_pad":
        # NZ physical layout: 8-row fragments; only the first
        # (vr // num_blocks) rows of each fragment are stored by tstore.
        golden = np.zeros((rows, cols), dtype=src.dtype)
        flat_golden = golden.reshape(-1)
        flat_in = np.asarray(src, dtype=src.dtype).reshape(-1)
        block_rows = 8
        block_size = block_rows * cols
        num_blocks = rows // block_rows
        valid_rows_per_block = vr // num_blocks
        for block in range(num_blocks):
            base = block * block_size
            valid_elems = valid_rows_per_block * cols
            flat_golden[base : base + valid_elems] = flat_in[base : base + valid_elems]
        return golden
    return np.asarray(src, dtype=src.dtype).copy()


CASES = []
for _spec in CASE_SPECS:
    _case_name = _spec["case_name"]
    _legacy_name = _spec["legacy_name"]
    _kind = _spec["kind"]
    _np_dtype = _spec["np_dtype"]
    _shape = _spec["shape"]
    _valid_shape = _spec["valid_shape"]
    _vr, _vc = _valid_shape if _valid_shape is not None else _shape
    _eps = _spec["eps"]

    CASES.append(
        golden_output_case(
            _case_name,
            _kernels[_case_name],
            inputs=lambda _case_name=_case_name, _legacy_name=_legacy_name,
            _np_dtype=_np_dtype, _shape=_shape: [
                _make_input(_case_name, _legacy_name, _np_dtype, _shape)
            ],
            expected=lambda src, _kind=_kind, _rows=_shape[0], _cols=_shape[1],
            _vr=_vr, _vc=_vc: _make_expected(src, _kind, _rows, _cols, _vr, _vc),
            rtol=_eps,
            atol=_eps,
        )
    )


auto_main(globals())
