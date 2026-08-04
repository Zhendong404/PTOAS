#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").

"""PTODSL migration of the legacy ``tsort32`` TileLang ST suite."""

from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


# The legacy launcher allocates a deliberately larger destination buffer.  The
# valid region contains interleaved (value, index) pairs: two f32 elements per
# pair, or four f16 elements per pair because the index is a u32 bit pattern.
CASE_SPECS = [
    ("f32_1x32", pto.f32, np.float32, (1, 32), (1, 32), None, (1, 128), (1, 32), (1, 64)),
    ("f32_1x64", pto.f32, np.float32, (1, 64), (1, 64), None, (1, 256), (1, 64), (1, 128)),
    ("f32_2x32", pto.f32, np.float32, (2, 32), (2, 32), None, (2, 128), (2, 32), (2, 64)),
    ("f32_16x32", pto.f32, np.float32, (16, 32), (16, 32), None, (16, 128), (16, 32), (16, 64)),
    ("f32_2x64_shared_idx", pto.f32, np.float32, (2, 64), (1, 64), None, (2, 256), (2, 64), (2, 128)),
    ("f32_16x64_shared_idx", pto.f32, np.float32, (16, 64), (1, 64), None, (16, 256), (16, 64), (16, 128)),
    ("f32_1x8192", pto.f32, np.float32, (1, 8192), (1, 8192), None, (1, 32768), (1, 8192), (1, 16384)),
    ("f32_2x13", pto.f32, np.float32, (2, 16), (2, 16), (1, 16), (2, 64), (2, 13), (2, 26)),
    ("f32_1x4164", pto.f32, np.float32, (1, 8192), (1, 8192), (1, 4168), (1, 32768), (1, 4164), (1, 8328)),
    ("f32_2x2084", pto.f32, np.float32, (2, 3072), (2, 3072), (1, 2088), (2, 12288), (2, 2084), (2, 4168)),
    ("f16_1x32", pto.f16, np.float16, (1, 32), (1, 32), None, (1, 128), (1, 32), (1, 128)),
    ("f16_4x64", pto.f16, np.float16, (4, 64), (4, 64), None, (4, 256), (4, 64), (4, 256)),
]


def _make_kernel(name, pto_dtype, src_shape, idx_shape, tmp_shape, dst_shape, src_valid, dst_valid):
    src_rows, src_cols = src_shape
    idx_rows, idx_cols = idx_shape
    dst_rows, dst_cols = dst_shape

    @pto.jit(name="tsort32_" + name, target="a5")
    def _kernel(src_ptr: pto.ptr(pto_dtype, "gm"), idx_ptr: pto.ptr(pto.ui32, "gm"), dst_ptr: pto.ptr(pto_dtype, "gm")):
        src_view = pto.make_tensor_view(src_ptr, shape=[src_rows, src_cols], strides=[src_cols, 1])
        idx_view = pto.make_tensor_view(idx_ptr, shape=[idx_rows, idx_cols], strides=[idx_cols, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[dst_rows, dst_cols], strides=[dst_cols, 1])
        src_tile = pto.alloc_tile(shape=[src_rows, src_cols], dtype=pto_dtype, valid_shape=list(src_valid))
        idx_tile = pto.alloc_tile(shape=[idx_rows, idx_cols], dtype=pto.ui32, valid_shape=[idx_rows, idx_cols])
        dst_tile = pto.alloc_tile(shape=[dst_rows, dst_cols], dtype=pto_dtype, valid_shape=list(dst_valid))
        tmp_tile = None if tmp_shape is None else pto.alloc_tile(shape=list(tmp_shape), dtype=pto_dtype)
        pto.tile.load(src_view, src_tile)
        pto.tile.load(idx_view, idx_tile)
        pto.tile.sort32(src_tile, idx_tile, dst_tile, tmp=tmp_tile)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


def _make_inputs(name, np_dtype, src_shape, idx_shape):
    np.random.seed(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)
    src = np.random.randint(1, 100, size=src_shape).astype(np_dtype)
    src_cols = src_shape[1]
    idx = np.arange(src_cols, dtype=np.uint32).reshape(1, src_cols)
    if idx_shape[0] != 1:
        idx = np.tile(idx, (idx_shape[0], 1))
    return [src, idx]


def _make_expected(src, idx, np_dtype, src_valid, dst_shape, dst_valid):
    rows, valid_cols = src_valid
    stride = 2 if np_dtype == np.float32 else 4
    result = np.zeros(dst_shape, dtype=np_dtype)
    for row in range(rows):
        for start in range(0, valid_cols, 32):
            end = min(start + 32, valid_cols)
            values = src[row, start:end]
            indices = idx[0 if idx.shape[0] == 1 else row, start:end]
            if end - start < 32:
                # Preserve the legacy gen_data.py padding exactly.  The
                # legacy source uses the numeric conversion
                # np.float32(0xFF800000) (rather than a bit reinterpretation),
                # and the simulator/template therefore places these values
                # first under descending sort.
                pad_value = np.float16(0xFC00) if np_dtype == np.float16 else np.float32(0xFF800000)
                padded_values = np.full(32, pad_value, dtype=np_dtype)
                padded_values[: end - start] = values
                padded_indices = np.zeros(32, dtype=np.uint32)
                padded_indices[: end - start] = indices
                order = np.argsort(-padded_values)
                sorted_values = padded_values[order]
                sorted_indices = padded_indices[order]
            else:
                order = np.argsort(-values)
                sorted_values = values[order]
                sorted_indices = indices[order]
            out = start * stride
            for i, value in enumerate(sorted_values):
                result[row, out + i * stride] = value
                bits = np.asarray(sorted_indices[i], dtype=np.uint32)
                if np_dtype == np.float32:
                    result[row, out + i * stride + 1] = bits.view(np.float32)
                else:
                    raw = bits.tobytes()
                    result[row, out + i * stride + 1] = np.frombuffer(raw[:2], dtype=np.float16)[0]
                    result[row, out + i * stride + 2] = np.frombuffer(raw[2:], dtype=np.float16)[0]
    return result


_kernels = {}
CASES = []
for _name, _pto_dtype, _np_dtype, _src_shape, _idx_shape, _tmp_shape, _dst_shape, _src_valid, _dst_valid in CASE_SPECS:
    _kernels[_name] = _make_kernel(_name, _pto_dtype, _src_shape, _idx_shape, _tmp_shape, _dst_shape, _src_valid, _dst_valid)
    CASES.append(
        golden_output_case(
            "tsort32_" + _name,
            _kernels[_name],
            inputs=lambda name=_name, dtype=_np_dtype, ss=_src_shape, ins=_idx_shape: _make_inputs(name, dtype, ss, ins),
            expected=lambda src, idx, dtype=_np_dtype, sv=_src_valid, ds=_dst_shape, dv=_dst_valid: _make_expected(src, idx, dtype, sv, ds, dv),
            rtol=1e-3 if _np_dtype == np.float16 else 1e-6,
            atol=1e-3 if _np_dtype == np.float16 else 1e-6,
        )
    )


auto_main(globals())
