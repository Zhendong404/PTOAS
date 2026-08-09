#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tcvt.
#
# tcvt is an element-wise type conversion: pto.tile.cvt(src, dst) with the
# destination tile dtype selecting the target type.  The full legacy case
# table (src/dst dtype, shape, valid_shape, dst_shape, round mode, eps) and
# the gen_data.py input/golden semantics are preserved; every case name is
# prefixed with "tcvt_".
#
# bf16 host inputs are passed as raw 16-bit storage (uint16 bit patterns)
# because the runtime torch cannot materialize ml_dtypes.bfloat16 tensors;
# this mirrors the legacy launcher which declared bf16 buffers as uint16_t.
# The golden logic reinterprets those bits back to bf16.

from pathlib import Path
import sys
import zlib

import ml_dtypes
import numpy as np
from ml_dtypes import bfloat16

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto

F8E4M3 = "f8e4m3"
F8E5M2 = "f8e5m2"
HIF8 = "hif8"
F4E1M2X2 = "f4e1m2x2"
F4E2M1X2 = "f4e2m1x2"

# 7 shapes (aligning with the legacy C++ INSTANTIATE_TCVT)
SHAPES = [
    (1, 128, 1, 128),
    (2, 64, 2, 64),
    (4, 32, 4, 32),
    (2, 128, 2, 128),
    (4, 128, 4, 65),   # Partial tiles
    (4, 256, 4, 200),  # Partial tiles
    (1, 256, 1, 129),  # Partial tiles
]

_LOW_PRECISION_DTYPES = frozenset({F8E4M3, F8E5M2, HIF8, F4E1M2X2, F4E2M1X2})

_NP_TO_PTO = {
    np.float32: pto.f32,
    np.float16: pto.f16,
    bfloat16: pto.bf16,
    # The legacy A5 tcvt cases use signed int8 tiles.  ``pto.i8`` is the
    # signless MLIR integer spelling and causes TileLib selection to see
    # ``i8`` instead of the legacy ``si8`` contract.
    np.int8: pto.si8,
    np.uint8: pto.ui8,
    np.int16: pto.i16,
    "si16": pto.si16,
    np.uint16: pto.ui16,
    np.int32: pto.i32,
    np.uint32: pto.ui32,
    np.int64: pto.i64,
    F8E4M3: pto.f8e4m3,
    F8E5M2: pto.f8e5m2,
    HIF8: pto.hif8,
    F4E1M2X2: pto.f4e1m2x2,
    F4E2M1X2: pto.f4e2m1x2,
}

_DTYPE_NAME = {
    np.float32: "f32",
    np.float16: "f16",
    bfloat16: "bf16",
    np.int8: "si8",
    np.uint8: "ui8",
    np.int16: "i16",
    "si16": "si16",
    np.uint16: "ui16",
    np.int32: "i32",
    np.uint32: "ui32",
    np.int64: "i64",
    np.uint64: "ui64",
    F8E4M3: F8E4M3,
    F8E5M2: F8E5M2,
    HIF8: HIF8,
    F4E1M2X2: F4E1M2X2,
    F4E2M1X2: F4E2M1X2,
}

_STR_DTYPE_MAP = {
    "si16": np.int16,
    F8E4M3: np.uint8,
    F8E5M2: np.uint8,
    HIF8: np.uint8,
    F4E1M2X2: np.uint8,
    F4E2M1X2: np.uint8,
}


def dtype_name(dtype):
    return _DTYPE_NAME.get(dtype, dtype)


def normalize_dtype(dtype):
    return _STR_DTYPE_MAP.get(dtype, dtype)


def is_low_precision_dtype(dtype):
    return dtype in _LOW_PRECISION_DTYPES


def eps_for_dtype(dtype):
    eps_map = {np.float32: 1e-6, np.float16: 1e-3, bfloat16: 1e-3}
    return eps_map.get(dtype, 0.0)


def _is_sub_float(dtype):
    if isinstance(dtype, str):
        return False
    return np.issubdtype(dtype, np.floating) or dtype == bfloat16


def _is_sub_int(dtype):
    if isinstance(dtype, str):
        return False
    return np.issubdtype(dtype, np.integer)


# ---------------------------------------------------------------------------
# Legacy case table (single source of truth, as in the legacy cases.py)
# ---------------------------------------------------------------------------

def _make_cases(src_dtype, dst_dtype):
    """Generate cases of 7 test shapes for src_dtype -> dst_dtype."""
    src_name = dtype_name(src_dtype)
    dst_name = dtype_name(dst_dtype)
    eps = eps_for_dtype(dst_dtype)

    cases = []
    for rows, cols, v_rows, v_cols in SHAPES:
        shape_name = f"{rows}x{cols}" if v_cols == cols else f"{v_rows}x{v_cols}"
        cases.append({
            "name": f"{src_name}_to_{dst_name}_{shape_name}",
            "src_dtype": src_dtype,
            "dst_dtype": dst_dtype,
            "shape": (rows, cols),
            "valid_shape": (v_rows, v_cols),
            "eps": eps,
        })
    return cases


def _make_low_precision_cases():
    """Targeted A5 low-precision tcvt coverage without multiplying all shapes."""
    shape = (16, 64)
    lowp_cases = []
    for src_dtype, dst_dtype in (
        (np.float32, F8E4M3),
        (np.float32, F8E5M2),
        (np.float32, HIF8),
        (np.float16, HIF8),
        (bfloat16, F4E1M2X2),
        (bfloat16, F4E2M1X2),
    ):
        case = {
            "name": f"{dtype_name(src_dtype)}_to_{dtype_name(dst_dtype)}_{shape[0]}x{shape[1]}",
            "src_dtype": src_dtype,
            "dst_dtype": dst_dtype,
            "shape": shape,
            "valid_shape": shape,
            "eps": eps_for_dtype(dst_dtype),
        }
        if dst_dtype in (F4E1M2X2, F4E2M1X2):
            case["name"] = f"{dtype_name(src_dtype)}_to_{dtype_name(dst_dtype)}_16x64_to_16x32"
            case["dst_shape"] = (16, 32)
            case["dst_valid_shape"] = (16, 32)
        lowp_cases.append(case)
    for src_dtype, dst_dtype in (
        (np.float32, F8E4M3),
        (np.float32, HIF8),
    ):
        shape = (4, 96)
        lowp_cases.append({
            "name": f"{dtype_name(src_dtype)}_to_{dtype_name(dst_dtype)}_{shape[0]}x{shape[1]}",
            "src_dtype": src_dtype,
            "dst_dtype": dst_dtype,
            "shape": shape,
            "valid_shape": shape,
            "eps": eps_for_dtype(dst_dtype),
        })
    return lowp_cases


_CASE_SPECS = [
    {
        "name": "f32_to_i32_rint_16x64",
        "src_dtype": np.float32,
        "dst_dtype": np.int32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "round_mode": "RINT",
        "eps": 0.0,
    },
    {
        "name": "f32_to_i32_round_16x64",
        "src_dtype": np.float32,
        "dst_dtype": np.int32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "round_mode": "ROUND",
        "eps": 0.0,
    },
    {
        "name": "i32_to_f32_rint_16x64",
        "src_dtype": np.int32,
        "dst_dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "round_mode": "RINT",
        "eps": 1e-6,
    },
    {
        "name": "f32_to_f16_rint_16x64",
        "src_dtype": np.float32,
        "dst_dtype": np.float16,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "round_mode": "RINT",
        "eps": 1e-3,
    },
    {
        "name": "f16_to_f32_rint_16x64",
        "src_dtype": np.float16,
        "dst_dtype": np.float32,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "round_mode": "RINT",
        "eps": 1e-6,
    },
    *_make_low_precision_cases(),
    # f32 -> f16, bf16, i16, i32, i64, f32
    *_make_cases(np.float32, np.float16),
    *_make_cases(np.float32, bfloat16),
    *_make_cases(np.float32, np.int16),
    *_make_cases(np.float32, np.int32),
    *_make_cases(np.float32, np.int64),
    *_make_cases(np.float32, np.float32),
    # f16 -> f32, i32, i16, si8, ui8
    *_make_cases(np.float16, np.float32),
    *_make_cases(np.float16, np.int32),
    *_make_cases(np.float16, np.int16),
    *_make_cases(np.float16, np.int8),
    *_make_cases(np.float16, np.uint8),
    # bf16 -> f32, f16, i32
    *_make_cases(bfloat16, np.float32),
    *_make_cases(bfloat16, np.float16),
    *_make_cases(bfloat16, np.int32),
    # ui8 -> f16, ui16
    *_make_cases(np.uint8, np.float16),
    *_make_cases(np.uint8, np.uint16),
    # si8 -> f16, si16, i32
    *_make_cases(np.int8, np.float16),
    *_make_cases(np.int8, "si16"),
    *_make_cases(np.int8, np.int32),
    # i16 -> ui8, f16, f32, ui32, i32
    *_make_cases(np.int16, np.uint8),
    *_make_cases(np.int16, np.float16),
    *_make_cases(np.int16, np.float32),
    *_make_cases(np.int16, np.uint32),
    *_make_cases(np.int16, np.int32),
    # i32 -> f32, i16, i64, ui8, ui16
    *_make_cases(np.int32, np.float32),
    *_make_cases(np.int32, np.int16),
    *_make_cases(np.int32, np.int64),
    *_make_cases(np.int32, np.uint8),
    *_make_cases(np.int32, np.uint16),
    # ui32 -> i16, ui16, ui8
    *_make_cases(np.uint32, np.int16),
    *_make_cases(np.uint32, np.uint16),
    *_make_cases(np.uint32, np.uint8),
    # i64 -> f32, i32
    *_make_cases(np.int64, np.float32),
    *_make_cases(np.int64, np.int32),
]


# ---------------------------------------------------------------------------
# Kernels: one @pto.jit auto-mode kernel per static dtype/shape variant
# ---------------------------------------------------------------------------

def _make_kernel(case):
    name = "tcvt_" + case["name"]
    src_pto = _NP_TO_PTO[normalize_dtype(case["src_dtype"])]
    dst_pto = _NP_TO_PTO[case["dst_dtype"]]
    rows, cols = case["shape"]
    v_rows, v_cols = case["valid_shape"]
    dst_rows, dst_cols = case.get("dst_shape", case["shape"])
    dv_rows, dv_cols = case.get("dst_valid_shape", case["valid_shape"])
    rmode = pto.RoundMode.ROUND if case.get("round_mode") == "ROUND" else None

    @pto.jit(name=name, target="a5")
    def _kernel(
        src_ptr: pto.ptr(src_pto, "gm"),
        dst_ptr: pto.ptr(dst_pto, "gm"),
    ):
        src_view = pto.make_tensor_view(src_ptr, shape=[rows, cols], strides=[cols, 1])
        dst_view = pto.make_tensor_view(dst_ptr, shape=[dst_rows, dst_cols], strides=[dst_cols, 1])

        src_tile = pto.alloc_tile(
            shape=[rows, cols], dtype=src_pto, valid_shape=[v_rows, v_cols]
        )
        dst_tile = pto.alloc_tile(
            shape=[dst_rows, dst_cols], dtype=dst_pto, valid_shape=[dv_rows, dv_cols]
        )

        pto.tile.load(src_view, src_tile)
        if rmode is None:
            pto.tile.cvt(src_tile, dst_tile)
        else:
            pto.tile.cvt(src_tile, dst_tile, rmode=rmode)
        pto.tile.store(dst_tile, dst_view)

    return _kernel


_kernels = {}


def _build_kernels():
    for case in _CASE_SPECS:
        name = "tcvt_" + case["name"]
        _kernels[name] = _make_kernel(case)


_build_kernels()


# ---------------------------------------------------------------------------
# Host input generation and golden (port of the legacy gen_data.py)
# ---------------------------------------------------------------------------

def _make_input_inner(src_dtype, shape):
    total = int(np.prod(shape))
    float_types = (np.float32, np.float16, bfloat16)

    if src_dtype in float_types:
        return np.random.random([total]) * 200 - 100
    elif src_dtype == np.int8:
        return np.random.randint(-128, 128, [total])
    elif src_dtype == np.uint8:
        return np.random.randint(0, 256, [total])
    elif src_dtype == np.int16:
        return np.random.randint(-1000, 1000, [total])
    elif src_dtype == np.uint16:
        return np.random.randint(0, 10000, [total])
    elif src_dtype in (np.int32, np.int64):
        return np.random.randint(-10000, 10000, [total])
    elif src_dtype == np.uint32:
        return np.random.randint(0, 10000, [total])
    else:
        return np.random.randint(-10000, 10000, [total])


def _make_input(src_dtype, shape):
    return _make_input_inner(src_dtype, shape).astype(normalize_dtype(src_dtype)).reshape(shape)


def _round_half_away_from_zero(values):
    return np.copysign(np.floor(np.abs(values) + 0.5), values)


def _default_saturation_off(src_dtype, dst_dtype):
    """Mirror the current A5 default saturation policy for supported pairs."""
    return (
        (src_dtype is np.float16 and dst_dtype is np.uint8)
        or (src_dtype is np.float16 and dst_dtype is np.int8)
        or (src_dtype is np.float32 and dst_dtype is np.int16)
        or (src_dtype is np.float16 and dst_dtype is np.int16)
        or (src_dtype is np.int64 and dst_dtype is np.int32)
        or (src_dtype is np.int32 and dst_dtype is np.int16)
    )


def _apply_round_mode(values, round_mode):
    rounding_funcs = {
        "RINT": np.rint,
        "ROUND": _round_half_away_from_zero,
        "FLOOR": np.floor,
        "CEIL": np.ceil,
        "TRUNC": np.trunc,
    }
    return rounding_funcs.get(round_mode, np.rint)(values)


def _truncate_to_int(values, dst_dtype):
    golden_list = []
    for val in values.flat:
        int_val = 0 if np.isnan(val) or np.isinf(val) else int(np.int64(val))

        if dst_dtype == np.int8:
            byte_val = int_val & 0xFF
            truncated_val = byte_val if byte_val < 128 else byte_val - 256
        elif dst_dtype == np.uint8:
            truncated_val = int_val & 0xFF
        elif dst_dtype == np.int16:
            word_val = int_val & 0xFFFF
            truncated_val = word_val if word_val < 32768 else word_val - 65536
        elif dst_dtype == np.int32:
            dword_val = int_val & 0xFFFFFFFF
            truncated_val = dword_val if dword_val < 2147483648 else dword_val - 4294967296
        else:
            truncated_val = int_val
        golden_list.append(truncated_val)
    return np.array(golden_list, dtype=dst_dtype).reshape(values.shape)


def _clamp_to_range_int(values, dst_dtype):
    info = ml_dtypes.iinfo(dst_dtype)
    is_int_type = _is_sub_int(values.dtype)
    temp_dtype = np.int64 if is_int_type else np.float64
    widened = values.astype(temp_dtype, copy=False)
    return np.clip(widened, info.min, info.max).astype(dst_dtype)


def _clamp_to_range_float(values, dst_dtype):
    info = ml_dtypes.finfo(dst_dtype)
    return np.clip(values, info.min, info.max).astype(dst_dtype)


def _convert(values, src_dtype, dst_dtype, round_mode=None):
    is_float_src = _is_sub_float(src_dtype)
    is_int_dst = _is_sub_int(dst_dtype)
    is_f32_to_f32 = src_dtype == np.float32 and dst_dtype == np.float32
    needs_rounding = is_float_src and (is_int_dst or is_f32_to_f32)

    if needs_rounding:
        values = _apply_round_mode(values, round_mode or "RINT")

    if is_int_dst:
        # Determine if this conversion has default saturation OFF (truncation)
        # or ON (clamping).
        if _default_saturation_off(src_dtype, dst_dtype):
            return _truncate_to_int(values, dst_dtype)
        return _clamp_to_range_int(values, dst_dtype)
    elif _is_sub_float(dst_dtype):
        return _clamp_to_range_float(values, dst_dtype)
    else:
        return values.astype(dst_dtype)


def _apply_valid_shape(values, valid_shape, dst_dtype):
    vr, vc = valid_shape
    masked = np.zeros_like(values, dtype=dst_dtype)
    masked[:vr, :vc] = values[:vr, :vc]
    return masked


def _bits_to_f32(bits):
    return np.array([bits], dtype=np.uint32).view(np.float32)[0]


def _f32_to_bits(value):
    return np.array([value], dtype=np.float32).view(np.uint32)[0]


def _f32_to_bf16_bits(value):
    bits = int(_f32_to_bits(value))
    lsb = (bits >> 16) & 1
    rounded = bits + 0x7FFF + lsb
    return np.uint16((rounded >> 16) & 0xFFFF)


def _decode_f8e4m3fn(byte):
    sign = -1.0 if byte & 0x80 else 1.0
    exp = (byte >> 3) & 0x0F
    mant = byte & 0x07
    if byte in (0x7F, 0xFF):
        return np.float32(np.nan)
    if exp == 0:
        return np.float32(sign * (mant / 8.0) * (2.0 ** -6))
    return np.float32(sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7)))


def _decode_f8e5m2(byte):
    sign = -1.0 if byte & 0x80 else 1.0
    exp = (byte >> 2) & 0x1F
    mant = byte & 0x03
    if exp == 0x1F:
        if mant == 0:
            return np.float32(sign * np.inf)
        return np.float32(np.nan)
    if exp == 0:
        return np.float32(sign * (mant / 4.0) * (2.0 ** -14))
    return np.float32(sign * (1.0 + mant / 4.0) * (2.0 ** (exp - 15)))


def _fp32_constructor(sign, exp, mant):
    return ((sign & 1) << 31) | ((exp & 0xFF) << 23) | (mant & 0x7FFFFF)


def _decode_hif8(byte):
    if byte == 0x00:
        return _bits_to_f32(0x00000000)
    if byte == 0x80:
        return np.float32(np.nan)
    if byte == 0x6F:
        return _bits_to_f32(0x7F800000)
    if byte == 0xEF:
        return _bits_to_f32(0xFF800000)

    input_sign = (byte >> 7) & 0x01
    bit6 = (byte >> 6) & 0x01
    bit5 = (byte >> 5) & 0x01
    bit4 = (byte >> 4) & 0x01
    bit3 = (byte >> 3) & 0x01
    if bit6 == 0 and bit5 == 0 and bit4 == 0 and bit3 == 0:
        return _bits_to_f32(_fp32_constructor(input_sign, (byte & 0x7) - 23 + 127, 0))

    if bit6 == 0:
        if bit5 == 0:
            exp_width = 0 if bit4 == 0 else 1
            d_width = 4 if bit4 == 0 else 3
        else:
            exp_width = 2
            d_width = 2
    else:
        exp_width = 3 if bit5 == 0 else 4
        d_width = 2
    man_width = 8 - d_width - exp_width - 1

    exp_mask = (1 << exp_width) - 1
    exp = 0
    if exp_width != 0:
        exp = ((byte >> man_width) & exp_mask) | (1 << (exp_width - 1))
        exp_msb = (byte >> (man_width + exp_width - 1)) & 0x1
        if exp_msb != 0:
            exp = -exp
    exp += 127

    man_mask = (1 << man_width) - 1
    mant = (byte & man_mask) << (23 - man_width)
    return _bits_to_f32(_fp32_constructor(input_sign, exp, mant))


_FP4E1M2_TO_BF16 = np.array(
    [
        0x0000, 0x3E80, 0x3F00, 0x3F40, 0x3F80, 0x3FA0, 0x3FC0, 0x3FE0,
        0x8000, 0xBE80, 0xBF00, 0xBF40, 0xBF80, 0xBFA0, 0xBFC0, 0xBFE0,
    ],
    dtype=np.uint16,
)

_FP4E2M1_TO_BF16 = np.array(
    [
        0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
        0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0,
    ],
    dtype=np.uint16,
)


def _f8e4_quantize_pairs():
    exact = [0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88, 0x38, 0xB8, 0x3C, 0xBC, 0x7E, 0xFE]
    pairs = [(_decode_f8e4m3fn(byte), byte) for byte in exact]
    # The f32->fp8 TileLang template uses #sat=1. In V300
    # instruction-controlled saturation, infinities/overflow clamp to max
    # finite and NaN is saturated to zero.
    pairs.extend([
        (np.float32(np.inf), 0x7E),
        (np.float32(-np.inf), 0xFE),
        (np.float32(np.nan), 0x00),
        (np.float32(1000.0), 0x7E),
        (np.float32(-1000.0), 0xFE),
        (np.float32(1.0625), 0x38),
        (np.float32(1.1875), 0x3A),
        (np.float32(-1.0625), 0xB8),
        (np.float32(-1.1875), 0xBA),
    ])
    return pairs


def _f8e5_quantize_pairs():
    exact = [0x00, 0x80, 0x01, 0x81, 0x03, 0x83, 0x04, 0x84, 0x3C, 0xBC, 0x40, 0xC0, 0x7B, 0xFB]
    pairs = [(_decode_f8e5m2(byte), byte) for byte in exact]
    # The f32->fp8 TileLang template uses #sat=1. In V300
    # instruction-controlled saturation, infinities/overflow clamp to max
    # finite and NaN is saturated to zero.
    pairs.extend([
        (np.float32(np.inf), 0x7B),
        (np.float32(-np.inf), 0xFB),
        (np.float32(np.nan), 0x00),
        (np.float32(1.0e10), 0x7B),
        (np.float32(-1.0e10), 0xFB),
        (np.float32(1.125), 0x3C),
        (np.float32(1.375), 0x3E),
        (np.float32(-1.125), 0xBC),
        (np.float32(-1.375), 0xBE),
    ])
    return pairs


def _hif8_quantize_pairs():
    exact = [
        0x00, 0x80, 0x6F, 0xEF, 0x01, 0x81, 0x07, 0x87,
        0x08, 0x88, 0x10, 0x90, 0x18, 0x98, 0x20, 0xA0,
        0x40, 0xC0, 0x50, 0xD0, 0x60, 0xE0, 0x70, 0xF0,
    ]
    return [(_decode_hif8(byte), byte) for byte in exact]


def _f4e1_quantize_pairs():
    pairs = [(bits, nibble) for nibble, bits in enumerate(_FP4E1M2_TO_BF16)]
    pairs.extend([
        (_f32_to_bf16_bits(0.625), 0x2),
        (_f32_to_bf16_bits(0.75), 0x3),
        (_f32_to_bf16_bits(0.875), 0x4),
        (_f32_to_bf16_bits(-0.625), 0xA),
        (_f32_to_bf16_bits(-0.75), 0xB),
        (_f32_to_bf16_bits(-0.875), 0xC),
    ])
    return pairs


def _f4e2_quantize_pairs():
    pairs = [(bits, nibble) for nibble, bits in enumerate(_FP4E2M1_TO_BF16)]
    pairs.extend([
        (_f32_to_bf16_bits(0.75), 0x2),
        (_f32_to_bf16_bits(1.25), 0x2),
        (_f32_to_bf16_bits(1.75), 0x4),
        (_f32_to_bf16_bits(-0.75), 0xA),
        (_f32_to_bf16_bits(-1.25), 0xA),
        (_f32_to_bf16_bits(-1.75), 0xC),
    ])
    return pairs


def _make_low_precision_quantize_golden(case):
    src_dtype = case["src_dtype"]
    dst_dtype = case["dst_dtype"]
    shape = case["shape"]
    dst_shape = case.get("dst_shape", shape)
    total = int(np.prod(shape))
    dst_total = int(np.prod(dst_shape))

    if dst_dtype == F8E4M3:
        pairs = _f8e4_quantize_pairs()
        values = np.resize(np.array([value for value, _ in pairs], dtype=np.float32), total)
        golden = np.resize(np.array([byte for _, byte in pairs], dtype=np.uint8), total)
        return values.astype(src_dtype).reshape(shape), golden.reshape(shape)

    if dst_dtype == F8E5M2:
        pairs = _f8e5_quantize_pairs()
        values = np.resize(np.array([value for value, _ in pairs], dtype=np.float32), total)
        golden = np.resize(np.array([byte for _, byte in pairs], dtype=np.uint8), total)
        return values.astype(src_dtype).reshape(shape), golden.reshape(shape)

    if dst_dtype == HIF8:
        pairs = _hif8_quantize_pairs()
        values = np.resize(np.array([value for value, _ in pairs], dtype=np.float32), total)
        golden = np.resize(np.array([byte for _, byte in pairs], dtype=np.uint8), total)
        return values.astype(src_dtype).reshape(shape), golden.reshape(shape)

    if dst_dtype == F4E1M2X2:
        pairs = _f4e1_quantize_pairs()
    elif dst_dtype == F4E2M1X2:
        pairs = _f4e2_quantize_pairs()
    else:
        raise ValueError(f"unsupported low-precision dst dtype: {dst_dtype}")

    bits = np.resize(np.array([bits for bits, _ in pairs], dtype=np.uint16), total)
    nibbles = np.resize(np.array([nibble for _, nibble in pairs], dtype=np.uint8), total)
    golden = np.array(
        [nibbles[i] | (nibbles[i + 1] << 4) for i in range(0, dst_total * 2, 2)],
        dtype=np.uint8,
    )
    return bits.view(bfloat16).reshape(shape), golden.reshape(dst_shape)


def _generate_golden(case):
    """Return (input_arr, golden) exactly as the legacy gen_data.generate_golden."""
    src_dtype = case["src_dtype"]
    dst_dtype = case["dst_dtype"]
    src_dtype_norm = normalize_dtype(src_dtype)
    dst_dtype_norm = normalize_dtype(dst_dtype)
    shape = case["shape"]
    round_mode = case.get("round_mode")

    if is_low_precision_dtype(dst_dtype):
        input_arr, golden = _make_low_precision_quantize_golden(case)
        dst_valid_shape = case.get("dst_valid_shape", case["valid_shape"])
        return input_arr, _apply_valid_shape(golden, dst_valid_shape, np.uint8)

    input_arr = _make_input(src_dtype, shape)
    converted = _convert(input_arr, src_dtype_norm, dst_dtype_norm, round_mode)
    golden = _apply_valid_shape(converted, case["valid_shape"], dst_dtype_norm)

    return input_arr, golden


def _make_inputs(case):
    """Deterministic per-case seed, mirroring st_common.setup_case_rng.

    The seed uses the *legacy* case name so the generated data is identical
    to the legacy gen_data.py output.
    """
    np.random.seed(zlib.crc32(case["name"].encode("utf-8")) & 0xFFFFFFFF)
    input_arr, _golden = _generate_golden(case)
    if normalize_dtype(case["src_dtype"]) is bfloat16:
        # Runtime torch cannot materialize ml_dtypes.bfloat16; pass the raw
        # 16-bit storage (identical to the legacy uint16_t host buffers).
        return [np.asarray(input_arr).view(np.uint16)]
    return [input_arr]


def _expected_for(case):
    src_norm = normalize_dtype(case["src_dtype"])
    dst_norm = normalize_dtype(case["dst_dtype"])
    low_precision = is_low_precision_dtype(case["dst_dtype"])
    dst_valid_shape = case.get("dst_valid_shape", case["valid_shape"])
    valid_shape = case["valid_shape"]
    round_mode = case.get("round_mode")

    def expected(src):
        if low_precision:
            _input_arr, golden = _make_low_precision_quantize_golden(case)
            return _apply_valid_shape(golden, dst_valid_shape, np.uint8)
        if src_norm is bfloat16:
            src = np.asarray(src).view(bfloat16)
        converted = _convert(src, src_norm, dst_norm, round_mode)
        masked = _apply_valid_shape(converted, valid_shape, dst_norm)
        # torch.from_numpy cannot materialize ml_dtypes.bfloat16.  Keep the
        # public kernel ABI as bf16, but compare/output through its uint16
        # storage representation, matching the legacy uint16 host buffers.
        if dst_norm is bfloat16:
            return np.asarray(masked).view(np.uint16)
        return masked

    return expected


def _build_case(case):
    name = "tcvt_" + case["name"]
    return golden_output_case(
        name,
        _kernels[name],
        inputs=lambda case=case: _make_inputs(case),
        expected=_expected_for(case),
        rtol=case["eps"],
        atol=case["eps"],
    )


CASES = [_build_case(case) for case in _CASE_SPECS]


auto_main(globals())
