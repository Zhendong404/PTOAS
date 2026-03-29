#!/usr/bin/env python3
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


ROWS = 16
COLS = 128
SEED = 19


def float32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32, copy=False).view(np.uint32)
    lsb = (bits >> 16) & 1
    rounding_bias = np.uint32(0x7FFF) + lsb
    return ((bits + rounding_bias) >> 16).astype(np.uint16)


def bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)

    v1 = rng.uniform(-3.0, 3.0, size=(ROWS, COLS)).astype(np.float32)
    v3 = np.zeros((ROWS, COLS), dtype=np.uint16)
    v4 = np.zeros((ROWS, 1), dtype=np.float32)
    v5 = np.zeros((ROWS, 1), dtype=np.float32)

    rowmax = np.max(v1, axis=1, keepdims=True)
    shifted = v1 - rowmax
    expv = np.exp(shifted).astype(np.float32, copy=False)
    expv_bf16 = float32_to_bf16_bits(expv)
    expv_roundtrip = bf16_bits_to_float32(expv_bf16)
    golden_v5 = np.sum(expv_roundtrip, axis=1, keepdims=True).astype(np.float32, copy=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in {
        "v1": v1,
        "v3": v3,
        "v4": v4,
        "v5": v5,
        "golden_v5": golden_v5,
    }.items():
        value.reshape(-1).tofile(output_dir / f"{name}.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
