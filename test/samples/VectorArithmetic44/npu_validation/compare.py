#!/usr/bin/python3
# coding=utf-8

import numpy as np

ROWS = 32
COLS = 32
TILE_SIZE = ROWS * COLS
OP_NAMES = [
    "tadd",
    "tsub",
    "tmul",
    "tdiv",
    "tmax",
    "tmin",
    "trem",
    "tprelu",
    "tadds",
    "tsubs",
    "tmuls",
    "tdivs",
    "tdivs_reverse",
    "tmaxs",
    "tmins",
    "trems",
    "taddc",
    "tsubc",
    "taddsc",
    "tsubsc",
    "tabs",
    "tneg",
    "texp",
    "tlog",
    "tsqrt",
    "trsqrt",
    "trecip",
    "trelu",
    "tlrelu",
    "tpartadd",
    "tpartmax",
    "tpartmin",
]


def compare_bin(golden_path, output_path, dtype, eps):
    golden = np.fromfile(golden_path, dtype=dtype)
    output = np.fromfile(output_path, dtype=dtype)
    if golden.shape != output.shape:
        print(f"[ERROR] Shape mismatch: {golden_path} {golden.shape} vs {output_path} {output.shape}")
        return False

    if golden.size % TILE_SIZE != 0:
        print(f"[ERROR] Unexpected flattened size {golden.size}, not divisible by tile size {TILE_SIZE}")
        return False

    op_count = golden.size // TILE_SIZE
    if op_count != len(OP_NAMES):
        print(f"[WARN] OP count mismatch: data has {op_count}, table has {len(OP_NAMES)}")

    all_ok = True
    for idx in range(op_count):
        op_name = OP_NAMES[idx] if idx < len(OP_NAMES) else f"op_{idx}"
        begin = idx * TILE_SIZE
        end = begin + TILE_SIZE
        golden_chunk = golden[begin:end]
        output_chunk = output[begin:end]
        close_mask = np.isclose(golden_chunk, output_chunk, atol=eps, rtol=eps)
        if np.all(close_mask):
            continue

        all_ok = False
        diff = np.abs(golden_chunk - output_chunk)
        mismatch_indices = np.flatnonzero(~close_mask)
        first_idx = int(mismatch_indices[0])
        row = first_idx // COLS
        col = first_idx % COLS
        max_diff = float(np.max(diff))
        print(
            f"[ERROR] {op_name} mismatch: max diff={max_diff}, "
            f"first mismatch at flat={first_idx} (row={row}, col={col}), "
            f"golden={golden_chunk[first_idx]}, output={output_chunk[first_idx]}"
        )

    if not all_ok:
        print(f"[ERROR] Mismatch: {golden_path} vs {output_path}")
    return all_ok


def main():
    ok = True
    ok = compare_bin("golden_v4.bin", "v4.bin", np.float32, 1e-3) and ok
    if not ok:
        print("[ERROR] compare failed")
        raise SystemExit(1)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
