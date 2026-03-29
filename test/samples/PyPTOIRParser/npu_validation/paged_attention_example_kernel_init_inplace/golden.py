#!/usr/bin/env python3
# coding=utf-8

import argparse
from pathlib import Path

import numpy as np


NUM_ELEMS = 1024
SEED = 19


def generate(output_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)

    v1 = rng.uniform(-2.0, 2.0, size=NUM_ELEMS).astype(np.float32)
    v2 = rng.uniform(-2.0, 2.0, size=NUM_ELEMS).astype(np.float32)
    v3 = rng.uniform(-2.0, 2.0, size=NUM_ELEMS).astype(np.float32)
    golden_v3 = v3.astype(np.float32, copy=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in {
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "golden_v3": golden_v3,
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
