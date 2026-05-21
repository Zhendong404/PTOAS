# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""End-to-end tadd JIT smoke test: Python DSL → IR → binary → launch → accuracy."""

from __future__ import annotations

import sys

import numpy as np
import torch
import torch_npu  # noqa: F401

from tadd_builder import TADD_f32_16x64, TADD_f32_32x32

_DEVICE = "npu:0"

CASES = [
    {
        "name": "f32_16x64",
        "kernel": TADD_f32_16x64,
        "shape": (16, 64),
        "valid_shape": (16, 64),
        "eps": 1e-6,
    },
    {
        "name": "f32_32x32",
        "kernel": TADD_f32_32x32,
        "shape": (32, 32),
        "valid_shape": (32, 32),
        "eps": 1e-6,
    },
]


def init_torch_npu() -> None:
    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)


def npu_tensor(np_arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np_arr).to(_DEVICE)


def empty_npu(shape, dtype) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=_DEVICE)


def make_case_data(case: dict):
    shape = case["shape"]
    valid_shape = case["valid_shape"]
    vr, vc = valid_shape

    rng = np.random.RandomState(hash(case["name"]) & 0xFFFFFFFF)
    input1 = rng.randint(1, 10, size=shape).astype(np.float32)
    input2 = rng.randint(1, 10, size=shape).astype(np.float32)
    golden = np.zeros(shape, dtype=np.float32)
    golden[:vr, :vc] = (input1[:vr, :vc] + input2[:vr, :vc]).astype(np.float32)
    return input1, input2, golden, valid_shape


def run_case(case: dict) -> None:
    input1, input2, golden, valid_shape = make_case_data(case)
    vr, vc = valid_shape

    a = npu_tensor(input1)
    b = npu_tensor(input2)
    c = empty_npu(case["shape"], torch.float32)

    compiled = case["kernel"].compile()
    compiled[1, None](a, b, c)
    torch.npu.synchronize()

    actual = c.cpu().numpy()
    torch.testing.assert_close(
        torch.from_numpy(golden[:vr, :vc]),
        torch.from_numpy(actual[:vr, :vc]),
        rtol=case["eps"],
        atol=case["eps"],
    )
    print(f"PASS {case['name']}")


def main() -> int:
    init_torch_npu()
    for case in CASES:
        run_case(case)
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
