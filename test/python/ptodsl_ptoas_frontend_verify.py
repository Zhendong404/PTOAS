#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ptodsl"))

from ptodsl import pto


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def resolve_ptoas_binary() -> Path:
    candidates = [
        REPO_ROOT / "build" / "tools" / "ptoas" / "ptoas",
        REPO_ROOT / "install" / "bin" / "ptoas",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    from_path = shutil.which("ptoas")
    if from_path:
        return Path(from_path)

    raise FileNotFoundError("unable to locate a ptoas binary under build/, install/, or PATH")


def load_flash_attention_demo():
    demo_path = REPO_ROOT / "ptodsl" / "demos" / "flash_attention_sketch.py"
    expect(demo_path.is_file(), f"canonical flash attention demo is missing: {demo_path}")

    spec = spec_from_file_location("ptodsl_flash_attention_demo", demo_path)
    expect(spec is not None and spec.loader is not None, f"unable to create import spec for {demo_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_ptoas_frontend_verify(ptoas_bin: Path, mlir_text: str, label: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False, encoding="utf-8") as handle:
        handle.write(mlir_text)
        input_path = Path(handle.name)

    try:
        result = subprocess.run(
            [str(ptoas_bin), str(input_path), "--emit-pto-ir", "-o", "-"],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        input_path.unlink(missing_ok=True)

    expect(
        result.returncode == 0,
        f"{label} should pass PTOAS frontend verification.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )
    expect(result.stdout.strip(), f"{label} should emit non-empty PTO IR after PTOAS frontend passes")
    return result.stdout


@pto.jit(target="a5")
def host_vec_copy(
    A: pto.tensor_spec(rank=2, dtype=pto.f32),
    O: pto.tensor_spec(rank=2, dtype=pto.f32),
    *,
    BLOCK: pto.constexpr = 128,
):
    rows = A.shape[0]
    cols = A.shape[1]
    a_view = pto.make_tensor_view(A, shape=A.shape, strides=A.strides)
    o_view = pto.make_tensor_view(O, shape=O.shape, strides=O.strides)
    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    out = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    pto.tload(part, a_tile)
    pto.tstore(o_tile, out)


def main() -> None:
    ptoas_bin = resolve_ptoas_binary()
    demo = load_flash_attention_demo()

    simple_text = host_vec_copy.compile().mlir_text()
    simple_frontend_text = run_ptoas_frontend_verify(
        ptoas_bin,
        simple_text,
        "host_vec_copy PTODSL artifact",
    )
    expect(
        "func.func @host_vec_copy" in simple_frontend_text,
        "host_vec_copy frontend verification output should preserve the kernel symbol",
    )
    expect(
        "pto.tload" in simple_frontend_text and "pto.tstore" in simple_frontend_text,
        "host_vec_copy frontend verification output should keep the tile IO contract visible",
    )

    flash_text = demo.emit_flash_attention_mlir(
        head_dim=128,
        causal=False,
        block_q=128,
        block_kv=128,
    )
    flash_frontend_text = run_ptoas_frontend_verify(
        ptoas_bin,
        flash_text,
        "flash attention PTODSL artifact",
    )
    expect(
        "func.func @flash_attention_kernel" in flash_frontend_text,
        "flash attention frontend verification output should preserve the JIT entry symbol",
    )
    expect(
        "func.func @materialize_tile_bounds" in flash_frontend_text,
        "flash attention frontend verification output should preserve the SIMT helper symbol",
    )
    expect(
        "math.exp" in flash_frontend_text,
        "flash attention frontend verification output should accept PTODSL scalar math ops such as math.exp",
    )

    print("ptodsl_ptoas_frontend_verify: PASS")


if __name__ == "__main__":
    main()
