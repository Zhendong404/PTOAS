#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ptodsl"))

from ptodsl import pto
from ptodsl._bootstrap import make_context
from mlir.ir import Module


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


def extract_child_module_texts(container_text: str, label: str) -> list[str]:
    with make_context() as ctx:
        parsed = Module.parse(container_text, ctx)
        top_level_ops = list(parsed.body.operations)
    expect(top_level_ops, f"{label} should contain at least one top-level operation")
    if all(op.operation.name == "builtin.module" for op in top_level_ops):
        return [str(op) for op in top_level_ops]
    return [container_text]


def run_ptoas_frontend_verify(ptoas_bin: Path, mlir_text: str, label: str) -> list[str]:
    child_modules = extract_child_module_texts(mlir_text, label)
    frontend_texts: list[str] = []

    for index, child_text in enumerate(child_modules, start=1):
        with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False, encoding="utf-8") as handle:
            handle.write(child_text)
            input_path = Path(handle.name)

        child_label = f"{label} [child {index}]"
        try:
            result = subprocess.run(
                [str(ptoas_bin), str(input_path), "--emit-pto-ir", "-o", "-"],
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            input_path.unlink(missing_ok=True)

        if result.returncode == 0 and result.stdout.strip():
            frontend_texts.append(result.stdout)
            continue

        if "object output requires an explicit file path passed with -o." in result.stderr:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_root = Path(temp_dir)
                fallback_input_path = temp_root / "input.mlir"
                output_path = temp_root / "kernel.o"
                fallback_input_path.write_text(child_text, encoding="utf-8")
                fallback_result = subprocess.run(
                    [str(ptoas_bin), str(fallback_input_path), "-o", str(output_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                artifact_exists = output_path.is_file()
                artifact_size = output_path.stat().st_size if artifact_exists else 0
            expect(
                fallback_result.returncode == 0,
                f"{child_label} should pass PTOAS fallback compilation when the VPTO fast path skips --emit-pto-ir.\n"
                f"stdout:\n{fallback_result.stdout}\nstderr:\n{fallback_result.stderr}",
            )
            expect(artifact_exists, f"{child_label} should produce an output artifact via fallback ptoas -o")
            expect(
                artifact_size > 0,
                f"{child_label} should produce a non-empty output artifact via fallback ptoas -o",
            )
            frontend_texts.append("")
            continue

        expect(
            result.returncode == 0,
            f"{child_label} should pass PTOAS frontend verification.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        expect(result.stdout.strip(), f"{child_label} should emit non-empty PTO IR after PTOAS frontend passes")
        frontend_texts.append(result.stdout)

    return frontend_texts


@pto.jit(target="a5")
def host_vec_copy(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.constexpr = 128,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    out = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    pto.tile.load(part, a_tile)
    pto.tile.store(o_tile, out)


def main() -> None:
    ptoas_bin = resolve_ptoas_binary()

    simple_text = host_vec_copy.compile().mlir_text()
    simple_frontend_texts = run_ptoas_frontend_verify(
        ptoas_bin,
        simple_text,
        "host_vec_copy PTODSL artifact",
    )
    expect(
        len(simple_frontend_texts) == 1,
        "host_vec_copy PTODSL artifact should lower to exactly one backend child module",
    )
    simple_frontend_text = simple_frontend_texts[0]
    expect(
        "func.func @host_vec_copy" in simple_frontend_text,
        "host_vec_copy frontend verification output should preserve the kernel symbol",
    )
    expect(
        "pto.tload" in simple_frontend_text and "pto.tstore" in simple_frontend_text,
        "host_vec_copy frontend verification output should keep the tile IO contract visible",
    )

    print("ptodsl_ptoas_frontend_verify: PASS")


if __name__ == "__main__":
    main()
