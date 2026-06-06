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
from importlib.util import module_from_spec, spec_from_file_location


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


def emit_example_mlir(example_path: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(example_path), "--emit-mlir"],
        capture_output=True,
        text=True,
        check=False,
    )
    expect(
        result.returncode == 0,
        f"{example_path.name} --emit-mlir should succeed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )
    expect(result.stdout.strip(), f"{example_path.name} --emit-mlir should print non-empty MLIR text")
    return result.stdout


def load_example_module(example_path: Path, module_name: str):
    spec = spec_from_file_location(module_name, example_path)
    expect(spec is not None and spec.loader is not None, f"unable to create import spec for {example_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def run_ptoas_frontend_verify_whole(ptoas_bin: Path, mlir_text: str, label: str) -> str:
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
        f"{label} should pass PTOAS frontend verification as one container.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )
    expect(result.stdout.strip(), f"{label} should emit non-empty PTO IR after PTOAS frontend passes")
    return result.stdout


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


@pto.jit(target="a5", entry=False, backend="vpto", mode="explicit", insert_sync=False)
def process_row_ptr_kernel_module(
    src_gm: pto.ptr(pto.f32, "gm"),
    dst_gm: pto.ptr(pto.f32, "gm"),
    row: pto.i32,
):
    with pto.simd():
        c0_i64 = pto.const(0, dtype=pto.i64)
        row_offset = row * 16
        src_row = pto.addptr(src_gm, row_offset)
        dst_row = pto.addptr(dst_gm, row_offset)
        ub_ptr = pto.castptr(c0_i64, pto.ptr(pto.f32, "ub"))

        pto.get_buf(pto.Pipe.MTE2, 0)
        pto.mte_gm_ub(src_row, ub_ptr, 0, 64, nburst=(1, 64, 64))
        pto.rls_buf(pto.Pipe.MTE2, 0)

        pto.get_buf(pto.Pipe.MTE3, 0)
        pto.mte_ub_gm(ub_ptr, dst_row, 64, nburst=(1, 64, 64))
        pto.rls_buf(pto.Pipe.MTE3, 0)
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5", backend="emitc")
def emitc_entry_calls_vpto_kernel_module_probe(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, 16], strides=[16, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, 16], strides=[16, 1])
    a_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, 16], dtype=pto.f32)

    with pto.for_(0, rows, step=1) as row:
        a_part = pto.partition_view(a_view, offsets=[row, 0], sizes=[1, 16])
        o_part = pto.partition_view(o_view, offsets=[row, 0], sizes=[1, 16])
        pto.tile.load(a_part, a_tile)
        pto.tile.adds(a_tile, 1.0, o_tile)
        pto.tile.store(o_tile, o_part)
        process_row_ptr_kernel_module(A_ptr, O_ptr, row)


PTR_LIKE_TILE_BUF_ADDR_MLIR = """
module attributes {pto.backend = "emitc", pto.target_arch = "a5"} {
  func.func private @consume(%arg0: !pto.ptr<f32, ub>)
  func.func @ptr_like_tile_buf_addr_probe() attributes {pto.aicore} {
    %tile = pto.alloc_tile : !pto.tile_buf<vec, 1x16xf32>
    %ptr = pto.tile_buf_addr %tile : !pto.tile_buf<vec, 1x16xf32> -> !pto.ptr<f32, ub>
    func.call @consume(%ptr) : (!pto.ptr<f32, ub>) -> ()
    return
  }
}
"""


def main() -> None:
    ptoas_bin = resolve_ptoas_binary()
    mixed_backend_example = REPO_ROOT / "ptodsl" / "examples" / "mixed_backend_kernel_module.py"
    cv_split_example = REPO_ROOT / "ptodsl" / "examples" / "hw_native_flash_attention_cv_split.py"
    fa_dn_ptodsl_example = REPO_ROOT / "ptodsl" / "examples" / "fa_dn_ptodsl.py"

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

    mixed_backend_text = emitc_entry_calls_vpto_kernel_module_probe.compile().mlir_text()
    mixed_backend_frontend_texts = run_ptoas_frontend_verify(
        ptoas_bin,
        mixed_backend_text,
        "emitc_entry_calls_vpto_kernel_module_probe PTODSL artifact",
    )
    expect(
        len(mixed_backend_frontend_texts) == 2,
        "mixed-backend PTODSL artifact should lower to separate caller/callee child modules",
    )
    mixed_backend_emitc_frontend_text = mixed_backend_frontend_texts[0]
    expect(
        'module attributes {pto.backend = "emitc", pto.target_arch = "a5"}'
        in mixed_backend_emitc_frontend_text,
        "mixed-backend caller child should stay on the EmitC backend through PTOAS frontend verification",
    )
    expect(
        "func.func @emitc_entry_calls_vpto_kernel_module_probe"
        in mixed_backend_emitc_frontend_text,
        "mixed-backend caller frontend verification output should preserve the entry symbol",
    )
    expect(
        mixed_backend_emitc_frontend_text.count("pto.section.vector {") == 1,
        "mixed-backend caller frontend verification should infer exactly one top-level vector section for the uncovered entry tile path",
    )
    expect(
        "pto.tload" in mixed_backend_emitc_frontend_text
        and "pto.tadds" in mixed_backend_emitc_frontend_text
        and "pto.tstore" in mixed_backend_emitc_frontend_text,
        "mixed-backend caller frontend verification output should preserve the entry tile path after inferred section normalization",
    )
    expect(
        "func.call @process_row_ptr_kernel_module__ptodsl_" in mixed_backend_emitc_frontend_text,
        "mixed-backend caller frontend verification output should keep the kernel-module call alongside the normalized tile path",
    )
    expect(
        mixed_backend_emitc_frontend_text.index("pto.section.vector {")
        < mixed_backend_emitc_frontend_text.index("pto.tload"),
        "mixed-backend caller frontend verification should place the entry tile path inside the inferred vector section",
    )
    expect(
        mixed_backend_frontend_texts[1] == "",
        "mixed-backend VPTO callee child should continue to compile through the fallback object path when --emit-pto-ir is unavailable",
    )

    ptr_like_addr_text = PTR_LIKE_TILE_BUF_ADDR_MLIR
    ptr_like_addr_frontend_texts = run_ptoas_frontend_verify(
        ptoas_bin,
        ptr_like_addr_text,
        "ptr_like_tile_buf_addr_probe IR artifact",
    )
    expect(
        len(ptr_like_addr_frontend_texts) == 1,
        "ptr_like_tile_buf_addr_probe should lower to exactly one backend child module",
    )
    ptr_like_addr_frontend_text = ptr_like_addr_frontend_texts[0]
    expect(
        "func.func @ptr_like_tile_buf_addr_probe" in ptr_like_addr_frontend_text,
        "ptr_like_tile_buf_addr_probe frontend verification output should preserve the kernel symbol",
    )
    expect(
        "memref<?xf32" in ptr_like_addr_frontend_text,
        "ptr-like tile_buf_addr lowering should materialize one memref<?xf32> address view during PTOViewToMemref",
    )
    expect(
        "call @consume" in ptr_like_addr_frontend_text,
        "ptr-like tile_buf_addr lowering should preserve call users after converting pointer-like operands",
    )

    example_mlir_text = emit_example_mlir(mixed_backend_example)
    example_child_texts = extract_child_module_texts(
        example_mlir_text,
        "mixed_backend_kernel_module.py --emit-mlir output",
    )
    expect(
        len(example_child_texts) == 2,
        "mixed_backend_kernel_module.py should materialize one EmitC caller child and one VPTO callee child",
    )
    example_emitc_child = example_child_texts[0]
    example_vpto_child = example_child_texts[1]
    expect(
        'module attributes {pto.backend = "emitc", pto.target_arch = "a5"}'
        in example_emitc_child,
        "mixed_backend_kernel_module.py EmitC child should carry the authored EmitC backend metadata",
    )
    expect(
        "func.func @emitc_entry_calls_vpto_module" in example_emitc_child
        and "func.func private @scale_row_kernel_module__ptodsl_" in example_emitc_child,
        "mixed_backend_kernel_module.py EmitC child should resemble mixed-external-vadd by keeping an entry symbol plus one private imported helper symbol",
    )
    expect(
        "pto.tload" in example_emitc_child
        and "pto.tadds" in example_emitc_child
        and "pto.tstore" in example_emitc_child
        and "func.call @scale_row_kernel_module__ptodsl_" in example_emitc_child,
        "mixed_backend_kernel_module.py EmitC child should keep the entry tile path plus one cross-backend helper call, like mixed-external-vadd's caller-side shape",
    )
    expect(
        'module attributes {pto.backend = "vpto", pto.target_arch = "a5"}'
        in example_vpto_child,
        "mixed_backend_kernel_module.py VPTO child should carry the callee backend metadata",
    )
    expect(
        "func.func public @scale_row_kernel_module__ptodsl_" in example_vpto_child
        and "pto.section.vector {" in example_vpto_child,
        "mixed_backend_kernel_module.py VPTO child should expose a public helper definition with explicit vector authoring, matching the vector-helper side of mixed-external-vadd",
    )

    example_frontend_texts = run_ptoas_frontend_verify(
        ptoas_bin,
        example_mlir_text,
        "mixed_backend_kernel_module.py --emit-mlir output",
    )
    expect(
        len(example_frontend_texts) == 2,
        "mixed_backend_kernel_module.py frontend verification should keep the two-child backend partition",
    )
    example_emitc_frontend_text = example_frontend_texts[0]
    expect(
        example_emitc_frontend_text.count("pto.section.vector {") == 1,
        "mixed_backend_kernel_module.py frontend verification should normalize the uncovered EmitC entry tile path into one vector section",
    )
    expect(
        example_emitc_frontend_text.index("pto.section.vector {")
        < example_emitc_frontend_text.index("pto.tload")
        < example_emitc_frontend_text.index("func.call @scale_row_kernel_module__ptodsl_"),
        "mixed_backend_kernel_module.py frontend verification should place the entry tile path and helper call inside the inferred caller-side vector section",
    )
    expect(
        example_frontend_texts[1] == "",
        "mixed_backend_kernel_module.py VPTO child should continue to compile through the fallback object path in frontend verification",
    )

    cv_split = load_example_module(
        cv_split_example,
        "ptodsl_hw_native_flash_attention_cv_split_example",
    )
    cv_split_text = cv_split.emit_flash_attention_mlir(
        head_dim=128,
        s1_tile=256,
        qk_preload=3,
        causal=False,
        q_rows=128,
    )
    cv_split_frontend_text = run_ptoas_frontend_verify_whole(
        ptoas_bin,
        cv_split_text,
        "hw_native_flash_attention_cv_split.py --emit-mlir output",
    )
    expect(
        cv_split_frontend_text.count('module attributes {pto.backend = "emitc", pto.target_arch = "a5"}') >= 3,
        "hw_native_flash_attention_cv_split.py frontend verification should preserve the outer entry child plus two EmitC helper children",
    )
    expect(
        "func.func public @hw_native_flash_attention_cv_split_cube_h128_s1t256_qp3_qr128__ptodsl_"
        in cv_split_frontend_text,
        "cv-split frontend verification should preserve the cube helper public ABI-specialized symbol",
    )
    expect(
        'pto.import_reserved_buffer{name = "fa_qk_c2v_fifo", peer_func = @hw_native_flash_attention_cv_split_vector_h128_s1t256_qp3_qr128}'
        in cv_split_frontend_text,
        "cv-split frontend verification should keep the logical vector peer_func reference that now resolves across helper containers",
    )
    expect(
        'pto.import_reserved_buffer{name = "fa_pv_c2v_fifo", peer_func = @hw_native_flash_attention_cv_split_vector_h128_s1t256_qp3_qr128}'
        in cv_split_frontend_text,
        "cv-split frontend verification should preserve the second logical vector peer_func reference",
    )
    expect(
        "pto.aic_initialize_pipe" in cv_split_frontend_text
        and "pto.tpush_to_aiv" in cv_split_frontend_text,
        "cv-split frontend verification should keep the cube helper pipe init and push path intact",
    )
    expect(
        "func.func public @hw_native_flash_attention_cv_split_vector_h128_s1t256_qp3_qr128__ptodsl_"
        in cv_split_frontend_text,
        "cv-split frontend verification should preserve the vector helper public ABI-specialized symbol",
    )
    expect(
        'pto.import_reserved_buffer{name = "fa_p_v2c_fifo", peer_func = @hw_native_flash_attention_cv_split_cube_h128_s1t256_qp3_qr128}'
        in cv_split_frontend_text,
        "cv-split frontend verification should keep the logical cube peer_func reference that now resolves across helper containers",
    )
    expect(
        "pto.reserve_buffer{name = \"fa_qk_c2v_fifo\"" in cv_split_frontend_text
        and "pto.reserve_buffer{name = \"fa_pv_c2v_fifo\"" in cv_split_frontend_text,
        "cv-split frontend verification should keep the local reserve_buffer owners for both imported cube peers",
    )
    expect(
        "pto.aiv_initialize_pipe" in cv_split_frontend_text
        and "pto.tpush_to_aic" in cv_split_frontend_text,
        "cv-split frontend verification should keep the vector helper pipe init and push path intact",
    )

    fa_dn_ptodsl = load_example_module(
        fa_dn_ptodsl_example,
        "ptodsl_fa_dn_ptodsl_example",
    )
    fa_dn_ptodsl_text = fa_dn_ptodsl.emit_fa_dn_mlir(
        head_dim=128,
        s1_tile=256,
        qk_preload=3,
        causal=False,
        q_rows=128,
    )
    fa_dn_ptodsl_frontend_text = run_ptoas_frontend_verify_whole(
        ptoas_bin,
        fa_dn_ptodsl_text,
        "fa_dn_ptodsl.py --emit-mlir output",
    )
    expect(
        fa_dn_ptodsl_frontend_text.count('module attributes {pto.backend = "emitc", pto.target_arch = "a5"}') >= 3,
        "fa_dn_ptodsl.py frontend verification should preserve the outer entry child plus two EmitC helper children",
    )
    expect(
        "func.func public @fa_dn_ptodsl_cube_h128_s1t256_qp3_qr128__ptodsl_"
        in fa_dn_ptodsl_frontend_text,
        "fa_dn_ptodsl frontend verification should preserve the cube helper public ABI-specialized symbol",
    )
    expect(
        'pto.import_reserved_buffer{name = "fa_dn_qk_c2v_fifo", peer_func = @fa_dn_ptodsl_vector_h128_s1t256_qp3_qr128}'
        in fa_dn_ptodsl_frontend_text,
        "fa_dn_ptodsl frontend verification should keep the QK logical vector peer reference",
    )
    expect(
        'pto.import_reserved_buffer{name = "fa_dn_pv_c2v_fifo", peer_func = @fa_dn_ptodsl_vector_h128_s1t256_qp3_qr128}'
        in fa_dn_ptodsl_frontend_text,
        "fa_dn_ptodsl frontend verification should keep the PV logical vector peer reference",
    )
    expect(
        'pto.import_reserved_buffer{name = "fa_dn_p_v2c_fifo", peer_func = @fa_dn_ptodsl_cube_h128_s1t256_qp3_qr128}'
        in fa_dn_ptodsl_frontend_text,
        "fa_dn_ptodsl frontend verification should keep the P logical cube peer reference",
    )
    expect(
        "pto.aic_initialize_pipe" in fa_dn_ptodsl_frontend_text
        and "pto.aiv_initialize_pipe" in fa_dn_ptodsl_frontend_text,
        "fa_dn_ptodsl frontend verification should keep both cube and vector pipe initialization paths",
    )

    print("ptodsl_ptoas_frontend_verify: PASS")


if __name__ == "__main__":
    main()
