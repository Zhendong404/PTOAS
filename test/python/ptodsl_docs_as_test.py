#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import re
import shutil
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
USER_GUIDE_ROOT = REPO_ROOT / "ptodsl" / "docs" / "user_guide"
sys.path.insert(0, str(REPO_ROOT / "ptodsl"))

from ptodsl._bootstrap import make_context
from mlir.ir import Module

FENCE_RE = re.compile(r"^```(?P<lang>[A-Za-z0-9_+-]*)\s*$")
META_RE = re.compile(r"^\s*<!--\s*ptodsl-doc-(?P<kind>test|ignore)\s*:\s*(?P<body>.*?)\s*-->\s*$")


@dataclass(frozen=True)
class MarkdownCodeBlock:
    path: Path
    start_line: int
    end_line: int
    language: str
    lines: tuple[str, ...]
    metadata: "DocBlockMetadata | None"

    @property
    def text(self) -> str:
        return "".join(self.lines)


@dataclass(frozen=True)
class MarkdownScanResult:
    path: Path
    blocks: tuple[MarkdownCodeBlock, ...]


@dataclass(frozen=True)
class DocBlockMetadata:
    kind: str
    body: str
    line: int
    raw: str


@dataclass(frozen=True)
class DocTestDirective:
    mode: str
    symbol: str
    compile_kwargs: dict[str, object]


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def format_doc_context(path: Path, start_line: int, symbol: str | None = None) -> str:
    symbol_text = symbol if symbol is not None else "<unknown>"
    return f"{path}:{start_line} [symbol={symbol_text}]"


def fail_doc(path: Path, start_line: int, message: str, symbol: str | None = None) -> None:
    raise AssertionError(f"{format_doc_context(path, start_line, symbol)}: {message}")


def iter_markdown_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.glob("*.md"))


def parse_metadata_line(path: Path, line: str, line_number: int) -> DocBlockMetadata | None:
    match = META_RE.match(line)
    if match is None:
        return None

    kind = match.group("kind")
    body = match.group("body").strip()
    expect(body, f"{format_doc_context(path, line_number)}: ptodsl-doc-{kind} metadata must not be empty")
    if kind == "test":
        try:
            json.loads(body)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"{format_doc_context(path, line_number)}: ptodsl-doc-test metadata must be valid JSON: {exc.msg}"
            ) from exc
    return DocBlockMetadata(kind=kind, body=body, line=line_number, raw=line.rstrip("\n"))


def find_block_metadata(path: Path, lines: list[str], fence_line: int) -> DocBlockMetadata | None:
    candidate = fence_line - 2
    while candidate >= 0 and not lines[candidate].strip():
        candidate -= 1
    if candidate < 0:
        return None
    line = lines[candidate]
    if line.lstrip().startswith("<!-- ptodsl-doc-") and parse_metadata_line(path, line, candidate + 1) is None:
        fail_doc(path, fence_line, "malformed ptodsl-doc metadata comment")
    return parse_metadata_line(path, line, candidate + 1)


def block_label(block: MarkdownCodeBlock, symbol: str | None = None) -> str:
    return format_doc_context(block.path, block.start_line, symbol)


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


def expect_parse_roundtrip_and_verify(text: str, label: str) -> None:
    with make_context() as ctx:
        parsed = Module.parse(text, ctx)
        parsed.operation.verify()
        roundtrip_text = str(parsed)
    expect(
        roundtrip_text == text,
        f"{label} should survive Module.parse(...) round-trip without textual drift",
    )


def run_ptoas_frontend_verify(ptoas_bin: Path, mlir_text: str, label: str) -> None:
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


def parse_test_directive(block: MarkdownCodeBlock) -> DocTestDirective:
    expect(block.metadata is not None, f"{block_label(block)}: python code block missing metadata")
    expect(block.metadata.kind == "test", f"{block_label(block)}: expected ptodsl-doc-test metadata")

    try:
        payload = json.loads(block.metadata.body)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"{block_label(block)}: ptodsl-doc-test metadata must be valid JSON: {exc.msg}"
        ) from exc

    expect(
        isinstance(payload, dict),
        f"{block_label(block)}: ptodsl-doc-test metadata must be a JSON object",
    )

    mode = payload.get("mode")
    symbol = payload.get("symbol")
    compile_kwargs = payload.get("compile")

    expect(
        isinstance(mode, str) and mode,
        f"{block_label(block)}: ptodsl-doc-test metadata must define a non-empty string 'mode'",
    )
    expect(
        isinstance(symbol, str) and symbol,
        f"{block_label(block)}: ptodsl-doc-test metadata must define a non-empty string 'symbol'",
    )
    expect(
        isinstance(compile_kwargs, dict),
        f"{block_label(block, symbol if isinstance(symbol, str) and symbol else None)}: "
        "ptodsl-doc-test metadata must define an object 'compile'",
    )
    expect(
        mode == "compile",
        f"{block_label(block, symbol if isinstance(symbol, str) and symbol else None)}: "
        f"unsupported ptodsl-doc-test mode {mode!r}; only 'compile' is supported",
    )
    return DocTestDirective(mode=mode, symbol=symbol, compile_kwargs=compile_kwargs)


def execute_snippet(block: MarkdownCodeBlock, symbol: str | None = None) -> dict[str, object]:
    namespace: dict[str, object] = {
        "__builtins__": __builtins__,
        "__name__": "__ptodsl_doc_snippet__",
        "__file__": str(block.path),
    }
    try:
        exec(compile(block.text, str(block.path), "exec"), namespace, namespace)
    except Exception as exc:
        raise AssertionError(
            f"{block_label(block, symbol)}: snippet execution failed: {exc.__class__.__name__}: {exc}"
        ) from exc
    return namespace


def run_compile_block(block: MarkdownCodeBlock, ptoas_bin: Path) -> None:
    directive = parse_test_directive(block)
    namespace = execute_snippet(block, directive.symbol)

    expect(
        directive.symbol in namespace,
        f"{block_label(block, directive.symbol)}: declared symbol is missing from snippet namespace",
    )

    target = namespace[directive.symbol]
    compile_attr = getattr(target, "compile", None)
    expect(
        callable(compile_attr),
        f"{block_label(block, directive.symbol)}: declared symbol does not expose a callable .compile(...) surface",
    )

    try:
        compiled = compile_attr(**directive.compile_kwargs)
    except Exception as exc:
        raise AssertionError(
            f"{block_label(block, directive.symbol)}: compile failed: {exc.__class__.__name__}: {exc}"
        ) from exc

    try:
        compiled.verify()
    except Exception as exc:
        raise AssertionError(
            f"{block_label(block, directive.symbol)}: compiled.verify() failed: {exc.__class__.__name__}: {exc}"
        ) from exc

    mlir_text = compiled.mlir_text()
    expect(
        isinstance(mlir_text, str) and mlir_text.strip(),
        f"{block_label(block, directive.symbol)}: compiled artifact should expose non-empty mlir_text()",
    )

    label = block_label(block, directive.symbol)
    expect_parse_roundtrip_and_verify(mlir_text, label)
    run_ptoas_frontend_verify(ptoas_bin, mlir_text, label)


def scan_markdown_file(path: Path) -> MarkdownScanResult:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    blocks: list[MarkdownCodeBlock] = []
    in_code_block = False
    block_language = ""
    block_start = 0
    block_lines: list[str] = []
    metadata: DocBlockMetadata | None = None

    for index, line in enumerate(lines, start=1):
        fence_match = FENCE_RE.match(line.rstrip("\n"))
        if fence_match:
            if not in_code_block:
                in_code_block = True
                block_language = fence_match.group("lang")
                block_start = index
                block_lines = []
                metadata = find_block_metadata(path, lines, index)
                if block_language == "python":
                    expect(
                        metadata is not None,
                        f"{format_doc_context(path, index)}: "
                        "python code block must be preceded by ptodsl-doc-test or ptodsl-doc-ignore metadata",
                    )
            else:
                blocks.append(
                    MarkdownCodeBlock(
                        path=path,
                        start_line=block_start,
                        end_line=index,
                        language=block_language,
                        lines=tuple(block_lines),
                        metadata=metadata,
                    )
                )
                in_code_block = False
                block_language = ""
                block_start = 0
                block_lines = []
                metadata = None
            continue

        if in_code_block:
            block_lines.append(line)

    expect(not in_code_block, f"unclosed fenced code block in {path}")
    return MarkdownScanResult(path=path, blocks=tuple(blocks))


def scan_user_guide() -> tuple[MarkdownScanResult, ...]:
    return tuple(scan_markdown_file(path) for path in iter_markdown_files(USER_GUIDE_ROOT))


def collect_python_blocks(results: Iterable[MarkdownScanResult]) -> tuple[MarkdownCodeBlock, ...]:
    blocks: list[MarkdownCodeBlock] = []
    for result in results:
        for block in result.blocks:
            if block.language == "python":
                blocks.append(block)
    return tuple(blocks)


def summarize_metadata(blocks: Iterable[MarkdownCodeBlock]) -> tuple[int, int]:
    test_count = 0
    ignore_count = 0
    for block in blocks:
        expect(block.metadata is not None, f"{block.path}:{block.start_line}: python code block missing metadata")
        if block.metadata.kind == "test":
            test_count += 1
        elif block.metadata.kind == "ignore":
            ignore_count += 1
        else:
            raise AssertionError(
                f"{block_label(block)}: unsupported ptodsl-doc metadata kind {block.metadata.kind!r}"
            )
    return test_count, ignore_count


def collect_test_blocks(blocks: Iterable[MarkdownCodeBlock]) -> tuple[MarkdownCodeBlock, ...]:
    return tuple(
        block
        for block in blocks
        if block.metadata is not None and block.metadata.kind == "test"
    )


def main() -> None:
    expect(USER_GUIDE_ROOT.is_dir(), f"missing PTODSL user guide directory: {USER_GUIDE_ROOT}")

    results = scan_user_guide()
    python_blocks = collect_python_blocks(results)
    test_count, ignore_count = summarize_metadata(python_blocks)
    test_blocks = collect_test_blocks(python_blocks)

    expect(bool(results), f"no markdown files found under {USER_GUIDE_ROOT}")
    expect(bool(python_blocks), f"no Python fenced code blocks found under {USER_GUIDE_ROOT}")

    if test_blocks:
        try:
            ptoas_bin = resolve_ptoas_binary()
        except FileNotFoundError as exc:
            fail_doc(test_blocks[0].path, test_blocks[0].start_line, str(exc))
    else:
        ptoas_bin = None
    for block in test_blocks:
        expect(ptoas_bin is not None, f"{block_label(block)}: missing ptoas binary for compile-mode docs test")
        run_compile_block(block, ptoas_bin)

    markdown_count = len(results)
    python_count = len(python_blocks)
    block_count = sum(len(result.blocks) for result in results)
    print(
        "ptodsl_docs_as_test: scanned "
        f"{markdown_count} markdown files, {block_count} fenced blocks, {python_count} python blocks "
        f"({test_count} test, {ignore_count} ignore)"
    )


if __name__ == "__main__":
    main()
