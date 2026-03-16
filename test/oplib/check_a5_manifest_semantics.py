#!/usr/bin/env python3
"""Validate A5 OpLib manifest evidence against the sibling pto-isa tree."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


APPROVED_PUBLIC_REWRITES = {
    "TRECIP": "TDIVS",
}

TARGETED_DTYPE_HEADERS = {
    "trecip": "include/pto/npu/a5/TDivS.hpp",
    "tand": "include/pto/npu/a5/TAnd.hpp",
    "tor": "include/pto/npu/a5/TOr.hpp",
    "txor": "include/pto/npu/a5/TXor.hpp",
    "tnot": "include/pto/npu/a5/TUnaryOp.hpp",
    "tshl": "include/pto/npu/a5/TShl.hpp",
    "tshr": "include/pto/npu/a5/TShr.hpp",
}

SUSPICIOUS_CONSTRAINT_TOKENS = (
    "MERCHANTABILITY",
    "#ifndef",
    "namespace pto",
    "PTO_UTILS_H",
)

CPP_TO_MANIFEST_DTYPE = {
    "uint8_t": "u8",
    "int8_t": "i8",
    "uint16_t": "u16",
    "int16_t": "i16",
    "uint32_t": "u32",
    "int32_t": "i32",
    "int": "i32",
    "half": "f16",
    "float16_t": "f16",
    "float": "f32",
    "float32_t": "f32",
}

DTYPE_ORDER = {
    "bf16": 0,
    "f16": 1,
    "f32": 2,
    "i16": 3,
    "i32": 4,
    "i8": 5,
    "u16": 6,
    "u32": 7,
    "u8": 8,
}


@dataclass(frozen=True)
class PublicApiEvidence:
    op: str
    line: int
    target: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that the A5 OpLib manifest keeps native A5 evidence, "
            "approved public API rewrites, and deferred boundaries aligned "
            "with the sibling pto-isa tree."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pto-isa-root", type=Path)
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def find_repo_root(start: Path) -> Path | None:
    for candidate in [start, *start.parents]:
        if (candidate / "docs" / "PTO_IR_manual.md").exists():
            return candidate
    return None


def discover_pto_isa_root(manifest_path: Path, explicit_root: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(explicit_root)
    for env_name in ("PTO_ISA_PATH", "PTO_ISA_ROOT"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value))

    repo_root = find_repo_root(manifest_path.resolve())
    if repo_root is not None:
        candidates.append(repo_root.parent / "pto-isa")

    seen: set[Path] = set()
    for raw in candidates:
        candidate = raw.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "include" / "pto" / "common" / "pto_instr.hpp").exists() and (
            candidate / "tests" / "npu" / "a5" / "src" / "st" / "testcase"
        ).exists():
            return candidate

    raise SystemExit(
        "failed to resolve pto-isa root; pass --pto-isa-root or export PTO_ISA_PATH/PTO_ISA_ROOT"
    )


def relpath(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def sort_dtypes(values: set[str] | list[str]) -> list[str]:
    return sorted(values, key=lambda value: (DTYPE_ORDER.get(value, 999), value))


def parse_public_api_evidence(common_header: Path) -> dict[str, PublicApiEvidence]:
    lines = common_header.read_text().splitlines()
    evidence: dict[str, PublicApiEvidence] = {}
    signature = re.compile(r"PTO_INST\s+RecordEvent\s+([A-Z0-9_]+)\s*\(")

    idx = 0
    while idx < len(lines):
        match = signature.search(lines[idx])
        if match is None:
            idx += 1
            continue

        op = match.group(1)
        line_no = idx + 1
        brace_depth = 0
        body_lines: list[str] = []

        while idx + 1 < len(lines):
            idx += 1
            line = lines[idx]
            body_lines.append(line)
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0 and "}" in line:
                break

        body = "\n".join(body_lines)
        target_match = re.search(r"MAP_INSTR_IMPL\(\s*([A-Z0-9_]+)\s*,", body)
        target = target_match.group(1) if target_match else None
        evidence[op] = PublicApiEvidence(op=op, line=line_no, target=target)
        idx += 1

    return evidence


def collect_native_evidence(a5_include_dir: Path, ops_upper: set[str]) -> dict[str, list[str]]:
    evidence: dict[str, list[str]] = {op: [] for op in ops_upper}
    for header in sorted(a5_include_dir.rglob("*.hpp")):
        text = header.read_text(errors="ignore")
        rel = relpath(a5_include_dir.parent.parent.parent, header)
        for op in ops_upper:
            if re.search(rf"\bOP_NAME\(\s*{re.escape(op)}\s*\)", text) or re.search(
                rf"\b{re.escape(op)}_IMPL\b", text
            ):
                evidence[op].append(rel)
    return evidence


def extract_supported_dtypes(header_text: str) -> list[str]:
    dtypes: set[str] = set()
    blocks = re.findall(r"static_assert\((.*?)\);", header_text, flags=re.S)
    for block in blocks:
        if "invalid data type" not in block.lower():
            continue
        compact = re.sub(r"\s+", "", block)
        if "sizeof(T)==4||sizeof(T)==2||sizeof(T)==1" in compact:
            dtypes.update({"i8", "u8", "i16", "u16", "i32", "u32"})
        for cpp_type, manifest_dtype in CPP_TO_MANIFEST_DTYPE.items():
            if re.search(rf"\b{re.escape(cpp_type)}\b", block):
                dtypes.add(manifest_dtype)
    return sort_dtypes(dtypes)


def extract_supported_dtypes_for_op(header_text: str, op_upper: str) -> list[str]:
    impl_match = re.search(rf"\b{re.escape(op_upper)}_IMPL\b", header_text)
    op_name_match = re.search(rf"\bOP_NAME\(\s*{re.escape(op_upper)}\s*\)", header_text)
    anchor = impl_match.start() if impl_match else (op_name_match.start() if op_name_match else -1)
    if anchor < 0:
        return extract_supported_dtypes(header_text)

    start = header_text.rfind("/*", 0, anchor)
    if start < 0:
        start = header_text.rfind("template <", 0, anchor)
    if start < 0:
        start = 0
    end = header_text.find("/*", anchor + 1)
    if end < 0:
        end = len(header_text)
    return extract_supported_dtypes(header_text[start:end])


def classify_entry(op_upper: str, native_headers: list[str], public_api: PublicApiEvidence | None) -> str:
    if native_headers:
        return "native_a5_impl"
    if public_api and APPROVED_PUBLIC_REWRITES.get(op_upper) == public_api.target:
        return "public_api_rewrite"
    return "missing_accepted_semantics"


def ensure(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def contains_path(entries: list[str], expected: str) -> bool:
    return any(item.endswith(expected) for item in entries)


def validate_generated_from(manifest: dict, errors: list[str]) -> None:
    generated = manifest.get("generated_from", {})
    ensure(
        generated.get("common_api_header") == "include/pto/common/pto_instr.hpp",
        "manifest generated_from.common_api_header must point to include/pto/common/pto_instr.hpp",
        errors,
    )


def validate_constraints(op: str, key_constraints: list[str], errors: list[str]) -> None:
    ensure(bool(key_constraints), f"{op}: key_constraints must be non-empty", errors)
    for constraint in key_constraints:
        ensure(
            isinstance(constraint, str) and constraint.strip(),
            f"{op}: key_constraints must contain non-empty strings",
            errors,
        )
        ensure(
            not any(token in constraint for token in SUSPICIOUS_CONSTRAINT_TOKENS),
            f"{op}: key_constraints contains leaked source text instead of structured constraints",
            errors,
        )


def validate_targeted_dtype_support(
    entry: dict, pto_isa_root: Path, errors: list[str]
) -> None:
    op = entry["op"]
    header_rel = TARGETED_DTYPE_HEADERS.get(op)
    if header_rel is None:
        return
    header_path = pto_isa_root / header_rel
    target_op = APPROVED_PUBLIC_REWRITES.get(entry["instruction"], entry["instruction"])
    supported = extract_supported_dtypes_for_op(header_path.read_text(errors="ignore"), target_op)
    manifest_dtypes = sort_dtypes(entry.get("dtype_support", []))
    ensure(
        manifest_dtypes == supported,
        f"{op}: dtype_support {manifest_dtypes} does not match {header_rel} {supported}",
        errors,
    )


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    pto_isa_root = discover_pto_isa_root(manifest_path, args.pto_isa_root)
    common_header = pto_isa_root / "include" / "pto" / "common" / "pto_instr.hpp"
    testcase_root = pto_isa_root / "tests" / "npu" / "a5" / "src" / "st" / "testcase"
    a5_include_dir = pto_isa_root / "include" / "pto" / "npu" / "a5"

    public_api_evidence = parse_public_api_evidence(common_header)
    ops_upper = {entry["instruction"] for entry in manifest["operators"]}
    native_evidence = collect_native_evidence(a5_include_dir, ops_upper)

    errors: list[str] = []
    classification_counter: Counter[str] = Counter()

    validate_generated_from(manifest, errors)

    for entry in manifest["operators"]:
        op = entry["op"]
        op_upper = entry["instruction"]
        classification = classify_entry(
            op_upper, native_evidence.get(op_upper, []), public_api_evidence.get(op_upper)
        )
        classification_counter[classification] += 1

        expected_status = "implemented" if classification != "missing_accepted_semantics" else "deferred"
        ensure(
            entry.get("a5_status") == expected_status,
            f"{op}: expected a5_status={expected_status} from {classification}, got {entry.get('a5_status')}",
            errors,
        )

        validate_constraints(op, entry.get("key_constraints", []), errors)

        for test_path in entry.get("test_paths", []):
            ensure(
                (pto_isa_root / test_path).exists(),
                f"{op}: manifest test path does not exist: {test_path}",
                errors,
            )

        public_api = public_api_evidence.get(op_upper)
        native_headers = native_evidence.get(op_upper, [])

        if classification == "native_a5_impl":
            ensure(
                bool(native_headers),
                f"{op}: native_a5_impl classification requires include/pto/npu/a5 evidence",
                errors,
            )
            ensure(
                any(path.startswith("include/pto/npu/a5/") for path in entry.get("header_paths", [])),
                f"{op}: native_a5_impl entries must keep A5 header_paths evidence",
                errors,
            )
        elif classification == "public_api_rewrite":
            expected_target = APPROVED_PUBLIC_REWRITES[op_upper]
            ensure(
                public_api is not None and public_api.target == expected_target,
                f"{op}: public_api_rewrite classification requires include/pto/common/pto_instr.hpp rewrite to {expected_target}",
                errors,
            )
            ensure(
                contains_path(entry.get("header_paths", []), "include/pto/common/pto_instr.hpp"),
                f"{op}: public_api_rewrite entries must retain include/pto/common/pto_instr.hpp in header_paths",
                errors,
            )
            ensure(
                contains_path(
                    entry.get("semantic_source_paths", []),
                    f"include/pto/common/pto_instr.hpp:{public_api.line}",
                ),
                f"{op}: semantic_source_paths must retain the public API line include/pto/common/pto_instr.hpp:{public_api.line}",
                errors,
            )
            ensure(
                bool(entry.get("test_paths")) and (testcase_root / op).exists(),
                f"{op}: public_api_rewrite entries require sibling A5 ST testcase evidence",
                errors,
            )
        else:
            ensure(
                entry.get("deferred_reason"),
                f"{op}: deferred entries must keep a deferred_reason",
                errors,
            )
            if public_api is not None:
                ensure(
                    contains_path(entry.get("header_paths", []), "include/pto/common/pto_instr.hpp"),
                    f"{op}: deferred entries with public API evidence must keep include/pto/common/pto_instr.hpp in header_paths",
                    errors,
                )
                ensure(
                    contains_path(
                        entry.get("semantic_source_paths", []),
                        f"include/pto/common/pto_instr.hpp:{public_api.line}",
                    ),
                    f"{op}: deferred entries with public API evidence must keep the public API semantic_source_paths entry",
                    errors,
                )

        validate_targeted_dtype_support(entry, pto_isa_root, errors)

    if errors:
        print("A5 manifest semantic alignment failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        "A5 manifest semantic alignment OK: "
        f"native_a5_impl={classification_counter['native_a5_impl']}, "
        f"public_api_rewrite={classification_counter['public_api_rewrite']}, "
        f"missing_accepted_semantics={classification_counter['missing_accepted_semantics']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
