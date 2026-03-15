#!/usr/bin/env python3
"""Check that every implemented manifest op has templates and lowering coverage."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that every implemented A5 OpLib V1 operator has at least "
            "one concrete template and one lowering use case."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    return parser.parse_args()


def load_implemented_ops(manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text())
    return sorted(
        entry["op"]
        for entry in manifest["operators"]
        if entry.get("a5_status") == "implemented"
    )


def collect_template_hits(template_dir: Path) -> dict[str, list[str]]:
    hits: dict[str, set[str]] = defaultdict(set)
    pattern = re.compile(r'pto\.oplib\.op = "([^"]+)"')
    for template in sorted(template_dir.glob("*.mlir")):
        for match in pattern.finditer(template.read_text()):
            hits[match.group(1)].add(template.name)
    return {op: sorted(files) for op, files in hits.items()}


def collect_lowering_hits(test_dir: Path, implemented_ops: list[str]) -> dict[str, list[str]]:
    op_patterns = {
        op: re.compile(rf"\bpto\.{re.escape(op)}\b") for op in implemented_ops
    }
    hits: dict[str, set[str]] = defaultdict(set)

    harness_refs: dict[str, list[str]] = defaultdict(list)
    for harness in sorted(test_dir.glob("*.mlir")):
        text = harness.read_text()
        if "--op-lib-dir" not in text:
            continue
        for fixture in sorted(test_dir.glob("*.pto")):
            if fixture.name in text:
                harness_refs[fixture.name].append(harness.name)
        for op, pattern in op_patterns.items():
            if pattern.search(text):
                hits[op].add(harness.name)

    for fixture in sorted(test_dir.glob("*.pto")):
        used_by = sorted(set(harness_refs.get(fixture.name, [])))
        if not used_by:
            continue
        text = fixture.read_text()
        use_case = f"{fixture.name} <- {', '.join(used_by)}"
        for op, pattern in op_patterns.items():
            if pattern.search(text):
                hits[op].add(use_case)

    return {op: sorted(entries) for op, entries in hits.items()}


def main() -> int:
    args = parse_args()
    implemented_ops = load_implemented_ops(args.manifest)
    template_hits = collect_template_hits(args.template_dir)
    lowering_hits = collect_lowering_hits(args.test_dir, implemented_ops)

    missing_templates = [
        op for op in implemented_ops if not template_hits.get(op)
    ]
    missing_lowering = [
        op for op in implemented_ops if not lowering_hits.get(op)
    ]

    if missing_templates or missing_lowering:
        if missing_templates:
            print("implemented ops missing concrete templates:", file=sys.stderr)
            for op in missing_templates:
                print(f"  - {op}", file=sys.stderr)
        if missing_lowering:
            print("implemented ops missing lowering use cases:", file=sys.stderr)
            for op in missing_lowering:
                print(f"  - {op}", file=sys.stderr)
        return 1

    print(
        f"implemented ops aligned: {len(implemented_ops)} ops with concrete templates "
        "and lowering use cases"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
