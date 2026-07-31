# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Conservatively select ci-sim suites from changed repository paths."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


SUITES = ("pypto", "tileop", "vpto", "tilelib", "ptodsl")
DIRECT_SUITE_ROOTS = {
    "test/vpto/": "vpto",
    "test/tilelib-st/": "tilelib",
    "test/dsl-st/": "ptodsl",
    "test/tilelang_st/": "tileop",
}
NON_CODE_ROOTS = ("docs/", "openspec/", ".github/ISSUE_TEMPLATE/")
NON_CODE_FILES = {
    ".editorconfig",
    ".gitattributes",
    ".gitignore",
    ".mailmap",
    ".github/CODEOWNERS",
}
NON_CODE_BASENAMES = {"LICENSE", "NOTICE"}
NON_CODE_SUFFIXES = {".md"}


@dataclass(frozen=True)
class Selection:
    suites: dict[str, bool]
    matched_paths: tuple[str, ...]
    reason: str

    @property
    def wheel(self) -> bool:
        return any(self.suites.values())

    def github_outputs(self) -> dict[str, str]:
        outputs = {"wheel": str(self.wheel).lower()}
        outputs.update({suite: str(selected).lower() for suite, selected in self.suites.items()})
        outputs["matched_paths"] = json.dumps(self.matched_paths, separators=(",", ":"))
        outputs["selection_reason"] = self.reason
        return outputs


def _normalize_path(path: str) -> str:
    normalized = PurePosixPath(path.strip().replace("\\", "/")).as_posix()
    return normalized.removeprefix("./")


def _is_non_code(path: str) -> bool:
    if path in NON_CODE_FILES or any(path.startswith(root) for root in NON_CODE_ROOTS):
        return True
    basename = PurePosixPath(path).name
    if basename in NON_CODE_BASENAMES or any(
        basename.startswith(f"{name}.") for name in NON_CODE_BASENAMES
    ):
        return True
    return PurePosixPath(path).suffix.lower() in NON_CODE_SUFFIXES


def classify(event_name: str, paths: Iterable[str]) -> Selection:
    normalized_paths = tuple(sorted({_normalize_path(path) for path in paths if path.strip()}))
    if event_name != "pull_request":
        return Selection(
            suites={suite: True for suite in SUITES},
            matched_paths=normalized_paths,
            reason=f"{event_name} events always run all ci-sim suites",
        )

    direct_suites: set[str] = set()
    direct_paths: list[str] = []
    non_code_paths: list[str] = []
    shared_or_unknown_paths: list[str] = []
    for path in normalized_paths:
        owner = next(
            (suite for root, suite in DIRECT_SUITE_ROOTS.items() if path.startswith(root)),
            None,
        )
        if owner:
            direct_suites.add(owner)
            direct_paths.append(path)
        elif _is_non_code(path):
            non_code_paths.append(path)
        else:
            shared_or_unknown_paths.append(path)

    if shared_or_unknown_paths:
        reason = "shared or unknown paths require all suites: " + json.dumps(
            shared_or_unknown_paths, separators=(",", ":")
        )
        return Selection(
            suites={suite: True for suite in SUITES},
            matched_paths=normalized_paths,
            reason=reason,
        )
    if direct_suites:
        selected = ", ".join(suite for suite in SUITES if suite in direct_suites)
        return Selection(
            suites={suite: suite in direct_suites for suite in SUITES},
            matched_paths=tuple(direct_paths),
            reason=f"direct suite-owned test paths selected: {selected}",
        )
    return Selection(
        suites={suite: False for suite in SUITES},
        matched_paths=tuple(non_code_paths),
        reason="all changed paths are explicitly classified as non-code",
    )


def _read_paths(path_file: Path | None, null_delimited: bool) -> list[str]:
    if path_file is None:
        return []
    contents = path_file.read_bytes()
    separator = b"\0" if null_delimited else b"\n"
    return [entry.decode("utf-8", errors="surrogateescape") for entry in contents.split(separator) if entry]


def _write_github_outputs(outputs: dict[str, str], output_path: Path) -> None:
    with output_path.open("a", encoding="utf-8") as stream:
        for name, value in outputs.items():
            print(f"{name}={value}", file=stream)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--paths-file", type=Path)
    parser.add_argument("--null", action="store_true", help="Read NUL-delimited paths")
    parser.add_argument(
        "--github-output",
        type=Path,
        default=Path(os.environ["GITHUB_OUTPUT"]) if "GITHUB_OUTPUT" in os.environ else None,
    )
    args = parser.parse_args()

    selection = classify(args.event_name, _read_paths(args.paths_file, args.null))
    outputs = selection.github_outputs()
    if args.github_output:
        _write_github_outputs(outputs, args.github_output)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
