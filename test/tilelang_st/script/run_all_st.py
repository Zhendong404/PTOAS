#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Batch runner for TileLang ST, suitable for CI/self-hosted runner usage."""

import argparse
import concurrent.futures
import os
import sys
import time
import traceback
from dataclasses import dataclass

import run_st


SOC_VERSION_MAP = {
    "a5": "Ascend950PR_9599",
}


@dataclass(frozen=True)
class ExecutionUnit:
    testcase: str
    case: str | None = None

    @property
    def label(self):
        if self.case is None:
            return self.testcase
        return f"{self.testcase}::{self.case}"


@dataclass(frozen=True)
class ExecutionResult:
    label: str
    log_path: str | None
    duration_seconds: float


def discover_testcases(testcase_root):
    testcases = []
    for entry in sorted(os.listdir(testcase_root)):
        testcase_dir = os.path.join(testcase_root, entry)
        if not os.path.isdir(testcase_dir):
            continue
        pto_file = os.path.join(testcase_dir, f"{entry}.pto")
        if os.path.isfile(pto_file):
            testcases.append(entry)
    return testcases


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all TileLang ST testcases for CI or local batch validation."
    )
    parser.add_argument(
        "-r", "--run-mode", default="sim",
        help="Run mode: sim or npu (default: sim)",
    )
    parser.add_argument(
        "-v", "--soc-version", default="a5",
        help="SoC version: a5 (default: a5)",
    )
    parser.add_argument(
        "-p", "--ptoas-bin", default=None,
        help="Path to ptoas binary (auto-detected if omitted)",
    )
    parser.add_argument(
        "-t", "--testcase", action="append", default=[],
        help="Run only selected testcase(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "-c", "--case", default=None,
        help="Run only a specific case within the selected testcase. Useful for local debugging.",
    )
    parser.add_argument(
        "-w", "--without-build", action="store_true",
        help="Skip build and reuse the existing build directory.",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Stop immediately after the first failed testcase.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List discovered testcases and exit.",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1,
        help=(
            "Number of execution units to run in parallel after the shared build "
            "(sim only; npu requires --jobs 1, default: 1)."
        ),
    )
    parser.add_argument(
        "--split-by-case", action="append", default=[],
        help="Split the specified testcase into per-case execution units. Can be passed multiple times.",
    )
    parser.add_argument(
        "--split-all-by-case", action="store_true",
        help="Split all selected testcases into per-case execution units.",
    )
    parser.add_argument(
        "--list-cases", action="store_true",
        help="List discovered case names for the selected testcase(s) and exit.",
    )
    return parser.parse_args()


def resolve_selected_testcases(all_testcases, requested):
    if not requested:
        return all_testcases

    requested_set = []
    seen = set()
    for testcase in requested:
        if testcase not in seen:
            requested_set.append(testcase)
            seen.add(testcase)

    missing = [testcase for testcase in requested_set if testcase not in all_testcases]
    if missing:
        raise ValueError(
            f"Unsupported testcase(s): {', '.join(missing)}; "
            f"supported: {', '.join(all_testcases)}"
        )
    return requested_set


def resolve_split_testcases(selected_testcases, requested, split_all_by_case):
    if split_all_by_case:
        return set(selected_testcases)

    resolved = []
    seen = set()
    for testcase in requested:
        if testcase in seen:
            continue
        if testcase not in selected_testcases:
            raise ValueError(
                f"Unsupported split-by-case testcase(s): {testcase}; "
                f"selected: {', '.join(selected_testcases)}"
            )
        resolved.append(testcase)
        seen.add(testcase)
    return set(resolved)


def build_execution_units(selected_testcases, split_testcases):
    units = []
    for testcase in selected_testcases:
        if testcase not in split_testcases:
            units.append(ExecutionUnit(testcase))
            continue

        case_names = run_st.discover_case_names(testcase)
        if not case_names:
            raise ValueError(f"No cases discovered for testcase: {testcase}")
        units.extend(ExecutionUnit(testcase, case_name) for case_name in case_names)
    return units


def validate_execution_constraints(run_mode, jobs):
    if run_mode == "npu" and jobs != 1:
        raise ValueError("--jobs > 1 is not supported in npu mode")


def run_execution_unit(execution_unit, log_dir):
    start_time = time.monotonic()
    log_path = run_st.execute_execution_unit(
        execution_unit.testcase,
        execution_unit.case,
        log_dir=log_dir,
    )
    duration_seconds = time.monotonic() - start_time
    return ExecutionResult(
        label=execution_unit.label,
        log_path=log_path,
        duration_seconds=duration_seconds,
    )


def main():
    args = parse_args()

    if args.soc_version not in SOC_VERSION_MAP:
        print(
            f"[ERROR] Unsupported soc-version: {args.soc_version}, "
            f"supported: {', '.join(sorted(SOC_VERSION_MAP))}",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.jobs < 1:
        print("[ERROR] --jobs must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.case is not None and len(args.testcase) != 1:
        print("[ERROR] --case requires exactly one selected testcase", file=sys.stderr)
        sys.exit(1)
    if args.case is not None and (args.split_by_case or args.split_all_by_case):
        print("[ERROR] --case cannot be combined with split-by-case options", file=sys.stderr)
        sys.exit(1)
    try:
        validate_execution_constraints(args.run_mode, args.jobs)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    tilelang_st_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    testcase_root = os.path.join(
        tilelang_st_root, "npu", args.soc_version, "src", "st", "testcase"
    )
    target_dir = os.path.dirname(testcase_root)

    if not os.path.isdir(testcase_root):
        print(f"[ERROR] Testcase root not found: {testcase_root}", file=sys.stderr)
        sys.exit(1)

    all_testcases = discover_testcases(testcase_root)
    if not all_testcases:
        print(f"[ERROR] No testcases found in: {testcase_root}", file=sys.stderr)
        sys.exit(1)

    if args.list:
        for testcase in all_testcases:
            print(testcase)
        return

    try:
        selected_testcases = resolve_selected_testcases(all_testcases, args.testcase)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    ptoas_bin = args.ptoas_bin or run_st.find_ptoas_bin()
    if not ptoas_bin:
        print(
            "[ERROR] Cannot find ptoas binary. Set PTOAS_BIN env or use -p flag.",
            file=sys.stderr,
        )
        sys.exit(1)
    ptoas_bin = os.path.abspath(ptoas_bin)

    default_soc_version = SOC_VERSION_MAP[args.soc_version]
    print(f"[INFO] run_mode={args.run_mode}")
    print(f"[INFO] soc_version={args.soc_version} ({default_soc_version})")
    print(f"[INFO] ptoas={ptoas_bin}")
    print(f"[INFO] target_dir={target_dir}")
    print(f"[INFO] selected_testcases={', '.join(selected_testcases)}")
    print(f"[INFO] jobs={args.jobs}")

    original_dir = os.getcwd()
    failures = []
    try:
        os.chdir(target_dir)
        log_dir = os.path.join(target_dir, "build", "logs")
        if args.list_cases:
            for testcase in selected_testcases:
                print(f"[INFO] testcase={testcase}")
                for case_name in run_st.discover_case_names(testcase):
                    print(case_name)
            return

        if args.case is not None:
            case_names = run_st.discover_case_names(selected_testcases[0])
            if args.case not in case_names:
                print(
                    f"[ERROR] Unsupported case: {args.case}; "
                    f"supported: {', '.join(case_names)}",
                    file=sys.stderr,
                )
                sys.exit(1)
            execution_units = [ExecutionUnit(selected_testcases[0], args.case)]
            split_testcases = set()
        else:
            try:
                split_testcases = resolve_split_testcases(
                    selected_testcases,
                    args.split_by_case,
                    args.split_all_by_case,
                )
                execution_units = build_execution_units(selected_testcases, split_testcases)
            except ValueError as exc:
                print(f"[ERROR] {exc}", file=sys.stderr)
                sys.exit(1)

        run_st.set_env_variables(args.run_mode, default_soc_version)
        if split_testcases:
            print(f"[INFO] split_by_case_testcases={', '.join(sorted(split_testcases))}")
        print(f"[INFO] execution_units={len(execution_units)}")
        print(f"[INFO] log_dir={log_dir}")

        if not args.without_build:
            build_target = "all" if selected_testcases == all_testcases else ";".join(selected_testcases)
            print(f"[INFO] build requested for {build_target}")
            run_st.build_project(args.run_mode, default_soc_version, build_target, ptoas_bin)

        total = len(execution_units)
        if args.jobs == 1:
            for index, execution_unit in enumerate(execution_units, start=1):
                print(f"[INFO] [{index}/{total}] running testcase: {execution_unit.label}")
                try:
                    result = run_execution_unit(execution_unit, log_dir)
                except Exception as exc:  # pragma: no cover - CI-side aggregation path
                    failures.append((execution_unit.label, str(exc)))
                    print(
                        f"[ERROR] testcase failed: {execution_unit.label} "
                        f"(log: {os.path.join(log_dir, run_st.get_execution_log_name(execution_unit.testcase, execution_unit.case))})"
                    )
                    traceback.print_exc()
                    if args.fail_fast:
                        break
                    continue

                print(
                    f"[INFO] completed testcase: {result.label} "
                    f"duration={result.duration_seconds:.1f}s log={result.log_path}"
                )
        else:
            print(f"[INFO] running testcases in parallel with jobs={args.jobs}")
            max_workers = min(args.jobs, total)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_testcase = {}
                for index, execution_unit in enumerate(execution_units, start=1):
                    print(f"[INFO] [{index}/{total}] queue testcase: {execution_unit.label}")
                    future = executor.submit(run_execution_unit, execution_unit, log_dir)
                    future_to_testcase[future] = execution_unit

                for future in concurrent.futures.as_completed(future_to_testcase):
                    execution_unit = future_to_testcase[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover - executor/host failure
                        failures.append((execution_unit.label, str(exc)))
                        print(
                            f"[ERROR] testcase runner crashed: {execution_unit.label} "
                            f"(log: {os.path.join(log_dir, run_st.get_execution_log_name(execution_unit.testcase, execution_unit.case))})"
                        )
                        traceback.print_exc()
                        if args.fail_fast:
                            break
                        continue

                    print(
                        f"[INFO] completed testcase: {result.label} "
                        f"duration={result.duration_seconds:.1f}s log={result.log_path}"
                    )

    except Exception as exc:
        print(f"[ERROR] batch run failed: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        os.chdir(original_dir)

    passed = len(execution_units) - len(failures)
    print("[INFO] TileLang ST summary")
    print(f"[INFO] passed={passed} failed={len(failures)} total={len(execution_units)}")
    if len(execution_units) != len(selected_testcases):
        print(
            f"[INFO] selected_testcases={len(selected_testcases)} "
            f"execution_units={len(execution_units)}"
        )
    if failures:
        for testcase, reason in failures:
            print(f"[INFO] failed testcase: {testcase} ({reason})")
        sys.exit(1)


if __name__ == "__main__":
    main()
