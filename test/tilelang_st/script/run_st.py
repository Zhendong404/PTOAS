#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""
TileLang ST runner — validates TileLang DSL template library on NPU / simulator.

Usage:
    python3 test/tilelang_st/script/run_st.py -r npu -v a5 -t tadd
    python3 test/tilelang_st/script/run_st.py -r sim -v a5 -t tadd
"""

import os
import sys
import subprocess
import shutil
import argparse
import re
import runpy
import traceback


def log_message(message, log_handle=None):
    print(message, file=log_handle or sys.stdout, flush=True)


def run_command(command, cwd=None, check=True, log_handle=None):
    try:
        log_message(f"run command: {' '.join(command)}", log_handle)
        subprocess.run(
            command,
            cwd=cwd,
            check=check,
            stdout=log_handle,
            stderr=log_handle,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        log_message(f"run command failed with return code {e.returncode}", log_handle)
        raise


def find_ptoas_bin():
    """Locate the ptoas binary by walking up from this script to the repo root."""
    env_bin = os.environ.get("PTOAS_BIN")
    if env_bin and os.path.isfile(env_bin):
        return os.path.abspath(env_bin)

    search_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        candidate = os.path.join(search_dir, "build", "tools", "ptoas", "ptoas")
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        parent = os.path.dirname(search_dir)
        if parent == search_dir:
            break
        search_dir = parent
    return None


def sanitize_case_name(case_name):
    return re.sub(r"[^0-9A-Za-z_.-]", "_", case_name)


def set_env_variables(run_mode, soc_version):
    if run_mode == "sim":
        ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
        if ld_lib_path:
            filtered_paths = [
                path for path in ld_lib_path.split(":")
                if "/runtime/lib64" not in path
            ]
            os.environ["LD_LIBRARY_PATH"] = ":".join(filtered_paths)

        ascend_home = os.environ.get("ASCEND_HOME_PATH")
        if not ascend_home:
            raise EnvironmentError("ASCEND_HOME_PATH is not set")

        os.environ["LD_LIBRARY_PATH"] = (
            f"{ascend_home}/runtime/lib64/stub:{os.environ.get('LD_LIBRARY_PATH', '')}"
        )
        setenv_path = os.path.join(ascend_home, "bin", "setenv.bash")
        if os.path.exists(setenv_path):
            print(f"run env shell: {setenv_path}")
            result = subprocess.run(
                f"source {setenv_path} && env",
                shell=True,
                executable=shutil.which("bash") or "bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for line in result.stdout.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value
        else:
            print(f"warning: not found {setenv_path}")

        simulator_lib_path = os.path.join(
            ascend_home, "tools", "simulator", soc_version, "lib"
        )
        os.environ["LD_LIBRARY_PATH"] = (
            f"{simulator_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        )


def get_testcase_source_dir(testcase):
    return os.path.join("testcase", testcase)


def get_testcase_work_dir(testcase, case_filter=None):
    work_dir = os.path.join("build", "testcase", testcase)
    if case_filter is None:
        return work_dir
    return os.path.join(work_dir, "_case_runs", sanitize_case_name(case_filter))


def get_testcase_binary_path(testcase):
    return os.path.abspath(os.path.join("build", "bin", testcase))


def load_testcase_cases(testcase):
    cases_path = os.path.join(get_testcase_source_dir(testcase), "cases.py")
    if not os.path.isfile(cases_path):
        raise FileNotFoundError(f"cases.py not found for testcase: {testcase}")

    namespace = runpy.run_path(cases_path)
    cases = namespace.get("CASES")
    if not isinstance(cases, list):
        raise ValueError(f"CASES is not a list in: {cases_path}")
    return cases


def discover_case_names(testcase):
    return [str(case["name"]) for case in load_testcase_cases(testcase)]


def get_execution_log_name(testcase, case_filter=None):
    if case_filter is None:
        return f"{sanitize_case_name(testcase)}.log"
    return f"{sanitize_case_name(testcase)}__{sanitize_case_name(case_filter)}.log"


def build_project(run_mode, soc_version, testcase, ptoas_bin):
    build_dir = "build"
    if os.path.exists(build_dir):
        print(f"clean build: {build_dir}")
        shutil.rmtree(build_dir)
    os.makedirs(build_dir, exist_ok=True)

    try:
        cmake_cmd = [
            "cmake",
            f"-DRUN_MODE={run_mode}",
            f"-DSOC_VERSION={soc_version}",
            f"-DTEST_CASE={testcase}",
            f"-DPTOAS_BIN={ptoas_bin}",
            "..",
        ]
        subprocess.run(
            cmake_cmd,
            cwd=build_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        cpu_count = os.cpu_count() or 4
        make_cmd = ["make", "VERBOSE=1", "-j", str(cpu_count)]
        result = subprocess.run(
            make_cmd,
            cwd=build_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print("compile process:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"build failed: {e.stdout}")
        raise


def _link_or_copy(src, dst):
    src_abs = os.path.abspath(src)
    if os.path.lexists(dst):
        if os.path.islink(dst) and os.path.realpath(dst) == src_abs:
            return
        os.remove(dst)

    try:
        os.symlink(src_abs, dst)
    except OSError:
        shutil.copyfile(src_abs, dst)


def _write_filtered_cases_wrapper(source_path, work_dir, case_filter):
    filtered_cases_path = os.path.join(work_dir, "cases.py")
    all_cases_path = os.path.join(work_dir, "_all_cases.py")
    _link_or_copy(source_path, all_cases_path)

    with open(filtered_cases_path, "w", encoding="utf-8") as handle:
        handle.write(
            "# Auto-generated by run_st.py for single-case execution.\n"
            "from _all_cases import CASES as _ALL_CASES\n\n"
            f"CASES = [case for case in _ALL_CASES if case.get('name') == {case_filter!r}]\n"
            "if not CASES:\n"
            f"    raise ValueError('unknown case filter: {case_filter}')\n"
        )


def _copy_testcase_scripts(testcase, case_filter=None):
    """Copy shared and per-testcase Python scripts into the build work directory."""
    work_dir = get_testcase_work_dir(testcase, case_filter)
    os.makedirs(work_dir, exist_ok=True)
    # Shared scripts (testcase/ level).
    for name in ("st_common.py",):
        src = os.path.join("testcase", name)
        if os.path.isfile(src):
            _link_or_copy(src, os.path.join(work_dir, name))
    # Per-testcase scripts.
    testcase_src = get_testcase_source_dir(testcase)
    for name in ("cases.py", "gen_data.py", "compare.py"):
        src = os.path.join(testcase_src, name)
        if not os.path.isfile(src):
            continue
        if name == "cases.py" and case_filter is not None:
            _write_filtered_cases_wrapper(src, work_dir, case_filter)
            continue
        _link_or_copy(src, os.path.join(work_dir, name))


def run_gen_data(testcase, case_filter=None, log_handle=None):
    try:
        work_dir = get_testcase_work_dir(testcase, case_filter)
        _copy_testcase_scripts(testcase, case_filter)
        run_command([sys.executable, "gen_data.py"], cwd=work_dir, log_handle=log_handle)
    except Exception as e:
        log_message(f"gen golden failed: {e}", log_handle)
        raise


def run_binary(testcase, case_filter=None, log_handle=None):
    try:
        work_dir = get_testcase_work_dir(testcase, case_filter)
        cmd = [get_testcase_binary_path(testcase)]
        if case_filter:
            cmd.append(case_filter)
        run_command(cmd, cwd=work_dir, log_handle=log_handle)
    except Exception as e:
        log_message(f"run binary failed: {e}", log_handle)
        raise


def run_compare(testcase, case_filter=None, log_handle=None):
    try:
        work_dir = get_testcase_work_dir(testcase, case_filter)
        cmd = [sys.executable, "compare.py"]
        if case_filter:
            cmd.append(case_filter)
        run_command(cmd, cwd=work_dir, log_handle=log_handle)
    except Exception as e:
        log_message(f"compare failed: {e}", log_handle)
        raise


def execute_execution_unit(testcase, case_filter=None, log_dir=None):
    log_path = None
    log_handle = None
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, get_execution_log_name(testcase, case_filter))
        log_handle = open(log_path, "w", encoding="utf-8")

    try:
        log_message(f"[INFO] begin testcase={testcase} case={case_filter or '<all>'}", log_handle)
        run_gen_data(testcase, case_filter, log_handle=log_handle)
        run_binary(testcase, case_filter, log_handle=log_handle)
        run_compare(testcase, case_filter, log_handle=log_handle)
        log_message("[INFO] execution unit passed", log_handle)
        return log_path
    except Exception:
        traceback.print_exc(file=log_handle or sys.stderr)
        raise
    finally:
        if log_handle is not None:
            log_handle.close()


def main():
    parser = argparse.ArgumentParser(description="TileLang ST runner")
    parser.add_argument("-r", "--run-mode", required=True,
                        help="Run mode: sim or npu")
    parser.add_argument("-v", "--soc-version", required=True,
                        help="SoC version: a5")
    parser.add_argument("-t", "--testcase", required=True,
                        help="Test case name (e.g. tadd)")
    parser.add_argument("-p", "--ptoas-bin", required=False,
                        help="Path to ptoas binary (auto-detected if omitted)")
    parser.add_argument("-c", "--case", required=False, default=None,
                        help="Run a specific case within the testcase (e.g. f32_16x64)")
    parser.add_argument("-w", "--without-build", action="store_true",
                        help="Skip build (requires prior build)")

    args = parser.parse_args()

    if args.soc_version == "a5":
        default_soc_version = "Ascend950PR_9599"
    else:
        print(f"[ERROR] Unsupported soc-version: {args.soc_version}, only a5 is supported",
              file=sys.stderr)
        sys.exit(1)

    testcase = args.testcase

    ptoas_bin = args.ptoas_bin or find_ptoas_bin()
    if not ptoas_bin:
        print("[ERROR] Cannot find ptoas binary. "
              "Set PTOAS_BIN env or use -p flag.", file=sys.stderr)
        sys.exit(1)
    ptoas_bin = os.path.abspath(ptoas_bin)
    print(f"[INFO] ptoas: {ptoas_bin}")

    original_dir = os.getcwd()
    try:
        script_path = os.path.abspath(__file__)
        tilelang_st_root = os.path.dirname(os.path.dirname(script_path))
        target_dir = os.path.join(tilelang_st_root, "npu", args.soc_version, "src", "st")

        if not os.path.isdir(target_dir):
            print(f"[ERROR] Target dir not found: {target_dir}", file=sys.stderr)
            sys.exit(1)

        print(f"target_dir: {target_dir}")
        os.chdir(target_dir)

        set_env_variables(args.run_mode, default_soc_version)

        if not args.without_build:
            build_project(args.run_mode, default_soc_version, testcase, ptoas_bin)

        # gen golden → run binary → compare
        run_gen_data(testcase, args.case)
        run_binary(testcase, args.case)
        run_compare(testcase, args.case)

    except Exception as e:
        print(f"run failed: {str(e)}", file=sys.stderr)
        sys.exit(1)
    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
