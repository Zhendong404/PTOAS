#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import os
import runpy
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

run_st = load_module("tilelang_run_st", SCRIPT_DIR / "run_st.py")
run_all_st = load_module("tilelang_run_all_st", SCRIPT_DIR / "run_all_st.py")


class BatchRunnerTest(unittest.TestCase):
    def test_get_testcase_work_dir_uses_case_specific_subdir(self):
        work_dir = run_st.get_testcase_work_dir("demo", "case/1")
        self.assertTrue(work_dir.endswith("build/testcase/demo/_case_runs/case_1"))

    def test_discover_case_names_reads_cases_py(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            testcase_dir = root / "testcase" / "demo"
            testcase_dir.mkdir(parents=True)
            (testcase_dir / "cases.py").write_text(
                textwrap.dedent(
                    """
                    CASES = [
                        {"name": "alpha"},
                        {"name": "beta"},
                    ]
                    """
                ),
                encoding="utf-8",
            )

            cwd = os.getcwd()
            try:
                os.chdir(root)
                self.assertEqual(run_st.discover_case_names("demo"), ["alpha", "beta"])
            finally:
                os.chdir(cwd)

    def test_copy_testcase_scripts_writes_filtered_cases_wrapper(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shared_dir = root / "testcase"
            testcase_dir = shared_dir / "demo"
            testcase_dir.mkdir(parents=True)
            (shared_dir / "st_common.py").write_text("# shared\n", encoding="utf-8")
            (testcase_dir / "gen_data.py").write_text("# gen\n", encoding="utf-8")
            (testcase_dir / "compare.py").write_text("# compare\n", encoding="utf-8")
            (testcase_dir / "cases.py").write_text(
                textwrap.dedent(
                    """
                    CASES = [
                        {"name": "alpha"},
                        {"name": "beta"},
                    ]
                    """
                ),
                encoding="utf-8",
            )

            cwd = os.getcwd()
            try:
                os.chdir(root)
                run_st._copy_testcase_scripts("demo", "beta")
                work_dir = Path(run_st.get_testcase_work_dir("demo", "beta"))
                self.assertTrue((work_dir / "_all_cases.py").is_file())
                sys.path.insert(0, str(work_dir))
                try:
                    filtered_cases = runpy.run_path(str(work_dir / "cases.py"))["CASES"]
                finally:
                    sys.path.pop(0)
                self.assertEqual([case["name"] for case in filtered_cases], ["beta"])
            finally:
                os.chdir(cwd)

    def test_build_execution_units_splits_selected_testcases(self):
        with mock.patch.object(run_all_st.run_st, "discover_case_names", return_value=["c1", "c2"]):
            units = run_all_st.build_execution_units(
                ["tadd", "trowargmax"],
                {"trowargmax"},
            )

        labels = [unit.label for unit in units]
        self.assertEqual(labels, ["tadd", "trowargmax::c1", "trowargmax::c2"])

    def test_resolve_split_testcases_rejects_unselected_testcase(self):
        with self.assertRaises(ValueError):
            run_all_st.resolve_split_testcases(["tadd"], ["trowargmax"], False)

    def test_run_execution_unit_returns_log_path_and_duration(self):
        execution_unit = run_all_st.ExecutionUnit("trowargmax", "case_a")
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(run_all_st.run_st, "execute_execution_unit", return_value="/tmp/case.log"):
                result = run_all_st.run_execution_unit(execution_unit, temp_dir)

        self.assertEqual(result.label, "trowargmax::case_a")
        self.assertEqual(result.log_path, "/tmp/case.log")
        self.assertGreaterEqual(result.duration_seconds, 0.0)

    def test_validate_execution_constraints_allows_parallel_sim(self):
        run_all_st.validate_execution_constraints("sim", 64)

    def test_validate_execution_constraints_allows_serial_npu(self):
        run_all_st.validate_execution_constraints("npu", 1)

    def test_validate_execution_constraints_rejects_parallel_npu(self):
        with self.assertRaisesRegex(ValueError, "npu mode"):
            run_all_st.validate_execution_constraints("npu", 2)


if __name__ == "__main__":
    unittest.main()
