# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import unittest

from classify_ci_sim_changes import SUITES, classify


class ClassifyCiSimChangesTest(unittest.TestCase):
    def assert_selected(self, paths, *expected):
        selection = classify("pull_request", paths)
        self.assertEqual(selection.wheel, bool(expected))
        self.assertEqual(
            {suite for suite, selected in selection.suites.items() if selected},
            set(expected),
        )
        return selection

    def test_direct_vpto_change_selects_only_vpto(self):
        self.assert_selected(["test/vpto/cases/basic/test.py"], "vpto")

    def test_each_direct_test_area_has_one_owner(self):
        cases = {
            "test/tilelib-st/test_tilelib_st.py": "tilelib",
            "test/dsl-st/test_abs.py": "ptodsl",
            "test/tilelang_st/test_tileop.py": "tileop",
        }
        for path, suite in cases.items():
            with self.subTest(path=path):
                self.assert_selected([path], suite)

    def test_multiple_direct_areas_select_union(self):
        self.assert_selected(
            ["test/vpto/cases/a/golden.py", "test/dsl-st/test_abs.py"],
            "vpto",
            "ptodsl",
        )

    def test_shared_source_selects_all_suites(self):
        self.assert_selected(["lib/PTO/Transforms/PTOToEmitC.cpp"], *SUITES)

    def test_unknown_path_selects_all_suites(self):
        selection = self.assert_selected(["new-component/config.toml"], *SUITES)
        self.assertIn("unknown", selection.reason)

    def test_non_code_only_change_selects_nothing(self):
        selection = self.assert_selected(
            [
                "docs/development.md",
                "README.md",
                "openspec/changes/example/proposal.md",
                ".github/ISSUE_TEMPLATE/bug.yml",
                "LICENSE",
                ".gitignore",
            ]
        )
        self.assertIn("non-code", selection.reason)

    def test_direct_and_non_code_selects_direct_owner(self):
        selection = self.assert_selected(
            ["test/tilelib-st/test_tilelib_st.py", "docs/tilelib.md"], "tilelib"
        )
        self.assertEqual(selection.matched_paths, ("test/tilelib-st/test_tilelib_st.py",))

    def test_non_pr_events_select_all_suites(self):
        for event_name in ("schedule", "workflow_dispatch"):
            with self.subTest(event_name=event_name):
                selection = classify(event_name, [])
                self.assertTrue(selection.wheel)
                self.assertTrue(all(selection.suites.values()))

    def test_github_outputs_include_wheel_suites_paths_and_reason(self):
        selection = classify("pull_request", ["test/vpto/cases/basic/test.py"])
        outputs = selection.github_outputs()
        self.assertEqual(outputs["wheel"], "true")
        self.assertEqual(outputs["vpto"], "true")
        self.assertEqual(outputs["tilelib"], "false")
        self.assertEqual(outputs["matched_paths"], '["test/vpto/cases/basic/test.py"]')
        self.assertIn("vpto", outputs["selection_reason"])


if __name__ == "__main__":
    unittest.main()
