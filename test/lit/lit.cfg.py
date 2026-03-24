# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util
import lit.llvm

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'PTOIR'

if lit.llvm.llvm_config is None:
    test_dir = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(test_dir, os.pardir, os.pardir))
    build_root = os.environ.get('PTOAS_BUILD_DIR',
                                os.path.join(repo_root, 'build'))
    llvm_build_root = os.environ.get(
        'LLVM_BUILD_DIR',
        os.path.abspath(os.path.join(repo_root, os.pardir, 'llvm-project',
                                     'build-shared')))

    config.llvm_tools_dir = os.path.join(llvm_build_root, 'bin')
    config.llvm_shlib_ext = getattr(config, 'llvm_shlib_ext', '.so')
    config.lit_tools_dir = getattr(config, 'lit_tools_dir', '')
    config.ptoir_src_root = repo_root
    config.ptoir_obj_root = build_root

    lit.llvm.initialize(lit_config, config)

llvm_config = lit.llvm.llvm_config

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.pto']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.ptoir_obj_root, 'test/lit')
config.ptoir_tools_dir = os.path.join(config.ptoir_obj_root, 'tools/ptoas')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'Examples', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [config.ptoir_tools_dir, config.llvm_tools_dir]
tools = [
    'ptoas',
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
