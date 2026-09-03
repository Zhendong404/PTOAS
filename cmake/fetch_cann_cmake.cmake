# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(NOT PROJECT_SOURCE_DIR)
    # Keep the CANN CMake integration as a pinned source submodule.  This makes
    # normal source checkouts reproducible and avoids configure-time network
    # access or an empty CANN_3RD_LIB_PATH resolving to /cann-cmake.
    set(CANN_CMAKE_SOURCE_DIR
        "${CMAKE_CURRENT_LIST_DIR}/../third_party/cann-cmake")
    if(NOT EXISTS "${CANN_CMAKE_SOURCE_DIR}/function/prepare.cmake")
        message(FATAL_ERROR
                "missing CANN CMake submodule at ${CANN_CMAKE_SOURCE_DIR}; "
                "run `git submodule update --init --recursive "
                "third_party/cann-cmake`")
    endif()
    include("${CANN_CMAKE_SOURCE_DIR}/function/prepare.cmake")
endif()
