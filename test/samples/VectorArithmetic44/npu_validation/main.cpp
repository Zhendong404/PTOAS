/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include <cstdlib>
#include <iostream>

using namespace PtoTestCommon;

void LaunchVector_arith_44_kernel_2d(float *v1, float *v2, float *v3, float *v4, void *stream);

namespace {

bool CheckAclCall(const char *expr, aclError err, const char *file, int line) {
    if (err != ACL_SUCCESS) {
        std::cerr << "[ERROR] ACL call failed: " << expr
                  << ", err=" << static_cast<int>(err)
                  << " at " << file << ":" << line << std::endl;
        return false;
    }
    return true;
}

#define CHECK_ACL(expr) \
    do { \
        if (!CheckAclCall(#expr, (expr), __FILE__, __LINE__)) { \
            return 1; \
        } \
    } while (0)

}  // namespace

int main() {
    constexpr size_t kTileElemCount = 32 * 32;
    constexpr size_t kOutputTileCount = 32;
    size_t inputFileSize = kTileElemCount * sizeof(float);
    size_t outputFileSize = kTileElemCount * kOutputTileCount * sizeof(float);

    float *dstHost0 = nullptr;
    float *dstDevice0 = nullptr;
    float *srcHost0 = nullptr;
    float *srcHost1 = nullptr;
    float *srcHost2 = nullptr;
    float *srcDevice0 = nullptr;
    float *srcDevice1 = nullptr;
    float *srcDevice2 = nullptr;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));

    CHECK_ACL(aclrtMallocHost((void **)(&dstHost0), outputFileSize));
    CHECK_ACL(aclrtMallocHost((void **)(&srcHost0), inputFileSize));
    CHECK_ACL(aclrtMallocHost((void **)(&srcHost1), inputFileSize));
    CHECK_ACL(aclrtMallocHost((void **)(&srcHost2), inputFileSize));
    CHECK_ACL(aclrtMalloc((void **)&dstDevice0, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&srcDevice0, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&srcDevice1, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&srcDevice2, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    if (!ReadFile("./v1.bin", inputFileSize, srcHost0, inputFileSize) ||
        !ReadFile("./v2.bin", inputFileSize, srcHost1, inputFileSize) ||
        !ReadFile("./v3.bin", inputFileSize, srcHost2, inputFileSize)) {
        return 1;
    }
    CHECK_ACL(aclrtMemcpy(srcDevice0, inputFileSize, srcHost0, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(srcDevice1, inputFileSize, srcHost1, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(srcDevice2, inputFileSize, srcHost2, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    LaunchVector_arith_44_kernel_2d(srcDevice0, srcDevice1, srcDevice2, dstDevice0, stream);

    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(dstHost0, outputFileSize, dstDevice0, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST));

    if (!WriteFile("./v4.bin", dstHost0, outputFileSize)) {
        return 1;
    }

    CHECK_ACL(aclrtFree(srcDevice0));
    CHECK_ACL(aclrtFree(srcDevice1));
    CHECK_ACL(aclrtFree(srcDevice2));
    CHECK_ACL(aclrtFree(dstDevice0));
    CHECK_ACL(aclrtFreeHost(srcHost0));
    CHECK_ACL(aclrtFreeHost(srcHost1));
    CHECK_ACL(aclrtFreeHost(srcHost2));
    CHECK_ACL(aclrtFreeHost(dstHost0));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());

    return 0;
}
