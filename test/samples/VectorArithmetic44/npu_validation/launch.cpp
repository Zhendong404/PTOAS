#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

__global__ AICORE void vector_arith_44_kernel_2d(__gm__ float *v1, __gm__ float *v2, __gm__ float *v3, __gm__ float *v4);

void LaunchVector_arith_44_kernel_2d(float *v1, float *v2, float *v3, float *v4, void *stream) {
    vector_arith_44_kernel_2d<<<1, nullptr, stream>>>(v1, v2, v3, v4);
}
