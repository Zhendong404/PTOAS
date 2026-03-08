// RUN: ptoas %S/generic_shape_dtype_chain.pto --op-lib-dir=%S/oplib --pto-arch=a5 -o %t.generic.cpp
// RUN: FileCheck %s < %t.generic.cpp

// CHECK-DAG: __global__ AICORE void binary_chain_f16_64x64(
// CHECK-DAG: __global__ AICORE void binary_chain_f32_48x48(
// CHECK-DAG: vadd(
// CHECK-DAG: vmin(
// CHECK-DAG: CreatePredicate<half>(
// CHECK-NOT: PTOAS__OPLIB_
